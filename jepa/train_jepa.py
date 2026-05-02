import dataclasses
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa.configs import load_config
from jepa.data import FineWebBatcher, synthetic_batch
from jepa.ema import EMATarget
from jepa.eval import evaluate_val
from jepa.loss import infonce_loss
from jepa.model import JEPA


def setup_distributed() -> tuple[int, int, int, torch.device, bool]:
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        device = torch.device("cuda", local_rank)
        torch.cuda.set_device(device)
        dist.init_process_group(backend="nccl", device_id=device)
        return rank, world_size, local_rank, device, True
    rank, world_size, local_rank = 0, 1, 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return rank, world_size, local_rank, device, False


def lr_for_step(step: int, warmup: int, total: int, max_lr: float) -> float:
    if step < warmup:
        return max_lr * step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))


def make_synthetic_loader(seqs_per_step: int, seq_len: int, vocab_size: int, device: torch.device, rank: int, seed_start: int):
    seed = seed_start + rank * 7919
    while True:
        yield synthetic_batch(seqs_per_step, seq_len, vocab_size, device=device, seed=seed)
        seed += 1


def main(
    config_name: str = "baseline",
    synthetic: bool = False,
    total_steps_override: int | None = None,
    warmup_steps_override: int | None = None,
    use_bf16: bool = True,
    use_compile: bool = True,
    use_wandb: bool = True,
    wandb_run_name: str | None = None,
):
    rank, world_size, local_rank, device, is_distributed = setup_distributed()
    master = rank == 0
    cfg = load_config(config_name)
    torch.manual_seed(cfg.train.seed + rank)
    torch.set_float32_matmul_precision("high")

    log_dir = Path(__file__).resolve().parent / "logs"
    log_dir.mkdir(exist_ok=True)
    run_id = int(time.time())
    log_file = log_dir / f"run_{run_id}.log" if master else None

    state = {"step": 0}

    def log(msg: str):
        if master:
            line = f"[step={state['step']:6d}] {msg}"
            print(line, flush=True)
            with log_file.open("a") as f:
                f.write(line + "\n")

    total_steps = total_steps_override if total_steps_override is not None else cfg.train.total_steps
    warmup_steps = warmup_steps_override if warmup_steps_override is not None else cfg.train.warmup_steps

    wandb_run = None
    if use_wandb and master:
        import wandb
        wandb_config = {
            "model": dataclasses.asdict(cfg.model),
            "train": dataclasses.asdict(cfg.train),
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "use_bf16": use_bf16,
            "use_compile": use_compile,
            "synthetic": synthetic,
            "world_size": world_size,
        }
        wandb_run = wandb.init(
            project="jepa-modded-nanogpt",
            name=wandb_run_name or f"run_{run_id}",
            config=wandb_config,
        )

    log(f"world_size={world_size} rank={rank} device={device}")
    log(f"config_name={config_name} use_bf16={use_bf16} use_compile={use_compile} use_wandb={use_wandb}")
    log(f"total_steps={total_steps} warmup_steps={warmup_steps}")
    log(f"config: {cfg}")

    model = JEPA(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model parameters: {n_params/1e6:.2f}M")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    model_unwrapped = model.module if is_distributed else model

    use_ema = cfg.train.use_ema
    pred_lag = cfg.train.pred_lag
    ema = None
    if use_ema:
        log(f"EMA target encoder enabled (momentum={cfg.train.ema_momentum})")
        ema = EMATarget(model_unwrapped, momentum=cfg.train.ema_momentum)
    if pred_lag != 1:
        log(f"prediction lag = {pred_lag} (predicting z_{{t+{pred_lag}}} instead of z_{{t+1}})")

    if use_compile:
        log("torch.compile(model)...")
        model = torch.compile(model, dynamic=False)
        log("compile graph will warm up on first forward")

    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        if use_bf16 and device.type == "cuda"
        else nullcontext()
    )

    if synthetic:
        train_loader = make_synthetic_loader(cfg.train.seqs_per_step, cfg.train.seq_len, cfg.model.vocab_size, device, rank, seed_start=0)
        val_loader = make_synthetic_loader(cfg.train.seqs_per_step, cfg.train.seq_len, cfg.model.vocab_size, device, rank, seed_start=10_000_000)
    else:
        train_loader = FineWebBatcher(cfg.train.train_data_glob, cfg.train.seqs_per_step, cfg.train.seq_len, rank, world_size, device, shuffle=True, seed=cfg.train.seed)
        val_loader = FineWebBatcher(cfg.train.val_data_glob, cfg.train.seqs_per_step, cfg.train.seq_len, rank, world_size, device, shuffle=False, seed=cfg.train.seed)

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
    )

    t0 = time.time()
    for step in range(total_steps):
        state["step"] = step

        lr = lr_for_step(step, warmup_steps, total_steps, cfg.train.lr)
        for g in opt.param_groups:
            g["lr"] = lr

        x = next(train_loader)
        with autocast_ctx:
            p, z = model(x)
            target_z = ema.target_z(x) if ema is not None else z
            loss, metrics = infonce_loss(
                p, target_z, tau=cfg.train.tau,
                subsample_cap=cfg.train.neg_subsample_cap,
                pred_lag=pred_lag,
            )

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()
        if ema is not None:
            ema.update()

        if step % cfg.train.log_every == 0:
            sps = (step + 1) / (time.time() - t0)
            log(
                f"loss={loss.item():.4f} top1={metrics['top1'].item():.3f} "
                f"top5={metrics['top5'].item():.3f} diag_sim={metrics['diag_cos_sim'].item():.3f} "
                f"lr={lr:.2e} grad={grad_norm.item():.2f} sps={sps:.1f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss.item(),
                        "train/top1": metrics["top1"].item(),
                        "train/top5": metrics["top5"].item(),
                        "train/diag_cos_sim": metrics["diag_cos_sim"].item(),
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item(),
                        "train/steps_per_sec": sps,
                    },
                    step=step,
                )

        if step > 0 and step % cfg.train.val_every == 0:
            with autocast_ctx:
                val_metrics = evaluate_val(
                    model_unwrapped,
                    val_loader,
                    cfg.train.val_steps,
                    tau=cfg.train.tau,
                    subsample_cap=cfg.train.neg_subsample_cap,
                )
            for k, v in val_metrics.items():
                log(f"  {k}={v:.4f}")
            if wandb_run is not None:
                wandb_run.log({f"val/{k.removeprefix('val_')}": v for k, v in val_metrics.items()}, step=step)

    if master:
        ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"final_step{total_steps}.pt"
        torch.save(
            {
                "model": model_unwrapped.state_dict(),
                "step": total_steps,
                "cfg_model": cfg.model,
                "cfg_train": cfg.train,
            },
            ckpt_path,
        )
        log(f"saved checkpoint to {ckpt_path}")

    if wandb_run is not None:
        wandb_run.finish()
    if is_distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    synthetic = "--synthetic" in sys.argv
    no_bf16 = "--no-bf16" in sys.argv
    no_compile = "--no-compile" in sys.argv
    no_wandb = "--no-wandb" in sys.argv
    steps_arg = next((a for a in sys.argv if a.startswith("--steps=")), None)
    warmup_arg = next((a for a in sys.argv if a.startswith("--warmup=")), None)
    name_arg = next((a for a in sys.argv if a.startswith("--name=")), None)
    config_arg = next((a for a in sys.argv if a.startswith("--config=")), None)
    steps = int(steps_arg.split("=")[1]) if steps_arg else None
    warmup = int(warmup_arg.split("=")[1]) if warmup_arg else None
    name = name_arg.split("=")[1] if name_arg else None
    config_name = config_arg.split("=")[1] if config_arg else "baseline"
    main(
        config_name=config_name,
        synthetic=synthetic,
        total_steps_override=steps,
        warmup_steps_override=warmup,
        use_bf16=not no_bf16,
        use_compile=not no_compile,
        use_wandb=not no_wandb,
        wandb_run_name=name,
    )
