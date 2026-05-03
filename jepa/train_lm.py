"""Train a plain causal-LM (next-token CE) baseline with the same architecture
and training recipe as a JEPA config — for head-to-head feature comparison.

Mirrors train_jepa.py end-to-end (distributed setup, bf16+compile, wandb,
cosine LR schedule, FineWebBatcher data loader) but with cross-entropy on
token logits instead of InfoNCE on latents.
"""
import dataclasses
import math
import os
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa.configs import load_config
from jepa.data import FineWebBatcher, synthetic_batch
from jepa.gpt import GPT
from jepa.train_jepa import lr_for_step, make_synthetic_loader, setup_distributed


@torch.no_grad()
def eval_lm(model, val_loader, n_steps: int) -> dict:
    model.eval()
    losses, top1s, top5s = [], [], []
    for _ in range(n_steps):
        x = next(val_loader)
        logits = model(x).float()
        target = x[:, 1:]
        flat_logits = logits[:, :-1].reshape(-1, logits.size(-1))
        flat_target = target.reshape(-1)
        loss = F.cross_entropy(flat_logits, flat_target)
        with torch.no_grad():
            pred = flat_logits.argmax(dim=-1)
            top1 = (pred == flat_target).float().mean()
            top5 = (flat_logits.topk(5, dim=-1).indices == flat_target[:, None]).any(dim=-1).float().mean()
        losses.append(loss)
        top1s.append(top1)
        top5s.append(top5)
    model.train()
    return {
        "val_loss": torch.stack(losses).mean().item(),
        "val_top1": torch.stack(top1s).mean().item(),
        "val_top5": torch.stack(top5s).mean().item(),
        "val_perplexity": math.exp(torch.stack(losses).mean().item()),
    }


def main(
    config_name: str = "lm_large",
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
        wandb_run = wandb.init(
            project="jepa-modded-nanogpt",
            name=wandb_run_name or f"run_{run_id}",
            config={
                "kind": "lm",
                "model": dataclasses.asdict(cfg.model),
                "train": dataclasses.asdict(cfg.train),
                "total_steps": total_steps,
                "warmup_steps": warmup_steps,
                "use_bf16": use_bf16,
                "use_compile": use_compile,
                "synthetic": synthetic,
                "world_size": world_size,
            },
        )

    log(f"world_size={world_size} rank={rank} device={device}")
    log(f"config_name={config_name} use_bf16={use_bf16} use_compile={use_compile} use_wandb={use_wandb}")
    log(f"total_steps={total_steps} warmup_steps={warmup_steps}")
    log(f"config: {cfg}")

    model = GPT(cfg.model).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"model parameters: {n_params/1e6:.2f}M")

    if is_distributed:
        model = DDP(model, device_ids=[local_rank])
    model_unwrapped = model.module if is_distributed else model

    if use_compile:
        log("torch.compile(model)...")
        model = torch.compile(model, dynamic=False)

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
            logits = model(x).float()
            target = x[:, 1:]
            flat_logits = logits[:, :-1].reshape(-1, logits.size(-1))
            flat_target = target.reshape(-1)
            loss = F.cross_entropy(flat_logits, flat_target)

        opt.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        opt.step()

        if step % cfg.train.log_every == 0:
            with torch.no_grad():
                pred = flat_logits.argmax(dim=-1)
                top1 = (pred == flat_target).float().mean().item()
            sps = (step + 1) / (time.time() - t0)
            log(
                f"loss={loss.item():.4f} top1={top1:.3f} ppl={math.exp(loss.item()):.2f} "
                f"lr={lr:.2e} grad={grad_norm.item():.2f} sps={sps:.1f}"
            )
            if wandb_run is not None:
                wandb_run.log(
                    {
                        "train/loss": loss.item(),
                        "train/top1": top1,
                        "train/perplexity": math.exp(loss.item()),
                        "train/lr": lr,
                        "train/grad_norm": grad_norm.item(),
                        "train/steps_per_sec": sps,
                    },
                    step=step,
                )

        if step > 0 and step % cfg.train.val_every == 0:
            with autocast_ctx:
                val_metrics = eval_lm(model_unwrapped, val_loader, cfg.train.val_steps)
            for k, v in val_metrics.items():
                log(f"  {k}={v:.4f}")
            if wandb_run is not None:
                wandb_run.log({f"val/{k.removeprefix('val_')}": v for k, v in val_metrics.items()}, step=step)

    if master:
        ckpt_dir = Path(__file__).resolve().parent / "checkpoints"
        ckpt_dir.mkdir(exist_ok=True)
        ckpt_path = ckpt_dir / f"{config_name}_step{total_steps}.pt"
        torch.save(
            {
                "model": model_unwrapped.state_dict(),
                "step": total_steps,
                "cfg_model": cfg.model,
                "cfg_train": cfg.train,
                "kind": "lm",
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
    config_name = config_arg.split("=")[1] if config_arg else "lm_large"
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
