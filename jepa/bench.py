"""Quick benchmark of bf16/compile combinations on the training step.

Measures steady-state steps/sec (after warmup) over a fixed window.
"""
import argparse
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa.configs.baseline import CONFIG
from jepa.data import synthetic_batch
from jepa.loss import infonce_loss
from jepa.model import JEPA


def bench_one(use_bf16: bool, use_compile: bool, warmup: int, measure: int) -> dict:
    cfg = CONFIG
    torch.manual_seed(cfg.train.seed)
    torch.set_float32_matmul_precision("high")
    device = torch.device("cuda")

    model = JEPA(cfg.model).to(device)
    if use_compile:
        model = torch.compile(model, dynamic=False)
    autocast_ctx = (
        torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_bf16 else nullcontext()
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.train.lr,
        betas=cfg.train.betas,
        weight_decay=cfg.train.weight_decay,
    )

    batch = synthetic_batch(cfg.train.seqs_per_step, cfg.train.seq_len, cfg.model.vocab_size, device=device, seed=0)

    for step in range(warmup):
        with autocast_ctx:
            p, z = model(batch)
            loss, _ = infonce_loss(p, z, tau=cfg.train.tau, subsample_cap=cfg.train.neg_subsample_cap)
        opt.zero_grad()
        loss.backward()
        opt.step()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for step in range(measure):
        with autocast_ctx:
            p, z = model(batch)
            loss, _ = infonce_loss(p, z, tau=cfg.train.tau, subsample_cap=cfg.train.neg_subsample_cap)
        opt.zero_grad()
        loss.backward()
        opt.step()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens_per_step = cfg.train.seqs_per_step * cfg.train.seq_len
    sps = measure / elapsed
    tps = sps * tokens_per_step
    return {
        "sps": sps,
        "tokens_per_sec": tps,
        "ms_per_step": 1000 * elapsed / measure,
        "final_loss": loss.item(),
    }


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--measure", type=int, default=50)
    args = ap.parse_args()

    print(f"warmup={args.warmup} measure={args.measure}")
    print(f"{'config':30s} {'sps':>7s} {'tok/s':>10s} {'ms/step':>10s} {'loss':>8s}")
    for use_bf16 in [False, True]:
        for use_compile in [False, True]:
            label = f"bf16={use_bf16}, compile={use_compile}"
            r = bench_one(use_bf16, use_compile, args.warmup, args.measure)
            print(f"{label:30s} {r['sps']:7.2f} {r['tokens_per_sec']:10.0f} {r['ms_per_step']:10.2f} {r['final_loss']:8.4f}")
