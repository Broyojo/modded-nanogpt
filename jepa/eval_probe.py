"""Standalone linear-probe evaluation of a trained JEPA checkpoint.

Trains a frozen-encoder Linear(model_dim -> vocab) head on next-token prediction
for `--train-steps`, then reports val perplexity. This is the closest analog to
"is this representation useful for LM" without giving the model a real LM head.

Usage:
  uv run python jepa/eval_probe.py --ckpt jepa/checkpoints/final_step50000.pt
"""
import argparse
import dataclasses
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa.configs.baseline import JEPAConfig
from jepa.data import FineWebBatcher
from jepa.eval import linear_probe
from jepa.model import JEPA


def load_jepa(ckpt_path: str, device: str = "cuda") -> tuple[JEPA, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_model = ckpt["cfg_model"]
    if isinstance(cfg_model, dict):
        cfg_model = JEPAConfig(**cfg_model)
    elif not isinstance(cfg_model, JEPAConfig):
        cfg_model = JEPAConfig(**dataclasses.asdict(cfg_model))
    model = JEPA(cfg_model).to(device)
    model.load_state_dict(ckpt["model"])
    return model, ckpt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--train-glob", default="data/fineweb10B/fineweb_train_*.bin")
    ap.add_argument("--val-glob", default="data/fineweb10B/fineweb_val_*.bin")
    ap.add_argument("--train-steps", type=int, default=200)
    ap.add_argument("--val-steps", type=int, default=32)
    ap.add_argument("--seqs", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=3e-4)
    args = ap.parse_args()

    device = "cuda"
    torch.set_float32_matmul_precision("high")
    model, ckpt = load_jepa(args.ckpt, device)
    print(f"loaded {args.ckpt} (step={ckpt['step']}, model_dim={model.cfg.model_dim})")

    train_loader = FineWebBatcher(args.train_glob, args.seqs, args.seq_len, 0, 1, device, shuffle=True, seed=0)
    val_loader = FineWebBatcher(args.val_glob, args.seqs, args.seq_len, 0, 1, device, shuffle=False, seed=0)

    print(f"running linear probe: train_steps={args.train_steps} val_steps={args.val_steps}")
    metrics = linear_probe(
        model,
        train_loader,
        val_loader,
        vocab_size=model.cfg.vocab_size,
        train_steps=args.train_steps,
        val_steps=args.val_steps,
        lr=args.lr,
        device=device,
    )
    for k, v in metrics.items():
        print(f"  {k}={v:.4f}")


if __name__ == "__main__":
    main()
