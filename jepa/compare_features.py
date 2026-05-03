"""Head-to-head feature-quality comparison: JEPA encoder@split vs LM hidden@split.

Loads two checkpoints (a JEPA and a plain-LM with the same arch + split index)
and runs three apples-to-apples probes on the layer-`split_index` hidden state
of each, plus the position-cheating diagnostic on the same features.

  1. SST-2 sentiment linear probe (sentence-level binary classification)
  2. Next-token CE linear probe (token-level perplexity)
  3. Position-cheating ratio on FineWeb val data

Usage:
  uv run python jepa/compare_features.py \\
      --jepa-ckpt jepa/checkpoints/large_step50000.pt \\
      --lm-ckpt   jepa/checkpoints/lm_large_step50000.pt
"""
import argparse
import dataclasses
import math
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".hf_cache"))
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from jepa.configs.baseline import JEPAConfig
from jepa.configs.lm_large import LMConfig
from jepa.data import BOS_ID, FineWebBatcher
from jepa.gpt import GPT
from jepa.loss import collapse_diagnostics
from jepa.model import JEPA


def _to_jepa_config(cfg) -> JEPAConfig:
    return JEPAConfig(**dataclasses.asdict(cfg))


def _to_lm_config(cfg) -> LMConfig:
    return LMConfig(**dataclasses.asdict(cfg))


def load_any_checkpoint(ckpt_path: str, device: str = "cuda"):
    """Load either a JEPA or LM checkpoint. Returns (model, kind, split_index)."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg_model = ckpt["cfg_model"]
    fields = {f.name for f in dataclasses.fields(cfg_model)}
    if "proj_dim" in fields:
        cfg = _to_jepa_config(cfg_model)
        model = JEPA(cfg).to(device)
        kind = "jepa"
    else:
        cfg = _to_lm_config(cfg_model)
        model = GPT(cfg).to(device)
        kind = "lm"
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, kind, cfg.split_index


def get_features(model, x: torch.Tensor, layer_idx: int) -> torch.Tensor:
    """Return (B, T, D) hidden state at the chosen depth.
    For JEPA: model.encode(x) returns the encoder output (depth = split_index).
    For GPT:  model.encode_at_layer(x, layer_idx) returns post-layer-k state.
    """
    if isinstance(model, JEPA):
        assert layer_idx == model.split, f"JEPA split is fixed at {model.split}, asked {layer_idx}"
        return model.encode(x)
    return model.encode_at_layer(x, layer_idx)


@torch.no_grad()
def encode_sst2_features(model, layer_idx: int, sentences: list[str], max_len: int = 64, batch_size: int = 64) -> torch.Tensor:
    import tiktoken
    enc = tiktoken.get_encoding("gpt2")
    device = next(model.parameters()).device
    out = []
    for i in range(0, len(sentences), batch_size):
        batch = sentences[i : i + batch_size]
        token_lists = []
        for s in batch:
            t = [BOS_ID] + enc.encode(s)
            token_lists.append(t[:max_len])
        max_T = max(len(t) for t in token_lists)
        max_T = max(max_T, 4)
        padded = torch.full((len(token_lists), max_T), BOS_ID, dtype=torch.int64, device=device)
        mask = torch.zeros(len(token_lists), max_T, dtype=torch.bool, device=device)
        for j, t in enumerate(token_lists):
            padded[j, : len(t)] = torch.tensor(t, device=device)
            mask[j, : len(t)] = True
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = get_features(model, padded, layer_idx)
        h = h.float()
        m = mask.unsqueeze(-1).float()
        pooled = (h * m).sum(dim=1) / m.sum(dim=1).clamp_min(1)
        out.append(pooled.cpu())
    return torch.cat(out, dim=0)


def linear_probe_classification(
    train_x: torch.Tensor,
    train_y: torch.Tensor,
    val_x: torch.Tensor,
    val_y: torch.Tensor,
    n_classes: int,
    epochs: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 0.01,
    batch_size: int = 256,
) -> dict:
    device = "cuda"
    D = train_x.shape[-1]
    head = nn.Linear(D, n_classes).to(device)
    nn.init.normal_(head.weight, std=0.02)
    nn.init.zeros_(head.bias)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    train_x = train_x.to(device)
    train_y = train_y.to(device)
    val_x = val_x.to(device)
    val_y = val_y.to(device)

    best_val_acc = 0.0
    for epoch in range(epochs):
        perm = torch.randperm(len(train_x), device=device)
        total_loss = 0.0
        for i in range(0, len(train_x), batch_size):
            idx = perm[i : i + batch_size]
            xb = train_x[idx]
            yb = train_y[idx]
            logits = head(xb)
            loss = F.cross_entropy(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(xb)
        with torch.no_grad():
            train_acc = (head(train_x).argmax(-1) == train_y).float().mean().item()
            val_acc = (head(val_x).argmax(-1) == val_y).float().mean().item()
        if val_acc > best_val_acc:
            best_val_acc = val_acc
    return {"train_acc": train_acc, "val_acc": val_acc, "best_val_acc": best_val_acc}


@torch.no_grad()
def position_cheating_on_features(model, layer_idx: int, val_loader, n_steps: int = 16) -> dict:
    accumulators = {k: [] for k in [
        "z_norm", "z_std_per_dim", "z_off_diag_cos_sim",
        "z_same_pos_cross_batch_sim", "z_position_cheating_ratio",
    ]}
    for _ in range(n_steps):
        x = next(val_loader)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = get_features(model, x, layer_idx)
        d = collapse_diagnostics(h)
        for k in accumulators:
            accumulators[k].append(d[k])
    return {k: torch.stack(v).mean().item() for k, v in accumulators.items()}


def next_token_probe(
    model,
    layer_idx: int,
    train_loader,
    val_loader,
    vocab_size: int,
    train_steps: int = 500,
    val_steps: int = 32,
    lr: float = 3e-4,
) -> dict:
    device = "cuda"
    D = model.cfg.model_dim
    head = nn.Linear(D, vocab_size, bias=False).to(device)
    nn.init.normal_(head.weight, std=0.02)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=0.0)

    for _ in range(train_steps):
        x = next(train_loader)
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            h = get_features(model, x, layer_idx)
        logits = head(h[:, :-1].float())
        target = x[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    head.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(val_steps):
            x = next(val_loader)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                h = get_features(model, x, layer_idx)
            logits = head(h[:, :-1].float())
            target = x[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
            val_losses.append(loss)
    val_loss = torch.stack(val_losses).mean().item()
    return {"probe_val_loss": val_loss, "probe_val_perplexity": math.exp(val_loss)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jepa-ckpt", required=True)
    ap.add_argument("--lm-ckpt", required=True)
    ap.add_argument("--train-glob", default="data/fineweb10B/fineweb_train_*.bin")
    ap.add_argument("--val-glob", default="data/fineweb10B/fineweb_val_*.bin")
    ap.add_argument("--probe-train-steps", type=int, default=500)
    ap.add_argument("--probe-val-steps", type=int, default=32)
    ap.add_argument("--sst2-epochs", type=int, default=20)
    ap.add_argument("--seqs-per-batch", type=int, default=8)
    ap.add_argument("--seq-len", type=int, default=1024)
    args = ap.parse_args()

    device = "cuda"
    torch.set_float32_matmul_precision("high")

    print(f"loading JEPA: {args.jepa_ckpt}")
    jepa, jepa_kind, jepa_layer = load_any_checkpoint(args.jepa_ckpt, device)
    print(f"  kind={jepa_kind} split_index={jepa_layer} model_dim={jepa.cfg.model_dim}")
    print(f"loading LM:   {args.lm_ckpt}")
    lm, lm_kind, lm_layer = load_any_checkpoint(args.lm_ckpt, device)
    print(f"  kind={lm_kind} split_index={lm_layer} model_dim={lm.cfg.model_dim}")
    assert jepa_kind == "jepa" and lm_kind == "lm"
    assert jepa_layer == lm_layer, "JEPA split_index must equal LM mid-layer index"
    assert jepa.cfg.model_dim == lm.cfg.model_dim, "model_dim mismatch"
    assert jepa.cfg.vocab_size == lm.cfg.vocab_size

    print("\n=== SST-2 sentiment linear probe ===")
    from datasets import load_dataset
    train = load_dataset("stanfordnlp/sst2", split="train")
    val = load_dataset("stanfordnlp/sst2", split="validation")
    train_sents = [r["sentence"].strip() for r in train]
    train_labels = torch.tensor([r["label"] for r in train], dtype=torch.long)
    val_sents = [r["sentence"].strip() for r in val]
    val_labels = torch.tensor([r["label"] for r in val], dtype=torch.long)
    print(f"  SST-2 train={len(train_sents)} val={len(val_sents)}")

    print("  encoding with JEPA encoder...")
    jepa_train_feat = encode_sst2_features(jepa, jepa_layer, train_sents)
    jepa_val_feat = encode_sst2_features(jepa, jepa_layer, val_sents)
    print(f"    JEPA train={jepa_train_feat.shape} val={jepa_val_feat.shape}")
    print("  encoding with LM mid-layer...")
    lm_train_feat = encode_sst2_features(lm, lm_layer, train_sents)
    lm_val_feat = encode_sst2_features(lm, lm_layer, val_sents)
    print(f"    LM   train={lm_train_feat.shape} val={lm_val_feat.shape}")

    print("  training linear classifiers...")
    jepa_sst = linear_probe_classification(jepa_train_feat, train_labels, jepa_val_feat, val_labels, n_classes=2, epochs=args.sst2_epochs)
    lm_sst = linear_probe_classification(lm_train_feat, train_labels, lm_val_feat, val_labels, n_classes=2, epochs=args.sst2_epochs)
    print(f"  SST-2 best val acc — JEPA: {jepa_sst['best_val_acc']:.4f} | LM mid: {lm_sst['best_val_acc']:.4f}")

    print("\n=== Position-cheating ratio on layer-{} features ===".format(jepa_layer))
    val_loader_diag_jepa = FineWebBatcher(args.val_glob, args.seqs_per_batch, args.seq_len, 0, 1, device, shuffle=False, seed=0)
    val_loader_diag_lm = FineWebBatcher(args.val_glob, args.seqs_per_batch, args.seq_len, 0, 1, device, shuffle=False, seed=0)
    jepa_diag = position_cheating_on_features(jepa, jepa_layer, val_loader_diag_jepa, n_steps=16)
    lm_diag = position_cheating_on_features(lm, lm_layer, val_loader_diag_lm, n_steps=16)
    print(f"  metric                              JEPA       LM")
    for k in ["z_norm", "z_std_per_dim", "z_off_diag_cos_sim", "z_same_pos_cross_batch_sim", "z_position_cheating_ratio"]:
        print(f"  {k:35s}  {jepa_diag[k]:8.4f}   {lm_diag[k]:8.4f}")

    print("\n=== Next-token CE linear probe on layer-{} features ===".format(jepa_layer))
    train_loader_jepa = FineWebBatcher(args.train_glob, args.seqs_per_batch, args.seq_len, 0, 1, device, shuffle=True, seed=42)
    val_loader_jepa = FineWebBatcher(args.val_glob, args.seqs_per_batch, args.seq_len, 0, 1, device, shuffle=False, seed=0)
    train_loader_lm = FineWebBatcher(args.train_glob, args.seqs_per_batch, args.seq_len, 0, 1, device, shuffle=True, seed=42)
    val_loader_lm = FineWebBatcher(args.val_glob, args.seqs_per_batch, args.seq_len, 0, 1, device, shuffle=False, seed=0)
    print("  probing JEPA features...")
    jepa_probe = next_token_probe(jepa, jepa_layer, train_loader_jepa, val_loader_jepa, jepa.cfg.vocab_size,
                                  train_steps=args.probe_train_steps, val_steps=args.probe_val_steps)
    print(f"    JEPA probe_val_loss={jepa_probe['probe_val_loss']:.4f} perplexity={jepa_probe['probe_val_perplexity']:.2f}")
    print("  probing LM mid-layer features...")
    lm_probe = next_token_probe(lm, lm_layer, train_loader_lm, val_loader_lm, lm.cfg.vocab_size,
                                train_steps=args.probe_train_steps, val_steps=args.probe_val_steps)
    print(f"    LM   probe_val_loss={lm_probe['probe_val_loss']:.4f} perplexity={lm_probe['probe_val_perplexity']:.2f}")

    print("\n=== Summary table ===\n")
    print(f"  {'metric':<35s}  {'JEPA':>10s}  {'LM mid':>10s}  {'winner':>10s}")
    print("  " + "-" * 70)
    rows = [
        ("SST-2 best val acc", jepa_sst["best_val_acc"], lm_sst["best_val_acc"], "higher"),
        ("Next-token probe val PPL", jepa_probe["probe_val_perplexity"], lm_probe["probe_val_perplexity"], "lower"),
        ("Position-cheating ratio", jepa_diag["z_position_cheating_ratio"], lm_diag["z_position_cheating_ratio"], "lower"),
        ("z_off_diag_cos_sim", jepa_diag["z_off_diag_cos_sim"], lm_diag["z_off_diag_cos_sim"], "lower"),
        ("z_std_per_dim", jepa_diag["z_std_per_dim"], lm_diag["z_std_per_dim"], "higher"),
    ]
    for name, jv, lv, direction in rows:
        if direction == "higher":
            winner = "JEPA" if jv > lv else "LM"
        else:
            winner = "JEPA" if jv < lv else "LM"
        print(f"  {name:<35s}  {jv:10.4f}  {lv:10.4f}  {winner:>10s}")


if __name__ == "__main__":
    main()
