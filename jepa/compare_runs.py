"""Pretty-print comparison of final val metrics across multiple training runs.

Greps the last "val_*" block from each log file and tabulates the metrics
side by side. Optionally also runs the linear probe on each checkpoint.

Usage:
  uv run python jepa/compare_runs.py
"""
import argparse
import re
import subprocess
import sys
from pathlib import Path


RUNS = [
    # (display_name, log_path, checkpoint_path, params_M)
    ("baseline-50k",   "/tmp/fineweb_50k.log",   "jepa/checkpoints/baseline_50k.pt",         51),
    ("baseline-500M",  "/tmp/fineweb_500M.log",  "jepa/checkpoints/baseline_500M_75k.pt",    51),
    ("large-12L-768D", "/tmp/fineweb_large.log", "jepa/checkpoints/large_step50000.pt",     124),
    ("ema-baseline",   "/tmp/fineweb_ema.log",   "jepa/checkpoints/ema_step50000.pt",        51),
    ("nope-baseline",  "/tmp/fineweb_nope.log",  "jepa/checkpoints/nope_step50000.pt",       51),
    ("lag4-baseline",  "/tmp/fineweb_lag4.log",  "jepa/checkpoints/lag4_step50000.pt",       51),
]

VAL_METRICS = [
    "val_loss",
    "val_top1",
    "val_top5",
    "val_z_norm",
    "val_z_std_per_dim",
    "val_z_off_diag_cos_sim",
    "val_z_same_pos_cross_batch_sim",
    "val_z_position_cheating_ratio",
]


def extract_last_val_block(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    text = log_path.read_text()
    metrics = {}
    for k in VAL_METRICS:
        matches = re.findall(rf"\b{k}=(-?[0-9.]+)", text)
        if matches:
            try:
                metrics[k] = float(matches[-1])
            except ValueError:
                pass
    return metrics


def fmt(v: float | None) -> str:
    if v is None:
        return "-"
    if abs(v) < 1.0:
        return f"{v:.4f}"
    return f"{v:.3f}"


def run_linear_probe(ckpt_path: Path, train_steps: int, val_steps: int) -> dict | None:
    if not ckpt_path.exists():
        return None
    res = subprocess.run(
        [
            "uv", "run", "python", "jepa/eval_probe.py",
            "--ckpt", str(ckpt_path),
            "--train-steps", str(train_steps),
            "--val-steps", str(val_steps),
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )
    if res.returncode != 0:
        return {"error": res.stderr.strip().splitlines()[-1] if res.stderr else "probe failed"}
    out = {}
    for line in res.stdout.splitlines():
        m = re.match(r"\s*(probe_\w+)=([-0-9.eE+]+)", line)
        if m:
            try:
                out[m.group(1)] = float(m.group(2))
            except ValueError:
                pass
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--probe", action="store_true", help="also run linear probe on each checkpoint")
    ap.add_argument("--probe-train-steps", type=int, default=500)
    ap.add_argument("--probe-val-steps", type=int, default=32)
    args = ap.parse_args()

    rows = []
    for name, log, ckpt, params_m in RUNS:
        m = extract_last_val_block(Path(log))
        rows.append((name, params_m, m))

    print("\n=== Final val metrics ===\n")
    cols = ["run", "params(M)"] + VAL_METRICS
    widths = [max(len(c), 14) for c in cols]
    for c, w in zip(cols, widths):
        print(c.ljust(w + 1), end=" ")
    print()
    print("-" * (sum(widths) + len(cols) * 2))
    for name, params_m, m in rows:
        cells = [name, str(params_m)] + [fmt(m.get(k)) for k in VAL_METRICS]
        for c, w in zip(cells, widths):
            print(c.ljust(w + 1), end=" ")
        print()

    if args.probe:
        print(f"\n=== Linear probe (train_steps={args.probe_train_steps}, val_steps={args.probe_val_steps}) ===\n")
        print(f"{'run':20s} {'probe_val_loss':>16s} {'probe_perplexity':>20s}")
        print("-" * 58)
        for name, _params_m, _m in rows:
            ckpt = Path(dict((n, c) for n, _, c, _ in RUNS)[name])
            res = run_linear_probe(ckpt, args.probe_train_steps, args.probe_val_steps)
            if res is None:
                print(f"{name:20s} {'(missing ckpt)':>16s}")
            elif "error" in res:
                print(f"{name:20s} ERROR: {res['error']}")
            else:
                pv = res.get("probe_val_loss")
                pp = res.get("probe_val_perplexity")
                print(f"{name:20s} {fmt(pv):>16s} {fmt(pp):>20s}")


if __name__ == "__main__":
    main()
