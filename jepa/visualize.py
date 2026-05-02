"""Visualize a trained JEPA on a single sentence: self-similarity matrices + predictor-vs-encoder cross-sim.

Usage:
  uv run python jepa/visualize.py --ckpt jepa/checkpoints/final_step50000.pt
  uv run python jepa/visualize.py --ckpt ... --text "Custom sentence to inspect."
"""
import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tiktoken
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa.eval_probe import load_jepa
from jepa.model import apply_rope, rms_norm

BOS_ID = 50256
DEFAULT_TEXT = (
    "The cat sat on the mat. The dog sat on the rug. "
    "Cats and dogs are common household pets. "
    "Many people prefer cats because they are independent."
)


@torch.no_grad()
def extract_encoder_attention_maps(model, x: torch.Tensor) -> list[torch.Tensor]:
    """Manually run encoder layers, replicating the model's attention but
    capturing the post-softmax attention weights at each encoder layer.
    Returns a list of (n_heads, T, T) tensors, one per encoder block.
    """
    cfg = model.cfg
    cos, sin = model.rope_cos, model.rope_sin
    h = rms_norm(model.embed(x))
    attn_maps = []
    for i in range(model.split):
        block = model.blocks[i]
        attn = block.attn
        x_norm = rms_norm(h)
        B, T, _ = x_norm.shape
        q, k, v = attn.qkv(x_norm).chunk(3, dim=-1)
        q = q.view(B, T, attn.n_heads, attn.head_dim)
        k = k.view(B, T, attn.n_heads, attn.head_dim)
        v = v.view(B, T, attn.n_heads, attn.head_dim)
        q = rms_norm(q)
        k = rms_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        scale = attn.head_dim ** -0.5
        scores = (q.float() @ k.float().transpose(-1, -2)) * scale
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=x.device), diagonal=1)
        scores = scores.masked_fill(causal_mask, float("-inf"))
        attn_w = F.softmax(scores, dim=-1)
        attn_maps.append(attn_w[0].detach().cpu())
        y = (attn_w.to(v.dtype) @ v).transpose(1, 2).contiguous().view(B, T, attn.n_heads * attn.head_dim)
        y = attn.out(y)
        x_after_attn = h + y
        x_norm2 = rms_norm(x_after_attn)
        mlp_out = block.mlp_w2(F.relu(block.mlp_w1(x_norm2)).square())
        h = x_after_attn + mlp_out
    return attn_maps


def analyze_semantic_attention(
    attn_maps: list[torch.Tensor],
    token_strs: list[str],
    output_dir: Path,
    query_tokens: list[str] | None = None,
    top_k: int = 4,
):
    """Strip BOS attention sink and inspect per-token attention targets.

    Two artifacts:
      - attn_no_bos.png   : per-layer attention with column 0 (BOS) zeroed +
                            row-renormalized, mean-over-heads. Shows the
                            "post-sink" residual pattern.
      - semantic_attn.txt : for each query token in `query_tokens`, prints
                            its top-k attended tokens at every (layer, head),
                            excluding the BOS column.
    """
    n_layers = len(attn_maps)
    n_heads = attn_maps[0].shape[0]
    T = attn_maps[0].shape[1]

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4.5), squeeze=False)
    for li, attn in enumerate(attn_maps):
        ax = axes[0, li]
        a = attn.clone()
        a[:, :, 0] = 0
        s = a.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        a = a / s
        avg = a.mean(dim=0).numpy()
        im = ax.imshow(avg, cmap="viridis", vmin=0, vmax=avg.max() if avg.max() > 0 else 1.0, aspect="equal")
        ax.set_title(f"layer {li} — BOS column zeroed, renormalized, mean over heads")
        ax.set_xlabel("attended-to position j")
        if li == 0:
            ax.set_ylabel("query position i")
        if T <= 40:
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(token_strs, rotation=90, fontsize=6)
            ax.set_yticklabels(token_strs, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_dir / "attn_no_bos.png", dpi=120, bbox_inches="tight")
    plt.close()

    if query_tokens is None:
        query_tokens = []
        for s in ["cat", "dog", "mat", "rug", "Cats", "dogs", "pets", "cats", "they", "people"]:
            for i, t in enumerate(token_strs):
                if t.strip() == s and i not in [j for j, q in query_tokens]:
                    query_tokens.append((i, t))
                    break
    else:
        query_tokens = [(i, t) for s in query_tokens for i, t in enumerate(token_strs) if t.strip() == s][:10]

    out_lines = []
    out_lines.append(f"Semantic attention summary (T={T} tokens; BOS column excluded)")
    out_lines.append(f"Tokens: {' '.join(f'{i}:{repr(t)}' for i, t in enumerate(token_strs))}")
    out_lines.append("")
    for q_idx, q_tok in query_tokens:
        out_lines.append(f"=== query position {q_idx} ('{q_tok}') ===")
        for li, attn in enumerate(attn_maps):
            row = attn[:, q_idx].clone()
            row[:, 0] = 0
            s = row.sum(dim=-1, keepdim=True).clamp_min(1e-9)
            row = row / s
            mean_row = row.mean(dim=0)
            top_vals, top_idx = mean_row.topk(min(top_k, T))
            entries = [f"{token_strs[j]!r}@{j}={top_vals[k].item():.3f}" for k, j in enumerate(top_idx.tolist()) if top_vals[k].item() > 0.01]
            out_lines.append(f"  layer {li} (head-mean): " + ", ".join(entries) if entries else f"  layer {li}: (all attention <1% on non-BOS)")
        out_lines.append("")
    out_path = output_dir / "semantic_attn.txt"
    out_path.write_text("\n".join(out_lines))


def plot_attention_maps(attn_maps: list[torch.Tensor], token_strs: list[str], output_dir: Path):
    n_layers = len(attn_maps)
    n_heads = attn_maps[0].shape[0]
    T = attn_maps[0].shape[1]

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4.5), squeeze=False)
    for li, attn in enumerate(attn_maps):
        ax = axes[0, li]
        avg = attn.mean(dim=0).numpy()
        im = ax.imshow(avg, cmap="viridis", vmin=0, vmax=avg.max(), aspect="equal")
        ax.set_title(f"layer {li} — mean over {n_heads} heads")
        ax.set_xlabel("attended-to position j")
        if li == 0:
            ax.set_ylabel("query position i")
        if T <= 40:
            ax.set_xticks(range(T))
            ax.set_yticks(range(T))
            ax.set_xticklabels(token_strs, rotation=90, fontsize=6)
            ax.set_yticklabels(token_strs, fontsize=6)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_dir / "attn_per_layer.png", dpi=120, bbox_inches="tight")
    plt.close()

    fig, axes = plt.subplots(n_layers, n_heads, figsize=(2.4 * n_heads, 2.4 * n_layers), squeeze=False)
    for li, attn in enumerate(attn_maps):
        for hi in range(n_heads):
            ax = axes[li, hi]
            mat = attn[hi].numpy()
            ax.imshow(mat, cmap="viridis", vmin=0, vmax=mat.max(), aspect="equal")
            ax.set_title(f"L{li} H{hi}", fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
    plt.suptitle(f"Encoder attention per (layer, head) — query rows attend to columns; T={T}", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "attn_per_head.png", dpi=120, bbox_inches="tight")
    plt.close()


def visualize(ckpt_path: str, text: str, output_dir: str, max_tokens: int = 64):
    device = "cuda"
    torch.set_float32_matmul_precision("high")
    model, ckpt = load_jepa(ckpt_path, device)
    model.eval()

    enc = tiktoken.get_encoding("gpt2")
    tokens = [BOS_ID] + enc.encode(text)
    tokens = tokens[:max_tokens]
    T = len(tokens)
    print(f"loaded {ckpt_path} (step={ckpt['step']}, model_dim={model.cfg.model_dim})")
    print(f"encoded {T} tokens; decoded='{enc.decode(tokens)}'")

    x = torch.tensor(tokens, device=device, dtype=torch.long).unsqueeze(0)
    with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        p, z = model(x)
        h = model.encode(x)
    p, z, h = p[0].float(), z[0].float(), h[0].float()

    p_n = F.normalize(p, dim=-1)
    z_n = F.normalize(z, dim=-1)
    h_n = F.normalize(h, dim=-1)

    z_self = (z_n @ z_n.T).cpu().numpy()
    h_self = (h_n @ h_n.T).cpu().numpy()
    pz_cross = (p_n @ z_n.T).cpu().numpy()

    token_strs = []
    for t in tokens:
        s = enc.decode([t])
        s = s.replace("\n", "\\n")
        if len(s) > 8:
            s = s[:7] + "…"
        token_strs.append(s)

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    for ax, mat, title in [
        (axes[0], h_self, "Encoder hidden state h (model_dim=512) self-similarity"),
        (axes[1], z_self, "Projected latent z (proj_dim=128) self-similarity"),
    ]:
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
        ax.set_title(f"{title}\ncos(·,·); diagonal=self=1.0")
        ax.set_xticks(range(T))
        ax.set_yticks(range(T))
        ax.set_xticklabels(token_strs, rotation=90, fontsize=7)
        ax.set_yticklabels(token_strs, fontsize=7)
        plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_dir / "self_sim.png", dpi=120, bbox_inches="tight")
    plt.close()

    fig, ax = plt.subplots(figsize=(11, 10))
    im = ax.imshow(pz_cross, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
    ax.set_title(
        f"Predictor p_i (rows) vs Encoder z_j (cols), cos(·,·)\n"
        f"Healthy: brightest cell in row i is at column i+1 (next-position match)"
    )
    ax.set_xticks(range(T))
    ax.set_yticks(range(T))
    ax.set_xticklabels(token_strs, rotation=90, fontsize=7)
    ax.set_yticklabels(token_strs, fontsize=7)
    if T > 1:
        ax.plot(np.arange(1, T) + 0.5, np.arange(0, T - 1) + 0.5, "g-", linewidth=1.5, alpha=0.8, label="j = i+1 (target diagonal)")
        ax.legend(loc="upper right")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(output_dir / "pred_vs_enc.png", dpi=120, bbox_inches="tight")
    plt.close()

    p_at = p_n[:-1]
    z_tgt = z_n[1:]
    sim = p_at @ z_tgt.T
    pred = sim.argmax(dim=-1).cpu().numpy()
    truth = np.arange(T - 1)
    offsets = pred - truth
    correct = (pred == truth).mean()

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    axes[0].hist(offsets, bins=range(int(offsets.min()) - 1, int(offsets.max()) + 2), align="left", color="steelblue", edgecolor="black")
    axes[0].axvline(0, color="green", linestyle="--", label="correct (offset=0)")
    axes[0].set_xlabel("argmax(j) − true_j  (true_j = i+1)")
    axes[0].set_ylabel("count")
    axes[0].set_title(f"Within-sentence retrieval offset distribution\ntop-1 accuracy = {correct:.3f}")
    axes[0].legend()

    z_norms = z.norm(dim=-1).cpu().numpy()
    p_norms = p.norm(dim=-1).cpu().numpy()
    h_norms = h.norm(dim=-1).cpu().numpy()
    ax2 = axes[1]
    ax2.plot(z_norms, label="‖z‖ (projected)", color="C0")
    ax2.plot(p_norms, label="‖p‖ (predictor proj)", color="C1")
    ax2_twin = ax2.twinx()
    ax2_twin.plot(h_norms, label="‖h‖ (encoder hidden)", color="C2", linestyle="--")
    ax2.set_xlabel("position")
    ax2.set_ylabel("proj-space norm")
    ax2_twin.set_ylabel("hidden norm")
    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="best")
    ax2.set_title("Latent magnitudes per position")
    plt.tight_layout()
    plt.savefig(output_dir / "retrieval_and_norms.png", dpi=120, bbox_inches="tight")
    plt.close()

    attn_maps = extract_encoder_attention_maps(model, x)
    plot_attention_maps(attn_maps, token_strs, output_dir)
    analyze_semantic_attention(attn_maps, token_strs, output_dir)

    print(f"\nWithin-sentence retrieval: {correct:.3f} top-1 ({(pred == truth).sum()}/{len(truth)})")
    print(f"saved figures to {output_dir.resolve()}/")
    print(f"  - self_sim.png        : encoder hidden + projected latent self-similarity")
    print(f"  - pred_vs_enc.png     : predictor vs encoder cross-similarity (the JEPA target diagonal)")
    print(f"  - retrieval_and_norms.png : within-sentence retrieval histogram + per-position norms")
    print(f"  - attn_per_layer.png  : encoder attention, mean over heads, per layer")
    print(f"  - attn_per_head.png   : encoder attention, every (layer, head)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--text", default=DEFAULT_TEXT)
    ap.add_argument("--output-dir", default="jepa/figs")
    ap.add_argument("--max-tokens", type=int, default=64)
    args = ap.parse_args()
    visualize(args.ckpt, args.text, args.output_dir, args.max_tokens)
