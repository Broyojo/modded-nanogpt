from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


def rms_norm(x: Tensor) -> Tensor:
    return F.rms_norm(x, (x.size(-1),))


def precompute_rope(head_dim: int, max_seq_len: int, base: float = 10000.0, device=None) -> tuple[Tensor, Tensor]:
    assert head_dim % 2 == 0
    inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim))
    t = torch.arange(max_seq_len, dtype=torch.float32, device=device)
    freqs = torch.outer(t, inv_freq)
    return freqs.cos(), freqs.sin()


def apply_rope(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
    """x: (B, T, H, D_head). cos/sin: (T_max, D_head/2)."""
    T = x.shape[-3]
    cos = cos[:T, None, :].to(x.dtype)
    sin = sin[:T, None, :].to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class CausalSelfAttention(nn.Module):
    def __init__(self, model_dim: int, n_heads: int, head_dim: int):
        super().__init__()
        assert n_heads * head_dim == model_dim, f"{n_heads}*{head_dim} != {model_dim}"
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.qkv = nn.Linear(model_dim, 3 * model_dim, bias=False)
        self.out = nn.Linear(model_dim, model_dim, bias=False)
        nn.init.zeros_(self.out.weight)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        B, T, _ = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q.view(B, T, self.n_heads, self.head_dim)
        k = k.view(B, T, self.n_heads, self.head_dim)
        v = v.view(B, T, self.n_heads, self.head_dim)
        q = rms_norm(q)
        k = rms_norm(k)
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        return self.out(y)


class Block(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.attn = CausalSelfAttention(cfg.model_dim, cfg.n_heads, cfg.head_dim)
        self.mlp_w1 = nn.Linear(cfg.model_dim, 4 * cfg.model_dim, bias=False)
        self.mlp_w2 = nn.Linear(4 * cfg.model_dim, cfg.model_dim, bias=False)
        nn.init.zeros_(self.mlp_w2.weight)

    def forward(self, x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:
        x = x + self.attn(rms_norm(x), cos, sin)
        x = x + self.mlp_w2(F.relu(self.mlp_w1(rms_norm(x))).square())
        return x


class JEPA(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        for blk in self.blocks:
            nn.init.normal_(blk.attn.qkv.weight, mean=0.0, std=0.02)
            nn.init.normal_(blk.mlp_w1.weight, mean=0.0, std=0.02)
        self.split = cfg.split_index
        self.proj = nn.Linear(cfg.model_dim, cfg.proj_dim, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        cos, sin = precompute_rope(cfg.head_dim, cfg.seq_len, cfg.rope_base)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

    def forward(self, input_seq: Tensor) -> tuple[Tensor, Tensor]:
        """input_seq: (B, T) int64. Returns (p, z), each (B, T, proj_dim).

        z = proj(rms_norm(encoder(x)))         — target side, will be detached in loss
        p = proj(rms_norm(predictor(encoder(x)))) — prediction side, gradient flows
        """
        x = rms_norm(self.embed(input_seq))
        cos, sin = self.rope_cos, self.rope_sin
        for blk in self.blocks[:self.split]:
            x = blk(x, cos, sin)
        h_enc = x
        for blk in self.blocks[self.split:]:
            x = blk(x, cos, sin)
        h_pred = x
        z = self.proj(rms_norm(h_enc))
        p = self.proj(rms_norm(h_pred))
        return p, z

    def encode(self, input_seq: Tensor) -> Tensor:
        """Encoder-only forward, returns raw (B, T, model_dim) hidden state for downstream probes."""
        x = rms_norm(self.embed(input_seq))
        cos, sin = self.rope_cos, self.rope_sin
        for blk in self.blocks[:self.split]:
            x = blk(x, cos, sin)
        return rms_norm(x)
