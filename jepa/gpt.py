"""Plain GPT (next-token prediction) baseline using the same Block/RoPE/etc as JEPA.

Identical architecture to jepa.model.JEPA but:
  - no projection head
  - LM head with weights tied to the embedding
  - forward() returns logits over vocabulary
  - encode_at_layer(x, k) returns the hidden state after the k-th block
    (used for the head-to-head feature comparison vs JEPA's encoder output).
"""
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from jepa.model import Block, precompute_rope, rms_norm


class GPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.model_dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=0.02)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layers)])
        for blk in self.blocks:
            nn.init.normal_(blk.attn.qkv.weight, mean=0.0, std=0.02)
            nn.init.normal_(blk.mlp_w1.weight, mean=0.0, std=0.02)
        if getattr(cfg, "use_rope", True):
            cos, sin = precompute_rope(cfg.head_dim, cfg.seq_len, cfg.rope_base)
            self.register_buffer("rope_cos", cos, persistent=False)
            self.register_buffer("rope_sin", sin, persistent=False)
        else:
            self.rope_cos = None
            self.rope_sin = None

    def forward(self, input_seq: Tensor) -> Tensor:
        """input_seq: (B, T) int64. Returns (B, T, vocab_size) logits."""
        x = rms_norm(self.embed(input_seq))
        cos, sin = self.rope_cos, self.rope_sin
        for blk in self.blocks:
            x = blk(x, cos, sin)
        x = rms_norm(x)
        return x @ self.embed.weight.T

    def encode_at_layer(self, input_seq: Tensor, layer_idx: int) -> Tensor:
        """Run forward up to `layer_idx` blocks; return rms_norm(hidden) at that depth.

        layer_idx == 0 returns the embedded input. layer_idx == n_layers returns
        the final pre-LM hidden state.
        """
        assert 0 <= layer_idx <= len(self.blocks)
        x = rms_norm(self.embed(input_seq))
        cos, sin = self.rope_cos, self.rope_sin
        for i, blk in enumerate(self.blocks):
            if i == layer_idx:
                break
            x = blk(x, cos, sin)
        return rms_norm(x)
