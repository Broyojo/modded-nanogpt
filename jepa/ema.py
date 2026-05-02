"""EMA (exponential-moving-average) target encoder for JEPA.

Maintains a slowly-updated copy of the online model's encoder + projection head.
Each training step:
  1. Run online model -> get p (predictor output)
  2. Run target model encoder-only -> get z_target (no grad)
  3. InfoNCE(p, z_target)
  4. After opt.step(), EMA update: target = momentum * target + (1-momentum) * online
"""
import copy

import torch
import torch.nn.functional as F

from jepa.model import JEPA, rms_norm


class EMATarget:
    def __init__(self, online_model: JEPA, momentum: float = 0.999):
        if hasattr(online_model, "_orig_mod"):
            online_model = online_model._orig_mod
        self.online = online_model
        self.target = copy.deepcopy(online_model).eval()
        for p in self.target.parameters():
            p.requires_grad_(False)
        self.momentum = momentum

    @torch.no_grad()
    def update(self):
        m = self.momentum
        for p_t, p_o in zip(self.target.parameters(), self.online.parameters()):
            p_t.data.mul_(m).add_(p_o.data, alpha=1.0 - m)

    @torch.no_grad()
    def target_z(self, input_seq: torch.Tensor) -> torch.Tensor:
        """Run target encoder + proj only (no predictor). Returns (B, T, proj_dim)."""
        cos, sin = self.target.rope_cos, self.target.rope_sin
        h = rms_norm(self.target.embed(input_seq))
        for blk in self.target.blocks[: self.target.split]:
            h = blk(h, cos, sin)
        return self.target.proj(rms_norm(h))
