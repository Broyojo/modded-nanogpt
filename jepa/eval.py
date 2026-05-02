import math
from typing import Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from jepa.loss import collapse_diagnostics, infonce_loss


@torch.no_grad()
def evaluate_val(
    model: nn.Module,
    val_loader: Iterator[Tensor],
    n_steps: int,
    tau: float = 0.1,
    subsample_cap: int = 8192,
) -> dict:
    model.eval()
    losses, top1s, top5s, diag_sims = [], [], [], []
    z_norms, z_stds, z_off_diags = [], [], []
    for _ in range(n_steps):
        x = next(val_loader)
        p, z = model(x)
        loss, m = infonce_loss(p, z, tau=tau, subsample_cap=subsample_cap)
        d = collapse_diagnostics(z)
        losses.append(loss.detach())
        top1s.append(m["top1"])
        top5s.append(m["top5"])
        diag_sims.append(m["diag_cos_sim"])
        z_norms.append(d["z_norm"])
        z_stds.append(d["z_std_per_dim"])
        z_off_diags.append(d["z_off_diag_cos_sim"])
    model.train()
    return {
        "val_loss": torch.stack(losses).mean().item(),
        "val_top1": torch.stack(top1s).mean().item(),
        "val_top5": torch.stack(top5s).mean().item(),
        "val_diag_cos_sim": torch.stack(diag_sims).mean().item(),
        "val_z_norm": torch.stack(z_norms).mean().item(),
        "val_z_std_per_dim": torch.stack(z_stds).mean().item(),
        "val_z_off_diag_cos_sim": torch.stack(z_off_diags).mean().item(),
    }


def linear_probe(
    model: nn.Module,
    train_loader: Iterator[Tensor],
    val_loader: Iterator[Tensor],
    vocab_size: int,
    train_steps: int = 200,
    val_steps: int = 16,
    lr: float = 3e-4,
    device: torch.device | str = "cuda",
) -> dict:
    """Frozen-encoder next-token linear probe. Trains a Linear(model_dim → vocab) head on z_t → token_{t+1} for `train_steps`, then reports val perplexity."""
    model.eval()
    probe = nn.Linear(model.cfg.model_dim, vocab_size, bias=False).to(device)
    nn.init.normal_(probe.weight, std=0.02)
    opt = torch.optim.AdamW(probe.parameters(), lr=lr, weight_decay=0.0)

    for _ in range(train_steps):
        x = next(train_loader)
        with torch.no_grad():
            z = model.encode(x)
        logits = probe(z[:, :-1].float())
        target = x[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    probe.eval()
    val_losses = []
    with torch.no_grad():
        for _ in range(val_steps):
            x = next(val_loader)
            z = model.encode(x)
            logits = probe(z[:, :-1].float())
            target = x[:, 1:]
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target.reshape(-1))
            val_losses.append(loss)

    model.train()
    val_loss = torch.stack(val_losses).mean().item()
    return {
        "probe_val_loss": val_loss,
        "probe_val_perplexity": math.exp(val_loss),
    }
