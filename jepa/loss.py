import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor


def all_gather_no_grad(x: Tensor) -> Tensor:
    if not dist.is_initialized() or dist.get_world_size() == 1:
        return x
    world = dist.get_world_size()
    out = [torch.empty_like(x) for _ in range(world)]
    dist.all_gather(out, x)
    return torch.cat(out, dim=0)


def infonce_loss(
    p: Tensor,
    z: Tensor,
    tau: float = 0.1,
    subsample_cap: int = 8192,
) -> tuple[Tensor, dict]:
    """
    InfoNCE on next-position latents.

    p: (B, T, D) predictor outputs
    z: (B, T, D) encoder latents

    For each position t in [0, T-1), match predictor(p_t) to stop_grad(z_{t+1}).
    Positives = matched pairs (diagonal). Negatives = all other (rank, batch, position)
    triples in the all-gathered target pool.
    """
    B, T, D = p.shape
    p_at_t = p[:, :-1].reshape(-1, D)
    z_tgt_local = z[:, 1:].detach().reshape(-1, D)

    p_at_t = F.normalize(p_at_t, dim=-1)
    z_tgt_local = F.normalize(z_tgt_local, dim=-1)

    z_neg = all_gather_no_grad(z_tgt_local)

    n_local = z_tgt_local.size(0)
    rank = dist.get_rank() if dist.is_initialized() else 0
    local_offset = rank * n_local

    if p_at_t.size(0) > subsample_cap:
        idx = torch.randperm(p_at_t.size(0), device=p_at_t.device)[:subsample_cap]
        p_at_t_sub = p_at_t[idx]
        labels = idx + local_offset
    else:
        p_at_t_sub = p_at_t
        labels = torch.arange(p_at_t.size(0), device=p_at_t.device) + local_offset

    logits = (p_at_t_sub @ z_neg.T).float() / tau
    loss = F.cross_entropy(logits, labels)

    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        top1 = (pred == labels).float().mean()
        top5 = (logits.topk(5, dim=-1).indices == labels[:, None]).any(dim=-1).float().mean()
        diag_sim = (p_at_t_sub * z_neg[labels]).sum(dim=-1).mean()
        n_anchors = labels.size(0)
        n_neg = z_neg.size(0)

    metrics = {
        "loss": loss.detach(),
        "top1": top1,
        "top5": top5,
        "diag_cos_sim": diag_sim,
        "n_anchors": n_anchors,
        "n_negatives": n_neg,
    }
    return loss, metrics


def collapse_diagnostics(z: Tensor) -> dict:
    """Collapse detectors. z: (B, T, D)."""
    z_flat = z.reshape(-1, z.size(-1))
    with torch.no_grad():
        z_norm = z_flat.norm(dim=-1).mean()
        z_std_per_dim = z_flat.float().std(dim=0).mean()
        z_n = F.normalize(z_flat, dim=-1)
        sim_matrix = z_n @ z_n.T
        n = sim_matrix.size(0)
        off_diag_mask = ~torch.eye(n, dtype=torch.bool, device=z.device)
        off_diag_mean = sim_matrix[off_diag_mask].mean()
    return {
        "z_norm": z_norm,
        "z_std_per_dim": z_std_per_dim,
        "z_off_diag_cos_sim": off_diag_mean,
    }
