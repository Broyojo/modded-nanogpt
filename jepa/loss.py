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


@torch.no_grad()
def collapse_diagnostics(z: Tensor) -> dict:
    """Collapse + position-cheating diagnostics on encoder latents. z: (B, T, D).

    Computed via the sum-of-vectors trick to avoid materializing the full
    (B*T) x (B*T) similarity matrix:
      sum_{i,j} <z_i, z_j>  =  || sum_i z_i ||^2
      sum_{i}   <z_i, z_i>  =  n   (when each z_i is L2-normalized)
      off-diag mean         =  (||sum z||^2 - n) / (n*(n-1))

    Returns:
      z_norm                       — mean L2 norm of z (norm-collapse signal)
      z_std_per_dim                — per-dim std averaged (variance-collapse signal)
      z_off_diag_cos_sim           — mean cos sim over all (b,t) != (b',t') pairs
      z_same_pos_cross_batch_sim   — mean cos sim of z_t[a] vs z_t[b] for a != b at the same position t
      z_position_cheating_ratio    — same_pos - off_diag. Significantly > 0 means
                                     latents at the same position across different
                                     sequences cluster, i.e. position info is encoded
                                     more strongly than content (= position cheating).
    """
    B, T, D = z.shape
    z_f = z.float()
    z_norm = z_f.norm(dim=-1).mean()
    z_std_per_dim = z_f.reshape(-1, D).std(dim=0).mean()

    z_n = F.normalize(z_f, dim=-1)

    n = B * T
    z_flat_sum = z_n.reshape(n, D).sum(dim=0)
    sum_all = (z_flat_sum * z_flat_sum).sum()
    off_diag_all = (sum_all - n) / max(1, n * (n - 1))

    if B > 1:
        z_sum_per_pos = z_n.sum(dim=0)
        sum_per_pos = (z_sum_per_pos * z_sum_per_pos).sum(dim=-1)
        off_diag_per_pos = (sum_per_pos - B) / (B * (B - 1))
        same_pos_mean = off_diag_per_pos.mean()
    else:
        same_pos_mean = z_f.new_zeros(())

    return {
        "z_norm": z_norm,
        "z_std_per_dim": z_std_per_dim,
        "z_off_diag_cos_sim": off_diag_all,
        "z_same_pos_cross_batch_sim": same_pos_mean,
        "z_position_cheating_ratio": same_pos_mean - off_diag_all,
    }
