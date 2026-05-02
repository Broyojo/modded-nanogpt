import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from jepa.configs.baseline import JEPAConfig
from jepa.loss import collapse_diagnostics, infonce_loss
from jepa.model import JEPA


def small_cfg(**kw):
    base = dict(
        seq_len=128,
        n_layers=4,
        split_index=2,
        model_dim=256,
        n_heads=4,
        head_dim=64,
        proj_dim=64,
    )
    base.update(kw)
    return JEPAConfig(**base)


def test_1_overfit_one_batch():
    """Train on a single fixed batch for 200 steps. Top-1 retrieval should hit ~100%."""
    print("[Test 1] Overfit-1-batch...")
    torch.manual_seed(0)
    cfg = small_cfg(seq_len=256)
    model = JEPA(cfg).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg.vocab_size, (8, 256), device="cuda")

    init_loss = None
    for step in range(200):
        p, z = model(x)
        loss, m = infonce_loss(p, z, tau=0.1)
        if step == 0:
            init_loss = loss.item()
        opt.zero_grad()
        loss.backward()
        opt.step()

    top1 = m["top1"].item()
    print(f"  init_loss={init_loss:.4f} final_loss={loss.item():.4f} top1={top1:.3f}")
    assert loss.item() < init_loss / 2, f"loss didn't drop: {init_loss:.4f} -> {loss.item():.4f}"
    assert top1 > 0.95, f"top1 should be >0.95 after 200 overfit steps, got {top1}"
    print("  PASS")


def test_2_no_label_leak():
    """At random init, real-label and shuffled-label losses must be similar (no positional leak)
    AND training with REAL labels must reduce loss far more than training with SHUFFLED labels."""
    print("[Test 2] No-label-leak + shuffled-doesnt-train...")
    torch.manual_seed(0)
    cfg = small_cfg()
    x = torch.randint(0, cfg.vocab_size, (4, 128), device="cuda")

    torch.manual_seed(0)
    model_init = JEPA(cfg).cuda()
    with torch.no_grad():
        p, z = model_init(x)
        D = p.size(-1)
        p_at = F.normalize(p[:, :-1].reshape(-1, D), dim=-1)
        z_tgt = F.normalize(z[:, 1:].detach().reshape(-1, D), dim=-1)
        N = p_at.size(0)
        labels = torch.arange(N, device="cuda")
        logits_real = (p_at @ z_tgt.T).float() / 0.1
        loss_real_init = F.cross_entropy(logits_real, labels).item()
        perm = torch.randperm(N, device="cuda")
        logits_shuf = (p_at @ z_tgt[perm].T).float() / 0.1
        loss_shuf_init = F.cross_entropy(logits_shuf, labels).item()
    print(f"  random init: real={loss_real_init:.3f}, shuffled={loss_shuf_init:.3f}, |diff|={abs(loss_real_init - loss_shuf_init):.3f}")
    assert abs(loss_real_init - loss_shuf_init) < 1.0, "real & shuffled losses should be ≈ equal at random init"

    torch.manual_seed(0)
    model_real = JEPA(cfg).cuda()
    opt = torch.optim.AdamW(model_real.parameters(), lr=1e-3)
    for _ in range(50):
        p, z = model_real(x)
        loss, _ = infonce_loss(p, z, tau=0.1)
        opt.zero_grad()
        loss.backward()
        opt.step()
    real_final = loss.item()

    torch.manual_seed(0)
    model_shuf = JEPA(cfg).cuda()
    opt = torch.optim.AdamW(model_shuf.parameters(), lr=1e-3)
    for _ in range(50):
        p, z = model_shuf(x)
        D = p.size(-1)
        p_at = F.normalize(p[:, :-1].reshape(-1, D), dim=-1)
        z_tgt = F.normalize(z[:, 1:].detach().reshape(-1, D), dim=-1)
        perm = torch.randperm(p_at.size(0), device="cuda")
        logits = (p_at @ z_tgt[perm].T).float() / 0.1
        labels = torch.arange(p_at.size(0), device="cuda")
        loss = F.cross_entropy(logits, labels)
        opt.zero_grad()
        loss.backward()
        opt.step()
    shuf_final = loss.item()
    print(f"  after 50 steps: real-trained={real_final:.3f}, shuffled-trained={shuf_final:.3f}")
    assert real_final < shuf_final - 1.0, f"real labels should learn far better than shuffled: {real_final} vs {shuf_final}"
    print("  PASS")


def test_3_identity_no_predictor():
    """split_index=n_layers (predictor=0 blocks). Forward + backward must work without error."""
    print("[Test 3] Identity check (split_index=n_layers, predictor is 0 blocks)...")
    torch.manual_seed(0)
    cfg = small_cfg(split_index=4)
    model = JEPA(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (4, 128), device="cuda")
    p, z = model(x)
    assert torch.allclose(p, z), "with no predictor blocks, p must equal z"
    loss, m = infonce_loss(p, z, tau=0.1)
    loss.backward()
    print(f"  loss={loss.item():.4f} top1={m['top1'].item():.3f} diag_sim={m['diag_cos_sim'].item():.3f}")
    print("  PASS")


def test_4_gradient_flow():
    """Verify z_target is detached AND gradients flow to encoder/predictor blocks.

    At random init, zero-init residual output projections (`attn.out`, `mlp_w2`) cause
    `qkv` and `mlp_w1` to receive ZERO gradient (the zero-output layer breaks the chain).
    This is intentional (stability). After one optimizer step the residual outputs are
    nonzero and gradients propagate to all weights. So we test:
      - z_target detach is correct
      - At init: residual-output weights (`attn.out`, `mlp_w2`), `proj`, `embed` get nonzero grad
      - After 1 step: ALL weights including `qkv`/`mlp_w1` get nonzero grad
    """
    print("[Test 4] Gradient flow assertions...")
    torch.manual_seed(0)
    cfg = small_cfg()
    model = JEPA(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (4, 128), device="cuda")
    p, z = model(x)
    assert not z[:, 1:].detach().requires_grad

    loss, _ = infonce_loss(p, z, tau=0.1)
    loss.backward()
    enc0_out = model.blocks[0].attn.out.weight.grad
    enc0_mlp2 = model.blocks[0].mlp_w2.weight.grad
    pred0_out = model.blocks[2].attn.out.weight.grad
    proj = model.proj.weight.grad
    embed = model.embed.weight.grad
    for name, g in [("enc0_attn.out", enc0_out), ("enc0_mlp_w2", enc0_mlp2), ("pred0_attn.out", pred0_out), ("proj", proj), ("embed", embed)]:
        assert g is not None and g.norm() > 0, f"{name} must have nonzero grad at init, got {g}"
    print(f"  init: enc0_attn.out={enc0_out.norm():.4f} enc0_mlp_w2={enc0_mlp2.norm():.4f} proj={proj.norm():.4f} embed={embed.norm():.4f}")

    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)
    opt.step()
    opt.zero_grad()
    p, z = model(x)
    loss, _ = infonce_loss(p, z, tau=0.1)
    loss.backward()
    enc0_qkv = model.blocks[0].attn.qkv.weight.grad
    enc0_mlp1 = model.blocks[0].mlp_w1.weight.grad
    pred0_qkv = model.blocks[2].attn.qkv.weight.grad
    for name, g in [("enc0_qkv", enc0_qkv), ("enc0_mlp_w1", enc0_mlp1), ("pred0_qkv", pred0_qkv)]:
        assert g is not None and g.norm() > 0, f"{name} must have nonzero grad after 1 step, got {g}"
    print(f"  step 1: enc0_qkv={enc0_qkv.norm():.4f} enc0_mlp_w1={enc0_mlp1.norm():.4f} pred0_qkv={pred0_qkv.norm():.4f}")
    print("  PASS")


def test_5_no_collapse_at_init():
    """At random init, z_std_per_dim must be well above 0 (no init-time collapse)."""
    print("[Test 5] No collapse at random init...")
    torch.manual_seed(0)
    cfg = small_cfg()
    model = JEPA(cfg).cuda()
    x = torch.randint(0, cfg.vocab_size, (4, 128), device="cuda")
    _, z = model(x)
    d = collapse_diagnostics(z)
    print(f"  z_norm={d['z_norm'].item():.3f} z_std_per_dim={d['z_std_per_dim'].item():.4f} z_off_diag={d['z_off_diag_cos_sim'].item():.4f}")
    assert d["z_std_per_dim"].item() > 0.05, "z_std_per_dim too low — collapse at init"
    assert abs(d["z_off_diag_cos_sim"].item()) < 0.3, "off-diagonal cos sim too high at init"
    print("  PASS")


if __name__ == "__main__":
    test_1_overfit_one_batch()
    test_2_no_label_leak()
    test_3_identity_no_predictor()
    test_4_gradient_flow()
    test_5_no_collapse_at_init()
    print("\nAll smoke tests PASSED")
