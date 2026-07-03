"""Task-2 smoke test: MeanFlow head + JVP loss math + verifier, CPU only."""
import sys
sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))

import torch
from models.transformer_smolvlm import SmolVLMActionTransformer, Attention
from models.verifier import ActionEnergyVerifier, ActionEnergyVerifierConfig, info_nce_loss

torch.manual_seed(0)
B, T_vlm, D_vlm, T_act, D_act, D_prop = 3, 37, 576, 10, 7, 8
H = 128

# ---------- 1. teacher-equivalence at init ----------
teacher = SmolVLMActionTransformer(
    hidden_size=H, vlm_hidden_size=D_vlm, depth=2, num_heads=4,
    dim_action=D_act, dim_propio=D_prop, use_adaln=False, use_meanflow=False,
)
student = SmolVLMActionTransformer(
    hidden_size=H, vlm_hidden_size=D_vlm, depth=2, num_heads=4,
    dim_action=D_act, dim_propio=D_prop, use_adaln=False, use_meanflow=True,
)
missing, unexpected = student.load_state_dict(teacher.state_dict(), strict=False)
assert all("interval_proj" in k for k in missing), missing
assert not unexpected, unexpected

teacher.eval(); student.eval()
feats = torch.randn(B, T_vlm, D_vlm)
x = torch.randn(B, T_act, D_act)
prop = torch.randn(B, D_prop)
t = torch.rand(B)

with torch.no_grad():
    v_teacher = teacher(feats, x, prop, t)
    v_student_none = student(feats, x, prop, t)          # r=None
    v_student_rt = student(feats, x, prop, t, r=t)       # r=t (zero-init => identical)
assert torch.allclose(v_teacher, v_student_none, atol=1e-6)
assert torch.allclose(v_teacher, v_student_rt, atol=1e-6)
print("[1] zero-init teacher equivalence OK")

# ---------- 2. MeanFlow JVP loss math ----------
for m in student.modules():
    if isinstance(m, Attention):
        m.fused_attn = False
student.train()

action_norm = torch.randn(B, T_act, D_act)
noise = torch.randn_like(action_norm)
t1, t2 = torch.rand(B) * 0.999 + 0.001, torch.rand(B) * 0.999 + 0.001
t = torch.maximum(t1, t2); r = torch.minimum(t1, t2)
x_t = t.view(-1, 1, 1) * noise + (1 - t.view(-1, 1, 1)) * action_norm
v_t = noise - action_norm

def u_fn(z, r_, t_):
    return student(feats, z, prop, t_, r_)

u, dudt = torch.func.jvp(u_fn, (x_t, r, t), (v_t, torch.zeros_like(r), torch.ones_like(t)))
u_tgt = (v_t - (t - r).view(-1, 1, 1) * dudt).detach()
err = u - u_tgt
delta_sq = err.pow(2).mean(dim=(1, 2))
loss = (delta_sq / (delta_sq.detach() + 1e-3)).mean()
loss.backward()
grads = [p.grad for p in student.parameters() if p.requires_grad and p.grad is not None]
assert len(grads) > 0 and all(torch.isfinite(g).all() for g in grads)
ip_grad = student.interval_proj[-1].weight.grad
assert ip_grad is not None and torch.isfinite(ip_grad).all()
print(f"[2] MeanFlow JVP loss OK: loss={loss.item():.4f}, {len(grads)} grads, interval_proj grad norm={ip_grad.norm():.4f}")

# ---------- 3. AdaLN variant ----------
student_adaln = SmolVLMActionTransformer(
    hidden_size=H, vlm_hidden_size=D_vlm, depth=2, num_heads=4,
    dim_action=D_act, dim_propio=D_prop, use_adaln=True, use_meanflow=True,
)
for m in student_adaln.modules():
    if isinstance(m, Attention):
        m.fused_attn = False
u2, dudt2 = torch.func.jvp(
    lambda z, r_, t_: student_adaln(feats, z, prop, t_, r_),
    (x_t, r, t), (v_t, torch.zeros_like(r), torch.ones_like(t)),
)
assert u2.shape == (B, T_act, D_act) and torch.isfinite(dudt2).all()
print("[3] AdaLN MeanFlow JVP OK")

# ---------- 4. Verifier ----------
cfg = ActionEnergyVerifierConfig(
    vlm_hidden_size=D_vlm, dim_action=D_act, dim_proprio=D_prop,
    hidden_size=96, depth=2, num_heads=4, num_pool_tokens=4, max_num_actions=T_act,
)
ver = ActionEnergyVerifier(cfg)
print(f"    verifier params: {ver.num_parameters/1e6:.2f}M")
e_pos = ver(feats, prop, action_norm)
assert e_pos.shape == (B,)
K = 5
cands = torch.randn(B, K, T_act, D_act)
e_neg = ver.score_candidates(feats, prop, cands)
assert e_neg.shape == (B, K)
mask = torch.ones(B, K, dtype=torch.bool); mask[:, 0] = False
out = info_nce_loss(e_pos, e_neg, temperature=0.1, neg_mask=mask)
out["loss"].backward()
assert torch.isfinite(out["loss"]) and 0 <= out["acc"] <= 1
print(f"[4] verifier + InfoNCE OK: loss={out['loss'].item():.4f} acc={out['acc'].item():.3f}")

# ---------- 5. score_candidates == per-candidate forward ----------
ver.eval()
with torch.no_grad():
    e_batch = ver.score_candidates(feats, prop, cands)
    e_loop = torch.stack([ver(feats, prop, cands[:, k]) for k in range(K)], dim=1)
assert torch.allclose(e_batch, e_loop, atol=1e-5), (e_batch - e_loop).abs().max()
print("[5] batched scoring == per-candidate scoring OK")

# ---------- 6. verifier save/load round-trip ----------
import tempfile, os
with tempfile.TemporaryDirectory() as d:
    ver.save_pretrained(d)
    ver2 = ActionEnergyVerifier.from_pretrained(d)
    with torch.no_grad():
        assert torch.allclose(ver(feats, prop, action_norm), ver2(feats, prop, action_norm), atol=1e-6)
print("[6] verifier save/load round-trip OK")

print("\nALL SMOKE TESTS PASSED")
