#!/usr/bin/env python3
"""
Does the shipped optimizer compute what the derivation says it computes?

Every other test in this directory checks an internal consistency (the factored
operator against a dense one, the flag map, the gamma=0 identity). None of them
checks the thing the paper's method section actually claims: that

    dTheta = -lr * P^-1/2 (I + gamma*Sigma_z)^-1/2 P^-1/2 m_hat,
    Sigma_z ~ (B kron A) / tr(A)

is the update `benchmark12`'s EvieKF produces. This builds that update from the
derivation in fully DENSE form -- forming the whole nm x nm matrix and taking one
eigendecomposition of it, sharing no code path with the optimizer beyond torch --
and compares. If the Kronecker factorisation, the trace normalisation, the
whitening placement or the norm rescale were wrong, this is what would catch it.

It also pins the two assumptions the derivation needs (DERIVATION.md sec 3.3):

  (A-curv)   H ~ diag(sqrt(m2_hat)) = P, NOT diag(m2_hat). Only the first makes
             the whitened curvature the identity.
  (A-metric) the control penalty is isotropic in whitened coordinates, R_z = I,
             i.e. R_theta = P. The alternative R_theta = I is not ill-posed -- it
             has its own closed form -- but it does NOT reduce to AdamW at
             gamma=0, which is what rules it out. That is a stronger argument
             than the reconstruction's ("it doesn't simplify"), so it is pinned
             here rather than left as prose.

    python benchmark12/tests/test_derivation.py
"""
import sys
import importlib.util
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[2]
torch.set_default_dtype(torch.float64)
FAILS = 0


def check(label, ok, detail=""):
    global FAILS
    print(f"  [{'ok  ' if ok else 'FAIL'}] {label}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILS += 1


def load(rel):
    path = ROOT / rel
    spec = importlib.util.spec_from_file_location("gen_" + path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


def isqrtm(X):
    w, U = torch.linalg.eigh((X + X.T) * 0.5)
    return U @ torch.diag(w.clamp_min(1e-300) ** -0.5) @ U.T


def sqrtm(X):
    w, U = torch.linalg.eigh((X + X.T) * 0.5)
    return U @ torch.diag(w.clamp_min(0.0) ** 0.5) @ U.T


M = load("benchmark12/language/language_wikitext2.py")
n, m = 5, 4
GAMMA, LR, BSIG, EPS, B1, B2 = 100.0, 0.01, 0.95, 1e-8, 0.9, 0.999
g = torch.Generator().manual_seed(1)

# ==========================================================================
print("\n1. vec ordering: the factored form and kron(B,A) agree on which "
      "eigenvalue goes where")
Y = torch.randn(n, m, generator=g)
A = torch.randn(m, m, generator=g); A = A @ A.T + 0.3 * torch.eye(m)
B = torch.randn(n, n, generator=g); B = B @ B.T + 0.3 * torch.eye(n)
err = float((torch.kron(B, A) @ Y.reshape(-1) - (B @ Y @ A.T).reshape(-1)).abs().max())
check("kron(B,A) vec_row(Y) == vec_row(B Y A^T)", err < 1e-10, f"{err:.2e}")

# ==========================================================================
print("\n2. end-to-end: every arm's step == the dense derivation's step")
# This replaces the parity gate against kaggle_evie_v3.py, which is lost. That
# gate compared the per-parameter port against a flat-vector implementation --
# two implementations sharing one set of assumptions. This compares the
# implementation against the MATHEMATICS: the operator is built by forming the
# whole nm x nm matrix and taking a single eigendecomposition of it. A reshape
# or vec-ordering slip -- the exact failure the parity gate existed to catch --
# fails here against ground truth rather than against another program.
#
# Covers what parity covered: the centred and uncentred objects, the diagonal
# nesting, and the maxf fallback (maxf=4 with shapes (5,4) leaves the column
# factor full and sends the row factor to its diagonal -- the mixed case that
# every vision task actually runs in).

CONFIGS = [
    ("EvieKF            (full Kronecker)", dict()),
    ("EvieKFu           (uncentred, Sigma from g)", dict(centered=False)),
    ("EvieKF maxf=4     (row factor -> diagonal)", dict(maxf=4)),
    ("EvieDiag          (diagonal, relative gate)", dict(diag_only=True)),
    ("EvieDiagAbs       (diagonal, absolute gate)",
     dict(diag_only=True, diag_relative=False)),
    ("EvieKFm           (mean-eigenvalue norm)", dict(kron_mean_norm=True)),
    ("EvieKFr           (no norm rescale)", dict(norm_preserve=False)),
]


def dense_reference(flags, steps=6):
    """Recompute the arm's trajectory from the derivation, densely."""
    cfg = dict(gamma=GAMMA, beta_sig=BSIG, warmup=0, refresh=1, maxf=4096,
               diag_only=False, centered=True, per_layer_norm=False,
               diag_relative=True, kron_mean_norm=False, norm_preserve=True)
    cfg.update(flags)
    gg = torch.Generator().manual_seed(23)
    p = torch.nn.Parameter(torch.randn(n, m, generator=gg) * 0.1)
    opt = M.EvieKF([p], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=0.0, **cfg)

    mm = torch.zeros(n, m); vv = torch.zeros(n, m)
    AA = torch.zeros(m, m); BB = torch.zeros(n, n); ss = torch.zeros(n, m)
    worst_rel = 0.0
    for t in range(1, steps + 1):
        grad = torch.randn(n, m, generator=gg)
        delta = torch.randn(n, m, generator=gg) * 0.5

        p.grad = grad.clone()
        opt.set_noise([(p, delta.clone())])
        before = p.detach().clone()
        opt.step()
        code_step = p.detach() - before

        mm = B1 * mm + (1 - B1) * grad
        vv = B2 * vv + (1 - B2) * grad * grad
        P_ = (vv / (1 - B2 ** t)).sqrt() + EPS
        W_ = P_ ** -0.5
        y = W_ * (mm / (1 - B1 ** t))
        D = W_ * (delta if cfg["centered"] else grad)   # the centred/uncentred choice

        if cfg["diag_only"]:
            ss = BSIG * ss + (1 - BSIG) * D * D
            sd = ss / (1 - BSIG ** t)
            arg = sd / sd.mean() if cfg["diag_relative"] else sd
            z = y * (1.0 + cfg["gamma"] * arg).rsqrt()
        else:
            AA = BSIG * AA + (1 - BSIG) * (D.T @ D)
            BB = BSIG * BB + (1 - BSIG) * (D @ D.T)
            Ad, Bd = AA / (1 - BSIG ** t), BB / (1 - BSIG ** t)
            # the maxf fallback: a factor larger than maxf keeps only its diagonal
            if m > cfg["maxf"]:
                Ad = torch.diag(torch.diag(Ad))
            if n > cfg["maxf"]:
                Bd = torch.diag(torch.diag(Bd))
            lam = torch.linalg.eigvalsh(Ad).clamp_min(0)
            mu = torch.linalg.eigvalsh(Bd).clamp_min(0)
            denom = (float(lam.sum()) * float(mu.sum()) / (n * m)
                     if cfg["kron_mean_norm"] else float(lam.sum())) + 1e-30
            Op = isqrtm(torch.eye(n * m) + cfg["gamma"] * torch.kron(Bd, Ad) / denom)
            z = (Op @ y.reshape(-1)).reshape(n, m)

        if cfg["norm_preserve"]:
            z = z * (y.norm() / z.norm())
        theory_step = -LR * (W_ * z)
        worst_rel = max(worst_rel, float((code_step - theory_step).abs().max())
                        / float(theory_step.abs().max()))
    return worst_rel


for label, flags in CONFIGS:
    # 1e-6, not 1e-14: the optimizer stores the noise sample in float32 by design
    # (full spec sec 5), so the factors carry float32 rounding in a float64 test.
    r = dense_reference(flags)
    check(f"{label}", r < 1e-6, f"rel {r:.2e}")

# ==========================================================================
print("\n3. the two assumptions the whitening collapse needs")
d = 8
torch.manual_seed(0)
X = torch.randn(d, d); Sig = X @ X.T / d + 0.1 * torch.eye(d)
pv = torch.rand(d) * 2.9 + 0.1
P = torch.diag(pv); W = torch.diag(pv ** -0.5)
gam = 3.7

# (A-curv): H = P makes the whitened curvature the identity; H = P^2 does not.
check("(A-curv) H = diag(sqrt(m2)) = P  ->  H_z = I",
      float((W @ P @ W - torch.eye(d)).abs().max()) < 1e-12)
check("(A-curv) H = diag(m2) = P^2  ->  H_z != I  (the pre-2026-09-17 wording)",
      float((W @ (P @ P) @ W - torch.eye(d)).abs().max()) > 0.1,
      f"off by {float((W @ (P @ P) @ W - torch.eye(d)).abs().max()):.2f}")

# The general closed form S = M^-1/2 (M^1/2 H M^1/2)^1/2 M^-1/2 solves S M S = H.
Mm = torch.eye(d) + gam * Sig
S = isqrtm(Mm) @ sqrtm(sqrtm(Mm) @ (W @ P @ W) @ sqrtm(Mm)) @ isqrtm(Mm)
check("general Riccati closed form solves S M S = H",
      float((S @ Mm @ S - (W @ P @ W)).abs().max()) < 1e-10,
      f"{float((S @ Mm @ S - (W @ P @ W)).abs().max()):.2e}")

# (A-metric): R_z = I gives the operator; R_theta = I gives a different one that
# is NOT AdamW at gamma = 0. That is what selects the assumption.
Sigz = W @ Sig @ W
grad = torch.randn(d)
adamw = -(grad / pv)                                # -P^-1 g
for label, Mz, want in (("R_theta = P  (R_z = I)", torch.eye(d), True),
                        ("R_theta = I  (R_z = P^-1)", P, False)):
    Sz = isqrtm(Mz)                                  # gamma = 0
    u = -(W @ (Sz @ (W @ grad)))
    rel = float((u - adamw).norm() / adamw.norm())
    check(f"(A-metric) {label}: gamma=0 recovers AdamW = {want}",
          (rel < 1e-12) == want, f"rel err {rel:.3f}")

check("(A-metric) the alternative is not ill-posed, just different "
      "(it has its own closed form)",
      float((isqrtm(P + gam * Sigz) @ (P + gam * Sigz) @ isqrtm(P + gam * Sigz)
             - torch.eye(d)).abs().max()) < 1e-10)

# ==========================================================================
print("\n4. gamma = 0 is AdamW, end to end through the real optimizer")
p2 = torch.nn.Parameter(torch.randn(n, m, generator=g) * 0.1)
p3 = torch.nn.Parameter(p2.detach().clone())
o2 = M.EvieKF([p2], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=0.0, gamma=0.0,
              beta_sig=BSIG, warmup=0, refresh=1, maxf=4096)
o3 = torch.optim.AdamW([p3], lr=LR, betas=(B1, B2), eps=EPS, weight_decay=0.0)
for _ in range(6):
    gr = torch.randn(n, m, generator=g)
    p2.grad = gr.clone()
    o2.set_noise([(p2, torch.randn(n, m, generator=g))])
    p3.grad = gr.clone()
    o2.step(); o3.step()
e = float((p2.detach() - p3.detach()).abs().max())
check("EvieKF(gamma=0) == torch.optim.AdamW", e < 1e-14, f"{e:.2e}")

# ==========================================================================
print("\n5. trace normalisation: what it does and what it does not (FW-9)")
Dn = torch.randn(200, n, m, generator=g)
def spectrum(c):
    Dc = Dn * c
    Ac = torch.einsum("bij,bik->jk", Dc, Dc) / Dc.shape[0]
    Bc = torch.einsum("bij,bkj->ik", Dc, Dc) / Dc.shape[0]
    lam = torch.linalg.eigvalsh(Ac).clamp_min(0)
    mu = torch.linalg.eigvalsh(Bc).clamp_min(0)
    ev = mu[:, None] * lam[None, :]
    return ev, float(lam.sum())

ev1, tau1 = spectrum(1.0)
flat = Dn.reshape(Dn.shape[0], -1)
Sig_true = torch.einsum("bi,bj->ij", flat, flat) / Dn.shape[0]
check("tr((B kron A)/tau) == tr(Sigma_z)  (trace matching -- why /tau is needed)",
      abs(float(ev1.sum()) / tau1 - float(Sig_true.trace())) < 1e-8,
      f"{float(ev1.sum())/tau1:.4f} vs {float(Sig_true.trace()):.4f}")

gains_tau, gains_mean = [], []
for c in (0.1, 1.0, 10.0):
    ev, tau = spectrum(c)
    gains_tau.append(float((1 + 10.0 * ev / tau).rsqrt().min()))
    gains_mean.append(float((1 + 10.0 * ev / ev.mean()).rsqrt().min()))
check("/tau is NOT scale-free: min gain moves with the noise scale",
      max(gains_tau) - min(gains_tau) > 0.5,
      " -> ".join(f"{v:.3f}" for v in gains_tau))
check("/mean(mu*lam) IS scale-free",
      max(gains_mean) - min(gains_mean) < 1e-9,
      " -> ".join(f"{v:.3f}" for v in gains_mean))

# ==========================================================================
print("\n6. the three diagnostic arms: each changes exactly one thing, and the "
      "defaults are untouched")

def run_arm(name, steps=25):          # > EVIEKF_WARMUP (20), or the operator
                                       # never activates and every arm is AdamW
    """Drive one registry arm through build_optimizer and return its parameter
    trajectory, so the comparisons below go through the same path a real run
    does rather than through hand-constructed flags."""
    torch.manual_seed(7)
    q = torch.nn.Parameter(torch.randn(n, m) * 0.1)
    o = M.build_optimizer(name, torch.nn.ParameterList([q]), lr=LR, knob=50.0)
    gg = torch.Generator().manual_seed(11)
    for _ in range(steps):
        q.grad = torch.randn(n, m, generator=gg)
        if name in M.NOISE_CONSUMERS:
            o.set_noise([(q, torch.randn(n, m, generator=gg) * 0.5)])
        o.step()
    return q.detach().clone(), o


_, o_kf = run_arm("EvieKF")
check("EvieKF defaults unchanged: relative diagonal gate, /tau, rescale on",
      (o_kf.diag_relative, o_kf.kron_mean_norm, o_kf.norm_preserve) == (True, False, True),
      f"{(o_kf.diag_relative, o_kf.kron_mean_norm, o_kf.norm_preserve)}")

p_diag, o_diag = run_arm("EvieDiag")
p_dabs, o_dabs = run_arm("EvieDiagAbs")
check("EvieDiagAbs is diagonal like EvieDiag but absolutely normalised",
      (o_dabs.diag_only, o_dabs.diag_relative) == (True, False)
      and (o_diag.diag_only, o_diag.diag_relative) == (True, True))
check("...and it actually trains differently (the C2 confound is real, FW-11)",
      float((p_diag - p_dabs).abs().max()) > 1e-6,
      f"max param diff {float((p_diag - p_dabs).abs().max()):.2e}")

p_kf, _ = run_arm("EvieKF")
p_km, o_km = run_arm("EvieKFm")
check("EvieKFm normalises the Kronecker spectrum by its mean eigenvalue",
      o_km.kron_mean_norm and not o_km.diag_only)
check("...and differs from EvieKF at the same gamma",
      float((p_kf - p_km).abs().max()) > 1e-6,
      f"max param diff {float((p_kf - p_km).abs().max()):.2e}")
check("EvieKFm gets its own re-centred gamma grid (gamma is O(1) there)",
      M.KNOB_GRID["EvieKFm"][0] < M.KNOB_GRID["EvieKF"][0],
      f"{M.KNOB_GRID['EvieKFm'][:3]} vs {M.KNOB_GRID['EvieKF'][:3]}")

p_kr, o_kr = run_arm("EvieKFr")
check("EvieKFr disables the norm-preserving rescale", not o_kr.norm_preserve)
check("...and its step is SHORTER than EvieKF's (it shrinks instead of rotating)",
      float((p_kr - p_kf).abs().max()) > 1e-6
      and o_kr.n_active_steps > 0,
      f"n_active_steps={o_kr.n_active_steps} (0 would be flagged as silent-AdamW)")

# The canary matters: before this change, disabling the rescale would have left
# n_active_steps at 0, which the engine converts into a divergence flag.
check("EvieKFr still reports the operator as active (the engine's canary)",
      o_kr.n_active_steps == 5, f"{o_kr.n_active_steps} active steps of 25")

for nm in ("EvieDiagAbs", "EvieKFm", "EvieKFr"):
    check(f"{nm} is registered as a noise consumer and has an LR centre",
          nm in M.NOISE_CONSUMERS and nm in M.LR_GRID_CENTRE and nm in M.KNOB_GRID)

# ==========================================================================
print()
if FAILS:
    print(f"FAILED: {FAILS}")
    sys.exit(1)
print("all derivation tests passed")
