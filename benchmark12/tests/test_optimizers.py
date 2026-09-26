#!/usr/bin/env python3
"""
Correctness tests for the SHARED OPTIMIZER BLOCK, run against the block as it is
actually pasted into the 12 generated files (extracted from
benchmark12/vision/vision_cifar10.py so a drift between fragment and generated
file is caught too).

    python benchmark12/tests/test_optimizers.py
    pytest benchmark12/tests/test_optimizers.py

CPU-only, a few seconds. No GPU, no datasets.
"""
import os
import re
import sys
import math
import types

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN_FILE = os.path.join(ROOT, "benchmark12", "vision", "vision_cifar10.py")


def _load_shared_block():
    """Exec ONLY the shared optimizer block from a generated file, in a namespace
    with the few globals it expects from the header."""
    src = open(GEN_FILE).read()
    a = src.index("# === SHARED OPTIMIZER BLOCK")
    b = src.index("# === END SHARED OPTIMIZER BLOCK ===") + len("# === END SHARED OPTIMIZER BLOCK ===")
    block = src[a:b]
    ns = {
        "torch": torch, "nn": nn, "math": math, "np": np,
        "WEIGHT_DECAY": 1e-4, "GRAD_CLIP_NORM": 5.0, "HESS_FREQ": 5, "N_HUTCH": 4,
        "log": lambda *a, **k: None,
    }
    exec(compile(block, "<shared_block>", "exec"), ns)
    return types.SimpleNamespace(**ns)


S = _load_shared_block()
FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


# --------------------------------------------------------------------------
def _quad_problem(d=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(d, d, generator=g)
    A = A @ A.T / d + torch.eye(d)          # SPD, well conditioned
    b = torch.randn(d, generator=g)
    x_star = torch.linalg.solve(A, b)
    return A, b, x_star


def _converges(make_opt, lr, steps=400, needs_hessian=False):
    A, b, x_star = _quad_problem()
    x = nn.Parameter(torch.zeros(20))
    opt = make_opt([x], lr)
    for _ in range(steps):
        opt.zero_grad()
        loss = 0.5 * x @ (A @ x) - b @ x
        loss.backward()
        if needs_hessian:
            # exact diagonal Hessian of the quadratic is diag(A)
            opt.set_hessian([(x, torch.diagonal(A).clone())])
        opt.step()
    err = (x.detach() - x_star).norm().item() / x_star.norm().item()
    return err


def test_convergence():
    print("[convergence on a well-conditioned quadratic; rel. error < 0.05]")
    check("AdamW", _converges(lambda p, lr: torch.optim.AdamW(p, lr=lr), 5e-2) < 0.05)
    check("AdaBelief", _converges(lambda p, lr: S.AdaBelief(p, lr=lr), 5e-2) < 0.05)
    check("SGDM", _converges(lambda p, lr: S.SGDM(p, lr=lr), 2e-2) < 0.05)
    check("Lion", _converges(lambda p, lr: S.Lion(p, lr=lr), 3e-3, steps=1500) < 0.08)
    check("Sophia", _converges(lambda p, lr: S.Sophia(p, lr=lr), 3e-1,
                               needs_hessian=True) < 0.05)


def test_decoupled_weight_decay():
    print("[decoupled WD: a zero-gradient param shrinks by exactly (1 - lr*wd)]")
    for name, mk in [("AdaBelief", lambda p: S.AdaBelief(p, lr=0.1, weight_decay=0.1)),
                     ("SGDM", lambda p: S.SGDM(p, lr=0.1, weight_decay=0.1)),
                     ("Lion", lambda p: S.Lion(p, lr=0.1, weight_decay=0.1)),
                     ("Sophia", lambda p: S.Sophia(p, lr=0.1, weight_decay=0.1))]:
        x = nn.Parameter(torch.ones(10))
        opt = mk([x])
        x.grad = torch.zeros(10)
        opt.step()
        check(f"{name} decoupled-WD factor", torch.allclose(x.detach(),
              torch.full((10,), 0.99), atol=1e-6),
              detail=f"got {x.detach()[0].item():.6f}, want 0.99")


def test_lion_bounded_step():
    print("[Lion update magnitude per coord == lr exactly (sign rule)]")
    x = nn.Parameter(torch.randn(1000))
    x0 = x.detach().clone()
    opt = S.Lion([x], lr=0.01, weight_decay=0.0)
    x.grad = torch.randn(1000) * 100.0            # huge gradient
    opt.step()
    step = (x.detach() - x0).abs()
    check("Lion step == lr", torch.allclose(step, torch.full((1000,), 0.01), atol=1e-7))


def test_adabelief_differs_from_adam():
    print("[AdaBelief != Adam on a signal with non-trivial gradient variance]")
    torch.manual_seed(0)
    xa = nn.Parameter(torch.zeros(50)); xb = nn.Parameter(torch.zeros(50))
    oa = torch.optim.Adam([xa], lr=1e-2)
    ob = S.AdaBelief([xb], lr=1e-2, weight_decay=0.0)
    for t in range(200):
        gcommon = torch.randn(50) + torch.sin(torch.tensor(float(t)))
        oa.zero_grad(); xa.grad = gcommon.clone(); oa.step()
        ob.zero_grad(); xb.grad = gcommon.clone(); ob.step()
    check("AdaBelief trajectory differs from Adam",
          (xa.detach() - xb.detach()).norm().item() > 1e-3)


def test_newton_schulz_orthogonalizes():
    print("[Newton-Schulz (5 steps) compresses the singular-value spectrum]")
    torch.manual_seed(0)
    # ill-conditioned input: singular values span [0.2, 5.0] (ratio 25)
    G = (torch.randn(64, 48) @ torch.diag(torch.linspace(0.2, 5.0, 48))
         @ torch.randn(48, 48))
    sv_in = torch.linalg.svdvals(G)
    O = S._zeropower_via_newtonschulz5(G, steps=5)
    sv = torch.linalg.svdvals(O)
    # 5-step NS is an APPROXIMATE orthogonalisation: it should sharply flatten
    # the spectrum and cap it near 1, not reach exactly unit svals.
    check("NS caps spectrum near 1", sv.max().item() < 1.35,
          detail=f"max sval {sv.max().item():.3f}")
    check("NS flattens spectrum",
          (sv.max() / sv.min()).item() < 0.25 * (sv_in.max() / sv_in.min()).item(),
          detail=f"ratio {sv.max()/sv.min():.2f} vs input {sv_in.max()/sv_in.min():.2f}")
    check("NS preserves shape", O.shape == G.shape)
    # NS is ~identity (up to scale) on an already-orthogonal input
    Q = torch.linalg.qr(torch.randn(32, 32))[0]
    svq = torch.linalg.svdvals(S._zeropower_via_newtonschulz5(Q, steps=5))
    check("NS ~fixes an orthogonal input", svq.min() > 0.95 and svq.max() < 1.05,
          detail=f"svals in [{svq.min():.3f}, {svq.max():.3f}]")


def test_muon_hybrid_split_and_step():
    print("[Muon: >=2D non-excluded -> Muon update; 1D / excluded -> AdamW]")

    class Tiny(nn.Module):
        def __init__(self):
            super().__init__()
            self.wte = nn.Embedding(20, 8)
            self.lin = nn.Linear(8, 8)
            self.ln = nn.LayerNorm(8)
            self.head = nn.Linear(8, 20, bias=False)
            self.head.weight = self.wte.weight
            self._muon_exclude_names = ["wte", "head"]

        def forward(self, idx):
            return self.head(self.ln(self.lin(self.wte(idx))))

    m = Tiny()
    muon_p, adamw_p = S.split_params_for_muon(m)
    muon_ids = {id(p) for p in muon_p}
    check("lin.weight -> Muon", id(m.lin.weight) in muon_ids)
    check("lin.bias -> AdamW", id(m.lin.bias) not in muon_ids)
    check("ln.weight (1D) -> AdamW", id(m.ln.weight) not in muon_ids)
    check("tied wte/head -> AdamW (excluded)", id(m.wte.weight) not in muon_ids)

    opt = S.build_optimizer("Muon", m, lr=2e-2, adamw_lr=1e-3)
    before = {n: p.detach().clone() for n, p in m.named_parameters()}
    idx = torch.randint(0, 20, (4, 3))
    loss = m(idx).float().pow(2).mean()
    loss.backward()
    opt.step()
    moved = {n for n, p in m.named_parameters()
             if not torch.allclose(p.detach(), before[n])}
    check("lin.weight moved", "lin.weight" in moved)
    check("ln.weight moved (adamw side)", "ln.weight" in moved)
    check("no NaNs after Muon step",
          all(torch.isfinite(p).all() for p in m.parameters()))


def test_lr_grid_shape():
    print("[7-point half-decade grid; same count for every optimizer]")
    for o, c in S.LR_GRID_CENTRE.items():
        g = S.half_decade_grid(c)
        check(f"{o} grid len 7", len(g) == 7)
        check(f"{o} grid half-decade ratio", abs(g[1] / g[0] / (10 ** 0.5) - 1) < 0.02)
    # EvieKFn stays a valid (buildable) name but is no longer SELECTED anywhere
    # (Fix 2 -- the shipped arm is plain EvieKF); it may still appear in the
    # mechanism/knob tables.
    # The four claim-ladder rungs, plus the three diagnostic arms added
    # 2026-09-24 (FW-9 / FW-10 / FW-11). The diagnostic arms are NOT selectable
    # from any protocol opt_set -- they are reachable only from a diag task's
    # explicit opt_list -- but they are real registry entries, so they must carry
    # an LR centre, a knob grid and the noise hook like the rest of the family.
    evie_family = {"EvieDiag", "EvieKFu", "EvieKF", "EvieKFn",
                   "EvieDiagAbs", "EvieKFm", "EvieKFr"}
    selectable = set(S.OPT_NAMES) | set(S.ABLATION_OPT_NAMES)
    check("shipped main arm is plain EvieKF, not EvieKFn",
          "EvieKF" in S.OPT_NAMES and "EvieKFn" not in S.OPT_NAMES)
    check("every selected optimizer has an LR-grid centre",
          selectable <= set(S.LR_GRID_CENTRE))
    check("every KNOB_GRID key is a known Evie-KF variant or a real optimizer",
          set(S.KNOB_GRID) <= (selectable | evie_family))
    check("NOISE_CONSUMERS are exactly the Evie-KF family",
          set(S.NOISE_CONSUMERS) == evie_family)


def test_hutchinson_diag_on_quadratic():
    print("[Hutchinson estimate ~ diag(A) on a quadratic]")
    A, b, _ = _quad_problem(d=12, seed=3)
    x = nn.Parameter(torch.randn(12))
    crit = None

    def compute_loss():
        return 0.5 * x @ (A @ x) - b @ x

    torch.manual_seed(0)
    acc = torch.zeros(12)
    trials = 40
    for _ in range(trials):
        h = S.hutchinson_diag_hessian(None, [x], compute_loss, n_hutch=8)[0]
        acc += h
    est = acc / trials
    rel = (est - torch.diagonal(A).abs()).norm() / torch.diagonal(A).abs().norm()
    check("Hutchinson ~ |diag(A)|", rel.item() < 0.25, detail=f"rel err {rel.item():.3f}")


if __name__ == "__main__":
    for fn in [test_convergence, test_decoupled_weight_decay, test_lion_bounded_step,
               test_adabelief_differs_from_adam, test_newton_schulz_orthogonalizes,
               test_muon_hybrid_split_and_step, test_lr_grid_shape,
               test_hutchinson_diag_on_quadratic]:
        fn()
    print()
    if FAILS:
        print(f"FAILED: {FAILS}")
        sys.exit(1)
    print("all optimizer tests passed")
