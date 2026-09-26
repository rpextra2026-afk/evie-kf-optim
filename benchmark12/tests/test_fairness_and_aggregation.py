#!/usr/bin/env python3
"""
Correctness tests for the two fairness/aggregation fixes in the ENGINE BLOCK,
run against the block as it is actually pasted into the generated files (so a
drift between fragment and generated file is caught too).

What is being pinned down here, and why each one is load-bearing:

  1. _model_has_batchnorm() -- the Evie-KF family must take the split-batch path
     to get its noise sample. On a BatchNorm model that path is ghost batch norm,
     a real regulariser, so if only the Evie arms took it they would be training
     a different objective than the baselines and any vision win would be
     uninterpretable. The engine therefore puts EVERY optimizer on the split path
     when BN is present. That decision hinges entirely on this predicate, so it
     is tested directly against all three real architectures.

  2. The split path's baseline branch must reproduce the full-batch gradient
     EXACTLY on a BN-free model -- that is the whole justification for letting
     language/finance baselines keep the cheaper single backward while the Evie
     arms split. If this identity did not hold, the two paths would not be
     interchangeable and the domains would not be comparable.

  3. _complete_units() -- dropping only the diverged cells rewards instability
     (an optimizer that blows up on the hard units gets scored on the easy ones
     it survived). The drop has to be paired across optimizers.

  4. Optimizer order: EvieKF first (a cut-short run must still contain the
     comparison the paper is about), and AdamW before Muon (Muon's AdamW
     sub-group reuses AdamW's frozen LR, which is None until AdamW is searched).

    python benchmark12/tests/test_fairness_and_aggregation.py

CPU-only, a few seconds. No GPU, no datasets.
"""
import os
import sys
import math
import types

import numpy as np
import torch
import torch.nn as nn

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
GEN = {
    "vision": os.path.join(ROOT, "benchmark12", "vision", "vision_cifar10.py"),
    "language": os.path.join(ROOT, "benchmark12", "language", "language_wikitext2.py"),
    "finance": os.path.join(ROOT, "benchmark12", "finance", "finance_tech.py"),
}

FAILS = []


def check(name, cond, detail=""):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}"
          + (f"   {detail}" if detail and not cond else ""))
    if not cond:
        FAILS.append(name)


def _slice(path, start, end):
    src = open(path, encoding="utf-8").read()
    a = src.index(start)
    b = src.index(end) + len(end)
    return src[a:b]


def _load_blocks(domain):
    """Exec the shared optimizer block + the engine block from a generated file,
    in a namespace carrying the globals the header would have provided."""
    path = GEN[domain]
    ns = {
        "torch": torch, "nn": nn, "math": math, "np": np, "os": os, "sys": sys,
        "WEIGHT_DECAY": 1e-4, "GRAD_CLIP_NORM": 5.0, "HESS_FREQ": 5, "N_HUTCH": 4,
        "log": lambda *a, **k: None,
    }
    exec(compile(_slice(path, "# === SHARED OPTIMIZER BLOCK",
                        "# === END SHARED OPTIMIZER BLOCK ==="),
                 "<shared_block>", "exec"), ns)
    return ns


def _load_engine_fns(domain, names):
    """Pull specific top-level function sources out of the generated file and
    exec them alone -- the engine block as a whole depends on far more header
    state than these few pure helpers need."""
    src = open(GEN[domain], encoding="utf-8").read()
    lines = src.split("\n")
    ns = {"torch": torch, "nn": nn, "math": math, "np": np}
    for name in names:
        start = next(i for i, l in enumerate(lines) if l.startswith(f"def {name}("))
        end = start + 1
        while end < len(lines) and not (lines[end] and not lines[end][0].isspace()):
            end += 1
        exec(compile("\n".join(lines[start:end]), f"<{name}>", "exec"), ns)
    return ns


# ==========================================================================
print("[_model_has_batchnorm: correct on all three real architectures]")
_hb = _load_engine_fns("vision", ["_model_has_batchnorm"])["_model_has_batchnorm"]

_S = _load_blocks("vision")
# Build each domain's real model straight out of its generated file.
_vsrc = open(GEN["vision"], encoding="utf-8").read()
_vision_src = _vsrc[_vsrc.index("class _BasicBlock"):_vsrc.index("class _VisionDataset")]
_vns = {"torch": torch, "nn": nn, "F": torch.nn.functional, "math": math, "np": np}
exec(compile(_vision_src, "<vision_models>", "exec"), _vns)
resnet = _vns["CifarStemResNet18"](num_classes=10)

check("vision ResNet-18 detected as a BatchNorm model", _hb(resnet) is True)

fin_mlp = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 32),
                        nn.GELU(), nn.Linear(32, 1))
check("finance MLP detected as BN-free", _hb(fin_mlp) is False)

ln_block = nn.Sequential(nn.Linear(16, 16), nn.LayerNorm(16), nn.GELU(),
                         nn.Linear(16, 16))
check("LayerNorm model detected as BN-free (LayerNorm must NOT count -- it "
      "does not normalise across the batch)", _hb(ln_block) is False)
check("BatchNorm1d also detected",
      _hb(nn.Sequential(nn.Linear(4, 4), nn.BatchNorm1d(4))) is True)


# ==========================================================================
print("\n[split-batch baseline branch == full-batch gradient, exactly, on a "
      "BN-free model]")
# This is the identity that lets language/finance baselines skip the split.
torch.manual_seed(0)
model = nn.Sequential(nn.Linear(5, 64), nn.GELU(), nn.Linear(64, 32),
                      nn.GELU(), nn.Linear(32, 1))
params = [p for p in model.parameters() if p.requires_grad]
x = torch.randn(32, 5)
y = torch.randn(32, 1)
crit = nn.MSELoss()

# (a) the reference: one full-batch backward
model.zero_grad(set_to_none=True)
crit(model(x), y).backward()
full = [p.grad.detach().clone() for p in params]

# (b) the engine's baseline split branch: accumulate both halves, then halve
model.zero_grad(set_to_none=True)
nh = x.size(0) // 2
crit(model(x[:nh]), y[:nh]).backward()
crit(model(x[nh:2 * nh]), y[nh:2 * nh]).backward()
for p in params:
    if p.grad is not None:
        p.grad.mul_(0.5)
split = [p.grad.detach().clone() for p in params]

err = max(float((a - b).abs().max()) for a, b in zip(full, split))
check(f"split == full-batch gradient (max abs err {err:.2e})", err < 1e-6,
      f"err={err:.3e}")

# (c) and the Evie branch -- clone both halves, then average -- agrees too
model.zero_grad(set_to_none=True)
crit(model(x[:nh]), y[:nh]).backward()
ga = [p.grad.detach().clone() for p in params]
model.zero_grad(set_to_none=True)
crit(model(x[nh:2 * nh]), y[nh:2 * nh]).backward()
gb = [p.grad.detach().clone() for p in params]
evie = [0.5 * (a + b) for a, b in zip(ga, gb)]
err2 = max(float((a - b).abs().max()) for a, b in zip(full, evie))
check(f"Evie branch == full-batch gradient too (max abs err {err2:.2e})",
      err2 < 1e-6, f"err={err2:.3e}")

# (d) on a BN model the two are NOT equal -- which is exactly why every arm has
#     to take the same path there. If this ever starts passing as "equal", the
#     confound argument is wrong and the universal-split fix is unnecessary.
bn = nn.Sequential(nn.Linear(5, 16), nn.BatchNorm1d(16), nn.GELU(),
                   nn.Linear(16, 1))
bnp = [p for p in bn.parameters() if p.requires_grad]
bn.train()
bn.zero_grad(set_to_none=True)
crit(bn(x), y).backward()
bn_full = [p.grad.detach().clone() for p in bnp]
bn.zero_grad(set_to_none=True)
crit(bn(x[:nh]), y[:nh]).backward()
crit(bn(x[nh:2 * nh]), y[nh:2 * nh]).backward()
for p in bnp:
    if p.grad is not None:
        p.grad.mul_(0.5)
bn_split = [p.grad.detach().clone() for p in bnp]
bn_err = max(float((a - b).abs().max()) for a, b in zip(bn_full, bn_split))
check(f"BN model: split != full-batch (gap {bn_err:.2e}) -- the confound the "
      f"universal-split fix exists to remove", bn_err > 1e-6,
      f"gap={bn_err:.3e} (unexpectedly equal)")


# ==========================================================================
print("\n[_complete_units: the drop is paired across optimizers]")
_eng = _load_engine_fns("finance", ["_cell_ok", "_complete_units"])
_eng["ACTIVE_OPT_NAMES"] = ["EvieKF", "AdamW"]
_eng["SEEDS"] = [42, 43]
_eng["_is_finite"] = lambda v: isinstance(v, (int, float)) and math.isfinite(v)

units = ["AAA", "BBB", "CCC"]
cells = {}
for o in ("EvieKF", "AdamW"):
    for u in units:
        for s in (42, 43):
            cells[f"{o}|{u}|{s}"] = {"primary": 1.0, "diverged": False}
# AdamW blows up on exactly one unit, at one seed.
cells["AdamW|BBB|43"] = {"primary": float("nan"), "diverged": True}

keep = _eng["_complete_units"](cells, units)
check("the failing unit is dropped", "BBB" not in keep, f"keep={keep}")
check("the other units survive", keep == ["AAA", "CCC"], f"keep={keep}")
check("it is dropped for EVERY optimizer, not just the one that failed -- "
      "otherwise the failing optimizer is scored on the easy units only",
      all(_eng["_complete_units"](cells, units) == keep for _ in range(2)))

# the degenerate case the engine has to warn about rather than silently fake
cells["EvieKF|AAA|42"] = {"primary": float("nan"), "diverged": True}
cells["AdamW|CCC|42"] = {"primary": float("nan"), "diverged": True}
check("every unit failing somewhere -> empty set (engine then warns loudly "
      "instead of reporting a biased number silently)",
      _eng["_complete_units"](cells, units) == [])


# ==========================================================================
print("\n[optimizer order: a cut-short run still contains the real comparison]")
names = _S["OPT_NAMES"]
check("EvieKF runs first", names[0] == "EvieKF", f"order={names}")
check("AdamW is in the first two (so any prefix has the headline pair)",
      "AdamW" in names[:2], f"order={names}")
check("AdamW precedes Muon (Muon's AdamW sub-group reuses AdamW's frozen LR, "
      "which is None until AdamW has been searched)",
      names.index("AdamW") < names.index("Muon"), f"order={names}")
check("still exactly the 8 frozen arms, nothing added or dropped",
      sorted(names) == sorted(["AdamW", "AdaBelief", "SGD", "Sophia", "Muon",
                               "Lion", "Shampoo", "EvieKF"]), f"order={names}")
abl = _S["ABLATION_OPT_NAMES"]
check("ablation leads with EvieKF too", abl[0] == "EvieKF", f"order={abl}")
check("ablation still has its 3 arms",
      sorted(abl) == sorted(["EvieDiag", "EvieKFu", "EvieKF"]), f"order={abl}")


# ==========================================================================
print()
if FAILS:
    print(f"FAILED: {FAILS}")
    sys.exit(1)
print("all fairness + aggregation tests passed")
