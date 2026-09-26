#!/usr/bin/env python3
"""
Correctness tests for the two diagnostic-only engine options, run against the
generated files as they are actually pasted (so fragment-vs-file drift is caught
too). No training happens: _score_lr is replaced with a synthetic score surface.

What is being pinned down, and why each one is load-bearing:

  1. TASK["fixed_knob"] (the B1 "same tuning budget as the baselines" arm) must
     collapse the second axis to the ONE stated value and must NOT run the knob
     boundary-extend rule -- there is no grid for the pick to sit at the edge of,
     and an extension would silently restore the joint search the arm exists to
     avoid. The synthetic surface is monotone in gamma on purpose, so the pick
     lands at the top of whatever knob grid exists: if the extension were still
     live, the test would see gamma grow past 3000.

  2. The LR axis must keep its ordinary 7-point half-decade grid and its own
     boundary rule under fixed_knob -- the whole claim of the arm is that it
     spends exactly the baselines' tuning budget, no less either.

  3. A protocol file must be UNAFFECTED by the edit: the same surface must still
     drive the full joint (lr x gamma) search with knob extensions. If this ever
     fails, the diagnostic option has leaked into the main run and every frozen
     gamma in the paper is suspect.

  4. TASK["eviekf_maxf"] must reach the optimizer (C1 raises the Kronecker size
     limit 1024 -> 4608 on CIFAR-10), and must be absent from every protocol
     task, i.e. those files still build EvieKF at the frozen 1024.

    python benchmark12/tests/test_diagnostic_knobs.py
"""
import sys
import importlib.util
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
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
    spec.loader.exec_module(mod)          # safe: main() only runs under __main__
    return mod


def probe(mod, opt_name="EvieKF"):
    """Run the real _joint_lr_knob_search against a synthetic surface: a bowl in
    lr (interior optimum) and monotone-decreasing in gamma (top-edge optimum)."""
    import math
    calls = []

    def fake_score(opt, data, units, lr, adamw_lr, knob=None):
        calls.append((float(lr), None if knob is None else float(knob)))
        v = (math.log10(lr) + 3.0) ** 2
        if knob is not None:
            v -= math.log10(knob) * 0.01
        return v if mod.PRIMARY_LOWER_BETTER else -v

    mod._score_lr = fake_score
    best_lr, best_knob, b_lr, b_knob = mod._joint_lr_knob_search(
        opt_name, None, ["full"], None, {}, lambda *a, **k: None)
    return dict(best_lr=best_lr, best_knob=best_knob, b_lr=b_lr, b_knob=b_knob,
                n_points=len(calls),
                lrs=sorted({l for l, _ in calls}),
                knobs=sorted({k for _, k in calls}))


# ==========================================================================
print("\n1-2. TASK['fixed_knob']: 7 LR points at one fixed gamma, no knob extension")
d = load("benchmark12/diagnostics/language_ptb_evie_fixed_gamma.py")
check("the file declares the fixed knob", d.TASK.get("fixed_knob") == {"EvieKF": 3000.0},
      str(d.TASK.get("fixed_knob")))
r = probe(d)
check("gamma axis is exactly the one stated value", r["knobs"] == [3000.0], str(r["knobs"]))
check("frozen gamma is that value", r["best_knob"] == 3000.0, str(r["best_knob"]))
check("knob boundary rule never ran", r["b_knob"] == 0, f"b_knob={r['b_knob']}")
check("LR grid still has its 7 half-decade points", len(r["lrs"]) == 7, str(r["lrs"]))
check("total budget is 7 scored points, not a joint grid", r["n_points"] == 7,
      f"n_points={r['n_points']}")

print("\n3. a protocol file is unaffected: still the full joint search")
m = load("benchmark12/language/language_ptb.py")
check("no fixed_knob on the protocol task", m.TASK.get("fixed_knob") is None)
rm = probe(m)
check("gamma grid still searched (>= its 7 points)", len(rm["knobs"]) >= 7,
      f"{len(rm['knobs'])} values")
check("knob extension still live on a top-edge pick", rm["b_knob"] > 0,
      f"b_knob={rm['b_knob']}")
check("joint grid, not 7 points", rm["n_points"] >= 49, f"n_points={rm['n_points']}")

print("\n4. TASK['eviekf_maxf'] reaches the optimizer, and only where declared")
import torch                                          # noqa: E402

c1 = load("benchmark12/diagnostics/vision_cifar10_evie_maxf4608.py")
check("the file declares the raised limit", c1.TASK.get("eviekf_maxf") == 4608,
      str(c1.TASK.get("eviekf_maxf")))
opt = c1.build_optimizer("EvieKF", torch.nn.Linear(8, 4), lr=1e-3, knob=3000.0,
                         eviekf_maxf=c1.TASK.get("eviekf_maxf"))
check("EvieKF honours the passed limit", opt.maxf == 4608, f"maxf={opt.maxf}")

v = load("benchmark12/vision/vision_cifar10.py")
check("no eviekf_maxf on the protocol task", v.TASK.get("eviekf_maxf") is None)
opt2 = v.build_optimizer("EvieKF", torch.nn.Linear(8, 4), lr=1e-3, knob=3000.0,
                         eviekf_maxf=v.TASK.get("eviekf_maxf"))
check("a task that declares nothing still gets the frozen 1024",
      opt2.maxf == v.EVIEKF_MAXF == 1024, f"maxf={opt2.maxf}")

# The wiring itself: the engine must actually pass the task's value at BOTH
# build_optimizer() call sites (fresh cell and resumed cell). Without this, the
# two checks above would pass while a real run silently used 1024 -- exactly the
# kind of silent-no-op the whole diagnostic depends on not happening. The shared
# optimizer block deliberately does NOT read TASK, so the engine is the only
# place this can be wired.
src = (ROOT / "benchmark12/diagnostics/vision_cifar10_evie_maxf4608.py").read_text(
    encoding="utf-8")
check("engine passes it at both build_optimizer() call sites",
      src.count("eviekf_maxf=TASK.get(\"eviekf_maxf\")") == 2,
      f"{src.count('eviekf_maxf=TASK.get')} call sites")
_shared = src[src.index("# === SHARED OPTIMIZER BLOCK"):
              src.index("# === END SHARED OPTIMIZER BLOCK")]
_reads = [ln.strip() for ln in _shared.splitlines()
          if ("TASK.get(" in ln or "TASK[" in ln) and not ln.lstrip().startswith("#")
          and '"""' not in ln and "-- a" not in ln and "passed in as" not in ln]
check("the shared block never reads TASK in code (docstrings aside)",
      not _reads, str(_reads[:2]))

# ==========================================================================
print()
if FAILS:
    print(f"FAILED: {FAILS}")
    sys.exit(1)
print("all diagnostic-knob tests passed")
