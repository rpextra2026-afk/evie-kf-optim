#!/usr/bin/env python3
"""
aggregate.py -- the confirmatory 12-task statistical test (CLAUDE.md sec 5).

Reads checkpoints/<task_name>/summary.json for all 12 tasks (whatever is
present) and runs the pre-registered analysis:

  * Unit of analysis = the TASK, not the seed. Each task contributes ONE number
    per optimizer: the median over the 5 seeds of that seed's mean-over-units
    primary metric (already computed by each task file -> summary.json).
  * Primary significance test: paired Wilcoxon signed-rank across the 12 tasks,
    each optimizer vs the reference (default: AdamW; pass --ref Evie once Evie
    exists). Because the 12 primary metrics live on different scales
    (accuracy %, perplexity, MSE), the pairing is done on the per-task
    DIRECTION-NORMALISED RELATIVE improvement d_i = (ref_i - opt_i)/|ref_i|
    (sign flipped for higher-is-better tasks) so no single large-scale task
    dominates the signed ranks. d_i > 0  <=>  the optimizer beat the reference
    on task i.
  * Robustness check: exact sign test (binomial) on the signs of d_i.
  * Multiplicity: Holm-Bonferroni across the family of optimizer-vs-reference
    comparisons, applied to the Wilcoxon p-values.
  * Only the primary metric is tested. Everything else is descriptive.

    python benchmark12/aggregate.py
    python benchmark12/aggregate.py --ckpt-root checkpoints --ref AdamW
"""
import os
import sys
import json
import glob
import math
import argparse

import numpy as np
from scipy.stats import wilcoxon, binomtest

DOMAIN_OF = {
    "vision_cifar10": "vision", "vision_cifar100": "vision",
    "vision_svhn": "vision", "vision_stl10": "vision",
    "language_wikitext2": "language", "language_wikitext103": "language",
    "language_ptb": "language", "language_enwik8": "language",
    "finance_tech": "finance", "finance_finance": "finance",
    "finance_healthcare": "finance", "finance_consumer_industrials": "finance",
}
ALL_TASKS = list(DOMAIN_OF)


def load_summaries(ckpt_root):
    out = {}
    for path in sorted(glob.glob(os.path.join(ckpt_root, "*", "summary.json"))):
        with open(path) as fh:
            s = json.load(fh)
        out[s["task_name"]] = s
    return out


def holm(pvals):
    idx = sorted(range(len(pvals)), key=lambda i: pvals[i])
    m = len(pvals)
    adj = [float("nan")] * m
    running = 0.0
    for rank, i in enumerate(idx):
        running = max(running, min((m - rank) * pvals[i], 1.0))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-root", default="checkpoints")
    ap.add_argument("--ref", default="AdamW",
                    help="reference optimizer. The paper's claim is EvieKF vs "
                         "every baseline, so run this a SECOND time with "
                         "--ref EvieKF: with --ref AdamW the comparison family "
                         "never contains EvieKF-vs-Muon or EvieKF-vs-Shampoo.")
    args = ap.parse_args()

    summ = load_summaries(args.ckpt_root)
    if not summ:
        sys.exit(f"no summary.json found under {args.ckpt_root}/*/ -- run the "
                 f"task files first")

    present = [t for t in ALL_TASKS if t in summ]
    missing = [t for t in ALL_TASKS if t not in summ]
    if not present:
        sys.exit(f"no MAIN 12-task summary.json found under {args.ckpt_root}/*/ "
                 f"(checkpoints/ablation_*/ summaries, if any, are not part of "
                 f"this confirmatory test) -- run the main task files first")
    # take the optimizer list from a present MAIN task, never from an arbitrary
    # summary (checkpoints/ablation_*/summary.json carry only the 3 ablation arms)
    opts = list(summ[present[0]]["task_number_by_optimizer"])
    ref = args.ref
    if ref not in opts:
        sys.exit(f"reference {ref!r} not among optimizers {opts}")

    print("=" * 96)
    print(f"  12-TASK CONFIRMATORY ANALYSIS   reference = {ref}")
    print(f"  tasks present: {len(present)}/12" +
          (f"   MISSING: {missing}" if missing else ""))
    print("=" * 96)

    # ---- the 12 (or fewer) x N_opt table of per-task numbers ----------------
    table = {}          # task -> {opt -> number}
    lower_better = {}
    for t in present:
        s = summ[t]
        table[t] = s["task_number_by_optimizer"]
        lower_better[t] = s["primary_lower_better"]

    hdr = f"  {'task':<28} {'metric':<16} " + "".join(f"{o:>13}" for o in opts)
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for t in present:
        s = summ[t]
        row = f"  {t:<28} {s['primary_metric']:<16} "
        for o in opts:
            v = table[t][o]
            row += f"{v:>13.5g}" if isinstance(v, (int, float)) and math.isfinite(v) \
                else f"{'nan':>13}"
        print(row)

    # ---- per-task direction-normalised relative improvement vs reference ----
    print("\n  per-task relative improvement d_i vs " + ref +
          "   ( d>0 => optimizer better; sign-normalised, direction-aware )")
    comps = [o for o in opts if o != ref]
    d_by_opt = {o: [] for o in comps}
    tasks_used = {o: [] for o in comps}
    imputed = []          # (task, optimizer, imputed d) -- never silent

    def _num(v):
        return isinstance(v, (int, float)) and math.isfinite(v)

    for t in present:
        rv = table[t][ref]
        if not (_num(rv) and rv != 0):
            print(f"    [skip] {t}: reference {ref} has no usable number here "
                  f"({rv!r}) -- the whole task drops, there is nothing to pair "
                  f"against")
            continue
        # Pass 1: the optimizers that actually produced a number on this task.
        ds_t = {}
        for o in comps:
            ov = table[t][o]
            if _num(ov):
                ds_t[o] = ((rv - ov) / abs(rv) if lower_better[t]
                           else (ov - rv) / abs(rv))
        # Pass 2: an optimizer with NO number on this task diverged on every seed.
        # Dropping it (the old behaviour) shrinks that optimizer's n and removes
        # precisely its hardest task -- it is rewarded for failing. Impute the
        # worst d observed on this task instead, so a total failure counts as a
        # loss. Conservative: it can only ever hurt the failing optimizer.
        if ds_t:
            worst = min(list(ds_t.values()) + [0.0])
            for o in comps:
                if o not in ds_t:
                    ds_t[o] = worst
                    imputed.append((t, o, worst))
        for o in comps:
            if o in ds_t:
                d_by_opt[o].append(ds_t[o])
                tasks_used[o].append(t)

    if imputed:
        print("\n    [imputed] these (task, optimizer) pairs had NO finite result "
              "(diverged at every seed) and were scored as the worst result on "
              "that task, not dropped:")
        for t, o, d in imputed:
            print(f"      {t:<30} {o:<12} d := {d:+.4f}")

    for o in comps:
        ds = d_by_opt[o]
        cells = "  ".join(f"{t.split('_', 1)[-1][:10]}:{d:+.3f}"
                          for t, d in zip(tasks_used[o], ds))
        print(f"    {o:<10} {cells}")

    # ---- Wilcoxon signed-rank + sign test + Holm --------------------------
    print("\n" + "=" * 96)
    print(f"  {'optimizer':<12} {'n':>3} {'wins':>6} {'median d':>10} "
          f"{'wilcoxon p':>12} {'holm p':>10} {'sign p':>10}   verdict (alpha=0.05)")
    print("  " + "-" * 92)
    w_p = []
    for o in comps:
        ds = np.array(d_by_opt[o], dtype=float)
        n = len(ds)
        if n < 6:
            w_p.append(float("nan"))
            print(f"  {o:<12} {n:>3}  -- too few tasks --")
            continue
        wins = int(np.sum(ds > 0))
        try:
            wp = wilcoxon(ds, alternative="two-sided", zero_method="wilcox").pvalue
        except ValueError:
            wp = float("nan")
        w_p.append(wp)

    holm_p = holm([p if math.isfinite(p) else 1.0 for p in w_p])
    for o, wp, hp in zip(comps, w_p, holm_p):
        ds = np.array(d_by_opt[o], dtype=float)
        n = len(ds)
        if n < 6:
            continue
        wins = int(np.sum(ds > 0))
        sp = binomtest(wins, n, 0.5, alternative="two-sided").pvalue
        med = float(np.median(ds))
        better = med > 0
        sig = math.isfinite(hp) and hp < 0.05
        verdict = (f"{'BETTER' if better else 'WORSE'} than {ref}, significant"
                   if sig else f"no significant difference from {ref}")
        print(f"  {o:<12} {n:>3} {wins:>4}/{n:<1} {med:>+10.4f} "
              f"{wp:>12.4g} {hp:>10.4g} {sp:>10.4g}   {verdict}")

    # ---- per-domain descriptive breakdown --------------------------------
    print("\n  per-domain mean of d_i (descriptive only, no test):")
    for dom in ("vision", "language", "finance"):
        dt = [t for t in present if DOMAIN_OF[t] == dom]
        if not dt:
            continue
        bits = []
        for o in comps:
            vals = [d for t, d in zip(tasks_used[o], d_by_opt[o]) if t in dt]
            if vals:
                bits.append(f"{o}:{np.mean(vals):+.3f}")
        print(f"    {dom:<10} ({len(dt)} tasks)  " + "  ".join(bits))

    out = os.path.join(args.ckpt_root,
                       f"aggregate_12task_ref_{ref}.json")
    with open(out, "w") as fh:
        json.dump({
            "reference": ref, "tasks_present": present, "tasks_missing": missing,
            "task_numbers": table,
            "relative_improvement": {o: dict(zip(tasks_used[o], d_by_opt[o]))
                                     for o in comps},
            "imputed_total_failures": [{"task": t, "optimizer": o, "d": d}
                                       for t, o, d in imputed],
            "wilcoxon_p": dict(zip(comps, w_p)),
            "holm_p": dict(zip(comps, holm_p)),
        }, fh, indent=2)
    print("\n  wrote " + out)
    if ref != "EvieKF" and "EvieKF" in opts:
        print("  NOTE: this run's comparison family is every optimizer vs "
              f"{ref}. The paper's claim is EvieKF vs every baseline -- run\n"
              "        python benchmark12/aggregate.py --ref EvieKF\n"
              "        as well, or EvieKF-vs-Muon and EvieKF-vs-Shampoo are "
              "never tested.")
    print("=" * 96)


if __name__ == "__main__":
    main()
