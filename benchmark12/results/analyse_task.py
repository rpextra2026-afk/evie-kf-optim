#!/usr/bin/env python3
"""
analyse_task.py -- per-task read-out of one finished benchmark12 checkpoint.

This is DESCRIPTIVE. The confirmatory test is the 12-task paired Wilcoxon in
benchmark12/aggregate.py, on the pre-registered primary metric only (CLAUDE.md
sec 5). Nothing printed here is a p-value that belongs in the paper's claims --
the within-task Wilcoxon below is reported because it says whether a per-task
gap is consistent across units, not to be tested for significance.

    python benchmark12/results/analyse_task.py checkpoints/finance_tech
    python benchmark12/results/analyse_task.py <dir> --md > results/finance_tech.md

Reads results.json + summary.json (+ lr_search.json when present) and reports:
  * the corrected task number per optimizer, and the unit-MEAN variant, so the
    effect of the median-vs-mean fix on this task is visible
  * complete-case bookkeeping: which units were dropped and why
  * per-unit paired comparison of the reference arm against each baseline
  * seed stability, which is where a well-behaved optimizer shows up and which
    the task number (a median over seeds) hides
  * divergences per optimizer
  * wall-clock per cell -- the fixed-step budget is NOT wall-clock matched, and
    this is the number a reviewer will ask for
  * the Evie-KF mechanism telemetry, including n_active_steps, whose being
    nonzero is the wiring canary that the operator was not silently AdamW
  * the knob (gamma) response at the frozen LR, i.e. how much the tuned second
    axis actually moved the score
"""
import os
import sys
import json
import math
import argparse

import numpy as np

try:
    from scipy.stats import wilcoxon
except ImportError:
    wilcoxon = None


def _f(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


def load(d):
    R = json.load(open(os.path.join(d, "results.json")))
    S = json.load(open(os.path.join(d, "summary.json")))
    lr = None
    p = os.path.join(d, "lr_search.json")
    if os.path.isfile(p):
        lr = json.load(open(p))
    return R, S, lr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_dir", help="checkpoints/<task_name>")
    ap.add_argument("--ref", default="EvieKF",
                    help="arm the per-unit comparison is centred on")
    args = ap.parse_args()

    R, S, LR = load(args.ckpt_dir)
    cells, meta = R["cells"], R["meta"]
    OPTS, SEEDS = meta["opt_names"], meta["seeds"]
    UNITS = meta["units"]
    lower = S["primary_lower_better"]
    CC = S.get("units_complete_case", UNITS)
    tn = S["task_number_by_optimizer"]
    tm = S.get("task_number_unit_mean_by_optimizer", {})
    out = []

    def p(s=""):
        out.append(s)
        print(s)

    p(f"TASK {S['task_name']}   primary = {S['primary_metric']} "
      f"({'lower' if lower else 'higher'} better)")
    n_rec = sum(1 for c in cells.values() if c.get("reconstructed_from_log"))
    p(f"  cells {len(cells)}   units {len(UNITS)}   seeds {SEEDS}")
    if n_rec:
        p(f"  {n_rec} cells are log-reconstructed: exact primary + divergence "
          f"flag, but NO descriptive telemetry (see kaggle/recovery/README.md)")
    if len(CC) != len(UNITS):
        p(f"  complete-case: {len(CC)}/{len(UNITS)} units; dropped "
          f"{sorted(set(UNITS) - set(CC))}")
        p(f"  (a unit is dropped for EVERY arm when ANY arm failed on it, so a "
          f"failing arm is not scored on the easy units it survived)")

    p("\nTASK NUMBER  (median over seeds of the per-seed unit-"
      f"{S.get('unit_reduction','median')})")
    p(f"  {'optimizer':<12}{'primary':>16}{'vs '+args.ref:>12}"
      + (f"{'unit-MEAN':>16}" if tm else ""))
    order = sorted((o for o in OPTS if _f(tn.get(o))),
                   key=lambda o: tn[o], reverse=not lower)
    ref_v = tn.get(args.ref)
    for o in order:
        rel = ""
        if _f(ref_v) and o != args.ref:
            d = (tn[o] - ref_v) / abs(tn[o]) * (1 if lower else -1)
            rel = f"{d*100:+.3f}%"
        p(f"  {o:<12}{tn[o]:>16.9g}{rel:>12}"
          + (f"{tm.get(o, float('nan')):>16.9g}" if tm else ""))
    if tm:
        p("  (unit-MEAN is the OLD aggregation, shown so its distortion is "
          "visible; the target is not standardised, so a mean over units is "
          "dominated by the highest-variance ones)")

    def unit_med(o):
        return [float(np.median([cells[f"{o}|{u}|{s}"]["primary"] for s in SEEDS
                                 if f"{o}|{u}|{s}" in cells
                                 and _f(cells[f"{o}|{u}|{s}"].get("primary"))]))
                for u in CC]

    if len(CC) >= 6 and args.ref in OPTS:
        p(f"\nPER-UNIT PAIRED COMPARISON across the {len(CC)} complete-case "
          f"units  (descriptive)")
        p(f"  {'baseline':<12}{'median rel':>12}{'ref wins':>12}"
          + (f"{'wilcoxon p':>13}" if wilcoxon else ""))
        ref = np.array(unit_med(args.ref))
        for o in OPTS:
            if o == args.ref:
                continue
            b = np.array(unit_med(o))
            rel = (b - ref) / b * (1 if lower else -1)
            wins = int((ref < b).sum() if lower else (ref > b).sum())
            pv = ""
            if wilcoxon:
                try:
                    pv = f"{wilcoxon(ref, b).pvalue:>13.2e}"
                except ValueError:
                    pv = f"{'n/a':>13}"
            p(f"  {o:<12}{np.median(rel)*100:>11.3f}%{wins:>8}/{len(CC)}{pv}")
        p("  NOTE: with many paired units a negligible gap can still be "
          "'significant'. Read the median relative difference, not the p-value.")

    p("\nSEED STABILITY  (spread of the per-seed task number, complete-case)")
    for o in order:
        ps = [float(np.median([cells[f"{o}|{u}|{s}"]["primary"] for u in CC
                               if _f(cells.get(f"{o}|{u}|{s}", {}).get("primary"))]))
              for s in SEEDS]
        ps = [v for v in ps if _f(v)]
        if ps:
            p(f"  {o:<12} spread {(max(ps)-min(ps))/min(ps)*100:>7.2f}%   "
              f"[{min(ps):.6g}, {max(ps):.6g}]")

    p("\nDIVERGENCE  (cells flagged, of "
      f"{len(UNITS)*len(SEEDS)} per optimizer)")
    for o in OPTS:
        d = [k for k, c in cells.items() if c.get("diverged") and c.get("opt") == o]
        if d:
            p(f"  {o:<12}{len(d):>4}   units {sorted({k.split('|')[1] for k in d})}")
        else:
            p(f"  {o:<12}{0:>4}")

    p("\nWALL-CLOCK per cell  (telemetry-bearing cells only; the step budget is "
      "fixed, NOT wall-clock matched)")
    base = None
    for o in OPTS:
        w = [c["wall_s"] for c in cells.values()
             if c.get("opt") == o and _f(c.get("wall_s"))]
        if not w:
            continue
        m = float(np.mean(w))
        base = m if o == "AdamW" else base
        p(f"  {o:<12} n={len(w):>4}  mean {m:>8.1f}s"
          + (f"   {m/base:>5.2f}x AdamW" if base else ""))

    tel = [c for c in cells.values()
           if c.get("opt", "").startswith("Evie") and c.get("eviekf_n_active_steps")]
    if tel:
        p("\nEVIE-KF MECHANISM TELEMETRY  (descriptive only, never tested)")
        for k, why in (
            ("eviekf_n_active_steps", "wiring canary: 0 would mean set_noise() "
                                      "never fired and this WAS silently AdamW"),
            ("eviekf_cos_mean", "cos(preconditioned dir, plain Adam dir); 1.0 = "
                                "the operator changed nothing"),
            ("eviekf_b_simple_mean", "tr(Sigma_z)/||y||^2 -- gradient noise "
                                     "relative to signal"),
            ("eviekf_eff_rank_mean", "effective rank of the Kronecker noise "
                                     "spectrum"),
            ("eviekf_gain_lo_min", "smallest gain applied; 1.0 = no shrinkage "
                                   "anywhere"),
            ("eviekf_gain_hi_max", "largest gain; always <= 1 by construction"),
        ):
            v = [c[k] for c in tel if _f(c.get(k))]
            if v:
                p(f"  {k:<26} mean {np.mean(v):>11.5g}  "
                  f"[{min(v):.5g}, {max(v):.5g}]")
                p(f"  {'':<26} {why}")

    if LR:
        p("\nTUNED SECOND AXIS (knob) RESPONSE at the frozen LR")
        for o, e in LR.items():
            if not isinstance(e, dict) or e.get("knob") is None:
                continue
            pts = {}
            for k, v in (e.get("grid") or {}).items():
                a, b = k.split("|")
                pts.setdefault(float(a), {})[float(b)] = v
            if not pts:
                continue
            row = pts.get(min(pts, key=lambda x: abs(x - e["lr"])), {})
            if len(row) < 2:
                continue
            vals = [row[g] for g in sorted(row)]
            p(f"  {o}: frozen lr={e['lr']:.4g} knob={e['knob']:g}   "
              f"lr-boundary-extends={e.get('boundary_checks_lr')} "
              f"knob-boundary-extends={e.get('boundary_checks_knob')}")
            for g in sorted(row):
                p(f"      knob={g:<9g} score={row[g]:.6g}"
                  + ("   <-- frozen" if g == e["knob"] else ""))
            p(f"      spread across the knob axis: "
              f"{(max(vals)-min(vals))/min(vals)*100:.2f}% over a "
              f"{max(row)/min(row):.0f}x range")
    return "\n".join(out)


if __name__ == "__main__":
    main()
