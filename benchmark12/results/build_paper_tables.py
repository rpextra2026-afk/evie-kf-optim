#!/usr/bin/env python3
"""
build_paper_tables.py -- every table and protocol fact the paper needs, rebuilt
from whatever tasks have finished.

    python benchmark12/results/build_paper_tables.py

Scans benchmark12/results/<task>/ (each holding summary.json, results.json,
lr_search.json, log.txt copied from a finished checkpoint) and writes
benchmark12/results/paper/:

  SETUP.md                     experimental protocol, filled from the generated
                               task files and the run logs -- not from memory
  table1_main_results          task number per optimizer x task, best in bold
  table2_ranks                 rank per task and mean rank across tasks
  table3_seed_robustness       per-seed values, median, mean +- sd, spread
  table4_evie_vs_baselines     EvieKF against each baseline, per task
  table5_relative_improvement  the per-task d_i that aggregate.py tests
  table6_tuning_budget         frozen LR / knob and how much search each arm got
  table7_knob_sensitivity      the tuned second axis at the frozen LR
  table8_compute               wall-clock and memory per cell
  table9_divergence            diverged cells and complete-case units
  table10_evie_telemetry       the mechanism diagnostics

each as .md (reading), .tex (booktabs, paste into the paper) and .csv (plots).

EVERYTHING HERE IS DESCRIPTIVE except table5, and table5 is only the input to
the confirmatory test: the test itself is aggregate.py's paired Wilcoxon across
the 12 tasks, with Holm over the 7 comparisons, run once with --ref AdamW and
once with --ref EvieKF. Until 12 tasks exist it is not a result, and the tables
say how many tasks they cover.
"""
import re
import csv
import json
import math
import types
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OUT = HERE / "paper"

OPT_ORDER = ["EvieKF", "AdamW", "AdaBelief", "SGD", "Sophia", "Muon", "Lion", "Shampoo"]
REF = "EvieKF"
DOMAIN_ORDER = {"vision": 0, "language": 1, "finance": 2}
METRIC = {
    "test_mse":         ("test MSE",          1e4, r"$\times10^{-4}$", 4),
    "test_top1_acc":    ("test top-1 acc (%)", 1.0, "",                2),
    "test_perplexity":  ("test perplexity",   1.0, "",                2),
}


def _f(v):
    return isinstance(v, (int, float)) and math.isfinite(v)


# ---------------------------------------------------------------------------
#  loading
# ---------------------------------------------------------------------------
def _task_file(name):
    for sub in ("vision", "language", "finance", "scalegen", "ablation"):
        p = REPO / "benchmark12" / sub / f"{name}.py"
        if p.is_file():
            return p
    return None


def _shared_block(src):
    """Exec the SHARED OPTIMIZER BLOCK out of a generated file so the protocol
    tables quote the constants that actually ran."""
    import torch
    import torch.nn as nn
    a = src.index("# === SHARED OPTIMIZER BLOCK")
    b = src.index("# === END SHARED OPTIMIZER BLOCK ===")
    ns = {"torch": torch, "nn": nn, "math": math, "np": np, "WEIGHT_DECAY": 1e-4,
          "GRAD_CLIP_NORM": 5.0, "HESS_FREQ": 5, "N_HUTCH": 4,
          "log": lambda *a, **k: None}
    exec(compile(src[a:b], "<shared>", "exec"), ns)
    return types.SimpleNamespace(**ns)


def _header_const(src, name):
    m = re.search(rf"^{name}\s*=\s*([^\n#]+)", src, re.M)
    return m.group(1).strip() if m else None


def load_tasks():
    tasks = []
    for d in sorted(HERE.iterdir()):
        if not (d.is_dir() and (d / "summary.json").is_file()):
            continue
        S = json.load(open(d / "summary.json"))
        R = json.load(open(d / "results.json"))
        LR = json.load(open(d / "lr_search.json")) if (d / "lr_search.json").is_file() else {}
        log = (d / "log.txt").read_text(encoding="utf-8", errors="replace") \
            if (d / "log.txt").is_file() else ""
        tf = _task_file(S["task_name"])
        src = tf.read_text(encoding="utf-8") if tf else ""
        cfg = json.loads(re.search(r'TASK = json\.loads\(r"""\s*(\{.*?\})\s*"""',
                                   src, re.S).group(1)) if src else {}
        m = re.search(r"\[lr-search\] units=(\[.*?\])", log)
        lr_units = len(eval(m.group(1))) if m else None          # noqa: S307 -- our own log
        gpus = sorted(set(re.findall(r"\[preflight\] OK: (.+?)\s+sm_", log)))
        tasks.append(dict(name=S["task_name"], dir=d, S=S, R=R, LR=LR, log=log,
                          src=src, cfg=cfg, lr_units=lr_units, gpus=gpus))
    tasks.sort(key=lambda t: (DOMAIN_ORDER.get(t["cfg"].get("domain"), 9), t["name"]))
    return tasks


# ---------------------------------------------------------------------------
#  per-task derived quantities
# ---------------------------------------------------------------------------
def opts_of(t):
    have = t["S"]["opt_names"]
    return [o for o in OPT_ORDER if o in have] + [o for o in have if o not in OPT_ORDER]


def cell_ok(c):
    return bool(c) and not c.get("diverged") and _f(c.get("primary"))


def complete_units(t):
    return t["S"].get("units_complete_case") or t["R"]["meta"]["units"]


def per_seed(t, o):
    ps = t["S"].get("per_seed_value_by_optimizer", {}).get(o, {})
    return {int(k): v for k, v in ps.items() if _f(v)}


def paired_vs_ref(t, o):
    """(median relative difference, ref wins, n pairs, pairing).
    Pairs across complete-case UNITS when the task has more than one (finance
    tickers); otherwise across SEEDS. rel > 0 means the reference is better."""
    cells, seeds = t["R"]["cells"], t["R"]["meta"]["seeds"]
    lower = t["S"]["primary_lower_better"]
    units = complete_units(t)
    if len(units) > 1:
        def val(opt, u):
            v = [cells[f"{opt}|{u}|{s}"]["primary"] for s in seeds
                 if cell_ok(cells.get(f"{opt}|{u}|{s}"))]
            return float(np.median(v)) if v else float("nan")
        pairs = [(val(REF, u), val(o, u)) for u in units]
        how = "units"
    else:
        a, b = per_seed(t, REF), per_seed(t, o)
        pairs = [(a[s], b[s]) for s in seeds if s in a and s in b]
        how = "seeds"
    pairs = [(r, b) for r, b in pairs if _f(r) and _f(b) and b != 0]
    if not pairs:
        return float("nan"), 0, 0, how
    rel = [((b - r) if lower else (r - b)) / abs(b) for r, b in pairs]
    wins = sum(1 for r, b in pairs if (r < b if lower else r > b))
    return float(np.median(rel)), wins, len(pairs), how


def d_i(t, o, ref):
    """aggregate.py's direction-normalised relative improvement of o over ref;
    > 0 means o beat ref on this task."""
    tn = t["S"]["task_number_by_optimizer"]
    rv, ov = tn.get(ref), tn.get(o)
    if not (_f(rv) and _f(ov)) or rv == 0:
        return float("nan")
    return ((rv - ov) if t["S"]["primary_lower_better"] else (ov - rv)) / abs(rv)


# ---------------------------------------------------------------------------
#  output
# ---------------------------------------------------------------------------
def _tex_esc(s):
    return str(s).replace("_", r"\_").replace("%", r"\%").replace("#", r"\#")


def emit(stem, title, caption, header, rows, bold=None, note=None):
    """rows: list of lists of display strings; bold: set of (r, c) to embolden."""
    bold = bold or set()
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{stem}.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(header)
        w.writerows(rows)
    md = [f"## {title}", "", caption, ""]
    md.append("| " + " | ".join(header) + " |")
    md.append("|" + "|".join("---" for _ in header) + "|")
    for i, r in enumerate(rows):
        md.append("| " + " | ".join(f"**{v}**" if (i, j) in bold and v else v
                                     for j, v in enumerate(r)) + " |")
    if note:
        md += ["", note]
    (OUT / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    cols = "l" + "r" * (len(header) - 1)
    tex = [r"\begin{table}[t]", r"\centering", r"\small",
           rf"\caption{{{_tex_esc(caption)}}}", rf"\label{{tab:{stem}}}",
           rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
           " & ".join(_tex_esc(h) for h in header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        tex.append(" & ".join((rf"\textbf{{{_tex_esc(v)}}}" if (i, j) in bold and v
                               else _tex_esc(v)) for j, v in enumerate(r)) + r" \\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (OUT / f"{stem}.tex").write_text("\n".join(tex) + "\n", encoding="utf-8")
    return "\n".join(md)


def fmt(t, v):
    name, scale, _, dp = METRIC[t["S"]["primary_metric"]]
    return "--" if not _f(v) else f"{v * scale:.{dp}f}"


def short(name):
    return name.replace("finance_", "fin/").replace("vision_", "vis/").replace("language_", "lang/")


# ---------------------------------------------------------------------------
def main():
    T = load_tasks()
    if not T:
        raise SystemExit(f"no finished task folders under {HERE}")
    n = len(T)
    cover = f"Covers {n} of 12 tasks: {', '.join(t['name'] for t in T)}."
    all_opts = [o for o in OPT_ORDER if any(o in t["S"]["opt_names"] for t in T)]
    sections = []

    # ---- table 1: main results ------------------------------------------------
    header = ["Optimizer"] + [short(t["name"]) for t in T]
    rows, bold = [], set()
    for i, o in enumerate(all_opts):
        rows.append([o] + [fmt(t, t["S"]["task_number_by_optimizer"].get(o)) for t in T])
    for j, t in enumerate(T, start=1):
        tn = t["S"]["task_number_by_optimizer"]
        lower = t["S"]["primary_lower_better"]
        best = (min if lower else max)((v for v in tn.values() if _f(v)))
        for i, o in enumerate(all_opts):
            if tn.get(o) == best:
                bold.add((i, j))
    units = "; ".join(f"{short(t['name'])}: {METRIC[t['S']['primary_metric']][0]}"
                      f"{' x1e' + str(-round(math.log10(METRIC[t['S']['primary_metric']][1]))) if METRIC[t['S']['primary_metric']][1] != 1 else ''}"
                      for t in T)
    sections.append(emit(
        "table1_main_results", "Table 1 -- main results",
        f"Primary metric per task: median over {len(T[0]['R']['meta']['seeds'])} "
        f"seeds of the per-seed value (for multi-unit tasks, the median over "
        f"complete-case units). Best per task in bold. {units}. Complete-case units "
        f"per task: " + ", ".join(
            f"{short(t['name'])} {len(complete_units(t))}/{len(t['R']['meta']['units'])}"
            for t in T) + f" (Table 9b re-scores with failures counted). {cover}",
        header, rows, bold))

    # ---- table 2: ranks ---------------------------------------------------------
    rank_of = {}
    for t in T:
        tn = t["S"]["task_number_by_optimizer"]
        lower = t["S"]["primary_lower_better"]
        order = sorted((o for o in all_opts if _f(tn.get(o))), key=lambda o: tn[o],
                       reverse=not lower)
        rank_of[t["name"]] = {o: order.index(o) + 1 for o in order}
    rows = []
    for o in all_opts:
        rs = [rank_of[t["name"]].get(o) for t in T]
        good = [r for r in rs if r]
        rows.append([o] + [str(r) if r else "--" for r in rs] +
                    [f"{np.mean(good):.2f}" if good else "--"])
    best_mean = min(float(r[-1]) for r in rows if r[-1] != "--")
    bold = {(i, len(T) + 1) for i, r in enumerate(rows) if r[-1] != "--" and float(r[-1]) == best_mean}
    sections.append(emit(
        "table2_ranks", "Table 2 -- rank per task",
        f"Rank of each optimizer on each task (1 = best) and mean rank. Descriptive "
        f"summary only. {cover}",
        ["Optimizer"] + [short(t["name"]) for t in T] + ["mean rank"], rows, bold))

    # ---- table 3: seed robustness ---------------------------------------------
    rows = []
    for t in T:
        seeds = t["R"]["meta"]["seeds"]
        for o in opts_of(t):
            ps = per_seed(t, o)
            v = [ps[s] for s in seeds if s in ps]
            if not v:
                continue
            sd = float(np.std(v, ddof=1)) if len(v) > 1 else 0.0
            spread = (max(v) - min(v)) / abs(min(v)) * 100 if min(v) else float("nan")
            rows.append([short(t["name"]), o] + [fmt(t, ps.get(s)) for s in seeds] +
                        [fmt(t, float(np.median(v))),
                         f"{fmt(t, float(np.mean(v)))} ± {fmt(t, sd)}",
                         f"{spread:.2f}%"])
    seeds0 = T[0]["R"]["meta"]["seeds"]
    sections.append(emit(
        "table3_seed_robustness", "Table 3 -- robustness across seeds",
        "Per-seed primary metric (seeds " + ", ".join(map(str, seeds0)) +
        "), with the median that enters Table 1, mean ± sample sd, and the spread "
        "(max-min)/min. Same units and scaling as Table 1. " + cover,
        ["Task", "Optimizer"] + [f"seed {s}" for s in seeds0] +
        ["median", "mean ± sd", "spread"], rows))

    # ---- table 3b: sensitivity to the LR-search seed ------------------------------
    # The LR search runs at LR_SEARCH_SEED (42), and make_loaders() seeds the batch
    # order from the seed alone -- so the search run sees exactly the data order of
    # main-run seed 42, and every frozen LR is tuned to that order. Seed 42 is
    # therefore potentially optimistic. Recomputing every task number without it
    # shows whether any conclusion depends on it.
    search_seed = int(_header_const(T[0]["src"], "LR_SEARCH_SEED") or 42)
    rows = []
    s42_best = s42_n = 0
    for t in T:
        tn = t["S"]["task_number_by_optimizer"]
        lower = t["S"]["primary_lower_better"]
        excl = {}
        for o in opts_of(t):
            ps = per_seed(t, o)
            rest = [v for s, v in ps.items() if s != search_seed]
            excl[o] = float(np.median(rest)) if rest else float("nan")
            if search_seed in ps and len(ps) > 1:
                s42_n += 1
                s42_best += ps[search_seed] == (min if lower else max)(ps.values())
        r_all = sorted((o for o in excl if _f(tn.get(o))), key=lambda o: tn[o], reverse=not lower)
        r_exc = sorted((o for o in excl if _f(excl[o])), key=lambda o: excl[o], reverse=not lower)
        for o in opts_of(t):
            ch = (excl[o] - tn[o]) / abs(tn[o]) * 100 if _f(tn.get(o)) and _f(excl[o]) else float("nan")
            rows.append([short(t["name"]), o, fmt(t, tn.get(o)), fmt(t, excl[o]),
                         f"{ch:+.2f}%" if _f(ch) else "--",
                         str(r_all.index(o) + 1) if o in r_all else "--",
                         str(r_exc.index(o) + 1) if o in r_exc else "--"])
    sections.append(emit(
        "table3b_excluding_search_seed",
        f"Table 3b -- sensitivity: excluding the LR-search seed ({search_seed})",
        f"The learning-rate search runs at seed {search_seed}, and the batch order is "
        f"determined by the seed alone, so the search saw exactly the data order of "
        f"main-run seed {search_seed} and each frozen LR is tuned to it. This table "
        f"recomputes every task number as the median over the other seeds only. So "
        f"far seed {search_seed} was the best of the seeds in {s42_best} of {s42_n} "
        f"optimizer-task cases (chance: {s42_n / len(seeds0):.1f}). " + cover,
        ["Task", "Optimizer", "all seeds", f"excl. {search_seed}", "change",
         "rank (all)", f"rank (excl. {search_seed})"], rows))

    # ---- table 4: EvieKF vs each baseline ---------------------------------------
    rows = []
    for t in T:
        for o in opts_of(t):
            if o == REF:
                continue
            rel, wins, npair, how = paired_vs_ref(t, o)
            d = d_i(t, REF, o)
            rows.append([short(t["name"]), o,
                         f"{d * 100:+.2f}%" if _f(d) else "--",
                         f"{rel * 100:+.3f}%" if _f(rel) else "--",
                         f"{wins}/{npair} {how}"])
    sections.append(emit(
        "table4_evie_vs_baselines", "Table 4 -- EvieKF against each baseline",
        "Positive = EvieKF better. 'task-level' compares the Table 1 numbers. "
        "'paired median' pairs EvieKF with the baseline unit by unit (complete-case "
        "tickers) for multi-unit tasks, or seed by seed otherwise, and takes the "
        "median relative difference; 'EvieKF wins' counts those pairs. Descriptive: "
        "no p-values, because with many paired units a negligible gap is still "
        "'significant'. " + cover,
        ["Task", "Baseline", "task-level Δ", "paired median Δ", "EvieKF wins"], rows))

    # ---- table 5: d_i for aggregate.py ----------------------------------------
    rows = []
    for ref in ("AdamW", "EvieKF"):
        for o in all_opts:
            if o == ref:
                continue
            ds = [d_i(t, o, ref) for t in T]
            rows.append([ref, o] + [f"{d:+.4f}" if _f(d) else "--" for d in ds] +
                        [f"{sum(1 for d in ds if _f(d) and d > 0)}/{sum(1 for d in ds if _f(d))}"])
    sections.append(emit(
        "table5_relative_improvement", "Table 5 -- per-task relative improvement d_i",
        "d_i = (ref - opt)/|ref| for lower-is-better metrics, sign-flipped otherwise; "
        "d_i > 0 means the optimizer beat the reference on that task. These are "
        "exactly the values benchmark12/aggregate.py feeds to the paired Wilcoxon "
        "across tasks (with Holm over 7 comparisons). The test is run twice, "
        f"reference AdamW and reference EvieKF. NOT A RESULT until all 12 tasks "
        f"exist -- {cover}",
        ["Reference", "Optimizer"] + [short(t["name"]) for t in T] + ["beats ref"], rows))

    # ---- table 6: tuning budget --------------------------------------------------
    rows = []
    for t in T:
        steps = int(t["cfg"].get("lr_search_steps", 2500))
        u = t["lr_units"] or 1
        for o in opts_of(t):
            e = t["LR"].get(o) or {}
            pts = len(e.get("grid") or {})
            rows.append([short(t["name"]), o,
                         f"{e['lr']:.4g}" if _f(e.get("lr")) else "--",
                         "--" if e.get("knob") is None else f"{e['knob']:g}",
                         str(pts), str(pts * u), f"{pts * u * steps:,}",
                         str(e.get("boundary_checks_lr", 0)),
                         str(e.get("boundary_checks_knob", 0))])
    sections.append(emit(
        "table6_tuning_budget", "Table 6 -- frozen hyperparameters and tuning budget",
        "Frozen learning rate and second-axis knob (EvieKF: gamma; Shampoo: beta), "
        "and how much search produced them: grid points scored, training runs "
        "(points x LR-search units -- 8 tickers for finance, 1 otherwise), total "
        "search steps, and boundary extensions. EvieKF and Shampoo search a joint "
        "(lr, knob) grid; every other arm searches LR only, so the budget is NOT "
        "equal across arms -- this table is the disclosure. " + cover,
        ["Task", "Optimizer", "frozen lr", "frozen knob", "grid pts",
         "search runs", "search steps", "lr extends", "knob extends"], rows))

    # ---- table 7: knob sensitivity -----------------------------------------------
    rows = []
    for t in T:
        for o in opts_of(t):
            e = t["LR"].get(o) or {}
            if e.get("knob") is None or not e.get("grid"):
                continue
            pts = {}
            for k, v in e["grid"].items():
                a, b = k.split("|")
                pts.setdefault(float(a), {})[float(b)] = v
            row = pts.get(min(pts, key=lambda x: abs(x - e["lr"])), {})
            if len(row) < 2:
                continue
            vals = [row[g] for g in sorted(row)]
            for g in sorted(row):
                rows.append([short(t["name"]), o, f"{g:g}", fmt(t, row[g]),
                             "frozen" if g == e["knob"] else ""])
            rows.append([short(t["name"]), o, "spread", "",
                         f"{(max(vals) - min(vals)) / abs(min(vals)) * 100:.2f}% over "
                         f"{max(row) / min(row):.0f}x"])
    sections.append(emit(
        "table7_knob_sensitivity", "Table 7 -- sensitivity to the tuned second axis",
        "LR-search score (best validation value, 1 seed, 25% of the step budget) at "
        "the frozen learning rate, for each value of the knob. A flat response means "
        "the knob does not matter on that task. Same units and scaling as Table 1. "
        + cover, ["Task", "Optimizer", "knob", "val score", ""], rows))

    # ---- table 8: compute ----------------------------------------------------------
    rows = []
    for t in T:
        cells = t["R"]["cells"]
        base = None
        per = {}
        for o in opts_of(t):
            cs = [c for c in cells.values() if c.get("opt") == o and _f(c.get("wall_s"))]
            if cs:
                per[o] = cs
        if "AdamW" in per:
            base = float(np.mean([c["wall_s"] for c in per["AdamW"]]))
        for o, cs in per.items():
            w = float(np.mean([c["wall_s"] for c in cs]))
            mem = [c["peak_mem_mb"] for c in cs if _f(c.get("peak_mem_mb"))]
            rows.append([short(t["name"]), o, str(len(cs)), f"{w:.1f}",
                         f"{w / base:.2f}x" if base else "--",
                         f"{np.mean(mem):.0f}" if mem else "--"])
    sections.append(emit(
        "table8_compute", "Table 8 -- compute per training cell",
        "Mean wall-clock seconds and peak GPU memory per (optimizer, unit, seed) "
        "cell of 10,000 steps, on the GPU listed in SETUP.md. The step budget is "
        "fixed, not wall-clock matched. Cells recovered from logs carry no "
        "wall-clock, so 'n' can be below the full count. " + cover,
        ["Task", "Optimizer", "n", "s / cell", "vs AdamW", "peak MB"], rows))

    # ---- table 9: divergence -------------------------------------------------------
    rows = []
    for t in T:
        cells, units, seeds = t["R"]["cells"], t["R"]["meta"]["units"], t["R"]["meta"]["seeds"]
        tot = len(units) * len(seeds)
        for o in opts_of(t):
            dv = [k for k, c in cells.items() if c.get("opt") == o and c.get("diverged")]
            rows.append([short(t["name"]), o, f"{len(dv)}/{tot}",
                         ", ".join(sorted({k.split('|')[1] for k in dv})) or "--"])
        cc = complete_units(t)
        rows.append([short(t["name"]), "complete-case", f"{len(cc)}/{len(units)} units",
                     ", ".join(sorted(set(units) - set(cc))) or "none dropped"])
    sections.append(emit(
        "table9_divergence", "Table 9 -- divergence and complete-case units",
        "Cells flagged diverged (non-finite loss, outside the task's absolute sane "
        "range, or beyond the peer-median factor), and the units on which they "
        "occurred. A unit on which any arm failed is dropped for every arm, so no "
        "arm is scored only on the units it survived. " + cover,
        ["Task", "Optimizer", "diverged", "units"], rows))

    # ---- table 9b: failures counted as losses (sensitivity) -------------------------
    # The primary aggregation is paired complete-case: a unit on which ANY arm failed
    # is dropped for EVERY arm. That keeps the comparison paired, but it also means
    # an arm's own failures never enter its own task number -- an optimizer that
    # breaks on 18 of 49 tickers is scored only on the 31 it survived, same as one
    # that never broke. This sensitivity keeps every unit and counts each diverged
    # cell as the worst finite value observed on that unit by any arm at any seed
    # -- the same "failure = worst result" rule aggregate.py already applies at task
    # level. It is a sensitivity analysis, not the pre-registered primary.
    rows = []
    for t in T:
        cells, units, seeds = t["R"]["cells"], t["R"]["meta"]["units"], t["R"]["meta"]["seeds"]
        lower = t["S"]["primary_lower_better"]
        worst = {}
        for u in units:
            v = [c["primary"] for c in cells.values() if c.get("unit") == u and cell_ok(c)]
            if v:
                worst[u] = (max if lower else min)(v)
        pen = {}
        for o in opts_of(t):
            per = []
            for s in seeds:
                vals = []
                for u in units:
                    c = cells.get(f"{o}|{u}|{s}")
                    if cell_ok(c):
                        vals.append(c["primary"])
                    elif u in worst:
                        vals.append(worst[u])
                if vals:
                    per.append(float(np.median(vals)))
            pen[o] = float(np.median(per)) if per else float("nan")
        tn = t["S"]["task_number_by_optimizer"]
        r_cc = sorted((o for o in pen if _f(tn.get(o))), key=lambda o: tn[o], reverse=not lower)
        r_pen = sorted((o for o in pen if _f(pen[o])), key=lambda o: pen[o], reverse=not lower)
        for o in opts_of(t):
            nd = sum(1 for c in cells.values() if c.get("opt") == o and c.get("diverged"))
            rows.append([short(t["name"]), o, str(nd), fmt(t, tn.get(o)), fmt(t, pen[o]),
                         str(r_cc.index(o) + 1) if o in r_cc else "--",
                         str(r_pen.index(o) + 1) if o in r_pen else "--"])
    sections.append(emit(
        "table9b_failures_as_losses", "Table 9b -- sensitivity: failures counted as losses",
        "Primary aggregation (complete-case: a unit any arm failed on is dropped for "
        "all arms) against a sensitivity that keeps every unit and scores each "
        "diverged cell as the worst finite value any arm reached on that unit. Under "
        "complete-case an arm's own failures do not enter its task number; here they "
        "do. Same units and scaling as Table 1. Sensitivity analysis only. " + cover,
        ["Task", "Optimizer", "diverged cells", "complete-case", "failures as losses",
         "rank (CC)", "rank (failures)"], rows))

    # ---- table 9c: divergence rate across tasks ---------------------------------------
    rows = []
    for o in all_opts:
        per_task, tot_d, tot_c = [], 0, 0
        for t in T:
            if o not in t["S"]["opt_names"]:
                per_task.append("--"); continue
            n_c = len(t["R"]["meta"]["units"]) * len(t["R"]["meta"]["seeds"])
            n_d = sum(1 for c in t["R"]["cells"].values() if c.get("opt") == o and c.get("diverged"))
            per_task.append(str(n_d)); tot_d += n_d; tot_c += n_c
        rows.append([o] + per_task + [f"{tot_d}/{tot_c}", f"{tot_d / tot_c * 100:.2f}%" if tot_c else "--"])
    sections.append(emit(
        "table9c_divergence_rate", "Table 9c -- divergence across tasks",
        "Diverged cells per optimizer on each task, and the overall rate. Descriptive. "
        + cover, ["Optimizer"] + [short(t["name"]) for t in T] + ["total", "rate"], rows))

    # ---- table 10: EvieKF telemetry ------------------------------------------------
    keys = [("eviekf_cos_mean", "cos(update, Adam dir)"),
            ("eviekf_b_simple_mean", "noise/signal b"),
            ("eviekf_eff_rank_mean", "noise eff. rank"),
            ("eviekf_gain_lo_min", "min gain")]
    rows = []
    for t in T:
        cs = [c for c in t["R"]["cells"].values()
              if c.get("opt") == REF and c.get("eviekf_n_active_steps")]
        if not cs:
            continue
        r = [short(t["name"]), str(len(cs))]
        for k, _ in keys:
            v = [c[k] for c in cs if _f(c.get(k))]
            r.append(f"{np.mean(v):.3g} [{min(v):.3g}, {max(v):.3g}]" if v else "--")
        rows.append(r)
    sections.append(emit(
        "table10_evie_telemetry", "Table 10 -- EvieKF mechanism diagnostics",
        "Mean [min, max] over cells. cos(update, Adam dir) = 1 means the operator "
        "left the AdamW direction unchanged. b = tr(Sigma_z)/||y||^2, the gradient "
        "noise relative to signal. Effective rank of the Kronecker noise spectrum. "
        "min gain = 1 means no shrinkage. Caveats: gains are recorded only in the "
        "Kronecker (2-D) branch, not the 1-D/diagonal branch; and the active-step "
        "counter resets when a cell resumes across a session break, so it is omitted "
        "here. Descriptive only. " + cover,
        ["Task", "cells"] + [lbl for _, lbl in keys], rows))

    # ---- SETUP.md ------------------------------------------------------------------
    t0 = T[0]
    Sb = _shared_block(t0["src"])
    hc = lambda k: _header_const(t0["src"], k)
    lines = ["# Experimental setup", "",
             f"Generated by `build_paper_tables.py` from the finished task folders and "
             f"the generated task files. {cover}", "",
             "## Tasks", "",
             "| task | domain | model | units | primary metric | batch | steps | "
             "LR-search steps | seeds | eval every | complete-case units | GPU |",
             "|" + "---|" * 12]
    for t in T:
        c, S, meta = t["cfg"], t["S"], t["R"]["meta"]
        lines.append(
            f"| `{t['name']}` | {c.get('domain')} | {c.get('title', '').split(' / ')[-1]} | "
            f"{len(meta['units'])} | {METRIC[S['primary_metric']][0]} | {c.get('batch_size')} | "
            f"{meta.get('max_steps')} | {c.get('lr_search_steps', 2500)} | "
            f"{len(meta['seeds'])} ({', '.join(map(str, meta['seeds']))}) | {c.get('eval_every')} | "
            f"{S.get('n_units_complete_case', len(meta['units']))}/{len(meta['units'])} | "
            f"{', '.join(t['gpus']) or '?'} |")
    lines += ["", "## Training protocol (common to every arm)", "",
              f"- Fixed step budget: {t0['R']['meta']['max_steps']:,} optimizer steps per "
              f"cell -- not epochs, and not wall-clock matched.",
              f"- Learning-rate schedule: cosine from the base LR down to "
              f"{hc('ETA_MIN_FRAC')} x base LR over the budget.",
              f"- Weight decay {hc('WEIGHT_DECAY')}, decoupled (p <- p(1 - lr*lambda), "
              f"applied before the update), for every optimizer.",
              f"- Global gradient-norm clipping at {hc('GRAD_CLIP_NORM')}.",
              "- float32 throughout, no mixed precision. Xavier-uniform weights, zero biases.",
              "- Best-validation checkpoint restored before the single final test "
              "evaluation on the full, untouched test split.",
              "- Finance: 3-layer MLP (5 -> 64 -> 32 -> 1, GELU) on standardised OHLCV, "
              "next-day log return, MSE loss; one model per ticker; 80/10/10 "
              "chronological split, scaler fit on train only; tickers with >= 300 "
              "usable rows.",
              "- Models with BatchNorm: every arm uses the same split-batch "
              "forward/backward as EvieKF, so BatchNorm statistics are identical "
              "across arms and only the preconditioner differs.", "",
              "## Learning-rate search", "",
              f"- 7-point half-decade grid (factor sqrt(10)), centred per optimizer family; "
              f"{t0['cfg'].get('lr_search_steps', 2500):,} steps (25% of the budget), "
              f"one seed ({hc('LR_SEARCH_SEED')}).",
              "- Scored on the best validation checkpoint of the search run, never the "
              "last step. Finance: median over 8 search tickers drawn once with "
              "random.Random(42) from the panel and reused for every arm.",
              "- Boundary rule: if the pick lands on a grid edge the grid is extended by "
              "two half-decade steps and re-searched, up to 3 times per axis.",
              "- EvieKF and Shampoo search a JOINT (lr, knob) grid; all other arms search "
              "LR only. The resulting tuning-budget asymmetry is reported in Table 6.", "",
              "| optimizer | LR grid centre | LR grid (before boundary extension) | second axis |",
              "|---|---|---|---|"]
    for o in all_opts:
        c0 = Sb.LR_GRID_CENTRE.get(o)
        knob = Sb.KNOB_GRID.get(o)
        lines.append(f"| {o} | {c0:g} | {', '.join(f'{x:g}' for x in Sb.half_decade_grid(c0))} | "
                     f"{('gamma ' if 'Evie' in o else 'beta ') + ', '.join(f'{x:g}' for x in knob) if knob else '--'} |")
    lines += ["", "## Fixed optimizer hyperparameters", "",
              "| optimizer | fixed settings |", "|---|---|",
              f"| AdamW | betas (0.9, 0.999), eps 1e-8 |",
              f"| AdaBelief | betas {Sb.ADABELIEF_BETAS}, eps {Sb.ADABELIEF_EPS:g} |",
              f"| SGD | momentum {Sb.SGDM_MOMENTUM}, Nesterov {Sb.SGDM_NESTEROV} |",
              f"| Sophia | betas {Sb.SOPHIA_BETAS}, rho {Sb.SOPHIA_RHO}, eps {Sb.SOPHIA_EPS:g}; "
              f"Hutchinson diagonal Hessian every {hc('HESS_FREQ')} steps with {hc('N_HUTCH')} probes |",
              f"| Muon | momentum {Sb.MUON_MOMENTUM}, Nesterov {Sb.MUON_NESTEROV}, "
              f"{Sb.MUON_NS_STEPS} Newton-Schulz steps, coefficients {Sb.MUON_NS_COEFFS}, "
              f"update scale {Sb.MUON_UPDATE_SCALE}; >=2-D weights only, AdamW (betas "
              f"{Sb.MUON_ADAMW_BETAS}) on the rest at AdamW's frozen LR |",
              f"| Lion | betas {Sb.LION_BETAS} |",
              f"| Shampoo | eps {Sb.SHAMPOO_EPS:g}, root refresh every {Sb.SHAMPOO_UPDATE_FREQ} "
              f"steps, AdamW-norm graft (betas {Sb.SHAMPOO_GRAFT_BETAS}), factors > "
              f"{Sb.SHAMPOO_MAXF} fall back to the graft |",
              f"| EvieKF | betas {Sb.EVIEKF_BETAS}, eps {Sb.EVIEKF_EPS:g}, noise-covariance EMA "
              f"{Sb.EVIEKF_BETA_SIG}, warmup {Sb.EVIEKF_WARMUP} steps, eigendecomposition "
              f"every {Sb.EVIEKF_REFRESH} steps, Kronecker factors > {Sb.EVIEKF_MAXF} "
              f"use a diagonal approximation; centred noise from a split batch |", "",
              "## Aggregation and statistics", "",
              "- Multi-unit tasks (finance): a unit is kept only if every arm finished it "
              "at every seed (paired complete-case); per seed, the median over those "
              "units; then the median over seeds. Single-unit tasks: median over seeds.",
              "- A cell is diverged if its loss is non-finite, its primary metric is "
              "outside the task's absolute sane range, or it is beyond the task's "
              "peer-median factor.",
              "- Confirmatory test (pre-registered, primary metric only): paired "
              "Wilcoxon signed-rank across the 12 tasks on d_i (Table 5), sign test as a "
              "robustness check, Holm correction over the 7 comparisons; run with "
              "reference AdamW and with reference EvieKF. Every other quantity is "
              "descriptive.", "",
              "## Software", "",
              f"- PyTorch {', '.join(sorted({t['R']['meta'].get('torch', '?') for t in T}))}.",
              "- GPUs per task in the table above; all runs on Kaggle, resumed across "
              "sessions from per-cell checkpoints.", "",
              "## Provenance notes (must accompany the numbers)", ""]
    for t in T:
        cells = t["R"]["cells"]
        rec = sum(1 for c in cells.values() if c.get("reconstructed_from_log"))
        notes = []
        if rec:
            notes.append(f"{rec} of {len(cells)} cells recovered from run logs after a "
                         f"lost Kaggle working directory: exact primary metric and "
                         f"divergence flag, no wall-clock or telemetry "
                         f"(kaggle/recovery/README.md)")
        if (t["dir"] / "summary_OLD_ENGINE.json").is_file():
            notes.append("its first summary came from the pre-fix aggregation; the "
                         "numbers here are from re-aggregation on the fixed engine, and "
                         "the old summary is kept beside it for audit")
        n_resume = len(re.findall(r"\[resume\] ", t["log"]))
        if n_resume:
            notes.append(f"{n_resume} cell(s) resumed mid-training across a session break "
                         f"from their own checkpoint (results unaffected; only the "
                         f"active-step counter in Table 10 resets)")
        lines.append(f"- `{t['name']}`: " + ("; ".join(notes) if notes else
                     "no cells recovered from logs; full telemetry."))
    lines += ["",
              f"- Every task: the LR search runs at seed {search_seed} and shares that "
              f"seed's batch order, so seed {search_seed} may be optimistic. Table 3b "
              f"reports every number with it excluded."]
    (OUT / "SETUP.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    (OUT / "ALL_TABLES.md").write_text(
        "# Paper tables\n\nRegenerate with `python benchmark12/results/build_paper_tables.py`. "
        f"{cover} Each table also exists as .tex and .csv.\n\n" +
        "\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"wrote {len(list(OUT.glob('*')))} files to {OUT}  ({cover})")


if __name__ == "__main__":
    main()
