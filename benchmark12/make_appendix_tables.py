"""Appendix tables, built from the committed run artefacts. No hardcoded results.

Everything here is recomputed with the main-run aggregation protocol --
complete-case units, median over units per seed, median over seeds -- so the
sweep tables are comparable to the main tables rather than to the sweep engine's
own mean-over-tickers reduction (which does not drop failed units).

    python benchmark12/make_appendix_tables.py

Writes one .tex fragment per table into results/paper/appendix/.
"""

import collections
import csv
import io
import json
import os
import statistics as st

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "results")
OUT = os.path.join(RES, "paper", "appendix")

ARMS6 = ["AdamW", "AdaBelief", "Sophia", "Muon", "EvieKF", "EvieKFu"]
ARMS3 = ["AdamW", "AdaBelief", "EvieKF"]
ARMS5 = ["AdamW", "AdaBelief", "Sophia", "Muon", "EvieKF"]


# --------------------------------------------------------------------------
# aggregation

def _cells_from_json(path):
    """(label, opt, unit, seed) -> primary, from a sweep's sweep_results.json."""
    d = json.load(io.open(path, encoding="utf-8"))
    out = {}
    for key, c in d["cells"].items():
        label, opt, unit, seed = key.split("|")
        out[(label, opt, unit, int(seed))] = (c["primary"], bool(c["diverged"]))
    return out, d["meta"]


def _cells_from_csv(path):
    out = {}
    for r in csv.DictReader(io.open(path, encoding="utf-8")):
        try:
            v = float(r["primary"])
        except (TypeError, ValueError):
            v = float("nan")
        out[(r["label"], r["opt"], r["unit"], int(r["seed"]))] = (
            v, r["diverged"].strip().lower() == "true")
    return out


def task_level(cells, label, arms):
    """Main-run protocol: complete-case units across `arms`, median over units
    per seed, median over seeds. Returns {arm: value}, n_units."""
    ok = collections.defaultdict(dict)          # arm -> (unit, seed) -> value
    units = collections.defaultdict(set)        # arm -> units seen at all
    bad = set()
    for (lab, opt, unit, seed), (v, div) in cells.items():
        if lab != label or opt not in arms:
            continue
        units[opt].add(unit)
        if div or v != v:
            bad.add(unit)
        else:
            ok[opt][(unit, seed)] = v
    if len(ok) < len(arms):
        return {}, 0
    common = set.intersection(*[units[a] for a in arms]) - bad
    # a unit must also have every (arm, seed) present
    seeds = sorted({s for d in ok.values() for (_, s) in d})
    common = {u for u in common
              if all((u, s) in ok[a] for a in arms for s in seeds)}
    if not common:
        return {}, 0
    res = {}
    for a in arms:
        per_seed = [st.median([ok[a][(u, s)] for u in common]) for s in seeds]
        res[a] = st.median(per_seed)
    return res, len(common)


# --------------------------------------------------------------------------
# latex helpers

def fmt(v, dec):
    return "---" if v is None else ("%%.%df" % dec) % v


def bold_best(row, lower):
    vals = [v for v in row.values() if v is not None]
    if not vals:
        return set()
    b = (min if lower else max)(vals)
    return {k for k, v in row.items() if v == b}


def write(name, lines):
    if not os.path.isdir(OUT):
        os.makedirs(OUT)
    p = os.path.join(OUT, name)
    io.open(p, "w", encoding="utf-8").write("\n".join(lines) + "\n")
    print("wrote %s" % os.path.relpath(p, HERE))


# --------------------------------------------------------------------------
# G.2a -- batch-size sweep

BATCH_HOSTS = [
    ("vision_cifar10_batchsize", "vision_cifar10",
     "CIFAR-10 --- test top-1 accuracy (\\%), higher is better", False, 2),
    ("language_wikitext2_batchsize", "language_wikitext2",
     "WikiText-2 --- test perplexity, lower is better", True, 2),
    ("finance_tech_batchsize", "finance_tech",
     "Finance-tech --- test MSE $\\times10^{-4}$, lower is better", True, 4),
]


def main_run_point(base, arms):
    """The reused baseline cell, from the main run's own results.json."""
    p = os.path.join(RES, base, "results.json")
    if not os.path.isfile(p):
        return {}, 0
    cells = {}
    for key, c in json.load(io.open(p, encoding="utf-8"))["cells"].items():
        parts = key.split("|")
        opt, unit, seed = parts[0], parts[1], parts[2]
        if len(parts) == 4:
            opt, unit, seed = parts[1], parts[2], parts[3]
        cells[("main", c.get("opt", opt), c.get("unit", unit),
               int(c.get("seed", seed)))] = (c["primary"], bool(c["diverged"]))
    # the sweeps use seeds 42-44 only
    cells = {k: v for k, v in cells.items() if k[3] in (42, 43, 44)}
    return task_level(cells, "main", arms)


def table_batchsize():
    rows = ["\\toprule",
            "batch & " + " & ".join(ARMS6).replace("EvieKFu", "Evie-KFu")
            .replace("EvieKF ", "Evie-KF ") + " & units \\\\"]
    rows[-1] = ("batch & AdamW & AdaBelief & Sophia & Muon & Evie-KF & "
                "Evie-KFu & units \\\\")
    for sweep, base, header, lower, dec in BATCH_HOSTS:
        rows += ["\\midrule",
                 "\\multicolumn{8}{l}{\\emph{%s}} \\\\" % header]
        path = os.path.join(RES, "sweeps", sweep)
        if os.path.isfile(os.path.join(path, "results_raw.csv")):
            cells = _cells_from_csv(os.path.join(path, "results_raw.csv"))
            meta = json.load(io.open(os.path.join(path, "sweep_results.json"),
                                     encoding="utf-8"))["meta"]
        else:
            cells, meta = _cells_from_json(
                os.path.join(path, "sweep_results.json"))
        scale = 1e4 if dec == 4 else 1.0
        for label in meta["grid"]:
            vals, n = task_level(cells, label, ARMS6)
            note = ""
            if not vals:                      # the reused baseline point
                vals, n = main_run_point(base, ARMS5)
                note = " (main run)"
                if not vals:
                    continue
            row = {a: (vals.get(a) * scale if vals.get(a) is not None else None)
                   for a in ARMS6}
            best = bold_best(row, lower)
            cs = []
            for a in ARMS6:
                s = fmt(row[a], dec)
                cs.append("\\textbf{%s}" % s if a in best and row[a] is not None
                          else s)
            rows.append("\\quad %s%s & %s & %d \\\\"
                        % (label.replace("batch", ""), note, " & ".join(cs), n))
    rows.append("\\bottomrule")
    write("tab_batchsize.tex", rows)


# --------------------------------------------------------------------------
# G.2b -- model-size sweep

SIZE_HOSTS = [
    ("vision_cifar10_modelsize", "vision_cifar10",
     "CIFAR-10 --- test top-1 accuracy (\\%), higher is better", False, 2),
    ("language_wikitext2_modelsize", "language_wikitext2",
     "WikiText-2 --- test perplexity, lower is better", True, 2),
    ("finance_tech_modelsize", "finance_tech",
     "Finance-tech --- test MSE $\\times10^{-4}$, lower is better", True, 4),
]


def table_modelsize():
    rows = ["\\toprule",
            "size & params & AdamW & AdaBelief & Evie-KF & seeds & units \\\\"]
    for sweep, base, header, lower, dec in SIZE_HOSTS:
        rows += ["\\midrule",
                 "\\multicolumn{7}{l}{\\emph{%s}} \\\\" % header]
        path = os.path.join(RES, "sweeps", sweep)
        if os.path.isfile(os.path.join(path, "results_raw.csv")):
            cells = _cells_from_csv(os.path.join(path, "results_raw.csv"))
        else:
            cells, _ = _cells_from_json(
                os.path.join(path, "sweep_results.json"))
        meta = json.load(io.open(os.path.join(path, "sweep_results.json"),
                                 encoding="utf-8"))["meta"]
        raw, _ = _cells_from_json(os.path.join(path, "sweep_results.json"))
        full = json.load(io.open(os.path.join(path, "sweep_results.json"),
                                 encoding="utf-8"))["cells"]
        scale = 1e4 if dec == 4 else 1.0
        grid = [g for g in ("small", "baseline", "large")
                if g in meta["grid"]]
        for label in grid:
            vals, n = task_level(cells, label, ARMS3)
            note = ""
            if not vals:
                vals, n = main_run_point(base, ARMS3)
                note = " (main run)"
                if not vals:
                    continue
            nparam = None
            src = full
            if note:
                bp = os.path.join(RES, base, "results.json")
                src = json.load(io.open(bp, encoding="utf-8"))["cells"]                     if os.path.isfile(bp) else {}
            for k, c in src.items():
                if (note or k.split("|")[0] == label) and c.get("n_params"):
                    nparam = c["n_params"]
                    break
            row = {a: vals[a] * scale for a in ARMS3}
            best = bold_best(row, lower)
            cs = ["\\textbf{%s}" % fmt(row[a], dec) if a in best
                  else fmt(row[a], dec) for a in ARMS3]
            wins = seed_wins(cells if not note else None, label, "EvieKF",
                             "AdamW", lower)
            rows.append("\\quad %s%s & %s & %s & %s & %s \\\\"
                        % (label, note,
                           "---" if nparam is None else "{:,}".format(nparam),
                           " & ".join(cs), wins, n))
    rows.append("\\bottomrule")
    write("tab_modelsize.tex", rows)


def seed_wins(cells, label, a, b, lower):
    if cells is None:
        return "---"
    per = collections.defaultdict(dict)
    for (lab, opt, unit, seed), (v, div) in cells.items():
        if lab == label and opt in (a, b) and not div and v == v:
            per[seed].setdefault(opt, []).append(v)
    w = t = 0
    for seed, d in per.items():
        if a in d and b in d:
            t += 1
            va, vb = st.median(d[a]), st.median(d[b])
            if (va < vb) if lower else (va > vb):
                w += 1
    return "%d/%d" % (w, t) if t else "---"


# --------------------------------------------------------------------------
# G.4 -- cost and telemetry

def table_cost():
    rows = ["\\toprule",
            "& \\multicolumn{2}{c}{Vision} & \\multicolumn{2}{c}{Language}"
            " & \\multicolumn{2}{c}{Finance} \\\\",
            "\\cmidrule(lr){2-3}\\cmidrule(lr){4-5}\\cmidrule(lr){6-7}",
            "Optimizer & time & peak MB & time & peak MB & time & peak MB \\\\",
            "\\midrule"]
    by = collections.defaultdict(lambda: collections.defaultdict(list))
    for r in csv.DictReader(io.open(
            os.path.join(RES, "paper", "table8_compute.csv"), encoding="utf-8")):
        dom = r["Task"].split("/")[0]
        by[r["Optimizer"]][dom].append(
            (float(r["vs AdamW"].rstrip("x")), float(r["peak MB"])))
    order = ["AdamW", "AdaBelief", "SGD", "Sophia", "Muon", "Lion", "Shampoo",
             "EvieKF"]
    for opt in order:
        cs = []
        for dom in ("vis", "lang", "fin"):
            v = by[opt].get(dom)
            if not v:
                cs += ["---", "---"]
                continue
            cs.append("%.2f$\\times$" % st.median(x for x, _ in v))
            cs.append("%d" % st.median(y for _, y in v))
        name = "\\textbf{Evie-KF}" if opt == "EvieKF" else opt
        rows.append("%s & %s \\\\" % (name, " & ".join(cs)))
    rows.append("\\bottomrule")
    write("tab_cost.tex", rows)


def table_telemetry():
    src = os.path.join(RES, "paper", "table10_evie_telemetry.csv")
    rows = ["\\toprule",
            "Task & cells & $\\cos(\\Delta_{\\rm EvieKF},\\Delta_{\\rm AdamW})$"
            " & $B_{\\rm simple}$ & noise eff.\\ rank & min gain \\\\",
            "\\midrule"]
    for r in csv.DictReader(io.open(src, encoding="utf-8")):
        cells = [r["Task"].replace("_", "\\_"), r["cells"]]
        for k in ("cos(update, Adam dir)", "noise/signal b", "noise eff. rank",
                  "min gain"):
            cells.append(tex_num(r[k].split(" [")[0]))
        rows.append(" & ".join(cells) + " \\\\")
    rows.append("\\bottomrule")
    write("tab_telemetry.tex", rows)


# --------------------------------------------------------------------------
# H.3 -- frozen hyperparameters

def tex_num(s):
    if "e+" in s or "e-" in s:
        m, e = s.split("e")
        return r"$%s{\times}10^{%d}$" % (m, int(e))
    return s


def table_hparams():
    order = ["EvieKF", "AdamW", "AdaBelief", "SGD", "Sophia", "Muon", "Lion",
             "Shampoo"]
    by = collections.defaultdict(dict)
    tasks = []
    for r in csv.DictReader(io.open(
            os.path.join(RES, "paper", "table6_tuning_budget.csv"),
            encoding="utf-8")):
        if r["Task"] not in tasks:
            tasks.append(r["Task"])
        by[r["Task"]][r["Optimizer"]] = r
    rows = ["\\toprule",
            "Task & " + " & ".join(o.replace("EvieKF", "Evie-KF")
                                   for o in order) + " & Evie-KF $\\gamma$ \\\\",
            "\\midrule"]
    for t in tasks:
        cs = []
        for o in order:
            r = by[t].get(o)
            cs.append("---" if r is None else sci(float(r["frozen lr"])))
        g = by[t].get("EvieKF", {}).get("frozen knob", "--")
        rows.append("%s & %s & %s \\\\"
                    % (t.replace("_", "\\_"), " & ".join(cs),
                       "---" if g in ("--", "", None)
                       else "{:,}".format(int(float(g)))))
    rows.append("\\bottomrule")
    write("tab_hparams.tex", rows)


def sci(x):
    """Two significant figures in LaTeX, with the mantissa rounded first so
    0.09999 prints as 1e-1 rather than 10.0e-2."""
    import math
    e = int(math.floor(math.log10(x)))
    m = round(x / 10.0 ** e, 1)
    if m >= 10.0:
        m, e = m / 10.0, e + 1
    if abs(m - 1.0) < 0.05:
        return "1\\textsc{e}%d" % e
    return "%.1f\\textsc{e}%d" % (m, e)


# --------------------------------------------------------------------------
# G.1 -- Tiny-ImageNet

def table_tinyimagenet():
    p = os.path.join(RES, "scalegen", "vision_tinyimagenet", "results.json")
    if not os.path.isfile(p):
        print("skip tiny-imagenet: no results.json")
        return
    cells = {}
    for key, c in json.load(io.open(p, encoding="utf-8"))["cells"].items():
        cells[("t", c["opt"], c.get("unit", "full"), int(c["seed"]))] = \
            (c["primary"], bool(c["diverged"]))
    arms = sorted({k[1] for k in cells})
    vals, n = task_level(cells, "t", arms)
    rows = ["\\toprule", "Optimizer & test top-1 (\\%) & seeds vs Evie-KF \\\\",
            "\\midrule"]
    best = bold_best(vals, False)
    for a in sorted(vals, key=lambda x: -vals[x]):
        w = seed_wins(cells, "t", "EvieKF", a, False) if a != "EvieKF" else "---"
        name = "\\textbf{Evie-KF}" if a == "EvieKF" else a
        v = ("\\textbf{%.2f}" % vals[a]) if a in best else "%.2f" % vals[a]
        rows.append("%s & %s & %s \\\\" % (name, v, w))
    rows.append("\\bottomrule")
    write("tab_tinyimagenet.tex", rows)


# --------------------------------------------------------------------------
# G.3 -- the ablation arms and their contrasts

ABL_HOSTS = [
    ("finance_tech", "Finance-tech", "MSE $\times10^{-4}$", True, 1e4, 4),
    ("language_wikitext2", "WikiText-2", "perplexity", True, 1.0, 2),
    ("vision_cifar10", "CIFAR-10", "top-1 (\%)", False, 1.0, 2),
]
ABL_ARMS = [("EvieKF", "Kronecker, absolute, centred"),
            ("EvieKFu", "Kronecker, absolute, uncentred"),
            ("EvieDiagAbs", "diagonal, absolute"),
            ("EvieDiag", "diagonal, mean-normalised")]


def _per_seed():
    out = collections.defaultdict(dict)
    for r in csv.DictReader(io.open(
            os.path.join(RES, "paper", "figdata", "fig5_ablation_per_seed.csv"),
            encoding="utf-8")):
        out[(r["host"], r["arm"])][int(r["seed"])] = float(r["value"])
    return out


def wilcoxon_exact_two_sided(d):
    """Exact two-sided signed-rank p for small n; zeros dropped."""
    d = [x for x in d if x != 0]
    n = len(d)
    if n == 0:
        return 1.0
    order = sorted(range(n), key=lambda i: abs(d[i]))
    rank = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and abs(d[order[j + 1]]) == abs(d[order[i]]):
            j += 1
        r = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            rank[order[k]] = r
        i = j + 1
    w = sum(rank[i] for i in range(n) if d[i] > 0)
    total = sum(rank)
    obs = min(w, total - w)
    cnt = 0
    for mask in range(1 << n):
        s = sum(rank[i] for i in range(n) if mask >> i & 1)
        if min(s, total - s) <= obs + 1e-9:
            cnt += 1
    return min(1.0, cnt / float(1 << n))


def table_ablation_arms():
    per = _per_seed()
    rows = [r"\toprule",
            "Arm & gate & " + " & ".join(h[1] for h in ABL_HOSTS) + r" \\",
            r"\midrule"]
    for arm, gate in ABL_ARMS:
        cs = []
        for host, _t, _m, _lo, scale, dec in ABL_HOSTS:
            v = per.get((host, arm))
            cs.append("---" if not v
                      else ("%%.%df" % dec) % (st.median(v.values()) * scale))
        name = (r"\textbf{Evie-KF}" if arm == "EvieKF"
                else arm.replace("Evie", "Evie-"))
        rows.append(r"%s & %s & %s \\" % (name, gate, " & ".join(cs)))
    rows.append(r"\bottomrule")
    write("tab_ablation_arms.tex", rows)


CONTRASTS = [("EvieDiagAbs", "Kronecker structure alone"),
             ("EvieDiag", "structure and normalisation"),
             ("EvieKFu", "centring")]


def table_ablation_contrasts():
    per = _per_seed()
    raw = []
    rows = [r"\toprule",
            r"Host & contrast & what it isolates & effect & seeds "
            r"& $p$ & Holm $p$ \\"]
    body = []
    for host, title, _m, lower, _s, _d in ABL_HOSTS:
        first = True
        for arm, what in CONTRASTS:
            a, b = per.get((host, "EvieKF")), per.get((host, arm))
            if not a or not b:
                continue
            seeds = sorted(set(a) & set(b))
            rel = [((b[s] - a[s]) / b[s]) if lower else ((a[s] - b[s]) / b[s])
                   for s in seeds]
            wins = sum(1 for x in rel if x > 0)
            ta, tb = st.median(a.values()), st.median(b.values())
            eff = (tb - ta) / tb if lower else (ta - tb) / tb
            p = wilcoxon_exact_two_sided(rel)
            raw.append(p)
            body.append((host, title if first else "", arm, what, eff, wins,
                         len(seeds), p))
            first = False
    # Holm over every contrast reported here
    order = sorted(range(len(raw)), key=lambda i: raw[i])
    m = len(raw)
    holm = [0.0] * m
    run = 0.0
    for k, i in enumerate(order):
        run = max(run, min(1.0, (m - k) * raw[i]))
        holm[i] = run
    for i, (host, title, arm, what, eff, wins, n, p) in enumerate(body):
        rows.append(r"%s & Evie-KF vs Evie-%s & %s & %+.2f\%% & %d/%d & "
                    r"%.4f & %.4f \\"
                    % (title, arm.replace("Evie", ""),
                       what, 100 * eff, wins, n, p, holm[i]))
        if i == 0:
            rows.insert(2, r"\midrule")
    rows.append(r"\bottomrule")
    write("tab_ablation_contrasts.tex", rows)


if __name__ == "__main__":
    table_batchsize()
    table_modelsize()
    table_cost()
    table_telemetry()
    table_hparams()
    table_tinyimagenet()
    table_ablation_arms()
    table_ablation_contrasts()
