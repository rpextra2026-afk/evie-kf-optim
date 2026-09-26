"""Export tidy CSVs for the paper's figures.

Reads only committed run artefacts under ``benchmark12/results`` and writes
``benchmark12/results/paper/figdata/*.csv``. Nothing here computes a result: it
reshapes what the runs already recorded, so a re-run of the sweep changes the
figures automatically instead of silently disagreeing with hardcoded numbers.

No number in this file is typed by hand.

    python benchmark12/export_figure_data.py
"""

import csv
import json
import os
import statistics

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")
OUT = os.path.join(RESULTS, "paper", "figdata")

# Fig 5: the ablation ladder. Host -> (folder, extra folder for the deconfound arm).
FIG5_HOSTS = [
    ("finance_tech", "ablation_finance_tech", "ablation_finance_tech_diag_absolute"),
    ("language_wikitext2", "ablation_language_wikitext2",
     "ablation_language_wikitext2_diag_absolute"),
    ("vision_cifar10", "ablation_vision_cifar10", None),
]

# Fig 6: the (lr, gamma) surfaces. One panel per entry, in figure order:
# the 12 benchmark tasks by domain, then the two scale-generalization tasks.
# The folder is relative to results/; the scalegen tasks live under scalegen/,
# not beside the 12.
FIG6_TASKS = [
    ("vision_cifar10", "vision_cifar10"),
    ("vision_cifar100", "vision_cifar100"),
    ("vision_svhn", "vision_svhn"),
    ("vision_stl10", "vision_stl10"),
    ("language_wikitext2", "language_wikitext2"),
    ("language_wikitext103", "language_wikitext103"),
    ("language_ptb", "language_ptb"),
    ("language_enwik8", "language_enwik8"),
    ("finance_tech", "finance_tech"),
    ("finance_finance", "finance_finance"),
    ("finance_healthcare", "finance_healthcare"),
    ("finance_consumer_industrials", "finance_consumer_industrials"),
    ("vision_tinyimagenet", os.path.join("scalegen", "vision_tinyimagenet")),
    ("language_openwebtext", os.path.join("scalegen", "language_openwebtext")),
]


def _task_meta(task_name):
    """Metric name and direction from the generator's own task table.

    Used when a task has no ``summary.json`` -- openwebtext was stopped before
    any final cell ran, so only its search survives. This reads the definition
    rather than restating it here, so the figure cannot drift from the run.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_tasks", os.path.join(HERE, "_build", "tasks.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for task in module.ALL_TASKS:
        if task.get("task_name") == task_name:
            return task.get("primary_metric", ""), bool(task.get("primary_lower_better"))
    return "", False


def _load(path):
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _read_raw(folder):
    """(arm, unit, seed) -> primary, for non-diverged cells only."""
    path = os.path.join(RESULTS, "ablation", folder, "results_raw.csv")
    if not os.path.exists(path):
        return None
    out = {}
    with open(path, encoding="utf-8", newline="") as fh:
        for r in csv.DictReader(fh):
            if str(r.get("diverged", "")).strip().lower() == "true":
                continue
            try:
                value = float(r["primary"])
            except (TypeError, ValueError):
                continue
            out[(r["opt"], r["unit"], int(r["seed"]))] = value
    return out


def export_fig5():
    """Per-seed value for every ablation arm on every host.

    Recomputed from ``results_raw.csv`` rather than read from ``summary.json``,
    because the deconfound arm lives in a second run folder whose complete-case
    unit set differs from the original's. Taking each folder's own summary would
    compare arms over different sets of tickers. Units are intersected across
    every arm of every folder for the host first, then reduced exactly as the
    main protocol does: median over units per seed.
    """
    rows = []
    for host, folder, extra in FIG5_HOSTS:
        cells = {}
        metric, lower = "", False
        for src in [folder, extra]:
            if src is None:
                continue
            raw = _read_raw(src)
            if raw is None:
                print("  skip (not present): %s" % src)
                continue
            summary = _load(os.path.join(RESULTS, "ablation", src, "summary.json")) or {}
            metric = metric or summary.get("primary_metric", "")
            lower = lower or bool(summary.get("primary_lower_better"))
            for (arm, unit, seed), value in raw.items():
                cells.setdefault((arm, unit, seed), value)
        if not cells:
            continue

        arms = sorted({a for a, _, _ in cells})
        seeds = sorted({s for _, _, s in cells})
        # complete case: a unit must be finished by every arm at every seed
        units = sorted({u for _, u, _ in cells})
        units = [u for u in units
                 if all((a, u, s) in cells for a in arms for s in seeds)]
        if not units:
            continue
        print("  fig5 %-20s arms=%d seeds=%d complete-case units=%d"
              % (host, len(arms), len(seeds), len(units)))

        for arm in arms:
            for seed in seeds:
                vals = [cells[(arm, u, seed)] for u in units]
                rows.append({
                    "host": host,
                    "arm": arm,
                    "seed": seed,
                    "value": statistics.median(vals),
                    "n_units": len(units),
                    "metric": metric,
                    "lower_is_better": int(lower),
                })
    _write("fig5_ablation_per_seed.csv",
           ["host", "arm", "seed", "value", "n_units", "metric",
            "lower_is_better"], rows)

    # A companion file with the paired contrasts, so the figure can annotate
    # without recomputing them.
    summary_rows = []
    for host, _, _ in FIG5_HOSTS:
        by_arm = {}
        for r in rows:
            if r["host"] == host:
                by_arm.setdefault(r["arm"], {})[r["seed"]] = r["value"]
        if "EvieKF" not in by_arm:
            continue
        lower = next(r["lower_is_better"] for r in rows if r["host"] == host)
        for other in by_arm:
            if other == "EvieKF":
                continue
            shared = sorted(set(by_arm["EvieKF"]) & set(by_arm[other]))
            if not shared:
                continue
            deltas = []
            for s in shared:
                a, b = by_arm["EvieKF"][s], by_arm[other][s]
                d = (b - a) / abs(b) if lower else (a - b) / abs(b)
                deltas.append(d)
            # Two different statistics, both reported so a figure annotation can
            # match the text. The paper quotes the task-level one (the protocol's
            # own reduction: median over seeds first, then compare).
            ka = statistics.median(by_arm["EvieKF"][s] for s in shared)
            kb = statistics.median(by_arm[other][s] for s in shared)
            task_level = (kb - ka) / abs(kb) if lower else (ka - kb) / abs(kb)
            summary_rows.append({
                "host": host,
                "baseline_arm": other,
                "n_seeds": len(shared),
                "eviekf_wins": sum(1 for d in deltas if d > 0),
                "task_level_delta": task_level,
                "median_per_seed_delta": statistics.median(deltas),
            })
    _write("fig5_contrasts.csv",
           ["host", "baseline_arm", "n_seeds", "eviekf_wins",
            "task_level_delta", "median_per_seed_delta"],
           summary_rows)


def export_fig6():
    """The scored joint (lr, gamma) search surface for EvieKF, per task."""
    rows = []
    for panel, folder in FIG6_TASKS:
        search = _load(os.path.join(RESULTS, folder, "lr_search.json"))
        summary = _load(os.path.join(RESULTS, folder, "summary.json"))
        if search is None or "EvieKF" not in search:
            print("  skip (not present): %s" % folder)
            continue
        grid = search["EvieKF"].get("grid", {})
        if summary:
            metric = summary.get("primary_metric", "")
            lower = bool(summary.get("primary_lower_better"))
        else:
            metric, lower = _task_meta(panel)

        points = []
        for key, score in grid.items():
            if "|" not in key:
                continue
            lr_s, knob_s = key.split("|")
            points.append((float(lr_s), float(knob_s), score))

        chosen_lr = (summary or {}).get("lr_used", {}).get("EvieKF")
        chosen_knob = (summary or {}).get("knob_used", {}).get("EvieKF")
        if chosen_lr is None or chosen_knob is None:
            # No summary: recover the frozen point the way the engine chose it,
            # as the best scored point on the grid.
            best = (min if lower else max)(points, key=lambda p: p[2])
            chosen_lr, chosen_knob = best[0], best[1]

        for lr, knob, score in points:
            rows.append({
                "panel": panel,
                "lr": lr,
                "gamma": knob,
                "search_score": score,
                "metric": metric,
                "lower_is_better": int(lower),
                "is_frozen": int(abs(lr - float(chosen_lr)) < 1e-12
                                 and abs(knob - float(chosen_knob)) < 1e-9),
            })
    _write("fig6_gamma_surface.csv",
           ["panel", "lr", "gamma", "search_score", "metric",
            "lower_is_better", "is_frozen"], rows)


def _write(name, fields, rows):
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, name)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %s (%d rows)" % (os.path.relpath(path, HERE), len(rows)))


if __name__ == "__main__":
    export_fig5()
    export_fig6()
