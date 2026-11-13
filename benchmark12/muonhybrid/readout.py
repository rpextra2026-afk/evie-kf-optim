# >>> GENERATED FILE -- do not edit. Source: benchmark12/_build/*.py + tasks.py
# >>> Rebuild:  python benchmark12/_build/build.py
"""Dependency-light tabular Muon-hybrid readout (stdlib + numpy).

The long-form CSV is the single deliverable. Every raw scalar, diagnostic
trace point, tuning trial, metadata field and registered contrast has a row.
Internal JSON/checkpoints remain necessary for crash recovery and auditing.
"""
import csv
import json
import math
import os
import tempfile
from pathlib import Path

import numpy as np


COLUMNS = ["record_type", "task", "arm", "unit", "seed", "subset", "metric",
           "value", "status", "note", "source"]


def finite(value):
    return isinstance(value, (int, float)) and math.isfinite(value)


def flatten(value, prefix=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from flatten(item, f"{prefix}.{key}" if prefix else str(key))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            yield from flatten(item, f"{prefix}[{index}]")
    else:
        yield prefix, value


def collect_rows(paths):
    rows, pooled = [], {}
    seen_tasks = set()
    expected_tasks = 4
    smoke_modes = set()
    for path in map(Path, paths):
        results = json.loads(path.read_text())
        meta, cells = results.get("meta", {}), results.get("cells", {})
        if meta.get("opt_set") != "muonhybrid":
            raise ValueError(f"{path}: expected muonhybrid results, got {meta.get('opt_set')!r}")
        smoke_modes.add(bool(meta.get("smoke")))
        if len(smoke_modes) > 1:
            raise ValueError("Refusing to pool SMOKE and production runs")
        task = meta.get("task_name") or path.parent.name
        if task in seen_tasks:
            raise ValueError(f"Duplicate task {task}; merge split-arm JSON files before reporting")
        seen_tasks.add(task)
        source = str(path)

        def emit(kind, arm, unit, seed, subset, metric, value, status="ok", note=""):
            if value is None or (isinstance(value, float) and not math.isfinite(value)):
                value, status = "", "unavailable" if status == "ok" else status
            rows.append(dict(zip(COLUMNS, [kind, task, arm, unit, seed, subset,
                                          metric, value, status, note, source])))

        for metric, value in flatten(meta):
            emit("metadata", "", "", "", "", metric, value)
        for key, cell in cells.items():
            parts = key.split("|")
            arm, unit, seed = parts if len(parts) == 3 else (
                cell.get("opt", ""), cell.get("unit", ""), cell.get("seed", ""))
            status = "failed" if cell.get("diverged") else "aliased" if cell.get("aliased_from") else "ok"
            note = cell.get("fail_reason") or cell.get("aliased_from") or ""
            for metric, value in flatten(cell):
                emit("cell", arm, unit, seed, "tuning_seed" if str(seed) == "42" else "heldout_seeds",
                     metric, value, status, note)

        search_path = path.with_name("lr_search.json")
        if search_path.exists():
            for arm, entry in json.loads(search_path.read_text()).items():
                for metric, value in flatten(entry):
                    emit("search", arm, "", 42, "tuning", metric, value,
                         note="validation only; fixed grid; no boundary extension")
                emit("search", arm, "", 42, "tuning", "trials_scored", len(entry.get("grid", {})),
                     note="derived arm reuses four gamma=30 trials" if entry.get("derived_from") else "")
        audit_path = path.with_name("search_audit.json")
        if audit_path.exists():
            audit = json.loads(audit_path.read_text())
            for key, cell in audit.items():
                arm = key.split("|")[0]
                for metric, value in flatten(cell):
                    emit("search_trial", arm, key, 42, "tuning", metric, value,
                         "failed" if cell.get("diverged") else "ok")
            for arm in {k.split("|")[0] for k in audit}:
                elapsed = [c["wall_s"] for k,c in audit.items()
                           if k.split("|")[0] == arm and finite(c.get("wall_s"))]
                emit("search", arm, "", 42, "tuning", "search_training_wall_s", sum(elapsed),
                     note="successful measured trials; excludes setup and unmeasured failed-trial cost")

        for metric, definition in {
            "perplexity": "exp(token-weighted NLL); within-task tokenizer only; capped-vocabulary WikiText is not a literature leaderboard",
            "bits_per_token": "NLL / ln(2); legacy bpc has this meaning for non-byte tokenizers",
            "bits_per_byte": "NLL / ln(2) only on enwik8 byte tokens; unavailable elsewhere without evaluated byte counts; https://arxiv.org/abs/2101.00027",
            "ece_15": "15 equal-width confidence bins; weighted absolute accuracy-confidence gap; https://proceedings.mlr.press/v70/guo17a.html",
            "mce_15": "maximum accuracy-confidence gap among occupied ECE bins; descriptive, bin-sensitive",
            "brier_score": "mean sum_v (p_v - onehot(y)_v)^2; range [0,2]; multiclass proper score; https://doi.org/10.1198/016214506000001437",
            "mean_target_rank": "1 + number of logits strictly exceeding gold; optimistic ties",
            "mean_reciprocal_rank": "mean reciprocal target rank across teacher-forced tokens; descriptive",
            "mean_log_ratio_ci95": "10000 paired seed-bootstrap samples, RNG 20261109; no CI for fewer than 2 pairs; very few seeds implies coarse uncertainty",
            "tuning": "12 vs 12 trials (OWT 4 vs 4); fixed gamma reuses 4 trials; no boundary extension; https://arxiv.org/abs/1910.05446",
            "time_to_reference_val": "first observed validation checkpoint meeting paired MuonSub best validation; no interpolation; unreached is blank+censored; post-hoc descriptive; https://mlcommons.org/benchmarks/algorithms/",
            "gen_gap_loss": "best-val-selected test loss minus final-step training loss; different checkpoints/batches, descriptive only",
            "cost": "wall_s is training+validation, excludes final detailed test diagnostics; alias cost is copied, not a second measured run",
            "hessian": "unavailable for hybrid arms: existing Hessian telemetry is Sophia-only; missing is not zero",
        }.items():
            emit("definition", "", "", "", "", metric, definition)
        arms = meta.get("opt_names") or sorted({k.split("|")[0] for k in cells})
        seeds = meta.get("seeds", sorted({int(k.split("|")[-1]) for k in cells}))
        units = meta.get("units", sorted({k.split("|")[1] for k in cells}))
        ref = "MuonSub"
        for subset, selected in [("all_seeds", seeds), ("heldout_seeds", [s for s in seeds if s != 42]),
                                 ("tuning_seed", [s for s in seeds if s == 42])]:
            by_arm = {}
            for arm in arms:
                expected = [cells.get(f"{arm}|{u}|{s}") for u in units for s in selected]
                healthy = [c for c in expected if c and not c.get("diverged")
                           and finite(c.get("primary")) and c["primary"] > 0]
                by_arm[arm] = {s: float(np.median([c["primary"] for u in units
                               if (c := cells.get(f"{arm}|{u}|{s}")) and not c.get("diverged")
                               and finite(c.get("primary")) and c["primary"] > 0]))
                               for s in selected if all(
                                   (c := cells.get(f"{arm}|{u}|{s}")) and not c.get("diverged")
                                   and finite(c.get("primary")) and c["primary"] > 0 for u in units)}
                emit("summary", arm, "", "", subset, "expected_cells", len(expected))
                emit("summary", arm, "", "", subset, "missing_cells", sum(c is None for c in expected))
                emit("summary", arm, "", "", subset, "divergences", sum(bool(c and c.get("diverged")) for c in expected))
                emit("summary", arm, "", "", subset, "aliased_cells", sum(bool(c and c.get("aliased_from")) for c in expected))
                emit("summary", arm, "", "", subset, "healthy_cells", len(healthy))
                # All observed-arm statistics are marked so failures cannot disappear
                # behind a favourable complete-case number.
                numeric = {k for c in healthy for k,v in c.items() if finite(v) and not isinstance(v, bool)}
                for metric in sorted(numeric):
                    values = np.array([c[metric] for c in healthy if finite(c.get(metric))])
                    for stat, value in [("mean", values.mean()), ("median", np.median(values)),
                                        ("std", values.std(ddof=1) if len(values)>1 else float("nan")),
                                        ("min", values.min()), ("max", values.max()),
                                        ("q25", np.percentile(values,25)), ("q75", np.percentile(values,75))]:
                        emit("summary", arm, "", "", subset, f"{metric}.{stat}", float(value),
                             note=f"observed healthy cells only; n={len(values)}; inspect missing/divergences")
            baseline = by_arm.get(ref, {})
            for arm in arms:
                if arm == ref:
                    continue
                paired = sorted(set(baseline) & set(by_arm[arm]))
                n_expected = len(selected)
                note = ("SMOKE correctness data; " if meta.get("smoke") else "") + f"vs {ref}; paired seeds={paired}; expected={n_expected}; losses/ties retained"
                emit("paired", arm, "", "", subset, "n_paired_seeds", len(paired), note=note)
                emit("paired", arm, "", "", subset, "n_unpaired_seeds", n_expected-len(paired), note=note)
                if not paired:
                    emit("paired", arm, "", "", subset, "mean_log_ratio", None, note=note)
                    continue
                a = np.array([by_arm[arm][s] for s in paired])
                b = np.array([baseline[s] for s in paired])
                ratio = a/b
                logs = np.log(ratio)
                med_a = float(np.median(list(by_arm[arm].values())))
                med_b = float(np.median(list(baseline.values())))
                stats = dict(median_test_perplexity=float(np.median(a)),
                             median_reference_perplexity=float(np.median(b)),
                             median_seed_ratio=float(np.median(ratio)),
                             median_seed_ratio_pct=100*(float(np.median(ratio))-1),
                             ratio_of_arm_medians_pct=100*(med_a/med_b-1),
                             mean_log_ratio=float(logs.mean()),
                             geometric_mean_ratio=float(np.exp(logs.mean())),
                             seeds_won=int((a<b).sum()), seeds_tied=int((a==b).sum()),
                             seeds_lost=int((a>b).sum()),
                             mean_paired_perplexity_difference=float((a-b).mean()))
                if len(paired) >= 2:
                    rng = np.random.default_rng(20261109)
                    boot = logs[rng.integers(0,len(logs),size=(10000,len(logs)))].mean(1)
                    lo, hi = np.quantile(boot,[.025,.975])
                    stats.update(mean_log_ratio_ci95_low=float(lo), mean_log_ratio_ci95_high=float(hi),
                                 geometric_ratio_ci95_low=float(np.exp(lo)), geometric_ratio_ci95_high=float(np.exp(hi)))
                else:
                    stats.update(mean_log_ratio_ci95_low=None, mean_log_ratio_ci95_high=None)
                # Cost ratios use the same seed+unit pairing; aliases are excluded:
                # their copied wall time is a counterfactual, not measured compute.
                for metric in ["wall_s", "sec_per_step", "peak_mem_mb", "tokens_per_s", "test_eval_wall_s"]:
                    ratios = []
                    for seed in paired:
                        for unit in units:
                            ca, cb = cells[f"{arm}|{unit}|{seed}"], cells[f"{ref}|{unit}|{seed}"]
                            if (not ca.get("aliased_from") and finite(ca.get(metric))
                                    and finite(cb.get(metric)) and cb[metric] > 0):
                                ratios.append(ca[metric]/cb[metric])
                    stats[metric+".median_ratio"] = float(np.median(ratios)) if ratios else None
                for seed in paired:
                    for unit in units:
                        ca, cb = cells[f"{arm}|{unit}|{seed}"], cells[f"{ref}|{unit}|{seed}"]
                        target = cb.get("best_val_primary")
                        if finite(target):
                            for which, cell in [(arm, ca), (ref, cb)]:
                                hits = [(step,val) for step,val in cell.get("val_curve", [])
                                        if finite(val) and val <= target]
                                step = hits[0][0] if hits else None
                                wall = dict(cell.get("eval_wall_curve", [])).get(step)
                                emit("time_to_target", which, unit, seed, subset,
                                     "reference_best_val_target", target, note="post-hoc baseline validation target")
                                for metric, value in [("time_to_reference_val_steps", step),
                                                      ("time_to_reference_val_wall_s", wall)]:
                                    emit("time_to_target", which, unit, seed, subset, metric, value,
                                         "aliased" if cell.get("aliased_from") else "ok" if hits else "censored",
                                         note="observed evaluation grid only; never use test scores for selection")
                for metric, value in stats.items():
                    emit("paired", arm, "", "", subset, metric, value, note=note)
                if subset == "heldout_seeds":
                    pooled.setdefault(arm, []).append((task, float(logs.mean()), len(paired) == n_expected))
    for arm, values in pooled.items():
        complete = len(values) == expected_tasks and all(v[2] for v in values)
        rows.append(dict(zip(COLUMNS, ["pooled", "ALL", arm, "", "", "heldout_seeds",
                    "equal_task_mean_log_ratio", float(np.mean([v[1] for v in values])),
                    "ok" if complete else "incomplete", "equal task weights; tasks="+
                    ",".join(v[0] for v in values)+"; descriptive only, no p-values", ""])))
    return rows


def write_metrics(paths, output, extra_rows=()):
    rows = collect_rows(paths) + list(extra_rows)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=".metrics-", suffix=".csv", dir=output.parent)
    try:
        with os.fdopen(fd, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=COLUMNS)
            writer.writeheader()
            writer.writerows(rows)
            f.flush()
            os.fsync(f.fileno())
        os.replace(temp, output)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)
    return rows


def readout_main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", help="results.json from one or more hybrid tasks")
    parser.add_argument("--output", default="muonhybrid_metrics.csv")
    args = parser.parse_args()
    rows = write_metrics(args.results, args.output)
    for row in rows:
        if row["record_type"] in ("paired", "pooled") and row["metric"] in (
                "median_seed_ratio_pct", "mean_log_ratio_ci95_low", "mean_log_ratio_ci95_high",
                "seeds_won", "wall_s.median_ratio", "equal_task_mean_log_ratio"):
            print(" | ".join(str(row[k]) for k in ["task","arm","subset","metric","value","status"]))
    print(f"Wrote {len(rows)} metric rows to {args.output}")

if __name__ == "__main__":
    readout_main()
