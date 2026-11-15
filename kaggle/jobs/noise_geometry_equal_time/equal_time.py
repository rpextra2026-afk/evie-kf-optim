"""Run the synthetic optimizer comparison at a fixed CPU-time deadline."""

import argparse
import csv
import hashlib
import json
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch

from pilot import ARMS, covariance_inputs, optimizer, problem, rollout
from pilot_optimizers import EvieKF


@torch.no_grad()
def at_deadline(arm, inputs, lr, gamma, budget_s):
    initial, noise_a, noise_b, curvature = inputs
    weight = torch.nn.Parameter(initial.clone())
    opt = optimizer(arm, weight, lr, gamma)
    evie = isinstance(opt, EvieKF)
    steps = 0
    start = time.perf_counter()
    elapsed = 0.0
    for step, (a_noise, b_noise) in enumerate(zip(noise_a, noise_b), 1):
        grad = curvature * weight
        a, b = grad + a_noise, grad + b_noise
        weight.grad = (a + b) * 0.5
        if evie:
            opt.set_noise([(weight, (a - b) * 0.5)])
        opt.step()
        elapsed = time.perf_counter() - start
        steps = step
        if elapsed >= budget_s:
            break
    if elapsed < budget_s:
        raise RuntimeError("Noise stream ended before the deadline")
    value = float((curvature * weight.square()).sum() * 0.5)
    if evie and steps > opt.warmup and not opt.n_active_steps:
        raise RuntimeError(f"{arm} did not activate the gate")
    return dict(steps=steps, elapsed_s=elapsed,
                loss=value if np.isfinite(value) else None)


def summarize(rows):
    muon = {r["seed"]: r["loss"] for r in rows if r["optimizer"] == "Muon"}
    for arm in ARMS:
        group = [r for r in rows if r["optimizer"] == arm]
        valid = [r for r in group if r["loss"] is not None]
        pairs = [(r["loss"], muon[r["seed"]]) for r in valid
                 if muon[r["seed"]] is not None]
        ratios = [a / b for a, b in pairs if b > 0]
        yield dict(
            rotation=group[0]["rotation"], off_diagonal=group[0]["off_diagonal"],
            optimizer=arm, completed=len(valid), total=len(group),
            median_loss=median(r["loss"] for r in valid) if valid else None,
            median_steps=median(r["steps"] for r in group),
            median_elapsed_s=median(r["elapsed_s"] for r in group),
            paired_runs=len(pairs), wins_vs_muon=sum(a < b for a, b in pairs),
            median_paired_ratio_to_muon=median(ratios) if ratios else None,
        )


def run(design, out, smoke):
    device = torch.device("cpu")
    budget = 0.025 if smoke else design["budget_s"]
    tune_seeds = design["tune_seeds"][:1] if smoke else design["tune_seeds"]
    evaluation_seeds = design["evaluation_seeds"][:2] if smoke else design["evaluation_seeds"]
    grid = design["lr_grid"][3:6] if smoke else design["lr_grid"]
    assert not set(tune_seeds) & set(evaluation_seeds)
    trials, tuning, summary = [], [], []
    out.mkdir(parents=True, exist_ok=True)
    for fraction in design["rotations"]:
        _, _, off_diagonal = covariance_inputs(design, fraction)
        warmup = problem(design, fraction, 10, 45, device)
        for arm in ARMS:
            rollout(arm, warmup, 0.01, design["gamma"], 20, False)
        scores = {arm: {lr: [] for lr in grid} for arm in ARMS}
        for seed in tune_seeds:
            inputs = problem(design, fraction, seed, design["max_noise_steps"], device)
            cases = [(arm, lr) for arm in ARMS for lr in grid]
            for idx in np.random.default_rng(seed).permutation(len(cases)):
                arm, lr = cases[idx]
                scores[arm][lr].append(
                    at_deadline(arm, inputs, lr, design["gamma"], budget))
        selected = {}
        for arm, by_lr in scores.items():
            medians = {lr: median(r["loss"] for r in runs)
                       for lr, runs in by_lr.items() if all(r["loss"] is not None for r in runs)}
            if not medians:
                raise RuntimeError(f"No finite tuning trial for {arm} at rotation {fraction}")
            pick = min(medians, key=medians.get)
            selected[arm] = pick
            tuning.append(dict(rotation=fraction, optimizer=arm, selected_lr=pick,
                               edge_pick=pick in (grid[0], grid[-1]), trials=by_lr))
        rows = []
        for seed in evaluation_seeds:
            inputs = problem(design, fraction, seed, design["max_noise_steps"], device)
            for arm in np.random.default_rng(seed).permutation(ARMS).tolist():
                result = at_deadline(arm, inputs, selected[arm], design["gamma"], budget)
                rows.append(dict(rotation=fraction, off_diagonal=off_diagonal,
                                 seed=seed, optimizer=arm, lr=selected[arm], **result))
        trials.extend(rows)
        summary.extend(summarize(rows))
        result = dict(metadata=dict(smoke=smoke, budget_s=budget, torch=str(torch.__version__),
                                    completed_regimes=len(summary) // len(ARMS)),
                      design=design, tuning=tuning, trials=trials, summary=summary)
        (out / "results.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
        with (out / "summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
        print(f"Completed rotation {fraction:g}: {selected}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    raw = args.data.read_bytes()
    if hashlib.sha256(raw).hexdigest() != args.sha256:
        parser.error("dataset digest mismatch")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    run(json.loads(raw), args.out, args.smoke)


if __name__ == "__main__":
    main()
