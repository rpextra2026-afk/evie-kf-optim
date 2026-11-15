"""Compare benchmark optimizers on a matrix quadratic with rotated noise."""

import argparse
import csv
import hashlib
import json
import math
import time
from pathlib import Path
from statistics import median

import numpy as np
import torch

from pilot_optimizers import EvieKF, Muon


ARMS = ("AdamW", "Muon", "Evie-KF", "Evie diagonal", "Evie random basis")


def rotated_factor(spectrum, fraction):
    """Rotate opposite ends of the spectrum through disjoint planes."""
    dim = len(spectrum)
    basis = np.eye(dim)
    angle = fraction * math.pi / 4
    c, s = math.cos(angle), math.sin(angle)
    for i in range(dim // 2):
        pair = [i, dim - 1 - i]
        basis[np.ix_(pair, pair)] = [[c, -s], [s, c]]
    return basis * np.sqrt(spectrum)


def covariance_inputs(data, fraction):
    left = rotated_factor(data["row_eigenvalues"], fraction)
    right = rotated_factor(data["column_eigenvalues"], fraction)
    covariance = np.kron(left @ left.T, right @ right.T)
    expected = np.sort(np.outer(data["row_eigenvalues"], data["column_eigenvalues"]).ravel())
    np.testing.assert_allclose(np.linalg.eigvalsh(covariance), expected, rtol=1e-9, atol=1e-11)
    off_diagonal = covariance - np.diag(np.diag(covariance))
    return left, right, float(np.linalg.norm(off_diagonal) / np.linalg.norm(covariance))


def problem(data, fraction, seed, steps, device):
    left, right, _ = covariance_inputs(data, fraction)
    rng = np.random.Generator(np.random.PCG64(seed))
    shape = (left.shape[0], right.shape[0])
    initial = rng.standard_normal(shape)
    normals = rng.standard_normal((steps, 2, *shape))
    # Each half has twice the target covariance, so their average has it.
    halves = math.sqrt(2) * (left @ normals @ right.T)
    curvature = np.asarray(data["curvature"])
    return tuple(torch.as_tensor(x, device=device, dtype=torch.float32)
                 for x in (initial, halves[:, 0], halves[:, 1], curvature))


def optimizer(arm, param, lr, gamma):
    if arm == "AdamW":
        return torch.optim.AdamW([param], lr=lr, weight_decay=0)
    if arm == "Muon":
        return Muon([dict(params=[param], lr=lr, use_muon=True, weight_decay=0)])
    return EvieKF([param], lr=lr, gamma=gamma, weight_decay=0, kron_mean_norm=True,
                  diag_only=arm == "Evie diagonal", random_basis=arm == "Evie random basis")


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


@torch.no_grad()
def rollout(arm, inputs, lr, gamma, every, timed):
    initial, noise_a, noise_b, curvature = inputs
    weight = torch.nn.Parameter(initial.clone())
    opt = optimizer(arm, weight, lr, gamma)
    evie = isinstance(opt, EvieKF)
    loss = lambda: float((curvature * weight.square()).sum() * 0.5)
    trace = [[0, 0.0, loss()]]
    elapsed = 0.0
    sync(weight.device)
    start = time.perf_counter()
    steps = len(noise_a)
    for step in range(1, steps + 1):
        true_grad = curvature * weight
        a, b = true_grad + noise_a[step - 1], true_grad + noise_b[step - 1]
        weight.grad = (a + b) * 0.5
        if evie:
            opt.set_noise([(weight, (a - b) * 0.5)])
        opt.step()
        if step == steps or (timed and step % every == 0):
            sync(weight.device)
            elapsed += time.perf_counter() - start
            value = loss()
            if not math.isfinite(value):
                return None
            trace.append([step, elapsed, value])
            start = time.perf_counter()
    if evie and steps > opt.warmup and opt.n_active_steps == 0:
        raise RuntimeError(f"{arm} never activated the gate")
    return trace


def run(data, out, device, smoke=False):
    full_steps = data["steps"]
    steps = 60 if smoke else full_steps
    tune_seeds = data["tune_seeds"][:1] if smoke else data["tune_seeds"]
    eval_seeds = data["evaluation_seeds"][:2] if smoke else data["evaluation_seeds"]
    fractions = data["rotations"]
    gamma = data["gamma"]
    tuning, trials, summary = [], [], []
    assert not set(tune_seeds) & set(eval_seeds)
    out.mkdir(parents=True, exist_ok=True)

    for fraction in fractions:
        _, _, off_diagonal = covariance_inputs(data, fraction)
        lr_by_arm = {}
        tune_inputs = [problem(data, fraction, seed, steps, device) for seed in tune_seeds]
        for arm in ARMS:
            grid = data["lr_grids"][arm]
            if smoke:
                grid = grid[2:5]
            scores = []
            for lr in grid:
                curves = [rollout(arm, inputs, lr, gamma, data["eval_every"], False)
                          for inputs in tune_inputs]
                score = median(curve[-1][2] for curve in curves) if all(curves) else None
                scores.append({"lr": lr, "median_final_loss": score})
            valid = [row for row in scores if row["median_final_loss"] is not None]
            if not valid:
                raise RuntimeError(f"No finite tuning trial for {arm} at rotation {fraction}")
            pick = min(valid, key=lambda row: row["median_final_loss"])["lr"]
            lr_by_arm[arm] = pick
            tuning.append(dict(rotation=fraction, optimizer=arm, selected_lr=pick,
                               edge_pick=pick in (grid[0], grid[-1]), trials=scores))
            print(f"rotation={fraction:g} {arm}: lr={pick:.4g}", flush=True)

        regime_trials = []
        for seed in eval_seeds:
            inputs = problem(data, fraction, seed, steps, device)
            order = np.random.Generator(np.random.PCG64(seed)).permutation(ARMS)
            curves = {arm: rollout(arm, inputs, lr_by_arm[arm], gamma, data["eval_every"], True)
                      for arm in order.tolist()}
            # Shortest completed run defines the common observed time horizon.
            budget = min((curve[-1][1] for curve in curves.values() if curve), default=None)
            for arm in ARMS:
                curve = curves[arm]
                at_budget = next((p for p in reversed(curve) if p[1] <= budget), None) if curve else None
                regime_trials.append(dict(rotation=fraction, off_diagonal=off_diagonal,
                    seed=seed, optimizer=arm, lr=lr_by_arm[arm], completed=bool(curve),
                    final_loss=curve[-1][2] if curve else None,
                    elapsed_s=curve[-1][1] if curve else None, common_time_s=budget,
                    loss_at_common_time=at_budget[2] if at_budget else None,
                    steps_at_common_time=at_budget[0] if at_budget else None, curve=curve))
        trials.extend(regime_trials)
        for arm in ARMS:
            rows = [r for r in regime_trials if r["optimizer"] == arm]
            muon = {r["seed"]: r for r in regime_trials if r["optimizer"] == "Muon"}
            result = dict(rotation=fraction, off_diagonal=round(off_diagonal, 4), optimizer=arm,
                          completed=sum(r["completed"] for r in rows), total=len(rows))
            for metric in ("final_loss", "loss_at_common_time"):
                values = [r[metric] for r in rows if r[metric] is not None]
                ratios = [r[metric] / muon[r["seed"]][metric] for r in rows
                          if r[metric] is not None and muon[r["seed"]][metric]]
                result[f"median_{metric}"] = median(values) if values else None
                result[f"{metric}_paired_ratio_to_muon"] = median(ratios) if ratios else None
            summary.append(result)
        # Save each finished regime for inspection if the notebook is interrupted.
        metadata = dict(device=str(device), torch_version=str(torch.__version__),
                        gpu=torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                        steps=steps, smoke=smoke)
        (out / "results.json").write_text(json.dumps(dict(
            metadata=metadata, design=data, tuning=tuning, trials=trials, summary=summary),
            indent=2, allow_nan=False) + "\n", encoding="utf-8")
        with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(summary[0]))
            writer.writeheader()
            writer.writerows(summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--sha256")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    raw = args.data.read_bytes()
    if args.sha256 and hashlib.sha256(raw).hexdigest() != args.sha256:
        parser.error("pilot dataset digest mismatch")
    torch.set_num_threads(1)
    torch.set_float32_matmul_precision("highest")
    for row in run(json.loads(raw), args.out, torch.device(args.device), args.smoke):
        print(row)


if __name__ == "__main__":
    main()
