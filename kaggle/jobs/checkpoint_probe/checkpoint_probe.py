"""Replay the production Evie-KF step from fixed training checkpoints."""

import copy
import math
from pathlib import Path
import time

import numpy as np
from scipy import stats
import torch

from run_followup import train, write_json


def capture(task, output, steps, evie_arm):
    build = task.build_optimizer

    def build_with_capture(arm, model, *args, **kwargs):
        optimizer = build(arm, model, *args, **kwargs)
        if arm != evie_arm:
            return optimizer
        step = optimizer.step

        def step_and_capture(*args, **kwargs):
            answer = step(*args, **kwargs)
            completed = optimizer._global_step() - 1
            if completed in steps:
                path = output / f"state_{completed}.pt"
                temp = path.with_suffix(".tmp")
                torch.save(dict(model=model.state_dict(), optimizer=optimizer.state_dict()), temp)
                temp.replace(path)
            return answer

        optimizer.step = step_and_capture
        return optimizer

    task.build_optimizer = build_with_capture


def batch_grad(task, model, params, dataset, generator, split):
    indices = torch.randint(len(dataset), (task.BATCH_SIZE,), generator=generator)
    rows = [dataset[int(i)] for i in indices]
    x, y = task.batch_to_device(tuple(torch.stack(items) for items in zip(*rows)))
    criterion = task.make_criterion()
    if not split:
        return torch.autograd.grad(task.forward_loss(model, (x, y), criterion), params)
    n = x.shape[0] // 2
    a = torch.autograd.grad(task.forward_loss(model, (x[:n], y[:n]), criterion), params)
    b = torch.autograd.grad(task.forward_loss(model, (x[n:], y[n:]), criterion), params)
    for p, u, v in zip(params, a, b):
        p.grad = (u + v) * 0.5
    norm = float(torch.nn.utils.clip_grad_norm_(params, task.GRAD_CLIP_NORM))
    if not math.isfinite(norm):
        raise FloatingPointError("nonfinite gradient in diagnostic")
    coef = min(1.0, task.GRAD_CLIP_NORM / (norm + 1e-6))
    return [p.grad for p in params], [(u - v) * (0.5 * coef) for u, v in zip(a, b)], norm


@torch.no_grad()
def whitening(params, optimizer, grads=None):
    result = []
    position = 0
    for group in optimizer.param_groups:
        beta2, eps = group["beta2"], group["eps"]
        for p in group["params"]:
            assert p is params[position]
            state = optimizer.state[p]
            v, step = state["v"], state["step"]
            if grads is not None:
                g = grads[position]
                v = v.mul(beta2).addcmul_(g, g, value=1 - beta2)
                step += 1
            result.append((v / (1 - beta2 ** step)).sqrt_().add_(eps).rsqrt_())
            position += 1
    return result


def flatten(tensors):
    return torch.cat([tensor.reshape(-1) for tensor in tensors])


@torch.no_grad()
def direction(params, originals, optimizer, state, grads, delta, gamma):
    optimizer.load_state_dict(copy.deepcopy(state))
    optimizer.gamma = gamma
    for p, grad in zip(params, grads):
        p.grad = grad
        # EvieKF.step uses parameter values only to apply decay and add the step.
        # A zero base removes decay and avoids subtracting two close FP32 weights.
        p.zero_()
    optimizer.set_noise(zip(params, delta))
    optimizer.step()
    value = flatten([p.detach().neg() for p in params])
    for p, original in zip(params, originals):
        p.copy_(original)
    return value


def mean_ci(values):
    values = np.asarray(values, dtype=float)
    avg = float(values.mean())
    half = stats.t.ppf(0.975, len(values) - 1) * values.std(ddof=1) / len(values) ** 0.5
    return dict(mean=avg, ci95=[float(avg - half), float(avg + half)])


def channels_summary(sums, squares, along, baseline_sq, cross_products):
    n = len(along)
    # Independent batch pairs: the cross-products estimate ||E[d]||^2.
    # This unbiased point estimate can be negative from sampling error.
    return dict(signed_shift_along_frozen_Adam_direction=mean_ci(along),
                rms_update_shift_over_ungated_rms=float((squares / baseline_sq) ** 0.5),
                independent_pair_mean_shift_squared_over_ungated_rms_squared=
                mean_ci(np.asarray(cross_products) / (baseline_sq / n)) if len(cross_products) > 1 else None,
                mean_shift_squared_over_ungated_rms_squared=
                float((sums.square().sum().item() - squares) / (n - 1) / baseline_sq))


def risk_summary(projections, seed):
    # A common set of independent probe batches is reused for every direction,
    # so resample both rows (directions) and columns (probe batches).
    def ratio(values):
        variances = values.var(axis=1, ddof=1).mean(axis=0)
        return variances[0] / variances[1] if variances[1] > 0 else np.nan

    rng = np.random.default_rng(seed)
    n, m = projections.shape[:2]
    estimates = []
    for _ in range(2000):
        i, j = rng.integers(n, size=n), rng.integers(m, size=m)
        estimates.append(ratio(projections[i[:, None], j[None, :], :]))
    point = ratio(projections)
    finite_estimates = [value for value in estimates if np.isfinite(value)]
    interval = (np.quantile(finite_estimates, [0.025, 0.975]).tolist()
                if np.isfinite(point) and len(finite_estimates) >= 0.99 * len(estimates)
                else None)
    return dict(variance_ratio_Evie_over_ungated=float(point) if np.isfinite(point) else None,
                ci95=interval, valid_bootstrap_draws=len(finite_estimates),
                paired_variances=projections.var(axis=1, ddof=1).tolist())


def probe(task, data, seed, config, path, step, count, probes, smoke):
    start = time.monotonic()
    state = torch.load(path, map_location=task.DEVICE, weights_only=True)
    model = task.build_model(data, "full").to(task.DEVICE)
    model.load_state_dict(state["model"])
    model.train()
    params = list(p for p in model.parameters() if p.requires_grad)
    originals = [p.detach().clone() for p in params]
    optimizer = task.build_optimizer(config["arm"], model, config["lr"][config["arm"]],
                                     adamw_lr=config["lr"]["AdamW"], knob=config["gamma"],
                                     eviekf_maxf=task.TASK.get("eviekf_maxf"))
    if smoke:
        optimizer.warmup = 0
        optimizer.refresh = 2
    # The optimizer state stores the LR used at the saved step. A common
    # one-step scalar LR is sufficient for the relative contrasts and directions.
    optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
    fixed_pm12 = whitening(params, optimizer)
    frozen_reference = flatten([
        d * optimizer.state[p]["m"] for p, d in zip(params, fixed_pm12)
    ])
    frozen_reference /= frozen_reference.norm().clamp_min(1e-30)
    inv_scale = flatten([1 / d for d in fixed_pm12])
    loader, _, _ = task.make_loaders(data, "full", task.BATCH_SIZE, seed)
    dataset = loader.dataset
    task.set_seed(seed * 97 + step)
    source = torch.Generator().manual_seed(seed * 101 + step)
    independent = torch.Generator().manual_seed(seed * 103 + step)
    probe_rng = torch.Generator().manual_seed(seed * 107 + step)
    conditions = ("swap_clipped_delta", "crossfit_whitened_noise")
    sums = {key: torch.zeros_like(inv_scale) for key in conditions}
    squares = dict.fromkeys(conditions, 0.0)
    signed = {key: [] for key in conditions}
    previous = {}
    cross_products = {key: [] for key in conditions}
    baseline_sq = 0.0
    norm_events = []
    directions = torch.empty((2 * count, inv_scale.numel()), device=task.DEVICE)
    algebra_error = 0.0

    for i in range(count):
        grads, delta, norm = batch_grad(task, model, params, dataset, source, split=True)
        donor_grads, donor_delta, _ = batch_grad(task, model, params, dataset, independent, split=True)
        optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
        recipient_w = whitening(params, optimizer, grads)
        donor_w = whitening(params, optimizer, donor_grads)
        donor_input = [d * z / r for d, z, r in zip(donor_w, donor_delta, recipient_w)]
        same = direction(params, originals, optimizer, state["optimizer"], grads, delta, config["gamma"])
        base = direction(params, originals, optimizer, state["optimizer"], grads, delta, 0.0)
        if i == 0:
            # Check the gamma-zero replay against the algebraic AdamW step.
            optimizer.load_state_dict(copy.deepcopy(state["optimizer"]))
            expected = []
            for group in optimizer.param_groups:
                for p in group["params"]:
                    st = optimizer.state[p]
                    k = len(expected)
                    beta1 = group["beta1"]
                    m = st["m"].mul(beta1).add_(grads[k], alpha=1 - beta1)
                    mh = m / (1 - beta1 ** (st["step"] + 1))
                    d = recipient_w[k]
                    expected.append((d * (d * mh)) * group["lr"])
            expected = flatten(expected)
            algebra_error = float((base - expected).norm() / expected.norm().clamp_min(1e-30))
            if algebra_error > 1e-5:
                raise RuntimeError(f"Gamma-zero replay disagrees with AdamW algebra: {algebra_error}")
        basen = float((base * inv_scale).norm())
        baseline_sq += basen ** 2
        norm_events.append(norm > task.GRAD_CLIP_NORM)
        for key, noise in zip(conditions, (donor_delta, donor_input)):
            alternative = direction(params, originals, optimizer, state["optimizer"], grads, noise, config["gamma"])
            diff = (same - alternative) * inv_scale
            if i % 2 == 0:
                previous[key] = diff
            else:
                cross_products[key].append(float(diff.dot(previous.pop(key))))
            sums[key] += diff
            squares[key] += float(diff.square().sum())
            signed[key].append(float(diff.dot(frozen_reference)) / max(basen, 1e-30))
        for j, step_direction in enumerate((same, base)):
            directions[2 * i + j] = step_direction / (step_direction * inv_scale).norm().clamp_min(1e-30)

    projections = np.empty((count, probes, 2))
    for j in range(probes):
        gradient = flatten(batch_grad(task, model, params, dataset, probe_rng, split=False))
        projections[:, j, :] = (directions @ gradient).reshape(count, 2).cpu().numpy()
    return dict(step_saved=step, next_update=step + 1,
                phase="eigensystem_refresh" if (step + 1) % optimizer.refresh == 0 else "cached_eigensystem",
                direction_pairs=count, independent_probe_batches=probes,
                clip_rate=float(np.mean(norm_events)), gamma_zero_relative_error=algebra_error,
                coupling={key: channels_summary(sums[key], squares[key], signed[key], baseline_sq, cross_products[key])
                          for key in conditions},
                risk=risk_summary(projections, seed + step),
                elapsed_seconds=time.monotonic() - start)


def run(task, data, seed, config, output, smoke):
    steps = [1, 2] if smoke else [1999, 2000, 4999, 5000, 7999, 8000]
    original_build = task.build_optimizer
    if smoke:
        def smoke_optimizer(*args, **kwargs):
            optimizer = original_build(*args, **kwargs)
            if isinstance(optimizer, task.EvieKF):
                optimizer.warmup = 0
                optimizer.refresh = 2
            return optimizer
        task.build_optimizer = smoke_optimizer
    capture(task, output, steps, config["arm"])
    cells = train(task, data, "language_wikitext2", seed, config, output, arms=(config["arm"],))
    task.build_optimizer = original_build
    write_json(output / f"probe_training_{seed}.json", cells)
    if cells[0]["failed"]:
        raise RuntimeError("Evie-KF training failed; see saved outcome")
    for step in steps:
        path = output / f"state_{step}.pt"
        result = probe(task, data, seed, config, path, step, 4 if smoke else 24, 6 if smoke else 64, smoke)
        result.update(seed=seed, smoke=smoke)
        write_json(output / f"probe_{seed}_{step}.json", result)
        point = result["risk"]["variance_ratio_Evie_over_ungated"]
        risk = f"{point:.4f}" if point is not None else "undefined"
        print(f"[probe] step {step}: {result['phase']} risk ratio {risk}; "
              f"{result['elapsed_seconds']:.1f}s", flush=True)
