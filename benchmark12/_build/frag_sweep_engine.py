# ==============================================================================
#  SWEEP DRIVER  --  batch-size OR model-size sweep over one host task
#  (CLAUDE.md sec 9). GENERATED only into benchmark12/sweeps/<type>/<task>.py.
#  Reuses the shared optimizer block + the engine helpers above UNCHANGED and
#  adds only this outer grid loop. Entry point: sweep_main().
#
#  Key rules (CLAUDE.md sec 9.2 / 9.3):
#   - fixed step budget per grid point (MAX_STEPS, not samples-seen-matched);
#     absolute metric values are therefore NOT comparable across grid points --
#     only the WITHIN-grid-point delta between arms is fair. Logged below.
#   - LR search re-run per (grid point, arm); the reused-baseline point instead
#     reads the main run's frozen LR + its already-computed [42,43,44] cells
#     from checkpoints/<BASE_TASK_NAME>/.
#   - seeds [42,43,44] (first 3 of the main run's 5) so the baseline point IS
#     the main run's own cell.
#   - Evie-KF arms (EvieKF in _BATCH_ARMS/_MODEL_ARMS; EvieKFu in _BATCH_ARMS
#     for the C1 test) run at a FIXED gamma = EVIEKF_SWEEP_GAMMA at the new
#     grid points (the sweeps re-search LR only, not gamma -- see _sweep_knob).
#     The reused baseline point keeps whatever gamma the MAIN run's joint
#     search froze for EvieKF (EvieKFu has no baseline cell -- the main table
#     never runs it). So EvieKF's baseline column may sit at a different gamma
#     than its new-point columns; that is inside the existing "absolute values
#     not comparable across grid points" caveat, and the C1 spearman
#     (EvieKFu vs EvieKF) is computed over the new points only, all at the same
#     gamma, so it is unaffected.
# ==============================================================================
from scipy.stats import spearmanr

SWEEP_TYPE     = TASK["sweep_type"]            # "batchsize" | "modelsize"
BASE_TASK_NAME = TASK["base_task_name"]
SWEEP_GRID     = TASK["sweep_grid"]
SWEEP_ARMS_CFG = TASK["sweep_arms"]
BASE_BATCH     = int(TASK["batch_size"])
SWEEP_SEEDS    = [42] if SMOKE else list(TASK.get("seeds", [42, 43, 44]))

SWEEP_LR_JSON = os.path.join(TASK_DIR, "sweep_lr.json")
SWEEP_RESULTS = os.path.join(TASK_DIR, "sweep_results.json")
SWEEP_SUMMARY = os.path.join(TASK_DIR, "summary.json")
SWEEP_CSV     = os.path.join(TASK_DIR, "results_raw.csv")


def _sweep_arms():
    # Accept any arm build_optimizer() can construct -- the main-table 8 PLUS
    # the ablation-ladder names (EvieKFu / EvieDiag). _BATCH_ARMS carries
    # "EvieKFu" for the C1 test even though it is not in OPT_NAMES.
    _buildable = set(OPT_NAMES) | set(ABLATION_OPT_NAMES)
    arms = [o for o in SWEEP_ARMS_CFG if o in _buildable]
    dropped = [o for o in SWEEP_ARMS_CFG if o not in _buildable]
    if dropped:
        log(f"[sweep] WARNING: sweep arms {dropped} are not buildable "
            f"(not in OPT_NAMES or ABLATION_OPT_NAMES) -- skipped")
    # NOTE: the sweeps re-search LR per (grid point, arm) but do NOT joint-search
    # the second axis. Evie-KF-family arms are pinned to EVIEKF_SWEEP_GAMMA
    # (=3000, the PoC v3 optimum -- see _sweep_knob); Shampoo (not currently a
    # sweep arm) would use its build_optimizer() default beta.
    return arms


def _sweep_knob(arm):
    """Fixed second-axis value the sweep pins `arm` to (LR is still re-searched
    at this value). Evie-KF family -> EVIEKF_SWEEP_GAMMA; everything else ->
    None, i.e. build_optimizer()'s own default."""
    if arm in ("EvieKF", "EvieKFu", "EvieDiag", "EvieKFn"):
        return EVIEKF_SWEEP_GAMMA
    return None


def _grid_points():
    """[(label, cfg), ...] with the reused-baseline point LAST."""
    g = SWEEP_GRID
    if SWEEP_TYPE == "batchsize":
        pts = [(f"batch{int(b)}", {"batch": int(b)}) for b in g["new"]]
        pts.append((f"batch{int(g['baseline'])}",
                    {"batch": int(g["baseline"]), "baseline": True}))
    else:
        pts = [(k, {"model_size": v}) for k, v in g["points"].items()]
        pts.append(("baseline", {"model_size": g["baseline_spec"], "baseline": True}))
    if SMOKE:
        pts = pts[:2] + [p for p in pts if p[1].get("baseline")]
    return pts


def _grid_x(label, cfg):
    """numeric x-axis for a grid point: batch size, or a param-count proxy."""
    if SWEEP_TYPE == "batchsize":
        return float(cfg["batch"])
    ms = cfg["model_size"]
    if isinstance(ms, dict):                       # language: L * d * heads
        return float(ms["n_layer"] * ms["d_model"] * ms["n_head"])
    return float(ms[0] * ms[-1])                   # vision widths / finance dims


def _configure(cfg, data):
    global BATCH_SIZE
    data.pop("model_size", None)
    BATCH_SIZE = int(cfg["batch"]) if "batch" in cfg else BASE_BATCH
    if "model_size" in cfg:
        data["model_size"] = cfg["model_size"]


def _baseline_from_main(data, arms):
    base_dir = os.path.join(CKPT_ROOT, BASE_TASK_NAME)
    lr_j = _read_json(os.path.join(base_dir, "lr_search.json"), {})
    res_j = _read_json(os.path.join(base_dir, "results.json"), {"cells": {}})
    got_lr, got_cells, missing = {}, {}, []
    for arm in arms:
        e = lr_j.get(arm)
        if isinstance(e, dict) and "lr" in e:
            got_lr[arm] = float(e["lr"])
        else:
            missing.append(f"lr[{arm}]")
    for arm in arms:
        for unit in main_units(data):
            for seed in SWEEP_SEEDS:
                c = (res_j.get("cells", {}) or {}).get(f"{arm}|{unit}|{seed}")
                if c:
                    got_cells[f"{arm}|{unit}|{seed}"] = c
                else:
                    missing.append(f"cell[{arm}|{unit}|{seed}]")
    if missing:
        log(f"[sweep] baseline point: {len(missing)} pieces NOT found under "
            f"checkpoints/{BASE_TASK_NAME}/ -- run that main task first for a "
            f"complete baseline column. Missing e.g. {missing[:6]}"
            + (" ..." if len(missing) > 6 else ""))
    return got_lr, got_cells


def _grid_lr(label, arm, data, adamw_lr, cache):
    key = f"{label}|{arm}"
    e = cache.get(key)
    if isinstance(e, dict) and "lr" in e:
        return float(e["lr"])
    knob = _sweep_knob(arm)
    best, tried, bchecks = search_lr_for(arm, data, lr_search_units(data),
                                         adamw_lr, log_prefix=f"[{label}] ",
                                         knob=knob)
    cache[key] = {"lr": best, "knob": knob, "boundary_checks": bchecks,
                  "grid": {f"{k:.6e}": tried[k] for k in sorted(tried)}}
    _write_json(SWEEP_LR_JSON, cache)
    log(f"[{label}] [lr-search] {arm}: FROZEN lr={best:.3e}"
        + ("" if knob is None else f"  (fixed knob={knob:g})")
        + f"  (boundary checks: {bchecks})")
    return best


def sweep_main():
    log_sep("=")
    log(f"  {TASK_NAME}  |  {SWEEP_TYPE} sweep on {BASE_TASK_NAME}  "
        f"device={DEVICE}  torch={torch.__version__}")
    log(f"  NOTE: fixed {MAX_STEPS}-step budget at every grid point -- absolute "
        f"{PRIMARY_METRIC} is NOT comparable across grid points; only the "
        f"within-grid-point arm delta is (CLAUDE.md sec 9.2).")
    if TASK.get("nonstandard_ppl_caveat"):
        log(f"  [CAVEAT] {TASK['nonstandard_ppl_caveat']}")
    log_sep("=")
    preflight()
    data = prepare_data()
    arms = _sweep_arms()
    points = _grid_points()
    ordered = (["AdamW"] if "AdamW" in arms else []) + \
              [a for a in arms if a != "AdamW"]
    log(f"  arms={arms}  seeds={SWEEP_SEEDS}  grid={[p[0] for p in points]}")

    lr_cache = _read_json(SWEEP_LR_JSON, {})
    results = _read_json(SWEEP_RESULTS, {"cells": {}, "meta": {}})
    cells = results["cells"]

    for label, cfg in points:
        _configure(cfg, data)

        if cfg.get("baseline"):
            log_sep("-")
            log(f"[{label}] reusing the main run's frozen LR + "
                f"seed {SWEEP_SEEDS} cells from checkpoints/{BASE_TASK_NAME}/")
            got_lr, got_cells = _baseline_from_main(data, arms)
            for arm, v in got_lr.items():
                lr_cache[f"{label}|{arm}"] = {"lr": v, "reused_from_main": True}
            for ck, c in got_cells.items():
                arm, unit, seed = ck.split("|")
                cells[f"{label}|{arm}|{unit}|{seed}"] = dict(
                    c, reused_from_main=True, recorded=True, label=label)
            _write_json(SWEEP_LR_JSON, lr_cache)
            _write_json(SWEEP_RESULTS, results)
            continue

        log_sep("-")
        log(f"[{label}] LR search  ({SWEEP_TYPE} = "
            f"{cfg.get('batch', cfg.get('model_size'))})")
        frozen = {}
        for arm in ordered:
            frozen[arm] = _grid_lr(label, arm, data, frozen.get("AdamW"), lr_cache)

        log(f"[{label}] sweep cells")
        for arm in ordered:
            for unit in main_units(data):
                for seed in SWEEP_SEEDS:
                    key = f"{label}|{arm}|{unit}|{seed}"
                    if key in cells and cells[key].get("recorded"):
                        continue
                    if torch.cuda.is_available():
                        torch.cuda.reset_peak_memory_stats()
                    ckpt = os.path.join(
                        CELL_DIR, f"{_safe(label)}__{_safe(arm)}__{_safe(unit)}"
                                  f"__seed{seed}.pt")
                    try:
                        r = train_run(arm, data, unit, frozen[arm],
                                      frozen.get("AdamW"), MAX_STEPS, seed, ckpt,
                                      is_lr_search=False, knob=_sweep_knob(arm))
                        r["recorded"] = True
                        r["label"] = label
                        if _absurd_primary(r["primary"]):
                            r["diverged"] = True
                            r["fail_reason"] = (r.get("fail_reason")
                                                or "primary outside absolute range")
                        cells[key] = r
                    except SystemExit:
                        raise
                    except Exception as e:
                        if classify_exception(e) == "env":
                            log(f"[sweep] ENVIRONMENT error in {key}: {e!r}")
                            log(traceback.format_exc())
                            fail(f"environment/CUDA error at {key}: {e!r}. This is "
                                 f"NOT a diverged run -- fix the environment; "
                                 f"finished cells are safe, re-run to resume.")
                        log(f"[sweep] numerical failure in {key}: {e!r} -> cell "
                            f"marked failed, continuing")
                        log(traceback.format_exc())
                        cells[key] = {"label": label, "opt": arm, "unit": unit,
                                      "seed": seed, "lr_used": frozen[arm],
                                      "knob_used": _sweep_knob(arm),
                                      "primary": float("nan"), "diverged": True,
                                      "fail_reason": f"exception: {e!r}",
                                      "recorded": True}
                    _write_json(SWEEP_RESULTS, results)
                    log(f"[{label}] {key}  primary={cells[key].get('primary')}  "
                        f"diverged={cells[key].get('diverged')}")

        # peer-relative divergence WITHIN this grid point only (scale differs
        # across grid points by design -- CLAUDE.md sec 9.2)
        healthy = [c["primary"] for k, c in cells.items()
                   if k.startswith(label + "|") and not c.get("diverged")
                   and _is_finite(c.get("primary"))]
        pm = float(np.median(healthy)) if healthy else None
        if pm is not None:
            for k, c in cells.items():
                if (k.startswith(label + "|") and not c.get("diverged")
                        and _absurd_primary(c.get("primary"), pm)):
                    c["diverged"] = True
                    c["fail_reason"] = (c.get("fail_reason")
                                        or f"{c['primary']:.4g} is "
                                           f">{PEER_DIVERGENCE_FACTOR}x off the "
                                           f"grid-point peer median {pm:.4g}")
                    log(f"[diverge] {k}: {c['fail_reason']}")
        _write_json(SWEEP_RESULTS, results)

    results["meta"] = {
        "task_name": TASK_NAME, "sweep_type": SWEEP_TYPE,
        "base_task_name": BASE_TASK_NAME, "arms": arms, "seeds": SWEEP_SEEDS,
        "primary_metric": PRIMARY_METRIC, "max_steps": MAX_STEPS,
        "grid": [p[0] for p in points], "torch": torch.__version__,
        "nonstandard_ppl_caveat": TASK.get("nonstandard_ppl_caveat"),
        "timestamp": datetime.now().isoformat(),
    }
    _write_json(SWEEP_RESULTS, results)
    _sweep_aggregate(results, points, arms, data)


def _sweep_number(cells, label, arm, units):
    per_seed = []
    for seed in SWEEP_SEEDS:
        vs = [cells[f"{label}|{arm}|{u}|{seed}"]["primary"] for u in units
              if f"{label}|{arm}|{u}|{seed}" in cells
              and not cells[f"{label}|{arm}|{u}|{seed}"].get("diverged")
              and _is_finite(cells[f"{label}|{arm}|{u}|{seed}"].get("primary"))]
        if vs:
            per_seed.append(float(np.mean(vs)))
    return (float(np.median(per_seed)) if per_seed else float("nan")), per_seed


def _sweep_write_csv(cells, labels, arms, units):
    import csv
    cols = ["label", "opt", "unit", "seed", "lr_used", "knob_used", "primary",
            "best_val_primary", "diverged", "fail_reason", "reused_from_main",
            "steps_run", "wall_s", "sec_per_step", "mean_step_size", "clip_freq",
            "peak_mem_mb",
            "steps_to_90pct_best", "frac_budget_to_90pct_best",
            "final_train_loss", "gen_gap_loss", "auc_val_curve",
            "update_weight_ratio", "update_cos_sim_sampled",
            "grad_norm_mean", "grad_norm_p95", "weight_norm_growth",
            "throughput_samples_per_s", "flops_per_step_est",
            "hess_trace", "hess_top_eig",
            "eviekf_cos_mean", "eviekf_cos_min", "eviekf_b_simple_mean",
            "eviekf_eff_rank_mean", "eviekf_gain_lo_min", "eviekf_gain_hi_max",
            "eviekf_n_active_steps"] + list(AUX_METRIC_KEYS)
    with open(SWEEP_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for label in labels:
            for arm in arms:
                for unit in units:
                    for seed in SWEEP_SEEDS:
                        c = cells.get(f"{label}|{arm}|{unit}|{seed}")
                        if c:
                            w.writerow(dict(c, label=label, opt=arm, unit=unit,
                                            seed=seed))
    log(f"  wrote {SWEEP_CSV}")


def _sweep_aggregate(results, points, arms, data):
    cells = results["cells"]
    units = main_units(data)
    labels = [p[0] for p in points]
    is_base = {p[0]: bool(p[1].get("baseline")) for p in points}
    xvals = {p[0]: _grid_x(*p) for p in points}
    # batch-size sweep: reference is EvieKF, so `delta vs ref` for the EvieKFu
    # arm IS the pre-registered C1 quantity -- does the centred-vs-uncentred gap
    # widen with batch size (CLAUDE.md sec 9.2). model-size sweep: reference
    # stays AdamW (a fixed external baseline is the right question there, not a
    # KF-vs-KF contrast).
    if SWEEP_TYPE == "batchsize" and "EvieKF" in arms:
        ref = "EvieKF"
    else:
        ref = "AdamW"

    table, per_seed = {}, {}
    for label in labels:
        table[label], per_seed[label] = {}, {}
        for arm in arms:
            n, ps = _sweep_number(cells, label, arm, units)
            table[label][arm] = n
            per_seed[label][arm] = ps

    log_sep("=")
    log(f"  {SWEEP_TYPE.upper()} SWEEP -- {TASK_NAME}   primary={PRIMARY_METRIC} "
        f"({'lower' if PRIMARY_LOWER_BETTER else 'higher'} is better)")
    if TASK.get("nonstandard_ppl_caveat"):
        log(f"  [CAVEAT] {TASK['nonstandard_ppl_caveat']}")
    log_sep("-")
    log(f"  {'grid point':<16}{'x':>11}  " + "".join(f"{a:>13}" for a in arms))
    for label in labels:
        row = f"  {label:<16}{xvals[label]:>11.4g}  "
        for arm in arms:
            v = table[label][arm]
            row += f"{v:>13.5g}" if _is_finite(v) else f"{'nan':>13}"
        log(row + ("   [reused baseline]" if is_base[label] else ""))

    log_sep("-")
    log(f"  delta vs {ref}  (d>0 => arm beats {ref} at that grid point; "
        f"direction-normalised relative)")
    deltas = {a: {} for a in arms if a != ref}
    for label in labels:
        rv = table[label].get(ref, float("nan"))
        bits = []
        for arm in deltas:
            av = table[label][arm]
            if _is_finite(rv) and rv != 0 and _is_finite(av):
                d = ((rv - av) / abs(rv) if PRIMARY_LOWER_BETTER
                     else (av - rv) / abs(rv))
            else:
                d = float("nan")
            deltas[arm][label] = d
            bits.append(f"{arm}:{d:+.3f}")
        log(f"    {label:<16} " + "  ".join(bits))

    spearman = {}
    if SWEEP_TYPE == "batchsize":
        log_sep("-")
        log(f"  pre-registered quantity (CLAUDE.md sec 9.2): "
            f"Spearman(batch size, delta vs {ref}) per arm.")
        if ref == "EvieKF":
            log(f"  -> the EvieKFu row is the C1 test: rho < 0 (with d = "
                f"normalised EvieKF-minus-EvieKFu) means the centred edge grows "
                f"as batch size shrinks.")
        else:
            log(f"  NOTE: reference is {ref} (EvieKF not among the sweep arms) -- "
                f"this trends each arm's edge over {ref}, not the C1 gap.")
        for arm in deltas:
            xy = [(xvals[l], deltas[arm][l]) for l in labels
                  if _is_finite(deltas[arm][l])]
            if len(xy) >= 3:
                rho, p = spearmanr([a for a, _ in xy], [b for _, b in xy])
                spearman[arm] = {"rho": float(rho), "p": float(p), "n": len(xy)}
                log(f"    {arm:<12} rho={rho:+.3f}  p={p:.3g}  n={len(xy)}")
            else:
                spearman[arm] = None
                log(f"    {arm:<12} too few finite points")

    _sweep_write_csv(cells, labels, arms, units)
    summary = {
        "task_name": TASK_NAME, "sweep_type": SWEEP_TYPE,
        "base_task_name": BASE_TASK_NAME, "primary_metric": PRIMARY_METRIC,
        "primary_lower_better": PRIMARY_LOWER_BETTER, "reference": ref,
        "grid_labels": labels, "grid_x": xvals, "grid_is_baseline": is_base,
        "arms": arms, "seeds": SWEEP_SEEDS,
        "eviekf_sweep_gamma": (EVIEKF_SWEEP_GAMMA
                               if any(a in ("EvieKF", "EvieKFu", "EvieDiag")
                                      for a in arms) else None),
        "c1_row": ("EvieKFu" if ref == "EvieKF" and "EvieKFu" in arms else None),
        "table": table, "per_seed": per_seed, "deltas": deltas,
        "spearman_batch_vs_delta": spearman,
        "nonstandard_ppl_caveat": TASK.get("nonstandard_ppl_caveat"),
        "note": ("fixed step budget per grid point; absolute values not "
                 "comparable across grid points, only within-grid-point arm "
                 "deltas (CLAUDE.md sec 9.2/9.3)"),
    }
    _write_json(SWEEP_SUMMARY, summary)
    log_sep("=")
    log(f"  wrote {SWEEP_SUMMARY}  and  {SWEEP_RESULTS}")
    log(f"  DONE {datetime.now().isoformat()}")
    _log_fh.flush()
