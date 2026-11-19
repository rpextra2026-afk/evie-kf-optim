"""Run the WikiText-2 checkpoint coupling and variance diagnostic."""

import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import torch


def write_json(path, payload):
    path = Path(path)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_task(config, output, smoke):
    output.mkdir(parents=True, exist_ok=True)
    os.environ["BENCH12_CKPT_ROOT"] = str(output.resolve())
    os.environ["BENCH12_DATA_ROOT"] = str(Path("task_data").resolve())
    os.environ["BENCH12_TORCH_THREADS"] = "2"
    os.environ["BENCH12_TF32"] = "0"
    os.environ["SMOKE"] = "1" if smoke else "0"
    os.environ["SMOKE_STEPS"] = "3"
    source = Path("tasks") / config["source"]
    if hashlib.sha256(source.read_bytes()).hexdigest() != config["source_sha256"]:
        raise ValueError("Task source does not match the plan")
    spec = importlib.util.spec_from_file_location("frozen_benchmark_task", source)
    task = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(task)
    if not smoke and (not torch.cuda.is_available() or "T4" not in torch.cuda.get_device_name()):
        raise RuntimeError("Select Kaggle GPU T4 before running.")
    task.preflight()
    if smoke:
        task.TASK.update(d_model=24, n_layer=1, n_head=4, ctx=16)
    return task


def train(task, data, host, seed, config, output, arms=None):
    cells = []
    for unit in task.main_units(data):
        for arm in arms or (config["arm"],):
            checkpoint = Path(task.CELL_DIR) / f"{arm}_{unit}_{seed}.pt"
            result = task.train_run(
                arm, data, unit, config["lr"][arm], config["lr"]["AdamW"],
                task.MAX_STEPS, seed, str(checkpoint), knob=config["gamma"])
            primary = float(result["primary"])
            cells.append(dict(
                unit=unit, arm=arm, primary=primary if np.isfinite(primary) else None,
                steps=result["steps_run"], wall_s=result["wall_s"],
                failed=bool(result["diverged"] or task._absurd_primary(primary)
                            or result["steps_run"] != task.MAX_STEPS),
                fail_reason=result.get("fail_reason")))
    return cells


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=17101)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    plan = json.loads(Path("plan.json").read_text(encoding="utf-8"))
    if args.seed not in plan["seeds"]:
        parser.error("Seed must match the published diagnostic")
    output = Path("measurement_output") / ("smoke" if args.smoke else "probe") / str(args.seed) / "language_wikitext2"
    config = plan["probe"]
    task = load_task(config, output, args.smoke)
    data = task.prepare_data()
    from checkpoint_probe import run
    run(task, data, args.seed, config, output, args.smoke)


if __name__ == "__main__":
    main()
