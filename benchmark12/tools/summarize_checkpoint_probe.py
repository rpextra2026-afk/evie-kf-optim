"""Summarize the WikiText-2 checkpoint diagnostics from their JSON artifacts."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def summarize(directory):
    rows = []
    for path in sorted(directory.glob("probe_17101_*.json")):
        item = json.loads(path.read_bytes())
        if item["smoke"] or item["seed"] != 17101:
            raise ValueError(f"Unexpected run: {path}")
        risk = item["risk"]
        values = np.asarray(risk["paired_variances"])
        if values.shape != (24, 2) or item["independent_probe_batches"] != 64:
            raise ValueError(f"Unexpected probe counts: {path}")
        ratio = values[:, 0].mean() / values[:, 1].mean()
        if not np.isclose(ratio, risk["variance_ratio_Evie_over_ungated"], rtol=1e-12):
            raise ValueError(f"Inconsistent risk summary: {path}")
        control = item["coupling"]["crossfit_whitened_noise"]
        mean = control["signed_shift_along_frozen_Adam_direction"]
        rows.append(dict(
            saved_step=item["step_saved"], next_update=item["next_update"], phase=item["phase"],
            variance_ratio=float(ratio), ci95_low=risk["ci95"][0], ci95_high=risk["ci95"][1],
            variance_reduction_percent=100 * (1 - ratio),
            rms_step_change_percent=100 * control["rms_update_shift_over_ungated_rms"],
            projected_mean_change_percent=100 * mean["mean"],
            projected_ci95_low_percent=100 * mean["ci95"][0],
            projected_ci95_high_percent=100 * mean["ci95"][1],
            clipping_rate=item["clip_rate"], gamma_zero_error=item["gamma_zero_relative_error"],
            probe_seconds=item["elapsed_seconds"]))
    if [r["saved_step"] for r in rows] != [1999, 2000, 4999, 5000, 7999, 8000]:
        raise ValueError("Expected all six planned checkpoints")
    with (directory / "summary.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0], lineterminator="
")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"Update {row['next_update']}: variance ratio {row['variance_ratio']:.3f} "
              f"[{row['ci95_low']:.3f}, {row['ci95_high']:.3f}], "
              f"projected mean shift {row['projected_mean_change_percent']:.6f}%")
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    summarize(parser.parse_args().directory)
