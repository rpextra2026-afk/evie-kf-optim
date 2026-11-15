"""Read a merged Financials follow-up sweep using its frozen usability rule."""
import argparse
import json
import math
from pathlib import Path
from statistics import median

BUDGET_SECONDS = 3 * 3600


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, help="Merged results.json from both notebooks")
    args = parser.parse_args()
    data = json.loads(args.results.read_text())
    meta = data["meta"]
    if meta["task_name"] not in {"finance_financials_2025", "finance_financials_original_split"} or meta.get("smoke"):
        parser.error("expected production Financials follow-up results")
    cells = data["cells"]
    expected = set(meta["expected_cell_keys"])
    if set(cells) != expected or any(not c.get("recorded") for c in cells.values()):
        parser.error("merge the complete outputs from both notebooks before reporting")

    print("| Optimizer | Completed | Median test MSE (completed) | Training GPU-hours | "
          "Attempts in budget | Usable in budget |")
    print("| --- | ---: | ---: | ---: | ---: | ---: |")
    for arm in meta["opt_names"]:
        runs = sorted((c for c in cells.values() if c["opt"] == arm),
                      key=lambda c: (c["seed"], c["unit"]))
        complete = [c for c in runs if not c["diverged"]]
        elapsed, attempts, usable = 0.0, 0, 0
        for cell in runs:
            duration = cell.get("wall_s")
            if duration is None or not math.isfinite(duration) or duration <= 0:
                parser.error(f"missing runtime for {arm}/{cell['unit']}/{cell['seed']}")
            elapsed += duration
            if not cell["diverged"]:
                ratio = cell.get("mse_vs_zero")
                if ratio is None or not math.isfinite(ratio):
                    parser.error(f"missing zero-return comparison for {arm}/{cell['unit']}/{cell['seed']}")
            if elapsed <= BUDGET_SECONDS:
                attempts += 1
                if not cell["diverged"] and ratio <= 1.0:
                    usable += 1
        mse = f"{median(c['primary'] for c in complete):.6g}" if complete else "—"
        print(f"| {arm} | {len(complete)}/{len(runs)} | {mse} | "
              f"{elapsed / 3600:.2f} | {attempts} | {usable} |")


if __name__ == "__main__":
    main()
