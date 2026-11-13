#!/usr/bin/env python3
"""Run R1 sequentially, refresh one consolidated CSV, preserve task failures.

No CUDA jobs are launched until this command is explicitly run by the user.
The task files handle per-cell checkpoints; re-running this runner resumes.
"""
import argparse
import os
from pathlib import Path
import subprocess
import sys
import time

from readout import COLUMNS, write_metrics

TASKS = ["language_wikitext2_muonhybrid", "language_wikitext103_muonhybrid",
         "language_openwebtext_muonhybrid", "language_enwik8_muonhybrid"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", nargs="+", choices=TASKS, default=TASKS)
    parser.add_argument("--output", default="muonhybrid_metrics.csv")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    repo = here.parents[1]
    # Use absolute paths: data/ckpt roots remain stable regardless of cwd.
    env = dict(os.environ)
    ckpt = Path(env.get("BENCH12_CKPT_ROOT", str(repo / "checkpoints"))).resolve()
    env["BENCH12_CKPT_ROOT"] = str(ckpt)
    output = Path(args.output).resolve()
    failures = []
    runs = []
    try:
        for task in args.tasks:
            started = time.monotonic()
            result = subprocess.run([sys.executable, str(here / (task+".py"))],
                                    cwd=repo, env=env, check=False)
            status = "ok" if result.returncode == 0 else "failed"
            runs.append(dict(zip(COLUMNS, ["runner", task, "", "", "", "",
                        "process_exit_code", result.returncode, status,
                        f"process wall seconds={time.monotonic()-started:.3f}; "
                        "see task log; environment errors are not numerical divergence", ""])))
            if result.returncode:
                failures.append(task)
            # Include any task files from the same checkpoint root, so resuming
            # a subset cannot erase completed tasks from the final artifact.
            paths = [ckpt / t / "results.json" for t in TASKS if (ckpt/t/"results.json").exists()]
            write_metrics(paths, output, extra_rows=runs)
            print(f"[runner] refreshed {output}", flush=True)
    except KeyboardInterrupt:
        print("Interrupted; task checkpoints are preserved. Re-run to resume.", file=sys.stderr)
        return 130
    if failures:
        print("Tasks needing attention: " + ", ".join(failures), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
