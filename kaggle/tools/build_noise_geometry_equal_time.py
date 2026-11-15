"""Package the fixed-deadline follow-up to the noise-geometry pilot."""

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PILOT = ROOT / "kaggle/jobs/noise_geometry_pilot"
JOB = ROOT / "kaggle/jobs/noise_geometry_equal_time"


def main():
    baseline = json.loads((PILOT / "noise_geometry_data.json").read_bytes())
    design = {key: baseline[key] for key in
              ("row_eigenvalues", "column_eigenvalues", "curvature", "rotations", "gamma")}
    design.update(
        budget_s=0.45, max_noise_steps=24000, tune_seeds=[300, 301, 302],
        evaluation_seeds=list(range(400, 412)),
        lr_grid=[float(f"{10 ** (k / 2):.7g}") for k in range(-8, 1)],
    )
    raw = (json.dumps(design, indent=2) + "\n").encode()
    digest = hashlib.sha256(raw).hexdigest()
    dataset = "noise_geometry_equal_time_data.json"
    (JOB / dataset).write_bytes(raw)
    optimizer_source = (
        "import math\nimport numpy as np\nimport torch\nimport torch.nn as nn\n"
        "WEIGHT_DECAY = 0.0\nGRAD_CLIP_NORM = 5.0\nHESS_FREQ = 10\nN_HUTCH = 1\n"
        "log = print\n\n"
        + (ROOT / "benchmark12/_build/frag_shared_optim.py").read_text(encoding="utf-8")
    )
    files = {
        "pilot_optimizers.py": optimizer_source.encode(),
        "pilot.py": (PILOT / "pilot.py").read_bytes(),
        "equal_time.py": (JOB / "equal_time.py").read_bytes(),
        dataset: raw,
    }
    setup = "from pathlib import Path\n" + "".join(
        f"Path({name!r}).write_bytes({content!r})\n" for name, content in files.items())
    launch = f"""import subprocess, sys
subprocess.run([
    sys.executable, 'equal_time.py', '--data', '{dataset}', '--sha256', '{digest}',
    '--out', 'noise_geometry_equal_time_results',
], check=True)
"""
    display = """import pandas as pd
table = pd.read_csv('noise_geometry_equal_time_results/summary.csv')
display(table.round(6))
"""
    intro = """# Fixed-deadline gradient-noise pilot

Use **CPU** and **Save & Run All**. All inputs and code are embedded.
Download both files under `noise_geometry_equal_time_results/` after the run.

Every arm gets 0.45 seconds of elapsed training-loop time on one CPU thread.
The run stops after the first completed update at or after the deadline and
records the actual elapsed time. A shared nine-point LR grid is selected
separately for each optimizer and rotation on three fresh tuning seeds, using
loss at that same deadline. The 12 paired evaluation seeds are disjoint.

The full Evie-KF comparison is against Muon. AdamW, the diagonal gate, and
the random-basis gate show whether any advantage is specific to the learned
Kronecker gate. Check completed runs, paired wins, and median paired loss
ratios across all three rotation settings. These timings cover the synthetic
update loop, with noise precomputed. Real model training includes additional
forward and backward compute.
"""
    cells = []
    for index, (kind, content) in enumerate([
        ("markdown", intro), ("code", setup), ("code", launch), ("code", display)
    ]):
        cell = dict(cell_type=kind, id=f"deadline-{index}", source=content, metadata={})
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(name="python3", display_name="Python 3",
                                                  language="python")))
    (JOB / "noise_geometry_equal_time.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
