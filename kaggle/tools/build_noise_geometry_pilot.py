"""Freeze the synthetic design and embed it and the benchmark optimizers."""

import hashlib
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "kaggle/jobs/noise_geometry_pilot"


def build():
    source = (ROOT / "benchmark12/_build/frag_shared_optim.py").read_text(encoding="utf-8")
    optimizers = (
        "import math\nimport numpy as np\nimport torch\nimport torch.nn as nn\n"
        "WEIGHT_DECAY = 0.0\nGRAD_CLIP_NORM = 5.0\nHESS_FREQ = 10\nN_HUTCH = 1\n"
        "log = print\n\n" + source
    )
    row = np.geomspace(1, 0.032, 16)
    column = np.geomspace(1, 0.1, 24)
    row *= 4 / (row.sum() * column.sum())
    curvature = np.outer(np.geomspace(1, 0.01, 16), np.geomspace(1, 0.1, 24))
    grid = [float(f"{10 ** (k / 2):.7g}") for k in range(-8, -1)]
    data = dict(
        description="Fixed matrix quadratic, rotated Kronecker Gaussian noise, trace 4",
        row_eigenvalues=row.tolist(), column_eigenvalues=column.tolist(),
        curvature=curvature.tolist(), rotations=[0.0, 0.5, 1.0], gamma=30,
        steps=1200, eval_every=20, tune_seeds=[100, 101], evaluation_seeds=list(range(200, 208)),
        lr_grids={arm: grid for arm in
                  ("AdamW", "Muon", "Evie-KF", "Evie diagonal", "Evie random basis")},
    )
    dataset = json.dumps(data, indent=2) + "\n"
    sha = hashlib.sha256(dataset.encode()).hexdigest()
    (JOB / "noise_geometry_data.json").write_bytes(dataset.encode())
    setup = ("from pathlib import Path\n"
             f"Path('pilot_optimizers.py').write_text({optimizers!r}, encoding='utf-8')\n"
             f"Path('pilot.py').write_text({(JOB / 'pilot.py').read_text(encoding='utf-8')!r}, encoding='utf-8')\n"
             f"Path('noise_geometry_data.json').write_bytes({dataset.encode()!r})\n")
    launch = f"""import subprocess, sys
subprocess.run([
    sys.executable, 'pilot.py', '--data', 'noise_geometry_data.json',
    '--sha256', '{sha}', '--out', 'noise_geometry_results', '--device', 'cpu',
], check=True)
"""
    display = """from pathlib import Path
import pandas as pd
summary = pd.read_csv(Path('noise_geometry_results') / 'summary.csv')
display(summary.round(5))
"""
    markdown = """# Gradient-noise geometry pilot

Run on CPU using **Save & Run All**. The synthetic design and optimizer code are embedded. Download `noise_geometry_results/results.json` and `noise_geometry_results/summary.csv` after the run.

We rotate the two Kronecker factors from coordinate-aligned to correlated while keeping the full covariance eigenvalues, total variance, curvature and seed draws fixed. Each optimizer gets the same seven learning rates and two tuning seeds per regime. Evaluation uses eight separate paired seeds. The Evie variants use fixed mean-normalized gamma 30. The random-basis arm keeps the learned gain spectrum and replaces its directions, while the diagonal arm uses only elementwise noise variance.

Compare the paired loss ratios to Muon across regimes. Ratios below 1 favor the corresponding arm. Equal-time results are measured at the last observed checkpoint within a common horizon per seed, with precomputed gradient noise and a single CPU thread. They measure the pilot update loop and exclude tuning, input generation, and model forward/backward. This is a controlled mechanism test using synthetic gradients, with one matrix parameter and no clipping or weight decay.
"""
    cells = []
    for index, (kind, body) in enumerate([
        ("markdown", markdown), ("code", setup), ("code", launch), ("code", display),
    ]):
        cell = dict(cell_type=kind, id=f"noise-pilot-{index}", source=body, metadata={})
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(name="python3", display_name="Python 3",
                                                   language="python")))
    (JOB / "noise_geometry_pilot.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    build()
