"""Build the self-contained notebook for the published checkpoint diagnostic."""

import base64
import gzip
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
JOB = ROOT / "kaggle/jobs/checkpoint_probe"


def main():
    plan = json.loads((JOB / "plan.json").read_text(encoding="utf-8"))
    source = ROOT / "benchmark12/language/language_wikitext2.py"
    # Normalize checkout line endings when embedding the frozen task.
    task_bytes = source.read_bytes().replace(b"\r\n", b"\n")
    plan["probe"]["source_sha256"] = hashlib.sha256(task_bytes).hexdigest()
    files = {
        "plan.json": (json.dumps(plan, indent=2) + "\n").encode(),
        "run_followup.py": (JOB / "run_followup.py").read_bytes().replace(b"\r\n", b"\n"),
        "checkpoint_probe.py": (JOB / "checkpoint_probe.py").read_bytes().replace(b"\r\n", b"\n"),
        "tasks/language_wikitext2.py": task_bytes,
    }
    setup = "from pathlib import Path\nfrom base64 import b64decode\nfrom gzip import decompress\n"
    for name, content in files.items():
        encoded = base64.b64encode(gzip.compress(content, mtime=0)).decode()
        setup += f"Path({name!r}).parent.mkdir(parents=True, exist_ok=True)\n"
        setup += f"Path({name!r}).write_bytes(decompress(b64decode({encoded!r})))\n"
    launch = """import os
from pathlib import Path
import subprocess
import sys
from zipfile import ZIP_DEFLATED, ZipFile

subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'datasets', 'scipy', 'pandas'], check=True)
try:
    subprocess.run([sys.executable, '-u', 'run_followup.py'],
                   env=dict(os.environ, CUDA_VISIBLE_DEVICES='0'), check=True)
finally:
    with ZipFile('checkpoint_probe_17101.zip', 'w', ZIP_DEFLATED) as archive:
        for path in sorted(Path('measurement_output').rglob('*.json')):
            archive.write(path, path.as_posix())
"""
    intro = (
        "# WikiText-2 checkpoint diagnostic\n\n"
        "Use Kaggle GPU T4, Internet On, and Save & Run All. No dataset attachment is needed. "
        "WikiText-2 is downloaded automatically. The notebook trains the original Evie-KF "
        "configuration for 10,000 steps, then probes coupling and directional variance at "
        "six fixed checkpoints. It uses one T4. Download checkpoint_probe_17101.zip from Output.\n"
    )
    cells = []
    for i, (kind, content) in enumerate((("markdown", intro), ("code", setup), ("code", launch))):
        cell = dict(cell_type=kind, id=f"checkpoint-{i}", source=content, metadata={})
        if kind == "code":
            cell.update(execution_count=None, outputs=[])
        cells.append(cell)
    notebook = dict(cells=cells, nbformat=4, nbformat_minor=5,
                    metadata=dict(kernelspec=dict(name="python3", display_name="Python 3",
                                                  language="python")))
    (JOB / "checkpoint_probe_17101.ipynb").write_text(
        json.dumps(notebook, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
