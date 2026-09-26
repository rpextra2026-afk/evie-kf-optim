#!/usr/bin/env python3
"""
build.py -- assemble the self-contained ICLR-benchmark task files from the
fragments in this directory.

    python benchmark12/_build/build.py            # write all files
    python benchmark12/_build/build.py --check    # write + verify shared blocks
    python benchmark12/_build/build.py --diff     # show which files would change

Main benchmark (12 files, benchmark12/{vision,language,finance}/):
    frag_header (this task's TASK dict injected)
  + frag_<domain>_data
  + frag_shared_optim   <-- BYTE-IDENTICAL across every generated file
  + frag_engine         <-- BYTE-IDENTICAL across every generated file
  + `if __name__ == "__main__": main()`

Evie-KF ablation (3 files, benchmark12/ablation/): same 4 pieces; TASK carries
    opt_set="ablation" so the engine runs ABLATION_OPT_NAMES (EvieDiag/EvieKFu/
    EvieKF) instead of OPT_NAMES. One file per chosen host task
    (language_wikitext2, vision_cifar10, finance_tech) -- claude_optim.md sec 2.

Phase-2 sweeps (12 files, benchmark12/sweeps/{batchsize,modelsize}/):
    ... same 4 pieces ...
  + frag_sweep_engine
  + `if __name__ == "__main__": sweep_main()`

No imports of anything in this repo; every emitted file is paste-runnable.
To add Evie (or fix Muon): edit frag_shared_optim.py and re-run -- the change
lands in all files at once (CLAUDE.md sec 2).
"""
import os
import sys
import json
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.dirname(HERE)
ROOT = os.path.dirname(BENCH)

sys.path.insert(0, HERE)
from tasks import (TASKS, ABLATION_TASKS, SWEEP_TASKS, SCALEGEN_TASKS,  # noqa: E402
                   DIAG_TASKS)

SHARED_BEGIN = "# === SHARED OPTIMIZER BLOCK -- keep identical across all 12 files ==="
SHARED_END = "# === END SHARED OPTIMIZER BLOCK ==="
ENGINE_BEGIN = "#  ENGINE  --  preflight, checkpoint/resume, divergence & error handling,"
ENGINE_END = "# === END ENGINE BLOCK ==="

GEN_BANNER = (
    "# >>> GENERATED FILE -- do not edit. Source: benchmark12/_build/*.py + "
    "tasks.py\n"
    "# >>> Rebuild:  python benchmark12/_build/build.py\n"
)


def _read(name):
    with open(os.path.join(HERE, name), "r") as fh:
        return fh.read()


def _rel_path(task):
    if task.get("is_sweep"):
        return f"benchmark12/sweeps/{task['sweep_type']}/{task['task_name']}.py"
    if task.get("opt_set") == "ablation":
        return f"benchmark12/ablation/{task['task_name']}.py"
    if task.get("opt_set") == "diag":
        return f"benchmark12/diagnostics/{task['task_name']}.py"
    if task.get("opt_set") == "scalegen":
        # kept visually/operationally separate from the primary 12, so nobody
        # accidentally feeds them into the main-run's cross-task Wilcoxon by
        # pattern-matching a directory glob (evie-scalegen-spec.md sec 3.5).
        return f"benchmark12/scalegen/{task['task_name']}.py"
    return f"benchmark12/{task['domain']}/{task['task_name']}.py"


_MAIN_OPT_SUMMARY = ("AdamW, AdaBelief, SGD(+momentum, Nesterov, cosine), "
                     "Sophia, Muon, Lion, Shampoo, EvieKF")
_ABLATION_OPT_SUMMARY = ("EvieDiag, EvieKFu, EvieKF  (Evie-KF ablation ladder, "
                         "10 seeds; EvieKF is also the main table's 8th arm)")
_DIAG_OPT_SUMMARY = ("a pinned single-arm diagnostic re-run (see TASK[\"title\"]); "
                     "hyper-parameters are PINNED, not searched, and the result "
                     "is NOT part of the 12-task significance table")
_SCALEGEN_OPT_SUMMARY = ("AdamW, AdaBelief, Muon, EvieKF (scale-generalization "
                         "secondary check, 3 seeds, NOT part of the 12-task "
                         "significance table)")


def build_one(task):
    header = _read("frag_header.py")
    engine = _read("frag_engine.py")
    shared = _read("frag_shared_optim.py")
    data = _read(f"frag_{task['domain']}_data.py")
    is_sweep = bool(task.get("is_sweep"))
    rel_path = _rel_path(task)

    subs = {
        "TASK_TITLE": task["title"],
        "TASK_NAME": task["task_name"],
        "DOMAIN": task["domain"],
        "PRIMARY_METRIC": task["primary_metric"],
        "PRIMARY_DIR": "lower is better" if task["primary_lower_better"]
                       else "higher is better",
        "OPT_SUMMARY": (_ABLATION_OPT_SUMMARY if task.get("opt_set") == "ablation"
                        else _SCALEGEN_OPT_SUMMARY if task.get("opt_set") == "scalegen"
                        else _DIAG_OPT_SUMMARY if task.get("opt_set") == "diag"
                        else _MAIN_OPT_SUMMARY),
        "N_SEEDS": str(len(task["seeds"])),
        "MAX_STEPS": str(task["max_steps"]),
        "REL_PATH": rel_path,
        "TASK_JSON": json.dumps(task, indent=2),
    }
    for k, v in subs.items():
        header = header.replace("{" + k + "}", v)
    leftover = [k for k in subs if "{" + k + "}" in header]
    assert not leftover, f"{rel_path}: unfilled placeholders {leftover}"

    parts = [GEN_BANNER,
             header.rstrip() + "\n", "\n\n",
             data.rstrip() + "\n", "\n\n",
             shared.rstrip() + "\n", "\n\n",
             engine.rstrip() + "\n"]
    if is_sweep:
        parts += ["\n\n", _read("frag_sweep_engine.py").rstrip() + "\n"]
        entry = "sweep_main"
    else:
        entry = "main"
    parts += ["\n\n", f'if __name__ == "__main__":\n    {entry}()\n']
    return "".join(parts), rel_path


def _slice(text, begin, end):
    a = text.index(begin)
    return text[a:text.index(end, a) + len(end)]


def shared_block_of(text):
    return _slice(text, SHARED_BEGIN, SHARED_END)


def engine_block_of(text):
    return _slice(text, ENGINE_BEGIN, ENGINE_END)


def main(argv):
    check = "--check" in argv
    diff = "--diff" in argv
    written, shared_h, engine_h = [], {}, {}
    for task in TASKS + ABLATION_TASKS + SWEEP_TASKS + SCALEGEN_TASKS + DIAG_TASKS:
        content, rel_path = build_one(task)
        out_path = os.path.join(ROOT, rel_path)
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        old = open(out_path).read() if os.path.exists(out_path) else None
        if diff:
            print(f"{'CHANGED' if old != content else 'same   '}  {rel_path}")
        else:
            with open(out_path, "w") as fh:
                fh.write(content)
        shared_h[rel_path] = hashlib.sha256(shared_block_of(content).encode()).hexdigest()
        engine_h[rel_path] = hashlib.sha256(engine_block_of(content).encode()).hexdigest()
        written.append(rel_path)

    n_main = len(TASKS)
    n_ablation = len(ABLATION_TASKS)
    n_sweep = len(SWEEP_TASKS)
    n_scalegen = len(SCALEGEN_TASKS)
    n_diag = len(DIAG_TASKS)
    ok_shared = len(set(shared_h.values())) == 1
    ok_engine = len(set(engine_h.values())) == 1
    print(f"\n{'[diff]' if diff else '[build]'} {len(written)} files "
          f"({n_main} main + {n_ablation} ablation + {n_sweep} sweep + "
          f"{n_scalegen} scalegen + {n_diag} diagnostic)")
    print(f"[shared-optim-block] {'IDENTICAL' if ok_shared else 'MISMATCH!!'} "
          f"across all {len(shared_h)} files"
          + (f"  sha256={list(shared_h.values())[0][:16]}..." if ok_shared else ""))
    print(f"[engine-block]       {'IDENTICAL' if ok_engine else 'MISMATCH!!'} "
          f"across all {len(engine_h)} files"
          + (f"  sha256={list(engine_h.values())[0][:16]}..." if ok_engine else ""))
    if not (ok_shared and ok_engine):
        for lbl, hh in (("shared", shared_h), ("engine", engine_h)):
            if len(set(hh.values())) != 1:
                for p, h in hh.items():
                    print(f"   [{lbl}] {h[:16]}  {p}")
        return 1
    if check:
        print("[check] OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
