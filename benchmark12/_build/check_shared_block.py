#!/usr/bin/env python3
"""
check_shared_block.py -- CI guard. Verifies the SHARED OPTIMIZER BLOCK and the
ENGINE block are byte-for-byte identical across every generated file (12 main +
3 Evie-KF ablation + 12 sweep), and match the current fragments (i.e. build.py
was re-run after the last fragment edit). Exit non-zero on any drift.

    python benchmark12/_build/check_shared_block.py
"""
import os
import sys
import glob
import hashlib

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))

SHARED_BEGIN = "# === SHARED OPTIMIZER BLOCK -- keep identical across all 12 files ==="
SHARED_END = "# === END SHARED OPTIMIZER BLOCK ==="
ENGINE_BEGIN = "#  ENGINE  --  preflight, checkpoint/resume, divergence & error handling,"
ENGINE_END = "# === END ENGINE BLOCK ==="


def _slice(text, begin, end):
    a = text.index(begin)
    return text[a:text.index(end, a) + len(end)]


def main():
    main_files = sorted(f for d in ("vision", "language", "finance")
                        for f in glob.glob(os.path.join(ROOT, "benchmark12", d, "*.py")))
    ablation_files = sorted(glob.glob(os.path.join(ROOT, "benchmark12", "ablation",
                                                  "*.py")))
    sweep_files = sorted(glob.glob(os.path.join(ROOT, "benchmark12", "sweeps",
                                                "*", "*.py")))
    scalegen_files = sorted(glob.glob(os.path.join(ROOT, "benchmark12", "scalegen",
                                                   "*.py")))
    files = main_files + ablation_files + sweep_files + scalegen_files
    problems = 0
    if len(main_files) != 12:
        print(f"expected 12 main files, found {len(main_files)}")
        problems += 1
    if len(ablation_files) != 3:
        print(f"expected 3 ablation files, found {len(ablation_files)}")
        problems += 1
    if len(sweep_files) != 12:
        print(f"expected 12 sweep files, found {len(sweep_files)}")
        problems += 1
    if len(scalegen_files) != 2:
        print(f"expected 2 scalegen files, found {len(scalegen_files)}")
        problems += 1

    shared_frag = _slice(_read(os.path.join(HERE, "frag_shared_optim.py")),
                         SHARED_BEGIN, SHARED_END)
    shared_h, engine_h = {}, {}
    for f in files:
        t = _read(f)
        rel = os.path.relpath(f, ROOT)
        s = _slice(t, SHARED_BEGIN, SHARED_END)
        e = _slice(t, ENGINE_BEGIN, ENGINE_END)
        shared_h[rel] = hashlib.sha256(s.encode()).hexdigest()
        engine_h[rel] = hashlib.sha256(e.encode()).hexdigest()
        if s != shared_frag:
            print(f"DRIFT: shared block in {rel} != frag_shared_optim.py "
                  f"(re-run build.py)")
            problems += 1

    for label, hh in (("shared optimizer block", shared_h),
                      ("engine block", engine_h)):
        if len(set(hh.values())) != 1:
            print(f"MISMATCH: {label} differs between generated files:")
            for r, h in hh.items():
                print(f"   {h[:16]}  {r}")
            problems += 1

    if problems == 0:
        print(f"OK: shared optimizer block + engine block byte-identical across "
              f"all {len(files)} files (12 main + 3 ablation + 12 sweep + "
              f"2 scalegen)")
        print(f"    shared sha256 = {list(shared_h.values())[0][:24]}")
        print(f"    engine sha256 = {list(engine_h.values())[0][:24]}")
    return 1 if problems else 0


def _read(p):
    with open(p) as fh:
        return fh.read()


if __name__ == "__main__":
    sys.exit(main())
