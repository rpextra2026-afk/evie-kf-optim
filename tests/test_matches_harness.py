"""The package and the harness must be the same optimizer.

`src/evie_kf/optimizer.py` is sliced out of
`benchmark12/_build/frag_shared_optim.py`, the fragment pasted byte-for-byte
into all 43 generated task files. Every number in the paper came from the
fragment; anyone reproducing the method installs the package. If the two ever
diverge, the paper describes one optimizer and the release ships another.

This asserts they have not. It is the reason the package source is in this
repository rather than only on PyPI.

    python tests/test_matches_harness.py
    pytest  tests/test_matches_harness.py
"""

import io
import os
import re

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
PKG = os.path.join(ROOT, "src", "evie_kf", "optimizer.py")
FRAG = os.path.join(ROOT, "benchmark12", "_build", "frag_shared_optim.py")
GENERATED = os.path.join(ROOT, "benchmark12", "vision", "vision_cifar10.py")


def _slice(path, start, end):
    """The lines from the first line matching `start` up to, not including,
    the first later line matching `end`."""
    lines = io.open(path, encoding="utf-8").read().split("\n")
    try:
        a = next(i for i, l in enumerate(lines) if re.match(start, l))
    except StopIteration:
        raise AssertionError("%s: no line matching %r" % (path, start))
    b = next((i for i in range(a + 1, len(lines)) if re.match(end, lines[i])),
             len(lines))
    return "\n".join(lines[a:b]).rstrip()


CLASS = (r"^class EvieKF\(", r"^def eviekf_self_check\(")
# In the fragment the self-check is followed by build_optimizer(); in the
# package it is followed by the split_step section banner, or by end of file.
CHECK = (r"^def eviekf_self_check\(", r"^def build_optimizer\(|^# ----")
CONSTS = (r"^EVIEKF_BETAS\b", r"^EVIEKF_GAMMA_GRID\b")


def _diff(a, b, label):
    la, lb = a.split("\n"), b.split("\n")
    for i, (x, y) in enumerate(zip(la, lb)):
        if x != y:
            return ("%s: first difference at line %d\n  package: %r\n"
                    "  harness: %r" % (label, i + 1, x, y))
    if len(la) != len(lb):
        return "%s: %d lines in package, %d in harness" % (label, len(la),
                                                           len(lb))
    return None


def test_class_matches_fragment():
    d = _diff(_slice(PKG, *CLASS), _slice(FRAG, *CLASS), "class EvieKF")
    assert d is None, d


def test_self_check_matches_fragment():
    d = _diff(_slice(PKG, *CHECK), _slice(FRAG, *CHECK), "eviekf_self_check")
    assert d is None, d


def test_constants_match_fragment():
    d = _diff(_slice(PKG, *CONSTS), _slice(FRAG, *CONSTS), "EVIEKF_* defaults")
    assert d is None, d


def test_fragment_matches_a_generated_task_file():
    """And the fragment is what a task file actually runs, so the package is
    the code that produced the results, not a copy of a copy."""
    d = _diff(_slice(FRAG, *CLASS), _slice(GENERATED, *CLASS),
              "class EvieKF (fragment vs generated file)")
    assert d is None, d


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  %s" % fn.__name__)
    print("\n%d/%d passed" % (len(fns), len(fns)))
