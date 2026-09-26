"""Package-level gates for evie-kf.

These are the checks a user installing the package can run in a few seconds on
a CPU. The full suite that gates the harness is in benchmark12/tests/.

    python tests/test_evie_kf.py
    pytest  tests/test_evie_kf.py
"""

import copy
import os
import sys

import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "..", "src"))

from evie_kf import EvieKF, eviekf_self_check, split_step  # noqa: E402


def _model(seed=0, dtype=torch.float32):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(8, 16), nn.Tanh(),
                         nn.Linear(16, 4)).to(dtype)


def _worst(a, b):
    return max(float((p.detach() - q.detach()).abs().max())
               for p, q in zip(a.parameters(), b.parameters()))


def _loss(model, x, y):
    return nn.functional.mse_loss(model(x), y)


def test_self_check():
    """The factored operator against a dense construction of the same thing."""
    r = eviekf_self_check()
    assert r["factored_vs_dense"] < 1e-9
    assert r["gamma0_identity"] < 1e-9
    assert r["gains_le_1"]


def test_gamma0_is_adamw():
    """gamma=0 reproduces torch.optim.AdamW to float precision (Prop. 4(iv)).

    Run in float64, so the bound measures the construction rather than
    float32 rounding accumulated over the 25 steps.
    """
    torch.manual_seed(1)
    x = torch.randn(32, 8, dtype=torch.float64)
    y = torch.randn(32, 4, dtype=torch.float64)

    a, b = _model(dtype=torch.float64), _model(dtype=torch.float64)
    b.load_state_dict(copy.deepcopy(a.state_dict()))

    oa = torch.optim.AdamW(a.parameters(), lr=1e-3, betas=(0.9, 0.999),
                           eps=1e-8, weight_decay=1e-4)
    ob = EvieKF(b.parameters(), lr=1e-3, gamma=0.0, weight_decay=1e-4)

    for _ in range(25):
        oa.zero_grad(set_to_none=True)
        _loss(a, x, y).backward()
        oa.step()

        ob.zero_grad(set_to_none=True)
        _loss(b, x, y).backward()
        ob.set_noise([(p, torch.zeros_like(p)) for p in b.parameters()])
        ob.step()

    worst = _worst(a, b)
    assert worst < 1e-14, worst


def test_split_step_activates_the_operator():
    """split_step feeds the noise sample, so the gate actually turns on."""
    torch.manual_seed(2)
    x, y = torch.randn(64, 8), torch.randn(64, 4)
    m = _model()
    o = EvieKF(m.parameters(), lr=1e-3, gamma=1000.0, warmup=2)

    for _ in range(30):
        loss = split_step(o, m, _loss, x, y)
        assert loss == loss                       # not NaN

    assert o.n_active_steps > 0, "set_noise never fired"
    assert o.last_cos < 1.0, "operator did not move the update off AdamW's"
    assert 0.0 < o.gain_lo <= 1.0


def test_step_without_noise_is_adamw():
    """The documented failure mode: no noise sample means no operator."""
    torch.manual_seed(3)
    x, y = torch.randn(64, 8), torch.randn(64, 4)
    m = _model()
    o = EvieKF(m.parameters(), lr=1e-3, gamma=1000.0, warmup=2)

    for _ in range(30):
        o.zero_grad(set_to_none=True)
        _loss(m, x, y).backward()
        o.set_noise([(p, torch.zeros_like(p)) for p in m.parameters()])
        o.step()

    assert o.last_cos == 1.0
    assert o.gain_lo == 1.0


def test_checkpoint_roundtrip():
    """state_dict survives a save/load, including the step counters."""
    torch.manual_seed(4)
    x, y = torch.randn(64, 8), torch.randn(64, 4)
    m = _model()
    o = EvieKF(m.parameters(), lr=1e-3, gamma=100.0, warmup=2)
    for _ in range(10):
        split_step(o, m, _loss, x, y)

    sd_m = copy.deepcopy(m.state_dict())
    sd_o = copy.deepcopy(o.state_dict())

    m2 = _model(seed=99)
    m2.load_state_dict(sd_m)
    o2 = EvieKF(m2.parameters(), lr=1e-3, gamma=100.0, warmup=2)
    o2.load_state_dict(sd_o)

    for _ in range(5):
        split_step(o, m, _loss, x, y)
        split_step(o2, m2, _loss, x, y)

    worst = _worst(m, m2)
    assert worst < 1e-10, worst


def test_diag_only_and_uncentred_run():
    """The two ablation arms the paper reports are reachable from the package."""
    torch.manual_seed(5)
    x, y = torch.randn(64, 8), torch.randn(64, 4)
    for kwargs in ({"diag_only": True}, {"centered": False},
                   {"diag_only": True, "diag_relative": False}):
        m = _model()
        o = EvieKF(m.parameters(), lr=1e-3, gamma=100.0, warmup=2, **kwargs)
        for _ in range(12):
            loss = split_step(o, m, _loss, x, y)
            assert loss == loss


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for fn in fns:
        fn()
        print("ok  %s" % fn.__name__)
    print("\n%d/%d passed" % (len(fns), len(fns)))
