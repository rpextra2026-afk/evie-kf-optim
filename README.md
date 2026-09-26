# Evie-KF

A Kronecker-factored, risk-sensitive gradient-noise preconditioner for AdamW.

Adam adapts each update to the magnitude of its own coordinate and discards the
correlations in gradient noise. Evie-KF keeps Adam's diagonal whitening and adds
a matrix gate built from the **centred** noise covariance in those whitened
coordinates:

```
dtheta = -lr * P^-1/2 (I + gamma * Sigma_z)^-1/2 P^-1/2 m_hat
```

`P` is AdamW's own denominator and `Sigma_z ~ (B kron A) / tr(A)` is the
Kronecker-factored noise covariance. The gate's eigenvalues lie in `(0, 1]`, so
it shrinks along noisy directions; a global rescale then restores AdamW's step
length, which turns the shrink into a rotation. **At `gamma = 0` the gate is the
identity and the step is AdamW's, exactly** — verified against
`torch.optim.AdamW` to `1e-14` in the test suite.

This repository is the anonymised release accompanying the submission. It holds
the optimizer as an installable package, the full 12-task benchmark harness, and
the raw per-cell records behind every number in the paper.

## Install

```bash
pip install evie-kf
```

or, from a clone:

```bash
pip install -e .
```

## Use

Evie-KF needs a sample of the gradient noise, which it takes from two disjoint
half batches: the mean of the two half gradients is the ordinary minibatch
gradient and their difference is a noise sample provably uncorrelated with it.
Two half backward passes cost the same as one full one and use no extra data.

`split_step` does the whole step:

```python
import torch, torch.nn as nn
from evie_kf import EvieKF, split_step

model = MyModel()
opt = EvieKF(model.parameters(), lr=1e-3, gamma=300.0, weight_decay=1e-4)

def loss_fn(model, x, y):
    return nn.functional.cross_entropy(model(x), y)

for x, y in loader:
    loss = split_step(opt, model, loss_fn, x, y, grad_clip=5.0)
```

If you drive the loop yourself, call `set_noise()` before every `step()`:

```python
opt.zero_grad(set_to_none=True)
# ... build p.grad = (gA + gB) / 2 ...
opt.set_noise([(p, (gA_p - gB_p) / 2) for p, gA_p, gB_p in ...])
opt.step()
```

**If `set_noise()` is never called the operator never activates and Evie-KF is
silently AdamW.** `opt.n_active_steps` counts the steps on which the gate was
live; check it is non-zero at the end of a run.

## Hyperparameters

| argument | default | notes |
|---|---|---|
| `lr` | `1e-3` | tuned per task; Evie-KF often prefers a higher `lr` than AdamW |
| `gamma` | `100.0` | the one tuned knob; `0.0` is AdamW exactly |
| `betas`, `eps`, `weight_decay` | AdamW's | decoupled weight decay, applied before the update |
| `beta_sig` | `0.95` | EMA rate for the Kronecker factors |
| `warmup` | `20` | steps before the gate turns on |
| `refresh` | `20` | steps between eigendecompositions |
| `maxf` | `1024` | a factor larger than this is treated diagonally |

`gamma` is searched on a half-decade grid. Under the shipped `tr(A)`
normalisation it carries each task's absolute noise energy, so the selected
value varies widely across tasks (3 to 243,000 in the paper's twelve); the
appendix reports a mean-eigenvalue normalisation that makes it dimensionless.
Every frozen `(lr, gamma)` pair used in the paper is in
`benchmark12/results/paper/table6_tuning_budget.csv`.

Three flags select the ablation arms the paper reports: `diag_only=True` for the
diagonal gate, `diag_relative=False` for the derivation's own absolute diagonal
case, and `centered=False` for the uncentred covariance that published Kronecker
methods use.

## Cost

Per `n x m` tensor Evie-KF stores `A`, `B` and their eigenbases, about
`2(n^2 + m^2)` floats against AdamW's `2nm`. Each step costs `O(nm(n + m))` and
each refresh `O(n^3 + m^3)`, amortised over 20 steps. The relative overhead
falls as the model grows: `4.7x` AdamW on a 2,497-parameter MLP, `1.6x` at
9–11M parameters, and `0.98x` at 33.7M.

## Repository layout

```
src/evie_kf/          the optimizer, sliced verbatim out of the harness fragment
tests/                package-level gates, CPU-only, a few seconds
benchmark12/
  _build/             the eight fragments and the generator
  vision/ language/ finance/     the 12 main-run task files
  ablation/ diagnostics/ sweeps/ scalegen/    the secondary task files
  tests/              the suite that gates the harness
  results/            raw per-cell records, logs, and the table/figure pipeline
  RUNBOOK.md          how to run a task end to end
```

Every task is a single self-contained script, generated from the shared
fragments so that the optimizer block, the training engine and the data loaders
are byte-identical across all 43 files. `benchmark12/_build/check_shared_block.py`
refuses to build if a generated file has drifted. `src/evie_kf/optimizer.py` is
sliced out of `benchmark12/_build/frag_shared_optim.py` rather than retyped, so
the package and the harness cannot diverge.

## Reproducing the paper's tables

```bash
python benchmark12/results/build_paper_tables.py      # Tables 1-14
python benchmark12/make_appendix_tables.py            # the appendix tables
python benchmark12/make_curve_figure.py               # Figure 5
python benchmark12/make_gamma_figure.py               # Figure 7
python benchmark12/make_results_figures.py            # Figures 6 and 8
```

These read the committed records under `benchmark12/results/` and write LaTeX,
CSV and PDF. No number in the paper is entered by hand.

## Tests

```bash
python tests/test_evie_kf.py              # the package
python tests/test_matches_harness.py      # package == harness, line by line
python benchmark12/tests/test_eviekf.py   # the shared optimizer block
python benchmark12/tests/test_derivation.py   # the update against a dense build
```

`test_matches_harness.py` is why the optimizer source is in this repository and
not only on PyPI: it asserts that `src/evie_kf/optimizer.py` is line-for-line
the `EvieKF` class, self-check and defaults from
`benchmark12/_build/frag_shared_optim.py`, and that the fragment is in turn
what a generated task file runs. The package you install and the code that
produced every number in the paper are checked to be the same optimizer.

`test_derivation.py` forms the whole `nm x nm` operator densely, taking one
eigendecomposition of it and sharing no code path with the optimizer beyond
PyTorch, and checks the shipped factored version against it.

## Hardware

Every cell in the paper ran on one NVIDIA Tesla T4 under PyTorch 2.10.0+cu128,
with per-cell checkpointing and resume. Nothing here needs a GPU to import, test
or inspect.

## License

MIT.
