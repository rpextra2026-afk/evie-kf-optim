# benchmark12 — the 12-task ICLR optimizer benchmark harness

Supersedes the old `vision/` `language/` `finance/` code (kept untouched as
reference). Twelve **single, self-contained, paste-runnable** task files —
4 vision, 4 language, 4 finance — each running the same baseline optimizers
through a pre-registered LR-search + multi-seed protocol.

**Optimizers (main table, `OPT_NAMES`, 8):** AdamW, AdaBelief, SGD (+momentum,
Nesterov, cosine), Sophia, Muon, Lion, **Shampoo** (per-layer, AdamW-grafted),
**EvieKF** (Kronecker-factored risk-sensitive preconditioner, `centered=True`,
`per_layer_norm=False` — the variant validated in `Claude outputs/kaggle_evie_v3.py`:
10 seeds, Holm-corrected, beats AdamW/AdaBelief/Lion/Muon/Shampoo/EvieDiag at
p=0.0078, ties its own uncentred ablation). `EvieKFn` (`per_layer_norm=True`)
stays defined in `frag_shared_optim.py` but is **no longer selected anywhere**.
Shampoo and the Evie-KF family also joint-search a **second axis** (`KNOB_GRID`:
γ for Evie-KF, β for Shampoo) alongside the LR grid, same 7-point /
boundary-extend discipline.

**Evie-KF ablation (`ABLATION_OPT_NAMES`, 3 arms, 10 seeds — `benchmark12/ablation/`):**
`EvieDiag`, `EvieKFu`, `EvieKF` — the C1 (centred vs uncentred Σ) / C2
(Kronecker vs diagonal) ladder. Three generated files, one per chosen host task
(`language_wikitext2`, `vision_cifar10`, `finance_tech`); each is a normal
single-task file whose `TASK["opt_set"] == "ablation"` makes the engine iterate
`ABLATION_OPT_NAMES`. 10 seeds (`[42..51]`), not the main run's 5 — at n=5 no
result can survive Holm correction across 8 comparisons. `EvieKF` is **both**
the main table's 8th arm and one of these 3 arms, so the 3 hosts compute it
twice (5-seed main cell + 10-seed ablation cell, a superset); ~15 extra cells,
a known accepted redundancy. (`claude_optim.md` sec 2 phrases the ablation as
"a 13th file"; one file per domain is used instead so every fragment stays
byte-identical and no cross-domain primary-metric handling has to be bolted
onto the engine.)

```
benchmark12/
  _build/              generator + source fragments (the ONLY place to edit)
    frag_header.py         per-file config + logging + RNG
    frag_shared_optim.py   optimizer classes + Shampoo + Evie-KF  ── byte-identical everywhere
    frag_engine.py         preflight / checkpoint / divergence / LR+knob search /
                           split-batch noise hook / stats
    frag_sweep_engine.py   Phase-2 grid driver (sweep files only)
    frag_vision_data.py    CIFAR-stem ResNet-18 + loaders
    frag_language_data.py  small GPT + tokenizers + loaders
    frag_finance_data.py   OHLCV MLP + frozen-panel loader
    tasks.py               12 main + 3 Evie-KF ablation + 12 sweep task configs
    build.py               assembles all 27 files
    check_shared_block.py  CI guard: shared + engine blocks byte-identical
  vision/  language/  finance/     the 12 GENERATED main-run files (committed)
  ablation/                        3 GENERATED Evie-KF ablation files (committed)
  sweeps/batchsize/  sweeps/modelsize/   12 GENERATED Phase-2 sweep files
  finance/data/                    frozen OHLCV CSV snapshots (committed)
  tests/test_optimizers.py         CPU optimizer correctness tests
  tests/test_eviekf.py             CPU Evie-KF / Shampoo gates (parity, resume, γ=0)
  aggregate.py                     the confirmatory 12-task statistical test
  RUNBOOK.md                       start-to-finish operational guide (incl. sweeps + Evie)
```

## Run a task (one documented command)

```bash
# on the GPU machine, from the repo root:
python benchmark12/vision/vision_cifar10.py
```

That single command does the whole task: 7-point half-decade LR search per
optimizer (25% budget, 1 seed, boundary re-search), then the full sweep
(6 optimizers × 5 seeds × 10 000 steps), then the per-task report. Everything is
written under `checkpoints/<TASK_NAME>/`. **Re-running the same command resumes**:
finished `(optimizer, unit, seed)` cells are skipped, an interrupted cell resumes
from its last in-cell checkpoint, LR-search grid points already scored are
reused. Files run independently and can go 3-at-a-time across the 3 A6000s
(nothing is ever written outside a file's own `checkpoints/<TASK_NAME>/`).

Then, once some/all task files have finished:

```bash
python benchmark12/aggregate.py            # 12-task paired Wilcoxon + sign test
```

### First-time finance setup (once, then commit)

```bash
pip install yfinance pandas
python scripts/download_finance_panels.py
git add benchmark12/finance/data && git commit -m "freeze finance panels"
```

The 4 finance task files read only `benchmark12/finance/data/<panel>.csv` —
**no `yfinance` at runtime**.

## The 12 tasks (frozen — CLAUDE.md §3)

| # | file | data | model | primary metric |
|---|------|------|-------|----------------|
| 1 | `vision/vision_cifar10.py` | CIFAR-10 | CIFAR-stem ResNet-18 | test top-1 acc |
| 2 | `vision/vision_cifar100.py` | CIFAR-100 | ″ | test top-1 acc |
| 3 | `vision/vision_svhn.py` | SVHN | ″ | test top-1 acc |
| 4 | `vision/vision_stl10.py` | STL-10 (resized 96→32) | ″ | test top-1 acc |
| 5 | `language/language_wikitext2.py` | WikiText-2 | small GPT (6L, d256, 8h, ctx256) | test perplexity |
| 6 | `language/language_wikitext103.py` | WikiText-103 (token-budget cap) | ″ | test perplexity |
| 7 | `language/language_ptb.py` | Penn Treebank | ″ | test perplexity |
| 8 | `language/language_enwik8.py` | enwik8 (byte-level) | ″ | test perplexity |
| 9 | `finance/finance_tech.py` | Tech sector panel | 3-layer OHLCV MLP | test MSE (next-day log-return) |
| 10 | `finance/finance_finance.py` | Finance sector panel | ″ | test MSE |
| 11 | `finance/finance_healthcare.py` | Healthcare sector panel | ″ | test MSE |
| 12 | `finance/finance_consumer_industrials.py` | Consumer & Industrials panel | ″ | test MSE |

## Baseline optimizers (frozen 2026-09-08 — CLAUDE.md §4)

AdamW · AdaBelief · SGD (+ Nesterov momentum + cosine) · Sophia · Muon · Lion.
All 6 live in the **SHARED OPTIMIZER BLOCK** (`frag_shared_optim.py`), pasted
byte-for-byte identically into all 12 files. Common: decoupled weight decay
`1e-4` (`p ← p·(1 − lr·λ)` before the update), global grad-norm clip `5.0`,
float32, cosine LR to 10 % of peak, Sophia Hutchinson probe every 5 steps × 4
Rademacher vectors.

**Muon** (§8.2 was open; resolved here): heavy-ball Nesterov momentum `0.95`
→ 5-step Newton-Schulz quintic `(3.4445, −4.7750, 2.0315)` on every param with
`ndim ≥ 2` (conv weights reshaped `(O, I·kH·kW)`), orthogonalised update scaled
by `0.2·√max(fan_out, fan_in)`; AdamW for everything else (biases, norm params,
1-D params, and — for the LM — the token/positional embeddings and the tied
output projection). The **7-point LR search tunes only the Muon (≥2-D) group's
LR**; the AdamW sub-group's LR is fixed to that file's separately-frozen AdamW
LR (never a second search axis).

## Protocol (CLAUDE.md §5)

- **LR grid**: 7-point half-decade (×√10), per-optimizer-family centre
  (`LR_GRID_CENTRE` in the shared block); every optimizer gets the same *number*
  of points. **Boundary rule**: a pick on a grid edge triggers a logged
  extend-and-re-search (up to 3 rounds).
- **Tune** at 2 500 steps (25 % of 10 000), 1 seed (42); LR selection uses the
  **best-checkpoint** validation metric during the search run, never the last
  step.
- **Main sweep**: 5 seeds `[42, 43, 44, 45, 46]`, 10 000 steps, cosine schedule.
- **Best-checkpoint test**: restore the best-validation checkpoint before the
  single, final, full-test evaluation. In-loop validation may use a fast subset;
  the reported test metric always uses the full untouched test split.
- **One primary metric per domain**, pre-registered; stats run only on it.
  Everything else (wall-clock, peak mem, GFLOPs, step size, clip frequency,
  Hessian norm/CV) is descriptive — no p-values.
- **Unit of analysis = the task**. Per `(task, optimizer)`: median over seeds of
  the per-seed mean-over-units primary → one number. `aggregate.py` then runs a
  **paired Wilcoxon signed-rank across the 12 tasks** on the direction-normalised
  relative improvement vs the reference (default AdamW; `--ref Evie` later),
  with an exact **sign test** as a robustness check and **Holm-Bonferroni**
  across the optimizer family.

## Error handling & checkpointing (CLAUDE.md §6)

- **Task-name-scoped directories** — every path is under
  `checkpoints/<TASK_NAME>/…`; two files on two GPUs from the same CWD cannot
  collide. One checkpoint per `(optimizer, unit, seed)` cell; atomic write
  (`tmp` with PID+random suffix → `fsync` → rotate old to `.bak` → `os.replace`).
- **Resumable** at cell granularity and mid-cell (every `ckpt_every` steps);
  a crash loses at most the in-progress cell.
- **SIGTERM/SIGINT** flush the current cell checkpoint, then exit 0.
- **Preflight**: verifies CUDA is usable at the expected compute capability
  (`get_device_capability` vs `get_arch_list`) and runs a live matmul; a
  mismatched torch/GPU build fails in one sentence *before* any training.
- **Environment errors ≠ divergence**: CUDA/CUBLAS/cuDNN/OOM-shaped exceptions
  are re-raised immediately and abort the file (they are *not* swallowed as
  "this run diverged"); only NaN/Inf/shape-type numerical failures mark a single
  cell failed and continue.
- **Absolute divergence threshold**, not just "worse than this run's own best":
  a cell is flagged `diverged` if its primary is outside a hard sane range
  (`primary_abs_range`) or, in a post-sweep pass, more than
  `peer_divergence_factor`× off the task's peer median — so a run broken from
  step 0 (no "own best") is still caught.
- Any single-cell numerical failure is logged, recorded as `diverged` in
  `results.json`, and the sweep continues.

## Deviations from CLAUDE.md defaults (with rationale)

1. **Vision init**: standard ResNet init (Kaiming-normal/fan-out for the ReLU
   convs, BN γ=1 β=0, zero-init the last BN of each residual branch, Linear
   ∼ 𝒩(0, 0.01)) instead of the old 2-conv CNN's **Xavier-uniform** default
   (§4). Xavier-uniform on a deep 3×3-conv ReLU stack under-scales early-layer
   signal and would systematically handicap SGD relative to the adaptive
   optimizers — contaminating the very comparison this benchmark exists to make.
   Finance MLP keeps Xavier-uniform / zero-bias (§4) unchanged.
2. **Vision activation**: ReLU (ResNet standard) rather than the old CNN's GELU.
3. **Step budget**: `MAX_STEPS = 10 000` for all 12 tasks, step-based (not
   epoch-based). The old repo's 5 000 steps is far too small for ResNet-18 / a
   ~10 M-param GPT to be an ICLR-credible run.
4. **Vocabulary** (language): fixed cap of 16 000 word types (top-by-frequency
   from train + `<unk>`), byte-level 256 for enwik8. The *rule* is consistent
   across the 4 language tasks (§7); the size necessarily differs by corpus.
   Perplexity is a **within-task** optimizer comparison and is **not** comparable
   to external WikiText/PTB leaderboards (capped vocab, whitespace tokenisation).
5. **Weight tying** (language): token embedding and output projection share
   weights (standard for small LMs; keeps the model near the ~10 M-param target
   and routes a single embedding-type param to Muon's AdamW side).

## Phase-2 sweeps (batch-size + model-size)

12 extra generated files under `benchmark12/sweeps/{batchsize,modelsize}/`, one
per host task (6 of the 12: cifar10, svhn, wikitext2, ptb, tech, healthcare).
Same shared optimizer block + engine; a `frag_sweep_engine.py` grid driver
replaces `main()`. Each sweeps a batch-size or model-size grid at a fixed
10 000-step budget, re-searching LR per grid point, seeds `[42,43,44]`, reusing
the main run's baseline cell for free. Lower priority than the main run — see
`RUNBOOK.md` §7 for the grids, commands, and the "absolute values not comparable
across grid points, only within-grid deltas" caveat.

## Regenerating the files

Edit a fragment in `_build/`, then:

```bash
python benchmark12/_build/build.py --check     # rewrite all 24 + verify blocks
python benchmark12/_build/check_shared_block.py # CI guard (12 main + 12 sweep)
python benchmark12/tests/test_optimizers.py     # optimizer correctness
```

To add **Evie** later: add its class to `frag_shared_optim.py`, add `"Evie"` to
`OPT_NAMES`, add one branch to `build_optimizer()`, set `LR_GRID_CENTRE["Evie"]`,
rebuild. Nothing else in any of the 12 files changes.

### CPU smoke test (no GPU, no real downloads)

```bash
SMOKE=1 ALLOW_CPU=1 python benchmark12/finance/finance_tech.py
SMOKE=1 ALLOW_CPU=1 python benchmark12/language/language_ptb.py
SMOKE=1 ALLOW_CPU=1 python benchmark12/vision/vision_cifar10.py
```

`SMOKE=1` swaps in tiny synthetic data, 1 seed, ~12 steps — it exercises the
full pipeline (LR search → sweep → divergence checks → aggregation → CSV/JSON)
end-to-end in seconds-to-minutes on CPU.
