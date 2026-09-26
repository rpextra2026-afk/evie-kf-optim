# benchmark12 RUNBOOK — one file at a time, in isolation, no folder

Operational guide for the ICLR optimizer benchmark **when each task file is run
completely on its own**: copied out by itself, pasted into a Kaggle-style
notebook or dropped on a bare box, with **no repo checkout, no sibling files, no
`benchmark12/_build/`, no `benchmark12/tests/`, no `aggregate.py`, no
`scripts/`, no shared `checkpoints/` tree**. Nothing here coordinates with
anything else — no shared state, no shared history.

That constraint is fine: every task file under
`benchmark12/{vision,language,finance,ablation}/` (and `benchmark12/sweeps/`,
`benchmark12/scalegen/`) is **already fully self-contained** — it has zero cross-file imports, embeds the
whole shared optimizer block + training engine + stats pipeline, and
`pip install`s anything it still needs on first run. What you lose by not having
the folder is spelled out where it matters below (mainly: the test suite, the
finance data CSVs, and cross-task aggregation).

**What one file does when you run it:** joint LR (+ second-axis knob) search →
all optimizers for that file → all seeds → the stats → a `RESULTS` table and
`DONE`. One `python <file>.py` per task, start to finish.

**What ships (2026-09-09, `claude_optim.md` + `claudefixes.md`):**
- **Main task files — 8 optimizers:** AdamW, AdaBelief, SGD(+momentum,
  Nesterov, cosine), Sophia, Muon, Lion, **Shampoo**, **EvieKF**
  (`centered=True`, `per_layer_norm=False` — the variant validated in
  `Claude outputs/kaggle_evie_v3.py`). `EvieKFn` is defined but **not
  selected**.
- **EvieKF and Shampoo joint-search a second axis** alongside the LR grid
  (`KNOB_GRID`: γ for EvieKF via `EVIEKF_GAMMA_GRID`, β for Shampoo). Every
  other optimizer is LR-search only.
- **Ablation files — 3 arms, 10 seeds:** `EvieDiag`, `EvieKFu`, `EvieKF`, one
  file per host (`ablation_language_wikitext2`, `ablation_vision_cifar10`,
  `ablation_finance_tech`). C1 = EvieKF vs EvieKFu, C2 = EvieKF vs EvieDiag.

Contents:
1. The execution model (read first)
2. Per-box setup
3. Datasets — what has to travel with the file
4. You cannot run the test suite — what guards a lone file instead
5. Run one file
6. Outputs of one file
7. Collect results across isolated runs (the step that used to be `aggregate.py`)
8. The 3 ablation files
9. Optional: batch-size & model-size sweep files
10. The 2 scale-generalization secondary task files
11. Troubleshooting
12. Appendix: changing an optimizer or a grid needs the full checkout

---

## 1. The execution model

- **One file is the entire unit of work.** You transport a single `.py`, you
  run `python <that file>.py`, you collect that file's output directory. Repeat
  for the next file — on the same box later, or a different box, in any order.
- **Files never talk to each other.** No shared checkpoints, no shared LR
  search, no shared data cache *required* (vision/language will share a
  download cache if they happen to run in the same working directory, but they
  don't depend on it). Running file B does not read or need anything file A
  produced. The only place results ever come back together is a folder **you**
  assemble by hand at the end (§7).
- **The build system and the tests are not present.** Anything in the old
  runbook that said "edit `benchmark12/_build/frag_*.py` and rerun
  `build.py`" — you can't, and you don't need to for a run. If you must change
  a constant, hand-edit it near the top of the one file you have (§10, §11).
- **`CUDA_VISIBLE_DEVICES` still works** if you happen to be on the 3× A6000
  box and want to pin each lone process to its own GPU. On a single-GPU
  notebook it's irrelevant — just run the file.

---

## 2. Per-box setup

Wherever a file is going to run (GPU box, notebook, whatever), you need:

```bash
# Python 3.10+  (3.13 is fine)
python3 -m venv .venv && source .venv/bin/activate   # optional but tidy
pip install --upgrade pip

# PyTorch built for THIS GPU. A6000 = CUDA compute capability 8.6.
# Pick the CUDA build from https://pytorch.org/get-started/locally/. Example:
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

The file `pip install`s `numpy`, `scipy`, `pandas`, `datasets` etc. itself on
first run if they're missing. Installing torch up front just avoids a surprise
mid-run.

**Verify the GPU once, by hand** (this is what each file's `preflight()` checks
anyway):

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), \
torch.cuda.get_device_name(0), torch.cuda.get_device_capability(0))"
# expect:  2.x.x True NVIDIA RTX A6000 (8, 6)

# EvieKF needs float64 eigendecomposition on CUDA -- confirm it works here:
python -c "import torch; torch.linalg.eigh(torch.randn(64,64,device='cuda').double() \
@ torch.randn(64,64,device='cuda').double().T); print('cuda fp64 eigh OK')"
```

If either errors, **stop and fix the torch/CUDA install** — every file refuses
to run otherwise, and the eigh path is hit ~20 steps into the first EvieKF
cell.

---

## 3. Datasets — what has to travel with the file

### Vision (CIFAR-10/100, SVHN, STL-10) — automatic, nothing to carry
The file downloads via `torchvision` on first run into `./checkpoints/_data/`
(override with `export BENCH12_DATA_ROOT=/big/disk/...`). If you run several
vision files in the same directory on a cold cache, stagger their first launch
by a minute — the download isn't locked.

### Language (WikiText-2/103, PTB, enwik8) — automatic, nothing to carry
Via HuggingFace `datasets` on first run, cached under `~/.cache/huggingface`.
enwik8 falls back to `http://mattmahoney.net/dc/enwik8.zip`; PTB to the raw
Mikolov files. wikitext103 is ~500 MB.

### Finance — the ONE thing you must carry alongside the file
The frozen panel CSV is **not embedded in the file**. A finance file that can't
find its panel data stops immediately with a message. You have two ways to give
it the data:

**A. `SMOKE=1` needs nothing** — it synthesises a tiny fake panel (§4). Use
this for the correctness check.

**B. A real run needs the real CSV.** Build the snapshot once on any machine
with internet (laptop is fine):

```bash
pip install yfinance pandas
# you need scripts/download_finance_panels.py from the repo for this one step
python download_finance_panels.py
```

It writes `data/{tech,finance,healthcare,consumer_industrials}.csv` +
`manifest.json`. Sanity-check `manifest.json`: each panel ~40–48
`qualifying_tickers` (of 50 candidates), 8 `lr_search_tickers`, no Energy panel
(intentional). Then copy the panel CSV (and, ideally, `manifest.json` next to
it) to the box and point the file at it:

```bash
export BENCH12_FINANCE_DATA=/abs/path/to/tech.csv     # the panel this file needs
python finance_tech.py
```

The file reads `manifest.json` from the **same directory as the CSV** for the
8-ticker LR-search subset. If `manifest.json` isn't there, the file falls back
to a deterministic `random.Random(seed)` sample of the panel — reproducible,
but not the manifest's exact 8. Carry `manifest.json` if you want the intended
subset; otherwise note in your records that the fallback was used.

---

## 4. You cannot run the test suite — what guards a lone file instead

The old correctness gate (`_build/build.py --check`,
`_build/check_shared_block.py`, `tests/test_optimizers.py`,
`tests/test_eviekf.py`) lives in the folder you don't have. You can't run it,
and that's expected. Three things stand in for it:

1. **Every file self-checks at startup and aborts in one sentence.**
   `preflight()` verifies CUDA is real and at the right compute capability;
   `eviekf_self_check()` (CPU, cheap, runs every launch) verifies the
   dense-vs-factored `(B⊗A)^-1/2` Kronecker parity and the γ=0 identity. If the
   linear-algebra path is wrong on this hardware the file stops **before** any
   training, with a `FATAL: EvieKF self-check failed at startup: ...` line — do
   not trust a run that had to be forced past this.

2. **Run the file once in SMOKE mode first** — minutes, CPU, tiny synthetic
   data, exercises the whole pipeline (LR search → every optimizer → every
   seed → stats):

   ```bash
   SMOKE=1 ALLOW_CPU=1 python finance_tech.py
   SMOKE=1 ALLOW_CPU=1 python vision_cifar10.py
   SMOKE=1 ALLOW_CPU=1 python ablation_finance_tech.py
   ```

   Each must print a `RESULTS` block and `DONE`. If SMOKE fails, the real run
   will fail — fix it before spending GPU time. (`SMOKE=1` on a finance file
   needs no CSV.)

3. **The numerical parity gate against `Claude outputs/kaggle_evie_v3.py`
   already passed before these files were shipped.** That gate
   (`test_eviekf.py::test_parity_vs_kaggle_reference`) is a dev-time check that
   needs the reference file and the test folder. You are running the frozen,
   validated file — you are trusting that gate, not re-running it. The in-file
   `eviekf_self_check()` is what confirms the maths still holds on *your*
   hardware.

---

## 5. Run one file

Nothing here has been run on a GPU. **Run the smallest file first, alone, and
watch it** — a finance panel or `vision_cifar10` is a good canary.

```bash
# on the multi-GPU box, pin it; on a 1-GPU notebook, drop the prefix
CUDA_VISIBLE_DEVICES=0 python finance_tech.py
tail -f checkpoints/finance_tech/log.txt
```

You want to see, in order:
- `[preflight] OK: NVIDIA RTX A6000 sm_86 ...`
- `[preflight] EvieKF self-check OK ...`
- `[lr-search] ... optimizers=['AdamW', ..., 'Shampoo', 'EvieKF']`
- the **first `[lr-search] EvieKF lr=... knob=... score=...` line** — the first
  CUDA float64 eigh. A broken eigh path fails here, cheap.
- `[lr-search] EvieKF: FROZEN lr=... knob=<γ> (LR boundary checks: N, knob
  boundary checks: M)` — in a main run γ is joint-searched over
  `EVIEKF_GAMMA_GRID`; it is only *pinned* to 3000 in the sweep files (§9).
- `[lr-search] frozen: AdamW=..., ..., EvieKF=.../k=<γ>, Shampoo=.../k=<β>`
- `[sweep] 8 optimizers x <units> units x 5 seeds = <total> cells` then
  `[sweep] 1/<total> ...` counting up
- a final `RESULTS -- <task>` table and `DONE`.

If the canary completes clean, the CUDA eigh path, the split-batch noise hook,
memory, and the full pipeline are all proven. Run the rest, one at a time (or
several at once if you're on the 3-GPU box — they don't interact; keep one
heavy run per GPU with `CUDA_VISIBLE_DEVICES`):

```
vision_cifar10.py   vision_cifar100.py   vision_svhn.py   vision_stl10.py
language_wikitext2.py   language_ptb.py   language_wikitext103.py   language_enwik8.py
finance_tech.py   finance_finance.py   finance_healthcare.py   finance_consumer_industrials.py
```

**Runtime budget:** design target is **≤24 h/file on one A6000**. Vision and
finance have headroom. **`language_enwik8` and `language_wikitext103` are the
ones at risk** — the joint LR×γ search for EvieKF (7×7 probes) plus heavier
EvieKF cells (two half-batch backward passes + an eigh every 20 steps) makes a
file ~1.7× the old 6-arm cost. If either projects past ~20 h in its LR-search
phase, see §10.

**Survive an SSH disconnect:**

```bash
tmux new -s bench          # detach: Ctrl-b then d
# or:  nohup env CUDA_VISIBLE_DEVICES=0 python language_enwik8.py &
```

The file also writes `checkpoints/<TASK_NAME>/log.txt` regardless.

**If a run crashes or the box reboots: re-run the identical command in the same
working directory.** It resumes — scored LR/knob grid points are reused,
finished `(optimizer, seed)` cells are skipped, a cell interrupted mid-training
resumes from its last in-cell checkpoint. Nothing finished is recomputed. This
only works if `./checkpoints/<TASK_NAME>/` survived — on an ephemeral notebook,
put the working directory on persistent storage or you lose resume.

---

## 6. Outputs of one file — `./checkpoints/<TASK_NAME>/`

Relative to the working directory you launched from (override the root with
`BENCH12_CKPT_ROOT`). This whole directory is what you carry back off the box.

```
lr_search.json     frozen lr AND knob per optimizer (+ the full joint grid)
results.json       every cell: primary metric, diverged flag, timings,
                   EvieKF telemetry (cos, b_simple, eff_rank, gain spectrum,
                   n_active_steps), sec_per_step, ...
results_raw.csv    the same, flat, for spreadsheets
summary.json       ONE number per optimizer (median over seeds) <- this is the
                   file the cross-task stats need
log.txt            the full run log
cells/*.pt         per-cell checkpoints (safe to delete once summary.json exists)
```

---

## 7. Collect results across isolated runs

There is no shared tree and no `aggregate.py` on the run boxes, so **you**
assemble the final picture:

1. From each finished run, pull its `checkpoints/<TASK_NAME>/` directory
   (at minimum `summary.json`, plus `results_raw.csv` and `lr_search.json` for
   the paper) back to one machine, into a layout like
   `collected/<TASK_NAME>/summary.json`.

2. To run the pre-registered aggregation, bring `benchmark12/aggregate.py` from
   the repo to that machine and point it at the collected tree
   (it reads every `checkpoints/<task>/summary.json`, 12 main tasks; ablation
   summaries are skipped):

   ```bash
   # from a dir where ./checkpoints/ is your collected tree
   python aggregate.py                 # reference = AdamW
   python aggregate.py --ref EvieKF    # reference = EvieKF (the headline table)
   ```

   It runs: one number per (task, optimizer); paired **Wilcoxon signed-rank
   across the 12 tasks** vs the reference; exact **sign test** as a robustness
   check; **Holm-Bonferroni** across the optimizer family. Works with ≥ ~6
   tasks present. Writes `checkpoints/aggregate_12task.json`.

3. If you can't bring `aggregate.py`, the analysis is reproducible by hand from
   the 12 `summary.json` files: take each task's one-number-per-optimizer,
   then paired Wilcoxon signed-rank (+ sign test, + Holm across the family) of
   the 12 differences vs the reference optimizer.

**For the paper:**
- per main task: `<TASK_NAME>/summary.json`, `results_raw.csv`
- headline: `aggregate_12task.json` + the `--ref EvieKF` printed table
- per ablation task: `ablation_<host>/summary.json` (the C1/C2 rows, §8)
- frozen `(lr, knob)`: each `*/lr_search.json`

---

## 8. The 3 ablation files

Run them the same way as any other file — after the main tasks, or at least
after their 3 hosts. They are standard single-task files with
`opt_set="ablation"`: **10 seeds** (`[42..51]`), arms `EvieDiag / EvieKFu /
EvieKF`, each arm its **own** independent joint `(lr, γ)` search. Cost per file
≈ one main language file. `ablation_finance_tech.py` needs the `tech` panel CSV
via `BENCH12_FINANCE_DATA` (§3), same as `finance_tech.py`.

```bash
python ablation_language_wikitext2.py
python ablation_vision_cifar10.py
BENCH12_FINANCE_DATA=/abs/path/to/tech.csv python ablation_finance_tech.py
```

- Own output dir: `checkpoints/ablation_<host>/` — never collides with the main
  run's `checkpoints/<host>/`, resumable the same way.
- `EvieKF` is computed here at 10 seeds **and** in the main table at 5 seeds.
  Known, accepted overlap — not de-duplicated.
- Read each result from its `RESULTS -- ablation_<host>` block and
  `checkpoints/ablation_<host>/summary.json`:
  - **C1 (novelty claim):** EvieKF vs `EvieKFu` — does centring the noise
    covariance help?
  - **C2 (mechanism claim):** EvieKF vs `EvieDiag` — does the Kronecker
    structure beat a diagonal gate?
  - With only 3 tasks there is no cross-task significance test; the read is the
    per-task effect size and whether the direction is consistent across the 3.

---

## 9. Optional: batch-size & model-size sweep files (Phase 2)

**Lower priority than the main run + ablation.** 12 generated files in
`benchmark12/sweeps/{batchsize,modelsize}/`, one per host task (6 of the 12:
`vision_cifar10`, `vision_svhn`, `language_wikitext2`, `language_ptb`,
`finance_tech`, `finance_healthcare`). Each is self-contained the same way —
run it on its own with `python <file>.py`, finance ones need
`BENCH12_FINANCE_DATA`.

- Own output dirs: `checkpoints/<task>_batchsize/` or
  `checkpoints/<task>_modelsize/`. Seeds `[42, 43, 44]`.
- **The baseline grid point is normally "reused for free" from the main host
  run's `checkpoints/<host_task>/`.** In one-file isolation that directory
  isn't present, so the baseline column is recomputed (logged, not an error) —
  or, if you *do* place the main run's `checkpoints/<host_task>/` next to the
  sweep file's working dir, it's reused as designed.
- **EvieKF γ in the sweeps is FIXED at `EVIEKF_SWEEP_GAMMA` (= 3000)**, not
  joint-searched. LR is re-searched per `(grid point, arm)`.
- **Fixed 10 000-step budget at every grid point** (not samples-seen-matched) —
  absolute metrics are **not** comparable across grid points, only the
  within-grid-point arm delta. Written to each `summary.json` (`c1_row`,
  `eviekf_sweep_gamma`, `spearman_batch_vs_delta`).
- **Reference:** batch-size sweep → `EvieKF` (so the `EvieKFu` delta row is the
  pre-registered C1 quantity: `Spearman(batch size, normalised EvieKF−EvieKFu)`;
  `rho < 0` ⇒ the centred edge grows as batch shrinks). Model-size sweep →
  `AdamW`.

Priority if GPU time is short: **batch-size sweep wins over model-size** (it
backs the paper's falsifiable prediction).

---

## 10. The 2 scale-generalization secondary task files

`evie-scalegen-spec.md` — one genuinely harder task per domain (vision/
language only), answering "does the edge hold at real scale, or only in the
CIFAR/tiny-GPT regime." **Not** added to `TASKS`/`ABLATION_TASKS`/
`SWEEP_TASKS` and **not** part of the 12-task Wilcoxon/Holm significance
pipeline — own `opt_set="scalegen"`, own output directory
(`benchmark12/scalegen/`), reported descriptively only (§7 of the spec: no
p-value at n=3 seeds).

| task_name | dataset | model | own output dir |
|---|---|---|---|
| `vision_tinyimagenet` | Tiny-ImageNet-200, native 64x64, 200 classes | Same CIFAR-stem ResNet-18 family, +1 maxpool for the 64x64 stem | `checkpoints/vision_tinyimagenet/` |
| `language_openwebtext` | OpenWebText slice, ~220M train tokens, GPT-2 BPE (vocab 50257) | Scaled-up GPT: 8L/d384/8h/ctx512 (~33M params) | `checkpoints/language_openwebtext/` |

- **Trimmed arm/seed set:** `AdamW, AdaBelief, Muon, EvieKF` x **3 seeds**
  (`[42, 43, 44]`) — not the full 8-arm/5-seed main-run protocol.
- Run exactly like any other file: `python vision_tinyimagenet.py` /
  `python language_openwebtext.py`. Same checkpoint/resume, divergence
  detection, preflight and LR-search machinery as every other generated
  file (they go through the same `_build/` pipeline) — tighter
  `ckpt_every` (500 / 300 vs. the other tasks' 2000) since these are the
  two most expensive, highest-stakes cells to lose progress on.
- **Data auto-downloads and caches on first run, same as WikiText-103/
  enwik8/PTB** — nothing to carry manually:
  - `vision_tinyimagenet`: the cs231n `tiny-imagenet-200.zip` (~237MB),
    falling back to the HF mirror `zh-plus/tiny-imagenet` if that source is
    unreachable; decoded once and cached as a single `.npz` under
    `{BENCH12_DATA_ROOT}/tinyimagenet/`. Integrity-checked on every load
    (100,000 train / 10,000 val / 200 classes) — a corrupt/truncated cache
    is rebuilt, never silently trusted.
  - `language_openwebtext`: streamed from `Skylion007/openwebtext` via HF
    **streaming mode** (never downloads the full ~40GB corpus), tokenized
    with `tiktoken`'s GPT-2 BPE, cached as nanoGPT-style `.bin` files +
    `meta.json` under `{BENCH12_DATA_ROOT}/openwebtext_gpt2bpe/`.
    Token-count + sha256 integrity-checked on every load.
- **Kaggle notebooks:** `kaggle/notebooks/vision_tinyimagenet.ipynb` and
  `kaggle/notebooks/language_openwebtext.ipynb`, generated the same way as
  every other notebook in that folder (`kaggle/notebooks/build_notebook.py
  --task benchmark12/scalegen/<file>.py`, no `--panel`) — same
  self-imposed wall-clock budget / `--ckpt-in-auto` resume-across-sessions
  mechanics as the other 12 (see `kaggle/README.md`).
- **Do not compute or report a p-value from these 3-seed cells** (spec §7).
  Report per-arm mean ± seed spread, win-count out of 3 vs. each baseline,
  and the standard mechanism diagnostics the engine already logs for free.
  `aggregate.py` does not read these two tasks' `summary.json` — collect
  and report them separately, by hand, alongside the 12-task table.

---

## 11. Troubleshooting

| Symptom | Meaning / action |
|---|---|
| `FATAL: CUDA is not available ...` | No GPU visible — wrong machine, or CPU-only torch. Reinstall the CUDA build (§2). For a CPU correctness smoke only: prefix `ALLOW_CPU=1`. |
| `FATAL: CUDA is present ... trivial matmul failed` | torch built for a different arch. Reinstall for compute capability 8.6. |
| `FATAL: EvieKF self-check failed at startup: ...` | The Kronecker maths is wrong on this box (bad eigh / precision). Do **not** trust any run. Re-check the `torch.linalg.eigh` fp64 CUDA probe in §2. |
| `FATAL: frozen panel CSV not found ...` | Finance file with no data. Set `BENCH12_FINANCE_DATA` to the panel CSV, or run with `SMOKE=1` (§3). |
| `[data] ... manifest.json` absent / LR-search tickers via fallback | `manifest.json` wasn't next to the CSV. Deterministic fallback subset used — reproducible, just not the manifest's 8. Carry `manifest.json` if you need the intended subset. |
| `[lr-search] <opt>: ... at grid edge -> extend ...` | **Not an error.** The boundary rule extended the LR or knob grid and re-searched. It logs how many times (`LR boundary checks`, `knob boundary checks`). |
| `[lr-search] EvieKF ... knob boundary checks: 3` and still edge-pinned | The true best γ is outside `[3 … 3000] × 3^±2`. Hand-edit `EVIEKF_GAMMA_GRID` near the top of this file (see below), or accept the edge value. |
| `[diverge] <opt>/<unit>/seed<n>: ...` | One cell hit a NaN or an absurd metric. Marked failed in `results.json`; the sweep continues. Check `fail_reason`. Occasional, expected. |
| `[diverge] ... n_active_steps=0 ... silently AdamW` | The EvieKF split-batch noise hook never fired for that cell — a wiring fault, treated like a divergence. Should never happen in a normal run; if it does, stop and report it. |
| `[sweep] ENVIRONMENT error ...` then `FATAL` | A CUDA/CUBLAS/cuDNN/cuSOLVER/OOM error — treated as an environment fault, not a diverged run. Aborts on purpose. Finished cells are safe; fix the environment, re-run the same command. |
| Run died (crash / reboot / preemption) | Re-run the identical command in the same working directory. Resumes from the last checkpoint — if `./checkpoints/<TASK_NAME>/` survived. |
| `language_enwik8` / `language_wikitext103` LR search projecting past ~20 h | You have only this one file, so there's no fragment to rebuild. Hand-edit near the top of the file: `EVIEKF_GAMMA_GRID = [10.0, 100.0, 1000.0]` (3 points instead of 7). The file's header says "do not edit" because normally it's regenerated — with no `_build/` present, the in-file edit is the only lever. Or just accept the longer runtime / kill it. |
| Disk filling up | Delete `checkpoints/<TASK>/cells/*.pt` once that task's `summary.json` exists. Move the dataset cache with `BENCH12_DATA_ROOT`. |
| A task looks stuck | `tail -f checkpoints/<TASK_NAME>/log.txt`. Language and finance files are legitimately long. Truly hung with no log progress for hours → kill and re-run (it resumes). |
| Fast check before spending GPU time | `SMOKE=1 ALLOW_CPU=1 python <file>.py` — tiny synthetic data, ~minutes, exercises the full pipeline; must end with `RESULTS` + `DONE`. |

---

## 12. Appendix: changing an optimizer or a grid needs the full checkout

The single files are **generated**. Adding or changing an optimizer, or making
a grid change stick across all files, is done in a full repo checkout by
editing `benchmark12/_build/frag_shared_optim.py` (the "SHARED OPTIMIZER
BLOCK") and running `python benchmark12/_build/build.py`, which pastes the
change identically into all 27 files. You then re-export the individual file(s)
you need. None of that is possible from a lone file.

The only in-place change that's safe on a single generated file is tweaking a
top-of-file scalar constant for *that run only* — e.g. `EVIEKF_GAMMA_GRID`,
`SHAMPOO_KNOB_GRID`, `LR_GRID_CENTRE[...]` — understanding it will be
overwritten the next time the file is regenerated from the fragments upstream.
Never treat an in-file edit as the source of truth; fold it back into
`frag_shared_optim.py` in the repo.

For the full optimizer-registration checklist (`OPT_NAMES`, `LR_GRID_CENTRE`,
`HESSIAN_CONSUMERS`, `NOISE_CONSUMERS`, `KNOB_GRID`, `build_optimizer()`), work
from a checkout and see `benchmark12/README.md` / `CLAUDE.md` §2.
