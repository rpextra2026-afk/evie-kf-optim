# Diagnostics — pinned one-question re-runs

Each file here re-runs **one arm** of an already-finished task with its
hyper-parameters **stated instead of searched**, to answer one question. They are
descriptive: a diagnostic number **never replaces a protocol number** in Table 1,
in `CONFIRMATORY.md`, or in the 12-task Wilcoxon. Own `task_name`, own
`checkpoints/<task_name>/`, own results folder
(`benchmark12/results/diagnostics/<task_name>/`), `opt_set="diag"` — they cannot
touch the finished runs' checkpoints or be swept up by a directory glob.

Generated, like everything else, by `python benchmark12/_build/build.py`. Never
hand-edit a file here. Notebooks: `kaggle/notebooks/<task_name>.ipynb`, built by
`kaggle/notebooks/build_notebook.py`, run with Save & Run All.

---

## The B and C batch (built 2026-09-23, none run yet)

Priority order. Each row is one notebook, one Kaggle account, GPU + Internet on.

| # | notebook | question | est. GPU time | sessions |
|---|---|---|---|---|
| B1a | `language_ptb_evie_fixed_gamma` | does EvieKF do better on PTB with a default γ than with the one its joint search picked? | ~6.0 h | 1 |
| B1b | `vision_stl10_evie_fixed_gamma` | same, STL-10 (search froze γ = 3, operator nearly off) | ~3.4 h | 1 |
| B1c | `vision_svhn_evie_fixed_gamma` | same, SVHN (γ = 10, 7th of 8) | ~3.6 h | 1 |
| B1d | `vision_cifar10_evie_fixed_gamma` | same, CIFAR-10 (γ = 27,000, a boundary-extended pick) | ~3.3 h | 1 |
| B2 | `language_enwik8_evie_gamma81k` | what did the resume bug's wrong γ cost? | ~5.0 h | 1 |
| B3 | `language_wikitext2_large_evie_at_adamw_lr` | is the 17.6% loss on the large model real, or the lr? | ~5.4 h | 1 |
| C1 | `vision_cifar10_evie_maxf4608` | is vision flat because most of ResNet-18 runs half-diagonal? | ~12–35 h (uncertain) | 2–4 |

Estimates are the finished runs' own median EvieKF wall-clock per cell
(`results.json`), plus, for B1, the 7 search probes at 25% of a cell each. C1 is
the one real unknown: a 4608×4608 float64 eigendecomposition every 20 steps,
where the main run's 1024-capped cells took 1,765 s.

---

## B1 — EvieKF at the baselines' tuning budget

`<task>_evie_fixed_gamma`, 4 files, 5 seeds each.

γ is **fixed at 3000** (`EVIEKF_SWEEP_GAMMA`, the value the sweeps already use)
and the learning rate is searched over the **ordinary 7-point half-decade grid**
with the usual boundary rule. So this arm spends exactly what AdamW spends: 7
search points, not the protocol's joint 49–99.

Three things at once:

1. **Answers the tuning-budget objection.** The first reviewer question about
   EvieKF is that it searched two axes while the baselines searched one. This row
   is the comparison at an equal budget.
2. **Gives a default.** "Use γ = 3000, tune the learning rate as you already do"
   is what a practitioner needs; the paper cannot ship a method whose knob must
   be grid-searched per task.
3. **Tests whether searching γ is even worth it.** On these four tasks the γ row
   at the chosen lr is non-monotone with a large spread — an argmax over noise.
   If a fixed γ matches or beats the searched one, the joint search is costing
   compute and buying nothing.

Hosts are the four tasks where the γ pick looks like noise: PTB (γ 10, 6th of 8),
STL-10 (γ 3, i.e. the operator nearly off, 6th of 8), SVHN (γ 10, 7th of 8),
CIFAR-10 (γ 27,000, a boundary-extended pick).

**Reading the result.** Compare against the *same task's* protocol EvieKF and
AdamW cells, which already exist — nothing needs re-running for the comparison.
Better at a fixed γ on most of the four ⇒ report the fixed-γ row as the headline
configuration and the searched-γ row as the protocol's, and say plainly that the
extra search did not help. Worse ⇒ the joint search is doing real work and the
budget disclosure stays as it is. Either answer is reportable.

**Caveat.** These runs are *not* a substitute for the 12-task test: four tasks
re-tuned after seeing the results are a selected subset, and that must be stated
wherever the row appears.

Mechanism: `TASK["fixed_knob"]` (engine), tested in
`benchmark12/tests/test_diagnostic_knobs.py`.

## B2 — enwik8 at γ = 81,000

`language_enwik8_evie_gamma81k`, both axes pinned (lr 3.162e-4), 5 seeds.

`FUTURE_WORK.md` FW-2: the grid-reset-on-resume bug made the file freeze
γ = 27,000 even though γ = 81,000 at the same lr had **already been scored** and
was 0.40% better (3.6956 vs 3.7103 in `lr_search.json`). Nothing is re-tuned
here; both values come from the file's own search. This measures only what the
bug cost. A small effect is the expected outcome — it removes a disclosure, and
if the effect is large, the disclosure becomes a correction instead.

## B3 — WikiText-2 large model at AdamW's learning rate

`language_wikitext2_large_evie_at_adamw_lr`, 9L/d384/12h, lr 3.162e-4, γ 3000,
seeds 42–44 (the sweep's seeds). This is `FUTURE_WORK.md` FW-6.

In the model-size sweep EvieKF froze lr 1e-3 while AdamW and AdaBelief both froze
3.162e-4, and lost by 17.6%; its own search scored those two lrs 108.8 vs 110.4,
1.5% apart. Near AdamW here ⇒ the sweep reads "wins small, lr-confounded large";
still 17% behind ⇒ it is a real loss at capacity.

Runs on the **main** engine, not the sweep engine (the model shape comes from
`n_layer`/`d_model`/`n_head` in `TASK`), so the sweep engine's ticker-mean
aggregation issue does not apply — and with one unit it could not anyway.

## C1 — raising the Kronecker size limit on CIFAR-10

`vision_cifar10_evie_maxf4608`, lr 0.09999 and γ 27,000 pinned at the main run's
frozen values, 5 seeds, `eviekf_maxf = 4608`.

`EVIEKF_MAXF = 1024` sends any Kronecker factor larger than that to a **diagonal
approximation**. A CIFAR-stem ResNet-18's conv weights reshape to
(out, in·k·k), so the input-side factors are:

| block | factor | at the frozen limit |
|---|---|---|
| layer1 | 576 | full Kronecker |
| layer2 | 1,152 | diagonal |
| layer3 | 2,304 | diagonal |
| layer4 | 4,608 | diagonal |

Layers 2–4 hold nearly all of ResNet-18's parameters, so **on vision EvieKF has
been running half-diagonal over most of the network.** The language models
(d256, 1024-wide MLPs) stay fully Kronecker apart from the vocabulary side of the
embedding, and the finance MLP is tiny. That lines up with the results: the
Kronecker structure was significant in language and finance, vision is where
EvieKF looks like Adam-plus-a-bit, and the vision ablation could not tell.

The only difference from the main run's CIFAR-10 EvieKF cells is the limit — same
lr, same γ, same seeds — so **those cells are the control** and nothing needs
re-running. Vision improves ⇒ a strong mechanism result, and the 1024 limit
explains the domain gap. No change ⇒ the limit is ruled out and vision's flatness
needs another explanation.

**The 1024 limit belongs in the paper either way**, run or not.

---

## Already run

| folder | result |
|---|---|
| `language_ptb_evie_at_adamw_lr` | EvieKF at AdamW's lr on PTB: 184.66 vs 192.17 protocol (AdamW 179.35) — half the gap was the lr choice |

## When a result comes back

Same routine as any other run — `RESULT_INSTRUCTIONS.md` §6, with the folder
`benchmark12/results/diagnostics/<task_name>/`. Do **not** run
`build_paper_tables.py` for a diagnostic; it reads only the 12 main task folders,
which is the point.
