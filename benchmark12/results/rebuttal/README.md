# Rebuttal experiments

Each directory in [merged/](merged/) contains the final per-cell results and
frozen search choices for one task. A cell is an optimizer, data unit, and seed.
The records retain numerical failures and include validation traces and
available timing and memory measurements. Cluster run logs are included with
local filesystem paths redacted. Merge provenance uses numbered shard labels.

| Experiment | Records |
| --- | --- |
| Fixed mean-normalized γ = 30 at the baseline LR budget, with same-platform AdamW references | `merged/<task>_evie_m30/` and `merged/<task>_ref_adamw/` for all 12 main tasks |
| Longer CIFAR training, including the fresh-epoch loader correction | [CIFAR-10](merged/vision_cifar10_long_fresh/) and [CIFAR-100](merged/vision_cifar100_long_fresh/). Original-loader comparisons are in `merged/vision_cifar{10,100}_long/`. |
| Fresh-epoch language check | [WikiText-2](merged/language_wikitext2_fresh/) |
| Expanded baseline tuning budget | [WikiText-2, 49 trials per baseline](merged/language_wikitext2_basebudget/) |
| Lagged and random-basis controls | [WikiText-2](merged/ablation_language_wikitext2_controls/), [Finance-Tech](merged/ablation_finance_tech_controls/), [CIFAR-100 random basis](merged/vision_cifar100_evie_m30_rb/) |
| Muon hybrid | `merged/language_{wikitext2,wikitext103,enwik8,openwebtext}_muonhybrid/` |
| 124M GPT-2-small on OpenWebText | [Original comparison](merged/language_openwebtext_124m/), [fixed γ = 30](merged/language_openwebtext_124m_evie_m30/), [fixed γ with factor cap 3072](merged/language_openwebtext_124m_evie_m30_cap3072/) |
| Synthetic correlated-noise quadratic | [Per-seed results](noisy_quadratic.json) |
| Financials under the matched-budget follow-up protocol | [Original snapshot and split](merged/finance_financials_original_split/REPORT.md), [2025 held-out window](merged/finance_financials_2025/REPORT.md) |
| Controlled noise-geometry comparisons | [Initial pilot](noise_geometry_pilot/summary.csv), [fixed-deadline follow-up](noise_geometry_equal_time/REPORT.md) |

### Financials and noise-geometry follow-ups

The Financials reports use the same retrospective three-training-GPU-hour
budget per optimizer. A usable run reaches test MSE at or below the
zero-return forecast on the same ticker and dates. The 32 Muon cells flagged
in the original benchmark all finished 10,000 steps. They were flagged
because their test MSE exceeded 50 times the task peer median.

Run either of the following from the repository root with a CUDA-enabled
environment. Data and manifests are under `benchmark12/finance/data/followups/`.
Each task searches on its own validation split and evaluates on its own test
split. Set `BENCH12_CKPT_ROOT` to change where checkpoints and results are written.

```sh
python benchmark12/diagnostics/finance_financials_original_split.py
python benchmark12/diagnostics/finance_financials_2025.py
python benchmark12/tools/financials_holdout_report.py checkpoints/finance_financials_original_split/results.json
python benchmark12/tools/financials_holdout_report.py checkpoints/finance_financials_2025/results.json
```

Use the [initial pilot notebook](../../../kaggle/jobs/noise_geometry_pilot/noise_geometry_pilot.ipynb)
and [fixed-deadline notebook](../../../kaggle/jobs/noise_geometry_equal_time/noise_geometry_equal_time.ipynb)
to reproduce the synthetic comparisons on CPU. Both embed their optimizer
source and synthetic inputs.

## Recompute analyses

From the repository root, install NumPy and run the commands below. The analysis
commands write Markdown and JSON into this directory. The archived outputs
include [paired intervals with and without seed 42](seed42.md),
[finance ticker sensitivity](finance_filter.md), [tuning budget sensitivity](budget.md),
and [multiple-comparison resolution](holm_limit.md).

```sh
python benchmark12/tools/rebuttal_analysis.py selftest
python benchmark12/tools/rebuttal_analysis.py seed42
python benchmark12/tools/rebuttal_analysis.py finance_filter
python benchmark12/tools/rebuttal_analysis.py budget
python benchmark12/tools/rebuttal_analysis.py holm_limit
python benchmark12/tools/rebuttal_analysis.py r3a --roots benchmark12/results/rebuttal/merged
python benchmark12/tools/rebuttal_analysis.py r4 --host language_wikitext2 --roots benchmark12/results/rebuttal/merged
python benchmark12/tools/rebuttal_analysis.py r4 --host finance_tech --roots benchmark12/results/rebuttal/merged
python benchmark12/tools/noisy_quadratic.py
```

The `seed42` and `r3a` commands use paired observations and aggregate
finance results by taking the seed median for each ticker before the panel
median. The paper's task statistic takes the ticker median for each seed on
its complete-case panel before the seed median. The two aggregation rules
can produce different task values and test statistics. The `r3a` command
excludes tuning seed 42.

## Rerun training

Use the [benchmark runbook](../../RUNBOOK.md) for data and dependencies.
Rebuttal tasks live in [diagnostics/](../../diagnostics/) and
[muonhybrid/](../../muonhybrid/). Their filenames match the task names in
`merged/`. Run the task script for both search and training. To reproduce the
evaluation at the archived hyperparameters, copy its `lr_search.json` into
`<checkpoint-root>/<task>/`, point `BENCH12_CKPT_ROOT` at that checkpoint
root, and set `BENCH12_STAGE=sweep`.

For example, in a POSIX shell with the OpenWebText cache configured:

```sh
task=language_openwebtext_124m_evie_m30_cap3072
mkdir -p "checkpoints/$task"
cp "benchmark12/results/rebuttal/merged/$task/lr_search.json" "checkpoints/$task/"
BENCH12_CKPT_ROOT=checkpoints BENCH12_STAGE=sweep python "benchmark12/diagnostics/$task.py"
```

The 124M and OpenWebText hybrid records were produced on an H100 MIG slice.
The other rebuttal training sweeps used T4 GPUs. The two corrected CIFAR tasks
and the WikiText-2 loader check explicitly enable fresh epochs. The original
protocol replays first-epoch batches.
