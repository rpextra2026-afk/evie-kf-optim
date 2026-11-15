# Financials follow-up on the original data and split

Both Kaggle notebooks finished. The four verified worker outputs cover all
960 planned cells without overlap (48 shared tickers, five seeds, four
optimizers). Both notebooks selected identical hyperparameters from seven
learning-rate trials per optimizer. Evie-KF used fixed mean-normalized gamma
30. The dataset retains the original per-ticker 80/10/10 chronological split.

| Optimizer | Completed | Median test MSE among completed runs | Training GPU-hours | Usable in full sweep | Usable within 3 GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW | 240/240 | 0.000253014 | 1.24 | 7 | 7 |
| Evie-KF | 240/240 | 0.000261027 | 5.90 | 5 | 3 |
| Muon | 240/240 | 0.000185156 | 2.84 | 81 | 81 |
| SGD | 238/240 | 0.000185714 | 1.26 | 76 | 76 |

A usable run completes and achieves test MSE at or below the zero-return
forecast for its ticker and dates. The three-hour readout applies the
same retrospective training-time budget and ascending seed and ticker order
as the 2025 follow-up. Failed attempts count toward the budget. Tuning and
final test evaluation are excluded. Evie-KF fit 122 attempts within the
budget. The other optimizers fit all 240 attempts. SGD's two failures were
MKL seed 42 and PGR seed 45. SGD selected the highest LR in its fixed grid.

Muon and Evie-KF completed every run in both follow-ups. Muon led on median
MSE and usable forecasts within the common budget on the original split
(81 vs. 3) and the 2025 split (138 vs. 38). The original benchmark's
completion advantage for tuned Evie-KF did not recur even on the original
data and split. The shared-ticker subset, tuning and failure rules changed
together, so the comparison cannot identify which change drove the result.

[Merged per-cell results](results.json) · [Frozen hyperparameters](lr_search.json) · [2025 follow-up](../finance_financials_2025/REPORT.md)
