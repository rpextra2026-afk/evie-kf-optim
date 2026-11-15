# Financials forecasting with a 2025 holdout

Both Kaggle notebooks finished. All four worker archives passed integrity and
coverage checks. Their disjoint assignments account for all 960 planned cells
(48 tickers, five seeds and four optimizers). The validation searches used
seven learning rates per optimizer and selected identical settings across the
two notebooks. Evie-KF used fixed mean-normalized gamma 30.

| Optimizer | Runs completed | Median test MSE among completed runs | Training GPU-hours | Usable in full sweep | Usable within 3 GPU-hours |
| --- | ---: | ---: | ---: | ---: | ---: |
| AdamW | 240/240 | 0.000285153 | 1.29 | 96 | 96 |
| Evie-KF | 240/240 | 0.000286625 | 6.06 | 66 | 38 |
| Muon | 240/240 | 0.000277718 | 2.88 | 138 | 138 |
| SGD | 238/240 | 0.000280815 | 1.28 | 127 | 127 |

Usable means a completed run with test MSE at or below a zero-return forecast
on the same ticker and dates. The three-hour result is a retrospective readout
in ascending seed and ticker order using recorded training time. It charges
failed attempts and excludes tuning and final test evaluation. Evie-KF fit
117 attempts within the budget. Each other optimizer fit all 240 attempts.
SGD's two failures were HBAN seed 45 and HIG seed 44.

Muon and Evie-KF both completed every held-out run. Muon had the lowest median
MSE and produced the most usable forecasts under the common budget. The
completion advantage observed for tuned Evie-KF in the original Financials
sweep did not recur in this follow-up. The follow-up uses a new data split,
48 of the original 49 tickers and a matched LR search with fixed gamma for
Evie-KF. The experiment therefore evaluates the proposed follow-up protocol
as a whole.

[Merged per-cell results](results.json) · [Frozen hyperparameters](lr_search.json)
