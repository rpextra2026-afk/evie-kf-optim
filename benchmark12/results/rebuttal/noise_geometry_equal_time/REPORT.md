# Fixed-deadline noise-geometry pilot

All 180 evaluation runs completed. Each optimizer was tuned for the same
0.45-second deadline on three separate seeds per regime. Evaluation used 12
paired seeds per regime. Actual elapsed times ranged from 0.450001 to
0.450913 seconds, and none of the selected learning rates were at grid edges.

| Noise covariance | AdamW median loss | Muon median loss | Evie-KF median loss | Evie diagonal median loss | Evie-KF wins vs Muon |
| --- | ---: | ---: | ---: | ---: | ---: |
| Coordinate-aligned | 0.052713 | 0.137920 | 0.087108 | 0.075880 | 12/12 |
| Intermediate rotation | 0.104354 | 0.177688 | 0.222821 | 0.109670 | 1/12 |
| Largest rotation | 0.115529 | 0.210282 | 0.288611 | 0.168519 | 1/12 |

Median paired Evie-KF/Muon loss ratios were 0.642, 1.270, and 1.381. Evie-KF
also beat its random-basis control on 12/12, 9/12, and 8/12 seeds respectively.
AdamW and the diagonal control each beat full Evie-KF on all 12 seeds in
each regime. The coordinate-aligned result is a repeatable synthetic
pairwise win over Muon. The stronger-rotation settings go in the opposite
direction from the proposed prediction. This pilot varies the covariance's
alignment with the fixed objective as well as its off-diagonal structure.
The deadline measures the single-thread CPU update loop on this quadratic.

[Per-run and tuning records](results.json) · [Machine-readable summary](summary.csv)
