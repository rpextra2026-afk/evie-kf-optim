# Fixed-deadline follow-up

Import [the notebook](noise_geometry_equal_time.ipynb) on Kaggle. Set
**Accelerator → None (CPU)** and choose **Save & Run All**. The notebook embeds
and hashes [the synthetic design](noise_geometry_equal_time_data.json).
Download `noise_geometry_equal_time_results/results.json` and
`noise_geometry_equal_time_results/summary.csv`.

## Why this follow-up

In the original fixed-step-tuned pilot, Evie-KF beat Muon at the shared
checkpoint horizon in 23 of 24 seed–regime comparisons. The median paired loss
ratios to Muon were 0.264, 0.444, and 0.445 as off-diagonal structure increased.
AdamW and the diagonal control had lower median matched-time loss than full
Evie-KF in all three regimes. Evie-KF's relative advantage over Muon was
largest in the coordinate-aligned case. This evidence supports a synthetic
pairwise timing result and leaves the Kronecker mechanism claim open.

## Experimental design

The covariance spectra, rotation settings, curvature and fixed gamma 30
carry over from the pilot. Initializations and split noise draws are matched
within each seed and rotation. This follow-up tunes every optimizer at the
same fixed 0.45-second elapsed deadline with the same nine LR trials and three
new tuning seeds per regime. Evaluation uses 12 different paired seeds.
The output records boundary LR choices, all per-seed losses, elapsed times,
completed steps and paired wins against Muon. All regimes are retained.

The timer covers the CPU update loop using precomputed synthetic inputs, and
can overshoot by at most one finished update. Setup, noise generation,
terminal evaluation and tuning are outside the per-run deadline. Compare
the actual elapsed times before interpreting a small gap. CPU results are
specific to this synthetic workload and runtime environment.

Rebuild with `python kaggle/tools/build_noise_geometry_equal_time.py`.
