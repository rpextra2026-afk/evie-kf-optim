# Gradient-noise geometry pilot

Import [the notebook](noise_geometry_pilot.ipynb) into Kaggle. Leave the
accelerator set to **None (CPU)** and choose **Save & Run All**. The notebook
embeds [the frozen synthetic design](noise_geometry_data.json) and verifies its
SHA-256. No dataset attachment, Git checkout, or internet is needed. The
environment needs PyTorch, NumPy, and pandas, which the standard Kaggle image
provides.

Download `noise_geometry_results/results.json` for the tuning trials and
per-seed loss curves, and `noise_geometry_results/summary.csv` for the table.

The three settings rotate the covariance eigenvectors while preserving its
eigenvalues and total variance. The objective and Gaussian draws are shared
across settings and optimizer arms. This changes covariance alignment with
both the parameter coordinates and the objective's fixed curvature.

- Each arm searches the same seven learning rates per setting on two tuning
  seeds. The five arms are AdamW, Muon, Evie-KF, its diagonal gate, and its
  random-basis gate. Evie variants use fixed mean-normalized γ = 30.
- Final loss is measured at 1,200 steps on eight separate paired seeds.
  Equal-time loss uses the last recorded checkpoint within the fastest
  completed arm's elapsed time on each seed. Timing covers the synthetic
  update loop on one CPU thread.
- Read paired ratios to Muon across the three settings. A ratio below one
  favors the corresponding arm. Check completions and boundary LR picks in
  the JSON before interpreting the comparison.

This measures a synthetic matrix quadratic with precomputed half-gradient
noise. Time includes the optimizer and synthetic gradient construction.
Model training time would also include forward and backward passes.

To rebuild from the working repository, run
`python kaggle/tools/build_noise_geometry_pilot.py`.
