# WikiText-2 checkpoint probe, seed 17101

The 10,000-step training run completed on a Tesla T4 without a reported training failure. All six planned checkpoint probes completed. Source files in the downloaded archive match the locally built runner, probe and task byte for byte. The gamma-zero replay matched the AdamW algebra at every checkpoint. TF32 was disabled.

Each checkpoint uses 24 direction/donor batch pairs and 64 independent training-distribution probe batches. The compared directions have equal length under the fixed pre-step Adam whitening map. The ungated direction is evaluated at the same Evie-KF model and moment state.

| Next update | Gate phase | Variance ratio, Evie / ungated (95% CI) | Variance reduction | RMS step change from independent gate input | Projected mean shift (95% CI), relative to baseline step |
| --- | --- | --- | --- | --- | --- |
| 2000 | eigensystem_refresh | 0.366 (0.311, 0.433) | 63.4% | 4.0295% | 0.260585% (0.234646%, 0.286525%) |
| 2001 | cached_eigensystem | 0.447 (0.371, 0.535) | 55.3% | 0.0383% | 0.000010% (-0.000018%, 0.000039%) |
| 5000 | eigensystem_refresh | 0.434 (0.371, 0.515) | 56.6% | 4.6015% | 0.316931% (0.287687%, 0.346175%) |
| 5001 | cached_eigensystem | 0.543 (0.475, 0.630) | 45.7% | 0.0531% | 0.000015% (-0.000066%, 0.000096%) |
| 8000 | eigensystem_refresh | 0.524 (0.462, 0.611) | 47.6% | 6.0039% | 0.466340% (0.436079%, 0.496602%) |
| 8001 | cached_eigensystem | 0.650 (0.545, 0.806) | 35.0% | 0.0795% | 0.000012% (-0.000097%, 0.000121%) |

The table uses the crossfit-whitened-noise control. The clipped-delta swap gives nearly identical results. Every one of the 24 measured directions at every checkpoint had a lower probe variance than its ungated partner. All six pointwise bootstrap intervals lie below one.

At the three sampled refresh steps, the mean shift projected along the frozen Adam direction is positive with a pointwise interval excluding zero. The estimated projected shift is 0.26–0.47% of the baseline step length. Individual-step RMS changes are 4.0–6.0%. These are different quantities. The projected mean-shift intervals exclude zero at each sampled refresh step, giving pointwise evidence of a nonzero coupling effect. The projection does not bound the total mean shift in all directions. The measured effects do not establish whether coupling helps or harms training.

At the three sampled cached steps, individual-step RMS changes are 0.038–0.079%. Intervals for the projected mean shift include zero, so this projection is not statistically distinguishable from zero at those checkpoints. This does not establish absence of coupling. These observations apply to the sampled cached positions immediately after refresh; other positions in the refresh cycle were not measured.

Clipping was inactive in every sampled update batch. The diagnostic therefore leaves the effect of active clipping unmeasured. Historical moment and covariance state are fixed within each replay. The intervention does not isolate every source of coupling or establish unbiasedness.

This is one training trajectory with six checkpoints. The intervals are pointwise and conditional on those checkpoints. Independent probe batches sample the training distribution. This measurement concerns first-order loss-gradient projections along fixed update directions, and supplies checkpoint-conditional empirical evidence for variance reduction. Test performance, exact loss changes, training-process variance and risk-objective optimality require separate evaluations. The original 12-task comparison with AdamW remains statistically nonsignificant.

## Reproduction and artifacts

This diagnostic uses the original WikiText-2 Evie-KF configuration with absolute gamma 300 and learning rate 0.001. It retains the original loader cadence. Its comparator is the gamma-zero direction at the same model and moment state.

Use [checkpoint_probe_17101.ipynb](../../../../kaggle/jobs/checkpoint_probe/checkpoint_probe_17101.ipynb) with Kaggle GPU T4 and Internet enabled, then Save & Run All. No dataset attachment is needed. The notebook downloads WikiText-2, trains for 10,000 steps and measures both probes at the six fixed checkpoints. The published compact output archive contains the unmodified numeric results.

The notebook embeds the task from this anonymous repository. The optimizer, training loop, data loaders, model and loss implementations match those used for the reported run. A docstring in the optimizer factory is anonymized.

Regenerate the notebook from the repository root with

```sh
python kaggle/tools/build_checkpoint_probe.py
```

Recompute the summary CSV from the published JSON with

```sh
python benchmark12/tools/summarize_checkpoint_probe.py benchmark12/results/rebuttal/checkpoint_probe_17101
```

The per-checkpoint JSON files retain both gate-input controls, interval estimates and individual paired direction variances. The training outcome is recorded in probe_training_17101.json. summary.csv contains the compact numeric readout. provenance.json records hashes of the released files and the run configuration.
