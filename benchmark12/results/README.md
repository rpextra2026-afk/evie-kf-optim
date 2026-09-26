# benchmark12/results

One folder per finished task, plus the paper tables built from all of them.

```
<task_name>/        summary.json  results.json  results_raw.csv  lr_search.json  log.txt
                    ANALYSIS.md   readout.txt
paper/              SETUP.md, ALL_TABLES.md, table*.{md,tex,csv}
sweeps/<task>/       batch-size / model-size sweeps (secondary, descriptive)
diagnostics/<name>/  pinned one-question re-runs; never replace a protocol number
analyse_task.py     per-task descriptive read-out
build_paper_tables.py   every table + the experimental-setup section
```

## Adding a finished task

1. Copy `checkpoints/<task_name>/{summary.json,results.json,results_raw.csv,lr_search.json,log.txt}`
   from its DONE output into `benchmark12/results/<task_name>/`. The `cells/*.pt`
   files are not needed -- they only matter for resuming.
2. Check `summary.json` has a `unit_reduction` field. If it does not, that
   session ran the pre-fix engine: re-aggregate it on the current notebook first.
3. `python benchmark12/results/analyse_task.py benchmark12/results/<task_name> > benchmark12/results/<task_name>/readout.txt`
4. `python benchmark12/results/build_paper_tables.py` -- rebuilds every table
   for all present tasks.

`results.json` holds every cell's validation curve, for learning-curve figures.

Everything here is descriptive except Table 5, which is the input to the
confirmatory test in `benchmark12/aggregate.py` (12-task paired Wilcoxon, Holm
over 7, run with `--ref AdamW` and `--ref EvieKF`).
