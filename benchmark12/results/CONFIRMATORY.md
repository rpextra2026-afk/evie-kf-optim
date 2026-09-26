# Confirmatory 12-task test — final (2026-09-19)

The pre-registered test (CLAUDE.md §5): per (task, optimizer) the median over 5
seeds; per task the relative difference d_i against a reference optimizer;
**paired two-sided Wilcoxon signed-rank over the 12 tasks**, Holm-corrected over
the 7 comparisons; sign test as a robustness check. Run twice:

```
python benchmark12/aggregate.py --ckpt-root benchmark12/results --ref EvieKF
python benchmark12/aggregate.py --ckpt-root benchmark12/results --ref AdamW
```

Outputs: `aggregate_12task_ref_EvieKF.json`, `aggregate_12task_ref_AdamW.json`
(this folder). All 12 tasks present; no imputed failures.

## Result 1 — EvieKF vs every baseline (the paper's comparison family)

d > 0 means the **baseline** beat EvieKF.

| baseline | baseline better on | median d | Wilcoxon p | **Holm p** | sign p | verdict |
|---|---|---|---|---|---|---|
| **Sophia** | 1/12 | −9.34% | 0.00098 | **0.0068** | 0.0063 | **EvieKF significantly better** |
| Lion | 2/12 | −3.68% | 0.021 | 0.126 | 0.039 | not significant after Holm |
| AdamW | 4/12 | −3.44% | 0.233 | 1 | 0.388 | not significant |
| Shampoo | 4/12 | −2.00% | 0.266 | 1 | 0.388 | not significant |
| AdaBelief | 5/12 | −2.38% | 0.204 | 1 | 0.774 | not significant |
| SGD | 6/12 | 0.00% | 0.970 | 1 | 1 | not significant |
| Muon | **10/12** | +2.50% | 0.204 | 1 | 0.039 | not significant (Muon ahead on 10/12) |

**One pre-registered comparison reaches significance: EvieKF beats Sophia.**
EvieKF is ahead of AdamW on 8 of 12 tasks (median +3.4%) and of Lion on 10 of 12,
but neither survives the correction. Muon is ahead of EvieKF on 10 of 12 tasks;
that is not significant either, because EvieKF's two large finance wins over
Muon (Muon diverged heavily on those panels) dominate the signed ranks.

## Result 2 — every optimizer vs AdamW (the conventional framing)

| optimizer | better than AdamW on | median d | Wilcoxon p | Holm p | verdict |
|---|---|---|---|---|---|
| **EvieKF** | **8/12** | **+3.32%** | 0.204 | 1 | not significant |
| Muon | 8/12 | +4.74% | 0.266 | 1 | not significant |
| SGD | 8/12 | +0.91% | 0.733 | 1 | not significant |
| Shampoo | 7/12 | +0.30% | 0.677 | 1 | not significant |
| AdaBelief | 4/12 | −0.45% | 0.233 | 1 | not significant |
| Lion | 4/12 | −0.61% | 0.266 | 1 | not significant |
| Sophia | 2/12 | −4.75% | 0.0049 | **0.034** | **significantly worse than AdamW** |

**No optimizer is significantly better than AdamW** across 12 tasks at 5 seeds —
including Muon. EvieKF has the joint-highest win count (8/12, with Muon and SGD)
and the second-highest median gain.

## Descriptive summary (no test)

Mean rank over 12 tasks (`paper/table2_ranks.md`): **Muon 3.00, EvieKF 3.67**,
Shampoo 3.92, SGD 4.08, AdamW 4.42, AdaBelief 5.08, Lion 5.50, Sophia 6.33.

Per-domain mean d vs EvieKF (> 0 = baseline better): vision — Muon +2.5%, others
within ±1% except Sophia −3.0%; language — Muon +6.8%, every other arm worse
than EvieKF (AdamW −4.0%, SGD −21.1%); finance — SGD +2.0%, every other arm worse.

## How to report it

- The confirmatory claim that survives: **EvieKF > Sophia** (Holm p = 0.007).
- Everything else is descriptive: EvieKF is **2nd by mean rank** (behind Muon),
  beats AdamW on 8/12 tasks with a median gain of 3.3%, and beats AdamW **at a
  matched learning rate on every seed** on WikiText-2, enwik8 and WikiText-103.
- The benchmark could not separate any optimizer from AdamW except Sophia
  (worse). That is a finding about the benchmark's power at 12 tasks × 5 seeds as
  much as about any optimizer, and it should be stated plainly.
- The test is two-sided as pre-registered (`aggregate.py`). Earlier bounds in
  `STATUS.md` were computed one-sided; the conclusions are the same.
