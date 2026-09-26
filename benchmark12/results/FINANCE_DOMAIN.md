# Finance domain — complete (4 of 4 panels, 4 of 12 tasks)

Synthesis across `finance_tech`, `finance_consumer_industrials`,
`finance_healthcare`, `finance_finance`. Every number traces to
`paper/table*.csv`; rebuild with `python benchmark12/results/build_paper_tables.py`.

**Descriptive.** The confirmatory test needs all 12 tasks. Finance supplies 4 of
the 12 per-task values that test consumes, and this document says what they are.

## One lesson first

After two panels, the read was "EvieKF's competitor is Muon, not AdamW." After
four, **Muon ranks 1, 2, 8, 8** — best on two panels, worst on the other two,
where it broke on up to 18 of 49 tickers. A two-panel pattern in this benchmark is
not a pattern. The same caution applies to everything below until vision and
language land.

## Ranking (pre-registered: complete-case, unit median, seed median)

| optimizer | tech | consumer | healthcare | finance | **mean rank** |
|---|---|---|---|---|---|
| **SGD** | 4 | 1 | 1 | 2 | **2.00** |
| **EvieKF** | 2 | 3 | 2 | 5 | **3.00** |
| AdamW | 6 | 4 | 3 | 4 | 4.25 |
| Lion | 3 | 8 | 5 | 1 | 4.25 |
| Muon | 1 | 2 | 8 | 8 | 4.75 |
| Shampoo | 5 | 5 | 4 | 7 | 5.25 |
| AdaBelief | 7 | 7 | 6 | 3 | 5.75 |
| Sophia | 8 | 6 | 7 | 6 | 6.75 |

**SGD with Nesterov momentum is the best finance optimizer; EvieKF is second** and
the most consistent of the top group (never below 5th; Muon and Lion both reach 8th).

## Failures counted as losses (sensitivity, Table 9b)

| optimizer | complete-case mean rank | failures-as-losses mean rank |
|---|---|---|
| SGD | 2.00 | 2.00 |
| **EvieKF** | 3.00 | **2.50** |
| Lion | 4.25 | 4.50 |
| Muon | 4.75 | 4.75 |
| AdamW | 4.25 | 5.00 |

The pre-registered rule drops a ticker for all arms when any arm fails on it,
which means an arm's own failures never enter its own number. Counting them,
EvieKF closes on SGD (ranks 5, 2, 1, 2): on the tickers hard enough to break other
optimizers, EvieKF held up. **Report both.** The gap between them is widest on
finance_finance, where complete-case keeps only 23 of 49 tickers.

## Stability is EvieKF's clearest finance result

| optimizer | diverged cells, 4 panels | rate |
|---|---|---|
| AdamW | 0 / 985 | 0.00% |
| AdaBelief | 0 / 985 | 0.00% |
| **EvieKF** | **2 / 985** | **0.20%** |
| SGD | 13 / 985 | 1.32% |
| Sophia | 18 / 985 | 1.83% |
| Shampoo | 21 / 985 | 2.13% |
| Lion | 30 / 985 | 3.05% |
| Muon | 34 / 985 | 3.45% |

EvieKF diverged twice (both on one IBM ticker), matching the Adam family and
roughly 6–17× less often than every other non-Adam method. Seed stability, by
contrast, is not a consistent EvieKF advantage: its spread ranges 0.71% to 9.85%
across panels, and it was among the least seed-stable arms on healthcare.

## What finance hands the confirmatory test

Per-task d_i for each baseline against EvieKF (Table 5, reference EvieKF;
**> 0 means the baseline beat EvieKF**):

| baseline | tech | consumer | healthcare | finance | beats EvieKF |
|---|---|---|---|---|---|
| AdamW | −4.43% | −2.45% | −6.24% | +0.11% | 1 / 4 |
| AdaBelief | −5.42% | −6.87% | −7.35% | +0.38% | 1 / 4 |
| **SGD** | −0.10% | **+6.07%** | **+0.25%** | **+1.73%** | **3 / 4** |
| Sophia | −9.87% | −4.75% | −9.33% | −16.27% | 0 / 4 |
| Muon | +0.05% | +2.86% | −11.80% | −64.29% | 2 / 4 |
| Lion | −0.01% | −7.48% | −7.31% | +1.76% | 1 / 4 |
| Shampoo | −1.62% | −4.50% | −6.80% | −49.16% | 0 / 4 |

The confirmatory test is a paired Wilcoxon over 12 tasks with Holm over 7
comparisons. At n = 12 it survives at most **3 losing tasks, and only if those
are the smallest-magnitude effects in the set.**

- **EvieKF vs SGD is now the binding comparison.** Finance already contains 3
  losses, one of them +6.07% — not small. For this comparison to clear Holm,
  EvieKF essentially has to beat SGD on all 8 vision and language tasks, by
  margins larger than 6%.
- **EvieKF vs Muon** has 2 losses, both small (+0.05%, +2.86%), and 2 very large
  wins. Recoverable.
- **EvieKF vs AdamW, AdaBelief, Lion** each have 1 small loss. Fine so far.
- **EvieKF vs Sophia, Shampoo**: 4 / 4 wins.

And the paired, ticker-by-ticker evidence is weaker than those task-level signs.
EvieKF vs AdamW paired: +2.89% (31/37), +0.12% (23/45), +0.34% (21/33), −0.17%
(9/23). Only finance_tech shows a gap that holds ticker by ticker; the other three
task-level wins are mostly seed noise surviving the median.

## The mechanism: active, variable, and not what decides the rank

| | tech | consumer | healthcare | finance |
|---|---|---|---|---|
| frozen γ | 10 | 300 | 1000 | **9000** |
| γ response at frozen LR | 1.82% / 1000× | 2.04% / 1000× | 2.16% / 1000× | **0.65% / 9000×** |
| cos(update, Adam direction) | 0.922 | 0.798 | 0.757 | **0.596** |
| min gain | 0.98 | 0.60 | 0.36 | **0.096** |
| EvieKF rank | 2 | 3 | 2 | **5** |

1. **γ is flat on every panel** — never more than 2.2% across a thousand-fold
   range. The frozen γ therefore says nothing about the task: across four
   near-identical problems it landed on 10, 300, 1000 and 9000, which is the
   signature of an argmax over noise.
2. **Because the frozen γ drives how hard the operator acts, the operator's
   strength across panels is effectively set by that noise.** It rotates the
   update by anywhere from 23° to 53°, and shrinks the minimum gain from 0.98 to
   0.096.
3. **More operator activity did not produce a better rank.** The most aggressive
   panel (finance, γ = 9000, cos 0.60) is EvieKF's worst.

The implication for the paper is direct: on finance, EvieKF's benefit — its
stability — does not come from tuning γ, and its 63-point joint (lr, γ) search
bought nothing a 7-point LR search would not have. Whether γ matters on language
is the open question; the PoC effect was measured on WikiText-2.

## Cost

EvieKF costs **4.7–4.9× AdamW** per cell on every panel, against Muon 2.2–2.3×,
Sophia 2.3–2.5×, Shampoo ~2×. On a 5 → 64 → 32 → 1 MLP the split-batch backward
and eigendecomposition are dominated by overhead; the ratio should fall on the
vision and language models, and Table 8 will show whether it does.

## Disclosures that must travel with these numbers

- finance_tech: 1145 of 1960 cells recovered from run logs (exact primary and
  divergence flag, no telemetry); first summary from the pre-fix aggregation,
  re-aggregated.
- Complete-case units: tech 37/49, consumer 45/50, healthcare 33/49, **finance
  23/49**.
- The LR search shares seed 42's batch order. Excluding seed 42 (Table 3b) moves
  EvieKF on two panels — up to 3rd on finance, down to 3rd on tech — always among
  arms within 1%.
- Tuning budget is unequal by design (Table 6): EvieKF scores 63 grid points on
  every panel (504 search training runs), AdamW 7–9 (56–72).
- The EvieKF active-step counter resets on resume and is not reported.
