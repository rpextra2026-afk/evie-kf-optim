# B1 — EvieKF at the baselines' tuning budget (all 4 hosts, 2026-09-24)

**The question.** EvieKF searches a joint (lr, γ) grid of 49–99 points; AdamW
searches 7. Does the result survive at an equal budget — γ **fixed at 3000**, lr
over the ordinary 7-point half-decade grid with the usual boundary rule? And is a
fixed γ perhaps *better* than the searched one on the tasks where the γ row looks
like an argmax over noise?

**Answer: it is a coin flip at task level — one large win, one tie, two losses —
so a single default cannot replace the search. But the win is the informative
result, because it shows the γ search itself making a demonstrably wrong pick.**

5 seeds per host, one session each, zero resumes, zero diverged cells,
`knob boundary checks: 0` everywhere (the axis was genuinely fixed).

## Results

| host | fixed γ=3000 | protocol EvieKF | AdamW | vs protocol | seeds | protocol's γ |
|---|---|---|---|---|---|---|
| **STL-10** (acc ↑) | **51.49** | 48.59 | 51.29 | **+5.97%** | **5/5** | 3 |
| SVHN (acc ↑) | **94.699** | 94.634 | 94.664 | +0.07% | 4/5 | 10 |
| CIFAR-10 (acc ↑) | 80.40 | **81.94** | 81.55 | −1.88% | 0/5 | 27,000 |
| PTB (ppl ↓) | 197.53 | **192.17** | 179.35 | −2.79% | 1/5 | 10 |

Median across the four: **−0.9%**. Tuning spent: 7 points (CIFAR-10, PTB) or 9
(SVHN, STL-10 — one boundary extension each), against the protocol's 49–99. The
equal-budget claim is honest. γ costs nothing at run time — wall clock is within
4% of the protocol cells on every host; only *searching* for it costs.

## STL-10: the γ search made a wrong pick, and we can prove it

This is the cleanest result of the four, because **both arms froze the same
learning rate** (3.162e-2). There is no lr confound: the only difference is γ.

The protocol's own γ row at that lr, scored at the 2,500-step tuning budget:

| γ | 0 | 1 | **3** | 10 | 30 | 100 | 300 | 1000 | **3000** |
|---|---|---|---|---|---|---|---|---|---|
| search acc | 48.8 | 49.8 | **54.2** | 52.4 | 50.8 | 52.8 | 52.4 | 51.6 | **53.6** |

The search preferred γ = 3 over γ = 3000 by 0.6 pp. At the full 10,000-step
budget the ordering **reverses and widens**: 48.59 against 51.49, a 2.9 pp gap,
on 5 of 5 seeds. The tuning budget judged the wrong thing.

The mechanism telemetry says the same:

| | fixed γ=3000 | protocol (γ=3) |
|---|---|---|
| cos(step, Adam) | 0.445 | 0.882 |
| min gain | 0.020 | 0.756 |

At γ = 3 the operator is **nearly off** — a 28° rotation of the AdamW update, and
gains barely below 1. At γ = 3000 it is doing real work (63°), and it is worth
2.9 pp. EvieKF's worst-but-one task was worst because its operator had been
switched off by a noisy 2,500-step score.

**At γ = 3000, STL-10's EvieKF (51.49) would rank 1st of 8**, ahead of AdamW's
51.29, instead of 6th. See the honesty note below before using that sentence.

## PTB: clean, and it closes FW-5 in the other direction

At lr 1e-3 the protocol's γ row reads:

| γ | 3 | **10** | 30 | 100 | 300 | 1000 | 3000 |
|---|---|---|---|---|---|---|---|
| search ppl | 201.29 | **186.28** | 205.72 | 198.82 | 202.92 | 207.85 | 209.75 |

γ = 3000 is the **worst point on the row**, and the full-budget run preserves the
ordering: 197.53 against 192.17. The fixed-γ arm's lr optimum was interior, so
nothing is confounded.

This settles `FUTURE_WORK.md` **FW-5**: a strong fixed γ is not what PTB needed.
With `language_ptb_evie_at_adamw_lr` (lr 3.162e-4, γ 10 → **184.66**), PTB is
settled — **the learning rate was the problem and γ was not.**

It also retires a claim I made earlier: I described PTB's γ row as "an argmax over
noise". The row *is* non-monotone, but here the pick was right. STL-10 is where
that criticism actually lands.

## SVHN: a genuine tie, with the operator far more active

+0.07%, 4/5 seeds, on a task whose arms 2–8 span 0.34 pp. The operator went from
nearly idle to strongly active (cos 0.943 → 0.675, min gain 0.291 → 0.013) and the
answer did not move. **More operator is not automatically better** — on SVHN
there is nothing for the preconditioner to find at any γ.

## CIFAR-10: a loss, and **partly confounded**

The fixed-γ arm froze lr **1e-2**; the protocol froze **0.09999**. Its own search:

```
... 3.162e-3  77.30 | 1e-2  79.48  <- frozen | 3.162e-2  79.18
```

The optimum is **interior**, so the boundary rule never fired and the grid never
reached 0.1 — where the protocol's own γ = 3000 column scores 81.64. A 0.3 pp
difference at the tuning budget, on a task whose whole arm spread is 0.94 pp,
decided whether an extra half-decade got explored. Logged as **FW-12**.

Part of the loss is real, though: CIFAR-10 is the one host whose protocol γ
(27,000) is *above* 3000, so pinning it lowers γ. At lr 0.09999 the protocol
scored γ = 3000 at 81.64 against γ = 27,000 at 82.28.

CIFAR-10 also shows why the two axes cannot be separated:

| at lr | γ = 3 | γ = 27,000 |
|---|---|---|
| 0.09999 | 77.08 | **82.28** |
| 0.01 | 80.16 | 79.64 |

**γ and lr are coupled**: a large γ is worth +5.2 pp at the high lr and −0.5 pp at
the low one. γ is what makes the high learning rate usable — which fits the
norm-preserving reading (`DERIVATION.md` §6): the operator does not shorten the
step, it rotates it off the noisy directions that make a large step unstable.

## What to write in the paper

1. **A single default γ cannot replace the search.** One large win, one tie, two
   losses, median −0.9%. Anyone using EvieKF must tune γ, and the paper must
   count that cost rather than dodge it.
2. **But the search is not reliable either**, and STL-10 proves it at
   matched lr: a 0.6 pp preference at 25% of the budget inverted into a 2.9 pp
   loss at full budget, on 5/5 seeds. This is `FUTURE_WORK.md` **FW-4** — tuning
   at 25% of budget judges the wrong schedule — with a measured instance instead
   of an argument.
3. **The coupling is the mechanism claim**, and CIFAR-10 shows it: γ buys a
   usable learning rate an order of magnitude above AdamW's.
4. **Two protocol limitations are now documented with evidence**: FW-4 (the
   search budget picks wrongly where the γ row is flat) and FW-12 (a 7-point grid
   plus boundary-extend can miss an optimum one point past the edge).

### Honesty note — read before quoting STL-10

The four B1 hosts were **chosen after seeing the main results**, specifically
because their γ picks looked suspicious. A win on one of four selected tasks is
close to what noise alone would produce, so **STL-10's +5.97% is not evidence
that EvieKF is better than the main table says**, and the "1st of 8" framing must
not be used as a headline. What it is evidence for is narrower and more useful:
that the *tuning protocol* can switch the operator off on a task where it would
have helped. Report it as a protocol limitation, with the selection stated.

The protocol numbers in Table 1 stand unchanged. Nothing here replaces them.

## Provenance

`results (17)–(20)`, Kaggle, 2026-09-23 19:38 → 2026-09-24 08:21. One session per
host; zero `[resume]` lines; zero diverged cells; `knob boundary checks: 0` on all
four, confirming `TASK["fixed_knob"]` held the axis. Raw files in each task's
folder.
