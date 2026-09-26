# FW-9 — does γ transfer once the spectrum is normalised properly? (2026-09-24)

**The question.** The shipped code normalises the Kronecker spectrum by
`τ = tr(A)`, which matches the estimator's trace but is **not scale-free**
(`DERIVATION.md` §5): scaling the whitened noise by `c` scales `μλ/τ` by `c²`, so
the tuned γ has to absorb each task's absolute noise energy. That is why the 12
tasks froze γ anywhere from 3 to 243,000 while the specification's grid was
`{3, 10, 30, 100}`. `EvieKFm` normalises by the **mean eigenvalue** instead, the
only form in which γ is genuinely dimensionless.

Three hosts, chosen because their shipped γ values are as far apart as the
benchmark gets. lr **pinned** at each task's protocol value so that only γ moves;
γ searched over a re-centred 7-point grid. 5 seeds each, one session, no resumes,
no diverged cells.

## The result: γ becomes a constant

| host | shipped `/τ` γ | mean-norm γ |
|---|---|---|
| language_wikitext2 | 300 | **30** |
| language_ptb | 10 | **100** |
| vision_cifar10 | 27,000 | **30** |
| **spread** | **2,700×** | **3.3×** |

Two of three land on exactly the same value, and the third is one grid point
away. Better still, **a single γ = 30 is within 0.33% of every host's own pick** —
on PTB, γ = 30 scores 208.41 against the chosen γ = 100's 207.72.

So the diagnosis in `DERIVATION.md` §5 is confirmed by measurement: under the
shipped parameterisation γ is not a hyperparameter, it is a scale factor in
disguise. Fix the normalisation and **γ stops being a tuned axis**.

## What it costs in quality

| host | mean-norm (γ as picked) | protocol EvieKF | AdamW | vs protocol | seeds |
|---|---|---|---|---|---|
| WikiText-2 (ppl ↓) | **98.47** | 100.46 | 105.70 | **+1.97%** | 4/5 |
| CIFAR-10 (acc ↑) | 81.39 | **81.94** | 81.55 | −0.67% | 2/5 |
| PTB (ppl ↓) | 200.06 | **192.17** | 179.35 | −4.10% | 1/5 |

Mixed: one clear gain, one small loss, one real loss. **Read this with the caveat
that the learning rate was pinned to the value tuned for the *other* operator.**
`EvieKFm` is a different operator; its own preferred lr was never searched. A
fair head-to-head would re-tune lr for both, which is a joint search again and
was not affordable here.

The honest summary is therefore: **the transfer claim is established, the quality
claim is not.** γ = 30 works everywhere; whether the mean-normalised operator is
as good as the shipped one at its own best lr is untested.

## γ rows (val metric at the 2,500-step tuning budget)

```
WikiText-2  g   0.1    0.3      1      3     10     30    100
   ppl        102.1  102.8  103.5  106.8  103.2  102.0  105.2     <- 30
PTB         g   0.1    0.3      1      3     10     30    100    300    900
   ppl        217.4  214.1  223.5  216.2  214.5  208.4  207.7  218.6  221.0  <- 100
CIFAR-10    g   0.1    0.3      1      3     10     30    100
   acc         79.4   77.7   80.8   79.9   79.2   82.0   79.8     <- 30
```

These rows are flat and noisy — spreads of 4.7% (WikiText-2), 7.1% (PTB) and
4.3 pp (CIFAR-10) with no clean monotonicity. That cuts both ways: the agreement
on γ ≈ 30 is partly luck, **and** it means the exact value barely matters, which
is what a dimensionless knob should look like. PTB needed one boundary extension
(to 300 and 900) and still chose an interior point.

## What to do with this

**For this submission:** report it as a diagnostic. It changes the operator, so
it cannot be folded into the 12-task table, and its quality comparison is
lr-confounded. The defensible sentence is: *"the risk parameter in the shipped
parameterisation is not dimensionless; under a mean-eigenvalue normalisation the
tuned value collapses from a 2,700× range to 3.3×, and a single γ = 30 is within
0.33% of each task's own optimum."*

**For the next version of the method:** this is the fix. It turns the
tuning-budget objection from something to disclose into something that no longer
exists — EvieKF would search one axis, like every baseline. That is a stronger
paper than the one we are submitting, and it is the single most valuable thing on
the future-work list.

**Before claiming it works as well as the shipped form**, re-tune lr for
`EvieKFm` on at least two hosts. Until then the quality column above is
suggestive, not evidence.

## Provenance

`results (21)`, `(25)`, `(26)`, Kaggle 2026-09-24. One session per host; zero
`[resume]` lines; zero diverged cells; lr boundary checks 0 everywhere (the axis
was pinned, as intended). Raw files in each task's folder.
