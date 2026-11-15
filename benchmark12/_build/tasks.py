"""
The 12 frozen ICLR-benchmark tasks (benchmark protocol sec 3). build.py consumes this.

Frozen protocol knobs shared by every task:
  * MAX_STEPS = 10000 (step-based, not epoch-based)
  * LR_SEARCH_STEPS = 2500 (25% of budget), 1 seed
  * SEEDS = [42, 43, 44, 45, 46] (5 seeds; deliberate increase over the old
    3-seed benchmark, part of fixing the workshop paper's stats failure)
  * 7-point half-decade LR grid per optimizer family + boundary re-search
  * decoupled weight decay 1e-4, global grad-norm clip 5.0, float32, cosine LR
Only per-task fields live below.
"""

COMMON = dict(
    max_steps=10000,
    lr_search_steps=2500,
    seeds=[42, 43, 44, 45, 46],
)

# benchmark protocol sec 9.5: WikiText-2 and PTB use a 16k whitespace-tokenized vocab and
# a ~10M-param model at a fixed 10k-step budget -- their perplexity is valid ONLY
# as a within-task, cross-optimizer ranking and is NOT comparable to published
# WikiText/PTB leaderboard numbers (~33k vocab / BPE). Logged everywhere results
# are printed or written.
_PPL_CAVEAT = ("perplexity uses a 16k whitespace-tokenized vocab + ~10M-param "
               "model at a fixed 10k-step budget -- within-task cross-optimizer "
               "ranking ONLY, NOT comparable to published WikiText/PTB "
               "perplexities (~33k vocab / BPE)")

_VISION = dict(domain="vision", primary_metric="test_top1_acc",
               primary_lower_better=False, primary_abs_range=[0.0, 100.0],
               peer_divergence_factor=3.0, batch_size=128,
               eval_every=500, ckpt_every=2000)

_LANGUAGE = dict(domain="language", primary_metric="test_perplexity",
                 primary_lower_better=True, peer_divergence_factor=5.0,
                 batch_size=32, eval_every=500, ckpt_every=2000)

_FINANCE = dict(domain="finance", primary_metric="test_mse",
                primary_lower_better=True, primary_abs_range=[0.0, 1.0],
                peer_divergence_factor=50.0, batch_size=32,
                eval_every=250, ckpt_every=2000)

TASKS = [
    # ---- VISION (CIFAR-stem ResNet-18) --------------------------------------
    dict(_VISION, task_name="vision_cifar10", dataset="cifar10",
         title="Vision / CIFAR-10 / CIFAR-stem ResNet-18"),
    dict(_VISION, task_name="vision_cifar100", dataset="cifar100",
         title="Vision / CIFAR-100 / CIFAR-stem ResNet-18"),
    dict(_VISION, task_name="vision_svhn", dataset="svhn",
         title="Vision / SVHN / CIFAR-stem ResNet-18"),
    dict(_VISION, task_name="vision_stl10", dataset="stl10",
         title="Vision / STL-10 (resized 96->32) / CIFAR-stem ResNet-18"),

    # ---- LANGUAGE (small GPT: 6L, d256, 8h, ctx256) -----------------------
    dict(_LANGUAGE, task_name="language_wikitext2", dataset="wikitext2",
         primary_abs_range=[1.0, 60000.0], nonstandard_ppl_caveat=_PPL_CAVEAT,
         title="Language / WikiText-2 / small GPT"),
    dict(_LANGUAGE, task_name="language_wikitext103", dataset="wikitext103",
         primary_abs_range=[1.0, 60000.0], token_budget=120_000_000,
         title="Language / WikiText-103 (token-budget cap) / small GPT"),
    dict(_LANGUAGE, task_name="language_ptb", dataset="ptb",
         primary_abs_range=[1.0, 60000.0], nonstandard_ppl_caveat=_PPL_CAVEAT,
         title="Language / Penn Treebank / small GPT"),
    dict(_LANGUAGE, task_name="language_enwik8", dataset="enwik8",
         primary_abs_range=[1.0, 3000.0],
         title="Language / enwik8 (byte-level) / small GPT"),

    # ---- FINANCE (3-layer OHLCV MLP, next-day log-return) -----------------
    dict(_FINANCE, task_name="finance_tech", panel="tech",
         title="Finance / Tech sector panel / OHLCV MLP"),
    dict(_FINANCE, task_name="finance_finance", panel="finance",
         title="Finance / Finance sector panel / OHLCV MLP"),
    dict(_FINANCE, task_name="finance_healthcare", panel="healthcare",
         title="Finance / Healthcare sector panel / OHLCV MLP"),
    dict(_FINANCE, task_name="finance_consumer_industrials",
         panel="consumer_industrials",
         title="Finance / Consumer & Industrials sector panel / OHLCV MLP"),
]

for _t in TASKS:
    for _k, _v in COMMON.items():
        _t.setdefault(_k, _v)

assert len({t["task_name"] for t in TASKS}) == 12, "task_name collision"


# ==========================================================================
#  PHASE 2 (benchmark protocol sec 9) -- batch-size + model-size sweeps.
#  6 host tasks x 2 sweep types = 12 generated files under benchmark12/sweeps/.
#  Reuse the same fragments; each file gets its own checkpoints/<task>_<type>/.
# ==========================================================================
SWEEP_HOST_NAMES = ["vision_cifar10", "vision_svhn",
                    "language_wikitext2", "language_ptb",
                    "finance_tech", "finance_healthcare"]

_BASE = {t["task_name"]: t for t in TASKS}


# ==========================================================================
#  EVIE-KF ABLATION (optimizer specification sec 2) -- the C1 (centred vs uncentred)
#  and C2 (Kronecker vs diagonal) ladder, "the paper" per the v3 doc.
#
#  Arms: EvieDiag, EvieKFu, EvieKF (3, from ABLATION_OPT_NAMES). EvieKF is also
#  the main table's 8th arm (see frag_shared_optim.OPT_NAMES), so the 3 ablation
#  hosts recompute EvieKF at 10 seeds -- a superset of the main run's 5; the
#  overlap is a known, accepted redundancy, not de-duplicated on purpose. Each
#  arm gets its OWN independent joint (lr, gamma) search per task (diag_only /
#  centered=F change how the loss responds to gamma), same LR-search-then-freeze
#  protocol as the main run.
#
#  Structure decision (documented for the user): optimizer specification sec 2 says
#  "a 13th ... file". The 3 chosen tasks span 3 domains with different primary
#  metrics, directions and batch sizes, which the single-TASK engine cannot
#  carry in one file without invasive, harness-wide changes. Three
#  domain-native files (one per chosen task) keep every fragment byte-identical
#  and every checkpoint / divergence / preflight rule intact. Task list and arms
#  are exactly as the spec fixes them; the ablation uses 10 seeds (see below),
#  and only the file count differs from the spec's "a 13th file" phrasing.
# ==========================================================================
_ABLATION_HOSTS = [
    "language_wikitext2",   # where the PoC result was measured -- direct continuity
    "vision_cifar10",       # cheapest, most standard vision task
    "finance_tech",         # largest ticker pool
]

ABLATION_TASKS = []
for _hn in _ABLATION_HOSTS:
    _b = _BASE[_hn]
    _cfg = dict(_b)                       # inherit domain/dataset/panel/metric/etc.
    _cfg["task_name"] = f"ablation_{_hn}"
    _cfg["base_task_name"] = _hn
    _cfg["opt_set"] = "ablation"
    # 10 seeds, not the main run's 5: the exact one-sided Wilcoxon floor at
    # n=5 is 0.03125, and 0.03125 x 8 comparisons >= 1, so NO 5-seed result
    # can survive Holm correction. n=10 -> floor 0.00098 -> Holm 0.0078,
    # which is what actually let kaggle_evie_v3.py's poc run reach
    # significance. This is what C1/C2 need to be decisive at full scale.
    _cfg["seeds"] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]
    _cfg["title"] = (f"{_b['title']}  --  Evie-KF ablation "
                     f"(EvieDiag / EvieKFu / EvieKF, 10 seeds)")
    ABLATION_TASKS.append(_cfg)

assert len(ABLATION_TASKS) == 3, "expected 3 ablation host tasks"
assert len({t["task_name"] for t in ABLATION_TASKS}) == 3, "ablation task_name collision"
assert not ({t["task_name"] for t in ABLATION_TASKS}
            & {t["task_name"] for t in TASKS}), "ablation collides with a main task"

_BATCH_GRID = {           # new batch sizes (dyadic, below the baseline) + baseline
    "vision":   {"new": [4, 8, 16, 32], "baseline": 128},
    "language": {"new": [2, 4, 8, 16], "baseline": 32},
    "finance":  {"new": [2, 4, 8, 16], "baseline": 32},
}
_MODEL_GRID = {           # small / (baseline reused) / large -- benchmark protocol sec 9.3
    "vision":   {"points": {"small": [32, 64, 128, 256],
                            "large": [128, 256, 512, 1024]},
                 "baseline_spec": [64, 128, 256, 512]},
    "language": {"points": {"small": {"n_layer": 3, "d_model": 128, "n_head": 4},
                            "large": {"n_layer": 9, "d_model": 384, "n_head": 12}},
                 "baseline_spec": {"n_layer": 6, "d_model": 256, "n_head": 8}},
    "finance":  {"points": {"small": [32, 16], "large": [128, 64]},
                 "baseline_spec": [64, 32]},
}
# sec 9.2 -- both EvieKF and EvieKFu: the batch-size sweep exists to test the
# pre-registered C1 prediction (the centred-vs-uncentred gap should widen at
# small batch / high gradient noise, even though it was a tie at the PoC's
# batch 32), which needs both arms present to compare against each other.
_BATCH_ARMS = ["AdamW", "AdaBelief", "Sophia", "Muon", "EvieKF", "EvieKFu"]
# sec 9.3 (leaner) -- EvieKF only: the model-size axis is the secondary "does
# the edge survive at different capacity" check, not a C1 test, so EvieKFu is
# not needed here.
_MODEL_ARMS = ["AdamW", "AdaBelief", "EvieKF"]

_INHERIT = ("domain", "dataset", "panel", "token_budget", "primary_metric",
            "primary_lower_better", "primary_abs_range", "peer_divergence_factor",
            "eval_every", "ckpt_every", "nonstandard_ppl_caveat")

SWEEP_TASKS = []
for _hn in SWEEP_HOST_NAMES:
    _b = _BASE[_hn]
    _dom = _b["domain"]
    for _stype, _grid, _arms in (
        ("batchsize", _BATCH_GRID[_dom], _BATCH_ARMS),
        ("modelsize", _MODEL_GRID[_dom], _MODEL_ARMS),
    ):
        _cfg = dict(
            task_name=f"{_hn}_{_stype}",
            base_task_name=_hn,
            sweep_type=_stype,
            sweep_grid=_grid,
            sweep_arms=list(_arms),
            batch_size=_b["batch_size"],          # the baseline batch size
            max_steps=10000,
            lr_search_steps=2500,
            seeds=[42, 43, 44],                   # first 3 of the main run's 5
            title=(f"{_b['title']}  --  {_stype} sweep"),
            is_sweep=True,
        )
        for _k in _INHERIT:
            if _k in _b:
                _cfg.setdefault(_k, _b[_k])
        SWEEP_TASKS.append(_cfg)

assert len(SWEEP_TASKS) == 12, "expected 6 hosts x 2 sweep types"
assert len({t["task_name"] for t in SWEEP_TASKS}) == 12, "sweep task_name collision"


# ==========================================================================
#  SCALE-GENERALIZATION SECONDARY TASKS (evie-scalegen-spec.md) -- one
#  genuinely harder task per domain (vision/language only), answering "does
#  the edge hold at real scale, or only in the CIFAR/tiny-GPT regime." NOT
#  added to TASKS/ABLATION_TASKS/SWEEP_TASKS and NOT part of the 12-task
#  Wilcoxon/Holm significance pipeline (spec sec 7) -- own opt_set, own
#  output directory (benchmark12/scalegen/), reported descriptively only.
# ==========================================================================
SCALEGEN_TASKS = [
    dict(_VISION,
         task_name="vision_tinyimagenet", dataset="tinyimagenet",
         image_size=64,                      # NEW field, frag_vision_data.py
         opt_set="scalegen",
         seeds=[42, 43, 44],
         max_steps=10000,
         ckpt_every=500,                      # tighter than the default 2000
         title=("Vision / Tiny-ImageNet-200 (native 64x64) / ResNet-18 -- "
                "scale-generalization secondary check, NOT part of the "
                "12-task primary significance table")),

    dict(_LANGUAGE,
         task_name="language_openwebtext", dataset="openwebtext",
         n_layer=8, d_model=384, n_head=8, ctx=512,   # NEW fields, frag_language_data.py
         vocab_mode="gpt2_bpe",                        # NEW field
         token_budget=220_000_000,
         # RESCOPED 2026-09-24. The 4-arm / 12000-step plan needs ~82 GPU-h and
         # cannot land before submission; EvieKF's joint (lr, gamma) search is
         # already DONE and committed (63 points, lr 1e-3 / gamma 27,000 -- see
         # benchmark12/results/scalegen/language_openwebtext/ANALYSIS.md), so what
         # remains is the baselines' searches plus their final cells. Measured
         # per-step costs on THIS model: EvieKF 1.96 s (median of 53 logged search
         # points); the baselines from their measured ratios to AdamW on the four
         # main language tasks (AdaBelief 1.01x, Muon 1.06x, EvieKF 1.61x), giving
         # AdamW ~1.22 s. Per arm, 3 seeds: EvieKF 9.8 h (search already done),
         # AdamW 13.2 h, AdaBelief 13.3 h, Muon 14.0 h. One arm per machine via
         # BENCH12_ARMS -- ~50 GPU-h in total, ~14 h of wall clock.
         #
         # TASK_NAME is deliberately UNCHANGED so the finished EvieKF search is
         # still found by --ckpt-in-auto and is not re-run (that alone would be
         # ~103 h). Two deviations to disclose, both applied equally to both arms:
         # the budget is 6000 steps, not 12000 (98M tokens, still a single pass
         # over fresh data and still more than WikiText-103's 82M), and the LR
         # search stays at 3000 steps, which is 50% of the new budget rather than
         # the protocol's 25% -- kept there ON PURPOSE, because EvieKF's cached
         # search used 3000 and letting AdamW search at 1500 would hand EvieKF an
         # unequal tuning budget.
         max_steps=6000, lr_search_steps=3000, batch_size=32,
         opt_set="scalegen",               # all 4 scalegen arms remain available;
                                           # BENCH12_ARMS picks which ones a given
                                           # machine runs
         force_split_path=True,            # or AdamW OOMs on the logits gradient
         seeds=[42, 43, 44],
         ckpt_every=300, eval_every=400,
         primary_abs_range=[1.0, 5_000_000.0],   # wider than the other 4 language
                                                   # tasks' [1,60000] -- bigger vocab
                                                   # (50257 vs 16000) means a bad LR
                                                   # can plausibly reach higher PPL
                                                   # before NaN; don't let a real
                                                   # divergence get missed by too
                                                   # tight a range
         title=("Language / OpenWebText (~220M-token slice, GPT-2 BPE, "
                "8L/d384/ctx512 GPT) -- scale-generalization secondary check, "
                "NOT part of the 12-task primary significance table")),
]

for _t in SCALEGEN_TASKS:
    _t.setdefault("lr_search_steps", int(0.25 * _t["max_steps"]))

assert len({t["task_name"] for t in SCALEGEN_TASKS}) == 2
assert not ({t["task_name"] for t in SCALEGEN_TASKS}
            & {t["task_name"] for t in TASKS}), "scalegen task_name collision"

# Token math for OpenWebText, worked so nothing here is a guess dressed as a
# measurement: max_steps(12000) * batch_size(32) * ctx(512) = 196,608,000
# tokens actually seen during a full training run. The prepared cache holds
# at least that many train tokens (220M) plus held-out val/test slices, so
# the run is close to a single pass over fresh data rather than repeated
# epochs over a too-small slice.


# ==========================================================================
#  DIAGNOSTICS -- one-question re-runs of an ALREADY FINISHED task with the
#  hyper-parameters PINNED instead of searched. Descriptive only: never part of
#  TASKS/the 12-task Wilcoxon, own opt_set, own output directory
#  (benchmark12/diagnostics/), own TASK_NAME so it cannot touch the finished
#  run's checkpoints.
#
#  language_ptb_evie_at_adamw_lr: on language_ptb the joint (lr, gamma) search
#  froze EvieKF at lr 1e-3 / gamma 10 on a 2500-step, single-seed score of
#  186.28 -- the best of all eight arms at the tuning budget -- and EvieKF then
#  finished 6th of 8 at the full 10000 steps (192.17 vs AdamW 179.35). The
#  gamma row at that lr is non-monotone with a 12.6% spread, i.e. an argmax over
#  noise, and the lr it carried (1e-3) is a half-decade above the one AdamW,
#  AdaBelief and Sophia all chose (3.162e-4). This file runs EvieKF at AdamW's
#  frozen lr, keeping gamma=10, over the same 5 seeds. Near AdamW => the PTB
#  loss is a tuning-protocol artefact and the paper says so; still near 192 =>
#  EvieKF is genuinely worse on PTB. Either answer is reportable; neither is
#  assumed.
# ==========================================================================
DIAG_TASKS = [
    dict(_BASE["language_ptb"],
         task_name="language_ptb_evie_at_adamw_lr",
         base_task_name="language_ptb",
         opt_set="diag",
         opt_list=["EvieKF"],
         pinned_lr={"EvieKF": {"lr": 3.162e-4, "knob": 10.0,
                               "note": "AdamW's frozen PTB lr; gamma kept at the "
                                       "main run's frozen 10"}},
         title=("Language / Penn Treebank / small GPT  --  DIAGNOSTIC: EvieKF "
                "at AdamW's frozen lr (3.162e-4), gamma=10, 5 seeds. NOT part "
                "of the 12-task significance table")),
]

# --------------------------------------------------------------------------
#  B1 -- "same tuning budget as the baselines": EvieKF with gamma FIXED at
#  EVIEKF_SWEEP_GAMMA (3000) and the ordinary 7-point half-decade LR search,
#  instead of the joint (lr, gamma) grid the protocol runs (49-99 points vs
#  AdamW's 7). Reviewers will read the joint search as an unfair tuning
#  advantage; this row answers that directly, and doubles as the practitioner
#  default ("use gamma=3000, tune lr like you always do").
#
#  Hosts: the four tasks where the joint search's gamma pick looks like an
#  argmax over noise rather than a signal -- PTB (gamma 10, 6th of 8),
#  STL-10 (gamma 3, the operator nearly off, 6th of 8), SVHN (gamma 10, 7th of
#  8) and CIFAR-10 (gamma 27000, a boundary-extended pick). Each is a SECONDARY
#  row: it never replaces that task's protocol number in Table 1 or in the
#  12-task Wilcoxon. 5 seeds, so it is comparable to the main run's cells.
# --------------------------------------------------------------------------
_B1_HOSTS = ["language_ptb", "vision_stl10", "vision_svhn", "vision_cifar10"]
_B1_GAMMA = 3000.0

for _hn in _B1_HOSTS:
    _b = _BASE[_hn]
    _cfg = dict(_b)
    _cfg["task_name"] = f"{_hn}_evie_fixed_gamma"
    _cfg["base_task_name"] = _hn
    _cfg["opt_set"] = "diag"
    _cfg["opt_list"] = ["EvieKF"]
    _cfg["fixed_knob"] = {"EvieKF": _B1_GAMMA}
    _cfg["title"] = (f"{_b['title']}  --  DIAGNOSTIC: EvieKF with gamma FIXED "
                     f"at {_B1_GAMMA:g} and a 7-point LR search only (the "
                     f"baselines' tuning budget), 5 seeds. NOT part of the "
                     f"12-task significance table")
    DIAG_TASKS.append(_cfg)

# --------------------------------------------------------------------------
#  B2 -- enwik8 at the gamma the protocol should have frozen.
#  FUTURE_WORK.md FW-2: the grid-reset-on-resume bug made the file freeze
#  gamma=27,000 although gamma=81,000 at the same lr had ALREADY been scored
#  and was 0.40% better (3.6956 vs 3.7103 in lr_search.json). Both values come
#  from the file's own search; nothing is re-tuned here, both axes are pinned,
#  so this measures only what the bug cost. A small effect is the expected
#  outcome; it removes a disclosure either way.
# --------------------------------------------------------------------------
DIAG_TASKS.append(dict(
    _BASE["language_enwik8"],
    task_name="language_enwik8_evie_gamma81k",
    base_task_name="language_enwik8",
    opt_set="diag",
    opt_list=["EvieKF"],
    pinned_lr={"EvieKF": {"lr": 3.162e-4, "knob": 81000.0,
                          "note": "the best point the file's own search scored "
                                  "(3.6956); the run froze gamma=27000 (3.7103) "
                                  "because of the grid-reset-on-resume bug, "
                                  "FUTURE_WORK FW-2"}},
    title=("Language / enwik8 (byte-level) / small GPT  --  DIAGNOSTIC: EvieKF "
           "at lr 3.162e-4, gamma=81000 (the best scored search point), 5 "
           "seeds. NOT part of the 12-task significance table"),
))

# --------------------------------------------------------------------------
#  B3 (FUTURE_WORK.md FW-6) -- the WikiText-2 model-size sweep's LARGE point
#  (9L / d384 / 12h), EvieKF at the lr AdamW and AdaBelief both froze
#  (3.162e-4) instead of the 1e-3 its own search picked, gamma at the sweep's
#  pinned 3000. In the sweep EvieKF lost by 17.6% at the large size, while its
#  own search scored those two lrs 108.8 vs 110.4 -- 1.5% apart. Seeds 42-44,
#  matching the sweep's cells, which are the comparison.
#
#  This runs on the MAIN engine, not the sweep engine: the model shape comes
#  from n_layer/d_model/n_head, which frag_language_data.build_model reads from
#  TASK. One unit, so the two engines' aggregation difference does not apply.
# --------------------------------------------------------------------------
DIAG_TASKS.append(dict(
    _BASE["language_wikitext2"],
    task_name="language_wikitext2_large_evie_at_adamw_lr",
    base_task_name="language_wikitext2",
    opt_set="diag",
    opt_list=["EvieKF"],
    n_layer=9, d_model=384, n_head=12,     # the sweep's "large" point
    seeds=[42, 43, 44],                     # the sweep's seeds
    pinned_lr={"EvieKF": {"lr": 3.162e-4, "knob": 3000.0,
                          "note": "the lr AdamW and AdaBelief both froze at the "
                                  "large size; gamma at the sweep's pinned 3000"}},
    title=("Language / WikiText-2 / LARGE GPT (9L, d384, 12h)  --  DIAGNOSTIC: "
           "EvieKF at AdamW's frozen lr (3.162e-4), gamma=3000, 3 seeds. NOT "
           "part of the 12-task significance table"),
))

# --------------------------------------------------------------------------
#  C1 -- is vision's flat result caused by the Kronecker size limit?
#  EVIEKF_MAXF = 1024 sends any factor larger than that to a diagonal
#  approximation. A CIFAR-stem ResNet-18's conv weights reshape to
#  (out, in*k*k), so the input-side factors are 576 (layer1), 1152 (layer2),
#  2304 (layer3) and 4608 (layer4): at the frozen limit, the layers holding
#  nearly all of ResNet-18's parameters run HALF-DIAGONAL. That fits the
#  results -- the Kronecker structure was significant in language and finance,
#  and vision is where EvieKF looks like Adam-plus-a-bit and the ablation could
#  not tell. This raises the limit to 4608 (every factor fully Kronecker) with
#  BOTH axes pinned at the main run's frozen values, so the only difference
#  from the main CIFAR-10 EvieKF cells is the limit, and those cells are the
#  control -- nothing needs re-running for the comparison.
#
#  Cost warning: a 4608x4608 float64 eigendecomposition every 20 steps. Expect
#  several times the main run's 1765 s/cell; budget two Kaggle sessions.
# --------------------------------------------------------------------------
DIAG_TASKS.append(dict(
    _BASE["vision_cifar10"],
    task_name="vision_cifar10_evie_maxf4608",
    base_task_name="vision_cifar10",
    opt_set="diag",
    opt_list=["EvieKF"],
    eviekf_maxf=4608,                       # frag_shared_optim.build_optimizer
    ckpt_every=500,                          # slow cells -- checkpoint often
    pinned_lr={"EvieKF": {"lr": 0.09999, "knob": 27000.0,
                          "note": "the main run's frozen CIFAR-10 (lr, gamma); "
                                  "the ONLY difference from those cells is "
                                  "eviekf_maxf 1024 -> 4608"}},
    title=("Vision / CIFAR-10 / CIFAR-stem ResNet-18  --  DIAGNOSTIC: EvieKF "
           "with the Kronecker size limit raised 1024 -> 4608 (every conv "
           "factor fully Kronecker), lr and gamma pinned at the main run's, "
           "5 seeds. NOT part of the 12-task significance table"),
))


# --------------------------------------------------------------------------
#  FW-11 -- deconfound C2, the claim the paper now rests on.
#
#  The ablation's EvieDiag gates with (1 + gamma*s/mean(s))^-1/2 (RELATIVE) while
#  EvieKF gates with mu*lam/tr(A) (ABSOLUTE). So "Kronecker beats diagonal" is
#  currently also "absolute normalisation beats relative", and an independent
#  gamma search per arm does not separate them: the two conventions differ by the
#  mean eigenvalue, which varies per layer and over training, so no single gamma
#  maps one onto the other. With C1 (centring) null, C2 carries the submission
#  (DERIVATION.md sec 5.1 and sec 10).
#
#  EvieDiagAbs is the derivation's OWN diagonal special case: with Sigma
#  diagonal, the operator gives (1 + gamma*diag(Sigma_z))^-1/2 with no mean
#  normalisation. The full spec states s/mean(s) only for 1-D parameters; the
#  shipped EvieDiag extends that to every parameter, which the spec never says to
#  do. So this arm is both the deconfounded comparison AND the more faithful one.
#
#  Only the new arm runs. EvieKF's and EvieKFu's 10-seed cells already exist in
#  benchmark12/results/ablation/<host>/ at the same seeds, same task, same
#  engine, and are the comparison -- do not recompute them (that is ~10 h on the
#  language host alone). The ANALYSIS file fills that column from the existing
#  summary.json, the same way the sweeps reuse their baseline column.
# --------------------------------------------------------------------------
_FW11_HOSTS = ["finance_tech", "language_wikitext2"]

for _hn in _FW11_HOSTS:
    _b = _BASE[_hn]
    _cfg = dict(_b)
    _cfg["task_name"] = f"ablation_{_hn}_diag_absolute"
    _cfg["base_task_name"] = _hn
    _cfg["opt_set"] = "diag"
    _cfg["opt_list"] = ["EvieDiagAbs"]
    _cfg["seeds"] = [42, 43, 44, 45, 46, 47, 48, 49, 50, 51]   # the ablation's 10
    _cfg["title"] = (f"{_b['title']}  --  DIAGNOSTIC: EvieDiag with the "
                     f"derivation's absolute gate (no mean normalisation), 10 "
                     f"seeds, own (lr, gamma) search. Deconfounds C2 against the "
                     f"existing ablation's EvieKF cells. NOT part of the 12-task "
                     f"significance table")
    DIAG_TASKS.append(_cfg)

# --------------------------------------------------------------------------
#  FW-9 -- is gamma dimensionless once the spectrum is normalised properly?
#
#  `/tau = /tr(A)` matches the Kronecker product's trace to tr(Sigma_z), which is
#  necessary, but it is NOT scale-free: scaling the noise by c scales mu*lam/tau
#  by c^2. That is why the frozen gamma spans 3 -> 243,000 across the 12 tasks
#  while the spec's grid was {3, 10, 30, 100}. EvieKFm normalises by the MEAN
#  eigenvalue instead, in which form gamma genuinely is dimensionless.
#
#  The question is TRANSFER, so lr is pinned at the value the protocol already
#  froze for EvieKF on each host and only gamma is searched, over the re-centred
#  7-point grid. Three hosts spanning two domains and three very different frozen
#  gammas (300 / 10 / 27,000): if EvieKFm lands on the same gamma at all three,
#  gamma stops being a tuned axis.
#
#  Caveat to carry into the write-up: the operator changes, so its preferred lr
#  may differ from the pinned one. This is a transfer test at the protocol's lr,
#  not a re-tuned comparison, and it can never replace a protocol number.
# --------------------------------------------------------------------------
_FW9_HOSTS = {                       # host -> the lr EvieKF froze there
    "language_wikitext2": 1e-3,
    "language_ptb": 1e-3,
    "vision_cifar10": 0.09999,
}

for _hn, _lr in _FW9_HOSTS.items():
    _b = _BASE[_hn]
    _cfg = dict(_b)
    _cfg["task_name"] = f"{_hn}_gamma_meannorm"
    _cfg["base_task_name"] = _hn
    _cfg["opt_set"] = "diag"
    _cfg["opt_list"] = ["EvieKFm"]
    _cfg["fixed_lr"] = {"EvieKFm": float(_lr)}
    _cfg["title"] = (f"{_b['title']}  --  DIAGNOSTIC: EvieKF with the Kronecker "
                     f"spectrum normalised by its mean eigenvalue (gamma "
                     f"dimensionless), lr pinned at the protocol's {_lr:g}, gamma "
                     f"searched. Tests whether gamma transfers across tasks. NOT "
                     f"part of the 12-task significance table")
    DIAG_TASKS.append(_cfg)

# --------------------------------------------------------------------------
#  FW-10 -- is norm preservation load-bearing?
#
#  Every gain is <= 1, so the operator alone can only shrink; the shipped code
#  then rescales the whole update back to the Adam direction's norm, so it
#  ROTATES rather than shrinks. The full spec calls that "the whole reason the
#  method can exist at all", and it is the answer to the reviewer who suspects a
#  disguised learning-rate schedule -- but it appears in the ALGORITHM, not in
#  the derivation, and has never been ablated (DERIVATION.md sec 6, sec 9).
#
#  EvieKFr is EvieKF with the rescale disabled, both axes pinned at the main
#  run's frozen values, so the main run's own EvieKF cells are the control and
#  nothing else needs re-running.
# --------------------------------------------------------------------------
DIAG_TASKS.append(dict(
    _BASE["language_wikitext2"],
    task_name="language_wikitext2_evie_no_rescale",
    base_task_name="language_wikitext2",
    opt_set="diag",
    opt_list=["EvieKFr"],
    pinned_lr={"EvieKFr": {"lr": 1e-3, "knob": 300.0,
                           "note": "the main run's frozen WikiText-2 (lr, gamma); "
                                   "the ONLY difference from those cells is that "
                                   "the global ||y||/||z|| rescale is off"}},
    title=("Language / WikiText-2 / small GPT  --  DIAGNOSTIC: EvieKF with the "
           "global norm-preserving rescale DISABLED (shrink instead of rotate), "
           "lr and gamma pinned at the main run's, 5 seeds. NOT part of the "
           "12-task significance table"),
))


for _t in DIAG_TASKS:
    for _k, _v in COMMON.items():
        _t.setdefault(_k, _v)

assert len({t["task_name"] for t in DIAG_TASKS}) == len(DIAG_TASKS)
assert not ({t["task_name"] for t in DIAG_TASKS}
            & {t["task_name"] for t in TASKS}), "diagnostic task_name collision"



# Muon hybrid R1: explicit, budget-matched searches; never part of main 12.
MUONHYBRID_ARMS = ["MuonSub", "MuonEvie", "MuonEvieFix"]
_MUONHYBRID_ANCHORS = {'language_wikitext2': (0.006325, 0.001, [0.00031622776601683794, 0.001, 0.0031622776601683794, 0.01]), 'language_wikitext103': (0.002, 0.001, [0.00031622776601683794, 0.001, 0.0031622776601683794, 0.01]), 'language_enwik8': (0.006325, 0.0003162, [9.999121961452416e-05, 0.0003162, 0.0009999121961452415, 0.003162]), 'language_openwebtext': (0.002, 0.001, [0.00031622776601683794, 0.001, 0.0031622776601683794, 0.01])}
MUONHYBRID_TASKS = []
for _name, (_matrix_lr, _anchor, _lrs) in _MUONHYBRID_ANCHORS.items():
    _owt = _name == "language_openwebtext"
    _base = (next(t for t in SCALEGEN_TASKS if t["task_name"] == _name)
             if _owt else _BASE[_name])
    _cfg = dict(_base)
    _cfg.update(task_name=_name + "_muonhybrid", base_task_name=_name,
                opt_set="muonhybrid", opt_list=MUONHYBRID_ARMS[:2] if _owt else list(MUONHYBRID_ARMS),
                force_split_path=True, allow_tf32=True, ref_opt="MuonSub",
                muon_matrix_lr=_matrix_lr, max_steps=6000 if _owt else 10000,
                lr_search_steps=1500 if _owt else 2500,
                seeds=[42,43,44] if _owt else [42,43,44,45,46],
                search_grid={"MuonSub": {"lr": list(_lrs), "knob": [1e-4] if _owt else [1e-4,1e-3,1e-2]},
                             "MuonEvie": {"lr": list(_lrs), "knob": [30.] if _owt else [3.,30.,300.]}},
                title=_base["title"] + " -- Muon hybrid R1" +
                      ("; DEVIATION: 1500-step search, 6000-step full run" if _owt else ""))
    MUONHYBRID_TASKS.append(_cfg)

# Validate frozen anchors against authoritative artifacts when available.
import json as _json
from pathlib import Path as _Path
for _name, (_matrix_lr, _anchor, _lrs) in _MUONHYBRID_ANCHORS.items():
    _r = _Path(__file__).resolve().parents[1] / "results" / (
        "scalegen/" + _name if _name == "language_openwebtext" else _name)
    if (_r / "lr_search.json").exists():
        _d = _json.loads((_r / "lr_search.json").read_text())
        if "AdamW" in _d:
            assert _d["AdamW"]["lr"] == _anchor, "frozen AdamW anchor changed"
        elif (_r / "final_cells/AdamW/results.json").exists():
            assert _json.loads((_r / "final_cells/AdamW/results.json").read_text())["meta"]["lr_used"]["AdamW"] == _anchor
        if "Muon" in _d:
            assert _d["Muon"]["lr"] == _matrix_lr, "frozen Muon anchor changed"
        elif (_r / "final_cells/Muon/results.json").exists():
            _cells = _json.loads((_r / "final_cells/Muon/results.json").read_text())["cells"]
            assert {_c["lr_used"] for _c in _cells.values() if _c["opt"] == "Muon"} == {_matrix_lr}

ALL_TASKS = TASKS + ABLATION_TASKS + SWEEP_TASKS + SCALEGEN_TASKS + DIAG_TASKS + MUONHYBRID_TASKS
assert len({t["task_name"] for t in ALL_TASKS}) == len(ALL_TASKS)

# Master rebuttal tasks, isolated from the frozen main run.
import ast as _ast

def _shared_literal(name):
    tree = _ast.parse((_Path(__file__).parent / "frag_shared_optim.py").read_text())
    for node in tree.body:
        if isinstance(node, _ast.Assign) and any(isinstance(t, _ast.Name) and t.id == name for t in node.targets):
            return _ast.literal_eval(node.value)
    raise KeyError(name)

# The optimizer fragment is not an importable module (it expects torch/nn and
# header globals). Read the authoritative literal instead of importing it.
EVIEKF_FIXED_GAMMA_M = _shared_literal("EVIEKF_FIXED_GAMMA_M")
_CENTRES = _shared_literal("LR_GRID_CENTRE")

def _frozen(task_dir, arm, results_root=None):
    root = _Path(results_root) if results_root is not None else _Path(__file__).resolve().parents[1]/"results"
    path = root/task_dir/"lr_search.json"
    if path.exists():
        d = _json.loads(path.read_text())
        if arm in d and "lr" in d[arm]:
            return float(d[arm]["lr"]), d[arm].get("knob")
    final = root/task_dir/"final_cells"/arm/"results.json"
    if final.exists():
        d = _json.loads(final.read_text())
        meta = d.get("meta", {})
        if arm in meta.get("lr_used", {}):
            return float(meta["lr_used"][arm]), meta.get("knob_used", {}).get(arm)
        # Actual OWT Muon metadata omits Muon. Recorded cells and its search
        # log agree on one LR; require unique cell values, never invent a default.
        vals = {(float(c["lr_used"]), c.get("knob_used")) for c in d.get("cells", {}).values()
                if c.get("opt") == arm and "lr_used" in c}
        if len(vals) == 1:
            return vals.pop()
    raise FileNotFoundError(f"Frozen {arm} unavailable in {path} or {final}")


def _diag_from(base_name, new_name, arms, pinned=None, fixed_knob=None, **overrides):
    bases = {t["task_name"]: t for t in TASKS + ABLATION_TASKS + SCALEGEN_TASKS}
    cfg = dict(bases[base_name])
    cfg.update(task_name=new_name, base_task_name=base_name, opt_set="diag",
               opt_list=list(arms), pinned_lr=pinned or {}, fixed_knob=fixed_knob or {},
               extra_metrics=True, rebuttal_run=True)
    cfg.update(overrides)
    return cfg

REBUTTAL_TASKS = []
assert len(_BASE) == 12
for _name, _base in _BASE.items():
    REBUTTAL_TASKS.append(_diag_from(_name, _name+"_evie_m30", ["EvieKFm"],
        fixed_knob={"EvieKFm": EVIEKF_FIXED_GAMMA_M}, budget_no_extend=True,
        title=_base["title"]+" -- R3a: fixed dimensionless gamma=30, exactly 7 LR trials; no boundary extension"))
    _lr, _kn = _frozen(_name, "AdamW")
    REBUTTAL_TASKS.append(_diag_from(_name, _name+"_ref_adamw", ["AdamW"],
        pinned={"AdamW":dict(lr=_lr,knob=None,note="main-run frozen AdamW LR; optional metrics reference")},
        title=_base["title"]+" -- R3a-ref: optional reference rerun with exploratory metrics"))
# R2 weight-decay values, per optimizer family (amended 2026-11-09; see
# R2_WEIGHT_DECAY_NOTE.md at the repo root). Decay here is decoupled,
# p <- p * (1 - lr * wd), so its strength per step is lr * wd: the same wd
# number is ~30x stronger for SGD (lr ~3e-2) than for AdamW (lr ~1e-3).
# Each arm keeps 2 values (10 trials, unchanged): the paper's 1e-4 as the
# anchor, plus its family's standard CIFAR strength --
#   AdamW, EvieKF : 5e-2  (common AdamW value; PyTorch's default is 1e-2)
#   SGD           : 5e-3  (= textbook coupled 5e-4 through momentum 0.9,
#                          which multiplies its effect by 1/(1-0.9) = 10)
#   Muon          : 5e-3  (matrix lr ~2e-2, close to SGD's, so the same value
#                          gives a similar per-step shrink)
_R2_WD = {"AdamWwd": [1e-4, 5e-2], "EvieKFmwd": [1e-4, 5e-2],
          "SGDwd": [1e-4, 5e-3], "Muonwd": [1e-4, 5e-3]}
for _name in ["vision_cifar100", "vision_cifar10"]:
    _arms=["AdamWwd","SGDwd","Muonwd","EvieKFmwd"]
    REBUTTAL_TASKS.append(_diag_from(_name,_name+"_long",_arms,
        max_steps=30000,lr_search_steps=3000,seeds=[42,43,44],allow_tf32=True,eval_every=1000,
        title=_BASE[_name]["title"]+" -- R2: long schedule (30k steps, about 77 epochs); search 10% of budget, 10 trials per arm; weight decay per optimizer family",
        search_grid={a:dict(lr=[_CENTRES[a]*10**(k/2) for k in [-2,-1,0,1,2]],knob=list(_R2_WD[a])) for a in _arms}))
for _name, _arms in [("language_wikitext2",["AdamWwd","AdaBeliefwd","Lionwd","Muonwd"]),
                       ("finance_tech",["AdamWwd","AdaBeliefwd","Lionwd"])]:
    REBUTTAL_TASKS.append(_diag_from(_name,_name+"_basebudget",_arms,
        seeds=[42,43,44],budget_no_extend=True,
        title=_BASE[_name]["title"]+" -- R3b: exactly 7x7 LR/weight-decay trials; no boundary extension"))
for _host in ["finance_tech","language_wikitext2"]:
    _base=next(t for t in ABLATION_TASKS if t["task_name"] == "ablation_"+_host)
    _lr,_kn=_frozen("ablation/"+_base["task_name"],"EvieKF")
    REBUTTAL_TASKS.append(_diag_from(_base["task_name"],_base["task_name"]+"_controls",
        ["EvieKFlag","EvieKFrb"],
        pinned={a:dict(lr=_lr,knob=_kn,note="same lr and gamma as EvieKF; one change only") for a in ["EvieKFlag","EvieKFrb"]},
        title=_base["title"]+" -- R4: controls reuse EvieKF's tuned values, slightly favouring EvieKF"))
_owt=next(t for t in SCALEGEN_TASKS if t["task_name"]=="language_openwebtext")
_grids={}
for _arm in ["AdamW","Muon","EvieKF"]:
    _lr,_kn=_frozen("scalegen/language_openwebtext",_arm)
    _grids[_arm]=dict(lr=[f*_lr for f in [.25,.5,1.]],knob=[_kn])
REBUTTAL_TASKS.append(_diag_from("language_openwebtext","language_openwebtext_124m",
    ["AdamW","Muon","EvieKF"],n_layer=12,d_model=768,n_head=12,
    seeds=[42,43,44],allow_tf32=True,force_split_path=True,assert_param_range=[110000000,140000000],
    search_grid=_grids,title=_owt["title"]+" -- R5: 12L/d768/12 heads, approximately 124M; within-protocol scale check"))
# R2b (2026-11-11): R2 again with ONE change, fresh_epochs=True. The engine's
# default itertools.cycle(train_loader) replays the first epoch's batches, so
# R2 (and every earlier vision run) trained on one fixed set of augmented
# images; R2 memorised it (train loss ~1e-6) and stopped at 51-56% on
# CIFAR-100. Here the loader is re-iterated every epoch: new shuffle, new
# crops/flips. Arms, grids, steps, seeds and weight-decay values are R2's.
# Boundary extensions (2026-11-11). R2's fixed 5-point grid never extends, so an
# arm whose pick lands on the top edge gets ONE more half-decade point above it,
# at both wd values. The trials already scored are reused (each extension job's
# lr file carries them); only the two new (lr, wd) trials run. Applied to every
# arm that hit the edge, baseline or Evie-KF alike:
#   CIFAR-10 AdamWwd 1e-2 (+3.162e-2; done, pick unchanged),
#   CIFAR-100 SGDwd 0.3 (+0.9487), CIFAR-100 EvieKFmwd 1e-2 (+3.162e-2),
#   CIFAR-10 EvieKFmwd 1e-2 (+3.162e-2).
_R2B_EXTEND = {("vision_cifar10","AdamWwd"), ("vision_cifar100","SGDwd"),
               ("vision_cifar100","EvieKFmwd"), ("vision_cifar10","EvieKFmwd")}
def _r2b_lr_steps(name, arm):
    return [-2,-1,0,1,2,3] if (name, arm) in _R2B_EXTEND else [-2,-1,0,1,2]
for _name in ["vision_cifar100", "vision_cifar10"]:
    _arms=["AdamWwd","SGDwd","Muonwd","EvieKFmwd"]
    REBUTTAL_TASKS.append(_diag_from(_name,_name+"_long_fresh",_arms,
        max_steps=30000,lr_search_steps=3000,seeds=[42,43,44],allow_tf32=True,eval_every=1000,
        fresh_epochs=True,
        title=_BASE[_name]["title"]+" -- R2b: R2's long schedule (30k steps, about 77 epochs; search 10% of budget, 10 trials per arm; weight decay per optimizer family) with fresh shuffling and augmentation every epoch",
        search_grid={a:dict(lr=[_CENTRES[a]*10**(k/2) for k in _r2b_lr_steps(_name,a)],knob=list(_R2_WD[a])) for a in _arms}))
# R5b (2026-11-11): R5's Evie-KF arm again with ONE change: the R3a variant
# (mean-normalised noise spectrum, gamma fixed at 30) instead of the absolute
# gamma=27000 carried over from the 33.7M run. Same model, steps, seeds and
# the same three learning rates R5 searched for EvieKF. AdamW and Muon are not
# rerun: their R5 cells are the comparison.
REBUTTAL_TASKS.append(_diag_from("language_openwebtext","language_openwebtext_124m_evie_m30",
    ["EvieKFm"],n_layer=12,d_model=768,n_head=12,
    seeds=[42,43,44],allow_tf32=True,force_split_path=True,assert_param_range=[110000000,140000000],
    search_grid={"EvieKFm":dict(lr=list(_grids["EvieKF"]["lr"]),knob=[EVIEKF_FIXED_GAMMA_M])},
    title=_owt["title"]+" -- R5b: 12L/d768/12 heads, approximately 124M; Evie-KF with mean-normalised gamma fixed at 30, R5's three learning rates"))
# R5c (2026-11-12): R5b again with ONE change, the Kronecker factor cap raised
# from the frozen EVIEKF_MAXF=1024 to 3072. At 1024 only the 768x768 attention
# projections were gated (6.0% of weights; R5 telemetry); 3072 is the smallest
# cap that gates every transformer-block matrix (2304x768 qkv, 768x768 proj,
# 3072x768 and 768x3072 MLP): 68.8% of weights. Only the 50257x768 embedding
# and 1-D parameters stay on the diagonal fallback. Diagnostic only (EVIEKF_MAXF
# stays frozen for protocol tasks). Same lr grid, gamma, steps, seeds as R5b.
REBUTTAL_TASKS.append(_diag_from("language_openwebtext","language_openwebtext_124m_evie_m30_cap3072",
    ["EvieKFm"],n_layer=12,d_model=768,n_head=12,
    seeds=[42,43,44],allow_tf32=True,force_split_path=True,assert_param_range=[110000000,140000000],
    eviekf_maxf=3072,
    search_grid={"EvieKFm":dict(lr=list(_grids["EvieKF"]["lr"]),knob=[EVIEKF_FIXED_GAMMA_M])},
    title=_owt["title"]+" -- R5c: as R5b (124M, Evie-KF gamma 30) with the Kronecker factor cap raised to 3072 (68.8% of weights gated, vs 6.0%)"))
# Loader check outside vision (2026-11-11): WikiText-2 has no augmentation, so
# the replayed first epoch only fixes the batch composition and order. This
# reruns AdamW (main-run lr) and Evie-KF gamma 30 (R3a lr) with fresh_epochs
# as the ONE change, 5 seeds, nothing searched, to show whether the language
# comparison moves. References: main AdamW and language_wikitext2_evie_m30.
_lr_a,_kn_a=_frozen("language_wikitext2","AdamW")
_m30=_json.loads((_Path(__file__).resolve().parents[1]/"results/rebuttal/merged/language_wikitext2_evie_m30/lr_search.json").read_text())["EvieKFm"]
assert _m30.get("complete") and float(_m30["knob"])==EVIEKF_FIXED_GAMMA_M
REBUTTAL_TASKS.append(_diag_from("language_wikitext2","language_wikitext2_fresh",["AdamW","EvieKFm"],
    pinned={"AdamW":dict(lr=_lr_a,knob=None,note="main-run frozen AdamW LR"),
            "EvieKFm":dict(lr=float(_m30["lr"]),knob=float(_m30["knob"]),note="R3a frozen LR at fixed gamma 30")},
    fresh_epochs=True,
    title=_BASE["language_wikitext2"]["title"]+" -- loader check: fresh shuffling every epoch; AdamW and Evie-KF gamma 30 at their frozen learning rates"))
# R4b (2026-11-11): random-basis control under the R3a variant, on the host
# where gamma 30 gains most (CIFAR-100: 55.36 vs AdamW 49.51). R4's
# random-basis arm matched EvieKF on WikiText-2 and Finance-Tech, so those
# hosts do not show that the gate needs the estimated noise eigenvectors.
# EvieKFmrb = EvieKFm with every Kronecker eigenbasis replaced by a fixed
# random orthogonal one (same eigenvalues). Pinned to EvieKFm's R3a lr and
# gamma 30; same seeds, steps and (default, replayed) loader as
# vision_cifar100_evie_m30, whose cells are the reference.
_m30c=_json.loads((_Path(__file__).resolve().parents[1]/"results/rebuttal/merged/vision_cifar100_evie_m30/lr_search.json").read_text())["EvieKFm"]
assert _m30c.get("complete") and float(_m30c["knob"])==EVIEKF_FIXED_GAMMA_M
REBUTTAL_TASKS.append(_diag_from("vision_cifar100","vision_cifar100_evie_m30_rb",["EvieKFmrb"],
    pinned={"EvieKFmrb":dict(lr=float(_m30c["lr"]),knob=float(_m30c["knob"]),note="same lr and gamma as EvieKFm in R3a; one change only (random eigenbasis)")},
    title=_BASE["vision_cifar100"]["title"]+" -- R4b: random-basis control for Evie-KF gamma 30 (same eigenvalues, random eigenvectors); pinned to R3a's values"))
for _task in MUONHYBRID_TASKS:
    _task["extra_metrics"]=True
_holdout_manifest = _json.loads(
    (_Path(__file__).resolve().parents[1] /
     "finance/data/followups/finance_financials_2025/manifest.json").read_text()
)
REBUTTAL_TASKS.append(_diag_from(
    "finance_finance", "finance_financials_2025", ["AdamW", "Muon", "SGD", "EvieKFm"],
    panel="finance_financials_2025", finance_holdout=True,
    finance_sha256=_holdout_manifest["sha256"],
    fixed_knob={"EvieKFm": EVIEKF_FIXED_GAMMA_M}, budget_no_extend=True,
    title="Financials next-day return forecasting, 2025 held-out window",
))
_original_split_manifest = _json.loads(
    (_Path(__file__).resolve().parents[1] /
     "finance/data/followups/finance_financials_original_split/manifest.json").read_text()
)
_original_split_task = dict(REBUTTAL_TASKS[-1])
_original_split_task.update(
    task_name="finance_financials_original_split",
    panel="finance_financials_original_split",
    finance_original_split=True,
    finance_sha256=_original_split_manifest["sha256"],
    title="Financials forecasting, original data and split under the follow-up protocol",
)
REBUTTAL_TASKS.append(_original_split_task)
ALL_TASKS = TASKS+ABLATION_TASKS+SWEEP_TASKS+SCALEGEN_TASKS+DIAG_TASKS+MUONHYBRID_TASKS+REBUTTAL_TASKS
assert len(REBUTTAL_TASKS)==39
assert len({t["task_name"] for t in ALL_TASKS})==len(ALL_TASKS)
