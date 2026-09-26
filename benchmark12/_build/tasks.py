"""
The 12 frozen ICLR-benchmark tasks (CLAUDE.md sec 3). build.py consumes this.

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

# CLAUDE.md sec 9.5: WikiText-2 and PTB use a 16k whitespace-tokenized vocab and
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
#  PHASE 2 (CLAUDE.md sec 9) -- batch-size + model-size sweeps.
#  6 host tasks x 2 sweep types = 12 generated files under benchmark12/sweeps/.
#  Reuse the same fragments; each file gets its own checkpoints/<task>_<type>/.
# ==========================================================================
SWEEP_HOST_NAMES = ["vision_cifar10", "vision_svhn",
                    "language_wikitext2", "language_ptb",
                    "finance_tech", "finance_healthcare"]

_BASE = {t["task_name"]: t for t in TASKS}


# ==========================================================================
#  EVIE-KF ABLATION (claude_optim.md sec 2) -- the C1 (centred vs uncentred)
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
#  Structure decision (documented for the user): claude_optim.md sec 2 says
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
_MODEL_GRID = {           # small / (baseline reused) / large -- CLAUDE.md sec 9.3
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


ALL_TASKS = TASKS + ABLATION_TASKS + SWEEP_TASKS + SCALEGEN_TASKS + DIAG_TASKS
