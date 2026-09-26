# === SHARED OPTIMIZER BLOCK -- keep identical across all 12 files ===
# ---------------------------------------------------------------------------
# This block is the single source of truth for every baseline optimizer used
# in the ICLR benchmark harness. It is pasted BYTE-FOR-BYTE IDENTICAL into all
# 12 task files by benchmark12/_build/build.py. Do NOT hand-edit it inside a
# generated file -- edit benchmark12/_build/frag_shared_optim.py and re-run
# build.py so the change lands in all 12 at once (see CLAUDE.md sec 2).
#
# Contract with the rest of the file (defined above this block, in the header):
#   - torch, torch.nn as nn, math, numpy as np  are imported
#   - WEIGHT_DECAY            : float  (decoupled weight decay, all optimizers)
#   - GRAD_CLIP_NORM          : float  (global grad-norm clip, applied by engine)
#   - HESS_FREQ, N_HUTCH      : int    (Sophia Hutchinson probe cadence / count)
#   - log(msg)                : function (stdout + logfile)
#
# Optimizers provided here: AdaBelief, SGDM (SGD+momentum+Nesterov), Sophia,
# Muon (hybrid: Newton-Schulz on >=2D params, AdamW on the rest), Lion.
# AdamW itself is torch.optim.AdamW (stdlib) and is built in build_optimizer().
#
# Weight decay convention (CLAUDE.md sec 4): decoupled, applied as
#   p <- p * (1 - lr * lambda)
# BEFORE the gradient-based update, never as L2-in-loss. Every custom optimizer
# below implements it that way; torch.optim.AdamW already does.
# ---------------------------------------------------------------------------

# Optimizer-family hyperparameters (frozen 2026-09-08, CLAUDE.md sec 4).
SOPHIA_BETAS      = (0.965, 0.99)   # (grad EMA, Hessian EMA)
SOPHIA_RHO        = 0.04
SOPHIA_EPS        = 1e-8
SOPHIA_BETA_H     = 0.99            # Hessian EMA used inside set_hessian()
ADABELIEF_BETAS   = (0.9, 0.999)
ADABELIEF_EPS     = 1e-16          # AdaBelief paper default
SGDM_MOMENTUM     = 0.9
SGDM_NESTEROV     = True
LION_BETAS        = (0.9, 0.99)
MUON_MOMENTUM     = 0.95
MUON_NESTEROV     = True
MUON_NS_STEPS     = 5
MUON_NS_COEFFS    = (3.4445, -4.7750, 2.0315)   # Newton-Schulz quintic (Keller Jordan)
MUON_UPDATE_SCALE = 0.2            # update <- MUON_UPDATE_SCALE * sqrt(max(fan_out, fan_in)) * orthogonalized
MUON_ADAMW_BETAS  = (0.9, 0.999)  # AdamW sub-group inside the Muon hybrid
MUON_ADAMW_EPS    = 1e-8

# --- Shampoo (per-layer, AdamW-norm-grafted) -- the "matrix vs diagonal"
#     baseline the Evie-KF ablation needs (claude_optim.md sec 3). Ungrafted
#     Shampoo is unstable at this scale; we graft the AdamW update norm.
SHAMPOO_BETA        = 0.95         # preconditioner EMA (tuned jointly with LR)
SHAMPOO_EPS         = 1e-6
SHAMPOO_UPDATE_FREQ = 20           # steps between L^-1/4 / R^-1/4 refreshes
SHAMPOO_GRAFT_BETAS = (0.9, 0.999) # AdamW graft moments
SHAMPOO_MAXF        = 1024         # 2-D params with a dim > this fall to the graft
SHAMPOO_KNOB_GRID   = [0.9, 0.95, 0.99, 0.999]   # `beta`, the second tuned axis

# --- Evie-KF: Kronecker-factored risk-sensitive preconditioner
#     (claude_optim.md sec 4.1; full derivation in evie-kf-full-specification.md).
#     Numerics are ground-truthed against the validated flat-vector reference in
#     "Claude outputs/kaggle_evie_v3.py" -- see
#     benchmark12/tests/test_eviekf.py::test_parity_vs_kaggle_reference.
#
#     One class, four flags, four rungs of the claim ladder:
#       gamma=0        -> AdamW, exactly (bit-identical by construction)
#       diag_only=True -> Evie-diag  (Sigma diagonal; SDProp-2017 / AdaBelief family)
#       centered=False -> Evie-KFu   (Sigma = E[gg^T], the UNCENTRED object every
#                                     published Kronecker method uses: K-FAC,
#                                     Shampoo, SOAP)  -- the C1 novelty ablation
#       per_layer_norm -> Evie-KFn   (gains rescaled to unit RMS per layer;
#                                     the WikiText-2 PoC winner, the main-table entrant)
EVIEKF_BETAS       = (0.9, 0.999)
EVIEKF_EPS         = 1e-8
EVIEKF_BETA_SIG    = 0.95     # frozen -- noise-covariance EMA
EVIEKF_WARMUP      = 20       # frozen -- operator off for the first WARMUP steps
EVIEKF_REFRESH     = 20       # frozen -- steps between eigendecomposition refreshes
EVIEKF_MAXF        = 1024     # frozen -- a Kronecker factor larger than this uses
                             #           a diagonal approximation (no eigh). On a
                             #           CIFAR-stem ResNet-18 the reshaped conv
                             #           factors are 576/1152/2304/4608, so at this
                             #           limit layers 2-4 -- nearly all the
                             #           parameters -- run half-diagonal. A
                             #           DIAGNOSTIC file may raise it via
                             #           TASK["eviekf_maxf"] (passed in as
                             #           build_optimizer(eviekf_maxf=...)); no
                             #           protocol task does, and the paper must
                             #           state the limit either way.
EVIEKF_GAIN_CLIP   = 5.0      # frozen -- per_layer_norm gain cap
EVIEKF_GAMMA_GRID  = [3.0, 10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0]  # TUNED: the one knob
# Fixed gamma for the Phase-2 sweeps, which re-search LR per grid point but do
# NOT joint-search gamma (that would ~7x the sweep cost). build_optimizer()'s
# own default is 100.0, but kaggle_evie_v3.py's joint search -- the run whose
# numbers go in the paper -- pinned the optimum at the TOP of the grid (3000),
# ~30x higher. The batch-size sweep exists to detect the C1 gap widening at
# small batch; running Evie's arms at a gamma the evidence says is far
# off-optimum risks masking that effect, so the sweeps use 3000, not 100.
# (Top-edge-pinned even at PoC scale -- a real joint search is still the right
# fix if the compute budget allows it; see RUNBOOK sec 5 / sec 7.)
EVIEKF_SWEEP_GAMMA  = 3000.0

# Diagnostic-only gamma grid for the MEAN-NORMALISED Kronecker arm (EvieKFm,
# FUTURE_WORK FW-9). Under `/tau` the spectrum carries the absolute noise energy,
# so the tuned gamma has to absorb it and lands anywhere from 3 to 243,000 across
# the 12 tasks. Under `/mean(mu*lam)` the spectrum has mean eigenvalue 1, so
# gamma is genuinely dimensionless and should sit near O(1). Same 7-point
# half-decade discipline, re-centred -- exactly as LR_GRID_CENTRE differs per
# optimizer family.
EVIEKF_MEANNORM_GAMMA_GRID = [0.1, 0.3, 1.0, 3.0, 10.0, 30.0, 100.0]

# Optimizer registry. EvieKF (centered=True, per_layer_norm=False) fills the
# reserved slot -- it is the variant validated in kaggle_evie_v3.py (10 seeds,
# Holm-corrected: beats AdamW/AdaBelief/Lion/Muon/Shampoo/EvieDiag at p=0.0078,
# ties its own uncentred ablation EvieKFu). EvieKFn (per_layer_norm=True) stays
# defined in this file but is no longer selected anywhere -- it was the headline
# arm of an earlier 5-seed / no-Muon-no-Shampoo run and its per_layer_norm gains
# >1 are a divergence path plain EvieKF does not have.
# The ablation ladder (EvieDiag/EvieKFu/EvieKF) runs only in the generated
# benchmark12/ablation/ files (TASK["opt_set"] == "ablation"). EvieKF is
# therefore BOTH the main table's 8th arm AND one of the 3 ablation arms: the 3
# ablation hosts (wikitext2, cifar10, finance_tech) compute EvieKF twice -- once
# in the main file (5 seeds), once in the ablation file (10 seeds, a superset).
# ~15 extra cells total; a known, accepted redundancy, not de-duplicated here.
#
# ORDER MATTERS, for two reasons, and neither is cosmetic:
#  1. run_sweep() and lr_search() walk this list in order, so a run that is cut
#     short (a Kaggle session that never comes back, a cluster preemption) yields
#     a PREFIX of it. EvieKF used to be last, which meant a partial run produced
#     seven baselines and no Evie -- i.e. nothing. EvieKF and AdamW now come
#     first, so any partial result already contains the comparison the paper is
#     about. Cells are keyed by NAME ("opt|unit|seed"), never by index, so
#     reordering is safe for an in-flight run: nothing already computed is lost
#     or re-run.
#  2. AdamW must precede Muon. Muon's AdamW sub-group is built with
#     lr_used.get("AdamW"), which is None until AdamW's own LR search has
#     finished -- ordering Muon first would silently give that sub-group Muon's
#     LR instead. Asserted below so a future edit cannot reintroduce it.
OPT_NAMES          = ["EvieKF", "AdamW", "AdaBelief", "Muon", "Shampoo",
                      "Sophia", "Lion", "SGD"]
ABLATION_OPT_NAMES = ["EvieKF", "EvieKFu", "EvieDiag"]   # EvieKF's 10-seed cells
                                                         # here are a superset of
                                                         # the main run's 5
assert OPT_NAMES.index("AdamW") < OPT_NAMES.index("Muon"), \
    "AdamW must be searched before Muon (Muon's AdamW sub-group reuses its LR)"
# Scale-generalization secondary tasks (evie-scalegen-spec.md sec 1): trimmed
# 4-arm set, 3 seeds, run via benchmark12/scalegen/ (TASK["opt_set"]=="scalegen").
# Same two ordering rules as OPT_NAMES above: EvieKF/AdamW first (a cut-short
# run's prefix still contains the comparison), AdamW before Muon (its AdamW
# sub-group reuses AdamW's frozen LR). All 4 names already exist in
# build_optimizer() below -- no new optimizer code.
SCALEGEN_OPT_NAMES = ["EvieKF", "AdamW", "AdaBelief", "Muon"]
assert SCALEGEN_OPT_NAMES.index("AdamW") < SCALEGEN_OPT_NAMES.index("Muon"), \
    "AdamW must be searched before Muon (Muon's AdamW sub-group reuses its LR)"
HESSIAN_CONSUMERS = {"Sophia"}     # optimizers that need the Hutchinson probe
NOISE_CONSUMERS   = {"EvieDiag", "EvieKFu", "EvieKF", "EvieKFn",
                     "EvieDiagAbs", "EvieKFm", "EvieKFr"}       # need the
                                  # split-batch noise hook (engine calls set_noise())

# Optimizers with a SECOND tuned axis, searched JOINTLY with the LR grid (same
# 7-point discipline, same boundary-extend rule -- claude_optim.md sec 4.3.2).
# Anything not listed here is LR-search only, exactly as before.
KNOB_GRID = {
    "EvieKFn": list(EVIEKF_GAMMA_GRID), "EvieKF": list(EVIEKF_GAMMA_GRID),
    "EvieKFu": list(EVIEKF_GAMMA_GRID), "EvieDiag": list(EVIEKF_GAMMA_GRID),
    "Shampoo": list(SHAMPOO_KNOB_GRID),
    # diagnostic arms (DERIVATION.md sec 5.1 / sec 11, FW-9 and FW-11)
    "EvieDiagAbs": list(EVIEKF_GAMMA_GRID),      # same grid as EvieDiag, absolute gate
    "EvieKFr": list(EVIEKF_GAMMA_GRID),          # same as EvieKF, rescale disabled
    "EvieKFm": list(EVIEKF_MEANNORM_GAMMA_GRID), # re-centred: gamma is dimensionless here
}

# Per-optimizer-family LR grid centre. Every optimizer gets the SAME NUMBER of
# grid points (7, half-decade spacing); only the centre differs by family
# because effective LR scale is not comparable across families (CLAUDE.md sec 5).
LR_GRID_CENTRE = {
    "AdamW":     1e-3,
    "AdaBelief": 1e-3,
    "SGD":       3e-2,
    "Sophia":    1e-3,
    "Muon":      2e-2,
    "Lion":      1e-4,
    "Shampoo":   1e-3,          # AdamW-grafted -> Adam-family centring
    "EvieKFn":   1e-3,          # Evie-KF family: P^-1/2 is AdamW's own denominator,
    "EvieKF":    1e-3,          # so the effective LR scale matches the Adam family
    "EvieKFu":   1e-3,
    "EvieDiag":  1e-3,
    "EvieDiagAbs": 1e-3,        # diagnostic arms, same family centring
    "EvieKFm":   1e-3,
    "EvieKFr":   1e-3,
}


def half_decade_grid(centre, n_below=3, n_above=3):
    """7-point half-decade (sqrt(10)~=3.162x) grid centred on `centre`.
    Returns a sorted list of length n_below + 1 + n_above."""
    r = 10.0 ** 0.5
    pts = [centre * (r ** k) for k in range(-n_below, n_above + 1)]
    return [float(f"{p:.4g}") for p in pts]


def knob_grid_extend(grid, at_low):
    """Boundary-extend rule (CLAUDE.md sec 5) applied to a second-axis knob grid.
    Returns up to 2 new points past the edge the search pinned to. gamma-style
    grids (all values >= 1) extend multiplicatively by 3x; a beta-style grid
    bounded in (0, 1) densifies toward the bound instead."""
    g = sorted(float(x) for x in grid)
    if g and all(0.0 < x < 1.0 for x in g):
        if at_low:
            lo, d = g[0], (g[1] - g[0] if len(g) > 1 else g[0] * 0.5)
            return [round(lo - k * d, 6) for k in (1, 2) if lo - k * d > 0.0]
        hi = g[-1]
        return [round(1.0 - (1.0 - hi) / f, 6) for f in (3.0, 10.0)
                if 0.0 < 1.0 - (1.0 - hi) / f < 1.0]
    if at_low:
        return [float(f"{g[0] / f:.4g}") for f in (3.0, 9.0)]
    return [float(f"{g[-1] * f:.4g}") for f in (3.0, 9.0)]


# ---------------------------------------------------------------------------
#  AdaBelief -- Adam with the second moment tracking (g - m)^2 ("belief" in
#  the gradient direction) instead of g^2. Decoupled weight decay, bias
#  correction, no RAdam-style rectification (plain decoupled AdaBelief).
# ---------------------------------------------------------------------------
class AdaBelief(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=ADABELIEF_BETAS, eps=ADABELIEF_EPS,
                 weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"AdaBelief: invalid lr {lr}")
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, b1, b2 = group["lr"], group["beta1"], group["beta2"]
            eps, wd = group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("AdaBelief does not support sparse gradients")
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)                       # decoupled WD
                st = self.state[p]
                if len(st) == 0:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["s"] = torch.zeros_like(p)               # EMA of (g - m)^2
                st["step"] += 1
                t = st["step"]
                m, s = st["m"], st["s"]
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                diff = g - m
                s.mul_(b2).addcmul_(diff, diff, value=1.0 - b2)
                s.add_(eps)
                m_hat = m / (1.0 - b1 ** t)
                s_hat = s / (1.0 - b2 ** t)
                p.addcdiv_(m_hat, s_hat.sqrt().add_(eps), value=-lr)
        return loss


# ---------------------------------------------------------------------------
#  SGDM -- SGD + heavy-ball momentum + Nesterov, with DECOUPLED weight decay
#  (torch.optim.SGD only offers coupled L2, hence this thin re-implementation
#  so the WD convention matches every other optimizer in the harness).
#  Cosine LR schedule is applied by the engine, same as for all optimizers.
# ---------------------------------------------------------------------------
class SGDM(torch.optim.Optimizer):
    def __init__(self, params, lr=3e-2, momentum=SGDM_MOMENTUM,
                 nesterov=SGDM_NESTEROV, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"SGDM: invalid lr {lr}")
        defaults = dict(lr=lr, momentum=momentum, nesterov=nesterov,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, mom = group["lr"], group["momentum"]
            nesterov, wd = group["nesterov"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)                       # decoupled WD
                st = self.state[p]
                if len(st) == 0:
                    st["buf"] = torch.zeros_like(p)
                buf = st["buf"]
                buf.mul_(mom).add_(g)
                d = g.add(buf, alpha=mom) if nesterov else buf
                p.add_(d, alpha=-lr)
        return loss


# ---------------------------------------------------------------------------
#  Sophia (SophiaH) -- clipped Newton-style update with a Hutchinson diagonal
#  Hessian EMA. Hyperparameters frozen per CLAUDE.md sec 4:
#  beta1=0.965, beta2(Hessian EMA)=0.99, rho=0.04, eps=1e-8, HESS_FREQ=5,
#  N_HUTCH=4. The engine calls set_hessian() every HESS_FREQ steps with the
#  Hutchinson estimate from hutchinson_diag_hessian().
# ---------------------------------------------------------------------------
class Sophia(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=SOPHIA_BETAS, rho=SOPHIA_RHO,
                 eps=SOPHIA_EPS, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Sophia: invalid lr {lr}")
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], rho=rho, eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.last_h_norm = float("nan")

    @torch.no_grad()
    def set_hessian(self, param_h_pairs, beta_h=SOPHIA_BETA_H):
        sq = 0.0
        for p, h in param_h_pairs:
            st = self.state[p]
            h_new = h.detach().abs()
            if "h_hat" not in st:
                st["h_hat"] = h_new.clone()
            else:
                st["h_hat"].mul_(beta_h).add_(h_new, alpha=1.0 - beta_h)
            sq += float(h_new.pow(2).sum().item())
        self.last_h_norm = math.sqrt(sq)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, b1 = group["lr"], group["beta1"]
            rho, eps, wd = group["rho"], group["eps"], group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)                       # decoupled WD
                st = self.state[p]
                if "step" not in st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                if "h_hat" not in st:
                    st["h_hat"] = torch.ones_like(p)
                st["step"] += 1
                t = st["step"]
                m = st["m"]
                m.mul_(b1).add_(g, alpha=1.0 - b1)
                m_hat = m / (1.0 - b1 ** t)
                h = st["h_hat"].clamp(min=eps)
                update = (m_hat / (rho * h)).clamp_(-1.0, 1.0)
                p.add_(update, alpha=-lr)
        return loss


def _spatial_avg(h):
    """Per-row average for 2D, scalar average for 1D, spatial average for 4D
    (kept for parity with the historical Hutchinson pipeline)."""
    if h.dim() == 4:
        return h.mean(dim=[2, 3], keepdim=True).expand_as(h).clone()
    if h.dim() == 2:
        return h.mean(dim=1, keepdim=True).expand_as(h).clone()
    if h.dim() == 1:
        return h.mean().expand_as(h).clone()
    return h.clone()


def hutchinson_diag_hessian(model, params, compute_loss, n_hutch=N_HUTCH,
                            optimizer=None, feed_optimizer=False):
    """Hutchinson diagonal-Hessian estimator: for each of n_hutch Rademacher
    vectors v, form v (x) Hv via a double backward and accumulate; divide by
    n_hutch, take abs. Feeds Sophia via set_hessian() when feed_optimizer=True.
    `compute_loss` is a zero-arg closure that returns the (scalar) training loss
    for the current batch -- the engine passes the domain's forward_loss so the
    probe matches the real loss (e.g. the (B,T,V) reshape for language).
    Returns the list of per-parameter diagonal estimates."""
    with torch.enable_grad():
        loss_h = compute_loss()
    grads = torch.autograd.grad(loss_h, params, create_graph=True,
                                retain_graph=True, allow_unused=True)
    h_acc = [torch.zeros_like(p) for p in params]
    for k in range(n_hutch):
        vs = [(torch.randint(0, 2, p.shape, device=p.device, dtype=torch.float32)
               * 2.0 - 1.0).to(p.dtype) for p in params]
        gv = sum((g * v).sum() for g, v in zip(grads, vs) if g is not None)
        retain = (k < n_hutch - 1)
        hvps = torch.autograd.grad(gv, params, retain_graph=retain, allow_unused=True)
        for i, (v, hv) in enumerate(zip(vs, hvps)):
            if hv is not None:
                h_acc[i].add_(v * hv.detach())
        del vs, gv, hvps
    del grads
    for h in h_acc:
        h.div_(n_hutch).abs_()
    if feed_optimizer and optimizer is not None:
        optimizer.set_hessian(list(zip(params, h_acc)), beta_h=SOPHIA_BETA_H)
    return h_acc


# ---------------------------------------------------------------------------
#  Lion (EvoLved Sign Momentum) -- update = sign(b1*m + (1-b1)*g); the EMA is
#  then refreshed with b2. Decoupled weight decay. LR runs ~10x below Adam's,
#  hence the separate grid centre (LR_GRID_CENTRE["Lion"]).
# ---------------------------------------------------------------------------
class Lion(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-4, betas=LION_BETAS, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Lion: invalid lr {lr}")
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1],
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, b1, b2, wd = (group["lr"], group["beta1"], group["beta2"],
                              group["weight_decay"])
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)                       # decoupled WD
                st = self.state[p]
                if len(st) == 0:
                    st["m"] = torch.zeros_like(p)
                m = st["m"]
                update = m.mul(b1).add_(g, alpha=1.0 - b1).sign_()
                p.add_(update, alpha=-lr)
                m.mul_(b2).add_(g, alpha=1.0 - b2)
        return loss


# ---------------------------------------------------------------------------
#  Muon (MomentUm Orthogonalized by Newton-schulz) -- hybrid optimizer.
#  Params with ndim >= 2 that are NOT flagged as embedding/head (see
#  split_params_for_muon) get the Muon update: heavy-ball (Nesterov) momentum
#  followed by Newton-Schulz orthogonalization of the (reshaped-to-2D) update,
#  scaled by MUON_UPDATE_SCALE * sqrt(max(fan_out, fan_in)). Every other param
#  (biases, norm params, all 1D params, token embeddings, LM head) gets a
#  standard decoupled-WD AdamW update. Both sub-groups are carried in ONE
#  Optimizer so checkpoint/resume and the cosine LR scheduler work unchanged.
#
#  Per the frozen decision (CLAUDE.md sec 8.2 + user): the 7-point LR search
#  tunes ONLY the Muon (2D) sub-group's LR; the AdamW sub-group's LR is fixed
#  to this file's separately-frozen AdamW LR (passed as adamw_lr to
#  build_optimizer). Conv weights (4D) are reshaped (O, I*kH*kW) and DO go to
#  the Muon side -- the strict "2D only" reading would leave Muon == AdamW on
#  the entire vision domain, which is not a meaningful baseline. This split is
#  written once here and pasted identically into all 12 files.
# ---------------------------------------------------------------------------
def _zeropower_via_newtonschulz5(G, steps=MUON_NS_STEPS, coeffs=MUON_NS_COEFFS,
                                 eps=1e-7):
    """Quintic Newton-Schulz iteration -> approximate orthogonalization of G
    (2D). Operates in float32. Returns a matrix with the same shape as G whose
    singular values are pushed toward 1."""
    assert G.ndim == 2, "Newton-Schulz expects a 2D matrix"
    a, b, c = coeffs
    X = G.float()
    X = X / (X.norm() + eps)
    transpose = X.size(0) > X.size(1)
    if transpose:
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * (A @ A)
        X = a * X + B @ X
    if transpose:
        X = X.T
    return X.to(G.dtype)


class Muon(torch.optim.Optimizer):
    def __init__(self, param_groups):
        # Each incoming group must set: params, lr, use_muon (bool).
        defaults = dict(
            lr=2e-2, use_muon=True,
            momentum=MUON_MOMENTUM, nesterov=MUON_NESTEROV, ns_steps=MUON_NS_STEPS,
            betas=MUON_ADAMW_BETAS, eps=MUON_ADAMW_EPS, weight_decay=WEIGHT_DECAY,
        )
        super().__init__(param_groups, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr, wd = group["lr"], group["weight_decay"]
            if group["use_muon"]:
                mom, nesterov, ns_steps = (group["momentum"], group["nesterov"],
                                           group["ns_steps"])
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if g.ndim > 2:
                        g = g.reshape(g.size(0), -1)
                    st = self.state[p]
                    if len(st) == 0:
                        st["buf"] = torch.zeros_like(g)
                    buf = st["buf"]
                    buf.mul_(mom).add_(g)
                    d = g.add(buf, alpha=mom) if nesterov else buf
                    o = _zeropower_via_newtonschulz5(d, steps=ns_steps)
                    fan_out, fan_in = o.size(0), o.size(1)
                    scale = MUON_UPDATE_SCALE * math.sqrt(max(fan_out, fan_in))
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)                   # decoupled WD
                    p.add_(o.reshape(p.shape), alpha=-lr * scale)
            else:
                b1, b2 = group["betas"]
                eps = group["eps"]
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    g = p.grad
                    if wd != 0.0:
                        p.mul_(1.0 - lr * wd)                   # decoupled WD
                    st = self.state[p]
                    if len(st) == 0:
                        st["step"] = 0
                        st["m"] = torch.zeros_like(p)
                        st["v"] = torch.zeros_like(p)
                    st["step"] += 1
                    t = st["step"]
                    m, v = st["m"], st["v"]
                    m.mul_(b1).add_(g, alpha=1.0 - b1)
                    v.mul_(b2).addcmul_(g, g, value=1.0 - b2)
                    m_hat = m / (1.0 - b1 ** t)
                    v_hat = v / (1.0 - b2 ** t)
                    p.addcdiv_(m_hat, v_hat.sqrt().add_(eps), value=-lr)
        return loss


def split_params_for_muon(model):
    """Returns (muon_params, adamw_params). A parameter goes to the Muon side
    iff it has ndim >= 2 AND its qualified name does not match any entry in
    model._muon_exclude_names (set by the data fragment for token embeddings /
    positional embeddings / LM head). Everything else -> AdamW side."""
    exclude = set(getattr(model, "_muon_exclude_names", []))

    def excluded(name):
        return any(name == e or name.startswith(e + ".") or name.endswith("." + e)
                   or ("." + e + ".") in ("." + name + ".") for e in exclude)

    muon_params, adamw_params = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim >= 2 and not excluded(name):
            muon_params.append(p)
        else:
            adamw_params.append(p)
    return muon_params, adamw_params


# ---------------------------------------------------------------------------
#  Shampoo -- per-layer L^{-1/4} G R^{-1/4} preconditioner with AdamW norm
#  grafting. L = EMA(G G^T), R = EMA(G^T G). Grafting the AdamW update norm
#  keeps it stable at this scale (ungrafted Shampoo is not). 1-D params
#  (biases, norms) and 2-D params with a dimension > SHAMPOO_MAXF fall through
#  to the AdamW graft alone -- the latter guard is NOT in the Kaggle reference
#  and is added here on purpose: without it a tied 16k-vocab lm_head would
#  allocate a 16000x16000 preconditioner and eigh it. Decoupled weight decay,
#  same as every optimizer in this block. `beta` is the second tuned axis
#  (KNOB_GRID["Shampoo"]); `lr` is the LR-searched value.
# ---------------------------------------------------------------------------
class Shampoo(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, beta=SHAMPOO_BETA, eps=SHAMPOO_EPS,
                 update_freq=SHAMPOO_UPDATE_FREQ, graft_betas=SHAMPOO_GRAFT_BETAS,
                 maxf=SHAMPOO_MAXF, weight_decay=0.0):
        if lr <= 0.0:
            raise ValueError(f"Shampoo: invalid lr {lr}")
        if not 0.0 < beta < 1.0:
            raise ValueError(f"Shampoo: invalid beta {beta}")
        defaults = dict(lr=lr, beta=beta, eps=eps, update_freq=int(update_freq),
                        graft_betas=graft_betas, maxf=int(maxf),
                        weight_decay=weight_decay)
        super().__init__(params, defaults)

    @staticmethod
    def _inv_root(M, eps, power=-0.25):
        L, Q = torch.linalg.eigh(M.double())
        L = L.clamp_min(eps)
        return (Q @ torch.diag(L.pow(power)) @ Q.T).to(M.dtype)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            b1, b2 = group["graft_betas"]
            lr, wd, maxf = group["lr"], group["weight_decay"], group["maxf"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                if g.is_sparse:
                    raise RuntimeError("Shampoo does not support sparse gradients")
                st = self.state[p]
                if "t" not in st:
                    st["t"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                    if g.ndim == 2 and max(g.shape) <= maxf:
                        st["L"] = torch.zeros(g.size(0), g.size(0), device=g.device)
                        st["R"] = torch.zeros(g.size(1), g.size(1), device=g.device)
                st["t"] += 1
                t = st["t"]
                if wd != 0.0:
                    p.mul_(1.0 - lr * wd)                       # decoupled WD
                st["m"].mul_(b1).add_(g, alpha=1.0 - b1)
                st["v"].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                adam_dir = (st["m"] / (1.0 - b1 ** t)) / (
                    (st["v"] / (1.0 - b2 ** t)).sqrt().add(1e-8))
                if "L" not in st:                              # 1-D or oversized 2-D
                    p.add_(adam_dir, alpha=-lr)
                    continue
                st["L"].mul_(group["beta"]).add_(g @ g.T, alpha=1.0 - group["beta"])
                st["R"].mul_(group["beta"]).add_(g.T @ g, alpha=1.0 - group["beta"])
                if t % group["update_freq"] == 0 or "Li" not in st:
                    st["Li"] = self._inv_root(st["L"], group["eps"])
                    st["Ri"] = self._inv_root(st["R"], group["eps"])
                d = st["Li"] @ g @ st["Ri"]
                d = d * (adam_dir.norm() / (d.norm() + 1e-12))  # graft the norm
                p.add_(d, alpha=-lr)
        return loss


# ---------------------------------------------------------------------------
#  Evie-KF -- Kronecker-factored risk-sensitive preconditioner.
#
#      dtheta = -lr * P^-1/2 (I + gamma Sigma_z)^-1/2 P^-1/2 m1hat
#      P      = diag(sqrt(m2hat))                 (AdamW's own curvature proxy)
#      Sigma_z = P^-1/2 Sigma P^-1/2              (noise covariance, whitened)
#      Sigma  = Cov(g) from the split batch: delta = (gA - gB)/2, g = (gA + gB)/2
#      Sigma_z ~= (B kron A)/tr(A) per weight matrix; gain_ij = (1 + gamma mu_i lam_j/tr)^-1/2
#      Z = U_B [ (U_B^T Y U_A) * gain ] U_A^T     (exact; self-checked against the dense form)
#
#  This is a per-parameter (self.state[p]) port of the validated flat-vector
#  reference in "Claude outputs/kaggle_evie_v3.py". The port is required so the
#  optimizer state survives state_dict()/resume like every other optimizer in
#  this harness. Two structural points the port gets right (claude_optim.md 4.1):
#    1. step() cannot compute the split-batch noise itself -- the engine runs the
#       two half-batch backward passes and hands the result in via set_noise(),
#       exactly mirroring Sophia's set_hessian() pattern. p.grad is (gA+gB)/2.
#    2. Algorithm step 8 (z <- z * ||y||/||z||) is a norm across ALL parameters,
#       not per-layer: two passes over the param groups per step() -- accumulate
#       ||y||^2, ||z||^2 globally, then rescale. This keeps Evie-KF's step SIZE
#       identical to AdamW's and isolates the direction change (spec sec 1).
#
#  Flags: gamma=0 -> AdamW exactly; diag_only -> Evie-diag; centered=False ->
#  Evie-KFu; per_layer_norm -> Evie-KFn. Decoupled weight decay, same convention
#  as the rest of the block.
# ---------------------------------------------------------------------------
class EvieKF(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=EVIEKF_BETAS, eps=EVIEKF_EPS,
                 weight_decay=0.0, gamma=100.0, beta_sig=EVIEKF_BETA_SIG,
                 warmup=EVIEKF_WARMUP, refresh=EVIEKF_REFRESH, maxf=EVIEKF_MAXF,
                 diag_only=False, centered=True, per_layer_norm=False,
                 gain_clip=EVIEKF_GAIN_CLIP,
                 diag_relative=True, kron_mean_norm=False, norm_preserve=True):
        if lr <= 0.0 or gamma < 0.0 or not 0.0 < beta_sig < 1.0:
            raise ValueError("EvieKF: invalid hyperparameters")
        defaults = dict(lr=lr, beta1=betas[0], beta2=betas[1], eps=eps,
                        weight_decay=weight_decay)
        super().__init__(params, defaults)
        self.gamma, self.beta_sig = float(gamma), float(beta_sig)
        self.warmup, self.refresh, self.maxf = int(warmup), int(refresh), int(maxf)
        self.diag_only, self.centered = bool(diag_only), bool(centered)
        self.per_layer_norm, self.gain_clip = bool(per_layer_norm), float(gain_clip)
        # --- three diagnostic switches. Every default reproduces the shipped
        # behaviour EXACTLY, so no protocol run is affected by their existence.
        #   diag_relative   False -> the diagonal gate is (1 + gamma*s)^-1/2, the
        #                   derivation's own diagonal special case, instead of the
        #                   mean-normalised (1 + gamma*s/mean(s))^-1/2. Needed to
        #                   deconfound C2 (DERIVATION.md sec 5.1, FW-11).
        #   kron_mean_norm  True -> normalise the Kronecker spectrum by its MEAN
        #                   eigenvalue instead of tr(A), which is the only form in
        #                   which gamma is actually dimensionless (FW-9).
        #   norm_preserve   False -> skip the global ||y||/||z|| rescale, so the
        #                   operator shrinks instead of rotating. Tests whether
        #                   norm preservation is load-bearing (FW-10).
        self.diag_relative = bool(diag_relative)
        self.kron_mean_norm = bool(kron_mean_norm)
        self.norm_preserve = bool(norm_preserve)
        self._roles = {}                # {param: role_str}; set by set_param_roles()
        # descriptive diagnostics, refreshed every step(), read by the engine
        # (claude_optim.md sec 4.3.4/4.3.5). NEVER significance-tested.
        self.last_cos = 1.0
        self.b_simple = 0.0
        self.tr_sigma = 0.0
        self.eff_rank = 0.0
        self.gain_lo, self.gain_hi = 1.0, 1.0
        self.n_active_steps = 0        # wiring canary: 0 at end of run => set_noise()
                                      # was never called and this WAS silently AdamW
        self.role_gain = {}           # role -> [gain_lo, gain_hi, eff_rank_sum, n]

    def set_param_roles(self, param_roles):
        """{param: role} with role in {embedding, attention, ffn, head, other};
        enables the per-module-category gain/effective-rank breakdown (figures
        F4/F5). Optional -- without it role_gain stays empty."""
        self._roles = dict(param_roles)

    @staticmethod
    def _eigh(M):
        Ms = (M + M.transpose(-1, -2)) * 0.5
        L, W = torch.linalg.eigh(Ms.double())          # fp64 eigh regardless of
        return L.clamp_min(0).to(M.dtype), W.to(M.dtype)   # param dtype, then cast back

    @torch.no_grad()
    def set_noise(self, param_delta_pairs):
        """Called by the engine once per step BEFORE step(), with
        param_delta_pairs = [(p, (gradA - gradB)/2), ...] and p.grad already set
        to (gradA + gradB)/2. Mirrors Sophia.set_hessian()."""
        for p, delta in param_delta_pairs:
            self.state[p]["_delta"] = delta.detach()

    def _global_step(self):
        """The global step index, derived from the per-parameter counters so it
        survives load_state_dict() (self.__dict__ is not checkpointed). Equals a
        plain monotone counter as long as every param is stepped together."""
        s = 0
        for group in self.param_groups:
            for p in group["params"]:
                st = self.state.get(p)
                if st and "step" in st:
                    s = max(s, int(st["step"]))
        return s + 1

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        t = self._global_step()
        active_gate = (t > self.warmup) and (self.gamma > 0.0)

        cache = []                       # (p, group, y, z, pm12) for pass 2
        y_sq_total = z_sq_total = dot_total = 0.0
        tr_sigma_total = y_sq_for_bsimple = 0.0
        gains_lo, gains_hi, eff_ranks = [], [], []
        any_noise = False
        self.role_gain = {}

        # ---- pass 1: per-parameter y (whitened Adam dir) and z (preconditioned) --
        for group in self.param_groups:
            b1, b2, eps = group["beta1"], group["beta2"], group["eps"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                st = self.state[p]
                if "step" not in st:
                    st["step"] = 0
                    st["m"] = torch.zeros_like(p)
                    st["v"] = torch.zeros_like(p)
                st["step"] += 1
                pt = st["step"]
                st["m"].mul_(b1).add_(g, alpha=1.0 - b1)
                st["v"].mul_(b2).addcmul_(g, g, value=1.0 - b2)
                mh = st["m"] / (1.0 - b1 ** pt)
                vh = st["v"] / (1.0 - b2 ** pt)
                pm12 = vh.sqrt().add(eps).rsqrt()          # == torch AdamW's P^-1/2
                y = pm12 * mh
                z = y.clone()

                delta = st.pop("_delta", None)             # consume; never stale-reuse
                if delta is not None:
                    any_noise = True
                    src = delta if self.centered else g
                    dz = pm12 * src.detach().float()
                    tr_sigma_total += float((dz * dz).sum())
                    y_sq_for_bsimple += float((y * y).sum())

                    if p.dim() < 2 or self.diag_only:
                        if "s" not in st:
                            st["s"] = torch.zeros_like(dz)
                        st["s"].mul_(self.beta_sig).addcmul_(
                            dz, dz, value=1.0 - self.beta_sig)
                        if active_gate:
                            s = st["s"] / (1.0 - self.beta_sig ** pt)
                            gsc = (1.0 + self.gamma
                                   * (s / (s.mean() + 1e-30)
                                      if self.diag_relative else s)).rsqrt()
                            if self.per_layer_norm:
                                gsc = (gsc / gsc.pow(2).mean().sqrt()
                                       .clamp_min(1e-30)).clamp(max=self.gain_clip)
                            z = y * gsc
                    else:
                        d0 = p.size(0)
                        rest = p.numel() // d0
                        D = dz.reshape(d0, rest)
                        if "A" not in st:
                            st["A"] = (torch.zeros(rest, rest, device=p.device)
                                       if rest <= self.maxf
                                       else torch.zeros(rest, device=p.device))
                            st["B"] = (torch.zeros(d0, d0, device=p.device)
                                       if d0 <= self.maxf
                                       else torch.zeros(d0, device=p.device))
                        st["A"].mul_(self.beta_sig).add_(
                            (D.T @ D) if rest <= self.maxf else (D * D).sum(0),
                            alpha=1.0 - self.beta_sig)
                        st["B"].mul_(self.beta_sig).add_(
                            (D @ D.T) if d0 <= self.maxf else (D * D).sum(1),
                            alpha=1.0 - self.beta_sig)
                        if active_gate:
                            if (pt % self.refresh == 0) or ("lam" not in st):
                                A = st["A"] / (1.0 - self.beta_sig ** pt)
                                B = st["B"] / (1.0 - self.beta_sig ** pt)
                                st["lam"], st["UA"] = (
                                    self._eigh(A) if A.dim() == 2
                                    else (A.clamp_min(0), None))
                                st["mu"], st["UB"] = (
                                    self._eigh(B) if B.dim() == 2
                                    else (B.clamp_min(0), None))
                                _tr_a = float(st["lam"].sum())
                                # /tau matches tr(B kron A) to tr(Sigma_z); it is
                                # NOT scale-free. /mean(mu*lam) is (FW-9).
                                st["tr"] = ((_tr_a * float(st["mu"].sum())
                                             / max(1, p.numel()))
                                            if self.kron_mean_norm
                                            else _tr_a) + 1e-30
                            lam, mu = st["lam"], st["mu"]
                            UA, UB, tr = st["UA"], st["UB"], st["tr"]
                            C = y.reshape(d0, rest)
                            if UB is not None:
                                C = UB.T @ C
                            if UA is not None:
                                C = C @ UA
                            gain = (1.0 + self.gamma
                                    * (mu[:, None] * lam[None, :]) / tr).rsqrt()
                            if self.per_layer_norm:
                                gain = (gain / gain.pow(2).mean().sqrt()
                                        .clamp_min(1e-30)).clamp(max=self.gain_clip)
                            g_lo, g_hi = float(gain.min()), float(gain.max())
                            ev = (mu[:, None] * lam[None, :]).reshape(-1)
                            er = float(ev.sum() ** 2 / (ev.pow(2).sum() + 1e-30))
                            gains_lo.append(g_lo)
                            gains_hi.append(g_hi)
                            eff_ranks.append(er)
                            role = self._roles.get(p, "other")
                            acc_r = self.role_gain.setdefault(
                                role, [g_lo, g_hi, 0.0, 0])
                            acc_r[0] = min(acc_r[0], g_lo)
                            acc_r[1] = max(acc_r[1], g_hi)
                            acc_r[2] += er
                            acc_r[3] += 1
                            C = C * gain
                            if UB is not None:
                                C = UB @ C
                            if UA is not None:
                                C = C @ UA.T
                            z = C.reshape_as(p)

                y_sq_total += float((y * y).sum())
                z_sq_total += float((z * z).sum())
                dot_total += float((y * z).sum())
                cache.append((p, group, y, z, pm12))

        # ---- pass 2: global norm-preserving rescale, then apply the update ------
        ny = y_sq_total ** 0.5
        nz = z_sq_total ** 0.5
        # `op_active` is "the operator ran this step" and drives the telemetry and
        # the wiring canary; `do_rescale` is the separate question of whether the
        # global norm rescale is applied. They were one variable until EvieKFr
        # needed to disable the rescale WITHOUT making n_active_steps read 0,
        # which the engine treats as a silent-AdamW bug and flags as a divergence.
        op_active = active_gate and any_noise and nz > 1e-30
        do_rescale = op_active and self.norm_preserve
        scale = (ny / nz) if do_rescale else 1.0
        den = ny * nz
        self.last_cos = (dot_total / den) if (op_active and den > 1e-30) else 1.0
        self.tr_sigma = tr_sigma_total
        self.b_simple = (tr_sigma_total / (y_sq_for_bsimple + 1e-30)
                         if any_noise else 0.0)
        if gains_lo:
            self.gain_lo, self.gain_hi = min(gains_lo), max(gains_hi)
            self.eff_rank = sum(eff_ranks) / len(eff_ranks)
        if op_active:
            self.n_active_steps += 1

        for p, group, y, z, pm12 in cache:
            lr, wd = group["lr"], group["weight_decay"]
            if wd != 0.0:
                p.mul_(1.0 - lr * wd)                          # decoupled WD
            u = (pm12 * (z * scale)) if scale != 1.0 else (pm12 * z)
            p.add_(u.to(p.dtype), alpha=-lr)
        return loss


def eviekf_self_check():
    """Dense-vs-factored (B kron A)^-1/2 parity + the gamma=0 identity
    (claude_optim.md sec 4.3.3 / sec 5). Ported from kaggle_evie_v3.py's
    self_check(). Raises RuntimeError on failure -- the engine turns that into a
    one-sentence abort, exactly like the CUDA preflight. A vec-ordering / eigh
    bug here produces a plausible-looking but meaningless results table."""
    gen = torch.Generator().manual_seed(0)
    n, m, gam = 5, 4, 3.7
    D = torch.randn(40, n, m, generator=gen, dtype=torch.float64)
    A = torch.einsum("bij,bik->jk", D, D) / 40.0
    B = torch.einsum("bij,bkj->ik", D, D) / 40.0
    tr = float(A.trace())
    Y = torch.randn(n, m, generator=gen, dtype=torch.float64)
    lam, UA = torch.linalg.eigh(A)
    mu, UB = torch.linalg.eigh(B)
    gain = (1.0 + gam * (mu[:, None] * lam[None, :]) / tr).rsqrt()
    fast = UB @ ((UB.T @ Y @ UA) * gain) @ UA.T
    M = torch.eye(n * m, dtype=torch.float64) + gam * torch.kron(B, A) / tr
    w, V = torch.linalg.eigh(M)
    dense = (V @ torch.diag(w.rsqrt()) @ V.T) @ Y.reshape(-1)
    e1 = float((fast.reshape(-1) - dense).abs().max())
    g1 = (1.0 + 0.0 * (mu[:, None] * lam[None, :]) / tr).rsqrt()
    e2 = float((UB @ ((UB.T @ Y @ UA) * g1) @ UA.T - Y).abs().max())
    ok3 = bool((gain <= 1.0 + 1e-12).all())
    if not (e1 < 1e-9 and e2 < 1e-9 and ok3):
        raise RuntimeError(
            f"EvieKF Kronecker self-check FAILED (factored-vs-dense max err "
            f"{e1:.2e}, gamma=0 identity err {e2:.2e}, gains<=1 {ok3}); refusing "
            f"to run -- every Evie-KF number would be meaningless.")
    return {"factored_vs_dense": e1, "gamma0_identity": e2, "gains_le_1": ok3}


def build_optimizer(opt_name, model, lr, adamw_lr=None, knob=None,
                    eviekf_maxf=None):
    """Single optimizer-selection entry point (CLAUDE.md sec 1: keep this
    structured so a new optimizer pastes in as one extra branch). `lr` is the
    frozen, LR-searched value for `opt_name`. `adamw_lr` is only used by Muon
    (its AdamW sub-group). `knob` is the frozen value of the SECOND tuned axis
    for optimizers in KNOB_GRID -- gamma for the Evie-KF family, beta for
    Shampoo -- and is ignored by every other optimizer. Weight decay is
    WEIGHT_DECAY (decoupled) for every optimizer.

    `eviekf_maxf` overrides the frozen EVIEKF_MAXF Kronecker size limit and is
    passed by the engine ONLY when the task declares TASK["eviekf_maxf"] -- a
    diagnostic file, never a protocol task (see the note on EVIEKF_MAXF). It
    arrives as an argument rather than being read from TASK here so that this
    block stays free of any dependency on the task header."""
    params = [p for p in model.parameters() if p.requires_grad]
    if opt_name == "AdamW":
        return torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.999), eps=1e-8,
                                 weight_decay=WEIGHT_DECAY)
    if opt_name == "AdaBelief":
        return AdaBelief(params, lr=lr, betas=ADABELIEF_BETAS, eps=ADABELIEF_EPS,
                         weight_decay=WEIGHT_DECAY)
    if opt_name == "SGD":
        return SGDM(params, lr=lr, momentum=SGDM_MOMENTUM, nesterov=SGDM_NESTEROV,
                    weight_decay=WEIGHT_DECAY)
    if opt_name == "Sophia":
        return Sophia(params, lr=lr, betas=SOPHIA_BETAS, rho=SOPHIA_RHO,
                      eps=SOPHIA_EPS, weight_decay=WEIGHT_DECAY)
    if opt_name == "Lion":
        return Lion(params, lr=lr, betas=LION_BETAS, weight_decay=WEIGHT_DECAY)
    if opt_name == "Shampoo":
        return Shampoo(params, lr=lr,
                       beta=(float(knob) if knob is not None else SHAMPOO_BETA),
                       eps=SHAMPOO_EPS, update_freq=SHAMPOO_UPDATE_FREQ,
                       graft_betas=SHAMPOO_GRAFT_BETAS, maxf=SHAMPOO_MAXF,
                       weight_decay=WEIGHT_DECAY)
    if opt_name in ("EvieKFn", "EvieKF", "EvieKFu", "EvieDiag",
                    "EvieDiagAbs", "EvieKFm", "EvieKFr"):
        return EvieKF(params, lr=lr, betas=EVIEKF_BETAS, eps=EVIEKF_EPS,
                      weight_decay=WEIGHT_DECAY,
                      gamma=(float(knob) if knob is not None else 100.0),
                      beta_sig=EVIEKF_BETA_SIG, warmup=EVIEKF_WARMUP,
                      refresh=EVIEKF_REFRESH,
                      maxf=(EVIEKF_MAXF if eviekf_maxf is None
                            else int(eviekf_maxf)),
                      diag_only=(opt_name in ("EvieDiag", "EvieDiagAbs")),
                      centered=(opt_name != "EvieKFu"),
                      per_layer_norm=(opt_name == "EvieKFn"),
                      gain_clip=EVIEKF_GAIN_CLIP,
                      diag_relative=(opt_name != "EvieDiagAbs"),
                      kron_mean_norm=(opt_name == "EvieKFm"),
                      norm_preserve=(opt_name != "EvieKFr"))
    if opt_name == "Muon":
        muon_params, adamw_params = split_params_for_muon(model)
        a_lr = adamw_lr if adamw_lr is not None else lr
        groups = []
        if muon_params:
            groups.append(dict(params=muon_params, lr=lr, use_muon=True))
        if adamw_params:
            groups.append(dict(params=adamw_params, lr=a_lr, use_muon=False))
        if not groups:
            raise RuntimeError("Muon: model has no trainable parameters")
        return Muon(groups)
    raise ValueError(f"unknown optimizer {opt_name!r}")


def optimizer_lr_scale(optimizer):
    """Base LRs per param-group, so the engine's cosine scheduler can be rebuilt
    on resume with the right per-group base_lrs (Muon has two)."""
    return [g["lr"] for g in optimizer.param_groups]
# === END SHARED OPTIMIZER BLOCK ===
