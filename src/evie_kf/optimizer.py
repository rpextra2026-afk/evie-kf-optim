"""Evie-KF -- a Kronecker-factored, risk-sensitive preconditioner for AdamW.

The update is

    dtheta = -lr * P^-1/2 (I + gamma * Sigma_z)^-1/2 P^-1/2 m_hat,

with P AdamW's own denominator and Sigma_z ~ (B kron A) / tr(A) the centred
gradient-noise covariance in Adam-whitened coordinates. At gamma=0 the gate is
the identity and the step is AdamW's, exactly.

The class below is sliced verbatim out of the shared optimizer block that
produced every number in the paper (`benchmark12/_build/frag_shared_optim.py`),
so this package and the benchmark harness cannot drift apart.

Evie-KF needs a sample of the gradient noise, which it gets from two disjoint
half batches. `step()` alone does not produce one: call `set_noise()` before
each `step()`, or use `evie_kf.split_step()`, which does the whole thing. If
`set_noise()` is never called, `n_active_steps` stays 0 and the optimizer is
silently AdamW -- check that counter at the end of a run.
"""

import math

import torch
import torch.nn as nn
import numpy as np

__version__ = "0.1.0"

__all__ = ["EvieKF", "eviekf_self_check", "split_step", "__version__"]

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


# ---------------------------------------------------------------------------
# The split-batch step.
#
# Two disjoint half batches, one forward/backward each: the same examples and
# the same total FLOPs as one full-batch backward, with no extra data. The mean
# of the two half gradients is the ordinary minibatch gradient and their
# difference is a noise sample uncorrelated with it (Proposition 3 in the
# paper). The noise sample is clipped by the same coefficient as the gradient,
# so the signal-to-noise ratio the gate sees is undistorted.
#
# This is the loop `benchmark12/_build/frag_engine.py` runs; it is reproduced
# here so the package is usable on its own.
# ---------------------------------------------------------------------------

@torch.no_grad()
def _noop():
    pass


def split_step(optimizer, model, loss_fn, inputs, targets, grad_clip=5.0):
    """One Evie-KF training step, including the noise sample.

    `loss_fn(model, inputs, targets)` returns a scalar loss. Returns the mean
    loss over the two halves as a float. The caller does not zero gradients;
    this does.

    On a batch too small to split the optimizer still steps, with a zero noise
    sample, which is AdamW for that step.
    """
    params = [p for g in optimizer.param_groups for p in g["params"]
              if p.requires_grad]
    optimizer.zero_grad(set_to_none=True)

    n_half = inputs.size(0) // 2
    if n_half < 1:
        loss = loss_fn(model, inputs, targets)
        loss.backward()
        loss_v = float(loss.detach().item())
        torch.nn.utils.clip_grad_norm_(params, grad_clip)
        optimizer.set_noise([(p, torch.zeros_like(p)) for p in params])
        optimizer.step()
        return loss_v

    xa, xb = inputs[:n_half], inputs[n_half:2 * n_half]
    ya, yb = targets[:n_half], targets[n_half:2 * n_half]

    la = loss_fn(model, xa, ya)
    la.backward()
    ga = [(p.grad.detach().clone() if p.grad is not None else None)
          for p in params]
    optimizer.zero_grad(set_to_none=True)

    lb = loss_fn(model, xb, yb)
    lb.backward()
    gb = [(p.grad.detach().clone() if p.grad is not None else None)
          for p in params]

    for p, a, b in zip(params, ga, gb):
        if a is not None and b is not None:
            p.grad = 0.5 * (a + b)
        elif a is not None:
            p.grad = a.clone()
        elif b is not None:
            p.grad = b.clone()

    loss_v = 0.5 * (float(la.detach().item()) + float(lb.detach().item()))

    pre_clip = float(torch.nn.utils.clip_grad_norm_(params, grad_clip))
    coef = (grad_clip / (pre_clip + 1e-6)
            if (math.isfinite(pre_clip) and pre_clip > grad_clip) else 1.0)

    pairs = []
    for p, a, b in zip(params, ga, gb):
        if a is not None and b is not None:
            pairs.append((p, (0.5 * (a - b)) * coef))
        else:
            pairs.append((p, torch.zeros_like(p)))
    optimizer.set_noise(pairs)
    optimizer.step()
    return loss_v
