# ==============================================================================
#  ENGINE  --  preflight, checkpoint/resume, divergence & error handling,
#  LR search (7-point half-decade + boundary rule), training loop, stats.
#  GENERATED: identical across all 12 files. Edit _build/frag_engine.py.
#
#  Domain hooks this engine calls (provided by the data fragment above):
#    prepare_data()                                  -> data handle
#    lr_search_units(data)                           -> [unit, ...]  (LR search)
#    main_units(data)                                -> [unit, ...]  (main sweep)
#    build_model(data, unit)                         -> nn.Module on DEVICE
#                                                      (._muon_exclude_names set)
#    make_loaders(data, unit, batch_size, seed)      -> (train, val, test)
#    make_criterion()                                -> loss fn
#    forward_loss(model, batch, criterion)           -> scalar loss tensor
#    evaluate(model, loader, criterion)              -> (primary_value, aux_dict)
#    AUX_METRIC_KEYS                                 -> [str, ...] (descriptive)
# ==============================================================================

# Which optimizer set this file runs: the main 8 (default), the Evie-KF
# ablation ladder for the generated benchmark12/ablation/ files
# (TASK["opt_set"] == "ablation", claude_optim.md sec 2/3), or the trimmed
# scale-generalization arm set for benchmark12/scalegen/ files
# (TASK["opt_set"] == "scalegen", evie-scalegen-spec.md sec 1). Every loop
# below iterates ACTIVE_OPT_NAMES, never OPT_NAMES directly.
# A diagnostic file (TASK["opt_set"] == "diag") runs an explicit arm list from
# TASK["opt_list"] with its (lr, knob) PINNED by TASK["pinned_lr"] -- no search.
# TASK["fixed_knob"] is the weaker form: the second axis is FIXED at a stated
# value while the LR is still searched over the ordinary 7-point grid, i.e. the
# arm gets exactly the baselines' tuning budget. TASK["fixed_lr"] is its mirror:
# pin the LR and search the second axis only.
# It answers a single "was this a tuning artefact?" question about one already
# finished task and is never part of any significance table.
ACTIVE_OPT_NAMES = (list(TASK["opt_list"]) if TASK.get("opt_list")
                    else ABLATION_OPT_NAMES if TASK.get("opt_set") == "ablation"
                    else SCALEGEN_OPT_NAMES if TASK.get("opt_set") == "scalegen"
                    else OPT_NAMES)
# BENCH12_ARMS splits ONE task across several machines: each run does a subset of
# the arms, writing into the same checkpoints/<TASK_NAME>/ namespace, and the
# per-arm results are merged afterwards. Cells are keyed "opt|unit|seed", never by
# index, so a union of the parts is exactly the whole -- this is the same property
# that makes reordering OPT_NAMES safe mid-run. Used for language_openwebtext,
# where one machine cannot finish in the time available.
_ARMS_ENV = os.environ.get("BENCH12_ARMS", "").strip()
if _ARMS_ENV:
    _want = [a.strip() for a in _ARMS_ENV.split(",") if a.strip()]
    _unknown = [a for a in _want if a not in ACTIVE_OPT_NAMES]
    if _unknown:
        raise SystemExit(f"BENCH12_ARMS names {_unknown}, which this task does not "
                         f"run (its arms are {ACTIVE_OPT_NAMES})")
    ACTIVE_OPT_NAMES = [a for a in ACTIVE_OPT_NAMES if a in _want]

# ---- graceful stop: flush the in-progress cell before exiting (CLAUDE.md 6) --
_STOP = {"flag": False, "sig": None}


def _on_signal(signum, frame):
    # Signal handlers must stay minimal: log() does buffered/locked stdio via
    # Python's TextIOWrapper, which is NOT safe to re-enter from a handler --
    # if the main thread is itself mid-print() (or a second signal lands before
    # the first handler returns, e.g. someone mashing Ctrl-C), that throws
    # "RuntimeError: reentrant call". The training loop already logs and
    # checkpoints cleanly once it observes the flag (see the top of the step
    # loop below); os.write() is the one signal-safe way to also ack immediately.
    _STOP["flag"] = True
    _STOP["sig"] = signum
    try:
        os.write(2, f"\n[signal] caught signal {signum}; will checkpoint and "
                    f"exit at the next safe point\n".encode())
    except Exception:
        pass


for _s in (signal.SIGTERM, signal.SIGINT):
    try:
        signal.signal(_s, _on_signal)
    except Exception:
        pass


_ENV_ERROR_MARKERS = (
    "cuda error", "cublas", "cudnn", "device-side assert", "no kernel image",
    "cuda out of memory", "illegal memory access", "misaligned address",
    "cuda runtime", "nccl", "cufft", "curand", "cusolver", "not compiled with cuda",
    "device kernel image", "invalid device function",
)


def classify_exception(e):
    """'env'  -> environment/hardware/build problem: re-raise, abort the file.
    'numerical' -> this one cell blew up (NaN/inf/shape from a bad LR): record
    it as a failed cell and keep going.  (CLAUDE.md sec 6: never swallow the
    first kind as if it were the second.)"""
    txt = f"{type(e).__name__}: {e}".lower()
    if isinstance(e, (torch.cuda.OutOfMemoryError,)):
        return "env"
    if any(m in txt for m in _ENV_ERROR_MARKERS):
        return "env"
    if isinstance(e, (RuntimeError, FloatingPointError, ValueError, ArithmeticError)):
        return "numerical"
    return "env"


def _preflight_eviekf_self_check():
    # Evie-KF Kronecker self-check (claude_optim.md sec 4.3.3): dense-vs-factored
    # (B kron A)^-1/2 parity + the gamma=0 identity. Cheap, CPU-only math -- runs
    # for every file regardless of CUDA/CPU/SMOKE, because a vec-ordering/eigh
    # bug here would silently produce a meaningless table, so abort in one
    # sentence now, same as the CUDA check below. (Previously this only ran on
    # the CUDA branch, so ALLOW_CPU=1/SMOKE=1 runs -- exactly the runs meant to
    # be a cheap correctness gate before spending GPU time -- never exercised it.)
    try:
        _sc = eviekf_self_check()
        log(f"[preflight] EvieKF self-check OK  (factored-vs-dense "
            f"{_sc['factored_vs_dense']:.1e}, gamma=0 identity "
            f"{_sc['gamma0_identity']:.1e})")
    except Exception as e:
        fail(f"EvieKF self-check failed at startup: {e}")


def preflight():
    """Verify CUDA is actually usable BEFORE any real training, so a broken
    torch/GPU stack fails in one sentence now instead of 10 optimizers deep with
    a raw traceback (CLAUDE.md sec 6)."""
    if not torch.cuda.is_available():
        if os.environ.get("ALLOW_CPU") == "1" or SMOKE:
            log("[preflight] CUDA not available; ALLOW_CPU/SMOKE set -> running on CPU")
            _preflight_eviekf_self_check()
            return
        fail("CUDA is not available. This harness targets a single RTX A6000. "
             "Set ALLOW_CPU=1 only for a CPU correctness smoke test.")
    try:
        cap = torch.cuda.get_device_capability()
        sm = f"sm_{cap[0]}{cap[1]}"
        arch_list = list(torch.cuda.get_arch_list())
        name = torch.cuda.get_device_name()
    except Exception as e:
        fail(f"CUDA reports available but querying the device failed ({e!r}); "
             f"the torch/driver install is broken.")
        return
    if not any(a == sm for a in arch_list):
        log(f"[preflight] WARNING: torch arch_list={arch_list} has no exact match "
            f"for this GPU ({sm}, {name}); relying on the live op check below")
    try:
        a = torch.randn(256, 256, device="cuda")
        b = torch.randn(256, 256, device="cuda")
        _ = float((a @ b).sum().item())
        torch.cuda.synchronize()
    except Exception as e:
        fail(f"CUDA is present ({name}, {sm}) but a trivial matmul failed ({e!r}). "
             f"The torch build does not match this GPU -- reinstall torch for {sm}.")
    log(f"[preflight] OK: {name}  {sm}  torch={torch.__version__}  "
        f"arch_list={arch_list}")
    _preflight_eviekf_self_check()


# ---------------------------------------------------------------------------
#  Atomic checkpoint IO. tmp name carries PID + randomness so two processes
#  sharing a directory (e.g. a network mount) still cannot collide; the real
#  file is only ever replaced via os.replace after fsync; the previous good
#  copy is rotated to .bak first (CLAUDE.md sec 6).
# ---------------------------------------------------------------------------
def atomic_torch_save(obj, path):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp.{os.getpid()}."
                          f"{random.randint(0, 1 << 30):x}")
    with open(tmp, "wb") as fh:
        torch.save(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    if os.path.exists(path):
        try:
            os.replace(path, path + ".bak")
        except OSError:
            pass
    os.replace(tmp, path)


def load_cell_ckpt(path):
    for p in (path, path + ".bak"):
        if os.path.exists(p):
            try:
                return torch.load(p, map_location="cpu", weights_only=False)
            except Exception as e:
                log(f"[resume] {p} is unreadable ({e!r}); trying next")
    return None


def _read_json(path, default):
    if os.path.exists(path):
        try:
            with open(path) as fh:
                return json.load(fh)
        except Exception as e:
            log(f"[warn] {path} unreadable ({e!r}); starting fresh")
    return default


def _write_json(path, obj):
    d = os.path.dirname(path) or "."
    tmp = os.path.join(d, f".{os.path.basename(path)}.tmp.{os.getpid()}")
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, path)


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ---------------------------------------------------------------------------
#  LR schedule -- manual cosine so it works with Muon's two param groups
#  (different base LRs) and needs no scheduler state in the checkpoint.
# ---------------------------------------------------------------------------
def cosine_factor(step, total):
    if total <= 1:
        return 1.0
    frac = min(max(step / float(total), 0.0), 1.0)
    return ETA_MIN_FRAC + (1.0 - ETA_MIN_FRAC) * 0.5 * (1.0 + math.cos(math.pi * frac))


def _apply_lr(optimizer, base_lrs, factor):
    for g, base in zip(optimizer.param_groups, base_lrs):
        g["lr"] = base * factor


def _power_iteration_top_eig(model, params, compute_loss, iters=10, eps=1e-8):
    """|top Hessian eigenvalue| by power iteration on Hessian-vector products
    (one batch). Descriptive only; run once per cell for Hessian-consuming
    optimizers (AdaHessian/Sophia-style)."""
    with torch.enable_grad():
        loss = compute_loss()
    grads = torch.autograd.grad(loss, params, create_graph=True)
    v = [torch.randn_like(p) for p in params]
    nrm = math.sqrt(sum(float(x.pow(2).sum().item()) for x in v)) + eps
    v = [x / nrm for x in v]
    ev = 0.0
    for _ in range(iters):
        gv = sum((g * vv).sum() for g, vv in zip(grads, v))
        hv = torch.autograd.grad(gv, params, retain_graph=True)
        ev = sum(float((h * vv).sum().item()) for h, vv in zip(hv, v))
        nrm = math.sqrt(sum(float(h.pow(2).sum().item()) for h in hv)) + eps
        v = [h.detach() / nrm for h in hv]
    return abs(ev)


def _is_finite(x):
    return x is not None and math.isfinite(float(x))


def _better(a, b):
    """True iff primary value a is strictly better than b (direction aware)."""
    if not _is_finite(a):
        return False
    if not _is_finite(b):
        return True
    return a < b if PRIMARY_LOWER_BETTER else a > b


def _absurd_primary(v, peer_median=None):
    """Absolute + peer-relative sanity (CLAUDE.md sec 6: an absolute range, not
    just 'worse than this run's own best')."""
    if not _is_finite(v):
        return True
    lo, hi = PRIMARY_ABS_RANGE
    if v < lo or v > hi:
        return True
    if peer_median is not None and _is_finite(peer_median) and peer_median != 0:
        if PRIMARY_LOWER_BETTER and v > PEER_DIVERGENCE_FACTOR * peer_median:
            return True
        if (not PRIMARY_LOWER_BETTER) and peer_median > 0 and \
           v < peer_median / PEER_DIVERGENCE_FACTOR:
            return True
    return False


def _infer_param_roles(model):
    """{param: role} for the Evie-KF per-module-category gain breakdown
    (claude_optim.md sec 4.3.5). Substring match on the qualified parameter
    name; checked against the actual module names in the language GPT
    (wte/wpe/qkv/proj/fc1/fc2/lm_head), the vision ResNet (conv/bn/fc) and the
    finance MLP (net.N) -- vision/finance mostly land in "other", which is fine.
    """
    roles = {}
    for nm, p in model.named_parameters():
        if not p.requires_grad:
            continue
        low = nm.lower()
        if any(k in low for k in ("wte", "wpe", "embed", "emb.", "tok_emb",
                                  "pos_emb", "embedding")):
            r = "embedding"
        elif any(k in low for k in ("attn", "qkv", ".proj", "attention", "q_proj",
                                    "k_proj", "v_proj", "out_proj")):
            r = "attention"
        elif any(k in low for k in ("mlp", "ffn", "fc1", "fc2", "feed_forward",
                                    "w1", "w2", "w3")):
            r = "ffn"
        elif any(k in low for k in ("lm_head", "head", "classifier", "fc.weight",
                                    "fc.bias", ".fc.")):
            r = "head"
        else:
            r = "other"
        roles[p] = r
    return roles


def _model_has_batchnorm(model):
    """True if any module normalises ACROSS the batch dimension.

    Why this exists (fairness, not cosmetics): the Evie-KF family needs the
    split-batch noise sample, so it runs two half-batch forward/backwards per
    step instead of one full-batch one. On a LayerNorm model (the GPT) or a
    plain MLP (finance) that is mathematically free -- (gA + gB)/2 over two
    disjoint equal halves is EXACTLY the full-batch gradient, so the baselines
    can keep the cheaper single backward with no confound.

    On a BatchNorm model (the CIFAR-stem ResNet-18, i.e. all 4 vision tasks)
    it is NOT free: BN over two halves is ghost batch norm, a real and
    well-documented regulariser, and it also updates BN's running statistics
    twice per step from half-batch estimates. Left uncorrected, the Evie arms
    would be training a DIFFERENT objective than the baselines, and any vision
    win would be uninterpretable -- "is this the preconditioner or is it GBN?"
    So where BN is present, EVERY optimizer takes the split path: identical
    training signal, identical BN behaviour, and the preconditioner is then the
    only thing that differs. FLOP-neutral (two half backwards ~= one full one).
    """
    bn_types = tuple(
        t for t in (getattr(nn, n, None) for n in
                    ("BatchNorm1d", "BatchNorm2d", "BatchNorm3d",
                     "SyncBatchNorm", "InstanceNorm1d", "InstanceNorm2d",
                     "InstanceNorm3d"))
        if t is not None)
    return any(isinstance(m, bn_types) for m in model.modules())


# ---------------------------------------------------------------------------
#  One training run: (optimizer, unit, seed) for the main sweep, or one
#  (optimizer, unit) LR-search probe. Fully resumable from ckpt_path.
# ---------------------------------------------------------------------------
def train_run(opt_name, data, unit, lr, adamw_lr, total_steps, seed,
              ckpt_path, is_lr_search=False, knob=None):
    criterion = make_criterion()
    tag = f"{opt_name}/{unit}/seed{seed}" + ("/LR" if is_lr_search else "")

    ck = load_cell_ckpt(ckpt_path) if ckpt_path else None
    if ck is not None and ck.get("complete"):
        return ck["result"]
    if ck is not None and "model" not in ck:      # slim/partial file -> restart
        ck = None

    train_loader, val_loader, test_loader = make_loaders(data, unit, BATCH_SIZE, seed)

    if ck is None:
        set_seed(seed * 100003 + (7 if is_lr_search else 0))
        model = build_model(data, unit).to(DEVICE)
        optimizer = build_optimizer(opt_name, model, lr, adamw_lr=adamw_lr,
                                    knob=knob,
                                    eviekf_maxf=TASK.get("eviekf_maxf"))
        base_lrs = optimizer_lr_scale(optimizer)
        start_step = 0
        best_val = math.inf if PRIMARY_LOWER_BETTER else -math.inf
        best_state = None
        acc = dict(clip_events=0, step_sq_sum=0.0, step_sq_n=0, n_params=0,
                   wall=0.0, hess_norms=[], val_curve=[],
                   # --- descriptive-only diagnostics (CLAUDE.md sec 9.5 dir.):
                   # recorded per cell, NEVER significance-tested. Primary stays
                   # the one pre-registered metric per domain. ---
                   grad_norms=[], theta_sq_sum=0.0, uwr_sum=0.0, uwr_n=0,
                   upcos_sum=0.0, upcos_n=0, hess_trace_samples=[],
                   train_loss_ema=None, final_train_loss=float("nan"),
                   loss_at=[None, None, None], eval_wall=[], w0_norm=None,
                   flops=0.0)
        _tp = [p for p in model.parameters() if p.requires_grad]
        acc["n_params"] = int(sum(p.numel() for p in _tp))
        with torch.no_grad():
            acc["w0_norm"] = math.sqrt(sum(float(p.detach().pow(2).sum().item())
                                           for p in _tp))
        diverged, fail_reason = False, None
    else:
        model = build_model(data, unit).to(DEVICE)
        model.load_state_dict(ck["model"])
        optimizer = build_optimizer(opt_name, model, lr, adamw_lr=adamw_lr,
                                    knob=knob,
                                    eviekf_maxf=TASK.get("eviekf_maxf"))
        optimizer.load_state_dict(ck["optimizer"])
        base_lrs = ck["base_lrs"]
        start_step = ck["step"]
        best_val = ck["best_val"]
        best_state = ck["best_state"]
        acc = ck["acc"]
        diverged = ck.get("diverged", False)
        fail_reason = ck.get("fail_reason", None)
        load_rng_state(ck["rng"])
        log(f"[resume] {tag} from step {start_step}/{total_steps}")

    params = [p for p in model.parameters() if p.requires_grad]
    hessian_consumer = opt_name in HESSIAN_CONSUMERS
    noise_consumer = opt_name in NOISE_CONSUMERS      # split-batch noise hook
    # Every optimizer takes the split path on a BatchNorm model, so the Evie
    # family's required split cannot hand it ghost-batch-norm regularisation the
    # baselines never get. See _model_has_batchnorm() for the full argument.
    # TASK["force_split_path"] puts the baselines on the split path on a model
    # with no BatchNorm too. It is a MEMORY measure, not a fairness one: the two
    # half-batch backwards never materialise the full-batch logits tensor, which
    # on a 50k-vocab model is batch*ctx*vocab*4 bytes and is what OOM'd
    # language_openwebtext on a T4 (3.07 GiB, exactly). The gradient is unchanged
    # -- test_fairness_and_aggregation.py proves the split path reproduces the
    # full-batch gradient exactly on a BN-free model -- so no arm's arithmetic
    # moves, only its peak allocation.
    split_batch = (noise_consumer or _model_has_batchnorm(model)
                   or bool(TASK.get("force_split_path")))
    if split_batch and not noise_consumer:
        _why = ("BatchNorm model" if _model_has_batchnorm(model)
                else "force_split_path (peak-memory measure; gradient unchanged)")
        log(f"[fairness] {tag}: {_why} -> {opt_name} uses the same "
            f"split-batch forward/backward as the Evie arms (identical training "
            f"signal; preconditioner is the only difference)")
    if noise_consumer and hasattr(optimizer, "set_param_roles"):
        optimizer.set_param_roles(_infer_param_roles(model))
    prev_delta = None                     # for update_cos_sim_sampled (not persisted)
    try:
        fwd_flops = float(estimate_fwd_flops(model, BATCH_SIZE))
    except Exception:
        fwd_flops = float("nan")
    loader_iter = itertools.cycle(train_loader)
    if start_step > 0:
        # the DataLoader shuffle order is fixed by its seeded generator, so
        # skipping (start_step mod len) batches restores the exact (x, y)
        # sequence a fresh run would see from here (a resumed run can still
        # differ at the augmentation-RNG level; an uninterrupted run is the
        # reference).
        try:
            nb = len(train_loader)
            for _ in range(start_step % max(nb, 1)):
                next(loader_iter)
        except TypeError:
            pass
    t_wall = time.time()

    def save_cell(complete=False, result=None):
        if not ckpt_path:
            return
        if complete:
            # slim terminal record (~KB): the full result lives in results.json;
            # this just lets a re-run skip the cell. Drop the big .bak too.
            atomic_torch_save({"complete": True, "result": result,
                               "opt_name": opt_name, "unit": unit, "seed": seed},
                              ckpt_path)
            for p in (ckpt_path + ".bak",):
                try:
                    os.remove(p)
                except OSError:
                    pass
            return
        atomic_torch_save({
            "step": step, "opt_name": opt_name, "unit": unit, "seed": seed,
            "lr": lr, "adamw_lr": adamw_lr, "knob": knob, "total_steps": total_steps,
            "model": model.state_dict(), "optimizer": optimizer.state_dict(),
            "base_lrs": base_lrs, "best_val": best_val, "best_state": best_state,
            "acc": acc, "diverged": diverged, "fail_reason": fail_reason,
            "rng": rng_state(), "complete": False, "result": None,
        }, ckpt_path)

    step = start_step
    inputs = targets = None
    model.train()
    while step < total_steps:
        if _STOP["flag"]:
            acc["wall"] += time.time() - t_wall
            save_cell(complete=False)
            log(f"[signal] {tag}: checkpointed at step {step}; exiting cleanly")
            _log_fh.flush()
            sys.exit(0)

        batch = next(loader_iter)
        inputs, targets = batch_to_device(batch)

        measure_step = (step % EVAL_EVERY == 0)      # sample the step-size metric
        p_before = ([p.detach().clone() for p in params] if measure_step else None)

        optimizer.zero_grad(set_to_none=True)
        _ga = _gb = None
        if split_batch:
            # Split-batch (claude_optim.md sec 4.3.1): two DISJOINT half batches,
            # one fwd+bwd each -- same examples, same total FLOPs as one
            # full-batch backward, no extra data. p.grad <- (gA+gB)/2. For a
            # noise consumer the noise sample delta = (gA-gB)/2 is additionally
            # clipped by the SAME coefficient as g (else the SNR the gate sees is
            # distorted, spec sec 5) and handed to the optimizer via set_noise()
            # BEFORE step(). A non-noise-consumer only lands here on a BatchNorm
            # model, purely so its BN statistics match the Evie arms' -- it needs
            # no half-gradient copies, so it accumulates both halves in place
            # instead (same result, no extra memory).
            _n = inputs.size(0)
            _nh = _n // 2
            if _nh >= 1:
                _xa, _xb = inputs[:_nh], inputs[_nh:2 * _nh]
                _ya, _yb = targets[:_nh], targets[_nh:2 * _nh]
                _la = forward_loss(model, (_xa, _ya), criterion)
                _la.backward()
                if noise_consumer:
                    _ga = [(p.grad.detach().clone() if p.grad is not None else None)
                           for p in params]
                    optimizer.zero_grad(set_to_none=True)
                _lb = forward_loss(model, (_xb, _yb), criterion)
                _lb.backward()
                loss_v = 0.5 * (float(_la.detach().item())
                                + float(_lb.detach().item()))
                if noise_consumer:
                    _gb = [(p.grad.detach().clone() if p.grad is not None else None)
                           for p in params]
                    for p, a, b in zip(params, _ga, _gb):
                        if a is not None and b is not None:
                            p.grad = 0.5 * (a + b)
                        elif a is not None:
                            p.grad = a.clone()
                        elif b is not None:
                            p.grad = b.clone()
                else:
                    # grads were accumulated over both halves, i.e. gA + gB; the
                    # quantity we want is their mean, so halve in place.
                    for p in params:
                        if p.grad is not None:
                            p.grad.mul_(0.5)
            else:                                   # batch too small to split
                loss = forward_loss(model, (inputs, targets), criterion)
                loss.backward()
                loss_v = float(loss.detach().item())
        else:
            loss = forward_loss(model, (inputs, targets), criterion)
            loss.backward()
            loss_v = float(loss.detach().item())

        if not math.isfinite(loss_v):
            diverged, fail_reason = True, f"non-finite loss at step {step}"
            log(f"[diverge] {tag}: {fail_reason}")
            break

        pre_clip = torch.nn.utils.clip_grad_norm_(params, GRAD_CLIP_NORM)
        acc["grad_norms"].append(float(pre_clip))          # descriptive
        if float(pre_clip) > CLIP_METRIC_THRESHOLD:
            acc["clip_events"] += 1

        if noise_consumer:
            _pc = float(pre_clip)
            _coef = (GRAD_CLIP_NORM / (_pc + 1e-6)
                     if (math.isfinite(_pc) and _pc > GRAD_CLIP_NORM) else 1.0)
            if _ga is not None:
                _pairs = []
                for p, a, b in zip(params, _ga, _gb):
                    if a is not None and b is not None:
                        _pairs.append((p, (0.5 * (a - b)) * _coef))
                    else:
                        _pairs.append((p, torch.zeros_like(p)))
                optimizer.set_noise(_pairs)
            else:
                optimizer.set_noise([(p, torch.zeros_like(p)) for p in params])
            _ga = _gb = None

        # descriptive: EMA of the train loss + snapshots at 25/50/75% of budget
        b = 0.98
        acc["train_loss_ema"] = (loss_v if acc["train_loss_ema"] is None
                                 else b * acc["train_loss_ema"] + (1 - b) * loss_v)
        acc["final_train_loss"] = loss_v
        for _i, _f in enumerate((0.25, 0.5, 0.75)):
            if acc["loss_at"][_i] is None and step + 1 >= int(_f * total_steps):
                acc["loss_at"][_i] = acc["train_loss_ema"]

        if hessian_consumer and (step % HESS_FREQ == 0):
            try:
                _h = hutchinson_diag_hessian(
                    model, params,
                    lambda: forward_loss(model, (inputs, targets), criterion),
                    N_HUTCH, optimizer=optimizer, feed_optimizer=True)
                if _is_finite(optimizer.last_h_norm):
                    acc["hess_norms"].append(float(optimizer.last_h_norm))
                acc["hess_trace_samples"].append(
                    sum(float(h.sum().item()) for h in _h))          # descriptive
            except Exception as e:
                if classify_exception(e) == "env":
                    raise
                log(f"[warn] {tag}: Hutchinson probe failed at step {step} ({e!r})")

        _apply_lr(optimizer, base_lrs, cosine_factor(step, total_steps))
        optimizer.step()
        step += 1
        if math.isfinite(fwd_flops):
            acc["flops"] += 3.0 * fwd_flops + (
                N_HUTCH * 2.0 * fwd_flops
                if (hessian_consumer and (step - 1) % HESS_FREQ == 0) else 0.0)

        if p_before is not None:
            with torch.no_grad():
                dts = [(p.detach() - pb) for p, pb in zip(params, p_before)]
                d_sq = sum(float(x.pow(2).sum().item()) for x in dts)
                th_sq = sum(float(p.detach().pow(2).sum().item()) for p in params)
                flat = torch.cat([x.reshape(-1) for x in dts]).float().cpu()
            acc["step_sq_sum"] += d_sq
            acc["step_sq_n"] = acc.get("step_sq_n", 0) + 1
            acc["theta_sq_sum"] += th_sq
            if d_sq > 0 and th_sq > 0:                     # ||dtheta|| / ||theta||
                acc["uwr_sum"] += math.sqrt(d_sq / th_sq)
                acc["uwr_n"] += 1
            if (prev_delta is not None and prev_delta.numel() == flat.numel()
                    and flat.norm() > 0 and prev_delta.norm() > 0):
                acc["upcos_sum"] += float(torch.dot(flat, prev_delta).item()) / (
                    float(flat.norm()) * float(prev_delta.norm()))
                acc["upcos_n"] += 1
            prev_delta = flat
        p_before = None

        if (step % EVAL_EVERY == 0) or (step == total_steps):
            acc["eval_wall"].append([step, acc["wall"] + (time.time() - t_wall)])
            val_primary, _ = evaluate(model, val_loader, criterion)
            acc["val_curve"].append([step, val_primary])
            if _better(val_primary, best_val):
                best_val = val_primary
                best_state = {k: v.detach().cpu().clone()
                              for k, v in model.state_dict().items()}
            if noise_consumer:
                # Evie-KF mechanism telemetry (claude_optim.md sec 4.3.4/4.3.5):
                # descriptive ONLY -- never enters the Wilcoxon/Holm pipeline.
                acc.setdefault("eviekf_diag", []).append({
                    "step": step,
                    "cos": float(getattr(optimizer, "last_cos", float("nan"))),
                    "b_simple": float(getattr(optimizer, "b_simple", float("nan"))),
                    "tr_sigma": float(getattr(optimizer, "tr_sigma", float("nan"))),
                    "eff_rank": float(getattr(optimizer, "eff_rank", float("nan"))),
                    "gain_lo": float(getattr(optimizer, "gain_lo", float("nan"))),
                    "gain_hi": float(getattr(optimizer, "gain_hi", float("nan"))),
                    "n_active_steps": int(getattr(optimizer, "n_active_steps", 0)),
                })
                _rg = getattr(optimizer, "role_gain", {}) or {}
                if _rg:
                    acc["eviekf_role_gain"] = {
                        r: [float(v[0]), float(v[1]), float(v[2]), int(v[3])]
                        for r, v in _rg.items()}
            model.train()

        if (step % CKPT_EVERY == 0) and not is_lr_search:
            acc["wall"] += time.time() - t_wall
            save_cell(complete=False)
            t_wall = time.time()

    acc["wall"] += time.time() - t_wall

    if best_state is not None:
        model.load_state_dict(best_state)
    if is_lr_search:
        test_primary, test_aux = float("nan"), {}     # LR pick uses best_val only
    else:
        try:
            test_primary, test_aux = evaluate(model, test_loader, criterion)
        except Exception as e:
            if classify_exception(e) == "env":
                raise
            test_primary, test_aux = float("nan"), {}
            diverged, fail_reason = True, f"test eval failed: {e!r}"

    n_params = max(acc["n_params"], 1)
    step_sq_n = max(acc.get("step_sq_n", 0), 1)

    # -------- descriptive-only derived diagnostics (NO significance tests) -----
    vc = acc["val_curve"]
    steps_to_best_val = steps_to_90 = frac_90 = wall_to_90 = auc_val = float("nan")
    if len(vc) >= 2:
        vs = [v for _, v in vc]
        ss = [s for s, _ in vc]
        bi = (min(range(len(vs)), key=lambda i: vs[i]) if PRIMARY_LOWER_BETTER
              else max(range(len(vs)), key=lambda i: vs[i]))
        steps_to_best_val = ss[bi]
        first_v, best_v = vs[0], vs[bi]
        if best_v != first_v:
            for s, v in vc:
                if (v - first_v) / (best_v - first_v) >= 0.9:
                    steps_to_90 = s
                    break
            if _is_finite(steps_to_90):
                frac_90 = steps_to_90 / max(total_steps, 1)
                wall_to_90 = dict((s, w) for s, w in acc["eval_wall"]).get(
                    steps_to_90, float("nan"))
        area = sum(0.5 * (vs[i] + vs[i - 1]) * (ss[i] - ss[i - 1])
                   for i in range(1, len(vc)))
        auc_val = area / max(ss[-1] - ss[0], 1)

    with torch.no_grad():
        w_end = math.sqrt(sum(float(p.detach().pow(2).sum().item())
                              for p in params))
    w0 = acc.get("w0_norm") or w_end
    gn = acc.get("grad_norms", [])
    hts = acc.get("hess_trace_samples", [])
    final_tl = acc.get("final_train_loss", float("nan"))

    hess_top_eig = float("nan")
    if hessian_consumer and not is_lr_search and inputs is not None:
        try:
            hess_top_eig = _power_iteration_top_eig(
                model, params,
                lambda: forward_loss(model, (inputs, targets), criterion), iters=10)
        except Exception as e:
            if classify_exception(e) == "env":
                raise

    result = {
        "opt": opt_name, "unit": unit, "seed": seed, "lr_used": lr,
        "knob_used": knob,
        "primary": test_primary, "best_val_primary": best_val,
        "steps_run": step, "diverged": bool(diverged), "fail_reason": fail_reason,
        "wall_s": acc["wall"],
        "sec_per_step": acc["wall"] / max(step, 1),        # cost column (spec 4.3.6)
        "mean_step_size": math.sqrt(acc["step_sq_sum"] / step_sq_n / n_params),
        "clip_freq": acc["clip_events"] / max(step, 1),
        "peak_mem_mb": (torch.cuda.max_memory_allocated() / 1024 ** 2
                        if torch.cuda.is_available() else float("nan")),
        "hess_norm": (float(np.mean(acc["hess_norms"])) if acc["hess_norms"]
                      else float("nan")),
        "hess_cv": (float(np.std(acc["hess_norms"]) / np.mean(acc["hess_norms"]))
                    if acc["hess_norms"] and np.mean(acc["hess_norms"]) > 0
                    else float("nan")),
        "n_params": n_params,
        "val_curve": acc["val_curve"],
        # ---- descriptive-only diagnostics (see CLAUDE.md sec 5: primary metric
        # per domain is the ONLY thing significance-tested; these are appendix /
        # behaviour telemetry, no p-values) ----
        "steps_to_best_val": steps_to_best_val,
        "steps_to_90pct_best": steps_to_90,
        "frac_budget_to_90pct_best": frac_90,
        "wallclock_to_90pct_best_s": wall_to_90,
        "train_loss_at_25pct": acc["loss_at"][0] if acc["loss_at"][0] is not None
                               else float("nan"),
        "train_loss_at_50pct": acc["loss_at"][1] if acc["loss_at"][1] is not None
                               else float("nan"),
        "train_loss_at_75pct": acc["loss_at"][2] if acc["loss_at"][2] is not None
                               else float("nan"),
        "final_train_loss": final_tl,
        "gen_gap_loss": (test_aux.get("test_loss", float("nan")) - final_tl
                         if _is_finite(final_tl) else float("nan")),
        "auc_val_curve": auc_val,
        "update_weight_ratio": (acc["uwr_sum"] / acc["uwr_n"]
                                if acc.get("uwr_n") else float("nan")),
        "update_cos_sim_sampled": (acc["upcos_sum"] / acc["upcos_n"]
                                   if acc.get("upcos_n") else float("nan")),
        "grad_norm_mean": float(np.mean(gn)) if gn else float("nan"),
        "grad_norm_p95": float(np.percentile(gn, 95)) if gn else float("nan"),
        "weight_norm_start": w0,
        "weight_norm_end": w_end,
        "weight_norm_growth": (w_end / w0) if w0 > 0 else float("nan"),
        "throughput_samples_per_s": (step * BATCH_SIZE) / max(acc["wall"], 1e-9),
        "flops_per_step_est": acc.get("flops", 0.0) / max(step, 1),
        "total_flops_est": acc.get("flops", float("nan")),
        "hess_trace": float(np.mean(hts)) if hts else float("nan"),
        "hess_top_eig": hess_top_eig,
    }
    for k in AUX_METRIC_KEYS:
        result[k] = test_aux.get(k, float("nan"))

    # ---- Evie-KF mechanism telemetry (descriptive only, spec sec 4.3.4/4.3.5) --
    _edg = acc.get("eviekf_diag", [])

    def _dstat(key, fn=np.mean):
        _vs = [d[key] for d in _edg if _is_finite(d.get(key))]
        return float(fn(_vs)) if _vs else float("nan")

    _n_active = int(_edg[-1]["n_active_steps"]) if _edg else 0
    result.update({
        "eviekf_cos_mean": _dstat("cos"),
        "eviekf_cos_min": _dstat("cos", np.min),
        "eviekf_b_simple_mean": _dstat("b_simple"),
        "eviekf_tr_sigma_mean": _dstat("tr_sigma"),
        "eviekf_eff_rank_mean": _dstat("eff_rank"),
        "eviekf_gain_lo_min": _dstat("gain_lo", np.min),
        "eviekf_gain_hi_max": _dstat("gain_hi", np.max),
        "eviekf_n_active_steps": _n_active,
        "eviekf_role_gain": acc.get("eviekf_role_gain", {}),
        # The full trace, not just the reductions above. The spec
        # (evie-kf-full-specification.md sec 8.2/sec 12) is explicit: "log raw,
        # analyse afterwards -- never log only a summary statistic", because the
        # mechanism figures (F4: gain spectrum and cos(step, Adam) over training;
        # F5: noise anisotropy) need the trajectory and it cannot be
        # reconstructed from a mean. One dict per eval point (~20-40 per cell),
        # EvieKF-family cells only -- a few KB.
        "eviekf_diag_trace": _edg,
    })
    # n_active_steps == 0 at the end of a real run means set_noise() never fired
    # and Evie-KF silently ran as AdamW -- same severity class as a divergence
    # flag (claude_optim.md sec 4.3.4), not a silent zero.
    if (noise_consumer and not is_lr_search
            and step > int(getattr(optimizer, "warmup", EVIEKF_WARMUP)) + 1
            and _n_active == 0):
        diverged = True
        fail_reason = (fail_reason or
                       "Evie-KF noise hook never fired (n_active_steps=0): "
                       "set_noise() wiring bug -- this run was silently AdamW")
        result["diverged"] = True
        result["fail_reason"] = fail_reason
        log(f"[diverge] {tag}: {fail_reason}")

    if is_lr_search:
        # LR selection uses the BEST-checkpoint val metric, never the last step
        # (CLAUDE.md sec 6).
        result["lr_search_score"] = best_val
    save_cell(complete=True, result=result)
    return result


# ---------------------------------------------------------------------------
#  LR search: 7-point half-decade grid per optimizer family, tuned at
#  LR_SEARCH_STEPS (~25% of budget), 1 seed; boundary rule -> extend & re-search
#  if the pick lands on a grid edge; then freeze (CLAUDE.md sec 5). Optimizers
#  in KNOB_GRID (the Evie-KF family: gamma; Shampoo: beta) get a SECOND axis
#  searched jointly with LR, same discipline (claude_optim.md sec 4.3.2).
# ---------------------------------------------------------------------------
def _score_lr(opt_name, data, units, lr, adamw_lr, knob=None):
    # LR-search runs are short (25% budget) and write NO checkpoint -- a full
    # per-probe .pt would be ~100s of MB x dozens of probes. If interrupted, the
    # in-flight grid point is simply re-run; completed points are cached in
    # LR_JSON (see lr_search()).
    scores = []
    for unit in units:
        r = train_run(opt_name, data, unit, lr, adamw_lr, LR_SEARCH_STEPS,
                      LR_SEARCH_SEED, ckpt_path=None, is_lr_search=True, knob=knob)
        s = r["lr_search_score"]
        if _is_finite(s) and not _absurd_primary(s):
            scores.append(s)
    if not scores:
        return math.inf if PRIMARY_LOWER_BETTER else -math.inf
    return float(np.median(scores))


def _safe(s):
    return "".join(c if c.isalnum() or c in "-._" else "_" for c in str(s))


def search_lr_for(opt_name, data, units, adamw_lr, log_prefix="", knob=None):
    """One optimizer's 7-point half-decade LR grid + boundary-extend rule
    (CLAUDE.md sec 5). Returns (best_lr, tried_scores, boundary_checks). Used by
    lr_search() below and by the batch/model-size sweep drivers. `knob` (if
    given) is the fixed second-axis value the cells will run at -- the sweeps
    pass it so the LR is tuned at the SAME knob the sweep cell uses, not at
    build_optimizer()'s default. Writes no state -- the caller persists."""
    tried = {}

    def _eval(lrs):
        for lr in lrs:
            if lr in tried and _is_finite(tried[lr]):
                continue
            tried[lr] = _score_lr(opt_name, data, units, lr, adamw_lr, knob=knob)
            log(f"{log_prefix}[lr-search] {opt_name}  lr={lr:.3e}  "
                f"score={tried[lr]:.6g}")

    def _pick():
        return (min(tried, key=tried.get) if PRIMARY_LOWER_BETTER
                else max(tried, key=tried.get))

    grid = sorted(half_decade_grid(LR_GRID_CENTRE[opt_name]))
    _eval(grid)
    best = _pick()
    bchecks = 0
    while best in (grid[0], grid[-1]) and bchecks < 3:
        bchecks += 1
        r = 10.0 ** 0.5
        ext = [float(f"{(best / r ** k) if best == grid[0] else (best * r ** k):.4g}")
               for k in (1, 2)]
        log(f"{log_prefix}[lr-search] {opt_name}: pick {best:.3e} at grid edge -> "
            f"extend with {ext} and re-search (boundary check #{bchecks})")
        _eval(ext)
        grid = sorted(tried)
        best = _pick()
    return float(best), tried, bchecks


def _joint_lr_knob_search(opt_name, data, units, adamw_lr, prior, persist):
    """LR grid x optional knob grid (KNOB_GRID). Scored at LR_SEARCH_STEPS / 1
    seed on the best-checkpoint val metric (CLAUDE.md sec 6). Boundary-extend on
    BOTH axes: if the LR pick lands on a grid edge, extend the LR grid at the
    chosen knob and re-search; likewise extend the knob grid at the chosen LR
    (claude_optim.md sec 4.3.2 -- gamma=100 was edge-pinned once already at PoC
    scale). Returns (best_lr, best_knob, b_lr, b_knob)."""
    knobs = KNOB_GRID.get(opt_name)
    # FIXED knob (TASK["fixed_knob"]): search the LR grid ONLY, at a stated
    # second-axis value. This is the "same tuning budget as the baselines" arm
    # -- 7 LR points and a default gamma, instead of the joint 49-99-point
    # search -- so the knob grid collapses to one value and the knob
    # boundary-extend rule does not apply (there is no grid to be at the edge
    # of). Written into lr_search.json via knob_grid so the provenance travels.
    knob_fixed = (TASK.get("fixed_knob") or {}).get(opt_name)
    if knob_fixed is not None:
        knobs = [float(knob_fixed)]
    knob_vals = [None] if knobs is None else [float(k) for k in knobs]
    # FIXED lr (TASK["fixed_lr"]): the mirror image -- pin the learning rate at a
    # stated value and search the second axis only. Used where the question is
    # about the knob itself (does gamma transfer across tasks?) and re-tuning lr
    # would confound the answer. The LR boundary rule does not apply for the same
    # reason it does not under fixed_knob: there is no grid to sit at the edge of.
    lr_fixed = (TASK.get("fixed_lr") or {}).get(opt_name)
    lr_vals = ([float(lr_fixed)] if lr_fixed is not None
               else sorted(half_decade_grid(LR_GRID_CENTRE[opt_name])))
    scored = dict(prior.get("scored", {}))              # "lr|knob" -> median score
    b_lr = int(prior.get("boundary_checks_lr", 0))
    b_knob = int(prior.get("boundary_checks_knob", 0))

    def _key(lr, kn):
        return f"{lr:.6e}|{'none' if kn is None else format(kn, '.6e')}"

    def _ensure():
        for lr in lr_vals:
            for kn in knob_vals:
                k = _key(lr, kn)
                if k in scored and _is_finite(scored[k]):
                    continue
                scored[k] = _score_lr(opt_name, data, units, float(lr), adamw_lr,
                                      knob=kn)
                log(f"[lr-search] {opt_name}  lr={lr:.3e}  "
                    f"knob={'-' if kn is None else format(kn, 'g')}  "
                    f"score={scored[k]:.6g}")
                persist(scored, b_lr, b_knob, lr_vals, knob_vals, complete=False)

    def _best():
        pick, pv = None, (math.inf if PRIMARY_LOWER_BETTER else -math.inf)
        for lr in lr_vals:
            for kn in knob_vals:
                s = scored.get(_key(lr, kn))
                if not _is_finite(s):
                    continue
                if (s < pv) if PRIMARY_LOWER_BETTER else (s > pv):
                    pv, pick = s, (lr, kn)
        return pick or (lr_vals[len(lr_vals) // 2], knob_vals[0])

    _ensure()
    bl, bk = _best()
    while lr_fixed is None and bl in (lr_vals[0], lr_vals[-1]) and b_lr < 3:
        b_lr += 1
        r = 10.0 ** 0.5
        at_low = (bl == lr_vals[0])
        ext = [float(f"{(bl / r ** k) if at_low else (bl * r ** k):.4g}")
               for k in (1, 2)]
        log(f"[lr-search] {opt_name}: LR pick {bl:.3e} at grid edge -> extend "
            f"{ext} and re-search (boundary check #{b_lr})")
        lr_vals = sorted(set(lr_vals) | set(ext))
        _ensure()
        bl, bk = _best()
    if knobs is not None and knob_fixed is None:
        while bk in (knob_vals[0], knob_vals[-1]) and b_knob < 3:
            b_knob += 1
            ext = knob_grid_extend(knob_vals, at_low=(bk == knob_vals[0]))
            if not ext:
                break
            log(f"[lr-search] {opt_name}: knob pick {bk:g} at grid edge -> extend "
                f"{ext} and re-search (boundary check #{b_knob})")
            knob_vals = sorted(set(knob_vals) | set(ext))
            _ensure()
            bl, bk = _best()
    persist(scored, b_lr, b_knob, lr_vals, knob_vals, complete=True,
            lr=bl, knob=bk)
    return float(bl), (None if bk is None else float(bk)), b_lr, b_knob


def lr_search(data):
    cached = _read_json(LR_JSON, {})
    lr_used, knob_used = {}, {}
    for k, v in cached.items():
        if isinstance(v, dict) and v.get("complete") and "lr" in v:
            lr_used[k] = float(v["lr"])
            knob_used[k] = (None if v.get("knob") is None else float(v["knob"]))
    units = lr_search_units(data)
    log_sep("-")

    # PINNED (lr, knob): a diagnostic file states the hyper-parameters instead of
    # searching for them, so the question it asks ("does this arm behave
    # differently at THIS lr?") is not re-contaminated by another noisy search.
    # Pinned values are written into lr_search.json with pinned=true so the
    # provenance travels with the results.
    for opt_name, pin in (TASK.get("pinned_lr") or {}).items():
        if opt_name not in ACTIVE_OPT_NAMES:
            continue
        lr_used[opt_name] = float(pin["lr"])
        knob_used[opt_name] = (None if pin.get("knob") is None
                               else float(pin["knob"]))
        cached[opt_name] = {"lr": lr_used[opt_name], "knob": knob_used[opt_name],
                            "grid": {}, "complete": True, "pinned": True,
                            "pinned_note": pin.get("note", "")}
        _write_json(LR_JSON, cached)
        log(f"[lr-search] {opt_name}: PINNED lr={lr_used[opt_name]:.3e}"
            + ("" if knob_used[opt_name] is None
               else f"  knob={knob_used[opt_name]:g}")
            + "  (no search on this arm)"
            + (f"  -- {pin['note']}" if pin.get("note") else ""))

    log(f"[lr-search] units={units}  steps={LR_SEARCH_STEPS}  seed={LR_SEARCH_SEED}"
        f"  optimizers={ACTIVE_OPT_NAMES}")

    # Muon's AdamW sub-group runs at AdamW's frozen lr, so Muon cannot be searched
    # until AdamW has been. In a single whole-task run that is guaranteed by
    # OPT_NAMES' order. When the task is split across machines with BENCH12_ARMS,
    # it is not -- so Muon's machine must be given a checkpoint in which AdamW's
    # search is already complete, and this is where that is checked. Failing here
    # with a sentence beats silently training Muon's AdamW group at Muon's own lr,
    # which is a ~20x wrong learning rate and would look like a divergence.
    if "Muon" in ACTIVE_OPT_NAMES and lr_used.get("AdamW") is None:
        raise SystemExit(
            "Muon needs AdamW's frozen lr and this run does not have it. Either "
            "run AdamW in the same job (BENCH12_ARMS=AdamW,Muon), or attach a "
            "checkpoint whose lr_search.json already has AdamW complete -- the "
            "AdamW machine's PAUSED/DONE archive does exactly that.")

    for opt_name in ACTIVE_OPT_NAMES:
        if opt_name in lr_used:
            log(f"[lr-search] {opt_name}: cached lr={lr_used[opt_name]:.3e}"
                + ("" if knob_used.get(opt_name) is None
                   else f"  knob={knob_used[opt_name]:g}"))
            continue
        adamw_lr = lr_used.get("AdamW")            # Muon reuses frozen AdamW lr
        entry = (cached.get(opt_name) if isinstance(cached.get(opt_name), dict)
                 else {})
        prior = {
            "scored": entry.get("grid", {}) or {},
            "boundary_checks_lr": entry.get("boundary_checks_lr", 0),
            "boundary_checks_knob": entry.get("boundary_checks_knob", 0),
        }

        def _persist(scored, blr, bknob, lrv, knv, complete, lr=None, knob=None):
            cached[opt_name] = {
                "grid": scored,
                "boundary_checks_lr": blr, "boundary_checks_knob": bknob,
                "centre": LR_GRID_CENTRE[opt_name],
                "lr_grid": [float(x) for x in lrv],
                "knob_grid": (None if knv == [None] else [float(x) for x in knv]),
                "complete": bool(complete),
            }
            if complete:
                cached[opt_name]["lr"] = float(lr)
                cached[opt_name]["knob"] = (None if knob is None else float(knob))
            _write_json(LR_JSON, cached)

        best_lr, best_knob, blr, bknob = _joint_lr_knob_search(
            opt_name, data, units, adamw_lr, prior, _persist)
        lr_used[opt_name] = best_lr
        knob_used[opt_name] = best_knob
        log(f"[lr-search] {opt_name}: FROZEN lr={best_lr:.3e}"
            + ("" if best_knob is None else f"  knob={best_knob:g}")
            + f"  (LR boundary checks: {blr}, knob boundary checks: {bknob})")
    log("[lr-search] frozen: " + ", ".join(
        f"{k}={lr_used[k]:.3e}" + ("" if knob_used.get(k) is None
                                   else f"/k={knob_used[k]:g}")
        for k in ACTIVE_OPT_NAMES))
    return lr_used, knob_used


# ---------------------------------------------------------------------------
#  Main sweep
# ---------------------------------------------------------------------------
def run_sweep(data, lr_used, knob_used=None):
    knob_used = knob_used or {}
    results = _read_json(RESULTS_JSON, {"cells": {}, "meta": {}})
    cells = results["cells"]
    units = main_units(data)
    total = len(ACTIVE_OPT_NAMES) * len(units) * len(SEEDS)
    log_sep("-")
    log(f"[sweep] {len(ACTIVE_OPT_NAMES)} optimizers x {len(units)} units x "
        f"{len(SEEDS)} seeds = {total} cells")

    done = 0
    for opt_name in ACTIVE_OPT_NAMES:
        for unit in units:
            for seed in SEEDS:
                ckey = f"{opt_name}|{unit}|{seed}"
                if ckey in cells and cells[ckey].get("recorded"):
                    done += 1
                    continue
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                ckpt = os.path.join(CELL_DIR,
                                    f"{_safe(opt_name)}__{_safe(unit)}__seed{seed}.pt")
                try:
                    r = train_run(opt_name, data, unit, lr_used[opt_name],
                                  lr_used.get("AdamW"), MAX_STEPS, seed, ckpt,
                                  is_lr_search=False, knob=knob_used.get(opt_name))
                    r["recorded"] = True
                    if _absurd_primary(r["primary"]):
                        r["diverged"] = True
                        r["fail_reason"] = (r.get("fail_reason")
                                            or "primary outside absolute sane range")
                    cells[ckey] = r
                except SystemExit:
                    raise
                except Exception as e:
                    kind = classify_exception(e)
                    if kind == "env":
                        log(f"[sweep] ENVIRONMENT error in {ckey}: {e!r}")
                        log(traceback.format_exc())
                        fail(f"environment/CUDA error while running {ckey}: {e!r}. "
                             f"This is NOT a diverged run -- fix the environment. "
                             f"Completed cells are safe; re-run to resume.")
                    log(f"[sweep] numerical failure in {ckey}: {e!r} -> marking "
                        f"cell failed, continuing")
                    log(traceback.format_exc())
                    cells[ckey] = {
                        "opt": opt_name, "unit": unit, "seed": seed,
                        "lr_used": lr_used[opt_name],
                        "knob_used": knob_used.get(opt_name),
                        "primary": float("nan"),
                        "diverged": True, "fail_reason": f"exception: {e!r}",
                        "recorded": True,
                    }
                done += 1
                _write_json(RESULTS_JSON, results)
                log(f"[sweep] {done}/{total}  {ckey}  "
                    f"primary={cells[ckey].get('primary')}  "
                    f"diverged={cells[ckey].get('diverged')}")

    # ---- peer-relative divergence pass (CLAUDE.md sec 6): a cell broken from
    # early on has no 'own best' to compare against, so compare to peers. ----
    healthy = [c["primary"] for c in cells.values()
               if not c.get("diverged") and _is_finite(c.get("primary"))]
    peer_median = float(np.median(healthy)) if healthy else None
    if peer_median is not None:
        for ckey, c in cells.items():
            if not c.get("diverged") and _absurd_primary(c.get("primary"), peer_median):
                c["diverged"] = True
                c["fail_reason"] = (c.get("fail_reason")
                                    or f"primary {c.get('primary'):.4g} is "
                                       f">{PEER_DIVERGENCE_FACTOR}x off the task "
                                       f"peer median {peer_median:.4g}")
                log(f"[diverge] {ckey}: {c['fail_reason']}")
    results["meta"] = {
        "task_name": TASK_NAME, "domain": DOMAIN, "primary_metric": PRIMARY_METRIC,
        "primary_lower_better": PRIMARY_LOWER_BETTER, "lr_used": lr_used,
        "knob_used": knob_used, "opt_names": list(ACTIVE_OPT_NAMES),
        "opt_set": TASK.get("opt_set", "main"),
        "seeds": SEEDS, "max_steps": MAX_STEPS, "units": units,
        "peer_median_primary": peer_median,
        "nonstandard_ppl_caveat": TASK.get("nonstandard_ppl_caveat"),
        "torch": torch.__version__, "device": str(DEVICE),
        "timestamp": datetime.now().isoformat(),
    }
    _write_json(RESULTS_JSON, results)
    return results


# ---------------------------------------------------------------------------
#  Aggregation + per-task descriptive report. Cross-12-task Wilcoxon lives in
#  benchmark12/aggregate.py; per file we still print the one-task picture and a
#  paired Wilcoxon across UNITS when a task has >=6 of them (finance panels).
# ---------------------------------------------------------------------------
def _cell_ok(cells, opt_name, unit, seed):
    c = cells.get(f"{opt_name}|{unit}|{seed}")
    return bool(c) and not c.get("diverged") and _is_finite(c.get("primary"))


def _complete_units(cells, units):
    """The PAIRED complete-case unit set: a unit survives only if every
    optimizer produced a finite, non-diverged cell there at every seed.

    Why, and why this is not fussiness: the old behaviour reduced over whatever
    cells happened to be healthy, per optimizer independently. That silently
    REWARDS instability -- an optimizer that blows up on the 15 hardest tickers
    of a 49-ticker panel gets its panel number computed over the 34 easy ones it
    survived, and then looks better than an optimizer that finished all 49. It
    is the same class of bug CLAUDE.md sec 6 flags twice. Dropping a unit for
    EVERY optimizer whenever ANY optimizer failed there keeps the comparison
    paired, which is what the whole design rests on.
    """
    return [u for u in units
            if all(_cell_ok(cells, o, u, s)
                   for o in ACTIVE_OPT_NAMES for s in SEEDS)]


def _cell_value_by_seed(cells, opt_name, units, how="median"):
    """Reduction over units of the test primary, per seed -> {seed: value}.

    Default reduction is the MEDIAN, not the mean. Finance panel MSE is in raw
    squared-log-return units -- _build_ticker_arrays() standardises X but never
    the target y -- so per-ticker scale spans an order of magnitude by
    volatility, and a mean over ~49 tickers is dominated by the handful of most
    volatile ones. Vision and language have a single unit ("full"), where median
    and mean are identical, so this changes nothing outside finance. The mean is
    still computed and reported, descriptively.
    """
    red = np.median if how == "median" else np.mean
    out = {}
    for seed in SEEDS:
        vals = [cells[f"{opt_name}|{unit}|{seed}"]["primary"]
                for unit in units if _cell_ok(cells, opt_name, unit, seed)]
        out[seed] = float(red(vals)) if vals else float("nan")
    return out


def aggregate_and_report(results):
    cells = results["cells"]
    units = results["meta"]["units"]
    lr_used = results["meta"]["lr_used"]
    knob_used = results["meta"].get("knob_used", {}) or {}

    log_sep("=")
    log(f"  RESULTS  --  {TASK_NAME}   primary = {PRIMARY_METRIC} "
        f"({'lower' if PRIMARY_LOWER_BETTER else 'higher'} is better)")
    if TASK.get("nonstandard_ppl_caveat"):
        log(f"  [CAVEAT] {TASK['nonstandard_ppl_caveat']}")
    log_sep("-")

    # ---- paired complete-case unit set (see _complete_units) ---------------
    units_c = _complete_units(cells, units)
    dropped = [u for u in units if u not in units_c]
    biased_fallback = False
    if not units_c:
        biased_fallback = True
        units_c = units
        log("  [WARNING] no unit survived the paired complete-case filter -- "
            "EVERY unit had at least one diverged/failed cell. Falling back to "
            "per-optimizer healthy cells, which is SURVIVORSHIP-BIASED in favour "
            "of whichever optimizer failed most. Do not report these numbers "
            "without saying so; go read the divergence reasons first.")
    elif dropped:
        log(f"  [complete-case] using {len(units_c)}/{len(units)} units; dropped "
            f"{len(dropped)} where some optimizer diverged or failed: "
            f"{dropped[:12]}{' ...' if len(dropped) > 12 else ''}")
        log("  (dropped for ALL optimizers, not just the failing one -- keeps the "
            "comparison paired; see CLAUDE.md sec 6)")

    task_number = {}
    task_number_mean = {}
    per_seed = {}
    for opt_name in ACTIVE_OPT_NAMES:
        by_seed = _cell_value_by_seed(cells, opt_name, units_c, how="median")
        per_seed[opt_name] = by_seed
        finite = [v for v in by_seed.values() if _is_finite(v)]
        task_number[opt_name] = float(np.median(finite)) if finite else float("nan")
        _mn = [v for v in _cell_value_by_seed(cells, opt_name, units_c,
                                              how="mean").values() if _is_finite(v)]
        task_number_mean[opt_name] = float(np.median(_mn)) if _mn else float("nan")
        n_div = sum(1 for u in units for s in SEEDS
                    if cells.get(f"{opt_name}|{u}|{s}", {}).get("diverged"))
        aux_bits = []
        for k in ("wall_s", "mean_step_size", "clip_freq", "peak_mem_mb"):
            vs = [cells[f"{opt_name}|{u}|{s}"].get(k) for u in units for s in SEEDS
                  if f"{opt_name}|{u}|{s}" in cells]
            vs = [v for v in vs if _is_finite(v)]
            if vs:
                aux_bits.append(f"{k}={np.mean(vs):.4g}")
        _kb = knob_used.get(opt_name)
        log(f"  {opt_name:<10}  lr={lr_used.get(opt_name, float('nan')):.2e}"
            + ("" if _kb is None else f"  knob={_kb:g}") + "  "
            f"{PRIMARY_METRIC}(task median over {len(SEEDS)} seeds)="
            f"{task_number[opt_name]:.6g}   "
            f"seed spread={_fmt_spread(finite)}   diverged_cells={n_div}/"
            f"{len(units) * len(SEEDS)}")
        if len(units) > 1:
            log(f"             descriptive: unit-MEAN variant="
                f"{task_number_mean[opt_name]:.6g}  (primary uses the unit-MEDIAN; "
                f"y is not target-standardised, so a mean over units is dominated "
                f"by the highest-variance ones)")
        log(f"             descriptive: " + "  ".join(aux_bits))

    finite_task = {k: v for k, v in task_number.items() if _is_finite(v)}
    if finite_task:
        best_opt = (min(finite_task, key=finite_task.get) if PRIMARY_LOWER_BETTER
                    else max(finite_task, key=finite_task.get))
        log_sep("-")
        log(f"  best optimizer on this task: {best_opt}  "
            f"({PRIMARY_METRIC}={finite_task[best_opt]:.6g})")

    # optional within-task paired test across units (finance panels: ~30 tickers)
    ref = "AdamW" if "AdamW" in ACTIVE_OPT_NAMES else ACTIVE_OPT_NAMES[0]
    if len(units_c) >= 6:
        log_sep("-")
        log(f"  within-task paired Wilcoxon across {len(units_c)} complete-case "
            f"units (seed-median per unit), reference = {ref}:")
        ref_by_unit = _unit_medians(cells, ref, units_c)
        for opt_name in ACTIVE_OPT_NAMES:
            if opt_name == ref:
                continue
            opt_by_unit = _unit_medians(cells, opt_name, units_c)
            pairs = [(a, b) for a, b in zip(opt_by_unit, ref_by_unit)
                     if _is_finite(a) and _is_finite(b)]
            if len(pairs) < 6:
                log(f"    {opt_name:<10} vs {ref}: too few paired units")
                continue
            a = np.array([p[0] for p in pairs]); b = np.array([p[1] for p in pairs])
            try:
                w_p = wilcoxon(a, b).pvalue
            except ValueError:
                w_p = float("nan")
            better = int(np.sum((a < b) if PRIMARY_LOWER_BETTER else (a > b)))
            med_delta = float(np.median(a - b))
            log(f"    {opt_name:<10} vs {ref}: median Δ={med_delta:+.4g}  "
                f"wins {better}/{len(pairs)}  wilcoxon p={w_p:.3g}  "
                f"(descriptive; the confirmatory test is the 12-task one)")

    _write_csv(cells, units)
    summary = {
        "task_name": TASK_NAME, "primary_metric": PRIMARY_METRIC,
        "primary_lower_better": PRIMARY_LOWER_BETTER,
        "opt_set": TASK.get("opt_set", "main"),
        "opt_names": list(ACTIVE_OPT_NAMES),
        "task_number_by_optimizer": task_number,
        "task_number_unit_mean_by_optimizer": task_number_mean,   # descriptive
        "per_seed_value_by_optimizer": per_seed,
        "lr_used": lr_used, "knob_used": knob_used, "units": units, "seeds": SEEDS,
        # complete-case bookkeeping -- report these in the paper, they are the
        # audit trail for how divergences were handled (CLAUDE.md sec 6)
        "unit_reduction": "median",
        "units_complete_case": units_c,
        "n_units_total": len(units),
        "n_units_complete_case": len(units_c),
        "units_dropped_incomplete": dropped,
        "complete_case_fallback_biased": biased_fallback,
        "nonstandard_ppl_caveat": TASK.get("nonstandard_ppl_caveat"),
        # deviations a diagnostic file may declare -- None on every protocol
        # task, so a reader can tell a pinned/limit-raised re-run apart from a
        # protocol number without reading the file's docstring
        "pinned_lr": TASK.get("pinned_lr"),
        "fixed_knob": TASK.get("fixed_knob"),
        "eviekf_maxf": TASK.get("eviekf_maxf"),
    }
    _write_json(os.path.join(TASK_DIR, "summary.json"), summary)
    log_sep("=")
    log(f"  wrote {os.path.join(TASK_DIR, 'summary.json')} "
        f"(consumed by benchmark12/aggregate.py for the 12-task test)")
    log_sep("=")
    return summary


def _unit_medians(cells, opt_name, units):
    out = []
    for unit in units:
        vs = [cells[f"{opt_name}|{unit}|{s}"]["primary"] for s in SEEDS
              if f"{opt_name}|{unit}|{s}" in cells
              and not cells[f"{opt_name}|{unit}|{s}"].get("diverged")
              and _is_finite(cells[f"{opt_name}|{unit}|{s}"].get("primary"))]
        out.append(float(np.median(vs)) if vs else float("nan"))
    return out


def _fmt_spread(vals):
    if not vals:
        return "n/a"
    return f"[{min(vals):.5g}, {max(vals):.5g}]"


def _write_csv(cells, units):
    import csv
    cols = ["opt", "unit", "seed", "lr_used", "knob_used", "primary",
            "best_val_primary",
            "diverged", "fail_reason", "steps_run", "wall_s", "sec_per_step",
            "mean_step_size",
            "clip_freq", "peak_mem_mb", "hess_norm", "hess_cv", "n_params",
            # descriptive-only diagnostics (no p-values -- CLAUDE.md sec 5)
            "steps_to_best_val", "steps_to_90pct_best",
            "frac_budget_to_90pct_best", "wallclock_to_90pct_best_s",
            "train_loss_at_25pct", "train_loss_at_50pct", "train_loss_at_75pct",
            "final_train_loss", "gen_gap_loss", "auc_val_curve",
            "update_weight_ratio", "update_cos_sim_sampled",
            "grad_norm_mean", "grad_norm_p95",
            "weight_norm_start", "weight_norm_end", "weight_norm_growth",
            "throughput_samples_per_s", "flops_per_step_est", "total_flops_est",
            "hess_trace", "hess_top_eig",
            # Evie-KF mechanism telemetry (descriptive -- claude_optim.md 4.3.4/5)
            "eviekf_cos_mean", "eviekf_cos_min", "eviekf_b_simple_mean",
            "eviekf_tr_sigma_mean", "eviekf_eff_rank_mean", "eviekf_gain_lo_min",
            "eviekf_gain_hi_max", "eviekf_n_active_steps"] \
        + list(AUX_METRIC_KEYS)
    with open(CSV_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for opt_name in ACTIVE_OPT_NAMES:
            for unit in units:
                for seed in SEEDS:
                    c = cells.get(f"{opt_name}|{unit}|{seed}")
                    if c:
                        w.writerow(c)
    log(f"  wrote {CSV_PATH}")


def main():
    log_sep("=")
    log(f"  {TASK_NAME}  |  domain={DOMAIN}  device={DEVICE}  torch={torch.__version__}")
    log(f"  primary={PRIMARY_METRIC}  seeds={SEEDS}  max_steps={MAX_STEPS}  "
        f"lr_search_steps={LR_SEARCH_STEPS}  batch={BATCH_SIZE}")
    if TASK.get("nonstandard_ppl_caveat"):
        log(f"  [CAVEAT] {TASK['nonstandard_ppl_caveat']}")
    log_sep("=")
    preflight()
    data = prepare_data()
    lr_used, knob_used = lr_search(data)
    results = run_sweep(data, lr_used, knob_used)
    aggregate_and_report(results)
    log(f"  DONE {datetime.now().isoformat()}")
    _log_fh.flush()


# === END ENGINE BLOCK ===
# (the `if __name__ == "__main__": main()` / `sweep_main()` entry point is
#  appended per-file by benchmark12/_build/build.py)
