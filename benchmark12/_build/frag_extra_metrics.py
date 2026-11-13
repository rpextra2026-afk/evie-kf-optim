# Gated master-rebuttal diagnostics. Helpers are inert unless extra_metrics is on.
EVIE_EXTRA_KEYS = ["last_c_mean", "last_c_min", "offdiag_frac_A", "offdiag_frac_B",
    "anisotropy_A", "anisotropy_B", "energy_top_before", "energy_top_after",
    "frac_params_kron", "frac_params_diag_fallback", "frac_params_vector", "eig_refresh_time_s"]
EXTRA_DOMAIN_KEYS = {
    "vision": ["cls_acc_min","cls_acc_std","cls_acc_worst5_mean","brier_score",
        "predictive_entropy_nats","mean_confidence","acc_noise_a","acc_noise_b",
        "acc_shift","acc_lowres","acc_bright","acc_fgsm_small","corruption_mean_drop",
        "train_acc_subsample","gen_gap_acc"],
    "language": ["nll_pos_0_32","nll_pos_32_128","nll_pos_128_end"]+[f"nll_freq_q{i}" for i in range(4)],
    "finance": ["ic_spearman","pred_std_ratio","mse_vs_zero","beats_zero","bias",
        "mse_tail","dir_acc_big","sign_turnover","sign_sharpe_net","max_drawdown_sign","pred_autocorr1"]}


def _tensor_bytes(value):
    if torch.is_tensor(value):
        return value.numel()*value.element_size()
    if isinstance(value,dict):
        return sum(_tensor_bytes(v) for v in value.values())
    if isinstance(value,(tuple,list)):
        return sum(_tensor_bytes(v) for v in value)
    return 0


def _safe_extra(name, fn, errors):
    try:
        value = fn()
        if isinstance(value, (float,np.floating)) and not math.isfinite(value):
            raise ValueError("undefined/nonfinite value")
        return value
    except Exception as exc:
        if not isinstance(exc,TimeoutError) and classify_exception(exc) == "env":
            raise
        errors[name] = repr(exc)
        return None


def _sync_time():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.perf_counter()


def landscape_extras(model, loader, criterion, params, initial, roles, errors, deadline=None):
    """Validation-only, fixed local RNG; a signed Rayleigh estimate from power
    iteration may converge to the largest-magnitude Hessian eigenpair.
    Runtime budget is checked between operations; a single kernel is not preempted.
    """
    import contextlib
    model.eval()
    started = time.monotonic()
    budget = 5. if DOMAIN == "finance" else 90.
    result = dict.fromkeys(["x_role_dist_from_init","x_role_weight_norm","x_dist_from_init"])
    # Local generators never alter training or checkpoint RNG streams.
    gen = torch.Generator(device=DEVICE).manual_seed(0)
    def within():
        if time.monotonic()-started > budget or (deadline is not None and time.monotonic()>deadline):
            raise TimeoutError(f"diagnostic budget {budget:g}s exhausted; used deterministic first validation batches")
    def role_dist():
        within()
        out = {}
        for index,p in enumerate(params):
            role = roles.get(p,"other")
            v=out.setdefault(role,[0.,0.])
            v[1]+=float(p.detach().double().square().sum())
            if initial is not None:
                v[0]+=float((p.detach().cpu().double()-initial[index].double()).square().sum())
        result["x_role_weight_norm"]={r:math.sqrt(v[1]) for r,v in out.items()}
        result["x_role_dist_from_init"]={r:math.sqrt(v[0]/max(1e-30,sum(float(initial[i].double().square().sum()) for i,p in enumerate(params) if roles.get(p,"other")==r))) for r,v in out.items()} if initial is not None else None
        result["x_dist_from_init"] = math.sqrt(sum(v[0] for v in out.values())/max(1e-30,sum(float(t.double().square().sum()) for t in initial))) if initial is not None else None
    _safe_extra("x_role_dist_from_init",role_dist,errors)
    if initial is None:
        errors["x_dist_from_init"]="resumed cell: initial weights unavailable"
        errors["x_role_dist_from_init"]="resumed cell: initial weights unavailable"
    def rank_stats():
        ranks=[]; byrole={}
        with torch.no_grad():
            for p in params:
                within()
                if p.ndim!=2 or max(p.shape)>4096: continue
                v=torch.randn(p.size(1),device=p.device,dtype=p.dtype,generator=gen)
                for _ in range(10):
                    v=p.T@(p@v);v=v/v.norm().clamp_min(1e-30)
                sigma=(p@v).norm()
                value=float(p.square().sum()/sigma.square().clamp_min(1e-30))
                ranks.append(value);byrole.setdefault(roles.get(p,"other"),[]).append(value)
        return ranks,byrole
    rank=_safe_extra("x_stable_rank_mean",rank_stats,errors)
    result.update(x_stable_rank_mean=float(np.mean(rank[0])) if rank and rank[0] else None,
                  x_stable_rank_min=min(rank[0]) if rank and rank[0] else None,
                  x_stable_rank_by_role={r:float(np.mean(v)) for r,v in rank[1].items()} if rank else None)
    def kurtosis():
        within()
        # Sample uniformly from the pooled vector without copying all weights.
        total=sum(p.numel() for p in params)
        if total<=1000000:
            values=torch.cat([p.detach().reshape(-1).cpu().double() for p in params])
        else:
            rng=np.random.default_rng(0)
            indices=np.sort(rng.choice(total,1000000,replace=False))
            parts=[];offset=0
            for p in params:
                idx=indices[(indices>=offset)&(indices<offset+p.numel())]-offset
                if len(idx):parts.append(p.detach().reshape(-1)[torch.as_tensor(idx,device=p.device)].cpu().double())
                offset+=p.numel()
            values=torch.cat(parts)
        centered=values-values.mean();v=centered.square().mean()
        return float(centered.pow(4).mean()/v.square()-3) if v>0 else float("nan")
    result["x_weight_kurtosis"]=_safe_extra("x_weight_kurtosis",kurtosis,errors)
    # Keep CPU/GPU loader seeding isolated, including DataLoader base seeds.
    rng_state_before=rng_state()
    try:
        batches=[]
        for batch in itertools.islice(loader,2 if DOMAIN=="finance" else 4):
            within();batches.append(batch_to_device(batch))
        if not batches:raise ValueError("empty validation loader")
    except Exception as exc:
        if not isinstance(exc,TimeoutError) and classify_exception(exc)=="env":raise
        errors["landscape_batches"]=repr(exc);batches=[]
    finally:
        load_rng_state(rng_state_before)
    result["x_landscape_batches"]=len(batches)
    result["x_landscape_examples"]=sum(int(b[0].size(0)) for b in batches)
    def loss():
        within()
        if not batches:raise ValueError("no diagnostic validation batches")
        return sum(forward_loss(model,b,criterion) for b in batches)/len(batches)
    try:
        from torch.nn.attention import sdpa_kernel, SDPBackend
        ctx=sdpa_kernel(SDPBackend.MATH)
    except ImportError:
        ctx=contextlib.nullcontext()
    with ctx:
        def hessian_metrics():
            with torch.enable_grad():
                L=loss()
                grads=torch.autograd.grad(L,params,create_graph=True,allow_unused=True)
                def hvp(v):
                    within()
                    dot=sum((g*x).sum() for g,x in zip(grads,v) if g is not None)
                    h=torch.autograd.grad(dot,params,retain_graph=True,allow_unused=True)
                    return [torch.zeros_like(p) if t is None else t.detach() for p,t in zip(params,h)]
                v=[torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=gen) for p in params]
                for _ in range(20):
                    norm=sum(float(t.square().sum()) for t in v)**.5
                    v=[t/max(norm,1e-30) for t in v];h=hvp(v)
                    eigen=sum(float((a*b).sum()) for a,b in zip(v,h));v=h
                trace=[]
                for _ in range(8):
                    v=[torch.randint(0,2,p.shape,device=p.device,generator=gen).to(p.dtype)*2-1 for p in params]
                    h=hvp(v);trace.append(sum(float((a*b).sum()) for a,b in zip(v,h)))
            return eigen,float(np.mean(trace))
        hess=_safe_extra("x_sharp_top_eig",hessian_metrics,errors)
        result["x_sharp_top_eig"]=hess[0] if hess else None
        result["x_hess_trace_hutch"]=hess[1] if hess else None
        if hess is None:errors.setdefault("x_hess_trace_hutch",errors.get("x_sharp_top_eig","unavailable"))
        if deadline is not None and time.monotonic()>deadline:
            for name in ["x_flat_rho001","x_flat_rho003","x_flat_rho010"]:
                result[name]=None;errors[name]="end-of-run diagnostic budget exhausted"
            for name,value in result.items():
                if value is None:errors.setdefault(name,"unavailable for this cell")
            return result
        snapshot={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        # Exclude normalization parameters by module membership, including names
        # like downsample.1 that do not spell out 'bn'.
        excluded={id(p) for mod in model.modules() if isinstance(mod,(nn.LayerNorm,nn.modules.batchnorm._BatchNorm)) for p in mod.parameters(recurse=False)}
        def flatness(rho):
            base=float(loss().detach())
            values=[]
            try:
                for _ in range(5):
                    within();model.load_state_dict(snapshot)
                    with torch.no_grad():
                        for p in params:
                            if id(p) in excluded:continue
                            d=torch.randn(p.shape,device=p.device,dtype=p.dtype,generator=gen)
                            p.add_(d,alpha=rho*float(p.norm())/max(float(d.norm()),1e-30))
                    values.append(float(loss().detach())-base)
                return float(np.mean(values))
            finally:
                model.load_state_dict(snapshot)
                key=next(iter(snapshot))
                if not torch.equal(model.state_dict()[key].detach().cpu(),snapshot[key]):
                    raise RuntimeError("flatness probe did not restore weights exactly")
        for name,rho in [("x_flat_rho001",.01),("x_flat_rho003",.03),("x_flat_rho010",.1)]:
            result[name]=_safe_extra(name,lambda r=rho:flatness(r),errors)
    for name,value in result.items():
        if value is None:errors.setdefault(name,"unavailable for this cell")
    return result


def _extra_train_accuracy(model,data,criterion,result,deadline=None):
    def measure():
        dataset=_VisionDataset(*data["train"],data["mean"],data["std"],train=False,hflip=False)
        selected=torch.utils.data.Subset(dataset,range(min(10000,len(dataset))))
        loader=torch.utils.data.DataLoader(selected,batch_size=BATCH_SIZE,
            generator=torch.Generator().manual_seed(0))
        hit=total=0
        for batch in loader:
            if deadline is not None and time.monotonic()>deadline:
                raise TimeoutError("end-of-run diagnostic budget exhausted")
            x,y=batch_to_device(batch)
            with torch.no_grad():hit+=int((model(x).argmax(1)==y).sum())
            total+=len(y)
        return 100*hit/max(total,1)
    result["train_acc_subsample"]=_safe_extra("train_acc_subsample",measure,result["x_errors"])
    result["train_acc_augmented"]=False
    result["gen_gap_acc"]=(result["train_acc_subsample"]-result["primary"]
                           if result["train_acc_subsample"] is not None else None)


def _speedtest(data):
    steps=int(os.environ.get("BENCH12_SPEEDTEST_STEPS","200"))
    if steps<=20:fail("BENCH12_SPEEDTEST_STEPS must exceed 20 warm-up steps")
    output={};lrs={}
    unit=lr_search_units(data)[0]
    for arm in ACTIVE_OPT_NAMES:
        pin=(TASK.get("pinned_lr") or {}).get(arm,{})
        grid=(TASK.get("search_grid") or {}).get(arm,{})
        lr=pin.get("lr",(grid.get("lr") or [LR_GRID_CENTRE[arm]])[0] if grid else LR_GRID_CENTRE[arm])
        knob=pin.get("knob",(TASK.get("fixed_knob") or {}).get(arm,(grid.get("knob") or KNOB_GRID.get(arm) or [None])[0]))
        lrs[arm]=lr
        if torch.cuda.is_available():torch.cuda.reset_peak_memory_stats()
        r=train_run(arm,data,unit,lr,_adamw_lr(lrs),steps,42,None,is_lr_search=True,knob=knob)
        output[arm]=dict(sec_per_step=r["speed_sec_per_step"],peak_mem_mb=r["peak_mem_mb"],
            steps=r["steps_run"],tf32=bool(torch.backends.cuda.matmul.allow_tf32),
            device=torch.cuda.get_device_name() if torch.cuda.is_available() else "CPU",
            primary_after=(r["val_curve"][-1][1] if r.get("val_curve") else r["best_val_primary"]),warmup_steps_excluded=20)
    _write_json(os.path.join(TASK_DIR,"speedtest.json"),output)
