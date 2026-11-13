#!/usr/bin/env python3
"""R6 confirmatory primary-metric sensitivity and labelled exploratory mining.
Numpy + stdlib. Never imports torch or changes archived result files.
"""
import argparse
import csv
import itertools
import json
import math
from pathlib import Path
import random
import statistics
import numpy as np

BENCH=Path(__file__).resolve().parents[1]
ARCHIVE=BENCH/'results'
OUT=ARCHIVE/'rebuttal'
MAIN=sorted(p.name for p in ARCHIVE.iterdir() if p.is_dir() and p.name.startswith(('vision_','language_','finance_')) and (p/'results.json').exists())
LOWER='test_loss gen_gap_loss gen_gap_acc ece ece_15 mce_15 brier_score nll_std nll_p90 nll_p95 nll_p99 bits_per_token bits_per_byte bpc mean_target_rank nll_pos_0_32 nll_pos_32_128 nll_pos_128_end nll_freq_q0 nll_freq_q1 nll_freq_q2 nll_freq_q3 rmse mae mse_vs_zero mse_tail bias_abs sign_turnover max_drawdown_sign cls_acc_std corruption_mean_drop clip_freq grad_norm_p95 x_grad_norm_max x_grad_norm_cv x_loss_spike_count x_loss_spike_max_ratio hess_top_eig x_sharp_top_eig x_hess_trace_hutch x_flat_rho001 x_flat_rho003 x_flat_rho010 sec_per_step wall_s peak_mem_mb x_opt_state_mb x_opt_time_s x_opt_time_frac steps_to_best_val wallclock_to_90pct_best_s frac_budget_to_90pct_best'.split()
HIGHER='top5 token_acc token_top5_acc token_top10_acc mean_reciprocal_rank dir_acc dir_acc_big r2 sign_sharpe sign_sharpe_net ic_spearman beats_zero cls_acc_min cls_acc_worst5_mean acc_noise_a acc_noise_b acc_shift acc_lowres acc_bright acc_fgsm_small throughput_samples_per_s tokens_per_s x_stable_rank_mean'.split()
DIRECTION={**dict.fromkeys(LOWER,'lower'),**dict.fromkeys(HIGHER,'higher')}
class Cells(dict):
    """Cell map carrying provenance without changing tuple-key iteration."""
    pass


def finite(x):return isinstance(x,(int,float)) and not isinstance(x,bool) and math.isfinite(x)
def usable(c):return bool(c) and not c.get('diverged') and finite(c.get('primary'))
def higher(task):return task.startswith('vision_') or task.startswith('ablation_vision_')
def gain(new,old,task):return (new-old)/old if higher(task) else (old-new)/old


def ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);out=[0.]*len(values);i=0
    while i<len(order):
        j=i+1
        while j<len(order) and values[order[j]]==values[order[i]]:j+=1
        for k in range(i,j):out[order[k]]=(i+j+1)/2
        i=j
    return out


def wilcoxon(differences):
    values=[float(x) for x in differences if finite(x) and x!=0]
    if not values:return 1.
    if len(values)>20:raise ValueError('Exact Wilcoxon limited to 20 independent task effects')
    rr=ranks([abs(x) for x in values]);observed=abs(sum(r if d>0 else -r for d,r in zip(values,rr)))
    n=sum(abs(sum(s*r for s,r in zip(signs,rr)))>=observed-1e-12 for signs in itertools.product([-1,1],repeat=len(rr)))
    return n/2**len(rr)


def sign_test(wins,losses):
    n=wins+losses
    return min(1.,2*sum(math.comb(n,k) for k in range(min(wins,losses)+1))/2**n) if n else 1.


def holm(pvalues):
    ordered=sorted(pvalues,key=pvalues.get);out={};last=0.;m=len(ordered)
    for i,k in enumerate(ordered):
        last=max(last,min(1.,(m-i)*pvalues[k]));out[k]=dict(p_raw=pvalues[k],p_adjusted=last,reject=last<=.05)
    return out


def bootstrap(values):
    values=[float(v) for v in values if finite(v)]
    if len(values)<2:return [None,None]
    rng=random.Random(0);n=len(values)
    samples=[sum(values[rng.randrange(n)] for _ in range(n))/n for _ in range(10000)]
    return np.quantile(samples,[.025,.975]).tolist()


def random_effects(effects,ses):
    pairs=[(e,se) for e,se in zip(effects,ses) if finite(e) and finite(se) and se>=0]
    if len(pairs)<2:return dict(effect=None,ci=[None,None],tau2=None,n=len(pairs))
    e=np.array([p[0] for p in pairs]);v=np.maximum(np.array([p[1] for p in pairs])**2,1e-12)
    w=1/v;fixed=float(np.dot(w,e)/w.sum());Q=float(np.dot(w,(e-fixed)**2))
    C=float(w.sum()-(w*w).sum()/w.sum());tau=max(0.,(Q-(len(e)-1))/C)
    wr=1/(v+tau);effect=float(np.dot(wr,e)/wr.sum());se=float(1/math.sqrt(wr.sum()))
    return dict(effect=effect,ci=[effect-1.96*se,effect+1.96*se],tau2=tau,n=len(pairs),note='DL normal interval; seed-spread SE, exploratory heterogeneity summary')


def load(task_dir):
    path=Path(task_dir)
    if path.is_file():path=path.parent
    d=json.loads((path/'results.json').read_text());cells=Cells()
    for key,cell in d.get('cells',{}).items():
        a,u,s=key.split('|');cells[(a,u,int(s))]=cell
    cells.meta=d.get('meta',{})
    return cells


def compatible(a,b):
    if hasattr(a,'meta') and hasattr(b,'meta') and bool(a.meta.get('smoke'))!=bool(b.meta.get('smoke')):
        raise ValueError('Refusing a comparison between SMOKE correctness data and production results')


def task_name(path):
    d=json.loads((Path(path)/'results.json').read_text());return d.get('meta',{}).get('task_name',Path(path).name)


def aggregate(cells,arm,metric='primary',exclude42=False):
    # Per-ticker seed median then ticker median; also return per-seed panel
    # medians for explicit paired-seed uncertainty summaries.
    units={u for a,u,s in cells if a==arm};byunit=[];byseed={}
    for unit in units:
        v=[]
        for (a,u,s),c in cells.items():
            value=abs(c.get('bias')) if metric=='bias_abs' and finite(c.get('bias')) else c.get(metric)
            if a==arm and u==unit and (not exclude42 or s!=42) and usable(c) and finite(value):
                v.append(value);byseed.setdefault(s,[]).append(value)
        if v:byunit.append(float(np.median(v)))
    return (float(np.median(byunit)) if byunit else None),{s:float(np.median(v)) for s,v in byseed.items()}


def paired(newcells,oldcells,new,old,metric='primary',exclude42=False):
    pairs=[]
    for (arm,unit,seed),c in newcells.items():
        if arm!=new or (exclude42 and seed==42):continue
        d=oldcells.get((old,unit,seed))
        x=abs(c['bias']) if metric=='bias_abs' and finite(c.get('bias')) else c.get(metric)
        y=abs(d['bias']) if d and metric=='bias_abs' and finite(d.get('bias')) else d.get(metric) if d else None
        if usable(c) and usable(d) and finite(x) and finite(y):pairs.append((unit,seed,x,y))
    return pairs


def paired_points(pairs):
    left={};right={}
    for unit,seed,a,b in pairs:
        left.setdefault(unit,[]).append(a);right.setdefault(unit,[]).append(b)
    return (float(np.median([np.median(v) for v in left.values()])) if left else None,
            float(np.median([np.median(v) for v in right.values()])) if right else None)


def comparison(task,nc,oc,new,old,exclude=False):
    compatible(nc,oc)
    pairs=paired(nc,oc,new,old,exclude42=exclude)
    rel=[gain(a,b,task) for u,s,a,b in pairs if b!=0]
    # Task effect respects per-ticker seed medians. Bootstrap sensitivity uses
    # ticker x seed pairs for finance as requested, with dependence caveat.
    valid={(u,s) for u,s,a,b in pairs};n={k:c for k,c in nc.items() if (k[1],k[2]) in valid};o={k:c for k,c in oc.items() if (k[1],k[2]) in valid}
    a,sa=aggregate(n,new,exclude42=exclude);b,sb=aggregate(o,old,exclude42=exclude)
    seedrel=[gain(sa[s],sb[s],task) for s in sorted(set(sa)&set(sb)) if sb[s]!=0]
    return dict(task=task,new=new,ref=old,smoke=bool(getattr(nc,'meta',{}).get('smoke')),subset='exclude42' if exclude else 'all',new_median=a,ref_median=b,
                rel_gain=gain(a,b,task) if a is not None and b not in (None,0) else None,
                ci=bootstrap(rel),ci_statistic='mean paired relative gain (task point estimate uses nested medians)',n_pairs=len(pairs),n_seeds=len(seedrel),
                se=statistics.stdev(seedrel)/math.sqrt(len(seedrel)) if len(seedrel)>1 else None,
                new_per_seed=sa,ref_per_seed=sb,n_new_failed=sum(bool(v.get('diverged')) for (a,u,s),v in nc.items() if a==new and (not exclude or s!=42)),n_ref_failed=sum(bool(v.get('diverged')) for (a,u,s),v in oc.items() if a==old and (not exclude or s!=42)),note='paired observations; finance bootstrap resamples ticker x seed pairs, not independent market episodes')


def primary_summary(rows):
    eff=[r['rel_gain'] for r in rows if finite(r.get('rel_gain'))]
    return dict(tasks=len(eff),wins=sum(e>0 for e in eff),ties=sum(e==0 for e in eff),losses=sum(e<0 for e in eff),
                wilcoxon_p=wilcoxon(eff),sign_p=sign_test(sum(e>0 for e in eff),sum(e<0 for e in eff)),
                random_effects=random_effects([r.get('rel_gain') for r in rows],[r.get('se') for r in rows]))


def find(args,name):
    for root in args.roots:
        p=Path(root)/name
        if (p/'results.json').exists():return p
    return None


def discover(args):
    paths={}
    for root in reversed(args.roots):
        root=Path(root)
        if (root/'results.json').exists():paths[task_name(root)]=root
        elif root.exists():
            for p in sorted(root.rglob('results.json')):
                if 'final_cells' in p.parts or 'rebuttal' in p.parts:continue
                try:paths[task_name(p.parent)]=p.parent
                except (ValueError,KeyError):continue
    return paths


def primary_commands(args):
    rows=[];summary={};header='PRIMARY METRIC ONLY; sensitivity analysis, not a new preregistration'
    if args.command in ('seed42','r3a'):
        for name in MAIN:
            old=load(ARCHIVE/name)
            if args.command=='seed42':
                for exclude in [False,True]:rows.append(comparison(name,old,old,args.new,args.old,exclude))
            else:
                path=find(args,name+'_evie_m30')
                if path:rows.append(comparison(name,load(path),old,'EvieKFm','AdamW',True))
        subsets=sorted({r['subset'] for r in rows})
        summary={s:primary_summary([r for r in rows if r['subset']==s]) for s in subsets}
        if args.command=='r3a':summary['holm']=holm({s:d['wilcoxon_p'] for s,d in summary.items()})
    elif args.command=='finance_filter':
        for name in [n for n in MAIN if n.startswith('finance_')]:
            c=load(ARCHIVE/name);arms={a for a,u,s in c};seeds={s for a,u,s in c};units={u for a,u,s in c}
            for rule in ['all_arms','pair_only','all_with_failures']:
                selected=[u for u in units if rule=='all_with_failures' or all(usable(c.get((a,u,s))) for a in (arms if rule=='all_arms' else [args.new,args.old]) for s in seeds)]
                wins=losses=ties=0
                for u in selected:
                    dif=[]
                    for s in seeds:
                        a,b=c.get((args.new,u,s)),c.get((args.old,u,s))
                        if usable(a) and usable(b):dif.append(float(np.sign(b['primary']-a['primary'])))
                        elif usable(a):dif.append(1.)
                        elif usable(b):dif.append(-1.)
                        else:dif.append(0.)
                    d=float(np.median(dif));wins+=d>0;losses+=d<0;ties+=d==0
                row=dict(task=name,rule=rule,n=len(selected),total=len(units),wins=wins,losses=losses,ties=ties,sign_p=sign_test(wins,losses))
                if rule!='all_with_failures':
                    subset={k:v for k,v in c.items() if k[1] in selected}
                    row.update(comparison(name,subset,subset,args.new,args.old))
                rows.append(row)
    elif args.command=='budget':
        header='search scores, validation metric, seed 42; shows sensitivity to tuning budget, not a test result'
        for name in MAIN:
            path=ARCHIVE/name/'lr_search.json'
            if not path.exists():continue
            entries=json.loads(path.read_text());expected={}
            for arm,e in entries.items():
                raw=list(e.get('grid',{}).values())
                scores=[v if finite(v) else (-math.inf if higher(name) else math.inf) for v in raw]
                if not scores:continue
                rng=random.Random(1);k=min(7,len(scores));select=max if higher(name) else min
                bests=[select(rng.sample(scores,k)) for _ in range(2000)]
                expected[arm]=float(np.mean(bests))
                rows.append(dict(task=name,arm=arm,n_trials=len(e.get('grid',{})),n_finite=sum(finite(v) for v in raw),k=k,full_best=select(scores),expected_best_k7=expected[arm]))
            for r in rows:
                if r['task']==name:r['difference_to_AdamW_k7']=r['expected_best_k7']-expected['AdamW'] if 'AdamW' in expected else None
    elif args.command=='holm_limit':
        for k in range(1,13):
            dif=[-i for i in range(1,13-k)]+list(range(13-k,13))
            assert len(dif)==12
            rows.append(dict(n=12,wins=k,min_wilcoxon_p=wilcoxon(dif),holm_threshold=.05/args.m,note='constrained scenario: wins hold largest ranks; two-sided exact p, not a guarantee from win count alone'))
    elif args.command=='compare':
        name=task_name(args.task);nc=load(args.task);oc=load(args.ref)
        for arm in args.arms.split(','):
            r=comparison(name,nc,oc,arm,args.old,False)
            units={u for a,u,s in nc if a==arm}
            vals=[float(np.median([c['primary'] for (a,u,s),c in nc.items() if a==arm and u==unit and usable(c)])) for unit in units if any(a==arm and u==unit and usable(c) for (a,u,s),c in nc.items())] if name.startswith('finance_') else [c['primary'] for (a,u,s),c in nc.items() if a==arm and usable(c)]
            r['mean']=float(np.mean(vals)) if vals else None;rows.append(r)
    elif args.command=='r4':
        name='ablation_'+args.host;old=load(ARCHIVE/'ablation'/name);path=find(args,name+'_controls')
        if path:
            nc=load(path)
            compatible(nc,old)
            for arm in ['EvieKF','EvieDiag']:
                value,seeds=aggregate(old,arm,exclude42=True);rows.append(dict(task=name,arm=arm,subset='exclude42',median=value,per_seed=seeds))
            if args.include_adamw:
                maincells=load(ARCHIVE/args.host)
                value,seeds=aggregate(maincells,'AdamW',exclude42=False)
                rows.append(dict(task=args.host,arm='AdamW',subset='main_run_only',median=value,
                    per_seed={str(s):seeds.get(s,'unavailable') for s in range(42,52)},
                    note='Optional main-run reference: seeds 42..46 only; never a ten-seed ablation reference'))
            for arm in ['EvieKFlag','EvieKFrb']:
                pairs=paired(old,nc,'EvieKF',arm,exclude42=True)
                rows.append(dict(task=name,arm=arm,subset='exclude42',median=aggregate(nc,arm,exclude42=True)[0],EvieKF_minus_control=float(np.mean([a-b for u,s,a,b in pairs])) if pairs else None,ci=bootstrap([a-b for u,s,a,b in pairs]),n=len(pairs)))
    return dict(header=header,rows=rows,summary=summary)


def interpolate(curve,x):
    c=sorted((s,v) for s,v in curve if finite(s) and finite(v))
    if not c or x<c[0][0] or x>c[-1][0]:return None
    return float(np.interp(x,[s for s,v in c],[v for s,v in c]))


def efficiency_pair(new,ref,is_higher=False):
    target=ref.get('best_val_primary');curve=new.get('val_curve',[])
    if not finite(target):return None
    hits=[(s,v) for s,v in curve if finite(v) and (v>=target if is_higher else v<=target)]
    step=hits[0][0] if hits else None;exact=dict(new.get('eval_wall_curve',[]));per=new.get('sec_per_step')
    wall=exact.get(step) if step is not None else None;approx=wall is None
    if wall is None and step is not None and finite(per):wall=step*per
    return dict(reached=step is not None,steps_to_target=step,wall_to_target=wall,approx=approx,
        step_ratio=step/ref['steps_to_best_val'] if step is not None and finite(ref.get('steps_to_best_val')) and ref['steps_to_best_val']>0 else None,
        wall_ratio=wall/ref['wall_s'] if wall is not None and finite(ref.get('wall_s')) and ref['wall_s']>0 else None,
        val_at_ref_steps=interpolate(curve,ref.get('steps_run',0)),
        val_at_ref_time=interpolate([(s*per,v) for s,v in curve],ref['wall_s']) if finite(per) and finite(ref.get('wall_s')) else None)


def profile_points(errors,taus=None):
    taus=np.arange(1.,1.501,.01) if taus is None else np.asarray(taus)
    arms=sorted(set().union(*(r.keys() for r in errors)));ratios={a:[] for a in arms};ranked={a:[] for a in arms}
    for row in errors:
        if not all(a in row and (finite(row[a]) or row[a]==math.inf) and row[a]>=0 for a in arms):continue
        if not any(finite(row[a]) for a in arms):continue
        best=min(row.values());rr=ranks([row[a] for a in arms])
        for a,r in zip(arms,rr):
            ratio=(1. if row[a]==best else math.inf) if best==0 else row[a]/best
            ratios[a].append(ratio);ranked[a].append(r)
    out=[]
    for a in arms:
        values=ratios[a];curve=[float(np.mean(np.array(values)<=t)) if values else None for t in taus]
        area=float(np.trapezoid(curve,taus)) if values else None
        out.append(dict(arm=a,n_tasks=len(values),infinite_ratio_tasks=sum(not math.isfinite(v) for v in values),taus=taus.tolist(),rho=curve,area=area,
                        average_rank=float(np.mean(ranked[a])) if values else None,
                        geometric_mean_ratio=float(np.exp(np.mean(np.log(values)))) if values else None))
    return out


def panel_cells(cells,arm):
    """One descriptive median panel trajectory per seed, with coverage counts."""
    out={}
    for seed in sorted({s for a,u,s in cells if a==arm}):
        allrows=[c for (a,u,s),c in cells.items() if a==arm and s==seed]
        rows=[c for c in allrows if usable(c)]
        if not rows:continue
        result={}
        for key in ['best_val_primary','steps_to_best_val','steps_run','sec_per_step','wall_s']:
            vals=[c[key] for c in rows if finite(c.get(key))]
            result[key]=float(np.median(vals)) if vals else None
        for key in ['val_curve','eval_wall_curve']:
            points={}
            for c in rows:
                for step,value in c.get(key,[]):
                    if finite(value):points.setdefault(step,[]).append(value)
            result[key]=[[step,float(np.median(v))] for step,v in sorted(points.items())]
        result['healthy_units']=len(rows);result['expected_units']=len(allrows)
        out[seed]=result
    return out


def exploratory(args,command=None):
    cmd=command or args.command;rows=[];summary={};metrics=set();paths=discover(args)
    modes={bool(json.loads((p/'results.json').read_text()).get('meta',{}).get('smoke')) for p in paths.values()}
    if len(modes)>1:raise ValueError('Refusing to pool SMOKE correctness data and production results')
    header=('SMOKE correctness data; ' if modes=={True} else '')+'EXPLORATORY: no p-values; old-cell wall time estimates are approximate'
    if cmd=='landscape':header+='; flatter is not automatically better generalisation; read with test metrics'
    if cmd=='mechanism':header+='; n = number of tasks, descriptive, a hypothesis generator not a test'
    profile_errors=[]
    for name,path in paths.items():
        c=load(path);arms=args.arms.split(',') if args.arms else ([args.new,args.ref_arm] if cmd=='profiles' else sorted({a for a,u,s in c}));ref=args.ref_arm;new=args.new
        if cmd=='efficiency':
            present={a for a,u,s in c}
            if new not in present or ref not in present:continue
            matched={(u,s) for (a,u,s),cell in c.items() if a==new and usable(cell) and usable(c.get((ref,u,s)))}
            pc={key:value for key,value in c.items() if (key[1],key[2]) in matched}
            nc,rc=panel_cells(pc,new),panel_cells(pc,ref)
            for seed in sorted({s for a,u,s in c if a in (new,ref) and s!=42}):
                expected=len({u for a,u,s in c if a in (new,ref) and s==seed})
                if seed not in nc or seed not in rc:
                    rows.append(dict(task=name,seed=seed,reached=False,steps_to_target=None,wall_to_target=None,step_ratio=None,wall_ratio=None,approx=True,expected_units=expected,note="no healthy paired cells; counted as not reaching target"))
                    continue
                r=efficiency_pair(nc[seed],rc[seed],higher(name))
                if r:rows.append(dict(task=name,seed=seed,new_healthy_units=nc[seed]['healthy_units'],ref_healthy_units=rc[seed]['healthy_units'],expected_units=expected,**r))
            metrics.update(['steps_to_target','wall_to_target','val_at_ref_steps','val_at_ref_time'])
        elif cmd=='pareto':
            points=[]
            b=aggregate(c,ref)[0]
            for a in arms:
                primary=aggregate(c,a)[0];wall=aggregate(c,a,'wall_s')[0];mem=aggregate(c,a,'peak_mem_mb')[0]
                if primary is not None:points.append(dict(task=name,arm=a,primary=primary,error=-primary if higher(name) else primary,wall_s=wall,peak_mem_mb=mem,rel_gain=gain(primary,b,name) if b not in (None,0) else None))
            for r in points:
                for cost in ['wall_s','peak_mem_mb']:
                    r['frontier_'+cost]=(not any(q['error']<=r['error'] and q[cost]<=r[cost] and (q['error']<r['error'] or q[cost]<r[cost]) for q in points if finite(q.get(cost)))) if finite(r.get(cost)) else None
                rows.append(r)
            metrics.update(['primary','wall_s','peak_mem_mb'])
        elif cmd=='profiles':
            row={}
            if not all(a in {key[0] for key in c} for a in arms):continue
            for a in arms:
                value=aggregate(c,a)[0]
                row[a]=(max(0.,100-value) if higher(name) else value) if value is not None else math.inf
            if row:profile_errors.append(row)
            metrics.add('primary')
        elif cmd=='lr_robustness':
            lp=path/'lr_search.json'
            if lp.exists():
                for a,e in json.loads(lp.read_text()).items():
                    grid=e.get('grid',{});v=[(k,x) for k,x in grid.items() if finite(x)]
                    if not v:continue
                    best=(max if higher(name) else min)(v,key=lambda t:t[1]);tol=.05*abs(best[1]);good=[(k,x) for k,x in v if abs(x-best[1])<=tol]
                    knob=best[0].split('|')[-1];lrs=[float(k.split('|')[0]) for k,x in good if k.split('|')[-1]==knob]
                    rows.append(dict(task=name,arm=a,n_within_5pct=len(good),lr_decades_within_5pct=math.log10(max(lrs)/min(lrs)) if lrs else None,fail_frac=1-len(v)/len(grid)))
            metrics.update(['n_within_5pct','lr_decades_within_5pct','fail_frac'])
        elif cmd=='seed_variance':
            stats={}
            for a in arms:
                values=list(aggregate(c,a,exclude42=True)[1].values())
                stats[a]=dict(std=statistics.stdev(values) if len(values)>1 else None,iqr=float(np.subtract(*np.percentile(values,[75,25]))) if values else None,range=max(values)-min(values) if values else None,cv=statistics.stdev(values)/abs(statistics.mean(values)) if len(values)>1 and statistics.mean(values)!=0 else None,n=len(values))
            for a,r in stats.items():rows.append(dict(task=name,arm=a,**r,ratio_std_to_ref=r['std']/stats[ref]['std'] if ref in stats and r['std'] is not None and stats[ref]['std'] not in (None,0) else None))
            metrics.update(['std','iqr','range','cv'])
        elif cmd=='stability':
            for a in arms:
                vals=[v for (arm,u,s),v in c.items() if arm==a]
                row=dict(task=name,arm=a,fail_rate=sum(bool(v.get('diverged')) for v in vals)/len(vals) if vals else None)
                for k in ['clip_freq','grad_norm_p95','x_grad_norm_max','x_loss_spike_count']:
                    row[k]=aggregate(c,a,k)[0];metrics.add(k)
                rows.append(row)
            metrics.add('fail_rate')
        elif cmd in ('domain','landscape'):
            if cmd=='domain' and not name.startswith(args.domain+'_'):continue
            keys=set(DIRECTION)|{'predictive_entropy_nats','mean_confidence','nll_p50','unk_target_fraction','pred_std_ratio','bias','pred_autocorr1','train_acc_subsample'} if cmd=='domain' else {k for v in c.values() for k in v if k.startswith(('x_sharp','x_hess','x_flat','x_dist','x_stable','x_weight_kurt'))}
            if cmd=='landscape':
                for cell in c.values():
                    for role,value in (cell.get('x_role_dist_from_init') or {}).items():cell['x_role_dist_from_init.'+role]=value;keys.add('x_role_dist_from_init.'+role)
            for k in sorted(keys):
                pairs=paired(c,c,new,ref,k,True)
                if not pairs:continue
                metrics.add(k);a,b=paired_points(pairs)
                dif=[x-y for u,s,x,y in pairs]
                direction=DIRECTION.get(k,'neutral')
                if k=='auc_val_curve':direction='higher' if higher(name) else 'lower'
                delta=float(np.mean(dif))
                rows.append(dict(task=name,metric=k,new=a,ref=b,difference=delta,ci=bootstrap(dif),n=len(pairs),difference_statistic='mean paired difference',direction=direction,
                    rel_gain=((a-b)/abs(b)*(1 if direction=='higher' else -1)) if b not in (None,0) and direction!='neutral' else None))
        elif cmd=='mechanism':
            a=aggregate(c,args.arm)[0];b=aggregate(c,ref)[0]
            row=dict(task=name,arm=args.arm,rel_gain=gain(a,b,name) if a is not None and b not in (None,0) else None)
            keys={k for v in c.values() for k in v if k.startswith('eviekf_') and not isinstance(v[k],(list,dict))}
            for k in keys:row[k]=aggregate(c,args.arm,k)[0];metrics.add(k)
            before=aggregate(c,args.arm,'eviekf_energy_top_before_mean')[0];after=aggregate(c,args.arm,'eviekf_energy_top_after_mean')[0]
            row['energy_after_before_ratio']=after/before if after is not None and before not in (None,0) else None
            rows.append(row)
    if cmd=='profiles':
        rows=profile_points(profile_errors)
        # Section 16.0 forbids secondary-metric significance testing. Profiles
        # uses primary scores, but exploratory_report explicitly promises no
        # p-values, so report Friedman statistic only in all exploratory modes.
        a=[r['average_rank'] for r in rows if r['average_rank'] is not None];n=rows[0]['n_tasks'] if rows else 0;k=len(a)
        summary['friedman_statistic']=12*n/(k*(k+1))*sum(r*r for r in a)-3*n*(k+1) if k>1 else None
        summary['note']='No exploratory p-values (section 16.0/16.6 exploratory-report constraint takes precedence)'
    if cmd=='mechanism':
        summary['spearman']={}
        for key in metrics:
            pairs=[(r[key],r['rel_gain']) for r in rows if finite(r.get(key)) and finite(r.get('rel_gain'))]
            value=float(np.corrcoef(ranks([x for x,y in pairs]),ranks([y for x,y in pairs]))[0,1]) if len(pairs)>2 and len({x for x,y in pairs})>1 and len({y for x,y in pairs})>1 else None
            summary['spearman'][key]=dict(rho=value if finite(value) else None,n=len(pairs))
    if cmd=='efficiency':
        for domain in ['vision','language','finance']:
            grouped={}
            for r in rows:
                if r['task'].startswith(domain+'_'):grouped.setdefault(r['task'],[]).append(r)
            values=[float(np.median([r['wall_ratio'] for r in rs if finite(r.get('wall_ratio'))])) for rs in grouped.values() if any(finite(r.get('wall_ratio')) for r in rs)]
            summary[domain]=dict(tasks=len(grouped),tasks_faster=sum(v<1 for v in values),median_wall_ratio=float(np.median(values)) if values else None,reached_fraction=sum(r['reached'] for r in rows if r['task'].startswith(domain+'_'))/max(1,sum(r['task'].startswith(domain+'_') for r in rows)))
    if cmd=='domain':
        summary['direction_counts']={key:dict(n=sum(r['metric']==key for r in rows),wins=sum(r['metric']==key and finite(r.get('rel_gain')) and r['rel_gain']>0 for r in rows),ties=sum(r['metric']==key and r.get('rel_gain')==0 for r in rows),losses=sum(r['metric']==key and finite(r.get('rel_gain')) and r['rel_gain']<0 for r in rows)) for key in metrics if DIRECTION.get(key) in ('lower','higher')}
    if cmd=='stability':
        summary['by_arm']={a:{key:float(np.median([r[key] for r in rows if r['arm']==a and finite(r.get(key))])) if any(r['arm']==a and finite(r.get(key)) for r in rows) else None for key in metrics} for a in {r['arm'] for r in rows}}
    if cmd=='pareto':
        summary['frontier_counts']={a:{cost:sum(r['arm']==a and bool(r.get('frontier_'+cost)) for r in rows) for cost in ['wall_s','peak_mem_mb']} for a in {r['arm'] for r in rows}}
    return dict(header=f'EXPLORATORY: no p-values; {len(metrics)} metrics examined. '+header,rows=rows,summary=summary,metrics=sorted(metrics))


def clean(value):
    if isinstance(value,dict):return {str(k):clean(v) for k,v in value.items()}
    if isinstance(value,(list,tuple,np.ndarray)):return [clean(v) for v in value]
    if isinstance(value,np.generic):value=value.item()
    return None if isinstance(value,float) and not math.isfinite(value) else value


def markdown(result):
    rows=result.get('rows',[]);keys=sorted(set().union(*(r.keys() for r in rows))) if rows else []
    def render(v):return json.dumps(clean(v),ensure_ascii=False).replace('|','\\|').replace('\n',' ') if isinstance(v,(dict,list)) else str(v)
    lines=[result.get('header',''),'', '| '+' | '.join(keys)+' |','| '+' | '.join(['---']*len(keys))+' |'] if keys else [result.get('header',''),'','No data.']
    lines+=['| '+' | '.join(render(r.get(k,'')) for k in keys)+' |' for r in rows]
    lines+=['','Summary: '+json.dumps(clean(result.get('summary',{})),ensure_ascii=False)]
    return '\n'.join(lines)+'\n'


def write(name,result):
    OUT.mkdir(parents=True,exist_ok=True);result=clean(result)
    (OUT/(name+'.json')).write_text(json.dumps(result,indent=2,allow_nan=False))
    (OUT/(name+'.md')).write_text(markdown(result));print(markdown(result))
    # One tabular deliverable spans every analysis command already run.
    records=[]
    for path in sorted(OUT.glob('*.json')):
        d=json.loads(path.read_text())
        for i,row in enumerate(d.get('rows',[])):
            for metric,value in row.items():
                records.append(dict(analysis=path.stem,row=i,task=row.get('task',''),arm=row.get('arm',row.get('new','')),metric=metric,value=json.dumps(value),status='unavailable' if value is None else 'ok'))
    with (OUT/'rebuttal_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=['analysis','row','task','arm','metric','value','status']);w.writeheader();w.writerows(records)


def selftest():
    assert wilcoxon([1,2,3])==.25
    assert wilcoxon([0,0])==1
    assert wilcoxon([1,1,1])==.25
    assert sign_test(3,0)==.25 and sign_test(1,1)==1
    assert holm({'a':.01,'b':.04})['a']['reject']
    assert not holm({'a':.03,'b':.04})['b']['reject']
    dl=random_effects([1.,1.],[.1,.1]);assert abs(dl['effect']-1)<1e-12 and dl['tau2']==0
    e=efficiency_pair(dict(val_curve=[[10,8],[20,4]],sec_per_step=2),dict(best_val_primary=5,steps_to_best_val=40,wall_s=100,steps_run=20))
    assert e['steps_to_target']==20 and e['wall_to_target']==40
    p=profile_points([dict(a=1,b=2),dict(a=2,b=1),dict(a=1,b=1)],taus=[1,2])
    assert p[0]['rho']==[2/3,1.] and p[1]['rho']==[2/3,1.]
    assert not set(LOWER)&set(HIGHER)
    print('selftest PASS: exact ranks/sign/Holm/DL, crossing, profiles, directions')


def main():
    parser=argparse.ArgumentParser(description=__doc__);sub=parser.add_subparsers(dest='command',required=True)
    commands=['selftest','seed42','finance_filter','budget','holm_limit','compare','r4','r3a','efficiency','pareto','profiles','lr_robustness','seed_variance','stability','landscape','domain','mechanism','exploratory_report']
    for name in commands:
        p=sub.add_parser(name);p.add_argument('--roots',nargs='+',default=[str(BENCH.parent/'checkpoints'),str(ARCHIVE)])
        p.add_argument('--new',default='EvieKF');p.add_argument('--old',default='AdamW')
        p.add_argument('--ref',default=None);p.add_argument('--arms',default='');p.add_argument('--arm',default='EvieKF')
        p.add_argument('--task');p.add_argument('--host',choices=['finance_tech','language_wikitext2'],default='language_wikitext2')
        p.add_argument('--domain',choices=['vision','language','finance'],default='language')
        p.add_argument('--m',type=int,default=7)
        p.add_argument('--include-adamw',action='store_true')
    args=parser.parse_args();args.ref_arm=args.ref or 'AdamW'
    if args.command=='selftest':selftest();return
    if args.command=='exploratory_report':
        sections={cmd:exploratory(args,cmd) for cmd in ['efficiency','pareto','profiles','lr_robustness','seed_variance','stability','landscape','domain','mechanism']}
        domain_results=[]
        for domain in ['vision','language','finance']:
            domain_args=argparse.Namespace(**vars(args));domain_args.domain=domain
            domain_results.append(exploratory(domain_args,'domain'))
        sections['domain']=dict(header='EXPLORATORY: all three domains; no p-values',
            rows=[row for result in domain_results for row in result['rows']],
            summary={d:r['summary'] for d,r in zip(['vision','language','finance'],domain_results)},
            metrics=sorted(set().union(*(r['metrics'] for r in domain_results))))
        metrics=set().union(*(r['metrics'] for r in sections.values()))
        header=f'{len(metrics)} metrics examined. Exploratory. No p-values. Anything we want to claim is re-stated as a pre-registered prediction first.'
        OUT.mkdir(parents=True,exist_ok=True)
        (OUT/'exploratory_report.md').write_text(header+'\n\n'+'\n\n'.join('## '+cmd+'\n\n'+markdown(r) for cmd,r in sections.items()))
        scoreboard=[]
        for key,direction in DIRECTION.items():
            gains=[]
            for name,path in discover(args).items():
                c=load(path);a,b=paired_points(paired(c,c,args.new,args.ref_arm,key,True))
                if a is not None and b not in (None,0):gains.append((a-b)/abs(b)*(1 if direction=='higher' else -1))
            scoreboard.append(dict(metric=key,n=len(gains),wins=sum(v>0 for v in gains),ties=sum(v==0 for v in gains),losses=sum(v<0 for v in gains),median_relative_difference=float(np.median(gains)) if gains else None))
        write('metric_scoreboard',dict(header=header,rows=scoreboard))
        for cmd,r in sections.items():write(cmd,r)
        print('Wrote',OUT/'exploratory_report.md');return
    if args.command in ['seed42','finance_filter','budget','holm_limit','compare','r4','r3a']:
        if args.command=='compare' and (not args.task or not args.ref or not args.arms):parser.error('compare requires --task, --ref and --arms')
        result=primary_commands(args)
    else:result=exploratory(args)
    write(args.command,result)


if __name__=='__main__':main()
