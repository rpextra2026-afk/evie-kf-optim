#!/usr/bin/env python3
"""R7: fixed-protocol correlated-noise quadratic, numpy only, no tuning.

Noise and initialization are identical across methods within a seed. This is
preconditioned SGD without Adam whitening, not a reproduction of the full optimizer.
"""
import json
import math
from pathlib import Path
import time
import numpy as np

M,N=16,24
T=4000
N_SEEDS=20
SIGMA2=4.
BETA=.99
GAMMAS=[0.,1.,3.,10.,30.,100.]
METHODS=['plain','diag_gate','kron_gate','kron_gate_oracle']
OUT=Path(__file__).resolve().parents[1]/'results/rebuttal'


def experiment():
    start=time.monotonic();rng=np.random.default_rng(728)
    qa=np.linalg.qr(rng.normal(size=(M,M)))[0];qb=np.linalg.qr(rng.normal(size=(N,N)))[0]
    av=np.logspace(0,-1.5,M);bv=np.logspace(0,-1,N)
    av*=SIGMA2/(av.sum()*bv.sum())
    A=(qa*av)@qa.T;B=(qb*bv)@qb.T
    la=np.linalg.cholesky(A);lb=np.linalg.cholesky(B)
    assert np.isclose(np.trace(A)*np.trace(B),SIGMA2)
    H=np.logspace(0,-2,M)[:,None]*np.logspace(0,-1,N)[None,:]
    eta=.5/H.max();gam=np.array(GAMMAS)[:,None,None]
    oracle_gain=1/np.sqrt(1+gam*(av/av.mean())[:,None]*(bv/bv.mean())[None,:])
    floors={name:[] for name in METHODS}
    for seed in range(N_SEEDS):
        rng=np.random.default_rng(seed);initial=rng.normal(size=(M,N))
        plain=initial.copy();w={name:np.repeat(initial[None],len(GAMMAS),0) for name in METHODS[1:]}
        ah=np.zeros((M,M));bh=np.zeros((N,N));sh=np.zeros((M,N))
        sums={name:np.zeros(len(GAMMAS)) for name in METHODS[1:]};plain_sum=0.
        for step in range(T):
            xi=la@rng.normal(size=(M,N))@lb.T
            xi2=la@rng.normal(size=(M,N))@lb.T
            delta=(xi-xi2)/2
            sh=BETA*sh+(1-BETA)*delta**2
            ah=BETA*ah+(1-BETA)*(delta@delta.T)/N
            bh=BETA*bh+(1-BETA)*(delta.T@delta)/M
            ae,au=np.linalg.eigh(ah);be,bu=np.linalg.eigh(bh)
            ae=np.maximum(ae,0);be=np.maximum(be,0)
            gain=1/np.sqrt(1+gam*(ae/max(ae.mean(),1e-30))[:,None]*(be/max(be.mean(),1e-30))[None,:])
            grad=H*plain+xi;plain-=eta*grad
            for name in METHODS[1:]:
                g=H*w[name]+xi
                if name=='diag_gate':z=g/np.sqrt(1+gam*sh/max(sh.mean(),1e-30))
                else:
                    u,v,gc=(au,bu,gain) if name=='kron_gate' else (qa,qb,oracle_gain)
                    z=u@((u.T@g@v)*gc)@v.T
                norms=np.linalg.norm(g,axis=(1,2));zn=np.linalg.norm(z,axis=(1,2))
                z*=np.divide(norms,zn,out=np.ones_like(norms),where=zn>1e-12)[:,None,None]
                # Bypass spectral roundoff at gamma=0: exact plain identity.
                z[0]=g[0]
                w[name]-=eta*z
                if seed==0:assert np.array_equal(w[name][0],plain),name
                if step>=int(.8*T):sums[name]+=.5*np.sum(H*w[name]**2,axis=(1,2))
            if step>=int(.8*T):plain_sum+=.5*np.sum(H*plain**2)
        n=T-int(.8*T);floors['plain'].append(plain_sum/n)
        for name in METHODS[1:]:floors[name].append(sums[name]/n)
    plain=np.array(floors['plain']);rows=[]
    rows.append(dict(method='plain',gamma=0,mean=float(plain.mean()),se=float(plain.std(ddof=1)/math.sqrt(N_SEEDS)),ratio=1.))
    for name in METHODS[1:]:
        a=np.array(floors[name])
        for index,gamma in enumerate(GAMMAS):
            row=dict(method=name,gamma=gamma,mean=float(a[:,index].mean()),
                se=float(a[:,index].std(ddof=1)/math.sqrt(N_SEEDS)),ratio=float(a[:,index].mean()/plain.mean()))
            if name=='kron_gate':row['fraction_beating_diag']=float((a[:,index]<np.array(floors['diag_gate'])[:,index]).mean())
            rows.append(row)
    return dict(config=dict(m=M,n=N,steps=T,seeds=N_SEEDS,sigma2=SIGMA2,beta=BETA,eta=eta,gammas=GAMMAS),
                rows=rows,per_seed={k:np.asarray(v).tolist() for k,v in floors.items()},elapsed_s=time.monotonic()-start,
                note='Synthetic control, all outcomes retained; no p-values; no tuning.')


def main():
    result=experiment();OUT.mkdir(parents=True,exist_ok=True)
    path=OUT/'noisy_quadratic.json';path.write_text(json.dumps(result,indent=2))
    print('method | gamma | mean noise floor | SE | ratio to plain | kron seed wins vs diag')
    for r in result['rows']:print(' | '.join(str(r.get(k,'')) for k in ['method','gamma','mean','se','ratio','fraction_beating_diag']))
    print('seconds:',result['elapsed_s'],'output:',path)


if __name__=='__main__':main()
