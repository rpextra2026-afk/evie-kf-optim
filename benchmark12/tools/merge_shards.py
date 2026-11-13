#!/usr/bin/env python3
"""Merge one task's downloaded shard trees; fail if any expected cell is missing."""
import argparse,copy,json,math,os,tempfile
from pathlib import Path
from collections import Counter


def atomic(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.merge-')
    try:
        with os.fdopen(fd,'w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
        os.replace(tmp,path)
    finally:
        if os.path.exists(tmp):os.unlink(tmp)


def same(a,b):return a==b or isinstance(a,float) and isinstance(b,float) and math.isnan(a) and math.isnan(b)


def merge(dirs,out):
    inputs=[]
    for name in dirs:
        path=Path(name)
        candidates=[path] if path.name=='results.json' else [path/'results.json'] if (path/'results.json').exists() else sorted(path.rglob('results.json'))
        inputs.extend(candidates)
    inputs=list(dict.fromkeys(p.resolve() for p in inputs))
    if not inputs:raise ValueError('No results.json files found; extract the downloaded tarballs first')
    cells={};first=None;lr={};units=set();expected=set();all_arms=set()
    for path in inputs:
        d=json.loads(path.read_text());meta=d.get('meta',{})
        if first is None:first=d
        elif meta.get('task_name')!=first['meta'].get('task_name'):
            raise ValueError('Inputs contain multiple tasks; merge one task at a time')
        if first['meta'].get('data_provenance') and meta.get('data_provenance') and first['meta']['data_provenance']!=meta['data_provenance']:
            raise ValueError(f'{path}: data fingerprints differ')
        if bool(meta.get('smoke'))!=bool(first['meta'].get('smoke')):raise ValueError('Cannot merge smoke and production results')
        units.update(meta.get('all_units',meta.get('units',[])))
        defined_arms=meta.get('task_config',{}).get('opt_list') or meta.get('opt_names',[])
        all_arms.update(defined_arms)
        expected.update(f'{a}|{u}|{s}' for a in defined_arms for u in meta.get('all_units',meta.get('units',[])) for s in meta.get('seeds',[]))
        if meta.get('expected_cell_keys'):expected.update(meta['expected_cell_keys'])
        else:expected.update(f'{a}|{u}|{s}' for a in meta.get('opt_names',[]) for u in meta.get('all_units',meta.get('units',[])) for s in meta.get('seeds',[]))
        lp=path.with_name('lr_search.json')
        if not lp.exists():raise FileNotFoundError(lp)
        for arm,e in json.loads(lp.read_text()).items():
            if not e.get('complete'):raise ValueError(f'{lp}: incomplete {arm}')
            if arm in lr and (lr[arm].get('lr'),lr[arm].get('knob'))!=(e.get('lr'),e.get('knob')):raise ValueError(f'Conflicting frozen choice for {arm}')
            lr[arm]=e
        for key,c in d['cells'].items():
            prior=cells.get(key)
            if prior and prior.get('recorded') and c.get('recorded') and not same(prior.get('primary'),c.get('primary')):
                raise ValueError(f'Conflicting primary for {key}')
            if not prior or (c.get('recorded') and not prior.get('recorded')):cells[key]=copy.deepcopy(c)
    # Resolve aliases after union, including sources in another notebook.
    changed=True
    while changed:
        changed=False
        for key,c in list(cells.items()):
            if c.get('deferred'):
                source=cells.get(c.get('aliased_from'))
                if source and source.get('recorded') and not source.get('deferred'):
                    arm,unit,seed=key.split('|');value=copy.deepcopy(source)
                    value.update(opt=arm,unit=unit,seed=int(seed),aliased_from=c['aliased_from'],recorded=True,deferred=False)
                    cells[key]=value;changed=True
    unresolved=[k for k,c in cells.items() if c.get('deferred')]
    missing=sorted(k for k in expected if not cells.get(k,{}).get('recorded') or cells[k].get('deferred'))
    meta=dict(first['meta']);meta.update(merged_from=[str(p.parent) for p in inputs],n_cells=len(cells),
        units=sorted(units),all_units=sorted(units),opt_names=sorted(all_arms),unit_slice=None,cell_slice=None,shard_tag='',
        expected_cell_keys=sorted(expected),missing_cells=missing,deferred_cells=unresolved,
        lr_used={a:e['lr'] for a,e in lr.items()},knob_used={a:e.get('knob') for a,e in lr.items()})
    atomic(Path(out)/'results.json',dict(cells=cells,meta=meta));atomic(Path(out)/'lr_search.json',lr)
    for arm in sorted(all_arms):
        exp=sum(k.split('|')[0]==arm for k in expected)
        done=sum(k.split('|')[0]==arm and c.get('recorded',False) and not c.get('deferred') for k,c in cells.items())
        failed=sum(k.split('|')[0]==arm and bool(c.get('diverged')) for k,c in cells.items())
        print(f'{arm}: finished {done}/{exp}, numerical failures {failed}')
    if unresolved:print('DEFERRED:',unresolved)
    if missing:print('MISSING:',missing)
    return 1 if missing or unresolved else 0


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',required=True);p.add_argument('shards',nargs='+');a=p.parse_args()
    raise SystemExit(merge(a.shards,a.out))


if __name__=='__main__':main()
