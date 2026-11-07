"""Hand-off threshold sensitivity: re-run the three hybrids with
eps_meta in {1e-1, 1e-2, 1e-3}, seeds 0-49. Output: results/sensitivity_raw.csv"""
import sys, csv, itertools, time
from multiprocessing import Pool
from benchmarks import get_all_systems
from hybrid import run_hybrid
EPS=[1e-1,1e-2,1e-3]; ALGOS=['PSO','GA','GWO']; NR=50
def task(a):
    eps,name=a
    S={s.name:s for s in get_all_systems()}[name]
    out=[]
    for algo in ALGOS:
        for seed in range(NR):
            r=run_hybrid(algo,S,seed=seed,meta_kwargs={'eps_meta':eps})
            out.append(dict(eps=eps,system=name,algo=algo,seed=seed,n_meta=r['n_meta'],n_hcm=r['n_hcm'],cpu_meta=r['cpu_meta'],cpu_total=r['cpu_total'],success=int(r['success']),meta_converged=int(r['meta_converged'])))
    return out
if __name__=='__main__':
    names=[s.name for s in get_all_systems()]
    tasks=[(e,n) for n in names for e in EPS]
    t0=time.time(); rows=[]
    with Pool(2) as p:
        for i,o in enumerate(p.imap_unordered(task,tasks)):
            rows+=o; print(i+1,len(tasks),round(time.time()-t0),flush=True)
    with open('results/sensitivity_raw.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('done',flush=True)
