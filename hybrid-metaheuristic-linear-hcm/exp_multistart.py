"""Multi-start baseline: restart Standard Linear Newton-HCM from independent
uniform points (seeds 0-49, at most 12 starts, stop at first success).
Output: results/multistart_raw.csv"""
import json, csv, time, numpy as np
from multiprocessing import Pool
from benchmarks import get_all_systems
from hcm import LinearNewtonHCM
KMAX=12; NR=50
def task(name):
    S={s.name:s for s in get_all_systems()}[name]
    lo,hi=S.bounds; hcm=LinearNewtonHCM(); out=[]
    for seed in range(NR):
        rng=np.random.default_rng(seed); it=[];cp=[];su=[]
        for k in range(KMAX):
            x0=rng.uniform(lo,hi,size=S.n)
            r=hcm.solve(S,x0); it.append(int(r['n_iter'])); cp.append(float(r['cpu_time'])); su.append(int(r['converged']))
            if r['converged']: break
        out.append(dict(system=name,seed=seed,n_starts=len(it),success=su[-1],iters=json.dumps(it),cpus=json.dumps(cp)))
    return out
if __name__=='__main__':
    names=[s.name for s in get_all_systems()]; rows=[]; t0=time.time()
    with Pool(2) as p:
        for i,o in enumerate(p.imap_unordered(task,names)):
            rows+=o; print(i+1,len(names),round(time.time()-t0),flush=True)
    with open('results/multistart_raw.csv','w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    print('done',flush=True)
