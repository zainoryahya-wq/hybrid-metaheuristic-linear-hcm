"""Re-run 25 selected runs and compare with results/raw_results.csv
(reproducibility check), and confirm the seed protocol."""
import sys, time, numpy as np, pandas as pd
from benchmarks import get_all_systems
from hybrid import run_hybrid
df=pd.read_csv('results/raw_results.csv')
S={s.name:s for s in get_all_systems()}
bad=0;tot=0;t0=time.time()
for name,algo in [('System1_ExpTrig2D','PSO'),('System2_BroydenTridiagonal_n5','GWO'),('System2_BroydenTridiagonal_n20','Standard'),('System4_ExtendedPowellSingular_n4','GA'),('System2_BroydenTridiagonal_n20','GWO')]:
    for seed in range(5):
        r=run_hybrid(algo,S[name],seed=seed)
        row=df[(df.system==name)&(df.algo==algo)&(df.seed==seed)].iloc[0]
        ok=(r['n_meta']==row.n_meta and r['n_hcm']==row.n_hcm and int(r['success'])==row.success and int(r['meta_converged'])==row.meta_converged)
        tot+=1; bad+=(not ok)
        if not ok: print('DIFF',name,algo,seed,r['n_meta'],row.n_meta,r['n_hcm'],row.n_hcm)
print('checked',tot,'diff',bad,'time',round(time.time()-t0,1))
# seed protocol: identical initial populations?
lo,hi=S['System2_BroydenTridiagonal_n10'].bounds; n=10
pop=np.random.default_rng(3).uniform(lo,hi,size=(40,n)); x0=np.random.default_rng(3).uniform(lo,hi,size=n)
print('Standard x0 == first member of initial population:', np.allclose(pop[0],x0))
