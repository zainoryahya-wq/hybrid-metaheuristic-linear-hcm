"""Find and cluster the roots reached by the Standard solver on each
benchmark (used for the root column of Table 1)."""
import numpy as np
from scipy.optimize import fsolve
from benchmarks import get_all_systems
from hybrid import run_hybrid
for S in get_all_systems():
    sols=[]
    for seed in range(50):
        r=run_hybrid('Standard',S,seed)
        if r['success']: sols.append(r['x_sol'])
    # cluster
    cl=[]
    for x in sols:
        for c in cl:
            if np.linalg.norm(c[0]-x,np.inf)<1e-4: c[1]+=1; break
        else: cl.append([x,1])
    print(S.name,len(sols),'distinct roots',len(cl),[ (c[1], np.round(c[0][:4],4).tolist(), round(float(np.linalg.norm(c[0],np.inf)),4)) for c in cl])
