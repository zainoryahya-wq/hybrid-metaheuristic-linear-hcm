"""Within-environment (env 2) comparison: Linear Newton-HCM vs direct Newton
(undamped, damped) from the same start, and CPU-matched multi-start."""
import json, numpy as np, pandas as pd
from scipy.stats import fisher_exact
a = pd.read_csv('results/newton_ablation_raw.csv'); m = pd.read_csv('results/multistart_raw.csv')
a['hyb_cpu'] = a.cpu_meta + a.hcm_cpu
SYS = ['System1_ExpTrig2D', 'System2_BroydenTridiagonal_n5', 'System2_BroydenTridiagonal_n10', 'System2_BroydenTridiagonal_n20']
rows = []
for s in a.system.unique():
    d = a[a.system == s]; std = d[d.algo == 'Standard']; per = std.hcm_cpu.mean()
    ms = m[m.system == s].sort_values('seed')
    for al in ['Standard', 'PSO', 'GA', 'GWO']:
        h = d[d.algo == al].sort_values('seed')
        r = dict(system=s, algo=al, hcm=100 * h.hcm_ok.mean(), newton=100 * h.nwt_ok.mean(), damped=100 * h.dmp_ok.mean(),
                 cpu=h.hyb_cpu.mean(), newton_cpu=(h.cpu_meta + h.nwt_cpu).mean(), damped_cpu=(h.cpu_meta + h.dmp_cpu).mean())
        if al == 'Standard':
            K = 1
        else:
            K = int(min(12, max(1, round(h.hyb_cpu.mean() / per))))
        ok = ((ms.success == 1) & (ms.n_starts <= K)).values
        r['K'] = K; r['ms'] = 100 * ok.mean()
        r['ms_cpu'] = np.mean([sum(json.loads(c)[:K]) for c in ms.cpus])
        k1, k2 = int(h.hcm_ok.sum()), int(ok.sum())
        r['p_hyb_vs_ms'] = fisher_exact([[k1, 50 - k1], [k2, 50 - k2]])[1] if al != 'Standard' else np.nan
        rows.append(r)
out = pd.DataFrame(rows); out.to_csv('results/env2_summary.csv', index=False)
pd.set_option('display.width', 250); print(out.round(3).to_string())
# Std first start in multistart vs Standard in ablation
st = a[a.algo == 'Standard'].sort_values(['system', 'seed']); fs = m.sort_values(['system', 'seed'])
fs1 = ((fs.n_starts >= 1) & (fs.success == 1) & (fs.n_starts == 1)).astype(int).values
print('first-start agreement', (st.hcm_ok.values == fs1).mean())
# failed vs successful start cost at Broyden 20 (env2)
ss = []
for _, rr in m[m.system == 'System2_BroydenTridiagonal_n20'].iterrows():
    c = json.loads(rr.cpus)
    for i, x in enumerate(c): ss.append((x, int(rr.success == 1 and i == len(c) - 1)))
ss = pd.DataFrame(ss, columns=['cpu', 'ok']); print(ss.groupby('ok').cpu.mean())
