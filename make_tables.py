"""
make_tables.py
==============
Write the numbers of Tables 2-4 of the paper as CSV files in tables/.

Inputs : results/raw_results.csv, results/stats_tests.csv (from stats.py),
         results/env2_summary.csv (from analyze_env2.py)
Outputs: tables/table2_success.csv, tables/table3_cost.csv, tables/table4_newton_multistart.csv
"""
import os
from math import sqrt
import pandas as pd

os.makedirs('tables', exist_ok=True)
raw = pd.read_csv('results/raw_results.csv')
st = pd.read_csv('results/stats_tests.csv')
e2 = pd.read_csv('results/env2_summary.csv')
SYS = list(dict.fromkeys(raw.system))
ALG = ['Standard', 'PSO', 'GA', 'GWO']


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


# Table 2: successes out of 50, Wilson 95% interval, Holm-adjusted Fisher p
rows = []
for s in SYS:
    d = raw[raw.system == s]
    for a in ALG:
        k = int(d[d.algo == a].success.sum()); lo, hi = wilson(k, 50)
        p = st[(st.system == s) & (st.algo == a)].p_fisher_holm
        rows.append(dict(system=s, method=a, successes=k, runs=50, wilson_lo=round(lo, 1), wilson_hi=round(hi, 1),
                         p_fisher_holm=(None if a == 'Standard' else float(p.iloc[0]))))
pd.DataFrame(rows).to_csv('tables/table2_success.csv', index=False)

# Table 3: mean N_HCM, mean total CPU time (s), CPU time per successful run (s)
rows = []
for s in SYS:
    d = raw[raw.system == s]
    for a in ALG:
        h = d[d.algo == a]
        rows.append(dict(system=s, method=a, mean_n_hcm=round(h.n_hcm.mean(), 1), mean_cpu_total=round(h.cpu_total.mean(), 4),
                         cpu_per_success=(round(h.cpu_total.mean() / h.success.mean(), 4) if h.success.mean() > 0 else None)))
pd.DataFrame(rows).to_csv('tables/table3_cost.csv', index=False)

# Table 4: second environment, same start x0: Linear Newton-HCM, Newton, damped Newton; CPU-matched multi-start
t4 = e2[['system', 'algo', 'hcm', 'newton', 'damped', 'K', 'ms', 'p_hyb_vs_ms']].rename(
    columns={'algo': 'start', 'hcm': 'linear_newton_hcm_pct', 'newton': 'newton_pct', 'damped': 'damped_newton_pct',
             'ms': 'multistart_pct', 'p_hyb_vs_ms': 'p_fisher_hybrid_vs_multistart'})
t4.to_csv('tables/table4_newton_multistart.csv', index=False)
print('Wrote tables/table2_success.csv, tables/table3_cost.csv, tables/table4_newton_multistart.csv')
