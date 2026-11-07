"""
stats.py
========
Statistical tests of each hybrid against the Standard baseline (main experiment).

- Success: two-sided Fisher exact test, Holm-adjusted over the 30 comparisons
  (3 hybrids x 10 systems).
- Homotopy iterations N_HCM: Wilcoxon signed-rank test on the seed-paired runs
  (the hybrid and the baseline share the seed), Holm-adjusted over 30 comparisons.

Input : results/raw_results.csv
Output: results/stats_tests.csv
"""
import numpy as np
import pandas as pd
from scipy.stats import fisher_exact, wilcoxon

raw = pd.read_csv('results/raw_results.csv')
systems = list(dict.fromkeys(raw.system))
rows = []
for s in systems:
    d = raw[raw.system == s]
    std = d[d.algo == 'Standard'].sort_values('seed')
    k_std = int(std.success.sum())
    for a in ['PSO', 'GA', 'GWO']:
        h = d[d.algo == a].sort_values('seed')
        assert (h.seed.values == std.seed.values).all()
        k = int(h.success.sum())
        p_f = fisher_exact([[k, 50 - k], [k_std, 50 - k_std]])[1]
        p_w = wilcoxon(h.n_hcm.values, std.n_hcm.values).pvalue
        rows.append(dict(system=s, algo=a, k=k, kstd=k_std, p_fisher=p_f, p_nhcm_wilcoxon=p_w))
R = pd.DataFrame(rows)


def holm(p):
    p = np.asarray(p, float); m = len(p); order = np.argsort(p); adj = np.empty(m); run = 0.0
    for i, idx in enumerate(order):
        run = max(run, (m - i) * p[idx]); adj[idx] = min(1.0, run)
    return adj


R['p_fisher_holm'] = holm(R.p_fisher)
R['p_nhcm_wilcoxon_holm'] = holm(R.p_nhcm_wilcoxon)
R.to_csv('results/stats_tests.csv', index=False)
pd.set_option('display.width', 200)
print(R.round(4).to_string())
print('N_HCM significantly different after Holm:', int((R.p_nhcm_wilcoxon_holm < 0.05).sum()), 'of', len(R))
