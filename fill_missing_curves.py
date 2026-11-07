"""
One-off repair script: the original run_experiments.py process for this
project was interrupted before it could call save_curve_store() the first
time, so the seed=0 example curves for the systems it had already finished
(System1, System2 n=5, n=10, and GWO seed=0 for n=20) were never written to
results/curves.pkl even though their raw_results.csv rows exist. This
re-runs just seed=0 for each algo on the affected (system, algo) pairs to
regenerate those curves, without touching raw_results.csv.
"""
import pickle
import os
import numpy as np

from benchmarks import get_all_systems
from hybrid import run_hybrid

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CURVES_PKL = os.path.join(RESULTS_DIR, "curves.pkl")

with open(CURVES_PKL, "rb") as f:
    curve_store = pickle.load(f)

ALGOS = ["PSO", "GA", "GWO", "Standard"]
systems = get_all_systems()

for sysobj in systems:
    for algo in ALGOS:
        key = (sysobj.name, algo, 0)
        if key in curve_store:
            continue
        r = run_hybrid(algo, sysobj, seed=0)
        curve_store[key] = dict(
            conv_time=r["conv_time"], conv_error=r["conv_error"],
            t_path=r["t_path"], x_path=[x.tolist() for x in r["x_path"]],
        )
        print(f"filled {key}")

with open(CURVES_PKL, "wb") as f:
    pickle.dump(curve_store, f)

print("Done. Systems now in store:", sorted(set(k[0] for k in curve_store.keys())))
