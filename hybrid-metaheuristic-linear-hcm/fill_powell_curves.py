import pickle, os
from benchmarks import System4ExtendedPowellSingular
from hybrid import run_hybrid

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
CURVES_PKL = os.path.join(RESULTS_DIR, "curves.pkl")

with open(CURVES_PKL, "rb") as f:
    curve_store = pickle.load(f)

ALGOS = ["PSO", "GA", "GWO", "Standard"]
for n in (4, 8, 16):
    sysobj = System4ExtendedPowellSingular(n)
    for algo in ALGOS:
        key = (sysobj.name, algo, 0)
        if key in curve_store:
            continue
        r = run_hybrid(algo, sysobj, seed=0)
        curve_store[key] = dict(
            conv_time=r["conv_time"], conv_error=r["conv_error"],
            t_path=r["t_path"], x_path=[x.tolist() for x in r["x_path"]],
        )
        print("filled", key)

with open(CURVES_PKL, "wb") as f:
    pickle.dump(curve_store, f)
print("systems now:", sorted(set(k[0] for k in curve_store.keys())))
