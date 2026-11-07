"""
run_experiments.py
===================
Run the full experiment: 50 runs (seeds 0-49) of each method (PSO-Linear HCM,
GA-Linear HCM, GWO-Linear HCM, Standard Linear Newton-HCM) on
each benchmark system, and save the raw results to CSV.

Usage:
    python3 run_experiments.py                  # all systems
    python3 run_experiments.py --system NAME     # one system only (partial name match)
"""

import argparse
import csv
import os
import sys
import pickle

import numpy as np

from benchmarks import get_all_systems
from hybrid import run_hybrid

N_RUNS = 50
ALGOS = ["PSO", "GA", "GWO", "Standard"]

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)
RAW_CSV = os.path.join(RESULTS_DIR, "raw_results.csv")
CURVES_PKL = os.path.join(RESULTS_DIR, "curves.pkl")

FIELDNAMES = ["system", "n", "algo", "seed", "n_meta", "n_hcm", "cpu_meta",
              "cpu_hcm", "cpu_total", "residual_inf", "success", "meta_converged"]


def load_curve_store():
    if os.path.exists(CURVES_PKL):
        with open(CURVES_PKL, "rb") as f:
            return pickle.load(f)
    return {}


def save_curve_store(store):
    with open(CURVES_PKL, "wb") as f:
        pickle.dump(store, f)


def already_done(system_name):
    """Check existing rows in the CSV so runs can be resumed safely."""
    done = set()
    if os.path.exists(RAW_CSV):
        with open(RAW_CSV, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                if row["system"] == system_name:
                    done.add((row["algo"], int(row["seed"])))
    return done


def run_for_system(sysobj, writer, curve_store):
    done = already_done(sysobj.name)
    # keep only ONE example curve (seed=0) per algo/system for plotting
    curve_key_seeds = {0}

    for algo in ALGOS:
        for seed in range(N_RUNS):
            if (algo, seed) in done:
                continue
            r = run_hybrid(algo, sysobj, seed=seed)
            writer.writerow({
                "system": sysobj.name,
                "n": sysobj.n,
                "algo": algo,
                "seed": seed,
                "n_meta": r["n_meta"],
                "n_hcm": r["n_hcm"],
                "cpu_meta": r["cpu_meta"],
                "cpu_hcm": r["cpu_hcm"],
                "cpu_total": r["cpu_total"],
                "residual_inf": r["residual_inf"],
                "success": int(r["success"]),
                "meta_converged": int(r["meta_converged"]),
            })
            if seed in curve_key_seeds:
                key = (sysobj.name, algo, seed)
                curve_store[key] = dict(
                    conv_time=r["conv_time"], conv_error=r["conv_error"],
                    t_path=r["t_path"], x_path=[x.tolist() for x in r["x_path"]],
                )
        print(f"  [{sysobj.name}] {algo}: completed {N_RUNS} runs", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--system", default=None, help="Partial system name (e.g. 'n10', 'Rosenbrock_n8')")
    args = parser.parse_args()

    systems = get_all_systems()
    if args.system:
        systems = [s for s in systems if args.system in s.name]
        if not systems:
            print(f"No system matches '{args.system}'", file=sys.stderr)
            sys.exit(1)

    write_header = not os.path.exists(RAW_CSV)
    curve_store = load_curve_store()

    with open(RAW_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        if write_header:
            writer.writeheader()
        for sysobj in systems:
            print(f"Running experiments for {sysobj.name} (n={sysobj.n}) ...", flush=True)
            run_for_system(sysobj, writer, curve_store)
            f.flush()

    save_curve_store(curve_store)
    print("Done.")


if __name__ == "__main__":
    main()
