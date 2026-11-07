"""
hybrid.py
=========
Combines Phase 1 (Metaheuristic: PSO / GA / GWO) with Phase 2 (Linear
Newton-HCM) into a single hybrid run, and provides a "Standard Linear
Newton-HCM" (no metaheuristic phase, random x0) as the control baseline
for comparison.
"""

import time
import numpy as np

from metaheuristics import PSO_Solver, GA_Solver, GWO_Solver
from hcm import LinearNewtonHCM

META_SOLVERS = {
    "PSO": PSO_Solver,
    "GA": GA_Solver,
    "GWO": GWO_Solver,
}


def run_hybrid(algo_name, system, seed, meta_kwargs=None, hcm_kwargs=None):
    """
    Run one full run of the hybrid <algo_name>-Linear HCM method on
    <system> with random seed <seed>.

    algo_name: "PSO", "GA", "GWO", or "Standard" (Standard Linear
               Newton-HCM without a metaheuristic phase; x0 drawn uniformly
               at random within bounds)
    """
    meta_kwargs = meta_kwargs or {}
    hcm_kwargs = hcm_kwargs or {}
    hcm = LinearNewtonHCM(**hcm_kwargs)

    if algo_name == "Standard":
        rng = np.random.default_rng(seed)
        lo, hi = system.bounds
        x0 = rng.uniform(lo, hi, size=system.n)
        n_meta = 0
        cpu_meta = 0.0
        meta_history = [np.sqrt(np.sum(system.F(x0) ** 2))]
        meta_time_history = [0.0]
        meta_converged = meta_history[0] <= 1e-2
    else:
        solver_cls = META_SOLVERS[algo_name]
        solver = solver_cls(seed=seed, **meta_kwargs)
        meta_res = solver.solve(system)
        x0 = meta_res["x_best"]
        n_meta = meta_res["n_iter"]
        cpu_meta = meta_res["cpu_time"]
        meta_history = meta_res["history"]
        meta_time_history = meta_res["time_history"]
        meta_converged = meta_res["converged"]

    hcm_res = hcm.solve(system, x0)

    total_cpu = cpu_meta + hcm_res["cpu_time"]
    # align the HCM-phase timeline so it continues after the metaheuristic phase
    hcm_time_path = list(np.linspace(cpu_meta, total_cpu, len(hcm_res["residual_path"])))

    success = bool(hcm_res["converged"])

    return dict(
        algo=algo_name,
        system=system.name,
        seed=seed,
        n_meta=n_meta,
        n_hcm=hcm_res["n_iter"],
        cpu_meta=cpu_meta,
        cpu_hcm=hcm_res["cpu_time"],
        cpu_total=total_cpu,
        residual_inf=hcm_res["residual_inf"],
        success=success,
        meta_converged=meta_converged,
        x_sol=hcm_res["x_sol"],
        # for convergence plots (time vs error norm, both phases combined)
        conv_time=list(meta_time_history) + hcm_time_path,
        conv_error=list(meta_history) + hcm_res["residual_path"],
        # for homotopy path-tracking plots
        t_path=hcm_res["t_path"],
        x_path=hcm_res["x_path"],
    )


if __name__ == "__main__":
    from benchmarks import System1ExpTrig
    sysobj = System1ExpTrig()
    for algo in ["PSO", "GA", "GWO", "Standard"]:
        r = run_hybrid(algo, sysobj, seed=42)
        print(f"{algo:8s} success={r['success']!s:5s} ||F||_inf={r['residual_inf']:.3e} "
              f"N_meta={r['n_meta']:4d} N_HCM={r['n_hcm']:4d} cpu_total={r['cpu_total']:.4f}s")
