# Hybrid metaheuristic–linear homotopy continuation for nonlinear systems

Code and data for the paper:

> Yahya, Z. R., & Mohamad Nor, H. *A Hybrid Metaheuristic–Linear Homotopy Continuation Framework for Solving Systems of Nonlinear Equations.* Manuscript submitted to the Songklanakarin Journal of Science and Technology.

A metaheuristic (PSO, GA or GWO) minimises the residual norm ‖F(x)‖₂ and hands its best point to a linear parameter homotopy
H(x, t) = (1 − t)(x − x₀) + t F(x), tracked with Newton correction (Linear Newton-HCM). The study compares the three hybrids with a single-start baseline, a CPU-matched multi-start baseline and Newton's method applied directly to F(x) = 0, on four benchmark families (n = 2–20).

## Requirements

Python 3.10 or later and the packages in `requirements.txt`:

```
pip install -r requirements.txt
```

All scripts are run from the repository root.

## Files

### Solvers and benchmarks

| File | Content |
|---|---|
| `benchmarks.py` | S1 Exp-Trig (n = 2), S2 Broyden tridiagonal (n = 5, 10, 20), S3 Extended Rosenbrock (n = 4, 8, 16), S4 Extended Powell singular (n = 4, 8, 16), with analytical Jacobians |
| `metaheuristics.py` | PSO, GA and GWO (population 40, at most 200 iterations, hand-off threshold ε_meta = 10⁻²) |
| `hcm.py` | Linear Newton-HCM: 100 uniform t-steps, at most 25 Newton iterations per step, corrector tolerance 10⁻⁸, success if ‖F‖∞ ≤ 10⁻⁶ |
| `hybrid.py` | Two-phase hybrid and the Standard baseline (one uniform random start) |

### Experiments (each writes one file in `results/`)

| Script | Output | Paper |
|---|---|---|
| `run_experiments.py` | `raw_results.csv` (2,000 rows: 10 systems × 4 methods × seeds 0–49), `curves.pkl` (seed-0 curves) | Tables 2–3, Figures 1–2 |
| `exp_threshold.py` | `sensitivity_raw.csv` (4,500 rows: ε_meta ∈ {10⁻¹, 10⁻², 10⁻³} × 3 hybrids × 10 systems × 50 seeds) | Section 3.4 |
| `exp_multistart.py` | `multistart_raw.csv` (restarts of the baseline, at most 12 per seed) | Table 4 |
| `exp_newton.py` | `newton_ablation_raw.csv` (2,000 rows: Linear Newton-HCM, undamped Newton and damped Newton from the same start) | Table 4 |

`fill_missing_curves.py` and `fill_powell_curves.py` are one-off repair scripts. The original run of `run_experiments.py` was interrupted before it saved the seed-0 curves of some systems, so these scripts re-ran seed 0 for the missing (system, method) pairs and added them to `curves.pkl` without changing `raw_results.csv`.

### Analysis (run after the experiments)

| Script | Output | Paper |
|---|---|---|
| `stats.py` | `results/stats_tests.csv`: Fisher exact test on success and Wilcoxon signed-rank test on N_HCM, both Holm-adjusted over 30 comparisons | Table 2, Section 3.2 |
| `analyze.py` | `results/summary_table.csv`, `results/summary_table.md` | Figure 1 input |
| `sens_analyze.py` | printed summary of the threshold study | Section 3.4 |
| `analyze_env2.py` | `results/env2_summary.csv`: direct Newton and CPU-matched multi-start, all from one environment | Table 4, Section 3.3 |
| `make_tables.py` | `tables/table2_success.csv`, `tables/table3_cost.csv`, `tables/table4_newton_multistart.csv` | Tables 2–4 |
| `make_figures.py` | `figures/*.png` (Figure 1 = `success_rate_summary.png`, Figure 2 = `convergence_System2_BroydenTridiagonal_n20.png`) | Figures 1–2 |
| `verify_roots.py` | roots reached on each benchmark | Table 1 |
| `check_reproducibility.py` | re-runs 25 runs and compares them with `raw_results.csv` | Section 2.5 |

## Reproducing the paper

```
# experiments (already provided in results/)
python run_experiments.py
python exp_threshold.py
python exp_multistart.py
python exp_newton.py

# analysis
python stats.py
python analyze.py
python sens_analyze.py
python analyze_env2.py
python make_tables.py
python make_figures.py
```

The analysis scripts run in seconds from the provided `results/`. Rerunning all experiments takes about one hour of CPU time on a laptop.

## Computing environments and reproducibility

The main experiment (`raw_results.csv`) was run with Python 3, NumPy 2.2 and SciPy 1.15. The threshold, multi-start and direct-Newton experiments were run with Python 3.13, NumPy 2.5 and SciPy 1.18 on a different machine. Re-running the main experiment in the second environment reproduced the metaheuristic iteration counts exactly and the success outcome in 99.2% of hybrid runs, but the homotopy iteration count in only 48%, and aggregate success rates differed by up to six percentage points (for example, Broyden n = 20 baseline 26% against 20%). The differences are attributed to floating-point differences amplified on ill-conditioned homotopy paths. Table 4 of the paper uses only second-environment results, so none of its comparisons mixes environments.

CPU times are process times and depend on the machine and the implementation: PSO is vectorised, whereas GA and GWO loop over individuals in Python.

## Seed protocol

For seed s, the hybrids draw their initial population from `numpy.random.default_rng(s)` and the Standard baseline uses the first row of that population as its start, so the four methods are paired by seed.

## Citation

See `CITATION.cff`. Please cite the paper and the archived version of this repository (Zenodo DOI).

## Licence

MIT (see `LICENSE`).
