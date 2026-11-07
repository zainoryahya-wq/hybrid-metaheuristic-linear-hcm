"""
make_figures.py
===============
Generate Convergence Curve and Homotopy Path Tracking (t in [0,1]) plots
from results/curves.pkl.
"""

import os
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
FIG_DIR = os.path.join(os.path.dirname(__file__), "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(RESULTS_DIR, "curves.pkl"), "rb") as f:
    curves = pickle.load(f)

ALGO_ORDER = ["Standard", "PSO", "GA", "GWO"]   # baseline first, then hybrids in ascending order of overall mean success rate
ALGO_LABEL = {
    "PSO": "PSO-Linear HCM",
    "GA": "GA-Linear HCM",
    "GWO": "GWO-Linear HCM",
    "Standard": "Standard Linear Newton-HCM",
}
ALGO_COLOR = {"PSO": "#1f77b4", "GA": "#2ca02c", "GWO": "#d62728", "Standard": "#7f7f7f"}

SYSTEMS = sorted(set(k[0] for k in curves.keys()))


def plot_convergence(sysname):
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for algo in ALGO_ORDER:
        key = (sysname, algo, 0)
        if key not in curves:
            continue
        c = curves[key]
        t = np.array(c["conv_time"])
        e = np.array(np.maximum(c["conv_error"], 1e-16))
        ax.semilogy(t, e, label=ALGO_LABEL[algo], color=ALGO_COLOR[algo], linewidth=1.6)
    ax.axhline(1e-6, color="k", linestyle=":", linewidth=1, label="Stopping criterion (1e-6)")
    ax.set_xlabel("CPU time (s)")
    ax.set_ylabel("Error norm  ||F(x)||  (log scale)")
    ax.set_title(f"Convergence Curve — {sysname}")
    ax.legend(fontsize=8)
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(FIG_DIR, f"convergence_{sysname}.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


def plot_homotopy_path(sysname):
    fig, axes = plt.subplots(1, len(ALGO_ORDER), figsize=(16, 3.8), sharey=False)
    for ax, algo in zip(axes, ALGO_ORDER):
        key = (sysname, algo, 0)
        if key not in curves:
            ax.set_visible(False)
            continue
        c = curves[key]
        t_path = np.array(c["t_path"])
        x_path = np.array(c["x_path"])  # shape (len(t), n)
        n_show = min(x_path.shape[1], 6)  # avoid overcrowding the plot for large n
        for j in range(n_show):
            ax.plot(t_path, x_path[:, j], linewidth=1.3, label=f"$x_{{{j+1}}}$")
        ax.set_xlabel("t")
        ax.set_title(ALGO_LABEL[algo], fontsize=10)
        if n_show <= 6:
            ax.legend(fontsize=6, ncol=2)
    axes[0].set_ylabel("Component x(t)")
    fig.suptitle(f"Homotopy Path Tracking — {sysname}", y=1.03)
    fig.tight_layout()
    fname = os.path.join(FIG_DIR, f"homotopy_path_{sysname}.png")
    fig.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return fname


def wilson(k, n, z=1.959964):
    p = k / n; d = 1 + z * z / n; c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return 100 * (c - h), 100 * (c + h)


def plot_success_rate_summary(summary_csv):
    import pandas as pd
    raw = pd.read_csv(os.path.join(RESULTS_DIR, "raw_results.csv"))
    systems = list(dict.fromkeys(raw["system"]))
    labels = {s: s.replace("System1_ExpTrig2D", "S1 Exp-Trig n=2")
              .replace("System2_BroydenTridiagonal_n", "S2 Broyden n=")
              .replace("System3_ExtendedRosenbrock_n", "S3 Ext. Rosenbrock n=")
              .replace("System4_ExtendedPowellSingular_n", "S4 Ext. Powell n=") for s in systems}
    x = np.arange(len(systems))
    width = 0.2
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, algo in enumerate(ALGO_ORDER):
        vals, lo, hi = [], [], []
        for s_ in systems:
            d = raw[(raw.system == s_) & (raw.algo == algo)]
            k, n = int(d.success.sum()), len(d)
            a, b = wilson(k, n)
            vals.append(100.0 * k / n); lo.append(max(0.0, 100.0 * k / n - a)); hi.append(max(0.0, b - 100.0 * k / n))
        ax.bar(x + (i - 1.5) * width, vals, width, label=ALGO_LABEL[algo], color=ALGO_COLOR[algo],
               yerr=[lo, hi], capsize=2, error_kw=dict(linewidth=0.8))
    ax.set_xticks(x)
    ax.set_xticklabels([labels[s_] for s_ in systems], rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Success rate (%)  [50 runs; bars: 95% Wilson interval]")
    ax.set_ylim(0, 105)
    ax.set_title("Convergence success rate across benchmark systems")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fname = os.path.join(FIG_DIR, "success_rate_summary.png")
    fig.savefig(fname, dpi=150)
    plt.close(fig)
    return fname


if __name__ == "__main__":
    made = []
    for sysname in SYSTEMS:
        made.append(plot_convergence(sysname))
        made.append(plot_homotopy_path(sysname))
    made.append(plot_success_rate_summary(os.path.join(RESULTS_DIR, "summary_table.csv")))
    for f in made:
        print("Saved:", f)
