"""
analyze.py
==========
Generate performance comparison tables (CSV + Markdown) from
results/raw_results.csv.
"""

import os
import pandas as pd
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")
RAW_CSV = os.path.join(RESULTS_DIR, "raw_results.csv")

ALGO_ORDER = ["PSO", "GA", "GWO", "Standard"]
ALGO_LABEL = {
    "PSO": "PSO-Linear HCM",
    "GA": "GA-Linear HCM",
    "GWO": "GWO-Linear HCM",
    "Standard": "Standard Linear Newton-HCM",
}

SYSTEM_ORDER = [
    "System1_ExpTrig2D",
    "System2_BroydenTridiagonal_n5",
    "System2_BroydenTridiagonal_n10",
    "System2_BroydenTridiagonal_n20",
    "System3_ExtendedRosenbrock_n4",
    "System3_ExtendedRosenbrock_n8",
    "System3_ExtendedRosenbrock_n16",
    "System4_ExtendedPowellSingular_n4",
    "System4_ExtendedPowellSingular_n8",
    "System4_ExtendedPowellSingular_n16",
]
SYSTEM_LABEL = {
    "System1_ExpTrig2D": "System 1: Exp-Trig (n=2)",
    "System2_BroydenTridiagonal_n5": "System 2: Broyden Tridiagonal (n=5)",
    "System2_BroydenTridiagonal_n10": "System 2: Broyden Tridiagonal (n=10)",
    "System2_BroydenTridiagonal_n20": "System 2: Broyden Tridiagonal (n=20)",
    "System3_ExtendedRosenbrock_n4": "System 3: Extended Rosenbrock (n=4)",
    "System3_ExtendedRosenbrock_n8": "System 3: Extended Rosenbrock (n=8)",
    "System3_ExtendedRosenbrock_n16": "System 3: Extended Rosenbrock (n=16)",
    "System4_ExtendedPowellSingular_n4": "System 4: Extended Powell Singular (n=4)",
    "System4_ExtendedPowellSingular_n8": "System 4: Extended Powell Singular (n=8)",
    "System4_ExtendedPowellSingular_n16": "System 4: Extended Powell Singular (n=16)",
}


def summarize():
    df = pd.read_csv(RAW_CSV)
    rows = []
    for sysname in SYSTEM_ORDER:
        for algo in ALGO_ORDER:
            sub = df[(df.system == sysname) & (df.algo == algo)]
            if len(sub) == 0:
                continue
            succ = sub[sub.success == 1]
            rows.append(dict(
                system=SYSTEM_LABEL[sysname],
                n=int(sub["n"].iloc[0]),
                algo=ALGO_LABEL[algo],
                n_meta_mean=sub.n_meta.mean(),
                n_meta_std=sub.n_meta.std(ddof=0),
                n_hcm_mean=sub.n_hcm.mean(),
                n_hcm_std=sub.n_hcm.std(ddof=0),
                cpu_mean=sub.cpu_total.mean(),
                cpu_std=sub.cpu_total.std(ddof=0),
                residual_mean=succ.residual_inf.mean() if len(succ) else np.nan,
                residual_best=sub.residual_inf.min(),
                success_rate=100.0 * sub.success.mean(),
            ))
    return pd.DataFrame(rows)


def to_markdown_table(summary):
    lines = []
    lines.append("| System | Method | N_meta (mean+/-sd) | N_HCM (mean+/-sd) | CPU time (s) | ||F||inf mean (successful) | Success Rate (%) |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in summary.iterrows():
        res_str = f"{r.residual_mean:.2e}" if not np.isnan(r.residual_mean) else "--"
        lines.append(
            f"| {r.system} | {r.algo} | {r.n_meta_mean:.1f} +/- {r.n_meta_std:.1f} | "
            f"{r.n_hcm_mean:.1f} +/- {r.n_hcm_std:.1f} | {r.cpu_mean:.4f} +/- {r.cpu_std:.4f} | "
            f"{res_str} | {r.success_rate:.1f} |"
        )
    return "\n".join(lines)


if __name__ == "__main__":
    summary = summarize()
    summary.to_csv(os.path.join(RESULTS_DIR, "summary_table.csv"), index=False)
    md = to_markdown_table(summary)
    with open(os.path.join(RESULTS_DIR, "summary_table.md"), "w") as f:
        f.write(md)
    print(md)
