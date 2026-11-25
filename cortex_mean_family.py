"""Correction family of the whole-cortex AVERAGE RSA time course (PCoder 3).

Used by Fig 5B (Fig5_Fig6_compose.py) and S12 Fig (21_rsa_controls.py). With the
reported models the family has m = 2 clusters.
Each test: two-tailed paired temporal cluster permutation test over participants of
RSA(full model) - RSA(comparison model), averaged over all 137 parcels, at every
100-ms analysis window (20-ms steps; no further averaging over time), 5000
permutations, seed 42. The comparisons of
the full model with every recurrent control and ablation form ONE family with the
main comparison (full model vs feedforward state). Comparison models whose RSA is
identical to that of an earlier one give the same test and are counted once;
contrasts that are zero for every participant (error correction without
feedback) are not tests and are not counted (Tarone, 1990; Gilbert, 2005).
BH-FDR over ALL clusters of the tests (the rule of the whole-cortex and ROI
families, model_brain_fdr.py); a cluster is significant after correction if its p
is at or below the largest p rejected in the family.

Input:  ../data/rsa_controls/rsa_controls_parcels.nc (21_rsa_controls.py)
Output (cached): ../data/rsa_controls/cortex_mean_family.csv, one row per cluster.

    python cortex_mean_family.py     # recompute the family and print it
"""
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import xarray as xr
from statsmodels.stats.multitest import multipletests

from config import fname
from model_brain_fdr import N_PERMUTATIONS, SEED

DIR = Path(fname.data_dir) / "rsa_controls"
CACHE = DIR / "cortex_mean_family.csv"
ORDER = ["feedforward", "rnn", "gru", "mem_only", "alpha0", "beta0", "rand_fb"]


def cortex_mean_rsa():
    """(subject, time, model) RSA averaged over the 137 parcels."""
    return xr.load_dataarray(DIR / "rsa_controls_parcels.nc").mean("parcel")


def family(recompute=False):
    if CACHE.exists() and not recompute:
        return pd.read_csv(CACHE)
    rsa = cortex_mean_rsa()
    times = rsa.time.values * 1000
    controls = [m for m in ORDER if m in rsa.model.values]
    rows, cells = [], {}
    for i, c in enumerate(controls):
        same = [e for e in controls[:i] if np.array_equal(rsa.sel(model=c).values, rsa.sel(model=e).values)]
        diff = (rsa.sel(model="full PC") - rsa.sel(model=c)).values
        if same:
            rows.append({"contrast": c, "status": f"identical to {same[0]} (same test, counted once)"})
            continue
        if np.all(diff == 0):
            rows.append({"contrast": c, "status": "identical to the full model (not a test)"})
            continue
        _, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
            diff, n_permutations=N_PERMUTATIONS, tail=0, seed=SEED, verbose=False)
        cells[c] = float(np.min(pvals)) if len(pvals) else 1.0
        for cl, pv in zip(clusters, pvals):
            rows.append({"contrast": c, "status": "tested", "start_ms": times[cl][0], "end_ms": times[cl][-1],
                         "p_cluster": pv,
                         "mean_delta_rho": diff[:, cl[0]].mean()})
        if not len(pvals):
            rows.append({"contrast": c, "status": "tested", "p_cluster": np.nan})
    df = pd.DataFrame(rows)
    tested = df.p_cluster.notna()
    reject, adjusted, _, _ = multipletests(df.p_cluster[tested], alpha=0.05, method="fdr_bh")
    threshold = float(df.p_cluster[tested][reject].max()) if reject.any() else 0.0
    df["m"] = int(tested.sum())          # clusters in the family
    df["n_tests"] = len(cells)
    df["test_p"] = df.contrast.map(cells)  # smallest cluster p of the test
    df.loc[tested, "p_fdr"] = adjusted
    df["threshold"] = threshold
    df["significant_fdr"] = df.p_cluster <= threshold
    df.to_csv(CACHE, index=False)
    return df


if __name__ == "__main__":
    with pd.option_context("display.width", 200):
        print(family(recompute=True).round(4).to_string(index=False))
