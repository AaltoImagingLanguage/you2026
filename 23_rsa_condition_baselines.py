########################################################################
# Condition-label baselines, severity-partialled RSA and noise ceiling
# (S6, S8 Figs).
#
# A 6-entry condition RDM may mainly capture a broad ordering (RW-RL1-RL2-RL3).
# This script
#   1. compares the model RDMs (w/o and w/ feedback) with two condition-label
#      RDMs: the four-level misspelling-severity RDM (|difference in replaced
#      letters|), i.e. the ordinal condition structure, and the
#      lexicality RDM (real word vs misspelled word);
#   2. computes RSA of the model and label RDMs with the MEG RDMs of all
#      137 cortical parcels (descriptive: mean and 95% CI, no tests);
#   3. tests whether the predictive-coding advantage (PCoder 3, w/ - w/o)
#      remains after partialling out each label RDM in turn (partial Spearman,
#      computed from the stored MEG RDMs); whole-cortex cluster tests per
#      200-ms window (two-tailed, 5000 permutations), BH-FDR across the
#      clusters of the five windows within each contrast;
#   4. computes the between-participant noise ceiling (Nili et al., 2014) of
#      the MEG RDMs over time.
#
# MEG RDMs are built from anatomically matched (fsaverage-morphed) patterns,
# i.e. condition means of the single-trial patterns from
# 17_meg_trial_patterns.py.
#
# Inputs:  ../data/meg_trial_patterns/parcel*.npz (17_meg_trial_patterns.py),
#          ../data/behavior/pcoder_reps_conds_RL23.pkl
# Outputs: ../data/rsa_baselines/rsa_condition_baselines_RL23.nc
#          ../data/rsa_baselines/rsa_condition_baselines_{summary,clusters}_RL23.csv
#          ../data/rsa_baselines/model_vs_label_rdms_RL23.csv
#          figures/rsa_baselines/rsa_condition_baselines_RL23.{pdf,png}
# The tables and figures are assembled by 25_rsa_baseline_tables.py
# and S6Fig_S8Fig_rsa_baselines.py.
#
#   python 23_rsa_condition_baselines.py            # compute + stats + figure
#   python 23_rsa_condition_baselines.py --plot-only
########################################################################
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from scipy import stats

from config import fname, time_windows
from rsa_utils import (CONDITIONS, condition_means, condition_of_item,
                          correlation_rdm, cortical_labels, label_rdms,
                          load_parcel, model_item_reps, model_rdms, noise_ceiling,
                          parcel_adjacency, partial_spearman, spearman, window_labels, window_mask)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hps", default="RL23", choices=["RL1", "RL23"])
parser.add_argument("-j", "--n-jobs", type=int, default=6)
parser.add_argument("--plot-only", action="store_true")
parser.add_argument("--n-perm", type=int, default=5000)
args = parser.parse_args()

OUT_DIR = Path(f"{fname.data_dir}/rsa_baselines")
OUT_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(fname.figures_dir) / "rsa_baselines"
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULT = OUT_DIR / f"rsa_condition_baselines_{args.hps}.nc"
SEED = 42

labels = cortical_labels()
cond = condition_of_item()
full_weights = np.zeros((4, cond.size))
full_weights[cond, np.arange(cond.size)] = 1

model = model_rdms(model_item_reps(args.hps))
baselines = label_rdms()
candidates = {**model, **baselines}
cand_names = list(candidates)
cand_matrix = np.array([candidates[k] for k in cand_names])
P3_OFF, P3_ON = model["pcoder3 w/o"], model["pcoder3 w/"]
COVARIATES = {"severity": [baselines["severity"]], "lexicality": [baselines["lexicality"]]}


def parcel_rsa(parcel):
    X0, valid, times = load_parcel(parcel)
    rdms = correlation_rdm(condition_means(X0, valid, full_weights))   # (sub, t, 6)
    rho = spearman(rdms, cand_matrix)                                   # (sub, t, k)
    lower, upper = noise_ceiling(rdms)
    return rho, np.stack([lower, upper]), rdms, times


if not args.plot_only:
    results = Parallel(n_jobs=args.n_jobs)(delayed(parcel_rsa)(p) for p in range(len(labels)))
    times = results[0][3]
    subjects = np.load(Path(fname.data_dir) / "meg_trial_patterns" / "parcel0.npz")["subjects"]
    ds = xr.Dataset(
        {"rsa": (("parcel", "subject", "time", "measure"), np.stack([r[0] for r in results])),
         "noise_ceiling": (("parcel", "bound", "time"), np.stack([r[1] for r in results])),
         "meg_rdm": (("parcel", "subject", "time", "pair"), np.stack([r[2] for r in results]))},
        coords={"parcel": np.arange(len(labels)), "subject": subjects, "time": times,
                "measure": cand_names, "bound": ["lower", "upper"],
                "pair": ["RW-RL1", "RW-RL2", "RW-RL3", "RL1-RL2", "RL1-RL3", "RL2-RL3"]},
        attrs={"hps": args.hps})
    ds.to_netcdf(RESULT)
    print("saved ->", RESULT)
ds = xr.load_dataset(RESULT)
# partial RSA of PCoder 3, one covariate at a time: (parcel, subject, time)
partial = {f"pcoder3 {s} | {c}": partial_spearman(ds["meg_rdm"].values, rdm, cov)
           for c, cov in COVARIATES.items() for s, rdm in (("w/o", P3_OFF), ("w/", P3_ON))}
partial = xr.DataArray(np.stack(list(partial.values()), -1),
                       dims=("parcel", "subject", "time", "measure"),
                       coords={**{d: ds[d] for d in ("parcel", "subject", "time")},
                               "measure": list(partial)})
rsa = xr.concat([ds["rsa"].sel(measure=cand_names), partial], "measure")
times = ds.time.values
hemis = np.array([label.hemi for label in labels])

# ---------------------------------------------------------------- 1. model vs labels
order = ["pcoder3 w/o", "pcoder3 w/", "severity", "lexicality"]
rdm_corr = pd.DataFrame(spearman(np.array([candidates[k] for k in order]),
                                 np.array([candidates[k] for k in order])),
                        index=order, columns=order)
rdm_corr.to_csv(OUT_DIR / f"model_vs_label_rdms_{args.hps}.csv")
print("\nSpearman correlations between candidate RDMs\n", rdm_corr.round(3))

# ---------------------------------------------------------------- contrasts
CONTRASTS = {
    "w/ - w/o": ("pcoder3 w/", "pcoder3 w/o"),
    "w/ - w/o | severity": ("pcoder3 w/ | severity", "pcoder3 w/o | severity"),
    "w/ - w/o | lexicality": ("pcoder3 w/ | lexicality", "pcoder3 w/o | lexicality"),
}


def summarize(a, b):
    """Paired summary across participants: mean [95% CI], t, p, Cohen's d
    (mean difference / pooled SD of the two states)."""
    d = a - b
    ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
    t, p = stats.ttest_1samp(d, 0)
    pooled = np.sqrt((a.var(ddof=1) + b.var(ddof=1)) / 2)
    return {"mean": d.mean(), "ci_low": ci[0], "ci_high": ci[1], "t": t, "p": p,
            "d": d.mean() / pooled if pooled > 0 else np.nan}


# Regions: whole cortex (mean over all 137 parcels), each hemisphere, ROIs.
REGIONS = {"cortex": np.arange(len(labels)), "left": np.flatnonzero(hemis == "lh"),
           "right": np.flatnonzero(hemis == "rh")}
REGION_TITLES = {"cortex": "Whole cortex (mean of 137 parcels)",
          "left": "Left hemisphere (mean of 69 parcels)",
          "right": "Right hemisphere (mean of 68 parcels)"}
rows = []
for region, parcels in REGIONS.items():
    region_rsa = rsa.isel(parcel=parcels).mean("parcel")
    for w, wl in zip(time_windows, window_labels()):
        sel = region_rsa.isel(time=window_mask(times, w)).mean("time")
        for name, (m_a, m_b) in CONTRASTS.items():
            rows.append({"region": region, "window_ms": wl, "contrast": name,
                         **summarize(sel.sel(measure=m_a).values, sel.sel(measure=m_b).values)})
        for m in order + list(partial.measure.values):
            v = sel.sel(measure=m).values
            ci = stats.t.interval(0.95, len(v) - 1, loc=v.mean(), scale=stats.sem(v))
            rows.append({"region": region, "window_ms": wl, "contrast": f"rho {m}",
                         "mean": v.mean(), "ci_low": ci[0], "ci_high": ci[1],
                         "t": np.nan, "p": np.nan, "d": np.nan})
        nc = ds["noise_ceiling"].isel(parcel=parcels).mean("parcel").isel(
            time=window_mask(times, w)).mean("time")
        rows.append({"region": region, "window_ms": wl, "contrast": "noise ceiling",
                     "mean": np.nan, "ci_low": float(nc.sel(bound="lower")),
                     "ci_high": float(nc.sel(bound="upper"))})
summary = pd.DataFrame(rows)
# FDR across the 5 windows within each region x contrast family
summary["p_fdr"] = np.nan
for (region, contrast), idx in summary.dropna(subset=["p"]).groupby(["region", "contrast"]).groups.items():
    summary.loc[idx, "p_fdr"] = mne.stats.fdr_correction(summary.loc[idx, "p"])[1]
summary.to_csv(OUT_DIR / f"rsa_condition_baselines_summary_{args.hps}.csv", index=False)
print("saved ->", OUT_DIR / f"rsa_condition_baselines_summary_{args.hps}.csv")

# ---------------------------------------------------------------- cluster tests
# Same procedure as the whole-cortex analysis (per window, all parcels,
# two-tailed, 5000 permutations), FDR across windows within each contrast.
adjacency = parcel_adjacency(labels)

cluster_rows = []
for name, (m_a, m_b) in CONTRASTS.items():
    contrast = (rsa.sel(measure=m_a) - rsa.sel(measure=m_b)).transpose("subject", "time", "parcel")
    family = []
    for w, wl in zip(time_windows, window_labels()):
        values = contrast.isel(time=window_mask(times, w)).values
        _, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
            values, n_permutations=args.n_perm, tail=0, seed=SEED, adjacency=adjacency,
            out_type="indices", verbose=False, buffer_size=None)
        for c, p in zip(clusters, p_values):
            family.append({"contrast": name, "window_ms": wl, "p_cluster": p,
                           "n_parcels_lh": int((hemis[np.unique(c[1])] == "lh").sum()),
                           "n_parcels_rh": int((hemis[np.unique(c[1])] == "rh").sum()),
                           "sign": np.sign(values[:, c[0], c[1]].mean()),
                           "mean_delta": values[:, c[0], c[1]].mean()})
    if family:
        fdr = mne.stats.fdr_correction([r["p_cluster"] for r in family])[1]
        for r, q in zip(family, fdr):
            r["p_fdr"] = q
    cluster_rows += family
clusters = pd.DataFrame(cluster_rows)
clusters.to_csv(OUT_DIR / f"rsa_condition_baselines_clusters_{args.hps}.csv", index=False)
sig = clusters[clusters.p_fdr < 0.05] if len(clusters) else clusters
print("\nsignificant clusters (FDR < .05)\n", sig.to_string(index=False))

# ---------------------------------------------------------------- figure
COLORS = {"pcoder3 w/o": "#56B4E9", "pcoder3 w/": "#D55E00", "severity": "0.35",
          "lexicality": "#009E73"}
# panel B shows severity only; the lexicality-partialled results are written to the CSV tables of 25_rsa_baseline_tables.py (not reported)
DELTA_STYLE = {"w/ - w/o": ("k", "k", "Δρ"),
               "w/ - w/o | severity": ("0.55", "w", "Δρ, severity partialled out")}
fig = plt.figure(figsize=(15, 10.5), layout="constrained")
top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.5])
rdm_axes = top.subplots(1, len(order) + 1, width_ratios=[1] * len(order) + [1.5])
for ax, name in zip(rdm_axes, order):
    square = np.zeros((4, 4))
    square[np.triu_indices(4, 1)] = candidates[name]
    square = square + square.T
    ax.imshow(square / square.max(), cmap="viridis", vmin=0, vmax=1)
    ax.set_title(name.replace("pcoder3", "PCoder 3"), fontsize=11)
    ax.set_xticks(range(4), [c[:3] for c in CONDITIONS], fontsize=8, rotation=90)
    ax.set_yticks(range(4), [c[:3] for c in CONDITIONS], fontsize=8)
im = rdm_axes[-1].imshow(rdm_corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
short = ["PC3 w/o", "PC3 w/", "severity", "lexicality"]
rdm_axes[-1].set_xticks(range(len(order)), short, rotation=90, fontsize=8)
rdm_axes[-1].set_yticks(range(len(order)), short, fontsize=8)
for (i, j), v in np.ndenumerate(rdm_corr.values):
    rdm_axes[-1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                      color="w" if abs(v) > 0.6 else "k")
rdm_axes[-1].set_title("RDM correlation (Spearman)", fontsize=11)
top.suptitle("A  Model, misspelling-severity and lexicality RDMs", x=0.01, ha="left", fontweight="bold")

axes = bottom.subplots(2, 3, sharex="row")
for col, region in enumerate(["cortex", "left", "right"]):
    parcels = REGIONS[region]
    reg = rsa.isel(parcel=parcels).mean("parcel")
    nc = ds["noise_ceiling"].isel(parcel=parcels).mean("parcel")
    ax = axes[0, col]
    ax.fill_between(times * 1000 + 50, nc.sel(bound="lower"), nc.sel(bound="upper"),
                    color="0.85", label="noise ceiling (raw RSA)")
    # panel B: PCoder 3 RSA before (solid) and after (dashed) partialling out
    # severity; the RSA of the label RDMs themselves is written by 25_rsa_baseline_tables.py (not reported)
    for m in ["pcoder3 w/o", "pcoder3 w/", "pcoder3 w/o | severity", "pcoder3 w/ | severity"]:
        mean = reg.sel(measure=m).mean("subject")
        sem = reg.sel(measure=m).std("subject") / np.sqrt(reg.sizes["subject"])
        color = COLORS[m.split(" | ")[0]]
        label = (f"{m} RDM" if m in baselines else
                 m.replace("pcoder3", "PCoder 3").replace(" | severity", ", severity partialled out"))
        ls = ":" if m in baselines else "--" if "|" in m else "-"
        ax.plot(times * 1000 + 50, mean, color=color, ls=ls, label=label)
        ax.fill_between(times * 1000 + 50, mean - sem, mean + sem, color=color, alpha=0.2)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(REGION_TITLES[region])
    ax.set_ylabel("RSA (Spearman)" if col == 0 else "")
    ax.set_xlabel("Time (ms, window centre)")
    ax = axes[1, col]
    sub = summary[(summary.region == region) & summary.contrast.isin(DELTA_STYLE)]
    for k, (name, grp) in enumerate(sub.groupby("contrast", sort=False)):
        x = np.arange(len(time_windows)) + (k - 0.5) * 0.25
        color, face, label = DELTA_STYLE[name]
        ax.errorbar(x, grp["mean"], yerr=[grp["mean"] - grp.ci_low, grp.ci_high - grp["mean"]],
                    fmt="o", ms=4, capsize=2, color=color, mfc=face, label=label)
        for xi, (_, r) in zip(x, grp.iterrows()):
            if r.p_fdr < 0.05:
                ax.text(xi, r.ci_high + 0.02, "*", ha="center", fontsize=10)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(time_windows)), window_labels(), fontsize=8)
    ax.set_xlabel("Time window (ms)")
    ax.set_ylabel("PCoder 3 Δρ (w/ − w/o)" if col == 0 else "")
axes[0, 0].legend(fontsize=8, frameon=False)
axes[1, 0].legend(fontsize=7.5, frameon=False)
bottom.suptitle("B  RSA time courses with noise ceiling, and the feedback effect before/after partialling out misspelling severity",
                x=0.01, ha="left", fontweight="bold")
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"rsa_condition_baselines_{args.hps}.{ext}", dpi=300)
print("saved ->", FIG_DIR / f"rsa_condition_baselines_{args.hps}.pdf")
