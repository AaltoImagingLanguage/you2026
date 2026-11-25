########################################################################
# Reliability, uncertainty over participants AND stimuli, and
# cross-validated RDMs for the four-condition RSA (S8, S9 Figs).
#
# For every cortical parcel (single-trial patterns from
# 17_meg_trial_patterns.py):
#   1. Split-half reliability of each participant's MEG RDM over time
#      (random halves of the stimuli within each condition, Spearman between
#      the two half-RDMs, Spearman-Brown corrected; averaged over splits).
#   2. Cross-validated RDMs from independent STIMULUS partitions (same splits):
#      cross-validated Euclidean distance and cross-validated correlation
#      distance between the two halves. S9 Fig shows the cross-validated
#      Euclidean estimator; both are written to the summary CSV.
#   3. A crossed (two-way) bootstrap over participants and stimuli: in each
#      iteration participants are resampled, and the 135 stimuli of each
#      condition are resampled; the same stimulus sample is used for the MEG
#      and the model RDMs. Iterations use the same resamples in every parcel,
#      so whole-cortex averages have a valid bootstrap distribution too.
# The noise ceiling drawn in panel A is read from 23_rsa_condition_baselines.py.
#
# Inputs:  ../data/meg_trial_patterns/parcel*.npz, model representations
#          (../data/behavior/pcoder_reps_conds_RL23.pkl),
#          ../data/rsa_baselines/rsa_condition_baselines_RL23.nc (noise ceiling)
# Outputs: ../data/rsa_baselines/parcels_RL23/parcel*.npz (per-parcel cache)
#          ../data/rsa_baselines/rsa_{bootstrap_summary,bootstrap_parcel_counts,
#            reliability_summary,crossval_summary,crossval_clusters}_RL23.csv
#          figures/rsa_baselines/rsa_reliability_bootstrap_cv_RL23.{pdf,png}
# S9 Fig is assembled by S9Fig_reliability.py.
#
#   python 24_rsa_reliability_bootstrap_cv.py -j 6
#   python 24_rsa_reliability_bootstrap_cv.py --summary-only
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

from config import fname, subjects, time_windows
from rsa_utils import (condition_means, condition_of_item, correlation_rdm,
                          cortical_labels, crossval_correlation_rdm,
                          crossval_euclidean_rdm, label_rdms, load_parcel,
                          model_item_reps, model_rdms, parcel_adjacency,
                          partial_spearman, rank, spearman, window_labels,
                          window_mask)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hps", default="RL23", choices=["RL1", "RL23"])
parser.add_argument("-j", "--n-jobs", type=int, default=6)
parser.add_argument("--n-boot", type=int, default=1000)
parser.add_argument("--n-splits", type=int, default=100)
parser.add_argument("--n-perm", type=int, default=5000)
parser.add_argument("--summary-only", action="store_true")
parser.add_argument("--reuse-cv-clusters", action="store_true",
                    help="with --summary-only: read the saved cross-validated cluster tests instead of rerunning them")
args = parser.parse_args()

OUT_DIR = Path(f"{fname.data_dir}/rsa_baselines")
PARCEL_DIR = OUT_DIR / f"parcels_{args.hps}"
PARCEL_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR = Path(fname.figures_dir) / "rsa_baselines"
FIG_DIR.mkdir(parents=True, exist_ok=True)
SEED = 42

labels = cortical_labels()
hemis = np.array([label.hemi for label in labels])
cond = condition_of_item()
n_items = cond.size
full_weights = np.zeros((4, n_items))
full_weights[cond, np.arange(n_items)] = 1

item_reps = model_item_reps(args.hps)
item_reps = {"pcoder3": item_reps["pcoder3"]}      # the only layer whose RSA changes
base = model_rdms(item_reps)
SEV = label_rdms()["severity"]
CV_CANDIDATES = np.array([base["pcoder3 w/o"], base["pcoder3 w/"], SEV])
BOOT_MEASURES = ["w/o", "w/", "severity", "w/o | severity", "w/ | severity"]
CV_KINDS = ["stimulus cv-Euclidean", "stimulus cv-correlation"]

# --------------------------------------------------- resamples shared by parcels
rng = np.random.default_rng(SEED)
n_subjects = len(subjects)
split_weights = np.zeros((args.n_splits, 2, 4, n_items))
for k in range(args.n_splits):
    for c in range(4):
        items = rng.permutation(np.flatnonzero(cond == c))
        half = len(items) // 2
        split_weights[k, 0, c, items[:half]] = 1
        split_weights[k, 1, c, items[half:]] = 1
boot_weights = np.zeros((args.n_boot, 4, n_items), np.float32)
for b in range(args.n_boot):
    for c in range(4):
        drawn = rng.choice(np.flatnonzero(cond == c), size=(cond == c).sum(), replace=True)
        np.add.at(boot_weights[b, c], drawn, 1)
boot_subjects = rng.integers(0, n_subjects, size=(args.n_boot, n_subjects))
boot_models = [model_rdms(item_reps, w) for w in boot_weights]


def rsa_measures(rdms, m_off, m_on):
    """The bootstrap measures for data RDMs (..., 6); only the ordinal
    misspelling severity is partialled out."""
    rho = spearman(rdms, np.array([m_off, m_on, SEV]))
    partial = [partial_spearman(rdms, m, [SEV]) for m in (m_off, m_on)]
    return np.concatenate([rho, np.stack(partial, -1)], -1)


def run_parcel(parcel):
    out_path = PARCEL_DIR / f"parcel{parcel}.npz"
    if out_path.exists():
        return
    X0, valid, times = load_parcel(parcel)

    # 1 + 2: stimulus partitions
    reliability = np.zeros((n_subjects, len(times)))
    cv_stim = np.zeros((2, n_subjects, len(times), 6))
    for weights in split_weights:
        a = condition_means(X0, valid, weights[0])
        b = condition_means(X0, valid, weights[1])
        ra, rb = rank(correlation_rdm(a)), rank(correlation_rdm(b))
        ra -= ra.mean(-1, keepdims=True)
        rb -= rb.mean(-1, keepdims=True)
        r = (ra * rb).sum(-1) / np.sqrt((ra**2).sum(-1) * (rb**2).sum(-1))
        reliability += r / len(split_weights)
        cv_stim[0] += crossval_euclidean_rdm(a, b) / len(split_weights)
        cv_stim[1] += crossval_correlation_rdm(a, b) / len(split_weights)

    means = condition_means(X0, valid, full_weights)                 # (sub, 4, v, t)
    cv_rsa = spearman(cv_stim, CV_CANDIDATES)                        # (2, sub, t, cand)

    # 3: crossed bootstrap
    full = rsa_measures(correlation_rdm(means), base["pcoder3 w/o"], base["pcoder3 w/"])
    boot_two_way = np.zeros((args.n_boot, len(times), len(BOOT_MEASURES)), np.float32)
    boot_items = np.zeros_like(boot_two_way)
    for b in range(args.n_boot):
        rdms = correlation_rdm(condition_means(X0, valid, boot_weights[b]))
        per_subject = rsa_measures(rdms, boot_models[b]["pcoder3 w/o"],
                                   boot_models[b]["pcoder3 w/"])
        boot_items[b] = per_subject.mean(0)
        boot_two_way[b] = per_subject[boot_subjects[b]].mean(0)
    boot_subj = full[boot_subjects].mean(1)                          # subjects only

    np.savez(out_path, reliability=reliability, cv_rsa=cv_rsa, full=full,
             boot_two_way=boot_two_way, boot_items=boot_items, boot_subjects=boot_subj,
             times=times)
    print("parcel", parcel, "done", flush=True)


if not args.summary_only:
    Parallel(n_jobs=args.n_jobs)(delayed(run_parcel)(p) for p in range(len(labels)))

# ----------------------------------------------------------------- assemble
parcel_files = [np.load(PARCEL_DIR / f"parcel{p}.npz") for p in range(len(labels))]
times = parcel_files[0]["times"]
stack = lambda key: np.stack([f[key] for f in parcel_files])  # noqa: E731
reliability = stack("reliability")                    # (parcel, sub, t)
cv_rsa = stack("cv_rsa")                              # (parcel, kind, sub, t, cand)
full = stack("full")                                  # (parcel, sub, t, measure)
boots = {"participants": stack("boot_subjects"), "stimuli": stack("boot_items"),
         "participants + stimuli": stack("boot_two_way")}   # (parcel, B, t, measure)


def spearman_brown(r):
    return 2 * r / (1 + r)


def mean_split_half(r, axes):
    """Average split-half correlations on the Fisher-z scale. The Spearman-Brown
    correction is nonlinear, so r is averaged first and corrected once (correcting
    every noisy participant x parcel x time value would bias negative values)."""
    return np.tanh(np.arctanh(np.clip(r, -0.999, 0.999)).mean(axes))


REGIONS = {"cortex": np.arange(len(labels)), "left": np.flatnonzero(hemis == "lh"),
           "right": np.flatnonzero(hemis == "rh")}
REGION_TITLES = {"cortex": "Whole cortex (mean of 137 parcels)",
          "left": "Left hemisphere (mean of 69 parcels)",
          "right": "Right hemisphere (mean of 68 parcels)"}
m = {name: i for i, name in enumerate(BOOT_MEASURES)}
CONTRASTS = {"w/ - w/o": ("w/", "w/o"),
             "w/ - w/o | severity": ("w/ | severity", "w/o | severity"),
}

# bootstrap summaries
rows = []
for region, parcels in REGIONS.items():
    for w, wl in zip(time_windows, window_labels()):
        tm = window_mask(times, w)
        observed = full[parcels][:, :, tm].mean((0, 2)).mean(0)        # (measure,)
        for name, (a, b) in CONTRASTS.items():
            row = {"region": region, "window_ms": wl, "contrast": name,
                   "observed": observed[m[a]] - observed[m[b]]}
            for kind, boot in boots.items():
                dist = boot[parcels][:, :, tm].mean((0, 2))           # (B, measure)
                delta = dist[:, m[a]] - dist[:, m[b]]
                lo, hi = np.percentile(delta, [2.5, 97.5])
                p = min(1.0, 2 * min((delta <= 0).mean(), (delta >= 0).mean()))
                row.update({f"{kind}: ci_low": lo, f"{kind}: ci_high": hi,
                            f"{kind}: p": max(p, 1 / len(delta))})
            rows.append(row)
boot_summary = pd.DataFrame(rows)
two_way_p = "participants + stimuli: p"
boot_summary["participants + stimuli: p_fdr"] = np.nan
for _, idx in boot_summary.groupby(["region", "contrast"]).groups.items():
    boot_summary.loc[idx, "participants + stimuli: p_fdr"] = mne.stats.fdr_correction(
        boot_summary.loc[idx, two_way_p])[1]
boot_summary.to_csv(OUT_DIR / f"rsa_bootstrap_summary_{args.hps}.csv", index=False)

# per-parcel counts: two-way bootstrap CI of delta rho above / below zero
count_rows = []
for w, wl in zip(time_windows, window_labels()):
    tm = window_mask(times, w)
    for name, (a, b) in CONTRASTS.items():
        dist = boots["participants + stimuli"][:, :, tm].mean(2)
        delta = dist[..., m[a]] - dist[..., m[b]]                     # (parcel, B)
        lo, hi = np.percentile(delta, [2.5, 97.5], axis=1)
        count_rows.append({"window_ms": wl, "contrast": name,
                           "parcels_ci_above_0": int((lo > 0).sum()),
                           "parcels_ci_below_0": int((hi < 0).sum()),
                           "n_parcels": len(labels)})
pd.DataFrame(count_rows).to_csv(OUT_DIR / f"rsa_bootstrap_parcel_counts_{args.hps}.csv",
                                index=False)

# reliability summary
rel_rows = []
for region, parcels in REGIONS.items():
    for w, wl in zip(time_windows, window_labels()):
        tm = window_mask(times, w)
        r = mean_split_half(reliability[parcels][:, :, tm], (0, 2))      # per participant
        z = np.arctanh(r)
        r_grp, r_lo, r_hi = np.tanh([z.mean(), z.mean() - stats.sem(z), z.mean() + stats.sem(z)])
        rel_rows.append({"region": region, "window_ms": wl,
                         "split_half_rho": r_grp, "split_half_sem_low": r_lo, "split_half_sem_high": r_hi,
                         "spearman_brown": spearman_brown(r_grp),
                         "spearman_brown_sem_low": spearman_brown(r_lo),
                         "spearman_brown_sem_high": spearman_brown(r_hi)})
pd.DataFrame(rel_rows).to_csv(OUT_DIR / f"rsa_reliability_summary_{args.hps}.csv", index=False)

# cross-validated RDMs: paired tests across participants + cluster tests
adjacency = parcel_adjacency(labels)
cv_rows, cv_clusters = [], []
for k, kind in enumerate(CV_KINDS):
    delta = cv_rsa[:, k, :, :, 1] - cv_rsa[:, k, :, :, 0]              # (parcel, sub, t)
    for region, parcels in REGIONS.items():
        for w, wl in zip(time_windows, window_labels()):
            tm = window_mask(times, w)
            on = cv_rsa[parcels][:, k][:, :, tm, 1].mean((0, 2))
            off = cv_rsa[parcels][:, k][:, :, tm, 0].mean((0, 2))
            d = on - off
            ci = stats.t.interval(0.95, len(d) - 1, loc=d.mean(), scale=stats.sem(d))
            cv_rows.append({"rdm": kind, "region": region, "window_ms": wl,
                            "rho_w/o": off.mean(), "rho_w/": on.mean(),
                            "rho_severity": cv_rsa[parcels][:, k][:, :, tm, 2].mean(),
                            "delta": d.mean(), "ci_low": ci[0], "ci_high": ci[1],
                            "p": stats.ttest_1samp(d, 0).pvalue,
                            "d": d.mean() / np.sqrt((on.var(ddof=1) + off.var(ddof=1)) / 2)})
    family = []
    for w, wl in zip(time_windows, [] if args.reuse_cv_clusters else window_labels()):
        values = delta[:, :, window_mask(times, w)].transpose(1, 2, 0)   # (sub, t, parcel)
        _, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
            values, n_permutations=args.n_perm, tail=0, seed=SEED, adjacency=adjacency,
            out_type="indices", verbose=False, buffer_size=None)
        for c, p in zip(clusters, p_values):
            parcels = np.unique(c[1])
            family.append({"rdm": kind, "window_ms": wl, "p_cluster": p,
                           "n_parcels_lh": int((hemis[parcels] == "lh").sum()),
                           "n_parcels_rh": int((hemis[parcels] == "rh").sum()),
                           "mean_delta": values[:, c[0], c[1]].mean()})
    if family:
        for r, q in zip(family, mne.stats.fdr_correction([r["p_cluster"] for r in family])[1]):
            r["p_fdr"] = q
    cv_clusters += family
cv_summary = pd.DataFrame(cv_rows)
cv_summary["p_fdr"] = np.nan
for _, idx in cv_summary.groupby(["rdm", "region"]).groups.items():
    cv_summary.loc[idx, "p_fdr"] = mne.stats.fdr_correction(cv_summary.loc[idx, "p"])[1]
cv_summary.to_csv(OUT_DIR / f"rsa_crossval_summary_{args.hps}.csv", index=False)
if not args.reuse_cv_clusters:
    pd.DataFrame(cv_clusters).to_csv(OUT_DIR / f"rsa_crossval_clusters_{args.hps}.csv", index=False)

with pd.option_context("display.width", 200, "display.max_columns", 30):
    print(boot_summary[boot_summary.region.isin(["cortex", "left", "right"])].round(3).to_string(index=False))
    print(pd.DataFrame(rel_rows).round(3).to_string(index=False))
    print(cv_summary[cv_summary.region.isin(["cortex", "left", "right"])].round(3).to_string(index=False))

# ----------------------------------------------------------------- figure
nc_path = OUT_DIR / f"rsa_condition_baselines_{args.hps}.nc"
noise = xr.load_dataset(nc_path)["noise_ceiling"] if nc_path.exists() else None
fig, axes = plt.subplots(3, 3, figsize=(15, 12), layout="constrained")
KIND_COLORS = {"participants": "#0072B2", "stimuli": "#E69F00",
               "participants + stimuli": "k"}
for col, region in enumerate(["cortex", "left", "right"]):
    parcels = REGIONS[region]
    title = REGION_TITLES[region]
    # row 1: reliability
    ax = axes[0, col]
    z = np.arctanh(mean_split_half(reliability[parcels], 0))           # (sub, t)
    r_mean, r_lo, r_hi = (np.tanh(z.mean(0) + k * stats.sem(z)) for k in (0, -1, 1))
    ax.plot(times * 1000 + 50, spearman_brown(r_mean), color="k",
            label="within-participant split-half\n(Spearman-Brown corrected)")
    ax.fill_between(times * 1000 + 50, spearman_brown(r_lo), spearman_brown(r_hi), color="k", alpha=0.2)
    if noise is not None:
        nc = noise.isel(parcel=parcels).mean("parcel")
        ax.fill_between(times * 1000 + 50, nc.sel(bound="lower"), nc.sel(bound="upper"),
                        color="0.75", alpha=0.6, label="noise ceiling (lower-upper bound)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_title(title)
    ax.set_xlabel("Time (ms, window centre)")
    ax.set_ylabel("MEG RDM reliability" if col == 0 else "")
    # row 2: bootstrap CIs
    ax = axes[1, col]
    sub = boot_summary[boot_summary.region == region]
    for k, name in enumerate(["w/ - w/o", "w/ - w/o | severity"]):
        grp = sub[sub.contrast == name]
        for j, kind in enumerate(boots):
            x = np.arange(len(time_windows)) + (k - 0.5) * 0.36 + (j - 1) * 0.08
            ax.vlines(x, grp[f"{kind}: ci_low"], grp[f"{kind}: ci_high"],
                      color=KIND_COLORS[kind], lw=2 if j == 2 else 1.2,
                      label=f"CI: {kind}" if (k == 0) else None)
        ax.plot(np.arange(len(time_windows)) + (k - 0.5) * 0.36, grp["observed"],
                "o^"[k], color="k", mfc="w", ms=5, label=f"Δρ {name[9:] or '(raw)'}")
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(time_windows)), window_labels(), fontsize=8)
    ax.set_ylabel("PCoder 3 Δρ (w/ − w/o), 95% bootstrap CI" if col == 0 else "")
    # row 3: cross-validated RDMs (stimulus-partition cv-Euclidean estimator;
    # the cv-correlation estimator is in the summary CSV)
    ax = axes[2, col]
    sub = cv_summary[cv_summary.region == region]
    grp = sub[sub.rdm == "stimulus cv-Euclidean"]
    x = np.arange(len(time_windows))
    ax.errorbar(x, grp["delta"], yerr=[grp["delta"] - grp.ci_low, grp.ci_high - grp["delta"]],
                fmt="o", ms=5, capsize=3, color="k",
                label="cross-validated Euclidean, stimulus halves")
    for xi, (_, r) in zip(x, grp.iterrows()):
        if r.p_fdr < 0.05:
            ax.text(xi, r.ci_high + 0.01, "*", ha="center", fontsize=10)
    ax.axhline(0, color="k", lw=0.6)
    ax.set_xticks(range(len(time_windows)), window_labels(), fontsize=8)
    ax.set_xlabel("Time window (ms)")
    ax.set_ylabel("Δρ with cross-validated RDMs" if col == 0 else "")
for row in range(3):
    axes[row, 0].legend(fontsize=7.5, frameon=False, loc="upper left" if row == 2 else "best")
for ax, letter in zip(axes[:, 0], "ABC"):
    ax.text(-0.18, 1.05, letter, transform=ax.transAxes, fontsize=14, fontweight="bold")
for ext in ("pdf", "png"):
    fig.savefig(FIG_DIR / f"rsa_reliability_bootstrap_cv_{args.hps}.{ext}", dpi=300)
print("saved ->", FIG_DIR / f"rsa_reliability_bootstrap_cv_{args.hps}.pdf")
if False:  # disabled; S9 Fig is made by S9Fig_reliability.py
    from PIL import Image
    SI = Path(fname.figures_dir) / "supporting_information"
    fig.savefig(SI / "S12_Fig.tif", dpi=300, facecolor="white")
    Image.open(SI / "S12_Fig.tif").convert("RGB").save(SI / "S12_Fig.tif", dpi=(300, 300), compression="tiff_lzw")
    print("saved ->", SI / "S12_Fig.tif")
