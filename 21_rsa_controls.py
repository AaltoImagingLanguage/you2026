########################################################################
# Condition-wise RSA of the recurrent controls and the ablations, compared with the
# full predictive-coding model (S12 Fig; also the RSA of the feedforward state and the
# full model used for the cortex-average time course of Fig 5B).
#
# Model RDMs: correlation distance between the condition means of the PCoder 3
# (RNN/GRU: third recurrent site) representations at t = 25, the state used for the
# full model. The feedforward state is t = 0 (identical for all models). Random
# feedback: RSA computed for each of the five seeds and averaged. MEG RDMs and RSA
# are those of the main analysis (18_rsa_parcels.py): morphed condition patterns in
# 100-ms windows, correlation distance, Spearman RSA per participant. The script
# asserts that the full model and the feedforward state reproduce the main RDMs and
# the main RSA in parcels 40 and 65.
#
# Tests (contrast = RSA of the full model minus RSA of a comparison model):
#   figure (S12 Fig): temporal cluster test of the RSA averaged over the 137 parcels,
#     all comparisons in one FDR family with the main comparison (full model vs
#     feedforward state); computed in cortex_mean_family.py
#   whole_cortex_clusters.csv: in each 200-ms window a two-tailed spatiotemporal
#     cluster permutation test over all 137 parcels (5000 permutations, seed 42), the
#     cluster p values of all contrasts and windows in one FDR family (columns p_fdr,
#     p_fdr_main_family, p_fdr_distinct). Contrasts that are exactly zero for every
#     participant, parcel and time are not tested.
#   The ROIs (vOT, ST) are not tested.
#
# Inputs: ../data/behavior/controls_condmeans_RL23.pkl (20_controls_representations.py),
#         ../data/behavior/reconrec_{rnn,gru}_bn128_reps_conds.pkl (09_rnn_gru_representations.py),
#         ../data/meg_trial_patterns/parcel{p}.npz (17_meg_trial_patterns.py),
#         ../data/behavior/pcoder_reps_conds_RL23.pkl, ../data/rsa_tcs/rsa_tcs_parcel{40,65}_RL23.nc
# Outputs: ../data/rsa_controls/{model_rdms.pkl, control_rdms.csv, rsa_controls_parcels.nc,
#          whole_cortex_clusters.csv, cortex_mean_rsa.csv}, figures/rsa/rsa_controls.{pdf,png}
#
#   python 21_rsa_controls.py -j 8         # compute RSA, tests, tables
#   python 21_rsa_controls.py --plot_only  # figure from the saved outputs
########################################################################
import argparse
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import xarray as xr

from config import event_id, fb_timestep, fname, time_windows
from model_brain_fdr import SEED
from rsa_utils import (PAIRS, ROIS, condition_means, condition_of_item,
                          correlation_rdm, cortical_labels, load_parcel,
                          model_item_reps, model_rdms, parcel_adjacency,
                          spearman, window_labels)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--n_perm", type=int, default=5000)
parser.add_argument("-j", "--n_jobs", type=int, default=1, help="CPUs for the cluster tests")
parser.add_argument("--plot_only", action="store_true")
args = parser.parse_args()

ALPHA = 0.05
OUT_DIR = Path(f"{fname.data_dir}/rsa_controls")
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR = Path(f"{fname.figures_dir}/rsa")
CONDMEANS = f"{fname.data_dir}/behavior/controls_condmeans_RL23.pkl"
CONTROLS = ["feedforward", "rnn", "gru", "mem_only", "beta0", "alpha0", "rand_fb"]
LABELS = {
    "full PC": "full predictive-coding model",
    "feedforward": "feedforward state (t = 0)",
    "rnn": "parameter-matched RNN",
    "gru": "parameter-matched GRU",
    "mem_only": "memory only",
    "beta0": "prediction-error correction without feedback",
    "alpha0": "feedback without error correction",
    "rand_fb": "random feedback weights",
}
COLORS = {"full PC": "#000000", "feedforward": "#7f7f7f", "rnn": "#555555", "gru": "#999999",
          "mem_only": "#0072B2", "beta0": "#009E73", "alpha0": "#E69F00", "rand_fb": "#CC79A7"}
sti_types = list(event_id.keys())


def rdm_of(means):
    """(4, features) condition means -> (6,) correlation-distance RDM."""
    return correlation_rdm(means[:, :, None])[0]


def model_rdm_trajectories():
    """{model: [ (T, 6) RDMs over timesteps, one per seed ]} for layer 3."""
    blob = pickle.load(open(CONDMEANS, "rb"))
    traj = {name: [np.array([rdm_of(run["pcoder3"][:, t]) for t in blob["timesteps"]])
                   for run in runs]
            for name, runs in blob["models"].items()}
    for cell in ("rnn", "gru"):  # read out by 09_rnn_gru_representations.py at t = 0, 5, ..., 25
        rec = pickle.load(open(f"{fname.data_dir}/behavior/reconrec_{cell}_bn128_reps_conds.pkl", "rb"))
        X = np.array([np.mean(rec["rep_dict"]["rnn3"][c], 0) for c in sti_types])  # (4, n_ts, feat)
        traj[cell] = [np.array([rdm_of(X[:, i]) for i in range(len(rec["timesteps"]))])]
        traj[f"{cell}_timesteps"] = rec["timesteps"]
    traj["timesteps"] = blob["timesteps"]
    return traj


def comparison_rdms(traj):
    """RDMs entering the RSA: {model: (n_seeds, 6)} at t = fb_timestep."""
    rdms = {"full PC": traj["full PC"][0][fb_timestep][None],
            "feedforward": traj["full PC"][0][0][None]}
    for name in ("mem_only", "beta0", "alpha0", "rand_fb"):
        rdms[name] = np.array([run[fb_timestep] for run in traj[name]])
    for cell in ("rnn", "gru"):
        rdms[cell] = traj[cell][0][list(traj[f"{cell}_timesteps"]).index(fb_timestep)][None]
    return rdms


def check_full_model(rdms):
    """The recomputed full model must reproduce the RDMs of the main analysis:
    the same rank order (all the RSA uses), and values equal up to GPU
    floating-point differences (< 5e-4)."""
    main = model_rdms(model_item_reps("RL23"))
    for name, state in (("full PC", "w/"), ("feedforward", "w/o")):
        ours, theirs = rdms[name][0], main[f"pcoder3 {state}"]
        assert np.array_equal(np.argsort(ours), np.argsort(theirs)), f"{name}: rank order differs"
        assert np.allclose(ours, theirs, atol=5e-4), f"{name}: values differ"


def compute_rsa(rdms):
    """RSA of every model (seeds averaged) in every parcel:
    (parcel, subject, time, model)."""
    names = list(rdms)
    stacked = np.concatenate([rdms[n] for n in names])
    owner = np.concatenate([[i] * len(rdms[n]) for i, n in enumerate(names)])
    cond = condition_of_item()
    weights = np.zeros((4, cond.size))
    weights[cond, np.arange(cond.size)] = 1
    out = []
    for parcel in range(len(cortical_labels())):
        X0, valid, times = load_parcel(parcel)
        meg = correlation_rdm(condition_means(X0, valid, weights))       # (sub, time, 6)
        rsa = spearman(meg, stacked)                                     # (sub, time, n_rdms)
        out.append(np.stack([rsa[..., owner == i].mean(-1) for i in range(len(names))], -1))
        print(f"parcel {parcel}", end="\r", flush=True)
    return xr.DataArray(np.array(out), dims=["parcel", "subject", "time", "model"],
                        coords={"parcel": np.arange(len(out)), "time": np.round(times, 2),
                                "model": names})


def contrast(rsa, control):
    return rsa.sel(model="full PC") - rsa.sel(model=control)


def whole_cortex_tests(rsa):
    labels = cortical_labels()
    adjacency = parcel_adjacency(labels)
    left = np.array([label.hemi == "lh" for label in labels])
    family = []
    for control in CONTROLS:
        diff = contrast(rsa, control)
        for w_index, window in enumerate(time_windows):
            values = diff.sel(time=slice(*window)).transpose("subject", "time", "parcel").values
            if np.all(values == 0):
                print(f"{control}, {window}: exact zero contrast; not tested")
                continue
            _, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
                values, n_permutations=args.n_perm, tail=0, seed=SEED, adjacency=adjacency,
                out_type="indices", n_jobs=args.n_jobs, verbose=False, buffer_size=None)
            for cluster, p in zip(clusters, p_values):
                t_index, p_index = cluster
                per_subject = values[:, t_index, p_index].mean(-1)
                family.append({
                    "contrast": control, "window_ms": window_labels()[w_index],
                    "p_cluster": p, "n_parcels_lh": int(left[np.unique(p_index)].sum()),
                    "n_parcels_rh": int((~left[np.unique(p_index)]).sum()),
                    "mean_delta_rho": per_subject.mean(),
                    "d": per_subject.mean() / per_subject.std(ddof=1)})
    df = pd.DataFrame(family)
    df["p_fdr"] = mne.stats.fdr_correction(df.p_cluster, alpha=ALPHA)[1]
    # family of the main comparison only (full model vs feedforward state)
    main = df.contrast == "feedforward"
    df["p_fdr_main_family"] = np.nan
    df.loc[main, "p_fdr_main_family"] = mne.stats.fdr_correction(df.p_cluster[main], alpha=ALPHA)[1]
    # alternative: each distinct test counted once (contrasts whose comparison
    # model has the same RSA as an earlier one are identical tests)
    distinct = [c for i, c in enumerate(CONTROLS)
                if not any(np.array_equal(rsa.sel(model=c).values, rsa.sel(model=e).values)
                           for e in CONTROLS[:i])]
    df["distinct_test"] = df.contrast.isin(distinct)
    df["p_fdr_distinct"] = np.nan
    keep = df.distinct_test
    df.loc[keep, "p_fdr_distinct"] = mne.stats.fdr_correction(df.p_cluster[keep], alpha=ALPHA)[1]
    print("distinct tests:", distinct)
    return df


def rdm_table(traj, rdms):
    """Layer-3 RDMs at t = 25 and their Spearman correlation with the
    feedforward and full-model RDMs."""
    rows = []
    for name in ["full PC"] + CONTROLS:
        for seed, rdm in enumerate(rdms[name]):
            rho = spearman(rdm[None], np.concatenate([rdms["feedforward"], rdms["full PC"]]))[0]
            rows.append({"model": name, "seed": seed, **{f"{sti_types[i][:3]}-{sti_types[j][:3]}": rdm[n]
                         for n, (i, j) in enumerate(PAIRS)},
                         "rho_feedforward": rho[0], "rho_full_PC": rho[1]})
    return pd.DataFrame(rows)


if not args.plot_only:
    traj = model_rdm_trajectories()
    rdms = comparison_rdms(traj)
    check_full_model(rdms)
    pickle.dump({"traj": traj, "rdms": rdms}, open(OUT_DIR / "model_rdms.pkl", "wb"))
    rdm_table(traj, rdms).to_csv(OUT_DIR / "control_rdms.csv", index=False)
    print(rdm_table(traj, rdms).round(3).to_string(index=False))

    rsa = compute_rsa(rdms)
    rsa.to_netcdf(OUT_DIR / "rsa_controls_parcels.nc")
    # the full model and feed-forward RSA must equal those of the main analysis
    for parcel in ROIS.values():
        main = xr.load_dataarray(str(fname.rsa_tc(roi=parcel)).replace(".nc", "_RL23.nc"))
        assert np.allclose(main.isel(pcoder=2, feedback=1).values,
                           rsa.sel(parcel=parcel, model="full PC").values, atol=1e-6)
        assert np.allclose(main.isel(pcoder=2, feedback=0).values,
                           rsa.sel(parcel=parcel, model="feedforward").values, atol=1e-6)

    cortex = whole_cortex_tests(rsa)
    cortex.to_csv(OUT_DIR / "whole_cortex_clusters.csv", index=False)
    print(cortex.round(4).to_string(index=False))

    # cortex-averaged RSA per model and 200-ms window (descriptive)
    rows = []
    for w_index, window in enumerate(time_windows):
        per_subject = rsa.sel(time=slice(*window)).mean(["parcel", "time"])  # (subject, model)
        for model in rsa.model.values:
            values = per_subject.sel(model=model).values
            rows.append({"window_ms": window_labels()[w_index], "model": model,
                         "rho": values.mean(), "sem": values.std(ddof=1) / np.sqrt(values.size)})
    pd.DataFrame(rows).to_csv(OUT_DIR / "cortex_mean_rsa.csv", index=False)

# ---------------------------------------------------------------- figure
# Correction column of whole_cortex_clusters.csv in which comparisons whose RSA is
# identical to that of an earlier comparison model are counted once (p_fdr_distinct);
# "p_fdr" counts every comparison separately. The significance bars of the figure
# come from cortex_mean_family.py.
FAMILY = "p_fdr_distinct"
SHORT = {"full PC": "Full model", "feedforward": "Feedforward (t = 0)", "rnn": "RNN",
         "gru": "GRU", "mem_only": "Memory only", "beta0": "Error correction,\nno feedback",
         "alpha0": "Feedback, no\nerror correction", "rand_fb": "Random feedback\n(mean of 5 seeds)"}

cortex = pd.read_csv(OUT_DIR / "whole_cortex_clusters.csv")
rsa = xr.load_dataarray(OUT_DIR / "rsa_controls_parcels.nc").mean("parcel")  # (subject, time, model)
order = ["full PC"] + CONTROLS

# models with identical RSA are drawn as one line
groups = []
for name in order:
    for group in groups:
        if np.array_equal(rsa.sel(model=name).values, rsa.sel(model=group[0]).values):
            group.append(name)
            break
    else:
        groups.append([name])

# cortex-averaged RSA over time (mean +- SEM over participants), as in Fig 5B;
# bars: clusters of the temporal cluster test of the whole-cortex average (full model minus
# each comparison model), corrected in one family with the main comparison
# (cortex_mean_family.py; identical tests counted once). Solid bar: significant after
# FDR; light bar: p < 0.05 uncorrected only. Time = start of each 100-ms analysis window.
from cortex_mean_family import family as cortex_mean_family
fam = cortex_mean_family()
fig, ax = plt.subplots(figsize=(4.5, 3))
x = rsa.time.values * 1000  # window start, as in Fig 5
for group in groups:
    name = group[0]
    mean = rsa.sel(model=name).mean("subject")
    sem = rsa.sel(model=name).std("subject") / np.sqrt(rsa.sizes["subject"])
    color = COLORS["full PC" if "full PC" in group else "feedforward" if "feedforward" in group else name]
    label = " = ".join(SHORT[m].replace("\n", " ") for m in group)
    label = label.replace(" = GRU = ", " = GRU =\n")
    ax.plot(x, mean, color=color, lw=1.5, label=label)
    ax.fill_between(x, mean - sem, mean + sem, color=color, alpha=0.2, lw=0)
ax.axhline(0, color="k", lw=0.6)
y = ax.get_ylim()[0]
for row, contrast_group in enumerate(g for g in groups if "full PC" not in g):
    color = COLORS["feedforward" if "feedforward" in contrast_group else contrast_group[0]]
    hits = fam[fam.contrast.isin(contrast_group) & (fam.p_cluster < ALPHA)]
    for r in hits.itertuples():
        ax.plot([r.start_ms, r.end_ms], [y - 0.02 * (row + 1)] * 2, lw=3, solid_capstyle="butt",
                color=color, alpha=1.0 if r.significant_fdr else 0.35)
ax.set_xlabel("Time (ms)")
ax.set_ylabel("RSA (Spearman), mean of 137 parcels", fontsize=9)
ax.set_title("Whole cortex, PCoder 3", fontsize=10)
ax.legend(fontsize=6.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2))
ax.spines[["right", "top"]].set_visible(False)

fig.savefig(FIG_DIR / "rsa_controls.pdf", bbox_inches="tight")
fig.savefig(FIG_DIR / "rsa_controls.png", bbox_inches="tight", dpi=200)
print("saved ->", FIG_DIR / "rsa_controls.pdf")
