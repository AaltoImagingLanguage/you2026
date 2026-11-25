########################################################################
# Global prediction error and recurrent trajectory over t = 0..28, and the
# condition-wise RSA of PCoder 3 at every recurrent timestep (S10 Fig).
#
# Inputs
#   ../data/behavior/dynamics_long_RL23.npz   15_update_terms_dynamics.py (t = 0..28)
#   ../data/meg_trial_patterns/parcel*.npz    17_meg_trial_patterns.py (MEG RDMs; cached
#                                             in ../data/dynamics/meg_rdms.npz)
#
# 1 Global prediction error. E(t) = sum_i n_i eps_i(t), the summed squared prediction
#   error of the three PCoders (n_i predicted units), per stimulus; per PCoder, eps
#   against the current target and against the t = 0 target (eps_fixed), drift from
#   the t = 0 state and step size.
# 2 Trajectory. Step size, changes of the top-1 prediction and accuracy up to t = 28.
#   Condition-wise RSA of PCoder 3 at every timestep (model RDMs from the condition
#   means; MEG RDMs, Spearman RSA and windows as in the main analysis), as the cortex
#   mean per 200-ms window, Delta rho(t) = rho(t) - rho(0). Whole-cortex cluster tests
#   (two-tailed, 5000 permutations, parcel adjacency) at the prespecified t = 5, 10, 15,
#   20 and 25 versus t = 0 in each window; the correction family is formed in
#   S10Fig_rsa_timesteps.py.
#
# Outputs: ../data/dynamics/{dynamics_summary,dynamics_extra,rsa_trajectory,
#          rsa_long_cluster_tests}.csv, dynamics_curves.npz;
#          figures/dynamics/dynamics_long.pdf (diagnostic, not in the paper)
#
#   python 30_dynamics_rsa.py -j 8          # everything
#   python 30_dynamics_rsa.py --plot_only   # figure from the saved tables
########################################################################
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
from scipy import stats

from config import category_colors, fb_timestep, fname, time_windows
from model_brain_fdr import SEED
from effect_sizes import dz_ci
from rsa_utils import (condition_means, condition_of_item, correlation_rdm,
                          cortical_labels, load_parcel, model_item_reps, model_rdms,
                          parcel_adjacency, spearman, window_labels, window_mask)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--n_perm", type=int, default=5000)
parser.add_argument("-j", "--n_jobs", type=int, default=1)
parser.add_argument("--plot_only", action="store_true")
args = parser.parse_args()

ALPHA = 0.05
TEST_STEPS = [5, 10, 15, 20, 25]
REPORT_STEPS = [0, 1, 3, 5, 10, 15, 20, 25, 28]
DYN = f"{fname.data_dir}/behavior/dynamics_long_RL23.npz"
OUT_DIR = Path(f"{fname.data_dir}/dynamics")
FIG_DIR = Path(f"{fname.figures_dir}/dynamics")
OUT_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
COND_LABELS = ["RW", "RL1", "RL2", "RL3"]
WINDOW_COLORS = plt.cm.viridis(np.linspace(0, 0.9, len(time_windows)))


def rdm_of(means):
    """(4, features) condition means -> (6,) correlation-distance RDM."""
    return correlation_rdm(np.asarray(means, float)[:, :, None])[0]


# ---------------------------------------------------------------- MEG RDMs
def meg_rdms():
    """(parcel, subject, time, 6) MEG RDMs of the main analysis, cached."""
    cache = Path(f"{fname.data_dir}/dynamics/meg_rdms.npz")
    if cache.exists():
        z = np.load(cache)
        return z["rdms"], z["times"]
    cond = condition_of_item()
    weights = np.zeros((4, cond.size))
    weights[cond, np.arange(cond.size)] = 1
    out = []
    for parcel in range(len(cortical_labels())):
        X0, valid, times = load_parcel(parcel)
        out.append(correlation_rdm(condition_means(X0, valid, weights)))
        print(f"MEG RDMs parcel {parcel}", end="\r", flush=True)
    rdms, times = np.array(out), np.round(times, 2)
    cache.parent.mkdir(exist_ok=True)
    np.savez(cache, rdms=rdms, times=times)
    return rdms, times


def cortex_window_rsa(meg, times, model):
    """Cortex-mean RSA per participant and 200-ms window: (subject, window, k)."""
    masks = [window_mask(times, w) for w in time_windows]
    acc = 0
    for parcel in range(meg.shape[0]):
        rsa = spearman(meg[parcel], model)                       # (sub, time, k)
        acc = acc + np.stack([rsa[:, m].mean(1) for m in masks], 1)
    return acc / meg.shape[0]


def summary_ci(values):
    """mean, 95% t-CI and two-tailed p over participants (axis 0)."""
    n = values.shape[0]
    mean = values.mean(0)
    sem = values.std(0, ddof=1) / np.sqrt(n)
    half = stats.t.ppf(0.975, n - 1) * sem
    with np.errstate(invalid="ignore", divide="ignore"):
        p = 2 * stats.t.sf(np.abs(mean / sem), n - 1)
    return mean, mean - half, mean + half, p


# ---------------------------------------------------------------- 1-2 trajectory
def trajectory(meg, times):
    d = np.load(DYN)
    m = d["metrics"].astype(float)                               # (T, N, pcoder, metric)
    names = list(d["metric_names"])
    eps, eps_fixed = m[..., names.index("eps")], m[..., names.index("eps_fixed")]
    drift, step = m[..., names.index("drift")], m[..., names.index("step")]
    cond, T = d["condition"], m.shape[0]
    E = (eps * d["n_units"]).sum(-1)                             # (T, N)
    E_rel = eps / eps[:1]                                        # scale-free, per layer

    rows = []
    for t in [s for s in REPORT_STEPS if s < T]:
        row = {"t": t}
        for c, lab in enumerate(COND_LABELS + ["all"]):
            sel = cond == c if lab != "all" else np.ones_like(cond, bool)
            row[f"E_rel_{lab}"] = E[t, sel].mean() / E[0, sel].mean()
            row[f"E_below0_{lab}"] = (E[t, sel] < E[0, sel]).mean()
            row[f"acc_{lab}"] = d["correct"][t, sel].mean()
        row["E_scalefree_all"] = E_rel[t].mean()
        for i in range(3):
            row[f"eps{i + 1}"] = eps[t, :, i].mean()
            row[f"eps{i + 1}_fixed"] = eps_fixed[t, :, i].mean()
            row[f"drift{i + 1}"] = drift[t, :, i].mean()
            row[f"step{i + 1}"] = step[t, :, i].mean()
        if t > 0:
            row["pred_changed"] = (d["pred"][t] != d["pred"][t - 1]).mean()
        rows.append(row)
    dyn = pd.DataFrame(rows)
    changes = d["pred"][1:] != d["pred"][:-1]                    # (T-1, N)
    last_change = np.array([np.max(np.where(c)[0]) + 1 if c.any() else 0 for c in changes.T])
    extra = {"t_min_mean_E": int(np.argmin(E.mean(1))),
             "t_min_E_median_over_stimuli": float(np.median(np.argmin(E, 0))),
             "last_prediction_change_median": float(np.median(last_change)),
             "last_prediction_change_max": int(last_change.max()),
             "stimuli_changed_after_t25": float((last_change > fb_timestep).mean())}

    # PCoder 3 RDMs over timesteps; t = 0 and 25 must equal the main analysis
    cm = d["condmeans_pcoder3"]                                  # (4, T, 1000)
    rdms = np.array([rdm_of(cm[:, t]) for t in range(T)])
    main = model_rdms(model_item_reps("RL23"))
    for t, state in ((0, "w/o"), (fb_timestep, "w/")):
        assert np.array_equal(np.argsort(rdms[t]), np.argsort(main[f"pcoder3 {state}"]))
        assert np.allclose(rdms[t], main[f"pcoder3 {state}"], atol=5e-4)
    same_rank = [bool(np.array_equal(np.argsort(r), np.argsort(rdms[fb_timestep]))) for r in rdms]
    rsa = cortex_window_rsa(meg, times, rdms)                    # (sub, window, T)
    delta = rsa - rsa[..., :1]
    rsa_rows = []
    for t in range(T):
        for w, lab in enumerate(window_labels()):
            mean, lo, hi, p = summary_ci(delta[:, w, t])
            rsa_rows.append({"t": t, "window_ms": lab, "rho": rsa[:, w, t].mean(),
                             "delta_rho": mean, "ci_low": lo, "ci_high": hi, "p_uncorrected": p,
                             "same_rank_order_as_t25": same_rank[t]})
    return dyn, extra, pd.DataFrame(rsa_rows), rdms, E, eps, eps_fixed, drift, step, d


def cluster_tests(meg, times, rdms):
    """Whole-cortex cluster tests of RSA(t) - RSA(0), t in TEST_STEPS."""
    labels = cortical_labels()
    adjacency = parcel_adjacency(labels)
    left = np.array([label.hemi == "lh" for label in labels])
    steps = [t for t in TEST_STEPS if t < len(rdms)]
    rsa = np.array([spearman(meg[p], rdms[[0] + steps]) for p in range(meg.shape[0])])
    family = []                                                  # (parcel, sub, time, 1+steps)
    for si, t in enumerate(steps):
        diff = (rsa[..., si + 1] - rsa[..., 0]).transpose(1, 2, 0)  # (sub, time, parcel)
        for w, window in enumerate(time_windows):
            values = diff[:, window_mask(times, window)]
            if np.all(values == 0):
                continue
            _, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
                values, n_permutations=args.n_perm, tail=0, seed=SEED, adjacency=adjacency,
                out_type="indices", n_jobs=args.n_jobs, verbose=False, buffer_size=None)
            for (t_index, p_index), p in zip(clusters, p_values):
                per_subject = values[:, t_index, p_index].mean(-1)
                family.append({"t": t, "window_ms": window_labels()[w], "p_cluster": p,
                               "n_parcels_lh": int(left[np.unique(p_index)].sum()),
                               "n_parcels_rh": int((~left[np.unique(p_index)]).sum()),
                               "mean_delta_rho": per_subject.mean(),
                               **dict(zip(["d_z", "d_z_ci95_low", "d_z_ci95_high"],
                                          dz_ci(per_subject)))})
    df = pd.DataFrame(family)
    if len(df):
        df["p_fdr"] = mne.stats.fdr_correction(df.p_cluster, alpha=ALPHA)[1]
    return df


# ---------------------------------------------------------------- figures
def style(ax):
    ax.spines[["right", "top"]].set_visible(False)


def plot_dynamics(dyn_full, rsa_df, tests):
    E, eps, eps_fixed, drift, step, cond, correct = dyn_full
    T = E.shape[0]
    t = np.arange(T)
    # E(t) is shown in S1 Fig B (15_update_terms_dynamics.py). Top row: PCoder errors A-C; bottom row: D-G.
    fig = plt.figure(figsize=(15, 6.5))
    gs = fig.add_gridspec(2, 12)
    top = [fig.add_subplot(gs[0, 4 * k:4 * k + 4]) for k in range(3)]
    bottom = [fig.add_subplot(gs[1, 3 * k:3 * k + 3]) for k in range(4)]
    for i in range(3):
        ax = top[i]
        ax.plot(t, eps[:, :, i].mean(1), color="k", lw=1.5, label="current target $m_{i-1}(t)$")
        ax.plot(t, eps_fixed[:, :, i].mean(1), color="k", lw=1.5, ls="--",
                label="fixed target $m_{i-1}(0)$")
        ax.axvline(fb_timestep, color="grey", lw=0.6, ls="--")
        ax.set_title(f"{'ABC'[i]}  PCoder {i + 1} prediction error", loc="left", fontsize=10)
        if i == 0:
            ax.set_ylabel("MSE")
        if i == 2:
            ax.legend(frameon=False, fontsize=8)
    pc_colors = ["#0072B2", "#D55E00", "#009E73"]
    ax = bottom[0]
    for i in range(3):
        ax.plot(t, drift[:, :, i].mean(1), color=pc_colors[i], lw=1.5, label=f"PCoder {i + 1}")
    ax.set_ylabel(r"$\|m_i(t)-m_i(0)\| / \|m_i(0)\|$")
    ax.set_title("D  Distance from feedforward state", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8)
    ax = bottom[1]
    for i in range(3):
        ax.semilogy(t[1:], step[1:, :, i].mean(1), color=pc_colors[i], lw=1.5, label=f"PCoder {i + 1}")
    ax.set_ylabel(r"$\|m_i(t)-m_i(t-1)\| / \|m_i(t-1)\|$")
    ax.set_title("E  Step size", loc="left", fontsize=10)
    ax = bottom[2]
    for c in range(4):
        ax.plot(t, correct[:T, cond == c].mean(1), color=category_colors[c], lw=1.5, label=COND_LABELS[c])
    ax.axvline(fb_timestep, color="grey", lw=0.6, ls="--")
    ax.set_ylabel("Top-1 accuracy")
    ax.set_title("F  Classification", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right", bbox_to_anchor=(1, 0.2))
    ax = bottom[3]
    for w, lab in enumerate(window_labels()):
        sub = rsa_df[rsa_df.window_ms == lab]
        ax.plot(sub.t, sub.delta_rho, color=WINDOW_COLORS[w], lw=1.5, label=f"{lab} ms")
        ax.fill_between(sub.t, sub.ci_low, sub.ci_high, color=WINDOW_COLORS[w], alpha=0.15, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.axvline(fb_timestep, color="grey", lw=0.6, ls="--")
    ax.set_ylabel(r"Cortex $\Delta\rho$ = RSA(t) $-$ RSA(0)")
    ax.set_title("G  Condition-wise RSA, PCoder 3", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=7, ncol=1)
    for ax in top + bottom:
        ax.set_xlabel("Timestep")
        style(ax)
    fig.tight_layout()
    for ext in ("pdf", "png", "tif"):
        kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
        fig.savefig(FIG_DIR / f"dynamics_long.{ext}", bbox_inches="tight", dpi=300, **kw)
    print("saved ->", FIG_DIR / f"dynamics_long.pdf")


# ---------------------------------------------------------------- main
if not args.plot_only:
    meg, times = meg_rdms()
    dyn, extra, rsa_df, rdms, E, eps, eps_fixed, drift, step, d = trajectory(meg, times)
    dyn.to_csv(OUT_DIR / "dynamics_summary.csv", index=False)
    pd.Series(extra).to_csv(OUT_DIR / "dynamics_extra.csv")
    rsa_df.to_csv(OUT_DIR / "rsa_trajectory.csv", index=False)
    np.savez_compressed(OUT_DIR / "dynamics_curves.npz", E=E, eps=eps.mean(1), eps_fixed=eps_fixed.mean(1),
                        drift=drift.mean(1), step=step.mean(1), condition=d["condition"],
                        acc=np.array([d["correct"][:, d["condition"] == c].mean(1) for c in range(4)]))
    with pd.option_context("display.width", 250, "display.max_columns", 60):
        print(dyn.round(4).to_string(index=False))
        print(extra)
        print(rsa_df[rsa_df.t.isin(REPORT_STEPS)].round(3).to_string(index=False))
    tests = cluster_tests(meg, times, rdms)
    tests.to_csv(OUT_DIR / "rsa_long_cluster_tests.csv", index=False)
    print(tests.round(4).to_string(index=False))
    plot_dynamics((E, eps, eps_fixed, drift, step, d["condition"], d["correct"]), rsa_df, tests)

else:
    rsa_df = pd.read_csv(OUT_DIR / "rsa_trajectory.csv")
    z = np.load(OUT_DIR / "dynamics_curves.npz")
    # curves were saved as means over stimuli; rebuild per-condition E from saved arrays
    d = np.load(DYN)
    m = d["metrics"].astype(float)
    names = list(d["metric_names"])
    eps = m[..., names.index("eps")]
    E = (eps * d["n_units"]).sum(-1)
    plot_dynamics((E, eps, m[..., names.index("eps_fixed")], m[..., names.index("drift")],
                   m[..., names.index("step")], d["condition"], d["correct"]),
                  rsa_df, pd.read_csv(OUT_DIR / "rsa_long_cluster_tests.csv"))
