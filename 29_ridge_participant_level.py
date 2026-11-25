########################################################################
# Participant-level ridge regression (S15 Fig, S4 Table).
#
# The participant is the unit of analysis:
#   (A) subject-level: the ridge regression is fitted per participant (predicting
#       that participant's per-stimulus ROI response in each 100-ms window),
#       10-fold CV across stimuli -> one score per participant, PCoder, state
#       (w/o feedback t = 0, w/ feedback t = 25) and window. Stimulus
#       generalization comes from the held-out CV, population inference from the
#       test across the 23 participants.
#   (B) participant bootstrap: the grand-average target is kept, participants are
#       resampled with replacement, the ridge is refitted and percentile CIs are
#       taken for the score and the feedback difference.
# The reported two-tailed cluster tests and their FDR family (ROI ridge, m = 6
# tests: vOT/ST x PCoders 1-3) are computed from the saved scores by
# model_brain_fdr.py and 22_effect_sizes_clusters.py; S15 Fig is drawn by
# S15Fig_ridge_participant_level.py. This script prints one-tailed cluster tests
# and the bootstrap CIs and saves its own summary figure.
#
# Inputs:  ../data/meg_tcs/meg_tc_{40,65}_sti.nc (subject, 540, time;
#          28_ridge_grand_average.py --compute True)
#          ../data/behavior/pcoder_reps_conds_RL23.pkl, ../data/dataset/rep_stimuli.csv
# Outputs: ../data/behavior/ridge_subjinf_{vOT,ST}_RL23.pkl
#          ../data/behavior/ridge_subjinf_{vOT,ST}_RL23_boot_ckpt.pkl (checkpoint)
#          figures/ridge/ridge_subjinf_{vOT,ST}_RL23.{png,pdf,tif}
#
#   python 29_ridge_participant_level.py --parcel_ind 40 --hps RL23 --n_boot 500 -j 8
#   python 29_ridge_participant_level.py --parcel_ind 65 --hps RL23 --n_boot 500 -j 8
#   python 29_ridge_participant_level.py --parcel_ind 65 --hps RL23 --n_boot 500 -j 8 \
#       --resume_bootstrap --boot_chunk 8      # skip (A), continue (B)
########################################################################
import argparse
import os
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import mne
from joblib import Parallel, delayed

from ridge import Ridge
from config import (fname, subjects, time_len, time_step,
                    time_interval, n_splits, nc, fb_timestep)

pa = argparse.ArgumentParser(description=__doc__)
pa.add_argument("--parcel_ind", type=int, default=65, help="65=ST, 40=vOT")
pa.add_argument("--hps", type=str, default="RL23", choices=["RL1", "RL23"])
pa.add_argument("--n_boot", type=int, default=2000, help="participant bootstraps (0 to skip)")
pa.add_argument("--boot_chunk", type=int, default=8,
                help="save bootstrap checkpoint after this many resamples")
pa.add_argument("-j", "--n_jobs", type=int, default=8)
pa.add_argument("--plot_only", action="store_true")
pa.add_argument(
    "--resume_bootstrap",
    action="store_true",
    help="load the saved subject-level result and skip analysis A, then run "
         "analysis B (participant bootstrap)",
)
args, _ = pa.parse_known_args()
args.boot_chunk = max(1, args.boot_chunk)
roi_id = args.parcel_ind
roi_name = {40: "vOT", 65: "ST"}[roi_id]
suf = "_RL23" if args.hps == "RL23" else ""
REPS = (str(fname.pcoder_reps).replace(".pkl", "_RL23.pkl")
        if args.hps == "RL23" else fname.pcoder_reps)
OUT = f"{fname.data_dir}/behavior/ridge_subjinf_{roi_name}{suf}.pkl"
BOOT_CKPT = f"{fname.data_dir}/behavior/ridge_subjinf_{roi_name}{suf}_boot_ckpt.pkl"
STATES = ["w/o", "w/"]                     # timestep 0 (feed-forward) vs fb_timestep (converged)
STATE_TS = [0, fb_timestep]
FB_COLORS = ["lightblue", "lightcoral"]


def fit_score(X, y):
    """mean 10-fold CV prediction score (Pearson r) for a single ridge fit."""
    rg = Ridge(n_splits, alphas=np.logspace(-3, 3, 20), n_jobs=1,
               random_state=100, n_pca=nc, n_perm=0)
    return float(np.mean(rg.bv_linear(X, y)))


def _atomic_pickle(obj, path):
    tmp = f"{path}.tmp"
    with open(tmp, "wb") as f:
        pickle.dump(obj, f)
    os.replace(tmp, path)


def _valid_meta(saved, pcoders, times):
    return (saved.get("roi") == roi_name and saved.get("hps") == args.hps
            and list(saved.get("pcoders", [])) == list(pcoders)
            and np.array_equal(saved.get("times"), times.values * 1000))


if not args.plot_only:
    # MEG: (subject, trial=540, time). Reorder trials to the canonical stimulus order.
    csv = pd.read_csv(fname.stimuli_con_dir)
    events = csv.columns.tolist()
    stimuli = [w for c in events for w in csv[c].tolist()]
    data = xr.load_dataarray(fname.meg_tc_sti(roi=roi_id)).sel(trial=stimuli)
    times = data.time[::time_step]
    times = times[(times >= 0) & (times <= time_len - time_interval)]
    nwin = len(times)

    # model activations per PCoder and state: (pcoder, state) -> (540, feat)
    with open(REPS, "rb") as f:
        rep_dict = pickle.load(f)
    pcoders = list(rep_dict)
    X = {}
    for p in pcoders:
        reps_all = np.array([r for c in events for r in rep_dict[p][c]])   # (540, n_ts, feat)
        for si, it in enumerate(STATE_TS):
            X[(p, si)] = reps_all[:, it, :]

    # per-subject, per-window target: Ywin (subject, nwin, 540) with NaN for missing stimuli
    Ywin = np.full((len(subjects), nwin, len(stimuli)), np.nan)
    for su, subject in enumerate(subjects):
        ds = data.sel(subject=subject)
        for wi, t in enumerate(times):
            Ywin[su, wi] = ds.sel(time=slice(t, t + time_interval)).mean("time").data

    # ---- (A) subject-level scores : (pcoder, state, time, subject) -------------
    if args.resume_bootstrap:
        with open(OUT, "rb") as f:
            saved = pickle.load(f)
        expected_shape = (len(pcoders), len(STATES), nwin, len(subjects))
        if saved.get("subj") is None or saved["subj"].shape != expected_shape:
            raise ValueError(
                f"Cannot resume from {OUT}: expected subject-level shape "
                f"{expected_shape}, got "
                f"{None if saved.get('subj') is None else saved['subj'].shape}"
            )
        if not _valid_meta(saved, pcoders, times):
            raise ValueError(
                f"Cannot resume from {OUT}: saved ROI, hyperparameters, PCoders, "
                "or time coordinates do not match this run"
            )
        subj = saved["subj"]
        print(f"[A] skipped: loaded completed subject-level result from {OUT}")
    else:
        def subj_scores(su):
            out = np.zeros((len(pcoders), len(STATES), nwin))
            for wi in range(nwin):
                yv = Ywin[su, wi]
                valid = np.isfinite(yv)
                for pi in range(len(pcoders)):
                    for si in range(len(STATES)):
                        out[pi, si, wi] = fit_score(X[(pcoders[pi], si)][valid], yv[valid])
            return out
        print(f"[A] subject-level ridge: {len(subjects)} participants x {nwin} windows ...")
        subj = np.array(Parallel(n_jobs=args.n_jobs)(delayed(subj_scores)(su)
                                                     for su in range(len(subjects))))
        subj = np.moveaxis(subj, 0, -1)                  # (pcoder, state, time, subject)

        # Persist A immediately so a later run can resume directly from B.
        _atomic_pickle({"subj": subj, "boot": None, "times": times.values * 1000,
                        "pcoders": pcoders, "roi": roi_name, "hps": args.hps}, OUT)
        print("saved (subject-level only, boot pending) ->", OUT)

    # ---- (B) participant bootstrap on the grand-average -----------------------
    boot = saved.get("boot") if args.resume_bootstrap and "saved" in locals() else None
    if args.n_boot:
        total_boot = args.n_boot + 1                         # b=0 point estimate + resamples
        boot_shape = (total_boot, len(pcoders), len(STATES), nwin)

        def init_boot_state():
            boot_arr = np.full(boot_shape, np.nan, dtype=float)
            done_arr = np.zeros(total_boot, dtype=bool)
            if boot is not None:
                if boot.ndim != 4 or boot.shape[1:] != boot_shape[1:]:
                    raise ValueError(
                        f"Cannot reuse saved boot array from {OUT}: expected trailing "
                        f"shape {boot_shape[1:]}, got {boot.shape[1:]}"
                    )
                ncopy = min(total_boot, boot.shape[0])
                boot_arr[:ncopy] = boot[:ncopy]
                done_arr[:ncopy] = np.isfinite(boot_arr[:ncopy]).all(axis=(1, 2, 3))
            return boot_arr, done_arr

        boot, boot_done = init_boot_state()
        if os.path.exists(BOOT_CKPT):
            with open(BOOT_CKPT, "rb") as f:
                ckpt = pickle.load(f)
            if not _valid_meta(ckpt, pcoders, times):
                raise ValueError(
                    f"Cannot resume bootstrap from {BOOT_CKPT}: saved ROI, "
                    "hyperparameters, PCoders, or time coordinates do not match this run"
                )
            ckpt_boot = ckpt["boot"]
            if ckpt_boot.ndim != 4 or ckpt_boot.shape[1:] != boot_shape[1:]:
                raise ValueError(
                    f"Cannot resume bootstrap from {BOOT_CKPT}: expected trailing "
                    f"shape {boot_shape[1:]}, got {ckpt_boot.shape[1:]}"
                )
            ckpt_done = np.asarray(ckpt.get("done", np.isfinite(ckpt_boot).all(axis=(1, 2, 3))),
                                   dtype=bool)
            ncopy = min(total_boot, ckpt_boot.shape[0])
            boot[:ncopy] = ckpt_boot[:ncopy]
            boot_done[:ncopy] = ckpt_done[:ncopy] & np.isfinite(boot[:ncopy]).all(axis=(1, 2, 3))
            print(f"[B] loaded bootstrap checkpoint {BOOT_CKPT}: "
                  f"{boot_done.sum()}/{total_boot} completed")

        def one_boot(b):
            rng = np.random.RandomState(b)
            order = rng.randint(0, len(subjects), len(subjects)) if b > 0 else np.arange(len(subjects))
            grand = np.nanmean(Ywin[order], axis=0)      # (nwin, 540)
            out = np.zeros((len(pcoders), len(STATES), nwin))
            for wi in range(nwin):
                yv = grand[wi]; valid = np.isfinite(yv)
                for pi in range(len(pcoders)):
                    for si in range(len(STATES)):
                        out[pi, si, wi] = fit_score(X[(pcoders[pi], si)][valid], yv[valid])
            return out

        pending = np.where(~boot_done)[0].tolist()
        print(f"[B] participant bootstrap: {args.n_boot} resamples "
              f"(b=0 is the point estimate); {len(pending)}/{total_boot} pending; "
              f"checkpoint every {args.boot_chunk} resamples -> {BOOT_CKPT}")
        for start in range(0, len(pending), args.boot_chunk):
            chunk = pending[start:start + args.boot_chunk]
            print(f"[B] running bootstrap indices {chunk[0]}-{chunk[-1]} "
                  f"({boot_done.sum()}/{total_boot} already complete)", flush=True)
            vals = Parallel(n_jobs=args.n_jobs)(delayed(one_boot)(int(b)) for b in chunk)
            for b, val in zip(chunk, vals):
                boot[b] = val
                boot_done[b] = True
            _atomic_pickle({"boot": boot, "done": boot_done,
                            "times": times.values * 1000, "pcoders": pcoders,
                            "roi": roi_name, "hps": args.hps, "n_boot": args.n_boot},
                           BOOT_CKPT)
            # Also refresh OUT so plot_only can use any complete boot if this run finishes.
            _atomic_pickle({"subj": subj, "boot": boot if boot_done.all() else None,
                            "times": times.values * 1000, "pcoders": pcoders,
                            "roi": roi_name, "hps": args.hps}, OUT)
            print(f"[B] checkpoint saved: {boot_done.sum()}/{total_boot} completed", flush=True)
        # boot[0] = grand-average point estimate; boot[1:] = resamples
        if not boot_done.all():
            raise RuntimeError(f"Bootstrap checkpoint incomplete: {boot_done.sum()}/{total_boot}")

    payload = {"subj": subj, "boot": boot, "times": times.values * 1000,
               "pcoders": pcoders, "roi": roi_name, "hps": args.hps}
    _atomic_pickle(payload, OUT)
    print("saved ->", OUT)
else:
    payload = pickle.load(open(OUT, "rb"))

# ---- inference + effect sizes ------------------------------------------------
subj = payload["subj"]; boot = payload["boot"]; tms = payload["times"]
pcoders = payload["pcoders"]; roi_name = payload["roi"]


def cluster(diff):
    """One-tailed paired cluster test across subjects (w/ > w/o)."""
    _, cl, pv, _ = mne.stats.permutation_cluster_1samp_test(
        diff, n_permutations=5000, tail=1, seed=50, verbose=False)
    return [(cl[i][0], float(pv[i])) for i in np.where(pv < 0.05)[0]]


def classical_d(w_off, w_on):                            # (n, time) -> (time,)
    md = (w_on - w_off).mean(0)
    sd = np.sqrt((w_off.var(0, ddof=1) + w_on.var(0, ddof=1)) / 2)
    return md / sd


def report_clusters(label, mat):
    cls = cluster(mat)
    if not cls:
        print(f"   [A] {label}: no significant cluster (one-tailed p<0.05)")
        return
    for cl, pv in cls:
        subj_eff = mat[:, cl].mean(1)
        mean_eff = subj_eff.mean()
        ci = 1.96 * subj_eff.std(ddof=1) / np.sqrt(len(subj_eff))
        dz = mean_eff / subj_eff.std(ddof=1) if subj_eff.std(ddof=1) > 0 else np.nan
        direction = "> 0" if mean_eff > 0 else "< 0"
        print(f"   [A] {label} {direction}: {tms[cl].min():.0f}-{tms[cl].max():.0f} ms"
              f" (one-tailed p={pv:.4f}); mean {mean_eff:+.3f} +/- {ci:.3f} "
              f"(95% CI); dz={dz:.2f}; positive subjects={np.sum(subj_eff > 0)}/{len(subj_eff)}")


print("\n" + "=" * 68)
print(f"Ridge feedback contrast — {roi_name} ({payload['hps']})  [participant-level inference]")
print("=" * 68)
for pi, p in enumerate(pcoders):
    w_off, w_on = subj[pi, 0].T, subj[pi, 1].T           # (subject, time)
    diff = w_on - w_off
    cls = cluster(diff)
    print(f"\nPCoder {pi+1}:")
    report_clusters("w/o feedback vs 0", w_off)
    report_clusters("w/ feedback vs 0", w_on)
    if not cls:
        print("   [A] feedback contrast: no significant cluster (one-tailed p<0.05)")
    for cl, pv in cls:
        d = classical_d(w_off, w_on)[cl]                 # d_av (raw within-condition SD)
        mdiff = diff[:, cl].mean(1)                       # (subject,)
        ci = 1.96 * mdiff.std(ddof=1) / np.sqrt(len(mdiff))
        peak_d = d[np.argmax(np.abs(d))]
        direction = "w/ > w/o"
        print(f"   [A] subject-level cluster {tms[cl].min():.0f}-{tms[cl].max():.0f} ms"
              f" ({direction}, one-tailed p={pv:.4f}):"
              f" Cohen's d={d.mean():.2f} (peak |d| {peak_d:.2f});"
              f" mean diff {mdiff.mean():+.3f} +/- {ci:.3f} (95% CI);"
              f" w/o={w_off[:, cl].mean():.3f} w/={w_on[:, cl].mean():.3f}")
    if boot is not None:
        bd = boot[1:, pi, 1] - boot[1:, pi, 0]           # (nboot, time)
        lo, hi = np.percentile(bd, [2.5, 97.5], axis=0)
        pt = boot[0, pi, 1] - boot[0, pi, 0]             # point estimate (time,)
        sig = np.where((lo > 0) | (hi < 0))[0]             # windows whose 95% CI excludes 0
        if len(sig):
            print(f"   [B] bootstrap: 95% CI excludes 0 at {tms[sig].min():.0f}-{tms[sig].max():.0f} ms;"
                  f" peak diff {pt.max():+.3f} [{lo[pt.argmax()]:+.3f}, {hi[pt.argmax()]:+.3f}]")
        else:
            print("   [B] bootstrap: 95% CI includes 0 at all windows")


# ---- plot: subject-level (mean +/- 95% CI) + bootstrap grand-average ----------
def band(ax, x, m, se, color, label):
    ax.plot(x, m, color=color, lw=2, label=label)
    ax.fill_between(x, m - 1.96 * se, m + 1.96 * se, color=color, alpha=0.2)


fig, axes = plt.subplots(2, len(pcoders), figsize=(13, 6.4), sharex=True)
for pi, p in enumerate(pcoders):
    # top row: subject-level
    ax = axes[0, pi]
    w_off, w_on = subj[pi, 0].T, subj[pi, 1].T
    n = w_off.shape[0]
    band(ax, tms, w_off.mean(0), w_off.std(0, ddof=1) / np.sqrt(n), FB_COLORS[0], "w/o feedback")
    band(ax, tms, w_on.mean(0), w_on.std(0, ddof=1) / np.sqrt(n), FB_COLORS[1], "w/ feedback")
    ylo, yhi = ax.get_ylim()
    yrng = yhi - ylo
    for cl, _ in cluster(w_off):
        ax.plot(tms[cl], [ylo + yrng * 0.03] * len(cl), "s", color=FB_COLORS[0],
                markeredgecolor="k", markeredgewidth=0.3, ms=3)
    for cl, _ in cluster(w_on):
        ax.plot(tms[cl], [ylo + yrng * 0.06] * len(cl), "s", color=FB_COLORS[1],
                markeredgecolor="k", markeredgewidth=0.3, ms=3)
    for cl, _ in cluster(w_on - w_off):
        ax.plot(tms[cl], [ylo] * len(cl), "s", color="k", ms=3)
    ax.axhline(0, color="k", lw=0.5); ax.set_title(f"PCoder {pi+1}")
    if pi == 0:
        ax.set_ylabel(f"{roi_name}\nsubject-level score (r)")
    # bottom row: grand-average + participant bootstrap CI
    ax2 = axes[1, pi]
    if boot is not None:
        for si, (col, lab) in enumerate(zip(FB_COLORS, ["w/o feedback", "w/ feedback"])):
            pt = boot[0, pi, si]
            lo, hi = np.percentile(boot[1:, pi, si], [2.5, 97.5], axis=0)
            ax2.plot(tms, pt, color=col, lw=2, label=lab)
            ax2.fill_between(tms, lo, hi, color=col, alpha=0.2)
        bd = boot[1:, pi, 1] - boot[1:, pi, 0]
        loD = np.percentile(bd, 2.5, axis=0)
        hiD = np.percentile(bd, 97.5, axis=0)
        ylo2, _ = ax2.get_ylim()
        sig = np.where((loD > 0) | (hiD < 0))[0]
        if len(sig):
            ax2.plot(tms[sig], [ylo2] * len(sig), "s", color="k", ms=3)
    ax2.axhline(0, color="k", lw=0.5); ax2.set_xlabel("Time (ms)")
    if pi == 0:
        ax2.set_ylabel(f"{roi_name}\ngrand-avg score (r)\n(participant bootstrap 95% CI)")
axes[0, -1].legend(fontsize=8, frameon=False)
fig.suptitle(f"Ridge model->MEG, participant-level inference ({roi_name}, {payload['hps']}). "
             f"Top: subject-level (mean±95% CI, one-tailed cluster across 23 participants). "
             f"Bottom: grand-average (participant bootstrap 95% CI).", y=1.02, fontsize=9)
fig.tight_layout()
base = f"{fname.figures_dir}/ridge/ridge_subjinf_{roi_name}{suf}"
for ext in ("png", "pdf", "tif"):
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
    fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=200, **kw)
print("\nsaved ->", f"{base}.[png|pdf|tif]")
