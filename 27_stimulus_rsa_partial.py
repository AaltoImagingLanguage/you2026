########################################################################
# Stimulus-level RSA and partial RSA (S13 Fig).
#
# Instead of the four condition RDMs (6 dissimilarities), RDMs are computed over
# the 540 individual stimuli (~1.5e5 dissimilarities), and the model's
# representational geometry is correlated with the MEG geometry per participant
# and 100-ms window (Spearman). Partial RSA (partial Spearman) tests whether the
# model explains MEG variance beyond simpler stimulus predictors (edit distance,
# orthographic bigram overlap, visual pixel similarity). Model states: w/o
# feedback (t = 0) and w/ feedback (t = 25). Clusters (5000 permutations,
# p < 0.05): one-sided (> 0) for the model RSA, two-sided for the predictors.
# Rejected trials are excluded per participant (RSA on the valid stimulus subset).
#
# Inputs:  ../data/meg_tc/{40,65}_ds100Hz_sti_ver.nc (26_meg_vertex_time_courses.py)
#          ../data/behavior/pcoder_reps_conds_RL23.pkl
#          ../data/dataset/rep_stimuli.csv, ../data/images_dataset/stimuli.tar
# Outputs: ../data/behavior/stim_rsa_{vOT,ST}_RL23.pkl
#          figures/rsa/stim_rsa_ST_vOT_RL23.{png,pdf,tif} (S13 Fig, with --combine_rois)
#
#   python 27_stimulus_rsa_partial.py --parcel_ind 40 --hps RL23   # left vOT
#   python 27_stimulus_rsa_partial.py --parcel_ind 65 --hps RL23   # left ST
#   python 27_stimulus_rsa_partial.py --hps RL23 --combine_rois    # S13 Fig
########################################################################
import argparse
import pickle

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import mne
from scipy.spatial.distance import pdist, squareform
from scipy.stats import rankdata
from Levenshtein import distance as levenshtein

from config import (fname, subjects, time_len, time_step, time_interval,
                    fb_timestep, metric_rdm)

pa = argparse.ArgumentParser(description=__doc__)
pa.add_argument("--parcel_ind", type=int, default=40, help="40=vOT, 65=ST")
pa.add_argument("--hps", type=str, default="RL23", choices=["RL1", "RL23"])
pa.add_argument("--plot_only", action="store_true")
pa.add_argument("--combine_rois", action="store_true",
                help="plot saved ST and vOT stimulus-level RSA results in one figure")
args, _ = pa.parse_known_args()
roi_id = args.parcel_ind
ROI_NAMES = {40: "vOT", 65: "ST"}
roi_name = ROI_NAMES[roi_id]
suf = "_RL23" if args.hps == "RL23" else ""
REPS = str(fname.pcoder_reps).replace(".pkl", "_RL23.pkl") if args.hps == "RL23" else fname.pcoder_reps
OUT = f"{fname.data_dir}/behavior/stim_rsa_{roi_name}{suf}.pkl"
STATES = {"t0": 0, "t25": fb_timestep}          # feed-forward vs converged


# ---- stimuli, in the fixed order used everywhere (csv columns flattened) ------
csv = pd.read_csv(fname.stimuli_con_dir)
conds = csv.columns.tolist()
stimuli = [w for c in conds for w in csv[c].tolist()]        # 540 strings
N = len(stimuli)


def rdm(mat):
    """condensed correlation RDM over rows of `mat` (n_items, features)."""
    return pdist(mat, metric=metric_rdm)


def bigrams(w):
    return set(w[i:i + 2] for i in range(len(w) - 1))


# ---- simpler-predictor RDMs (pairwise, stimulus-to-stimulus) ------------------
def predictor_rdms():
    edit = np.array([levenshtein(stimuli[i], stimuli[j])
                     for i in range(N) for j in range(i + 1, N)], dtype=float)
    bgs = [bigrams(w) for w in stimuli]
    ortho = np.array([1 - len(bgs[i] & bgs[j]) / max(len(bgs[i] | bgs[j]), 1)
                      for i in range(N) for j in range(i + 1, N)], dtype=float)
    # visual similarity: correlation RDM over the rendered word images
    import glob
    import webdataset as wds
    from utility import transform
    img = {}
    for png, js in (wds.WebDataset(glob.glob(f"{fname.dataset_dir}/stimuli.tar"))
                    .decode("pil").map_dict(png=transform).to_tuple("png", "json")):
        img[js["word"]] = np.asarray(png).ravel()
    visual = rdm(np.stack([img[w] for w in stimuli]))
    # word length is not a usable predictor here: all stimuli are 7-8 letters, so the
    # length RDM has essentially no structure.
    return {"edit distance": edit, "orthographic": ortho, "visual": visual}


# ---- helpers -----------------------------------------------------------------
def _resid(rank_vec, cov_rank):
    A = np.column_stack([np.ones(len(rank_vec)), cov_rank])
    return rank_vec - A @ np.linalg.lstsq(A, rank_vec, rcond=None)[0]


def _sub(condensed, idx):
    """subset a condensed RDM to the items in idx (via the square form)."""
    return squareform(squareform(condensed)[np.ix_(idx, idx)])


if not args.plot_only and not args.combine_rois:
    # MEG: per-stimulus vertex patterns. Some subjects lack some stimuli (rejected
    # trials -> NaN patterns), so the RSA for each subject is computed on that
    # subject's valid stimulus subset.
    meg = xr.load_dataarray(fname.meg_tc_sti_ver(roi=roi_id, fs=100)).sel(trial=stimuli)
    meg = meg.sel(time=slice(0, time_len))
    times = meg.time[::time_step]; times = times[times <= time_len - time_interval]

    preds = predictor_rdms()                                    # dict of condensed RDMs (540)
    with open(REPS, "rb") as f:
        rep_dict = pickle.load(f)
    pcoders = list(rep_dict)
    model_full = {}                                            # full (540) model RDMs
    for p in pcoders:
        reps_all = np.array([r for c in conds for r in rep_dict[p][c]])   # (540, n_ts, feat)
        for sname, ts in STATES.items():
            model_full[(p, sname)] = rdm(reps_all[:, ts, :])

    res_full = {(p, s): [] for p in pcoders for s in STATES}   # -> (subj, time)
    res_part = {(p, s): [] for p in pcoders for s in STATES}
    pred_self = {k: [] for k in preds}
    for subject in subjects:
        ms = meg.sel(subject=subject)
        valid = np.isfinite(ms.data).all(axis=(1, 2))          # stimuli with finite pattern
        idx = np.where(valid)[0]
        cov = np.column_stack([rankdata(_sub(preds[k], idx)) for k in preds])   # (n_pairs, 3)
        mrank = {key: rankdata(_sub(model_full[key], idx)) for key in model_full}
        prank = {k: rankdata(_sub(preds[k], idx)) for k in preds}
        fs = {key: [] for key in res_full}; ps = {key: [] for key in res_part}
        pss = {k: [] for k in preds}
        for t in times:
            patt = (ms.isel(trial=idx).sel(time=slice(t, t + time_interval))
                    .mean("time").transpose("trial", "vertex").data)
            grank = rankdata(pdist(patt, metric=metric_rdm))
            gres = _resid(grank, cov)
            for k in preds:
                pss[k].append(np.corrcoef(prank[k], grank)[0, 1])
            for key in model_full:
                fs[key].append(np.corrcoef(mrank[key], grank)[0, 1])
                ps[key].append(np.corrcoef(_resid(mrank[key], cov), gres)[0, 1])
        for key in res_full:
            res_full[key].append(fs[key]); res_part[key].append(ps[key])
        for k in preds:
            pred_self[k].append(pss[k])
        print(f"subject {subject}: {len(idx)}/{N} valid stimuli")

    # model <-> predictor RSA (static, full 540 items): the model shares
    # orthographic/visual structure with the predictors.
    model_pred = {}
    for key in model_full:
        for kk in preds:
            model_pred[f"{key[0]}|{key[1]}|{kk}"] = float(
                np.corrcoef(rankdata(model_full[key]), rankdata(preds[kk]))[0, 1])
    print("model<->predictor RSA (full 540):")
    for kk in preds:
        print("  " + kk + ": " + "  ".join(
            f"{key[0]} {key[1]}={model_pred[f'{key[0]}|{key[1]}|{kk}']:+.3f}" for key in model_full))
    # correlations among the predictor RDMs
    print("predictor<->predictor RSA:")
    pk = list(preds)
    for a2 in range(len(pk)):
        for b2 in range(a2 + 1, len(pk)):
            r = np.corrcoef(rankdata(preds[pk[a2]]), rankdata(preds[pk[b2]]))[0, 1]
            print(f"  {pk[a2]} <-> {pk[b2]}: {r:+.3f}")

    payload = {"times": times.values * 1000,
               "full": {k: np.array(v) for k, v in res_full.items()},     # (subj, time)
               "partial": {k: np.array(v) for k, v in res_part.items()},
               "pred_self": {k: np.array(v) for k, v in pred_self.items()},
               "model_pred": model_pred, "pcoders": pcoders, "roi": roi_name}
    with open(OUT, "wb") as f:
        pickle.dump(payload, f)
    print("saved ->", OUT)
elif args.plot_only and not args.combine_rois:
    payload = pickle.load(open(OUT, "rb"))

# ---- plot: model raw vs partial RSA + predictors, per PCoder -----------------
if args.combine_rois:
    payloads = []
    for rid in (65, 40):   # ST, then vOT
        rname = ROI_NAMES[rid]
        path = f"{fname.data_dir}/behavior/stim_rsa_{rname}{suf}.pkl"
        with open(path, "rb") as f:
            payloads.append(pickle.load(f))
        print("loaded ->", path)
else:
    payloads = [payload]

tms = payloads[0]["times"]; pcoders = payloads[0]["pcoders"]
for pld in payloads[1:]:
    if not np.array_equal(pld["times"], tms) or list(pld["pcoders"]) != list(pcoders):
        raise ValueError("Cannot combine stimulus RSA payloads: time axis or PCoder labels differ")
PRED_COLORS = ["#0072B2", "#009E73", "#CC79A7"]   # edit, ortho, visual


def _cluster_idx(cluster):
    cluster = cluster[0] if isinstance(cluster, tuple) else cluster
    if isinstance(cluster, slice):
        return np.arange(len(tms))[cluster]
    return np.asarray(cluster, dtype=int)


def cluster_test(mat, tail=1):   # (subj, time); tail=1 one-sided >0, tail=0 two-sided
    _, cl, pv, _ = mne.stats.permutation_cluster_1samp_test(mat, n_permutations=5000, tail=tail, seed=50, verbose=False)
    return [(_cluster_idx(cl[i]), float(pv[i])) for i in np.where(pv < 0.05)[0]]


def clusters(mat, tail=1):
    return [idx for idx, _ in cluster_test(mat, tail=tail)]


def print_raw_delta_diagnostics(payload):
    print(f"raw t=25 vs t=0 clusters ({payload['roi']}; one-sided cluster p<0.05; printed, not drawn):")
    for pi, p in enumerate(pcoders):
        full25 = payload["full"][(p, "t25")]
        full0 = payload["full"][(p, "t0")]
        for label, mat in (("t25 - t0 > 0", full25 - full0),
                           ("t0 - t25 > 0", full0 - full25)):
            rows = cluster_test(mat, tail=1)
            if not rows:
                print(f"  PCoder {pi + 1} {label}: none")
                continue
            for idx, pv in rows:
                subj_eff = mat[:, idx].mean(axis=1)
                mean_eff = float(subj_eff.mean())
                sd_eff = float(subj_eff.std(ddof=1))
                cohend = mean_eff / sd_eff if sd_eff > 0 else np.nan
                print(f"  PCoder {pi + 1} {label}: {tms[idx[0]]:.0f}-{tms[idx[-1]]:.0f} ms, "
                      f"p={pv:.4f}, mean={mean_eff:+.6f}, d={cohend:+.2f}, "
                      f"positive subjects={int(np.sum(subj_eff > 0))}/{len(subj_eff)}")


nroi = len(payloads)
fig, axes = plt.subplots(nroi, len(pcoders), figsize=(13, 3.4 * nroi),
                         sharex=True, sharey=True)
axes = np.asarray(axes).reshape(nroi, len(pcoders))
for ri, payload in enumerate(payloads):
    for pi, p in enumerate(pcoders):
        ax = axes[ri, pi]
        full25 = payload["full"][(p, "t25")]           # (subj, time)
        full0 = payload["full"][(p, "t0")]
        part25 = payload["partial"][(p, "t25")]
        show_label = (ri == 0 and pi == len(pcoders) - 1)
        ax.plot(tms, full25.mean(0), color="#000000", lw=2,
                label="w/ feedback (raw)" if show_label else None)
        ax.plot(tms, part25.mean(0), color="#D55E00", lw=2, ls="--",
                label="w/ feedback (partial)" if show_label else None)
        ax.plot(tms, full0.mean(0), color="0.6", lw=1.2,
                label="w/o feedback" if show_label else None)
        for k, col in zip(payload["pred_self"], PRED_COLORS):
            ax.plot(tms, payload["pred_self"][k].mean(0), color=col, lw=1, alpha=0.8,
                    label=k if show_label else None)
        ylo, yhi = ax.get_ylim()
        for cl in clusters(full25):
            ax.plot(tms[cl], [ylo] * len(cl), "-", color="#000000", ms=3)
        for cl in clusters(part25):
            ax.plot(tms[cl], [ylo + (yhi - ylo) * 0.03] * len(cl), "s", color="#D55E00", ms=2)
        # t=25 vs t=0 clusters are not drawn (they are printed by
        # print_raw_delta_diagnostics); at stimulus level they can flag very small
        # differences in this low-SNR single-trial analysis.
        # predictor RSA clusters (two-sided)
        for j, (k, col) in enumerate(zip(payload["pred_self"], PRED_COLORS)):
            for cl in clusters(payload["pred_self"][k], tail=0):
                ax.plot(tms[cl], [ylo + (yhi - ylo) * (0.06 + 0.03 * j)] * len(cl), "-", color=col, lw=2)
        ax.axhline(0, color="k", lw=0.5)
        if ri == 0:
            ax.set_title(f"PCoder {pi+1}")
        if ri == nroi - 1:
            ax.set_xlabel("Time (ms)")
        if pi == 0:
            ax.set_ylabel(f"{payload['roi']}\nstimulus-level RSA\n(spearman)")
        ax.spines[["right", "top"]].set_visible(False)
axes[0, -1].legend(fontsize=7, frameon=False, bbox_to_anchor=(1.02, 1),
                   loc="upper left")
# fig.suptitle(f"Stimulus-level RSA and unique variance beyond simpler predictors "
#              f"({payload['roi']}, {args.hps})", y=1.03)
fig.tight_layout()
base = (f"{fname.figures_dir}/rsa/stim_rsa_ST_vOT{suf}" if args.combine_rois
        else f"{fname.figures_dir}/rsa/stim_rsa_{payloads[0]['roi']}{suf}")
for ext in ("png", "pdf", "tif"):
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
    fig.savefig(f"{base}.{ext}", bbox_inches="tight", dpi=200, **kw)
print("saved ->", f"{base}.[png|pdf|tif]")
for payload in payloads:
    print_raw_delta_diagnostics(payload)
