"""Effect sizes and multiple-comparison correction of the ROI contrasts, t = 0 (w/o
feedback) vs t = 25 (w/ feedback), for the condition-wise RSA and the
participant-level ridge regression in the post hoc ROIs left vOT (parcel 40) and
left ST (parcel 65): S4 Table (participant-level ridge; assembled
by S4Tab_effect_sizes_vOT_ST.py) and the ROI families of S5 Table (S5Tab_correction_families.py).

For each ROI and PCoder, the significant clusters are recovered with the two-tailed
paired cluster test of the time courses (5000 permutations), and reported inside
each cluster:
  - the paired effect size d_z (mean / SD of the participant-wise differences) with
    its exact noncentral-t 95% CI, and the raw mean difference with its 95% t CI,
    computed on the participant-wise means over the cluster (descriptive: the
    cluster was selected from the same data) and over the whole epoch (no
    selection); d_av for reference
  - the absolute scores (t = 0 and t = 25)
RSA and ridge are paired across the 23 participants (ridge: participant-level
scores, each cross-validated across stimuli; 29_ridge_participant_level.py).

Each cluster test controls the family-wise error over time within a test.
Benjamini-Hochberg FDR is applied within two families (model_brain_fdr.py): RSA
(vOT/ST x PCoder 3) and ridge regression (vOT/ST x PCoders 1-3); every cluster
of every test enters the correction of its family. Effect sizes are reported
separately for each cluster with p < .05.

Inputs: ../data/rsa_tcs/rsa_tcs_parcel{40,65}_RL23.nc (18_rsa_parcels.py),
        ../data/behavior/ridge_subjinf_{vOT,ST}_RL23.pkl (29_ridge_participant_level.py)
Outputs: ../data/effect_sizes_rsa_ridge_RL23.csv, ../data/fdr_families_roi_RL23.csv,
         ../data/fdr_families_roi_clusters_RL23.csv

  python 22_effect_sizes_clusters.py --hps RL23
Run after 19_rsa_wholecortex.py.
"""
import argparse
import os
import pickle

import mne
import numpy as np
import xarray as xr
from config import fname
from model_brain_fdr import SEED, model_brain_fdr
from effect_sizes import d_av, dz_ci, mean_ci

pa = argparse.ArgumentParser(description=__doc__)
pa.add_argument("--hps", type=str, default="RL23", choices=["RL1", "RL23"],
                help="hyperparameters of the model representations: RL23 (default, reported) or RL1")
args, _ = pa.parse_known_args()


def rl23(path):
    """append _RL23 before the extension when --hps RL23. The contrast is always
    t=0 (feedforward, hyperparameter-independent) vs t=25; with RL23 the t=25 state
    uses the hyperparameters optimized on RL2+RL3."""
    if args.hps != "RL23":
        return str(path)
    root, ext = os.path.splitext(str(path))
    return f"{root}_RL23{ext}"


ROIS = [("vOT", 40), ("ST", 65)]
PCODERS = [0, 1, 2]  # PCoder 1..3
def cluster_masks(diff, seed=SEED):
    """Two-tailed paired cluster test over time; return one boolean time
    mask per cluster with p < .05, their p-values, and the smallest cluster p."""
    _, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
        diff, n_permutations=5000, tail=0, seed=seed)
    masks, sig_p = [], []
    for idx in np.where(pvals < 0.05)[0]:
        mask = np.zeros(diff.shape[1], bool)
        mask[clusters[idx][0]] = True
        masks.append(mask)
        sig_p.append(pvals[idx])
    min_p = float(np.min(pvals)) if len(pvals) else 1.0
    return masks, sig_p, min_p


ROWS = []  # one row per summary, saved as CSV


def report(kind, roi, times, w_off, w_on, n_label):
    """w_off/w_on: (n, time) scores without feedback (t=0) / with feedback (t=25).
    Reports every significant cluster (descriptive: selected from the same
    participants) and, without any selection, the mean over the predefined
    epoch (all time points)."""
    diff = w_on - w_off
    masks, sig_p, min_p = cluster_masks(diff, seed=1)
    print(f"\n[{kind}] {roi}  (paired across {diff.shape[0]} {n_label})")
    if not masks:
        print(f"   no significant cluster (smallest cluster p={min_p:.4f})")
    for mask, p in zip(masks, sig_p):
        report_cluster(kind, roi, times, w_off, w_on, mask, p, "cluster (descriptive)")
    if not np.all(diff == 0):
        report_cluster(kind, roi, times, w_off, w_on, np.ones(diff.shape[1], bool), np.nan,
                       "whole epoch (predefined)")


def report_cluster(kind, roi, times, w_off, w_on, mask, p, selection):
    """Participant-wise means over the time points of `mask`, then the paired
    mean difference with its 95% t CI and d_z with its exact CI (effect_sizes.py)."""
    tms = times[mask] * 1000
    off, on = w_off[:, mask].mean(1), w_on[:, mask].mean(1)  # (n,)
    mdiff = on - off
    mean, low, high = mean_ci(mdiff)
    d_z, dz_low, dz_high = dz_ci(mdiff)
    print(f"   {selection}: {tms.min():.0f}-{tms.max():.0f} ms (p={p:.4f}), {mask.sum()} timepoints")
    print(f"   mean diff {mean:.3f} [{low:.3f}, {high:.3f}], d_z {d_z:.2f} [{dz_low:.2f}, {dz_high:.2f}], "
          f"d_av {d_av(off, on):.2f}")
    print(f"   absolute: w/o={off.mean():.3f}  w/={on.mean():.3f}")
    ROWS.append({"analysis": kind, "roi": roi, "selection": selection,
                 "start_ms": tms.min(), "end_ms": tms.max(), "p_cluster": p,
                 "without": off.mean(), "with": on.mean(), "mean_diff": mean,
                 "ci95_low": low, "ci95_high": high, "d_z": d_z,
                 "d_z_ci95_low": dz_low, "d_z_ci95_high": dz_high, "d_av": d_av(off, on)})


# ------------------------------------------------------------------ RSA
print("=" * 70)
print(f"RSA effect sizes (t=0 vs t=25), paired d_z, across participants [hps={args.hps}]")
print("=" * 70)
for roi_name, roi in ROIS:
    data = xr.load_dataarray(rl23(fname.rsa_tc(roi=roi)))  # (pcoder, subject, time, feedback)
    times = data.time.values
    for p in PCODERS:
        w_off = data.isel(pcoder=p, feedback=0).data  # (subject, time)
        w_on = data.isel(pcoder=p, feedback=1).data
        report("RSA", f"{roi_name} PCoder{p+1}", times, w_off, w_on, "participants")

# ------------------------------------------------------------------ ridge
print("\n" + "=" * 70)
print(f"Ridge effect sizes (t=0 vs t=25), paired d_z, across participants [hps={args.hps}]")
print("=" * 70)
for roi_name, roi in ROIS:
    suffix = "_RL23" if args.hps == "RL23" else ""
    with open(f"{fname.stimuli_dir}/ridge_subjinf_{roi_name}{suffix}.pkl", "rb") as f:
        payload = pickle.load(f)
    subj = payload["subj"]  # (pcoder, state, time, subject)
    times = np.asarray(payload["times"]) / 1000
    for p in PCODERS:
        w_off = subj[p, 0].T  # (subject, time)
        w_on = subj[p, 1].T
        report("ridge", f"{roi_name} PCoder{p+1}", times, w_off, w_on, "participants")


# ------------------------------------------ FDR (families in model_brain_fdr.py)
# Each cluster test controls the family-wise error over time within a test;
# every cluster of every test enters the BH correction of its family.
print("\n" + "=" * 70)
print(f"Benjamini-Hochberg FDR within the RSA and ridge families [hps={args.hps}]")
print("=" * 70)
for (kind, roi, pcoder), r in model_brain_fdr(args.hps).items():
    roi_name = dict((v, k) for k, v in ROIS)[roi]
    print(f"   {kind:<6} {roi_name:<4} PCoder{pcoder + 1}  (m = {r['m']} clusters in the family)")
    for c in r["clusters"]:
        print(f"      {c['start_ms']:.0f}-{c['end_ms']:.0f} ms  raw p={c['p']:.4f}  BH p={c['p_fdr']:.4f} "
              f"({'sig' if c['reject'] else 'n.s.'})")

# save all effect sizes and the FDR families
out = f"{fname.data_dir}/effect_sizes_rsa_ridge_{args.hps}.csv"
import pandas as pd  # noqa: E402
pd.DataFrame(ROWS).to_csv(out, index=False)
FDR = model_brain_fdr(args.hps)
names = dict((v, k) for k, v in ROIS)
fam = [{"family": kind, "roi": names[roi], "pcoder": pcoder + 1,
        **{k: v for k, v in r.items() if k != "clusters"}} for (kind, roi, pcoder), r in FDR.items()]
pd.DataFrame(fam).to_csv(f"{fname.data_dir}/fdr_families_roi_{args.hps}.csv", index=False)
# every cluster of every ROI test with its unadjusted and BH-adjusted p (all clusters enter the family)
cl = [{"family": kind, "roi": names[roi], "pcoder": pcoder + 1, **c}
      for (kind, roi, pcoder), r in FDR.items() for c in r["clusters"]]
pd.DataFrame(cl).to_csv(f"{fname.data_dir}/fdr_families_roi_clusters_{args.hps}.csv", index=False)
print("saved ->", out)
