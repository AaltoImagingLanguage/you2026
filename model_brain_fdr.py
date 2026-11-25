"""FDR correction for the post hoc follow-up ROI tests (left vOT and ST),
used by Fig 6 (Fig5_Fig6_compose.py), 22_effect_sizes_clusters.py and S5 Table.

RSA and ridge regression are corrected as separate families.
  - RSA: vOT and ST x PCoder 3 (2 cells). PCoders 1 and 2 are not tested:
    feedback leaves the rank order of their six condition dissimilarities
    unchanged, so their RSA difference is exactly zero for every participant.
  - ridge: vOT and ST x PCoders 1--3 (6 cells), tested across the 23
    participants on the participant-level ridge scores (each participant's
    score cross-validated across stimuli; 29_ridge_participant_level.py), as
    reported in the paper (S15 Fig, S4 Table).
Every cluster of every test enters the BH correction of its family, the same
rule as the whole-cortex family. A cell (test) is summarized by its
smallest cluster: ``raw_p`` and ``adjusted_p`` are that cluster's unadjusted and
FDR-adjusted p; all clusters are listed in ``clusters``.
A joint 12-cell family (RSA and ridge x ROIs x PCoders 1--3, smallest cluster
p per cell) is also returned as ``adjusted_p_12``.

Inputs: ../data/rsa_tcs/rsa_tcs_parcel{40,65}_RL23.nc (18_rsa_parcels.py),
        ../data/behavior/ridge_subjinf_{vOT,ST}_RL23.pkl (29_ridge_participant_level.py)
Library module (no command-line interface).
"""

import os
import pickle

import mne
import numpy as np
import xarray as xr
from statsmodels.stats.multitest import multipletests

from config import fname


ROIS = (40, 65)  # left vOT and ST
PCODERS = (0, 1, 2)
N_PERMUTATIONS = 5000
SEED = 42


def _hps_path(path, hps):
    if hps != "RL23":
        return str(path)
    root, ext = os.path.splitext(str(path))
    return f"{root}_RL23{ext}"


def _clusters(diff, times_ms):
    """All clusters of the two-tailed temporal cluster test: list of
    (start_ms, end_ms, p). Empty if no cluster formed."""
    _, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
        diff,
        n_permutations=N_PERMUTATIONS,
        tail=0,  # two-tailed, as stated in the Methods
        seed=SEED,
        verbose=False,
    )
    return [(float(times_ms[c[0]][0]), float(times_ms[c[0]][-1]), float(p)) for c, p in zip(clusters, pvals)]


def _cells(hps):
    """All clusters of every (analysis, roi, pcoder) cell."""
    cells = {}
    for roi in ROIS:
        data = xr.load_dataarray(_hps_path(fname.rsa_tc(roi=roi), hps))
        times_ms = data.time.values * 1000
        for pcoder in PCODERS:
            diff = (
                data.isel(pcoder=pcoder, feedback=1).data
                - data.isel(pcoder=pcoder, feedback=0).data
            )
            cells[("RSA", roi, pcoder)] = _clusters(diff, times_ms)
    for roi in ROIS:
        roi_name = {40: "vOT", 65: "ST"}[roi]
        suffix = "_RL23" if hps == "RL23" else ""
        with open(f"{fname.stimuli_dir}/ridge_subjinf_{roi_name}{suffix}.pkl", "rb") as f:
            payload = pickle.load(f)
        subj = payload["subj"]  # (pcoder, state, time, subject)
        times_ms = np.asarray(payload["times"], float)
        for pcoder in PCODERS:
            diff = (subj[pcoder, 1] - subj[pcoder, 0]).T  # (subject, time)
            cells[("ridge", roi, pcoder)] = _clusters(diff, times_ms)
    return cells


def _bh(keys, cells, alpha):
    """BH-FDR over ALL clusters of the tests in ``keys``. A test without any
    cluster has p = 1 for every time point and contributes nothing. Returns per
    test its smallest cluster (raw and adjusted p), all clusters with their
    adjusted p, the family size m, and ``threshold``: the largest raw cluster p
    rejected in the family (0 if none); a cluster is significant after correction
    if its p is at or below it."""
    flat = [(k, i) for k in keys for i in range(len(cells[k]))]
    raw = np.asarray([cells[k][i][2] for k, i in flat])
    if len(raw):
        reject, adjusted, _, _ = multipletests(raw, alpha=alpha, method="fdr_bh")
    else:
        reject, adjusted = np.zeros(0, bool), np.zeros(0)
    threshold = float(raw[reject].max()) if reject.any() else 0.0
    out = {}
    for k in keys:
        cl = [{"start_ms": s, "end_ms": e, "p": p, "p_fdr": float(adjusted[j]), "reject": bool(reject[j])}
              for j, (kk, i) in enumerate(flat) if kk == k for s, e, p in [cells[k][i]]]
        best = min(cl, key=lambda c: c["p"]) if cl else None
        out[k] = {"raw_p": best["p"] if best else 1.0, "adjusted_p": best["p_fdr"] if best else 1.0,
                  "reject": bool(best["reject"]) if best else False, "threshold": threshold,
                  "m": len(raw), "clusters": cl}
    return out


def _bh_min_p(keys, cells, alpha):
    """BH over the smallest cluster p of each test; used only for adjusted_p_12."""
    raw = np.asarray([min([c[2] for c in cells[k]], default=1.0) for k in keys])
    _, adjusted, _, _ = multipletests(raw, alpha=alpha, method="fdr_bh")
    return {k: float(adjusted[i]) for i, k in enumerate(keys)}


def model_brain_fdr(hps="RL23", alpha=0.05):
    """BH-FDR within the RSA family (vOT/ST x PCoder 3) and within the ridge
    family (vOT/ST x PCoder 1--3).

    Returns
    -------
    dict
        Keys are ``(analysis, roi_id, pcoder_index)`` and values contain
        ``raw_p``/``adjusted_p``/``reject`` (smallest cluster of the test),
        ``threshold`` (largest raw cluster p rejected in the family), ``m``
        (clusters in the family), ``clusters`` (all clusters of the test) and
        ``adjusted_p_12`` (joint 12-cell family, smallest cluster p per cell).
    """
    cells = _cells(hps)
    families = [
        [("RSA", roi, 2) for roi in ROIS],
        [("ridge", roi, pcoder) for roi in ROIS for pcoder in PCODERS],
    ]
    results = {}
    for keys in families:
        results.update(_bh(keys, cells, alpha))
    twelve = _bh_min_p(list(cells), cells, alpha)
    for key, value in results.items():
        value["adjusted_p_12"] = twelve[key]
    return results
