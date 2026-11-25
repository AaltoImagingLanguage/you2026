"""Shared helpers for the condition-wise RSA on single-trial MEG patterns
(18_rsa_parcels.py, S5Fig_meg_rdms.py, 21_rsa_controls.py, 23_-25_, 30_dynamics_rsa.py,
32_hp_perturbation_rsa.py). Library module; not run directly.

Everything works on four-condition RDMs stored as condensed vectors in the
pair order of scipy/mne_rsa: (RW,RL1) (RW,RL2) (RW,RL3) (RL1,RL2) (RL1,RL3)
(RL2,RL3). MEG RDMs are correlation distances between condition-mean vertex
patterns (as in the main RSA); RSA is the Spearman correlation between RDMs.
"""
import pickle
from pathlib import Path

import mne
import numpy as np
from scipy.stats import rankdata

from config import fb_timestep, fname, parc, time_windows

TRIAL_DIR = Path(f"{fname.data_dir}/meg_trial_patterns")
CONDITIONS = ["RW", "RL1PW", "RL2PW", "RL3PW"]
N_PER_CONDITION = 135
PAIRS = np.array([(i, j) for i in range(4) for j in range(i + 1, 4)])
STATES = {"w/o": 0, "w/": fb_timestep}
ROIS = {"vOT": 40, "ST": 65}


def cortical_labels():
    mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
    annotation = mne.read_labels_from_annot("fsaverage", parc=parc, verbose=False)
    return [label for label in annotation if "Unknown" not in label.name]


def parcel_adjacency(labels):
    """Hemisphere-correct parcel adjacency (as in 19_rsa_wholecortex.py)."""
    from scipy import sparse
    src = mne.read_source_spaces(fname.fsaverage_src, verbose=False)
    vertex_adjacency = mne.spatial_src_adjacency(src, verbose=False).tocoo()
    vertex_to_parcel = np.full(vertex_adjacency.shape[0], -1, dtype=int)
    for p, label in enumerate(labels):
        vertex_to_parcel[label_positions(label, src)] = p
    rows = vertex_to_parcel[vertex_adjacency.row]
    cols = vertex_to_parcel[vertex_adjacency.col]
    keep = (rows >= 0) & (cols >= 0)
    adj = sparse.coo_matrix((np.ones(int(keep.sum()), np.int8), (rows[keep], cols[keep])),
                            shape=(len(labels), len(labels))).tocsr()
    adj = adj.maximum(adj.T)
    adj.data[:] = 1
    adj.setdiag(0)
    adj.eliminate_zeros()
    return adj


def label_positions(label, src):
    """Global positions of a label's vertices in the fsaverage source space."""
    hemi = 0 if label.hemi == "lh" else 1
    vertno = np.asarray(src[hemi]["vertno"])
    pos = np.searchsorted(vertno, np.intersect1d(label.vertices, vertno))
    return pos + (len(src[0]["vertno"]) if hemi else 0)


def condition_of_item():
    return np.repeat(np.arange(4), N_PER_CONDITION)


# ---------------------------------------------------------------- MEG patterns
def load_parcel(parcel):
    """Single-trial window-averaged patterns, X: (subject, 540, vertex, time)."""
    archive = np.load(TRIAL_DIR / f"parcel{parcel}.npz")
    X = archive["X"]
    valid = ~np.isnan(X[:, :, 0, 0])
    return np.nan_to_num(X), valid, archive["times"]


def condition_means(X0, valid, weights):
    """Weighted condition means of single-trial patterns.

    X0: (subject, item, vertex, time) with missing trials set to 0;
    valid: (subject, item); weights: (4, item) non-negative item weights
    (an indicator for a stimulus partition, or bootstrap counts).
    Returns (subject, 4, vertex, time).
    """
    n_sub, n_item = valid.shape
    sums = weights.astype(X0.dtype) @ X0.reshape(n_sub, n_item, -1)  # (sub, 4, v*t)
    counts = valid.astype(float) @ weights.T                # (sub, 4)
    means = sums / counts[:, :, None]
    return means.reshape(n_sub, 4, *X0.shape[2:])


def _zscore_vertices(patterns):
    """z-score condition patterns across vertices (axis -2)."""
    centred = patterns - patterns.mean(-2, keepdims=True)
    return centred / np.sqrt((centred**2).mean(-2, keepdims=True))


def correlation_rdm(means):
    """Correlation-distance RDMs: (..., 4, vertex, time) -> (..., time, 6)."""
    z = _zscore_vertices(means)
    i, j = PAIRS.T
    return 1 - (z[..., i, :, :] * z[..., j, :, :]).mean(-2).swapaxes(-1, -2)


def crossval_euclidean_rdm(means_a, means_b):
    """Cross-validated squared Euclidean distance between two independent
    partitions (unbiased: expectation 0 when two conditions do not differ)."""
    i, j = PAIRS.T
    diff_a = means_a[..., i, :, :] - means_a[..., j, :, :]
    diff_b = means_b[..., i, :, :] - means_b[..., j, :, :]
    return (diff_a * diff_b).mean(-2).swapaxes(-1, -2)


def crossval_correlation_rdm(means_a, means_b):
    """Cross-validated correlation distance: mean within-condition split
    correlation minus between-condition split correlation (expectation 0 when
    two conditions share the same pattern)."""
    za, zb = _zscore_vertices(means_a), _zscore_vertices(means_b)
    corr = np.einsum("...cvt,...dvt->...tcd", za, zb) / za.shape[-2]
    corr = (corr + corr.swapaxes(-1, -2)) / 2
    i, j = PAIRS.T
    within = (corr[..., i, i] + corr[..., j, j]) / 2
    return within - corr[..., i, j]


# ------------------------------------------------------------------- statistics
def rank(x):
    return rankdata(x, axis=-1)


def spearman(data_rdms, model_rdms):
    """Spearman between data RDMs (..., 6) and model RDMs (k, 6) -> (..., k)."""
    a = rank(data_rdms)
    b = rank(model_rdms)
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    num = a @ b.T
    den = np.sqrt((a**2).sum(-1))[..., None] * np.sqrt((b**2).sum(-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return num / den


def partial_spearman(data_rdms, model_rdm, covariate_rdms):
    """Partial Spearman between data RDMs (..., 6) and one model RDM (6,),
    controlling for covariate RDMs (m, 6): ranks of both are residualized on
    the covariate ranks (plus intercept), then correlated."""
    design = np.column_stack([np.ones(6), rank(np.atleast_2d(covariate_rdms)).T])
    hat = design @ np.linalg.pinv(design)
    resid = lambda r: r - r @ hat.T  # noqa: E731 (hat is symmetric)
    a = resid(rank(data_rdms))
    b = resid(rank(model_rdm))
    den = np.sqrt((a**2).sum(-1)) * np.sqrt((b**2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return (a @ b) / den


def noise_ceiling(subject_rdms):
    """Spearman noise ceiling (Nili et al., 2014): subject_rdms (subject, ..., 6)
    -> lower and upper bounds (...)."""
    ranks = rank(subject_rdms)
    total = ranks.sum(0)
    n = ranks.shape[0]
    upper = np.mean([_pearson(ranks[s], total / n) for s in range(n)], 0)
    lower = np.mean([_pearson(ranks[s], (total - ranks[s]) / (n - 1)) for s in range(n)], 0)
    return lower, upper


def _pearson(a, b):
    a = a - a.mean(-1, keepdims=True)
    b = b - b.mean(-1, keepdims=True)
    return (a * b).sum(-1) / np.sqrt((a**2).sum(-1) * (b**2).sum(-1))


def window_mask(times, window):
    """RSA window starts inside a 200-ms analysis window (inclusive, as in the
    whole-cortex cluster tests)."""
    return (times >= window[0] - 1e-9) & (times <= window[1] + 1e-9)


def window_labels():
    return [f"{int(w[0] * 1000)}-{int(w[1] * 1000)}" for w in time_windows]


# --------------------------------------------------------------- model + labels
def model_item_reps(hps="RL23"):
    """Item-level model representations at the two states, cached as npz:
    {pcoder: {state: (540, features)}} in the fixed stimulus order."""
    cache = TRIAL_DIR / f"model_item_reps_{hps}.npz"
    if not cache.exists():
        path = (str(fname.pcoder_reps).replace(".pkl", "_RL23.pkl")
                if hps == "RL23" else fname.pcoder_reps)
        with open(path, "rb") as f:
            rep_dict = pickle.load(f)
        arrays = {}
        for p in rep_dict:
            reps = np.concatenate([np.asarray(rep_dict[p][c]) for c in CONDITIONS])
            for state, t in STATES.items():
                arrays[f"{p}|{state}"] = reps[:, t, :].astype(np.float32)
        np.savez(cache, **arrays)
    archive = np.load(cache)
    out = {}
    for key in archive.files:
        p, state = key.split("|")
        out.setdefault(p, {})[state] = archive[key]
    return out


def model_rdms(item_reps, weights=None):
    """Condition-level correlation-distance RDMs of the model from item
    representations (optionally with item weights (4, 540)).
    Returns {f"{pcoder} {state}": (6,)}."""
    if weights is None:
        weights = np.zeros((4, 4 * N_PER_CONDITION))
        weights[condition_of_item(), np.arange(weights.shape[1])] = 1
    weights = weights / weights.sum(1, keepdims=True)
    rdms = {}
    for p, states in item_reps.items():
        for state, reps in states.items():
            means = weights @ reps                              # (4, features)
            rdms[f"{p} {state}"] = correlation_rdm(means[:, :, None])[0]
    return rdms


def label_rdms():
    """Condition-label baseline RDMs, defined by the design alone. Severity:
    |difference in number of replaced letters|; lexicality: real word vs
    misspelled word."""
    i, j = PAIRS.T
    return {"severity": np.abs(np.arange(4)[i] - np.arange(4)[j]).astype(float),
            "lexicality": ((i == 0) != (j == 0)).astype(float)}
