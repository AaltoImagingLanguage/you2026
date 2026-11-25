########################################################################
# Single-trial MEG source patterns, the input of the condition-wise RSA
# (18_rsa_parcels.py; Figs 5-6, Table 2) and of the reliability, bootstrap,
# cross-validated RDM and condition-label analyses (23_rsa_condition_baselines.py,
# 24_rsa_reliability_bootstrap_cv.py; S6, S8, S9 Figs). Requires the
# participant-level MEG data (YOU2026_PRIVATE_DATA, see config.py).
#
# Epochs (baseline -200-0 ms, resampled to 100 Hz) are source-localized per trial
# (MNE, lambda2 = 1/snr_epoch**2, normal orientation) and morphed to fsaverage. Because
# averaging, the inverse and the morph are linear, the mean of the single-trial patterns
# within a condition equals the morphed evoked response; the script verifies this at
# the end for two participants in parcels 40 (left vOT) and 65 (left ST).
# Each trial is averaged in the 100-ms windows (20-ms steps, 51 windows) of the RSA
# time courses. Trials are stored in the fixed stimulus order of
# ../data/dataset/rep_stimuli.csv (16_model_rdms.py).
#
#   python 17_meg_trial_patterns.py -j 4          # all participants, then parcels
#   python 17_meg_trial_patterns.py --subject sub-01 --overwrite   # one participant only
#   python 17_meg_trial_patterns.py --parcels-only
#
# Outputs:
#   ../data/meg_trial_patterns/{subject}.npy       (trial, vertex, window)
#   ../data/meg_trial_patterns/parcel{p}.npz       X: (subject, 540, vertex, window)
#   ../data/meg_trial_patterns/stimulus_metadata.csv
########################################################################
import argparse
import os
from pathlib import Path

import mne
import numpy as np
import pandas as pd
import xarray as xr
from joblib import Parallel, delayed
from mne import compute_source_morph
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

from config import (baseline_window, f_down_sampling, fname, parc, pick_ori,
                    snr_epoch, subjects, time_interval, time_len, time_step)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("-j", "--n-jobs", type=int, default=4)
parser.add_argument("--subject", help="extract only this subject and exit (cluster arrays)")
parser.add_argument("--overwrite", action="store_true",
                    help="re-extract subjects whose pattern file already exists")
parser.add_argument("--parcels-only", action="store_true",
                    help="skip the per-subject extraction and only (re)build parcel files")
args = parser.parse_args()

OUT = Path(f"{fname.data_dir}/meg_trial_patterns")
OUT.mkdir(parents=True, exist_ok=True)

mne.set_config("SUBJECTS_DIR", fname.private_mri_subjects_dir)
annotation = mne.read_labels_from_annot("fsaverage", parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]

# fixed stimulus order shared with the model representations (16_model_rdms.py)
csv = pd.read_csv(fname.stimuli_con_dir)
conditions = csv.columns.tolist()
stimuli = [w for c in conditions for w in csv[c].tolist()]       # 540
stim_index = {w: i for i, w in enumerate(stimuli)}


def window_samples(times):
    """Sample indices of the 100-ms RSA windows
    (xarray's inclusive time slices at 20-ms steps, last window starting at
    or before time_len - time_interval). Times are rounded to the 10-ms grid
    first; with raw float times the slices drop the end sample (90-ms windows)."""
    times = np.round(times, 3)
    t_all = xr.DataArray(np.arange(len(times)), coords={"time": times}, dims="time")
    t_all = t_all.sel(time=slice(0, time_len))
    starts = t_all.time[::time_step]
    starts = starts[starts <= time_len - time_interval]
    windows = [t_all.sel(time=slice(t, round(t + time_interval, 3))).values
               for t in starts.values]
    return starts.values, windows


def extract_subject(subject):
    out_path = OUT / f"{subject}.npy"
    if out_path.exists() and not args.overwrite:
        return subject
    src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)
    inverse_operator = read_inverse_operator(fname.inv(subject=subject), verbose=False)
    morph = compute_source_morph(inverse_operator["src"], subject_from=subject,
                                 subject_to="fsaverage", src_to=src_to, verbose=False)
    epochs = (mne.read_epochs(fname.epo(subject=subject), verbose=False)
              .crop(baseline_window[0], time_len)
              .resample(f_down_sampling)
              .apply_baseline(baseline_window))
    starts, windows = window_samples(epochs.times)
    words = epochs.metadata["stimuli"].tolist()

    stcs = apply_inverse_epochs(epochs, inverse_operator, 1.0 / snr_epoch**2,
                                pick_ori=pick_ori, return_generator=True, verbose=False)
    n_vertices = sum(len(s["vertno"]) for s in src_to)
    patterns = np.full((len(stimuli), n_vertices, len(windows)), np.nan, np.float32)
    for word, stc in zip(words, stcs):
        data = morph.apply(stc).data
        patterns[stim_index[word]] = np.stack(
            [data[:, w].mean(1) for w in windows], axis=1)
    np.save(OUT / f"{subject}.tmp.npy", patterns)
    os.replace(OUT / f"{subject}.tmp.npy", out_path)
    epochs.metadata.to_csv(OUT / f"{subject}_metadata.csv", index=False)
    print(subject, "done", flush=True)
    return subject


def label_positions(label, src):
    hemi = 0 if label.hemi == "lh" else 1
    vertno = np.asarray(src[hemi]["vertno"])
    pos = np.searchsorted(vertno, np.intersect1d(label.vertices, vertno))
    return pos + (len(src[0]["vertno"]) if hemi else 0)


if args.subject:
    extract_subject(args.subject)
    raise SystemExit
if not args.parcels_only:
    Parallel(n_jobs=args.n_jobs)(delayed(extract_subject)(s) for s in subjects)

# stimulus properties (identical across subjects) for condition-label baselines
meta = pd.concat([pd.read_csv(OUT / f"{s}_metadata.csv") for s in subjects])
meta = meta.drop_duplicates("stimuli").set_index("stimuli").reindex(stimuli)
meta.index.name = "stimuli"
meta["condition"] = [c for c in conditions for _ in csv[c]]
meta.to_csv(OUT / "stimulus_metadata.csv")

# regroup per parcel: (subject, 540, vertex, window)
src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)
epochs_times = np.round(np.arange(-0.2, time_len, 1 / f_down_sampling), 2)
starts, _ = window_samples(epochs_times)
subject_data = [np.load(OUT / f"{s}.npy", mmap_mode="r") for s in subjects]
for p, label in enumerate(labels):
    pos = label_positions(label, src_to)
    X = np.stack([d[:, pos, :] for d in subject_data])
    np.savez(OUT / f"parcel{p}.npz", X=X, vertices=pos, times=starts,
             subjects=np.asarray(subjects), stimuli=np.asarray(stimuli),
             conditions=np.asarray(conditions), label=label.name)
print("saved parcel files ->", OUT)

# Verification: condition means of the single trials == the morphed evoked response.
from mne.minimum_norm import apply_inverse
for subject in subjects[:2]:
    si = subjects.index(subject)
    inverse_operator = read_inverse_operator(fname.inv(subject=subject), verbose=False)
    morph = compute_source_morph(inverse_operator["src"], subject_from=subject,
                                 subject_to="fsaverage", src_to=src_to, verbose=False)
    epochs = (mne.read_epochs(fname.epo(subject=subject), verbose=False)
              .crop(baseline_window[0], time_len).resample(f_down_sampling))
    _, windows = window_samples(epochs.times)
    for p in (40, 65):
        X = np.load(OUT / f"parcel{p}.npz")["X"]
        pos = label_positions(labels[p], src_to)
        for ci, c in enumerate(conditions):
            evoked = epochs[c].average().apply_baseline(baseline_window)
            stc = morph.apply(apply_inverse(evoked, inverse_operator, 1.0 / snr_epoch**2,
                                            pick_ori=pick_ori, verbose=False))
            ref = np.stack([stc.data[pos][:, w].mean(1) for w in windows], axis=1)
            trial_mean = np.nanmean(X[si, ci * 135:(ci + 1) * 135], axis=0)
            r = np.corrcoef(trial_mean.ravel(), ref.ravel())[0, 1]
            print(f"{subject} parcel {p} {c}: r(trial mean, morphed evoked) = {r:.6f}")
