########################################################################
# Stimulus-level source time courses of the vertices of one parcel, for the
# stimulus-level RSA (27_stimulus_rsa_partial.py, S13 Fig; S14Fig_stimulus_rdms.py, S14 Fig).
#
# For every participant and condition, the epochs (resampled to 100 Hz) are
# source-localized with the participant's inverse operator (lambda2 = 1 / snr_sti**2,
# normal orientation) and morphed to fsaverage; the time courses of the vertices of the
# parcel are kept for every trial (one trial per stimulus).
# Needs the participant-level MEG data (environment variable YOU2026_PRIVATE_DATA).
#
# Output: {data_dir}/meg_tc/{parcel}_ds100Hz_sti_ver.nc,
#         DataArray (subject, trial = stimulus, vertex, time)
#
#   python 26_meg_vertex_time_courses.py --parcel_ind 40 -j 8   # left vOT
#   python 26_meg_vertex_time_courses.py --parcel_ind 65 -j 8   # left ST
########################################################################
import argparse

import mne
import numpy as np
import xarray as xr
from joblib import Parallel, delayed
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator

from config import SUBJECT, event_id, f_down_sampling, fname, parc, pick_ori, snr_sti, subjects

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--parcel_ind", type=int, required=True, help="40: left vOT, 65: left ST")
parser.add_argument("-j", "--n_jobs", type=int, default=1)
args = parser.parse_args()

LAMBDA2 = 1.0 / snr_sti ** 2
mne.set_config("SUBJECTS_DIR", fname.private_mri_subjects_dir)
labels = [lab for lab in mne.read_labels_from_annot(SUBJECT, parc=parc, verbose=False)
          if "Unknown" not in lab.name]
roi = labels[args.parcel_ind]


def subject_time_courses(subject):
    src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)
    inv = read_inverse_operator(fname.inv(subject=subject), verbose=False)
    morph = mne.compute_source_morph(inv["src"], subject_from=subject, subject_to=SUBJECT,
                                     src_to=src_to, verbose=False)
    data, stimuli, conditions = [], [], []
    for condition in event_id:
        epochs = mne.read_epochs(fname.epo_con(subject=subject, condition=condition), preload=True,
                                 verbose=False).resample(f_down_sampling)
        stcs = [morph.apply(stc) for stc in
                apply_inverse_epochs(epochs, inv, LAMBDA2, pick_ori=pick_ori, verbose=False)]
        lh, rh = stcs[0].vertices
        if roi.hemi == "lh":
            idx = np.searchsorted(lh, np.intersect1d(roi.vertices, lh))
        else:
            idx = len(lh) + np.searchsorted(rh, np.intersect1d(roi.vertices, rh))
        data.append(np.array([stc.data[idx] for stc in stcs]))             # (trial, vertex, time)
        stimuli += epochs.metadata["stimuli"].tolist()
        conditions += [condition[:3]] * len(stcs)
    da = xr.DataArray(np.concatenate(data), dims=["trial", "vertex", "time"],
                      coords={"trial": stimuli, "vertex": idx, "time": stcs[0].times,
                              "condition_type": ("trial", conditions)})
    print(subject, da.shape, flush=True)
    return da.expand_dims("subject").assign_coords(subject=[subject])


out = xr.concat(Parallel(n_jobs=args.n_jobs)(delayed(subject_time_courses)(s) for s in subjects),
                dim="subject")
out.attrs = {"description": f"parcel {args.parcel_ind} stimulus-level vertex time courses",
             "sampling_frequency": f_down_sampling, "lambda2": LAMBDA2}
path = fname.meg_tc_sti_ver(roi=args.parcel_ind, fs=f_down_sampling)
out.to_netcdf(path)
print("saved ->", path)
