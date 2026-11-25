# %%
"""
Ridge regression from PCoder representations to the grand-average MEG time
course of one parcel (Fig 7).

For each 100-ms analysis window (20-ms steps), the MEG source time course
(mean_flip over the parcel, averaged over the 23 participants and over the
window) of the 540 stimuli is predicted from the PCoder 1-3 representations
without (t = 0) and with feedback (t = 25). Score: correlation between predicted
and observed responses in 10-fold cross-validation over stimuli
(ridge.Ridge: z-scoring, RidgeCV with 20 alphas in 10^-3..10^3 chosen within
each training set; no PCA, nc = 0 in config.py). With --n_perm > 0 the
stimulus labels are permuted to build a null distribution.

Inputs:  ../data/meg_tcs/meg_tc_{40,65}_sti.nc (trial-wise parcel time courses;
         computed from the epochs and inverse operators with --compute True)
         ../data/behavior/pcoder_reps_conds_RL23.pkl, ../data/dataset/rep_stimuli.csv
Output:  ../data/ridge_tcs/ridge_tc_{40,65}_perm{n_perm}_RL23.nc
         (plotted by Fig7_ridge.py)

    python 28_ridge_grand_average.py --parcel_ind 40 --hps RL23 -j 16   # left vOT
    python 28_ridge_grand_average.py --parcel_ind 65 --hps RL23 -j 16   # left ST
    python 28_ridge_grand_average.py --parcel_ind 40 --hps RL23 --n_perm 1000 -j 16
"""
from joblib import Parallel, delayed

import numpy as np
import pickle
from ridge import Ridge

import mne
import xarray as xr
import pandas as pd
from mne import compute_source_morph, read_epochs
from mne.minimum_norm import apply_inverse_epochs, read_inverse_operator
import argparse
from config import (
    event_id,
    f_down_sampling,
    fname,
    snr_sti,
    parc,
    subjects,
    time_len,
    time_step,
    time_interval,
    n_splits,
    nc,
    SUBJECT,
    fb_timestep,
)
import time

mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)

annotation = mne.read_labels_from_annot(SUBJECT, parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "-j",
    "--n_jobs",
    type=int,
    default=1,
    help="number of CPU cores to use",
)
parser.add_argument(
    "--compute",
    type=bool,
    default=False,
    help="whether to compute the trial-wise time courses or just load existing data",
)
parser.add_argument(
    "--parcel_ind",
    type=int,
    default=65,
    help="parcel index to use, e.g., 65 for ST and 40 for vOT",
)
parser.add_argument(
    "--n_perm",
    type=int,
    default=0,  # (0 for real result or 1000 for permutation tests)
    help="number of permutations to compute or 0 for no permutation",
)

parser.add_argument(
    "--hps", type=str, default="RL23", choices=["RL1", "RL23"],
    help="hyperparameter set of the model representations: RL1 or RL23 (RL2+RL3, reported)",
)
parser.add_argument(
    "--tsens", action="store_true",
    help="timestep-sensitivity mode: run ridge at model timesteps "
    "[22, 23, 24, 25, 26, 27, 28] (stored as a 'timestep' dim) instead of the 2-state "
    "t=0 vs t=fb_timestep output. Writes a separate _tsens file.",
)

arg, _ = parser.parse_known_args()

# model timesteps read out: sweep for --tsens, else feedforward vs converged.
TIMESTEPS = [22, 23, 24, 25, 26, 27, 28] if arg.tsens else [0, fb_timestep]

roi_id = arg.parcel_ind
roi = labels[roi_id]
# RL23 reads the *_RL23 representations and writes a *_RL23 output file
REPS_PATH = (str(fname.pcoder_reps).replace(".pkl", "_RL23.pkl")
             if arg.hps == "RL23" else fname.pcoder_reps)

# Initialize list to store data from all subjects
all_data = []
start_time = time.monotonic()
n_perm = arg.n_perm
if n_perm:
    np.random.seed(n_perm)


def linear_reg_time_courses(
    data,
    rep_dict,
    events,
    stimuli,
    time_step=time_step,
    time_interval=time_interval,
    nc=nc,
    n_perm=1000,
):
    """Perform ridge regression from model layer activations to MEG time courses."""
    data_avg = data.mean(dim=["subject"])

    times = data_avg.time[::time_step]
    times = times[
        times <= time_len - time_interval
    ]  # make sure the 100 ms time window extracted is within the data range
    times = times[times >= 0]  #
    if n_perm:
        gof_ev_md = np.zeros(
            [
                len(rep_dict),  # pcoders
                len(TIMESTEPS),  # model time steps
                len(times),
                n_splits,
                n_perm,
            ]
        )
    else:
        gof_ev_md = np.zeros(
            [
                len(rep_dict),  # pcoders
                len(TIMESTEPS),  # model time steps
                len(times),
                n_splits,
            ]
        )

    data_reordered = data_avg.sel(trial=stimuli)

    for ii, t in enumerate(times):
        y_scaled = (
            data_reordered.sel(time=slice(t, t + time_interval)).mean(dim="time").data
        )  # (540,)

        for p, pcoder in enumerate(rep_dict):
            reps = []
            for condition in events:
                reps.extend(rep_dict[pcoder][condition])  #
            x_scaled_all = np.array(reps)
            # Read out the selected model timesteps explicitly (correct regardless of how
            # many timesteps the reps file stores; reps hold the full 0..max_timestep range).
            # Default: feed-forward (t=0) vs converged state (t=fb_timestep); --tsens: full sweep.
            for it_i, it in enumerate(TIMESTEPS):
                x_scaled = x_scaled_all[:, it, :]  # (540, feature_dim)

                ridge = Ridge(
                    n_splits,
                    alphas=np.logspace(-3, 3, 20),
                    n_jobs=arg.n_jobs,
                    random_state=100,
                    n_pca=nc,
                    n_perm=n_perm,
                    
                )  # alphas:(10^-3 - 10^3)
                corr2 = ridge.bv_linear(x_scaled, y_scaled)
                gof_ev_md[p, it_i, ii, ...] = corr2

                print(
                    f"Pcoder: {p}, Feedback: {bool(it_i)}, t={it}, Time at {t.data}: Goodness of fit correlation: {gof_ev_md[p,it_i,ii].mean():.4f}"
                )
    # --tsens: a 'timestep' dim over the full sweep; else the 2-state 'fb' dim.
    step_dim = "timestep" if arg.tsens else "fb"
    step_coord = TIMESTEPS if arg.tsens else ["w/o", "w/"]
    if n_perm:
        subject_corr = xr.DataArray(
            gof_ev_md,
            dims=["pcoder", step_dim, "time", "split", "perm"],
            coords={
                "pcoder": range(len(rep_dict)),
                step_dim: step_coord,
                "time": times,
                "split": range(n_splits),
                "perm": range(n_perm),
            },
        )
    else:
        subject_corr = xr.DataArray(
            gof_ev_md,
            dims=[
                "pcoder",
                step_dim,
                "time",
                "split",
            ],
            coords={
                "pcoder": range(len(rep_dict)),
                step_dim: step_coord,
                "time": times,
                "split": range(10),
            },
        )
    return subject_corr


def compute_roi_time_courses(subject):
    """Compute ROI time courses for a single subject across all conditions."""
    print(subject)
    src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)
    inverse_operator = read_inverse_operator(
        fname.inv(subject=subject),
        verbose=False,
    )
    morph = compute_source_morph(
        inverse_operator["src"],
        subject_from=subject,
        subject_to=SUBJECT,
        src_to=src_to,
        verbose=False,
    )
    # Collect all data first, then create single DataArray
    all_roi_data = []
    all_metadata = []
    all_condition_types = []

    for condition in event_id.keys():
        epoch_condition = read_epochs(
            fname.epo_con(subject=subject, condition=condition),
            preload=True,
            verbose=False,
        ).resample(f_down_sampling)
        metadata = epoch_condition.metadata

        # Get evoked data (averaging across trials in sensor space)
        #
        stcs = apply_inverse_epochs(
            epoch_condition,
            inverse_operator,
            1.0 / snr_sti**2,
            pick_ori="normal",
            verbose=False,
        )
        stcs_morph = [morph.apply(stc) for stc in stcs]  # (135,)

        roi_tc = mne.extract_label_time_course(
            stcs_morph, roi, mode="mean_flip", src=src_to, verbose=False
        )

        roi_tc = np.array(roi_tc).squeeze()  # (trials,timepoints)

        all_roi_data.append(roi_tc)
        all_metadata.extend(metadata["stimuli"].tolist())
        all_condition_types.extend([condition[:3]] * roi_tc.shape[0])

    # Concatenate all data along trial dimension
    combined_roi_data = np.concatenate(all_roi_data, axis=0)
    # Create single DataArray with condition_type only varying along trial

    subject_combined = xr.DataArray(
        combined_roi_data,
        dims=["trial", "time"],
        coords={
            "trial": all_metadata,
            "condition_type": (
                "trial",
                all_condition_types,
            ),  # Only along trial dimension
            "time": stcs_morph[0].times,
        },
    )
    subject_combined = subject_combined.expand_dims("subject")
    subject_combined = subject_combined.assign_coords(subject=[subject])

    return subject_combined


if arg.compute:

    # compute trial-wise source time courses in parallel
    print(f"Processing {len(subjects)} subjects with {arg.n_jobs} jobs...")
    all_subject_data = Parallel(n_jobs=arg.n_jobs)(
        delayed(compute_roi_time_courses)(subject) for subject in subjects
    )
    exec_time = (time.monotonic() - start_time) / 60
    print(f"Execution time: {exec_time:.2f} minutes")
    # Concatenate all subjects
    print("Concatenating data from all subjects...")
    data = xr.concat(all_subject_data, dim="subject")

    # Optional: Add attributes for better documentation
    data.attrs = {
        "description": f"{roi_id} time courses for all subjects and trails",
        "sampling_frequency": f_down_sampling,
        "snr": snr_sti,
    }

    data.to_netcdf(fname.meg_tc_sti(roi=roi_id))

else:
    sti_types = list(event_id.keys())
    data = xr.load_dataarray(fname.meg_tc_sti(roi=roi_id))  # (23,540,130)

# Data are averaged by stimulus over participants inside linear_reg_time_courses.
# get model layer features
with open(REPS_PATH, "rb") as f:
    rep_dict = pickle.load(f)

# # Get all unique stimuli
unique_stimuli = pd.read_csv(fname.stimuli_con_dir)
events = unique_stimuli.columns.tolist()
stimuli = []  # (540, 19)
for condition in events:
    stimuli.extend(unique_stimuli[condition].tolist())

final_data = linear_reg_time_courses(data, rep_dict, events, stimuli, n_perm=n_perm)

exec_time = (time.monotonic() - start_time) / 60
print(f"Execution time: {exec_time:.2f} minutes")
print("Concatenating data from all subjects...")

# Optional: Add attributes for better documentation
final_data.attrs = {
    "description": f"ridge regression from layer activations to grand average source activation in {roi_id}",
    "time step": time_step,
    "time interval": time_interval,
    "n_components": nc,
}
ridge_out = str(fname.ridge_tc_sti(roi=roi_id, perm=n_perm))
if arg.hps == "RL23":
    ridge_out = ridge_out.replace(".nc", "_RL23.nc")
if arg.tsens:
    ridge_out = ridge_out.replace(".nc", "_tsens.nc")
final_data.to_netcdf(ridge_out)
print("saved ->", ridge_out)
