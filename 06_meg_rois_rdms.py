# %%
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt
import numpy as np
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)

# from utility import get_pmodel
import mne
import xarray as xr
from mne import compute_source_morph
from mne.minimum_norm import read_inverse_operator, apply_inverse
import argparse
from config import (
    event_id,
    f_down_sampling,
    fname,
    parc,
    rois,
    subjects,
    time_windows,
    snr_epoch,
    baseline_window,
    time_len,
    pick_ori,
)
import time

# vOt and ST
mne.set_config("SUBJECTS_DIR", fname.private_mri_subjects_dir)
SUBJECT = "fsaverage"
annotation = mne.read_labels_from_annot("fsaverage", parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]
print("downsampling:", f_down_sampling)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--parcel_id",
    type=str,
    default=65,
    help="ids of the parcel to analyze, range from [0,136], 40->vOT, 65->ST",
)
parser.add_argument(
    "-j",
    "--n-jobs",
    type=int,
    default=3,
    help="number of CPU cores to use",
)
parser.add_argument(
    "--compute",
    type=bool,
    default=False,
    help="whether to compute and save the condition-wise time courses or to load and analyze them",
)
parser.add_argument(
    "--metric",
    type=str,
    default="correlation",
    help="RDM distance metric: cosine, correlation, euclidean",
)

args = parser.parse_args()
roi_id = args.parcel_id
parcel = [parcel for parcel, idx in rois.items() if idx == roi_id][0]
roi = labels[roi_id]

start_time = time.monotonic()


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
    subject_data = []
    print("Reading epochs...")
    epochs = (
        mne.read_epochs(fname.epo(subject=subject))
        .crop(baseline_window[0], time_len)
        .resample(f_down_sampling)
    )

    ave = {
        cat: epochs[cat].average().apply_baseline(baseline_window) for cat in event_id
    }
    for condition in event_id.keys():

        stcs = apply_inverse(
            ave[condition],
            inverse_operator,
            1.0 / snr_epoch**2,
            pick_ori=pick_ori,
            verbose=False,
        )

        stcs_morph = morph.apply(stcs)  # (135,)
        stcs_data = np.array(stcs.data)  # (trials,vertices/n_labels,timepoints)

        vertices_lh, vertices_rh = stcs_morph.vertices
        if roi.hemi == "lh":
            roi_ind = np.searchsorted(
                vertices_lh, np.intersect1d(roi.vertices, vertices_lh)
            )
        else:
            roi_ind = len(vertices_lh) + np.searchsorted(
                vertices_rh, np.intersect1d(roi.vertices, vertices_rh)
            )

        roi_tc = stcs_data[roi_ind, :]
        subject_data.append(roi_tc)
    subject_data = np.array(subject_data)
    subject_combined = xr.DataArray(
        subject_data,
        dims=["trial", "vertex", "time"],
        coords={
            "trial": list(event_id.keys()),
            "vertex": roi_ind,
            "time": stcs_morph.times,
        },
    )
    # Add subject coordinate
    subject_combined = subject_combined.expand_dims("subject")
    subject_combined = subject_combined.assign_coords(subject=[subject])
    return subject_combined


if args.compute:

    # Run parallel processing
    print(f"Processing {len(subjects)} subjects with {args.n_jobs} jobs...")
    all_subject_data = Parallel(n_jobs=args.n_jobs)(
        delayed(compute_roi_time_courses)(subject) for subject in subjects
    )
    exec_time = (time.monotonic() - start_time) / 60
    print(f"Execution time: {exec_time:.2f} minutes")
    # Concatenate all subjects
    print("Concatenating data from all subjects...")
    final_data = xr.concat(all_subject_data, dim="subject")

    # Optional: Add attributes for better documentation
    final_data.attrs = {
        "description": f"{parcel} time courses for all subjects and conditions",
        "sampling_frequency": f_down_sampling,
        "lambda2": snr_epoch,
        "pick_ori": "normal",
    }

    final_data.to_netcdf(fname.meg_tc_con(roi=roi_id))


else:
    # for tmin, tmax in [(start_time+i * duration, start_time+(i + 1) * duration) for i in range(5)]:
    final_data = xr.load_dataarray(fname.meg_tc_con(roi=roi_id))

data_avg = final_data.mean(dim=["subject"]) #(4,34,130)
# Plot RDMs for each time window
rdms = []
names = []
metric = args.metric
import mne_rsa
from utility import plot_meg_rdms

for tmin, tmax in time_windows:
    print(f"Analyzing time window: {tmin} - {tmax} seconds")

    data_avg_time = data_avg.sel(time=slice(tmin, tmax)).mean(dim="time")
    # Compute RDM
    rdm = mne_rsa.compute_rdm(data_avg_time.data, metric=metric)
    rdms.append(rdm)
    names.append(f"{int(tmin*1000)}-{int(tmax*1000)} ms")
fig = plot_meg_rdms(
    rdms,
    names=names,
    title=parcel,
    n_rows=1,
    vmax=0.45,
    colorbar_label=f"{metric} distance",
)

plt.savefig(
    fname.fig_meg_rdms(roi=parcel),
    dpi=300,
    bbox_inches="tight",
)
plt.show()
