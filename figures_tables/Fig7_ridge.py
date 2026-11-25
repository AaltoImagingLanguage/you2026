"""
Fig 7: time courses of the ridge-regression prediction scores of one parcel.

Plots the cross-validated prediction score (correlation) of the grand-average
MEG response from PCoder 1-3 without and with feedback, mean +- SEM over the 10
CV folds. If the permutation null (28_ridge_grand_average.py --n_perm 1000) is
available, the dashed lines show its 95th percentile (chance-level threshold).

Inputs:  ../data/ridge_tcs/ridge_tc_{40,65}_perm0_RL23.nc and, if present,
         ../data/ridge_tcs/ridge_tc_{40,65}_perm1000_RL23.nc (28_ridge_grand_average.py)
Output:  figures/ridge/ridge_{40,65}_RL23.pdf

    python figures_tables/Fig7_ridge.py --parcel_ind 40 --hps RL23   # left vOT
    python figures_tables/Fig7_ridge.py --parcel_ind 65 --hps RL23   # left ST
"""

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
import os

import mne
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import argparse
from config import fname, parc, SUBJECT, n_splits, fb_colors, rois
import time

mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
annotation = mne.read_labels_from_annot(SUBJECT, parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--parcel_ind",
    type=int,
    default=65,
    help="parcel index to use, e.g., 40 for vOT, 65 for ST",
)
parser.add_argument(
    "--hps", type=str, default="RL23", choices=["RL1", "RL23"],
    help="hyperparameter set of the model representations: RL23 (RL2+RL3, reported) "
    "or RL1. RL23 reads the per-parcel _RL23 result and writes a _RL23 figure; the "
    "dashed chance-level thresholds are drawn only if the RL23 permutation null "
    "(perm=1000) is available.",
)

arg = parser.parse_args()

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

roi_id = arg.parcel_ind
roi = labels[roi_id]

# Initialize list to store data from all subjects
all_data = []
start_time = time.monotonic()

compute = False  # True: gather the per-parcel RL1 results into one whole-brain file
region_name = "whole_brain"

if compute:

    for perm in [0, 1000]:
        data_all = []
        for roi_id in range(len(labels)):
            data = xr.load_dataarray(
                fname.ridge_tc_sti(roi=roi_id, perm=perm)
            )  # (3,2,51,10)-> (pcoder, fb, time, split)
            if perm == 1000:
                data = data.mean(dim="split")  # (3,2,51)
            data = data.expand_dims("parcel_ind")
            data = data.assign_coords(parcel_ind=[roi_id])

            data_all.append(data)
        data_all = xr.concat(data_all, dim="parcel_ind")
        data_all.to_netcdf(fname.ridge_tc_sti(roi=region_name, perm=perm))

# data loading
if arg.hps == "RL23":
    # read the per-parcel RL23 result; the dashed chance-level thresholds are
    # drawn only if the permutation null (perm=1000) exists.
    ridge_path = str(fname.ridge_tc_sti(roi=roi_id, perm=0)).replace(".nc", "_RL23.nc")
    data = xr.load_dataarray(ridge_path)  # (pcoder, fb, time, split)
    print("loaded data from:", ridge_path)
    perm_path = str(fname.ridge_tc_sti(roi=roi_id, perm=1000)).replace(".nc", "_RL23.nc")
    data_perm = xr.load_dataarray(perm_path) if os.path.exists(perm_path) else None
    if data_perm is None:
        print("no RL23 permutation null found -> omitting dashed chance thresholds")
    elif "split" in data_perm.dims:
        # per-parcel null keeps the split dim; average over it so the 95th
        # percentile is taken over permutations.
        data_perm = data_perm.mean("split")  # (pcoder, fb, time, perm)
else:
    data = xr.load_dataarray(fname.ridge_tc_sti(roi=region_name, perm=0)).sel(
        parcel_ind=roi_id
    )
    data_perm = xr.load_dataarray(fname.ridge_tc_sti(roi=region_name, perm=1000)).sel(
        parcel_ind=roi_id
    )  # (3,2 ,51,10, 1000)


time = data.time * 1000

fig, axes = plt.subplots(1, data.shape[0], figsize=(12, 3), sharex=True, sharey=True)
fig.subplots_adjust(left=0.09)
for pcoder in range(data.shape[0]):
    ax = axes[pcoder]
    y0 = data.sel(pcoder=pcoder, fb="w/o")  # (51,10)
    y0_mean = y0.mean(dim=["split"])  # (51,)
    y0_sem = y0.std(dim=["split"], ddof=1) / np.sqrt(n_splits)

    a0 = ax.plot(
        time,
        y0_mean,
        color=fb_colors[0],
        label="w/o feedback - mean",
    )
    ax.fill_between(
        time,
        y0_mean - y0_sem,
        y0_mean + y0_sem,
        color=fb_colors[0],
        alpha=0.3,
    )
    a1 = None
    if data_perm is not None:
        y0_perm = data_perm.sel(pcoder=pcoder, fb="w/o")  # (51,10,1000)
        significance_threshold = np.sort(y0_perm, axis=-1)[
            :, int(0.95 * y0_perm.shape[-1])
        ]
        a1 = ax.plot(
            time,
            significance_threshold,
            color=fb_colors[0],
            linestyle="--",
            label="w/o feedback - significance threshold",
        )

    y1 = data.sel(pcoder=pcoder, fb="w/")  # (51,10)
    y1_mean = y1.mean(dim=["split"])  # (51,)
    y1_sem = y1.std(dim=["split"], ddof=1) / np.sqrt(n_splits)

    a2 = ax.plot(
        time,
        y1_mean,
        color=fb_colors[1],
        label="w/ feedback - mean",
    )
    ax.fill_between(
        time,
        y1_mean - y1_sem,
        y1_mean + y1_sem,
        color=fb_colors[1],
        alpha=0.3,
    )
    a3 = None
    sig_time = None
    if data_perm is not None:
        y1_perm = data_perm.sel(pcoder=pcoder, fb="w/")  # (51,10,1000)
        significance_threshold = np.sort(y1_perm, axis=-1)[
            :, int(0.95 * y1_perm.shape[-1])
        ]
        where = np.where(
            np.array(y1_mean) > significance_threshold
        )  # pcoder =2: (480,500., 520., 540., 560) ms , pcoder1: (440., 460., 480., 500., 520.) ms
        sig_time = time[where]
        print(sig_time.data)
        a3 = ax.plot(
            time,
            significance_threshold,
            color=fb_colors[1],
            linestyle="--",
            label="w/ feedback - significance threshold",
        )
    if pcoder == 2:
        if data_perm is not None:
            legend1 = ax.legend(
                [a0[0], a1[0]],
                ["mean", "significance threshold"],
                title="w/o feedback",
                loc="upper right",
                bbox_to_anchor=(1.68, 1),
            )
            legend1.get_title().set_fontweight("bold")

            # Create second legend for Group 2
            legend2 = ax.legend(
                [a2[0], a3[0]],
                ["mean", "significance threshold"],
                title="w/ feedback",
                loc="upper right",
                bbox_to_anchor=(1.68, 0.7),
            )
            legend2.get_title().set_fontweight("bold")

            ax.add_artist(legend1)
        else:
            ax.legend(
                [a0[0], a2[0]],
                ["w/o feedback", "w/ feedback"],
                loc="upper right",
                bbox_to_anchor=(1.68, 1),
            )
    # t_obs, clusters, pvals, H0 = mne.stats.permutation_cluster_1samp_test(
    #     y1.data.T - y0.data.T,
    #     n_permutations=5000,
    #     tail=1,
    # )

    # good_clusters_idx = np.where(pvals < 0.05)[0]
    # good_clusters = [clusters[idx] for idx in good_clusters_idx]
    # for jj in range(len(good_clusters)):
    #     plt.plot(
    #         time[good_clusters[jj]],
    #         [0.25] * len(good_clusters[jj][0]),
    #         "s",
    #         color="k",
    #         markersize=4,
    #     )
    #     if sig_time is not None:
    #         print(
    #             "intersection",
    #             jj,
    #             np.intersect1d(time[good_clusters[jj]].data, sig_time.data),
    #         )
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"PCoder {pcoder+1}")
    if pcoder == 0:
        ax.set_ylabel("Prediction score (correlation)")


# fig.suptitle(roi_id, y=1.05, fontweight="bold")
fig.suptitle(
     f"{[k for k, v in rois.items() if v == roi_id][0]}",
    y=1.0,
    x=0.1,
    ha="left",
    fontweight="bold",
)
fig_roi = f"{roi_id}_RL23" if arg.hps == "RL23" else roi_id
plt.savefig(
    fname.fig_ridge_tc(roi=fig_roi),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
print("done")

# %%
