"""
plot the time course of ridge regression prediction scores for a given parcel
"""

import mne
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

import numpy as np
import mne
import xarray as xr
import argparse
from config import fname, parc, SUBJECT, n_splits, fb_colors
import time

mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
annotation = mne.read_labels_from_annot(SUBJECT, parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--parcel_ind",
    type=int,
    default=40,
    help="parcel index to use, e.g., 40 for vOT, 65 for ST",
)

arg = parser.parse_args()

default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]

roi_id = arg.parcel_ind
roi = labels[roi_id]

# Initialize list to store data from all subjects
all_data = []
start_time = time.monotonic()

compute = False  # whether to compute and save the whole brain data
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
    y0_perm = data_perm.sel(pcoder=pcoder, fb="w/o")  # (51,10,1000)

    significance_threshold = np.sort(y0_perm, axis=-1)[:, int(0.95 * y0_perm.shape[-1])]

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
    y1_perm = data_perm.sel(pcoder=pcoder, fb="w/")  # (51,10,1000)

    significance_threshold = np.sort(y1_perm, axis=-1)[:, int(0.95 * y1_perm.shape[-1])]

    where = np.where(
        np.array(y1_mean) > significance_threshold
    )  # pcoder =2: (480,500., 520., 540., 560) ms , pcoder1: (440., 460., 480., 500., 520.) ms, pcoder
    sig_time = time[where]
    print(sig_time.data)
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
    a3 = ax.plot(
        time,
        significance_threshold,
        color=fb_colors[1],
        linestyle="--",
        label="w/ feedback - significance threshold",
    )
    if pcoder == 2:
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
    t_obs, clusters, pvals, H0 = mne.stats.permutation_cluster_1samp_test(
        y1.data.T - y0.data.T,
        n_permutations=5000,
        tail=1,
    )

    good_clusters_idx = np.where(pvals < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    for jj in range(len(good_clusters)):
        plt.plot(
            time[good_clusters[jj]],
            [0.25] * len(good_clusters[jj][0]),
            "s",
            color="k",
            markersize=4,
        )
        print(
            "intersection",
            jj,
            np.intersect1d(time[good_clusters[jj]].data, sig_time.data),
        )
    ax.set_xlabel("Time (ms)")
    ax.set_title(f"PCoder {pcoder+1}")
    if pcoder == 0:
        ax.set_ylabel("Prediction score (correlation)")


fig.suptitle(roi_id, y=1.05, fontweight="bold")
plt.savefig(
    fname.fig_ridge_tc(roi=roi_id),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
print("done")

# %%
