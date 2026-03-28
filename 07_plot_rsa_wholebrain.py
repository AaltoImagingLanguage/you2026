# %%
import mne

import xarray as xr

import numpy as np
import matplotlib.pyplot as plt
import os
from config import (
    fname,
    parc,
    metric_rsa,
    time_windows,
)
from utility import create_labels_adjacency_matrix, plot_cluster_label

mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
annotation = mne.read_labels_from_annot("fsaverage", parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]


# For the cluster permutation stats, we need the adjacency between ROIs.
src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)
labels_adjacency_matrix = create_labels_adjacency_matrix(labels, src_to)
import argparse


# metric = "spearman"

# plt.figure(figsize=(6, 4))
##plot
from mne.viz import Brain
import matplotlib as mpl

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--pcoder",
    type=int,
    default=2,
    help="Index of PCoder to plot (0-indexed)",
)

arg = parser.parse_args()
metric = metric_rsa

rsa_all = []
if not os.path.exists(fname.rsa_tc(roi="whole_brain")):
    for roi_ind in range(len(labels)):

        data = xr.load_dataarray(fname.rsa_tc(roi=roi_ind))  # (3,23,51,2))
        time = data.time * 1000
        data = data.expand_dims("parcel_ind")
        data = data.assign_coords(parcel_ind=[roi_ind])
        rsa_all.append(data)
    rsa_all = xr.concat(rsa_all, dim="parcel_ind")
    rsa_all.to_netcdf(
        fname.rsa_tc(
            roi="whole_brain",
        )
    )  # (3,2,51,10)-> (pcoder, fb, time, split)
else:
    print("loading precomputed whole brain rsa scores...")
    rsa_all = xr.load_dataarray(fname.rsa_tc(roi=f"whole_brain"))  #


fig, axs = plt.subplots(3, 5, figsize=(15, 4.5))

norm_bupu = plt.Normalize(vmin=0, vmax=0.8)
cmap_bupu = plt.get_cmap("BuPu")
norm_coolwarm = plt.Normalize(vmin=-0.2, vmax=0.2)
cmap_coolwarm = plt.get_cmap("coolwarm")
j = arg.pcoder
for w in range(len(time_windows)):
    time_window = time_windows[w]
    for i in range(3):
        brain = Brain(
            subject="fsaverage",
            surf="inflated",
            hemi="split",
            views=["lateral", "ventral"],
            view_layout="vertical",
            cortex="0.8",
            background="white",
        )
        if i < 2:
            data = rsa_all.isel(pcoder=j, feedback=i)
            data = data.sel(time=slice(time_window[0], time_window[1])).mean(dim="time")
            data_avg = data.mean(dim="subject")  # (137,)

            colors = cmap_bupu(norm_bupu(data_avg))
            for r, color in enumerate(colors):

                brain.add_label(labels[r], color=color, borders=False, alpha=1)
            brain.add_annotation(
                parc, borders=True, color="k", remove_existing=False, alpha=0.1
            )
            axs[i, 0].set_ylabel("w/o feedback" if i == 0 else "w/ feedback")
        else:
            data = rsa_all.isel(pcoder=j, feedback=1) - rsa_all.isel(
                pcoder=j, feedback=0
            )
            data = data.sel(time=slice(time_window[0], time_window[1]))  # (137,23,10)
            data_avg = data.mean(dim="subject").mean(dim="time")  # (137,)

            colors = cmap_coolwarm(norm_coolwarm(data_avg))
            for r, color in enumerate(colors):

                brain.add_label(labels[r], color=color, borders=False, alpha=1)

            t0, clusters, pvals, _ = mne.stats.spatio_temporal_cluster_1samp_test(
                data.data.transpose(1, 2, 0),  # (23,  10,137,)
                n_permutations=5000,
                tail=1,
                n_jobs=1,
                seed=42,
                adjacency=labels_adjacency_matrix,
                verbose=False,
                buffer_size=None,
            )  # (events, subjects, len(labels), length)

            # We can't call clusters with an associated p-value "significant". We will
            # call them "good" instead.
            good_clusters_idx = np.where(pvals < 0.05)[0]
            good_clusters = [clusters[idx] for idx in good_clusters_idx]
            print("n_clusters=", len(good_clusters))

            for cluster in good_clusters:
                plot_cluster_label(
                    cluster, labels, brain, color="black", width=2, alpha=0.6
                )
            axs[i, 0].set_ylabel("w/ - w/o")
        brain.show_view()
        screenshot = brain.screenshot()
        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        axs[i, w].imshow(cropped_screenshot)
        brain.close()
        axs[0, w].set_title(f"{int(time_window[0]*1000)}-{int(time_window[1]*1000)} ms")

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

cbar1 = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm_bupu, cmap=cmap_bupu),
    ax=axs[0:2, :],
    orientation="vertical",
    shrink=0.5,
)
cbar1.set_label(
    f"RSA score ({metric})",
    rotation=270,
    labelpad=15,
)

# Colorbar for last row (coolwarm colormap)
cbar2 = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm_coolwarm, cmap=cmap_coolwarm),
    ax=axs[2, :],
    orientation="vertical",
)
cbar2.set_label(
    f"RSA score ({metric})",
    rotation=270,
    labelpad=15,
)

plt.savefig(
    fname.fig_rsa_whole(p=j),
    dpi=300,
    bbox_inches="tight",
)
plt.show()
