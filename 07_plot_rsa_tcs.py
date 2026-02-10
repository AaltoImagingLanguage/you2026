"""Plot RSA time courses for different PCoders in a given ROI,  comparing feedback vs. no-feedback conditions."""

# %%
import mne
import xarray as xr

import numpy as np
import matplotlib.pyplot as plt
from config import (
    fname,
    subjects,
    rois,
    metric_rsa,
    fb_colors,
)

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--parcel_ind",
    type=int,
    default=65,
    help="id of the parcel to analyze, e.g., 40 for vOT, 65 for ST",
)

arg = parser.parse_args()

roi_id = arg.parcel_ind  # vOT, ST

# load data
data = xr.load_dataarray(fname.rsa_tc(roi=roi_id))
print(
    "loaded data from:", fname.rsa_tc(roi=roi_id)
)  # (pcoder, subject, time, feedback)
times = data.time.values

fig, axes = plt.subplots(1, data.shape[0], figsize=(12, 3), sharex=True, sharey=True)
for p in range(data.shape[0]):

    mean_data = data.isel(pcoder=p).mean(dim=["subject"])
    sem_data = data.isel(pcoder=p).std(dim=["subject"], ddof=1) / np.sqrt(len(subjects))

    for n in range(2):

        axes[p].plot(
            times * 1000,
            mean_data.data[:, n],
            color=fb_colors[n],
            label=["w/ feedback" if n else "w/o feedback"] if p > 1 else None,
        )
        axes[p].fill_between(
            times * 1000,
            mean_data[:, n] - sem_data[:, n],
            mean_data[:, n] + sem_data[:, n],
            color=fb_colors[n],
            alpha=0.3,
        )

    t_obs, clusters, pvals, H0 = mne.stats.permutation_cluster_1samp_test(
        data.isel(pcoder=p, feedback=1).data - data.isel(pcoder=p, feedback=0).data,
        n_permutations=5000,
        tail=1,
        seed=50,
    )  # (events,subjects,len(rois), length)

    good_clusters_idx = np.where(pvals < 0.05)[0]
    good_clusters = [clusters[idx] for idx in good_clusters_idx]
    for jj in range(len(good_clusters)):
        axes[p].plot(
            times[good_clusters[jj]] * 1000,
            [-0.3] * len(good_clusters[jj][0]),
            "s",
            color="k",
            alpha=1,
            markersize=4,
        )

    axes[p].set_xlabel("Time (ms)")
    if p == 0:
        axes[p].set_ylabel(f"RSA score ({metric_rsa})")
    axes[p].set_title(f"PCoder {p+1}")
fig.legend(
    bbox_to_anchor=(1, 1),
)
fig.suptitle(
     f"{[k for k, v in rois.items() if v == roi_id][0]}",
    y=1.05,
    x=0.05,
    ha="left",
    fontweight="bold",
)

plt.savefig(
    fname.fig_rsa_roi(roi=f"{[k for k, v in rois.items() if v == roi_id][0]}_rl1"),
    dpi=300,
    bbox_inches="tight",
)

plt.show()
print("done")
