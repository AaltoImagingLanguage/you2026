# %%
import mne
import os
import xarray as xr
from statsmodels.stats.multitest import multipletests
import numpy as np
from config import fname, parc, SUBJECT, ctn_tps

mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
annotation = mne.read_labels_from_annot(SUBJECT, parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]

from scipy.ndimage import label

import matplotlib.pyplot as plt

compute = False  # whether to compute and save the whole brain data
region_name = "whole_brain"


# load whole brain data
data_all = xr.load_dataarray(
    fname.ridge_tc_sti(roi=region_name, perm=0)
)  # (137,3,2,51)-> (parcels, pcoders, fb, )

data_all_perm = xr.load_dataarray(
    fname.ridge_tc_sti(roi=region_name, perm=1000)
)  # (137,3,2,51, 1000)-> (parcels, pcoders, fb, perms )

from mne.viz import Brain
import matplotlib as mpl

cmap0 = plt.get_cmap("BuPu")
norm0 = plt.Normalize(vmin=0.08, vmax=0.15)
cmap1 = plt.get_cmap("GnBu")
norm1 = plt.Normalize(vmin=0, vmax=0.1)

fig, axs = plt.subplots(4, 3, figsize=(9, 6))
for j in range(3):
    validr_dir = {}
    results = []
    for i in range(4):

        brain = Brain(
            subject=SUBJECT,
            surf="inflated",
            hemi="split",
            views=["lateral", "ventral"],
            view_layout="vertical",
            cortex="0.8",
            background="white",
        )

        if i < 2:

            data = data_all.isel(pcoder=j, fb=i)
            data = data.mean(dim="split")  # （137,51）

            data_perm = data_all_perm.isel(pcoder=j, fb=i)  # （137,51,1000）

            significance_threshold = np.sort(data_perm, axis=-1)[
                ..., int(0.95 * data_perm.shape[-1])
            ]  # (137,51)
            mask = data > significance_threshold
            mask_new = np.zeros_like(mask, dtype=bool)

            # Process each parcel (row) separately
            for ii in range(mask.shape[0]):
                labeled, n = label(mask[ii])
                for jj in range(1, n + 1):
                    if np.sum(labeled == jj) > 3:  # ≥3 consecutive True
                        mask_new[ii, labeled == jj] = True
            masked_data = np.where(mask_new, data, np.nan)
            result = np.nanmean(masked_data, axis=1)
            result = np.nan_to_num(result, nan=0.0)  # (137)
            colors = cmap0(norm0(result))
            results.append(result)
            for r, color in enumerate(colors):
                if result[r] > 0:
                    if r in validr_dir:
                        validr_dir[r][i] = np.where(~np.isnan(masked_data[r]))[0]
                    else:
                        # if pval_correct[r] < 0.05:
                        validr_dir[r] = {i: np.where(~np.isnan(masked_data[r]))[0]}
                    brain.add_label(labels[r], color=color, borders=False, alpha=1)

            brain.add_annotation(
                parc, borders=True, color="k", remove_existing=False, alpha=0.1
            )
            axs[i, 0].set_ylabel("w/o feedback" if i == 0 else "w/ feedback")
        elif i == 3:
            id = []
            validr = np.unique(list(validr_dir.keys()))
            data = data_all.isel(pcoder=j, fb=1) - data_all.isel(pcoder=j, fb=0)
            data = data.sel(parcel_ind=validr)
            data1 = data.mean(dim="split")  # （137,51）
            data_perm = data_all_perm.isel(pcoder=j, fb=1) - data_all_perm.isel(
                pcoder=j, fb=0
            )
            data_perm = data_perm.sel(parcel_ind=validr)

            pvalues = []
            parcel_summary = np.zeros(len(validr))
            for r in range(len(data.parcel_ind)):
                X = data.data[r, ...]
                t0, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
                    X.transpose(1, 0),  # (23,  10,137,)
                    n_permutations=5000,
                    tail=1,
                    n_jobs=1,
                    seed=42,
                    verbose=False,
                    buffer_size=None,
                )  # (events, subjects, len(labels), length)

                sig_mask = np.zeros_like(t0, dtype=bool)
                for cl, p_val in zip(clusters, pvals):

                    if p_val < 0.05:
                        ids = []
                        for c in cl[0]:
                            if (
                                1 in validr_dir[validr[r]]
                                and c in validr_dir[validr[r]][1]
                            ):  # index=1: w fb
                                ids.append(c)
                        if len(ids) > ctn_tps:  # ≥3 consecutive time points (60 ms)

                            sig_mask[ids] = True

                if sig_mask.any():

                    # average model difference (across splits) within significant times
                    parcel_summary[r] = data1.data[r, :][sig_mask].mean()
                else:
                    parcel_summary[r] = 0
                    pvals[:] = 0.06

                if pvals.size > 0:
                    p_value = np.min(pvals)
                else:
                    p_value = 0.06
                pvalues.append(p_value)

            # pvalues = multipletests(pvalues, method="fdr_bh")[1]
            colors = cmap1(norm1(parcel_summary))

            for r, color in enumerate(colors):
                if pvalues[r] < 0.05:
                    id.append(data.parcel_ind[r].data)

                    brain.add_label(
                        labels[data.parcel_ind[r].data],
                        color=color,
                        borders=False,
                        alpha=1,
                    )
                    print("pcoder", j, "index", r, "parcel", data.parcel_ind[r].data)
            axs[i, 0].set_ylabel("w/ > w/o")
        else:
            validr = np.unique(list(validr_dir.keys()))
            data = data_all.isel(pcoder=j, fb=0) - data_all.isel(pcoder=j, fb=1)
            data = data.sel(parcel_ind=validr)
            data1 = data.mean(dim="split")  # （137,51）
            data_perm = data_all_perm.isel(pcoder=j, fb=0) - data_all_perm.isel(
                pcoder=j, fb=1
            )
            data_perm = data_perm.sel(parcel_ind=validr)

            id1 = []
            pvalues1 = []
            parcel_summary = np.zeros(len(validr))
            for r in range(len(data.parcel_ind)):

                X = data.data[r, ...]
                t0, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
                    X.transpose(1, 0),  # (23,  10,137,)
                    n_permutations=5000,
                    tail=1,
                    n_jobs=1,
                    seed=42,
                    verbose=False,
                    buffer_size=None,
                )  # (events, subjects, len(labels), length)
                sig_mask = np.zeros_like(t0, dtype=bool)
                for cl, p_val in zip(clusters, pvals):
                    if p_val < 0.05:
                        if validr[r] == 24:
                            print("significant", data.parcel_ind[r].data)
                        ids = []
                        for c in cl[0]:
                            if (
                                1 in validr_dir[validr[r]]
                                and c in validr_dir[validr[r]][1]
                            ):  # index=1: w fb
                                ids.append(c)
                        if len(ids) > 3:

                            sig_mask[ids] = True
                        print(data.parcel_ind[r].data, data.time[cl].data)

                if sig_mask.any():
                    parcel_summary[r] = data1.data[r, :][sig_mask].mean()
                else:
                    parcel_summary[r] = 0
                    pvals[:] = 0.06
                if pvals.size > 0:
                    p_value = np.min(pvals)
                else:
                    p_value = 0.06
                pvalues1.append(p_value)

            # pvalues1 = multipletests(pvalues1, method="fdr_bh")[1]

            colors = cmap1(norm1(parcel_summary))
            for r, color in enumerate(colors):
                if pvalues1[r] < 0.05:
                    id1.append(data.parcel_ind[r].data)

                    brain.add_label(
                        labels[data.parcel_ind[r].data],
                        color=color,
                        borders=False,
                        alpha=1,
                    )
                    print("pcoder", j, "index", r, "parcel", data.parcel_ind[r].data)
            axs[i, 0].set_ylabel("w/o > w/")

        brain.show_view()
        screenshot = brain.screenshot()
        nonwhite_pix = (screenshot != 255).any(-1)
        nonwhite_row = nonwhite_pix.any(1)
        nonwhite_col = nonwhite_pix.any(0)
        cropped_screenshot = screenshot[nonwhite_row][:, nonwhite_col]
        axs[i, j].imshow(cropped_screenshot)
        brain.close()
        axs[0, j].set_title(f"PCoder {j+1}")

for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)

cbar1 = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm0, cmap=cmap0),
    ax=axs[0:2, :],
    orientation="vertical",
    shrink=0.8,
)
cbar1.set_label(
    "Prediction score",
    rotation=270,
    labelpad=15,
)
# Colorbar for last row (coolwarm colormap)
cbar2 = fig.colorbar(
    mpl.cm.ScalarMappable(norm=norm1, cmap=cmap1),
    ax=axs[2:, :],
    orientation="vertical",
    shrink=0.8,
)
cbar2.set_label(
    "Difference in prediction score",
    rotation=270,
    labelpad=15,
)

plt.savefig(
    fname.fig_ridge_tc(
        roi=region_name,
    ),
    dpi=300,
    bbox_inches="tight",
)
plt.show()
print("done")

# %%
