"""Whole-cortex RSA maps and cluster tests: PCoder 3 for Fig 5A, PCoders 1-2 for S7 Fig.

Each PCoder is shown as RSA without feedback (t = 0), RSA with feedback (t = 25), and
their difference, averaged over participants in five 200-ms windows (100-300, ...,
900-1100 ms) on the 137 cortical parcels. Cluster tests (spatio-temporal cluster
1-sample test, two-tailed, 5000 permutations, parcel adjacency from the fsaverage
source space) are run only on the difference maps; each test controls the
family-wise error over parcels and samples within its window, and the cluster p-values
of all 3 PCoders x 5 windows are then corrected together with Benjamini-Hochberg FDR
(S5 Table, S5Tab_correction_families.py). Effect sizes without cluster selection are computed
in the same windows, averaged over all parcels of the cortex and of each hemisphere
(input of Tab2_effect_sizes_predefined_windows.py, Table 2).

Input: ../data/rsa_tcs/rsa_tcs_parcel{p}_RL23.nc (18_rsa_parcels.py), assembled into
../data/rsa_tcs/rsa_tcs_parcelwhole_brain_RL23.nc
Outputs (figures/rsa/rsa_whole_brain_pcoderall_RL23_*):
  _pcoder3_main.{tif,pdf}     Fig 5A (composite: Fig5_Fig6_compose.py)
  _pcoder12.{tif,pdf}         S7 Fig
  _fdr_family.csv             all clusters of the FDR family (S5 Table)
  _cluster_stats.csv          effect sizes within the significant clusters (descriptive)
  _fdr_window_family.csv      one p-value per window (smallest cluster p), BH over windows
  _predefined_windows.csv     effect sizes per window without cluster selection

    python 19_rsa_wholecortex.py --hps RL23
    python 19_rsa_wholecortex.py --hps RL23 --stats-only     # statistics, no brain maps
Rendering the
brain maps needs an off-screen OpenGL (EGL) context.
"""

# %%
import argparse
import io
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Render headless with NVIDIA EGL; without this, screenshots come back blank.
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
# The notebook backend renders off-screen here; the pyvista backend does not.
os.environ.setdefault("MNE_3D_BACKEND", "notebook")

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import xarray as xr
from mne.viz import Brain
from PIL import Image
from scipy import sparse, stats

from config import fname, metric_rsa, parc, time_windows
from utility import plot_cluster_label
from effect_sizes import d_av, dz_ci, mean_ci


N_PERMUTATIONS = 5000
SEED = 42
ALPHA = 0.05
PLOS_MAX_WIDTH_PX = 2250  # 7.5 in at 300 dpi
RSA_COMPUTE_SCRIPT = Path(__file__).with_name("18_rsa_parcels.py")  # computes all parcels
# post hoc follow-up ROIs (left vOT, left ST) and parcel 82 (pC); flags in the cluster table
FOLLOW_UP_PARCELS = {"vOT": 40, "ST": 65, "pC": 82}


parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--hps",
    choices=["RL1", "RL23"],
    default="RL23",
    help=(
        "RL1: hyperparameters optimized on RL1; RL23: hyperparameters "
        "optimized on RL2+RL3 (default, reported)."
    ),
)
parser.add_argument(
    "--compute-missing",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Run 18_rsa_parcels.py (all parcels) if per-parcel RSA files are missing.",
)
parser.add_argument(
    "--parcel-jobs",
    type=int,
    default=1,
    help="Number of independent parcels to compute concurrently.",
)
parser.add_argument(
    "--figure",
    choices=["main", "supp", "all"],
    default="all",
    help="main: PCoder 3 (Fig 5A); supp: PCoders 1-2 (S7 Fig); all: both.",
)
parser.add_argument(
    "--stats-only",
    action="store_true",
    help="Run and save the statistics without rendering the brain figure.",
)
arg = parser.parse_args()
if arg.parcel_jobs < 1:
    parser.error("--parcel-jobs must be positive")


mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
annotation = mne.read_labels_from_annot("fsaverage", parc=parc, verbose=False)
labels = [label for label in annotation if "Unknown" not in label.name]


def label_vertex_indices(label, src):
    """Map hemisphere-local FreeSurfer vertex IDs to global source positions."""
    hemi_index = 0 if label.hemi == "lh" else 1
    source_vertices = np.asarray(src[hemi_index]["vertno"])
    _, source_positions, _ = np.intersect1d(
        source_vertices,
        np.asarray(label.vertices),
        assume_unique=False,
        return_indices=True,
    )
    if hemi_index == 1:
        source_positions = source_positions + len(src[0]["vertno"])
    return np.asarray(source_positions, dtype=int)


def parcel_adjacency(labels, src):
    """Collapse vertex adjacency into hemisphere-correct parcel adjacency."""
    vertex_adjacency = mne.spatial_src_adjacency(src).tocoo()
    vertex_to_parcel = np.full(vertex_adjacency.shape[0], -1, dtype=int)
    for parcel_index, label in enumerate(labels):
        vertices = label_vertex_indices(label, src)
        if vertices.size == 0:
            raise ValueError(f"Parcel contains no source vertices: {label.name}")
        vertex_to_parcel[vertices] = parcel_index

    row_parcels = vertex_to_parcel[vertex_adjacency.row]
    column_parcels = vertex_to_parcel[vertex_adjacency.col]
    keep = (row_parcels >= 0) & (column_parcels >= 0)
    adjacency = sparse.coo_matrix(
        (
            np.ones(int(keep.sum()), dtype=np.int8),
            (row_parcels[keep], column_parcels[keep]),
        ),
        shape=(len(labels), len(labels)),
    ).tocsr()
    adjacency = adjacency.maximum(adjacency.T)
    adjacency.data[:] = 1
    adjacency.setdiag(0)
    adjacency.eliminate_zeros()
    return adjacency


src_to = mne.read_source_spaces(fname.fsaverage_src, verbose=False)
labels_adjacency_matrix = parcel_adjacency(labels, src_to)


def rsa_path(roi):
    """RSA file of one parcel; RL23 files carry the suffix _RL23."""
    path = str(fname.rsa_tc(roi=roi))
    return path.replace(".nc", "_RL23.nc") if arg.hps == "RL23" else path


def compute_parcel(roi_ind):
    """Compute one missing parcel with the project's RSA implementation."""
    output_path = rsa_path(roi_ind)
    if os.path.exists(output_path):
        return output_path
    command = [
        sys.executable,
        str(RSA_COMPUTE_SCRIPT),
        "--hps",
        arg.hps,
    ]
    print(f"computing RSA parcel {roi_ind} -> {output_path}", flush=True)
    subprocess.run(command, check=True)
    if not os.path.exists(output_path):
        raise RuntimeError(
            f"RSA computation completed without creating: {output_path}"
        )
    return output_path


def compute_missing_parcels(roi_ids):
    """Compute missing parcels and retain completed files for safe restarts."""
    print(
        f"computing {len(roi_ids)} missing {arg.hps} RSA parcels with "
        f"{arg.parcel_jobs} concurrent parcel job(s)",
        flush=True,
    )
    if arg.parcel_jobs == 1:
        for position, roi_ind in enumerate(roi_ids, start=1):
            print(f"[{position}/{len(roi_ids)}]", flush=True)
            compute_parcel(roi_ind)
        return

    with ThreadPoolExecutor(max_workers=arg.parcel_jobs) as executor:
        futures = {executor.submit(compute_parcel, roi_ind): roi_ind for roi_ind in roi_ids}
        completed = 0
        for future in as_completed(futures):
            roi_ind = futures[future]
            future.result()
            completed += 1
            print(
                f"[{completed}/{len(roi_ids)}] finished RSA parcel {roi_ind}",
                flush=True,
            )


def load_whole_brain():
    """Load or assemble the selected hyperparameter set's whole-brain RSA."""
    whole_path = rsa_path("whole_brain")
    if os.path.exists(whole_path):
        print("loaded data from:", whole_path)
        return xr.load_dataarray(whole_path)

    parcel_paths = [rsa_path(roi_ind) for roi_ind in range(len(labels))]
    missing_ids = [
        roi_ind for roi_ind, path in enumerate(parcel_paths) if not os.path.exists(path)
    ]
    if missing_ids and arg.compute_missing:
        compute_missing_parcels(missing_ids)
        missing_ids = [
            roi_ind
            for roi_ind, path in enumerate(parcel_paths)
            if not os.path.exists(path)
        ]
    if missing_ids:
        preview = ", ".join(map(str, missing_ids[:20]))
        raise FileNotFoundError(
            f"Cannot plot {arg.hps} whole-brain RSA: {len(missing_ids)} of "
            f"{len(labels)} parcel files are missing ({preview}); run "
            f"18_rsa_parcels.py --hps {arg.hps} first."
        )

    parcel_data = []
    for roi_ind, path in enumerate(parcel_paths):
        data = xr.load_dataarray(path).expand_dims(parcel_ind=[roi_ind])
        parcel_data.append(data)
    rsa_data = xr.concat(parcel_data, dim="parcel_ind")
    rsa_data.to_netcdf(whole_path)
    print("assembled and saved:", whole_path)
    return rsa_data


rsa_all = load_whole_brain()
n_pcoders = rsa_all.sizes["pcoder"]
if n_pcoders != 3:
    raise ValueError(f"Expected three PCoders, found {n_pcoders}")
if rsa_all.sizes["parcel_ind"] != len(labels):
    raise ValueError("RSA parcel count does not match the cortical annotation")

contrast = (rsa_all.isel(feedback=1) - rsa_all.isel(feedback=0)).transpose(
    "pcoder", "subject", "time", "parcel_ind"
)
# Each MNE test controls clusterwise error over space and samples within its
# panel. The resulting cluster p-values are collected across every PCoder and
# time window, then corrected together using Benjamini-Hochberg FDR.
panel_clusters = {}
panel_p_values = {}
for pcoder_index in range(n_pcoders):
    for window_index, window in enumerate(time_windows):
        panel_key = (pcoder_index, window_index)
        data = contrast.isel(pcoder=pcoder_index).sel(
            time=slice(window[0], window[1])
        ).transpose("subject", "time", "parcel_ind")
        values = np.asarray(data.values, dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(
                f"Non-finite RSA contrast in PCoder {pcoder_index + 1}, "
                f"window {window}"
            )

        if np.all(values == 0):
            panel_clusters[panel_key] = []
            panel_p_values[panel_key] = np.array([], dtype=float)
            print(
                f"PCoder {pcoder_index + 1}, {window}: exact zero contrast; "
                "no clusters"
            )
            continue

        _, clusters, raw_p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
            values,
            n_permutations=N_PERMUTATIONS,
            tail=0,
            n_jobs=1,
            seed=SEED,
            adjacency=labels_adjacency_matrix,
            out_type="indices",
            verbose=False,
            buffer_size=None,
        )
        panel_clusters[panel_key] = clusters
        panel_p_values[panel_key] = raw_p_values

all_p_values = np.concatenate(
    [p_values for p_values in panel_p_values.values() if p_values.size]
)
reject_all, fdr_p_values_all = mne.stats.fdr_correction(
    all_p_values, alpha=ALPHA, method="indep"
)

corrected_clusters = {}
cluster_rows = []
family_rows = []  # every cluster of the FDR family, significant or not
offset = 0
for pcoder_index in range(n_pcoders):
    for window_index, window in enumerate(time_windows):
        panel_key = (pcoder_index, window_index)
        raw_p_values = panel_p_values[panel_key]
        n_clusters = raw_p_values.size
        reject = reject_all[offset : offset + n_clusters]
        fdr_p_values = fdr_p_values_all[offset : offset + n_clusters]
        offset += n_clusters
        keep = np.flatnonzero(reject)
        corrected_clusters[panel_key] = [
            panel_clusters[panel_key][index] for index in keep
        ]
        if not n_clusters:
            continue
        for index in range(n_clusters):
            _, cl_parcels = panel_clusters[panel_key][index]
            cl_hemis = np.array([labels[p].hemi for p in np.unique(cl_parcels)])
            family_rows.append({
                "pcoder": pcoder_index + 1,
                "window_ms": f"{int(window[0] * 1000)}-{int(window[1] * 1000)}",
                "cluster": index,
                "n_parcels_lh": int((cl_hemis == "lh").sum()),
                "n_parcels_rh": int((cl_hemis == "rh").sum()),
                "p_cluster": raw_p_values[index],
                "p_fdr": fdr_p_values[index],
                "significant": bool(reject[index]),
            })
        print(
            f"PCoder {pcoder_index + 1}, {window}: "
            f"{len(keep)} corrected cluster(s); "
            f"minimum raw p={raw_p_values.min() if raw_p_values.size else np.nan:.6f}; "
            f"minimum FDR p={fdr_p_values.min():.6f}"
        )

        # Cluster-level effect sizes: average each participant's RSA over all
        # (time, parcel) points of the cluster, then summarize across people.
        states = rsa_all.isel(pcoder=pcoder_index).sel(
            time=slice(window[0], window[1])
        ).transpose("feedback", "subject", "time", "parcel_ind").values
        for index in keep:
            time_index, parcel_index = panel_clusters[panel_key][index]
            w_off = states[0][:, time_index, parcel_index].mean(1)
            w_on = states[1][:, time_index, parcel_index].mean(1)
            difference = w_on - w_off
            ci = stats.t.interval(
                0.95,
                len(difference) - 1,
                loc=difference.mean(),
                scale=stats.sem(difference),
            )
            d_z, dz_low, dz_high = dz_ci(difference)
            parcels = np.unique(parcel_index)
            hemis = np.array([labels[p].hemi for p in parcels])
            row = {
                "pcoder": pcoder_index + 1,
                "window_ms": f"{int(window[0] * 1000)}-{int(window[1] * 1000)}",
                "p_cluster": raw_p_values[index],
                "p_fdr": fdr_p_values[index],
                "n_parcels_lh": int((hemis == "lh").sum()),
                "n_parcels_rh": int((hemis == "rh").sum()),
                "mean_delta_rho": difference.mean(),
                "ci95_low": ci[0],
                "ci95_high": ci[1],
                # paired d_z (SD of the differences) with its exact CI;
                # d_av (pooled SD of the two states) for reference
                "d_z": d_z,
                "d_z_ci95_low": dz_low,
                "d_z_ci95_high": dz_high,
                "d_av": d_av(w_off, w_on),
                # these values are conditional on the cluster selected from the
                # same participants: descriptive only
                "selection": "cluster (descriptive)",
            }
            for name, parcel in FOLLOW_UP_PARCELS.items():
                row[f"{name}_in_cluster"] = bool(np.isin(parcel, parcels))
            cluster_rows.append(row)
            print(
                f"    cluster: {row['n_parcels_lh']} lh + "
                f"{row['n_parcels_rh']} rh parcels, p={row['p_cluster']:.4f}, "
                f"FDR p={row['p_fdr']:.4f}, mean delta rho="
                f"{row['mean_delta_rho']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}], "
                f"d_z={row['d_z']:.2f} [{row['d_z_ci95_low']:.2f}, {row['d_z_ci95_high']:.2f}], "
                + ", ".join(
                    f"{name}={'in' if row[f'{name}_in_cluster'] else 'out'}"
                    for name in FOLLOW_UP_PARCELS
                )
            )

figure_stem = str(fname.fig_rsa_whole(p="all"))
if arg.hps == "RL23":
    figure_stem += "_RL23"
stats_path = f"{figure_stem}_cluster_stats.csv"
pd.DataFrame(cluster_rows).to_csv(stats_path, index=False)
print("saved ->", stats_path)

# the complete FDR family (all clusters of all tested PCoders and windows)
family = pd.DataFrame(family_rows)
family_path = f"{figure_stem}_fdr_family.csv"
family.to_csv(family_path, index=False)
print(f"FDR family: {len(family)} clusters from "
      f"{family.groupby(['pcoder', 'window_ms']).ngroups} tests -> {family_path}")

# alternative family: one hypothesis per tested window (its smallest cluster p, the
# family-wise-corrected test of 'no effect anywhere in this window'), BH over the
# tested windows
window_family = family.groupby(["pcoder", "window_ms"], sort=False).p_cluster.min().reset_index()
window_family["p_fdr_window_family"] = mne.stats.fdr_correction(
    window_family.p_cluster, alpha=ALPHA)[1]
window_family.to_csv(f"{figure_stem}_fdr_window_family.csv", index=False)
print(window_family.to_string(index=False))

# effect sizes in the predefined 200-ms windows, averaged over ALL parcels
# of each hemisphere (and of the cortex) without any cluster selection
hemi_of = np.array([label.hemi for label in labels])
predefined_rows = []
for pcoder_index in range(n_pcoders):
    for window_index, window in enumerate(time_windows):
        states = rsa_all.isel(pcoder=pcoder_index).sel(
            time=slice(window[0], window[1])
        ).transpose("feedback", "subject", "time", "parcel_ind").values
        for region, mask in (("cortex", np.ones(hemi_of.size, bool)),
                             ("left", hemi_of == "lh"), ("right", hemi_of == "rh")):
            w_off = states[0][:, :, mask].mean((1, 2))
            w_on = states[1][:, :, mask].mean((1, 2))
            difference = w_on - w_off
            mean, low, high = mean_ci(difference)
            if np.all(difference == 0):
                d_z = dz_low = dz_high = np.nan
            else:
                d_z, dz_low, dz_high = dz_ci(difference)
            predefined_rows.append({
                "pcoder": pcoder_index + 1, "region": region,
                "window_ms": f"{int(window[0] * 1000)}-{int(window[1] * 1000)}",
                "rho_without": w_off.mean(), "rho_with": w_on.mean(),
                "mean_delta_rho": mean, "ci95_low": low, "ci95_high": high,
                "d_z": d_z, "d_z_ci95_low": dz_low, "d_z_ci95_high": dz_high,
                "selection": "predefined window, all parcels",
            })
predefined = pd.DataFrame(predefined_rows)
predefined.to_csv(f"{figure_stem}_predefined_windows.csv", index=False)
print(predefined[predefined.pcoder == 3].round(3).to_string(index=False))
if arg.stats_only:
    sys.exit()


norm_bupu = plt.Normalize(vmin=0, vmax=0.8)
cmap_bupu = plt.get_cmap("BuPu")
norm_coolwarm = plt.Normalize(vmin=-0.4, vmax=0.4)
cmap_coolwarm = plt.get_cmap("coolwarm")
row_names = ["w/o feedback", "w/ feedback", "w/ \u2212 w/o"]


def render_brain(pcoder_index, window_index, state_index):
    """Screenshot one brain: RSA w/o or w/ feedback, or their difference."""
    window = time_windows[window_index]
    brain = Brain(
        subject="fsaverage",
        surf="inflated",
        hemi="split",
        views=["lateral", "ventral"],
        view_layout="vertical",
        cortex="0.8",
        background="white",
        show=False,
    )
    if state_index < 2:
        data = rsa_all.isel(pcoder=pcoder_index, feedback=state_index).sel(
            time=slice(window[0], window[1])
        )
        data_avg = data.mean(dim="subject").mean(dim="time")
        colors = cmap_bupu(norm_bupu(data_avg))
        for parcel_index, color in enumerate(colors):
            brain.add_label(labels[parcel_index], color=color, borders=False, alpha=1)
        brain.add_annotation(
            parc, borders=True, color="k", remove_existing=False, alpha=0.1
        )
    else:
        data = contrast.isel(pcoder=pcoder_index).sel(
            time=slice(window[0], window[1])
        )
        data_avg = data.mean(dim="subject").mean(dim="time")
        colors = cmap_coolwarm(norm_coolwarm(data_avg))
        for parcel_index, color in enumerate(colors):
            brain.add_label(labels[parcel_index], color=color, borders=False, alpha=1)
        for cluster in corrected_clusters[pcoder_index, window_index]:
            plot_cluster_label(
                cluster, labels, brain, color="black", width=2, alpha=0.6
            )

    brain.show_view()
    screenshot = brain.screenshot()
    brain.close()
    nonwhite = (screenshot != 255).any(-1)
    return screenshot[nonwhite.any(1)][:, nonwhite.any(0)]


def plot_figure(pcoder_indices, figure_path):
    """One panel per PCoder, each with three rows (w/o, w/, difference);
    the colorbars sit in their own column to the right of the panels. With
    several PCoders, each panel is framed and labelled "PCoder X"."""
    n_panels = len(pcoder_indices)
    label_panels = n_panels > 1
    # The figure is drawn 15 in wide and printed at 7.5 in, so font sizes
    # here print at half size.
    fig = plt.figure(figsize=(15, 6.2 * n_panels))
    panels_fig, cbar_fig = fig.subfigures(
        1, 2, width_ratios=[0.91, 0.09], wspace=0
    )
    pcoder_figs = np.atleast_1d(
        panels_fig.subfigures(
            n_panels, 1, hspace=0.03, height_ratios=[1.07] + [1] * (n_panels - 1)
        )
    )
    for panel_index, (pcoder_index, pcoder_fig) in enumerate(
        zip(pcoder_indices, pcoder_figs)
    ):
        if label_panels:
            pcoder_fig.add_artist(
                mpl.patches.FancyBboxPatch(
                    (0.006, 0.012),
                    0.988,
                    0.976,
                    boxstyle="round,pad=0,rounding_size=0.015",
                    transform=pcoder_fig.transSubfigure,
                    facecolor="none",
                    edgecolor="0.6",
                    linewidth=1.2,
                )
            )
        axs = pcoder_fig.subplots(3, len(time_windows))
        pcoder_fig.subplots_adjust(
            left=0.1 if label_panels else 0.05,
            right=0.985,
            top=0.89 if panel_index == 0 else 0.96,
            bottom=0.03,
            wspace=0.05,
            hspace=0.25,
        )
        if label_panels:
            pcoder_fig.supylabel(
                f"PCoder {pcoder_index + 1}", fontsize=20, fontweight="bold", x=0.025
            )
        for state_index, row_name in enumerate(row_names):
            axs[state_index, 0].set_ylabel(row_name, fontsize=18)
            for window_index in range(len(time_windows)):
                axs[state_index, window_index].imshow(
                    render_brain(pcoder_index, window_index, state_index)
                )
        if panel_index == 0:
            for window_index, window in enumerate(time_windows):
                axs[0, window_index].set_title(
                    f"{int(window[0] * 1000)}-{int(window[1] * 1000)} ms",
                    fontsize=20,
                )
        for axis in axs.flat:
            axis.set_xticks([])
            axis.set_yticks([])
            axis.set_frame_on(False)

    cbar_raw_axis = cbar_fig.add_axes([0.12, 0.55, 0.1, 0.33])
    cbar_difference_axis = cbar_fig.add_axes([0.12, 0.06, 0.1, 0.33])
    for cax, norm, cmap, label in [
        (cbar_raw_axis, norm_bupu, cmap_bupu, "RSA score"),
        (cbar_difference_axis, norm_coolwarm, cmap_coolwarm, "RSA difference"),
    ]:
        cbar = fig.colorbar(
            mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
            cax=cax,
            orientation="vertical",
        )
        cbar.set_ticks(np.linspace(norm.vmin, norm.vmax, 3))
        cbar.ax.tick_params(labelsize=14)
        cbar.set_label(
            f"{label}\n({metric_rsa})", rotation=270, labelpad=40, fontsize=16
        )

    # Render at 300 dpi, then downsample to the PLOS maximum width (7.5 in =
    # 2250 px at 300 dpi) and save as an LZW-compressed RGB TIFF.
    pdf_path = figure_path.replace(".tif", ".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    print("saved ->", pdf_path)
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    buffer.seek(0)
    image = Image.open(buffer).convert("RGB")
    height = round(image.height * PLOS_MAX_WIDTH_PX / image.width)
    image = image.resize((PLOS_MAX_WIDTH_PX, height), Image.LANCZOS)
    image.save(figure_path, compression="tiff_lzw", dpi=(300, 300))
    print("saved ->", figure_path)


# Fig 5A: PCoder 3, the only layer whose RSA changes with feedback.
# S7 Fig: PCoders 1 and 2, whose difference maps are zero.
if arg.figure in ("main", "all"):
    plot_figure([2], f"{figure_stem}_pcoder3_main.tif")
if arg.figure in ("supp", "all"):
    plot_figure([0, 1], f"{figure_stem}_pcoder12.tif")
print("done")
