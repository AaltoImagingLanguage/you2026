########################################################################
# Whole-cortex maps of the PCoder 3 feedback effect after partialling out
# misspelling severity (S8 Fig C).
#
# Rows: severity-partialled RSA without and with feedback, and their difference.
# Cluster tests are rerun exactly as in 23_rsa_condition_baselines.py (per
# window, all parcels, two-tailed, 5000 permutations, seed 42, FDR across the
# five windows within each contrast) and checked against its cluster CSV.
# Significant clusters are outlined in black.
#
# Inputs:  ../data/rsa_baselines/rsa_condition_baselines_RL23.nc and
#          ../data/rsa_baselines/rsa_condition_baselines_clusters_RL23.csv
#          (23_rsa_condition_baselines.py)
# Outputs: figures/rsa_baselines/rsa_severity_partial_maps_RL23.{pdf,png}
#          (assembled into S8 Fig by S6Fig_S8Fig_rsa_baselines.py)
#
#   python figures_tables/S8FigC_severity_partial_maps.py
########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
import argparse
import os
from pathlib import Path

# Render headless with EGL (as in 19_rsa_wholecortex.py).
os.environ.setdefault("VTK_DEFAULT_OPENGL_WINDOW", "vtkEGLRenderWindow")
os.environ.setdefault("PYVISTA_OFF_SCREEN", "true")
os.environ.setdefault("MNE_3D_BACKEND", "notebook")

import matplotlib as mpl
import matplotlib.pyplot as plt
import mne
import numpy as np
import pandas as pd
import xarray as xr
from mne.viz import Brain

from config import fname, time_windows
from rsa_utils import cortical_labels, parcel_adjacency, window_labels, window_mask

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hps", default="RL23", choices=["RL1", "RL23"])
parser.add_argument("--n-perm", type=int, default=5000)
args = parser.parse_args()

OUT_DIR = Path(f"{fname.data_dir}/rsa_baselines")
FIG_DIR = Path(fname.figures_dir) / "rsa_baselines"
SEED = 42
ALPHA = 0.05

mne.set_config("SUBJECTS_DIR", fname.mri_subjects_dir)
labels = cortical_labels()
hemis = np.array([label.hemi for label in labels])
rsa = xr.load_dataset(OUT_DIR / f"rsa_condition_baselines_{args.hps}.nc")["rsa"]
times = rsa.time.values

CONTRASTS = {
    "w/ - w/o": ("pcoder3 w/", "pcoder3 w/o"),
    "w/ - w/o | severity": ("pcoder3 w/ | severity", "pcoder3 w/o | severity"),
}

# ---------------------------------------------------------------- cluster tests
adjacency = parcel_adjacency(labels)
sig_clusters = {}
rows = []
for name, (m_a, m_b) in CONTRASTS.items():
    contrast = (rsa.sel(measure=m_a) - rsa.sel(measure=m_b)).transpose("subject", "time", "parcel")
    family = []
    for w_index, (w, wl) in enumerate(zip(time_windows, window_labels())):
        values = contrast.isel(time=window_mask(times, w)).values
        _, clusters, p_values, _ = mne.stats.spatio_temporal_cluster_1samp_test(
            values, n_permutations=args.n_perm, tail=0, seed=SEED, adjacency=adjacency,
            out_type="indices", verbose=False, buffer_size=None)
        family += [(w_index, wl, c, p) for c, p in zip(clusters, p_values)]
    fdr = mne.stats.fdr_correction([f[3] for f in family])[1]
    for (w_index, wl, c, p), q in zip(family, fdr):
        rows.append({"contrast": name, "window_ms": wl, "p_cluster": p, "p_fdr": q,
                     "n_parcels_lh": int((hemis[np.unique(c[1])] == "lh").sum()),
                     "n_parcels_rh": int((hemis[np.unique(c[1])] == "rh").sum())})
        if q < ALPHA:
            sig_clusters.setdefault((name, w_index), []).append(c)

# the clusters must be the ones reported in rsa_condition_baselines_clusters_*.csv
reported = pd.read_csv(OUT_DIR / f"rsa_condition_baselines_clusters_{args.hps}.csv")
reported = reported[reported.contrast.isin(CONTRASTS)]   # the CSV also holds "| lexicality"
cols = ["contrast", "window_ms", "n_parcels_lh", "n_parcels_rh"]
ours = pd.DataFrame(rows)
for df in (ours, reported):
    df.sort_values(cols + ["p_cluster"], inplace=True, ignore_index=True)
assert ours[cols].equals(reported[cols]), "clusters differ from the reported CSV"
assert np.allclose(ours[["p_cluster", "p_fdr"]], reported[["p_cluster", "p_fdr"]])
print(ours[ours.p_fdr < ALPHA].round(4).to_string(index=False))

# ---------------------------------------------------------------- figure
ROWS = [  # (row label, measures to subtract or show, contrast for clusters)
    ("w/o\n| severity", ("pcoder3 w/o | severity", None), None),
    ("w/\n| severity", ("pcoder3 w/ | severity", None), None),
    ("w/ − w/o\n| severity",
     ("pcoder3 w/ | severity", "pcoder3 w/o | severity"), "w/ - w/o | severity"),
]  # "| severity": severity RDM partialled out
# Partial correlations can be negative, so the score maps use a diverging scale.
norm_score = plt.Normalize(vmin=-0.6, vmax=0.6)
cmap_score = plt.get_cmap("PuOr_r")
norm_diff = plt.Normalize(vmin=-0.6, vmax=0.6)
cmap_diff = plt.get_cmap("coolwarm")


def window_mean(measures, w):
    m_a, m_b = measures
    data = rsa.sel(measure=m_a) - (rsa.sel(measure=m_b) if m_b else 0)
    return data.isel(time=window_mask(times, w)).mean(["subject", "time"]).values


def plot_cluster_label(cluster, brain):
    """Outline a cluster's parcels at its most populated time point (as
    utility.plot_cluster_label, which cannot be imported without torch)."""
    t_index, p_index = cluster
    steps, counts = np.unique(t_index, return_counts=True)
    for p in np.asarray(p_index)[t_index == steps[np.argmax(counts)]]:
        brain.add_label(labels[p], borders=2, color="black", alpha=0.6)


def render_brain(values, cmap, norm, clusters):
    brain = Brain(subject="fsaverage", surf="inflated", hemi="split",
                  views=["lateral", "ventral"], view_layout="vertical", cortex="0.8",
                  background="white", show=False)
    for label, color in zip(labels, cmap(norm(values))):
        brain.add_label(label, color=color, borders=False, alpha=1)
    for cluster in clusters:
        plot_cluster_label(cluster, brain)
    brain.show_view()
    screenshot = brain.screenshot()
    brain.close()
    nonwhite = (screenshot != 255).any(-1)
    return screenshot[nonwhite.any(1)][:, nonwhite.any(0)]


fig = plt.figure(figsize=(15, 6.4))
panels_fig, cbar_fig = fig.subfigures(1, 2, width_ratios=[0.91, 0.09], wspace=0)
axs = panels_fig.subplots(len(ROWS), len(time_windows))
panels_fig.subplots_adjust(left=0.1, right=0.985, top=0.93, bottom=0.02,
                           wspace=0.05, hspace=0.25)
for r, (row_name, measures, contrast_name) in enumerate(ROWS):
    is_diff = measures[1] is not None
    axs[r, 0].set_ylabel(row_name, fontsize=15)
    for w_index, w in enumerate(time_windows):
        axs[r, w_index].imshow(render_brain(
            window_mean(measures, w),
            cmap_diff if is_diff else cmap_score, norm_diff if is_diff else norm_score,
            sig_clusters.get((contrast_name, w_index), [])))
for w_index, wl in enumerate(window_labels()):
    axs[0, w_index].set_title(f"{wl} ms", fontsize=18)
for ax in axs.flat:
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
for cax_rect, norm, cmap, label in [
    ([0.12, 0.42, 0.1, 0.45], norm_score, cmap_score, "partial RSA score"),   # rows 1-2
    ([0.12, 0.04, 0.1, 0.24], norm_diff, cmap_diff, "RSA difference"),       # row 3
]:
    cbar = fig.colorbar(mpl.cm.ScalarMappable(norm=norm, cmap=cmap),
                        cax=cbar_fig.add_axes(cax_rect), orientation="vertical")
    cbar.set_ticks(np.linspace(norm.vmin, norm.vmax, 3))
    cbar.ax.tick_params(labelsize=13)
    cbar.set_label(f"{label}\n(Spearman)", rotation=270, labelpad=38, fontsize=14)

for ext in ("pdf", "png"):
    path = FIG_DIR / f"rsa_severity_partial_maps_{args.hps}.{ext}"
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print("saved ->", path)
