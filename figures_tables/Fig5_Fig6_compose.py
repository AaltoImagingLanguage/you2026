########################################################################
# Main-text Figs 5 and 6:
#   Fig 5  whole cortex: A RSA maps (19_rsa_wholecortex.py, unchanged PDF;
#          family i, m = 73 clusters of the 5 PCoder 3 window tests) stacked with
#          B the RSA time course averaged over all 137 parcels (family ii, m = 2:
#          full model vs feedforward and vs every control, identical comparisons
#          counted once; cortex_mean_family.py; controls drawn in S12 Fig).
#   Fig 6  post hoc ROIs: A left vOT, B ST (family iii, m = 2; model_brain_fdr.py).
# All time courses: PCoder 3, mean +- SEM over participants, 100-ms analysis windows
# (20-ms steps), two-tailed paired temporal cluster tests (5000 permutations, seed 42);
# black bars: significant after FDR within the family, grey: p < 0.05 uncorrected only.
# Fig 5 is assembled with LaTeX (graphicx), so the map PDF is embedded unchanged;
# nothing is re-rendered.
#
# Inputs:  figures/rsa/rsa_whole_brain_pcoderall_RL23_pcoder3_main.pdf (19_rsa_wholecortex.py)
#          ../data/rsa_tcs/rsa_tcs_parcel{0-136}_RL23.nc (18_rsa_parcels.py)
#          ../data/rsa_controls/cortex_mean_family.csv (cortex_mean_family.py, cached)
#          ../data/behavior/ridge_subjinf_{vOT,ST}_RL23.pkl (read by model_brain_fdr.py)
# Outputs: figures/rsa/Figure6_wholecortex_tc.{pdf,png}, figures/rsa/Figure7_rois.{pdf,png}
#          figures/Figure5.{pdf,tif}, figures/Figure6.{pdf,tif}
#          manuscript/figures/Figure_wholecortex.pdf (Fig 5), manuscript/figures/Figure6.pdf (Fig 6)
#
#   python figures_tables/Fig5_Fig6_compose.py
########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np
import xarray as xr
from PIL import Image

from config import fb_colors, fname, metric_rsa
from cortex_mean_family import family
from model_brain_fdr import N_PERMUTATIONS, SEED, model_brain_fdr

MAPS = Path("figures/rsa/rsa_whole_brain_pcoderall_RL23_pcoder3_main.pdf")
RSA = Path(fname.figures_dir) / "rsa"
PLOS_MAX_WIDTH_PX = 2250


def load(roi):
    return xr.load_dataarray(str(fname.rsa_tc(roi=roi)).replace(".nc", "_RL23.nc")).isel(pcoder=2)


def draw(ax, d, bars, times, fs):
    m = d.mean("subject")
    se = d.std("subject", ddof=1) / np.sqrt(d.sizes["subject"])
    for n, lab in enumerate(["w/o feedback", "w/ feedback"]):
        ax.plot(times, m[:, n], color=fb_colors[n], label=lab, lw=1.5 * fs / 10)
        ax.fill_between(times, m[:, n] - se[:, n], m[:, n] + se[:, n], color=fb_colors[n], alpha=0.3, lw=0)
    for start, end, pv, sig in bars:
        if pv < 0.05:
            ax.plot([start, end], [-0.3, -0.3], color="k" if sig else "0.65", lw=0.4 * fs, solid_capstyle="butt")
            print(f"   cluster {start:.0f}-{end:.0f} ms, p = {pv:.4f} ({'FDR' if sig else 'uncorrected only'})")
    ax.axhline(0, color="k", lw=0.6 * fs / 10)
    ax.set_xlim(-10, 1010)
    ax.set_xlabel("Time (ms)")
    ax.spines[["right", "top"]].set_visible(False)


def to_tif(pdf, tif):
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["pdftoppm", "-png", "-singlefile", "-scale-to-x", str(PLOS_MAX_WIDTH_PX), "-scale-to-y", "-1",
                        str(pdf), f"{tmp}/x"], check=True)
        Image.open(f"{tmp}/x.png").convert("RGB").save(tif, compression="tiff_lzw", dpi=(300, 300))


cortex = xr.concat([load(r) for r in range(137)], "parcel").mean("parcel")
times = cortex.time.values * 1000

# ---- Fig 5B: whole-cortex average, drawn at the scale of the map PDF (15 in wide, printed at 7.5 in)
fam = family()
ff = fam[(fam.contrast == "feedforward") & fam.p_cluster.notna()]
print("Fig 5B (family ii, m = %d):" % fam.m.iloc[0])
plt.rcParams.update({"font.size": 18})
fig = plt.figure(figsize=(14.8, 4.0))
ax = fig.add_axes([0.075, 0.2, 0.62, 0.68])
draw(ax, cortex, [(r.start_ms, r.end_ms, r.p_cluster, bool(r.significant_fdr)) for r in ff.itertuples()], times, 18)
ax.set_ylabel(f"RSA score\n({metric_rsa})")
ax.set_title("Whole cortex (average of 137 parcels)", loc="left", fontsize=20)
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=18)
fig.text(0.005, 0.97, "B", fontsize=24, fontweight="bold", va="top")
fig.savefig(RSA / "Figure6_wholecortex_tc.pdf")
fig.savefig(RSA / "Figure6_wholecortex_tc.png", dpi=100)
plt.close(fig)

# ---- Fig 5: stack the unchanged map PDF (A) and the time course (B) with LaTeX
with tempfile.TemporaryDirectory() as tmp:
    shutil.copy(MAPS, f"{tmp}/maps.pdf")
    shutil.copy(RSA / "Figure6_wholecortex_tc.pdf", f"{tmp}/tc.pdf")
    Path(f"{tmp}/fig6.tex").write_text(r"""\documentclass{standalone}
\usepackage{graphicx}
\begin{document}
\begin{tabular}{@{}l@{}}
{\fontsize{24}{24}\selectfont\sffamily\bfseries A}\\[8pt]
\includegraphics[width=14.8in]{maps.pdf}\\[2pt]
\includegraphics[width=14.8in]{tc.pdf}
\end{tabular}
\end{document}
""")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "fig6.tex"], cwd=tmp, check=True,
                   stdout=subprocess.DEVNULL)
    shutil.copy(f"{tmp}/fig6.pdf", "figures/Figure5.pdf")
to_tif("figures/Figure5.pdf", "figures/Figure5.tif")
MS = Path("manuscript/figures")  # copies for the LaTeX manuscript, if present
if MS.is_dir():
    shutil.copy("figures/Figure5.pdf", MS / "Figure_wholecortex.pdf")

# ---- Fig 6: post hoc ROIs (family iii)
fdr = model_brain_fdr("RL23")
plt.rcParams.update({"font.size": 10})
fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.4), sharey=True)
for ax, (letter, roi, title) in zip(axes, [("A", 40, "vOT"), ("B", 65, "ST")]):
    d = load(roi)
    diff = (d.sel(feedback="w/") - d.sel(feedback="w/o")).data
    _, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
        diff, n_permutations=N_PERMUTATIONS, tail=0, seed=SEED, verbose=False)
    cell = fdr[("RSA", roi, 2)]
    print(f"Fig 6{letter} {title} (family iii): raw p = {cell['raw_p']:.4f}, FDR p = {cell['adjusted_p']:.4f}")
    draw(ax, d, [(times[c][0], times[c][-1], pv, pv <= cell["threshold"]) for c, pv in zip(clusters, pvals)],
         times, 10)
    ax.set_title(f"{letter}  {title}", loc="left", fontsize=10)
axes[0].set_ylabel(f"RSA score ({metric_rsa})")
axes[0].legend(loc="upper left", frameon=False, fontsize=8.5)
fig.tight_layout()
fig.savefig(RSA / "Figure7_rois.pdf", bbox_inches="tight")
fig.savefig(RSA / "Figure7_rois.png", dpi=200, bbox_inches="tight")
plt.close(fig)
shutil.copy(RSA / "Figure7_rois.pdf", "figures/Figure6.pdf")
to_tif("figures/Figure6.pdf", "figures/Figure6.tif")
if MS.is_dir():
    shutil.copy("figures/Figure6.pdf", MS / "Figure6.pdf")
print("saved -> figures/Figure5.{pdf,tif}, figures/Figure6.{pdf,tif}")
