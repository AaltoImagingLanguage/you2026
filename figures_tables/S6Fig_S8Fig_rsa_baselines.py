########################################################################
# S6 and S8 Figs:
#   S6 Fig  condition-label baselines: condition RDMs of PCoder 3 without
#            and with feedback, of the misspelling-severity and of the lexicality
#            baseline, and the Spearman correlations between them.
#   S8 Fig  noise ceiling and partialling out the ordinal severity structure:
#            A  noise ceiling of the MEG RDMs, averaged over the whole cortex (lower-upper bound);
#            B  RSA averaged over the whole cortex at every 100-ms analysis window
#               (20-ms steps, no window averaging), without and with feedback, before
#               (solid) and after (dashed) partialling out severity; noise ceiling of the
#               raw RSA in grey. Descriptive: inference rests on the cluster test in C;
#            C  whole-cortex maps of the severity-partialled RSA (rendered on a GPU node
#               by S8FigC_severity_partial_maps.py;
#               embedded unchanged) with the outlines of the significant clusters.
# Inputs: ../data/rsa_baselines/table_condition_rdms_RL23.csv, model_vs_label_rdms_RL23.csv (25_rsa_baseline_tables.py),
#         rsa_condition_baselines_RL23.nc (23_rsa_condition_baselines.py),
#         figures/rsa_baselines/rsa_severity_partial_maps_RL23.pdf (S8FigC_severity_partial_maps.py)
# Outputs: figures/rsa_baselines/S6_condition_baselines.{pdf,png}, S8_severity_partial.{pdf,png}
#          figures/supporting_information/S6_Fig.tif, S8_Fig.tif (PDFs in src/)
#
#   python figures_tables/S6Fig_S8Fig_rsa_baselines.py
########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
import shutil
import subprocess
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

from config import fb_colors, fname
from si_utils import pdf_to_tif

D = Path(fname.data_dir) / "rsa_baselines"
FIG = Path("figures/rsa_baselines")
MAPS = FIG / "rsa_severity_partial_maps_RL23.pdf"
COND = ["RW", "RL1", "RL2", "RL3"]



# ------------------------------------------------------------------ S6 Fig
rdms = pd.read_csv(D / "table_condition_rdms_RL23.csv", index_col=0)
corr = pd.read_csv(D / "model_vs_label_rdms_RL23.csv", index_col=0)
names = ["PCoder 3 w/o", "PCoder 3 w/", "severity", "lexicality"]
titles = ["PCoder 3\nw/o feedback", "PCoder 3\nw/ feedback", "Misspelling\nseverity", "Lexicality"]
fig, axes = plt.subplots(1, 5, figsize=(15, 3.6), width_ratios=[1, 1, 1, 1, 1.5], layout="constrained")
for ax, name, title in zip(axes, names, titles):
    square = np.zeros((4, 4))
    square[np.triu_indices(4, 1)] = rdms.loc[name].values
    square = square + square.T
    ax.imshow(square / square.max(), cmap="viridis", vmin=0, vmax=1)
    ax.set_title(title, fontsize=11)
    ax.set_xticks(range(4), COND, fontsize=9, rotation=90)
    ax.set_yticks(range(4), COND, fontsize=9)
    ax.grid(False)
short = ["PC3 w/o", "PC3 w/", "severity", "lexicality"]
axes[-1].imshow(corr.values, cmap="RdBu_r", vmin=-1, vmax=1)
axes[-1].set_xticks(range(4), short, rotation=90, fontsize=9)
axes[-1].set_yticks(range(4), short, fontsize=9)
for (i, j), v in np.ndenumerate(corr.values):
    axes[-1].text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11, fontweight="bold",
                  color="white" if abs(v) > 0.6 else "black")
axes[-1].grid(False)
axes[-1].set_title("RDM correlation (Spearman)", fontsize=11)
fig.savefig(FIG / "S6_condition_baselines.pdf")
fig.savefig(FIG / "S6_condition_baselines.png", dpi=200)
plt.close(fig)
pdf_to_tif(FIG / "S6_condition_baselines.pdf", "S6_Fig")

# ------------------------------------------------------------------ S8 Fig A, B
ds = xr.load_dataset(D / "rsa_condition_baselines_RL23.nc")
cortex = ds["rsa"].mean("parcel")                       # (subject, time, measure)
nc = ds["noise_ceiling"].mean("parcel")
t = ds.time.values * 1000                               # start of each 100-ms analysis window, as in Fig 5B
plt.rcParams.update({"font.size": 18})
fig = plt.figure(figsize=(14.8, 4.2))
axA = fig.add_axes([0.07, 0.19, 0.29, 0.68])
axB = fig.add_axes([0.455, 0.19, 0.29, 0.68], sharey=axA)
axA.fill_between(t, nc.sel(bound="lower"), nc.sel(bound="upper"), color="0.7", lw=0)
axA.set_ylabel("Spearman ρ")
axA.set_title("Noise ceiling of the MEG RDMs", fontsize=17)
axB.fill_between(t, nc.sel(bound="lower"), nc.sel(bound="upper"), color="0.88", lw=0, label="noise ceiling")
for k, (state, lab) in enumerate([("w/o", "w/o feedback"), ("w/", "w/ feedback")]):
    for suffix, ls, extra in [("", "-", ""), (" | severity", "--", ", severity\npartialled out")]:
        d = cortex.sel(measure=f"pcoder3 {state}{suffix}")
        m, se = d.mean("subject"), d.std("subject", ddof=1) / np.sqrt(d.sizes["subject"])
        axB.plot(t, m, color=fb_colors[k], ls=ls, lw=2.7, label=lab + extra)
        axB.fill_between(t, m - se, m + se, color=fb_colors[k], alpha=0.25, lw=0)
axB.set_ylabel("RSA score (Spearman)")
axB.set_title("RSA of PCoder 3", fontsize=17)
axB.legend(loc="upper left", bbox_to_anchor=(1.02, 1.05), frameon=False, fontsize=14)
for ax, letter in [(axA, "A"), (axB, "B")]:
    ax.axhline(0, color="k", lw=1.08)
    ax.set_xlim(-10, 1010)
    ax.set_xlabel("Time (ms)")
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(-0.2, 1.02, letter, transform=ax.transAxes,
            fontsize=24, fontweight="bold", va="bottom")
fig.savefig(FIG / "S8_timecourse.pdf")
plt.close(fig)

# ------------------------------------------------------------------ S8 Fig = time courses (A, B) + maps (C)
with tempfile.TemporaryDirectory() as tmp:
    shutil.copy(MAPS, f"{tmp}/maps.pdf")
    shutil.copy(FIG / "S8_timecourse.pdf", f"{tmp}/tc.pdf")
    Path(f"{tmp}/s8.tex").write_text(r"""\documentclass{standalone}
\usepackage{graphicx}
\begin{document}
\begin{tabular}{@{}l@{}}
\includegraphics[width=14.8in]{tc.pdf}\\[6pt]
{\fontsize{24}{24}\selectfont\sffamily\bfseries C}\\[8pt]
\includegraphics[width=14.8in]{maps.pdf}
\end{tabular}
\end{document}
""")
    subprocess.run(["pdflatex", "-interaction=nonstopmode", "s8.tex"], cwd=tmp, check=True, stdout=subprocess.DEVNULL)
    shutil.copy(f"{tmp}/s8.pdf", FIG / "S8_severity_partial.pdf")
subprocess.run(["pdftoppm", "-png", "-singlefile", "-r", "60", str(FIG / "S8_severity_partial.pdf"),
                str(FIG / "S8_severity_partial")], check=True)
pdf_to_tif(FIG / "S8_severity_partial.pdf", "S8_Fig")
