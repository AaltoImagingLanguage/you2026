"""S4 Table, effect sizes in the post hoc follow-up regions left vOT and ST, with the
mean paired difference (with minus without feedback) and its 95% t CI, and the paired
d_z with its exact CI (noncentral t).
  A: condition-wise RSA of PCoder 3 in the predefined 200-ms windows of the
     whole-cortex test (one value per participant, averaged over the window).
  B: participant-level ridge regression over the whole analyzed epoch, PCoders 1-3.
Reads ../data/rsa_tcs/rsa_tcs_parcel{40,65}_RL23.nc (18_rsa_parcels.py) and
../data/effect_sizes_rsa_ridge_RL23.csv (22_effect_sizes_clusters.py), and writes
figures/supporting_information/S4_Tab.pdf (LaTeX source: src/S4_Tab.tex).

    python figures_tables/S4Tab_effect_sizes_vOT_ST.py
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
from pathlib import Path

import pandas as pd
import xarray as xr

from config import fname, time_windows
from si_utils import build_table
from effect_sizes import dz_ci, mean_ci

PCODER = 2  # PCoder 3


def neg(x, fmt):
    return f"{x:{fmt}}".replace("-", "$-$")


rows_a = []
for name, parcel in (("vOT", 40), ("ST", 65)):
    d = xr.load_dataarray(f"{fname.data_dir}/rsa_tcs/rsa_tcs_parcel{parcel}_RL23.nc")
    for w in time_windows:
        x = d.isel(pcoder=PCODER).sel(time=slice(w[0], w[1])).mean("time")  # same slicing as the cluster test
        diff = (x.isel(feedback=1) - x.isel(feedback=0)).values
        m, lo, hi = mean_ci(diff)
        z, zlo, zhi = dz_ci(diff)
        rows_a.append(f"{name} & {int(w[0] * 1000)}--{int(w[1] * 1000)} & {neg(m, '.3f')} "
                      f"[{neg(lo, '.3f')}, {neg(hi, '.3f')}] & {neg(z, '.2f')} [{neg(zlo, '.2f')}, {neg(zhi, '.2f')}] \\\\")

d = pd.read_csv(f"{fname.data_dir}/effect_sizes_rsa_ridge_RL23.csv")
d = d[(d.analysis == "ridge") & (d.selection == "whole epoch (predefined)")]
rows_b = []
for _, r in d.iterrows():
    roi, pc = r.roi.split(" PCoder")
    rows_b.append(f"{roi} & {pc} & {neg(r.mean_diff, '.4f')} [{neg(r.ci95_low, '.4f')}, {neg(r.ci95_high, '.4f')}] & "
                  f"{neg(r.d_z, '.2f')} [{neg(r.d_z_ci95_low, '.2f')}, {neg(r.d_z_ci95_high, '.2f')}] \\\\")

tex = r"""\documentclass[9pt]{extarticle}
\usepackage[margin=2cm,a4paper]{geometry}
\usepackage{booktabs,amsmath}
\pagestyle{empty}
\begin{document}
\noindent\textbf{A}\quad RSA, PCoder 3\\[3pt]
\begin{tabular}{llll}
\toprule
Region & Window (ms) & $\Delta\rho$ [95\% CI] & $d_z$ [95\% CI] \\
\midrule
""" + "\n".join(rows_a) + r"""
\bottomrule
\end{tabular}

\vspace{12pt}
\noindent\textbf{B}\quad Ridge regression, whole epoch\\[3pt]
\begin{tabular}{llll}
\toprule
Region & PCoder & $\Delta r$ [95\% CI] & $d_z$ [95\% CI] \\
\midrule
""" + "\n".join(rows_b) + r"""
\bottomrule
\end{tabular}
\end{document}
"""
build_table(tex, "S4_Tab")
