########################################################################
# S5 Table, the correction families of the main inferences: one block per
# Benjamini-Hochberg family, stating the number and identity of the tests and the
# number of clusters entered (m), with the exact unadjusted and FDR-adjusted cluster
# p values: whole-cortex RSA in five windows, RSA and ridge regression in vOT and ST. The sizes of the families of the supplementary
# analyses (severity/lexicality partialled, controls, timesteps, cortex-mean t
# tests) are given in the Methods next to each analysis.
# Inputs (outputs of earlier scripts; nothing is recomputed):
#   figures/rsa/rsa_whole_brain_pcoderall_RL23_fdr_family.csv  (19_rsa_wholecortex.py)
#   ../data/rsa_baselines/rsa_condition_baselines_clusters_RL23.csv    (23_rsa_condition_baselines.py; same 73 clusters, with sign)
#   ../data/fdr_families_roi_RL23.csv, ../data/effect_sizes_rsa_ridge_RL23.csv  (22_effect_sizes_clusters.py)
# Output: figures/supporting_information/S5_Tab.pdf (LaTeX source: src/S5_Tab.tex)
#
#   python figures_tables/S5Tab_correction_families.py
########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
from pathlib import Path

import numpy as np
import pandas as pd

from config import fname
from si_utils import build_table

D = Path(fname.data_dir)
ALPHA = 0.05
fam = pd.read_csv(f"{fname.figures_dir}/rsa/rsa_whole_brain_pcoderall_RL23_fdr_family.csv")
base = pd.read_csv(D / "rsa_baselines/rsa_condition_baselines_clusters_RL23.csv")
raw = base[base.contrast == "w/ - w/o"].reset_index(drop=True)
for k in ["window_ms", "n_parcels_lh", "n_parcels_rh", "p_cluster"]:  # same clusters, same order
    assert (fam[k].values == raw[k].values).all(), k
fam = fam.assign(sign=raw.sign.values)
roi = pd.read_csv(D / "fdr_families_roi_RL23.csv")
roicl = pd.read_csv(D / "effect_sizes_rsa_ridge_RL23.csv")
roicl = roicl[(roicl.analysis == "RSA") & roicl.selection.str.startswith("cluster")]


def fp(p):
    """Cluster p values are multiples of 1/5000, so 4 decimals are exact."""
    return f"{p:.4f}"


def fq(q):
    return fp(q) + ("*" if q < ALPHA else "")


def win(w):
    return w.replace("-", "--")


blocks = []  # (title, rows); row = (test, cluster, p, FDR p)

# 1 whole-cortex RSA, five windows
rows = []
for w, g in fam.groupby("window_ms", sort=False):
    shown = g[g.p_cluster < ALPHA].sort_values("n_parcels_lh")
    assert (shown.sign > 0).all()
    for j, r in enumerate(shown.itertuples()):
        hemi = f"left, {r.n_parcels_lh}" if r.n_parcels_lh else f"right, {r.n_parcels_rh}"
        rows.append((f"{win(w)} ms ({len(g)} clusters)" if j == 0 else "", f"{hemi} parcels",
                     fp(r.p_cluster), fq(r.p_fdr)))
hidden = fam[fam.p_cluster >= ALPHA]
blocks.append((f"1. Whole-cortex RSA: 5 tests, one per window; $m={len(fam)}$ clusters$^a$", rows))

# 2-3 families of vOT and ST: every cluster of every test entered the correction (as in block 1); all are listed
rcl = pd.read_csv(D / "fdr_families_roi_clusters_RL23.csv")
roi = pd.read_csv(D / "fdr_families_roi_RL23.csv")


def roi_rows(fam_name, label):
    """Clusters with unadjusted p < .05 only (as in block 1); the others are counted in the note."""
    rows = []
    for t in roi[roi.family == fam_name].itertuples():
        c = rcl[(rcl.family == fam_name) & (rcl.roi == t.roi) & (rcl.pcoder == t.pcoder)].sort_values("start_ms")
        shown = c[c.p < ALPHA]
        for j, r in enumerate(shown.itertuples()):
            n = f" ({len(c)} cluster{'s' * (len(c) > 1)})"
            rows.append((label(t) + n if j == 0 else "", f"{r.start_ms:.0f}--{r.end_ms:.0f} ms", fp(r.p), fq(r.p_fdr)))
    return rows


def hidden_txt(df, p, q):
    h = df[df[p] >= ALPHA]
    return f"{len(h)} clusters with $p\\geq0.05$ (FDR $p\\geq{h[q].min():.2f}$)"


m_rsa = int(roi[roi.family == "RSA"].m.iloc[0])
blocks.append((f"2. RSA in left vOT and ST (post hoc): 2 tests; $m={m_rsa}$ clusters$^a$",
               roi_rows("RSA", lambda t: t.roi)))
m_rid = int(roi[roi.family == "ridge"].m.iloc[0])
rows = roi_rows("ridge", lambda t: f"{t.roi}, PCoder {t.pcoder}")
blocks.append((f"3. Ridge regression in vOT and ST: 6 tests, 2 regions $\\times$ PCoders 1--3; $m={m_rid}$ clusters$^a$",
               rows or [(r"\multicolumn{4}{l}{No cluster with $p<0.05$}", None, None, None)]))
NOTE_A = (f"$^a$ Not listed: block 1, {hidden_txt(fam, 'p_cluster', 'p_fdr')}; "
          f"block 2, {hidden_txt(rcl[rcl.family == 'RSA'], 'p', 'p_fdr')}; "
          f"block 3, {hidden_txt(rcl[rcl.family == 'ridge'], 'p', 'p_fdr')}.")

L = [r"\documentclass[10pt]{article}", r"\usepackage[margin=2cm,a4paper]{geometry}",
     r"\usepackage{booktabs,amsmath,array}", r"\pagestyle{empty}", r"\begin{document}",
     r"\begin{center}", r"\begin{tabular}{p{5.2cm} p{3.6cm} >{\raggedleft\arraybackslash}p{1.8cm} >{\raggedleft\arraybackslash}p{1.8cm}}", r"\toprule",
     r"Test & Cluster & $p$ & FDR $p$ \\", r"\midrule"]
for i, (title, rows) in enumerate(blocks):
    if i:
        L.append(r"\addlinespace[5pt]")
    L.append(rf"\multicolumn{{4}}{{p{{13.4cm}}}}{{\textbf{{{title}}}}} \\")
    L += [(row[0] if row[1] is None else " & ".join(row)) + r" \\" for row in rows]
L += [r"\bottomrule", r"\end{tabular}", r"\end{center}", r"\footnotesize",
      r"\noindent " + r" \\ ".join([r"$p$, unadjusted; FDR $p$, adjusted within the Benjamini--Hochberg family of each block; *, FDR $p<0.05$.", NOTE_A]), r"\end{document}"]
tex = "\n".join(L) + "\n"
build_table(tex, "S5_Tab")
print(tex)
