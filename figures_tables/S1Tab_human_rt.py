########################################################################
# S1 Table, human catch-trial behavior (reaction time).
#   A  catch trials per participant and condition: correct, incorrect and
#      missed responses, usable (correct) RTs, mean RT
#   B  the one-way repeated-measures ANOVA of the manuscript (participant means
#      of correct-trial RTs) with Mauchly's test, Greenhouse-Geisser correction
#      and effect sizes.
# Inputs (outputs of 11_human_rt_stats.py; nothing is recomputed except counts):
#   ../data/behavior/rt_trials.csv, ../data/behavior/rt_stats.csv
# Output: figures/supporting_information/S1_Tab.pdf (LaTeX source: src/S1_Tab.tex)
#
#   python figures_tables/S1Tab_human_rt.py
########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
from pathlib import Path

import numpy as np
import pandas as pd

from config import fname
from si_utils import build_table

D = Path(fname.data_dir) / "behavior"
C = ["RW", "RL1", "RL2", "RL3"]

trials = pd.read_csv(D / "rt_trials.csv")
st = pd.read_csv(D / "rt_stats.csv").set_index("analysis")
correct = trials[(trials.outcome == "hit") & trials.rt_ms.notna()]
assert ((trials.outcome == "hit") & trials.rt_ms.isna()).sum() == 0
n_sub = trials.subject.nunique()

cnt = trials.groupby(["subject", "condition"]).outcome.value_counts().unstack(fill_value=0)
for k in ["hit", "incorrect", "miss"]:
    assert k in cnt.columns
cnt["n"] = cnt.sum(axis=1)
assert (cnt["n"] == 15).all()
rt_means = correct.groupby(["subject", "condition"]).rt_ms.mean().unstack()


def mr(x):
    """mean (min--max) over participants"""
    return f"{x.mean():.1f} ({x.min():.0f}--{x.max():.0f})"


def fp(p):
    if p >= 0.001:
        return f"{p:.3f}"
    m, e = f"{p:.1e}".split("e")
    return f"{m}\\times10^{{{int(e)}}}"


rowsA = []
for c in C:
    x = cnt.xs(c, level="condition")
    rowsA.append([c, "15", mr(x["hit"]), mr(x["incorrect"]), mr(x["miss"]),
                  str(int(x["hit"].sum())), f"{rt_means[c].mean():.0f} ({rt_means[c].std(ddof=1):.0f})"])


def anova_row(name, key, note=""):
    r = st.loc[key]
    return [name + note,
            f"$F({int(r.df1)},{int(r.df2)})={r.F:.2f}$, $p={fp(r.p)}$",
            f"$W={r.mauchly_W:.2f}$, $p={fp(r.mauchly_p)}$",
            f"$\\varepsilon={r.eps_GG:.2f}$; $F({r.df1_GG:.2f},{r.df2_GG:.2f})={r.F:.2f}$, $p={fp(r.p_GG)}$",
            f"$\\eta^2_G={r.eta_G2:.2f}$"]


k_main = list(st.index)[0]   # mean RT of correct trials, as in the main text
rowsB = [anova_row("Participant mean RT of correct trials", k_main)]

L = [r"\documentclass[9pt]{extarticle}", r"\usepackage[margin=1.5cm,a4paper,landscape]{geometry}",
     r"\usepackage{booktabs,amsmath,array}", r"\pagestyle{empty}", r"\begin{document}",
     r"\noindent\textbf{A}\par\smallskip",
     r"\begin{tabular}{l c c c c c c}", r"\toprule",
     r"Condition & Catch trials & Correct & Incorrect & Missed & Usable RTs & Mean RT (ms) \\", r"\midrule"]
L += [" & ".join(x) + r" \\" for x in rowsA]
L += [r"\bottomrule", r"\end{tabular}", r"\par\bigskip",
      r"\noindent\textbf{B}\par\smallskip",
      r"\begin{tabular}{>{\raggedright}p{3.6cm} >{\raggedright}p{3.6cm} >{\raggedright}p{2.6cm} "
      r">{\raggedright}p{4.4cm} >{\raggedright\arraybackslash}p{4.8cm}}", r"\toprule",
      r"Dependent variable & Uncorrected $F$ test (sphericity assumed) & Mauchly's test of sphericity & Greenhouse--Geisser corrected $F$ test & Effect size \\", r"\midrule"]
L += [" & ".join(x) + r" \\ \addlinespace[3pt]" for x in rowsB]
L += [r"\bottomrule", r"\end{tabular}", r"\par\medskip",
      r"\noindent{\footnotesize 15 catch trials per participant and condition. Correct, incorrect and missed: mean (minimum--maximum) over participants; usable RTs: total number of correct-trial RTs; mean RT: mean (SD) over participants of the participant means of correct-trial RTs.\par}",
      r"\end{document}"]
build_table("\n".join(L) + "\n", "S1_Tab")
for x in rowsA + rowsB:
    print(" | ".join(x))
