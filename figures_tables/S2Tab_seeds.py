########################################################################
# S2 Table, reliability of the accuracy change (t = 0 -> t = 25) of the
# predictive-coding model across training seeds and stimuli (paired bootstrap).
#   A  per run (original model and training seeds 5, 15, 25, 35) and condition RL1-RL3,
#      held-out model: accuracy at t = 0 and t = 25, paired change with its 95%
#      bootstrap interval over the 135 stimuli, stimuli gained / lost (descriptive)
#   B  across the 5 runs: mean change with the 95% crossed bootstrap interval
#      (runs and stimuli resampled independently), range over runs, runs with a
#      positive change, two-sided bootstrap p and BH-adjusted p (family: held-out
#      model in RL1, RL2, RL3; m = 3). Model with the median hyperparameters (Fig 3B;
#      RL2/RL3 in-sample) for comparison, descriptive.
# RW is omitted from A and B: >= 98.5% at both timesteps, at most one stimulus changed (not a test).
# Inputs (12_eval_runs_behavior.py, then 13_runs_behavior_stats.py): ../data/behavior/seeds/behavior_per_seed.csv,
#   behavior_across_seeds.csv
# Output: figures/supporting_information/S2_Tab.pdf (LaTeX source: src/S2_Tab.tex)
#
#   python figures_tables/S2Tab_seeds.py
########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
from pathlib import Path

import pandas as pd

from config import fname
from si_utils import build_table

D = Path(fname.data_dir) / "behavior" / "seeds"
C = ["RL1", "RL2", "RL3"]
SEEDS = {"orig": "Original", "seed5": "5", "seed15": "15", "seed25": "25", "seed35": "35"}

per = pd.read_csv(D / "behavior_per_seed.csv").set_index(["model", "seed", "condition"])
acr = pd.read_csv(D / "behavior_across_seeds.csv").set_index(["model", "condition"])
# RW is near ceiling in every run (>= 98.5%, at most one stimulus gained or lost)
rw = per.xs("RW", level="condition")
assert (rw[["acc_t0", "acc_t25"]] >= 98.5).all().all() and (rw.gained + rw.lost <= 1).all()


def ci(d, lo, hi, a0=None, a25=None):
    """change [CI]; * if the change differs from the difference of the rounded accuracies"""
    star = "$^{*}$" if a0 is not None and abs(round(d, 1) - (round(a25, 1) - round(a0, 1))) > 0.05 else ""
    return f"{d:+.1f}{star} [{lo:.1f}, {hi:.1f}]".replace("-", "$-$")


def p(x):
    return f"{x:.3f}" if x >= 0.001 else "$<0.001$"


rowsA = []
for s, lab in SEEDS.items():
    r = [lab]
    for c in C:
        x = per.loc[("PC held-out", s, c)]
        r += [f"{x.acc_t0:.1f}", f"{x.acc_t25:.1f}", ci(x.delta, x.ci_low, x.ci_high, x.acc_t0, x.acc_t25), f"{int(x.gained)}/{int(x.lost)}"]
    rowsA.append(r)

rowsB = []
for m, lab in [("PC held-out", "Held-out (confirmatory)"), ("PC median", "Median hyperparameters (Fig 3B)")]:
    for c in C:
        x = acr.loc[(m, c)]
        rowsB.append([lab if c == "RL1" else "", c, f"{x.acc_t0:.1f}", f"{x.acc_t25:.1f}",
                      ci(x.delta, x.ci_low, x.ci_high, x.acc_t0, x.acc_t25),
                      f"{x.delta_min:.1f}--{x.delta_max:.1f}".replace("-", "$-$", 1) if x.delta_min < 0 else f"{x.delta_min:.1f}--{x.delta_max:.1f}", f"{int(x.seeds_positive)}/{int(x.n_seeds)}",
                      *([p(x.p_boot), p(x.p_boot_fdr)] if m == "PC held-out" else ["--", "--"])])

L = [r"\documentclass[9pt]{extarticle}", r"\usepackage[margin=1.5cm,a4paper,landscape]{geometry}",
     r"\usepackage{booktabs,amsmath,array}", r"\pagestyle{empty}", r"\begin{document}",
     r"\noindent\textbf{A}\par\smallskip",
     r"\begin{tabular}{l" + " c c c c" * 3 + "}", r"\toprule",
     r" & \multicolumn{4}{c}{RL1} & \multicolumn{4}{c}{RL2} & \multicolumn{4}{c}{RL3} \\",
     r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}",
     r"Seed" + r" & $t=0$ & $t=25$ & Change [95\% CI] & Gained/lost" * 3 + r" \\", r"\midrule"]
L += [" & ".join(x) + r" \\" for x in rowsA]
L += [r"\bottomrule", r"\end{tabular}", r"\par\bigskip", r"\noindent\textbf{B}\par\smallskip",
      r"\begin{tabular}{l l c c c c c c c}", r"\toprule",
      r"Model & Condition & $t=0$ & $t=25$ & Mean change [95\% CI] & Range over seeds & Seeds $>0$ & "
      r"$p$ & $p_{\mathrm{FDR}}$ \\", r"\midrule"]
for i, x in enumerate(rowsB):
    L.append(" & ".join(x) + r" \\" + (r" \midrule" if i == 2 else ""))
L += [r"\bottomrule", r"\end{tabular}", r"\par\medskip",
      r"\noindent{\footnotesize $^{*}$Differs by 0.1 from the difference of the rounded accuracies because changes were computed from unrounded accuracies (stimuli correct/135).\par}", r"\end{document}"]
build_table("\n".join(L) + "\n", "S2_Tab")
for x in rowsA + rowsB:
    print(" | ".join(x))
