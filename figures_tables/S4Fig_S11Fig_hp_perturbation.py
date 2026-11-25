########################################################################
# S4 Fig (behaviour) and S11 Fig (RSA) of the hyperparameter perturbation
# (31_hp_perturbation.py; analysed by 32_hp_perturbation_rsa.py). Every optimized hyperparameter is
# multiplied by a factor drawn uniformly from [1 - m, 1 + m], m = 0.2, 0.4, 0.6
# (100 sets each; mixing weights renormalized), readout timestep t = 25.
#
#   S4 Fig  A-C accuracy over t = 0..28 per magnitude (thin: perturbed sets, thick:
#           optimized model); D change in accuracy t = 0 -> 25 per set, RL1-RL3
#   S11 Fig A-C cortex-average RSA of PCoder 3 at every 100-ms MEG analysis window
#           (w/o feedback = t = 0, shared by all sets; w/ feedback at t = 25);
#           D time-averaged RSA per set (mean over all analysis windows)
#   SEM across participants for w/o feedback and the optimized model.
#
# Outputs: figures/dynamics/{S3_scale_behavior,S5_scale_rsa}.{pdf,png},
#          rendered to figures/supporting_information/S4_Fig.tif, S11_Fig.tif (PDFs in src/)
#
#   python figures_tables/S4Fig_S11Fig_hp_perturbation.py
########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from config import category_colors, fb_colors, fb_timestep, fname
from si_utils import pdf_to_tif

D = Path(fname.data_dir) / "dynamics"
FIG = Path(fname.figures_dir) / "dynamics"
CONDS = ["RW", "RL1", "RL2", "RL3"]
LEVELS = [("scale20", "±20%"), ("scale40", "±40%"), ("scale60", "±60%")]
T_MAX = 29
plt.rcParams.update({"font.size": 10, "axes.spines.right": False, "axes.spines.top": False})

p = pd.read_csv(D / "hp_perturbation_per_set.csv")
z = np.load(D / "hp_perturbation_curves.npz", allow_pickle=True)
assert list(z["name"]) == list(p.name)
assert not p.diverged.any()
acc = 100 * z["acc"]                                                # (set, T, cond)
i_opt = list(p.name).index("median")
steps = np.arange(acc.shape[1])[:T_MAX]
rng = np.random.default_rng(0)                                      # jitter of the dots only


def save(fig, name, si_name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}", bbox_inches="tight", dpi=300)
    pdf_to_tif(FIG / f"{name}.pdf", si_name)


def sel(level):
    return (p.level == level).values


# ---- S4 Fig: behaviour ---------------------------------------------------------------
fig = plt.figure(figsize=(11, 7.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.45, wspace=0.12)
for gi, (lev, lab) in enumerate(LEVELS):
    ax = fig.add_subplot(gs[0, gi])
    for k in range(4):
        for a in acc[sel(lev), :T_MAX, k]:
            ax.plot(steps, a, color=category_colors[k], alpha=0.12, lw=0.6)
    for k in range(4):
        ax.plot(steps, acc[i_opt, :T_MAX, k], color=category_colors[k], lw=2.2)
    ax.axvline(fb_timestep, color="grey", ls="--", lw=0.8)
    ax.set(ylim=(0, 101), yticks=range(0, 101, 20), xticks=[0, 5, 10, 15, 20, 25], xlim=(0, 28),
           xlabel="Timestep")
    if gi:
        ax.set_yticklabels([])
    else:
        ax.set_ylabel("Accuracy (%)")
    ax.set_title(f"{'ABC'[gi]}  {lab} ({int(sel(lev).sum())} sets)", loc="left", fontsize=10)
handles = [Line2D([], [], color=category_colors[k], lw=2.2) for k in range(4)]
handles += [Line2D([], [], color="0.3", lw=2.2), Line2D([], [], color="0.3", lw=0.7, alpha=0.4)]
fig.legend(handles, CONDS + ["optimized model", "perturbed sets (condition colour)"],
           loc="upper center", ncol=6, frameon=False, bbox_to_anchor=(0.5, 0.515), fontsize=9)

ax = fig.add_subplot(gs[1, :])
width = 0.22
GAP = 4                                     # per condition: the three magnitudes, then a gap
for k in (1, 2, 3):
    x0 = (k - 1) * GAP
    for gi, (lev, lab) in enumerate(LEVELS):
        g = acc[sel(lev), fb_timestep, k] - acc[sel(lev), 0, k]
        ax.scatter(x0 + gi + rng.uniform(-width, width, len(g)), g, s=8, color=category_colors[k],
                   alpha=0.6, lw=0, zorder=2)
    opt = acc[i_opt, fb_timestep, k] - acc[i_opt, 0, k]
    ax.hlines(opt, x0 - 0.45, x0 + 2.45, color=category_colors[k], lw=1.2, ls=(0, (4, 3)), zorder=4)
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks([(k - 1) * GAP + gi for k in (1, 2, 3) for gi in range(3)],
              [lab for _ in (1, 2, 3) for _, lab in LEVELS], fontsize=9)
for k in (1, 2, 3):
    ax.text((k - 1) * GAP + 1, -0.17, CONDS[k], transform=ax.get_xaxis_transform(), ha="center",
            color=category_colors[k], fontweight="bold")
ax.set_ylabel(f"Change in accuracy, t = 0 to {fb_timestep}\n(percentage points)")
ax.set_title("D  Change in accuracy per perturbed set", loc="left", fontsize=10)
ax.legend([Line2D([], [], marker="o", ls="", ms=4, color="0.5", alpha=0.6),
           Line2D([], [], color="0.5", ls=(0, (4, 3)), lw=1.2)],
          ["perturbed sets (condition colour)", "optimized model (condition colour)"], frameon=False,
          loc="upper right", fontsize=9)
save(fig, "S3_scale_behavior", "S4_Fig")

# ---- S11 Fig: RSA (raw scores) -----------------------------------------------------------
mt = z["meg_times"] * 1000
k25 = list(z["rsa_steps"]).index(fb_timestep)
R, RS = z["rsa_tc"], z["rsa_tc_sem"]                               # (set, step, MEG time)
fig = plt.figure(figsize=(11, 7.4))
gs = fig.add_gridspec(2, 3, height_ratios=[1, 1.05], hspace=0.45, wspace=0.12)
top = []
for gi, (lev, lab) in enumerate(LEVELS):
    ax = fig.add_subplot(gs[0, gi], sharey=top[0] if top else None)
    top.append(ax)
    for a in R[sel(lev), k25]:
        ax.plot(mt, a, color=fb_colors[1], alpha=0.12, lw=0.6)
    for k, col, lw in ((0, fb_colors[0], 1.8), (k25, fb_colors[1], 2.2)):
        ax.plot(mt, R[i_opt, k], color=col, lw=lw)
        ax.fill_between(mt, R[i_opt, k] - RS[i_opt, k], R[i_opt, k] + RS[i_opt, k], color=col, alpha=0.25, lw=0)
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlim=(0, 1000), xticks=range(0, 1001, 250), xlabel="MEG time (ms)")
    if gi:
        plt.setp(ax.get_yticklabels(), visible=False)
    else:
        ax.set_ylabel("Cortex-average RSA (Spearman)")
    ax.set_title(f"{'ABC'[gi]}  {lab} ({int(sel(lev).sum())} sets)", loc="left", fontsize=10)
fig.legend([Line2D([], [], color=fb_colors[0], lw=1.8), Line2D([], [], color=fb_colors[1], lw=2.2),
            Line2D([], [], color=fb_colors[1], lw=0.8, alpha=0.6)],
           ["w/o feedback", "w/ feedback, optimized model", "w/ feedback, perturbed sets"],
           loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.515), fontsize=9)

ax = fig.add_subplot(gs[1, :])
E0, E0s = z["rsa_epoch"][i_opt, 0], z["rsa_epoch_sem"][i_opt, 0]
for gi, (lev, lab) in enumerate(LEVELS):
    y = z["rsa_epoch"][sel(lev), k25]
    ax.scatter(gi + rng.uniform(-width, width, len(y)), y, s=10, color=fb_colors[1], alpha=0.6, lw=0, zorder=2)
for gi in range(3):                         # same segment length for both reference lines
    ax.hlines(E0, gi - 0.35, gi + 0.35, color=fb_colors[0], lw=1.8, zorder=3)
    ax.hlines(z["rsa_epoch"][i_opt, k25], gi - 0.35, gi + 0.35, color=fb_colors[1], lw=1.2, ls=(0, (4, 3)), zorder=4)
ax.set_xlim(-0.6, 2.6)
ax.set_xticks(range(3), [lab for _, lab in LEVELS])
ax.set_xlabel("Perturbation magnitude")
ax.set_ylabel("Cortex-average RSA,\naveraged over time")
ax.set_title("D  Time-averaged RSA per perturbed set", loc="left", fontsize=10)
ax.legend([Line2D([], [], color=fb_colors[0], lw=1.8),
           Line2D([], [], marker="o", ls="", ms=4, color=fb_colors[1], alpha=0.6),
           Line2D([], [], color=fb_colors[1], ls=(0, (4, 3)), lw=1.2)],
          ["w/o feedback", "w/ feedback, perturbed sets", "w/ feedback, optimized model"],
          loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False, fontsize=8.5)
save(fig, "S5_scale_rsa", "S11_Fig")

for lev, lab in LEVELS:
    g = p[p.level == lev]
    print(lab, "gain>0 (RL1-3) %:", [int(round(100 * (g[f"gain25_RL{i}"] > 0).mean())) for i in (1, 2, 3)],
          "| RSA advantage > 0 %:", int(round(100 * (g.drho25_epoch > 0).mean())))
