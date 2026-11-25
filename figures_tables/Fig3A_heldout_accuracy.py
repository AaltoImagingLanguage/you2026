#########################################################################
# Fig 3A: held-out accuracy over recurrent timesteps. One line per outer fold f of
# the nested cross-validation (hyperparameters trained on three folds, selected on
# fold f+1):
#   RW, RL1  all 135 stimuli (never used for tuning)
#   RL2, RL3 the 27 stimuli of test fold f (never used for training or selection)
# so every point is held out. Thick line: mean over folds (for RL2/RL3, equal to the
# pooled held-out accuracy over all 135 stimuli). Same style as Fig 3B
# (Fig3BC_model_behavior.py).
# Input: ../data/behavior/seeds/item_preds_orig.csv (12_eval_runs_behavior.py)
# Output: figures/behav/iterate_accs_mean_heldout.{pdf,png,tif}
#
#   python figures_tables/Fig3A_heldout_accuracy.py
#########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from brokenaxes import brokenaxes
from matplotlib.lines import Line2D

from config import category_colors, fb_timestep, fname, k

CONDS = ["RW", "RL1", "RL2", "RL3"]
FS = 18
d = pd.read_csv(f"{fname.data_dir}/behavior/seeds/item_preds_orig.csv")
curves = {c: [] for c in CONDS}
for f in range(k):
    x = d[d.model == f"PC nested {f}"]
    x = x[x.condition.isin(["RW", "RL1"]) | (x.fold == f)]
    n = x[x.t == 0].condition.value_counts()
    assert n.RW == n.RL1 == 135 and n.RL2 == n.RL3 == 27, n
    acc = x.groupby(["condition", "t"]).correct.mean().unstack() * 100
    for c in CONDS:
        curves[c].append(acc.loc[c].to_numpy())
curves = {c: np.array(v) for c, v in curves.items()}
for c in CONDS:
    print(c, "fold range", curves[c].min().round(1), curves[c].max().round(1),
          "mean t0 -> t25", curves[c].mean(0)[0].round(1), curves[c].mean(0)[fb_timestep].round(1))

fig = plt.figure(figsize=(7, 7))
bax = brokenaxes(ylims=((0, 32), (45, 73), (82, 93), (99, 101)), hspace=0.2, fig=fig,
                 diag_color=plt.rcParams["axes.edgecolor"])
for i, c in enumerate(CONDS):
    t = np.arange(curves[c].shape[1])
    for y in curves[c]:
        bax.plot(t, y, color=category_colors[i], alpha=0.3, linewidth=2)
    bax.plot(t, curves[c].mean(0), color=category_colors[i], linewidth=2, label=c)
bax.set_xlabel("Timesteps", fontsize=20, labelpad=25)
bax.set_ylabel("Accuracy (%)", fontsize=20, labelpad=40)
bax.tick_params(axis="both", labelsize=FS)
for y0, title, alpha in ((0.72, "Mean", 1), (0.32, "Per fold", 0.3)):
    leg = fig.legend([Line2D([0], [0], color=category_colors[i], lw=2, alpha=alpha) for i in range(4)], CONDS,
                     bbox_to_anchor=(0.92, y0), loc="center left", fontsize=FS, title=title, title_fontsize=FS,
                     frameon=False)
out = "figures/behav/iterate_accs_mean_heldout"
for ext in ("pdf", "png"):
    fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=300)
fig.savefig(f"{out}.tif", bbox_inches="tight", dpi=300, pil_kwargs={"compression": "tiff_lzw"})
print("saved ->", out)
