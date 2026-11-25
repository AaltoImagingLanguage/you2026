"""
Fig 2: human behavioral results of the MEG experiment (catch trials), accuracy and
reaction time per condition (RW, RL1-RL3). Reaction times come from the
sentence-completion catch trials (a semantic plausibility judgment), a different task
from the model's classification; they characterize the human behavioral pattern
(accuracy and the graded increase in RT with misspelling severity) and are not used as
a model benchmark. The statistics of the RTs are in 11_human_rt_stats.py (S1 Table).

Input: ../data/behavior/performance.csv (one row per participant and condition)
Output: figures/behav/human_acc_rt.{pdf,png,tif}

    python figures_tables/Fig2_human_behavior.py
"""
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
import os

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from config import fname, figures_dir, category_colors

plt.rcParams["font.size"] = 14

# ---- data --------------------------------------------------------------
perf = pd.read_csv(fname.human_behav, index_col=0)
perf["x"] = "x"
if "type" not in perf.columns:
    perf["type"] = perf["Type"]

order = ["RW", "RL1", "RL2", "RL3"]
# condition colors shared with the model figures (config.category_colors)
palette = dict(zip(order, category_colors))

# ---- plot: accuracy (A) and reaction time (B) --------------------------
# Style: unfilled boxplots with condition-colored edges + overlaid points.
panels = [("accuracy", "Accuracy (%)", "A"), ("rt", "Reaction time (ms)", "B")]
fig, axs = plt.subplots(1, len(panels), figsize=(8, 2.5))

for ax, (y, ylabel, plabel) in zip(axs, panels):
    sns.boxplot(data=perf, x="x", y=y, hue="type", hue_order=order,
                palette=palette, fill=False, linewidth=1.5, gap=0.15,
                showfliers=False, ax=ax)
    sns.stripplot(data=perf, x="x", y=y, hue="type", hue_order=order,
                  dodge=True, jitter=True, size=3.5, alpha=0.7, color="0.2",
                  legend=False, ax=ax)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set(xticklabels=[])
    ax.set_xlabel("")
    ax.set_ylabel(ylabel)
    # ax.text(-0.12, 1.05, plabel, transform=ax.transAxes,
    #         fontweight="bold", fontsize=16)
    if ax.get_legend() is not None:
        ax.get_legend().remove()

# single shared legend (condition colors) at the top right
handles, labels = axs[0].get_legend_handles_labels()
axs[-1].legend(handles[:len(order)], labels[:len(order)],
               bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

fig.tight_layout()

# ---- save --------------------------------------------------------------
out_dir = f"{figures_dir}/behav"
os.makedirs(out_dir, exist_ok=True)
base = f"{out_dir}/human_acc_rt"
fig.savefig(f"{base}.pdf", bbox_inches="tight")
fig.savefig(f"{base}.png", bbox_inches="tight", dpi=300)
fig.savefig(f"{base}.tif", bbox_inches="tight", dpi=300,
            pil_kwargs={"compression": "tiff_lzw"})
print("saved ->", f"{base}.[pdf|png|tif]")

# ---- summary printout (mean per condition) ----------------------------
print("\nmean by condition:")
print(perf.groupby("type")[["accuracy", "rt"]].mean().reindex(order).round(1))
