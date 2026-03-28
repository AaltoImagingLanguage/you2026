
from plotnine import *
import numpy as np
import matplotlib.pyplot as plt
import pickle
from matplotlib.lines import Line2D
from config import (
    category_colors,
    event_id,
    fname,
 k
)


with open(fname.word2idx_dir, "rb") as file:
    word2idx10k = pickle.load(file)

words1k = list(word2idx10k.keys())
sti_types = list(event_id.keys())

fontsize = 18


overall_accs = {sti: [] for sti in sti_types}
for n_fold in range(k):

    with open(
        fname.accs(n_fold=n_fold),
        "rb",
    ) as f:
        accs_sti_dict = pickle.load(f)

    for i, (sti, data) in enumerate(accs_sti_dict.items()):
        acc_means = {k: np.mean(v) * 100 for k, v in data.items()}
        mean=list(acc_means.values())
        overall_accs[sti].append(mean)

averaged_accs = {sti: np.mean(values, axis=0) for sti, values in overall_accs.items()}

ylims=((8,23), (50, 67), (76, 98), (99, 101))


from brokenaxes import brokenaxes

current_color = plt.rcParams["axes.edgecolor"]

fig = plt.figure(figsize=(7, 7))
bax = brokenaxes(
    ylims=ylims,
    hspace=0.2,
    fig=fig,
    diag_color=current_color,
)

for i, (sti, data) in enumerate(averaged_accs.items()):
    data = np.asarray(data)
    x = np.arange(data.shape[0])
    # plot per-fold accuracy lines in a lighter color
    folds = np.asarray(overall_accs[sti])  # shape: (n_folds, timesteps)
    if folds.ndim == 2:
        for f in folds:
            bax.plot(x, f, color=category_colors[i], 
                    #  marker="o",
                        alpha=0.3, linewidth=2
                        )

    # plot the mean across folds (bolder)
    bax.plot(
        x,
        data,
        # marker="o",
        label=sti[:3],
        color=category_colors[i],
        linewidth=2,
    )

bax.set_xlabel("Timesteps", fontsize=20, labelpad=25)
# plt.xlim(0, 29)
bax.set_ylabel("Accuracy (%)", fontsize=20, labelpad=40)
mean_handles = []
mean_labels = []
for i, sti in enumerate(averaged_accs.keys()):
    h = Line2D([0], [0], color=category_colors[i], linewidth=2,)
    mean_handles.append(h)
    mean_labels.append(sti[:3])

ax = plt.gca()

# Place both legends inside the axes at the same vertical position (axes coords).
# This keeps them visually aligned and prevents clipping in the saved figure.

left_x = 1


# Mean legend (one entry per stimulus group) - inside axes, solid line + marker
mean_legend = ax.legend(
    mean_handles,
    mean_labels,
    bbox_to_anchor=(left_x, 0.8),
    loc="center left",
    fontsize=fontsize,
    title_fontsize=fontsize,
    title="Mean",
    bbox_transform=ax.transAxes,
    frameon=False,
)
ax.add_artist(mean_legend)
# Per-fold legend: show a faint, colored handle per condition (one per stimulus)
# This mirrors the per-fold traces (which use the group color with low alpha).
fold_handles = []
fold_labels = []
for i, sti in enumerate(averaged_accs.keys()):
    fh = Line2D(
        [0], [0], color=category_colors[i], alpha=0.3, linewidth=2,
    )
    fold_handles.append(fh)
    fold_labels.append(sti[:3])

# place per-condition faint legend at the same vertical level but to the right
faint_legend = ax.legend(
    fold_handles,
    fold_labels,
    bbox_to_anchor=(left_x, 0.3),
    loc="center left",
    fontsize=fontsize,
    title="Per fold",
    title_fontsize=fontsize,
    bbox_transform=ax.transAxes,
    frameon=False,
)
ax.add_artist(faint_legend)

bax.tick_params(axis="both", labelsize=fontsize)
fig = plt.gcf()
plt.savefig(
    fname.fig_acc_mean,
    bbox_inches="tight",
        bbox_extra_artists=[mean_legend, faint_legend],
)
plt.show()
print("done")
