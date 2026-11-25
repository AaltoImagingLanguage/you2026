#########################################################################
# Reliability of the accuracy change t = 0 -> t = 25 across stimuli and
# training runs (original model and seeds 5, 15, 25, 35), from the per-stimulus
# predictions of 12_eval_runs_behavior.py (S2 Table, S2 Fig).
#
# Models
#   PC held-out  RW and RL1 (never used for tuning): median hyperparameters;
#                RL2 and RL3: nested cross-validation, each item classified with
#                the hyperparameters of the outer loop in which its fold was the
#                test fold (never used for training or selection)
#   PC median    median hyperparameters for all conditions (as in Fig 3B;
#                RL2/RL3 in-sample)
# Uncertainty is quantified by a paired bootstrap, which gives an interval for the
# size of the change and extends to the seed level.
# Per seed and condition (descriptive): accuracy at t = 0 and t = 25, the paired
# change, the numbers of stimuli gained/lost, and a 95% paired-bootstrap interval
# over stimuli.
# Across seeds (inferential): mean change with a 95% crossed bootstrap interval that
# resamples seeds and stimuli in every iteration, and a two-sided bootstrap p value
# (2 x the smaller tail of the bootstrap distribution beyond 0). Correction family: the
# held-out PC model in RL1, RL2, RL3 (m = 3, Benjamini-Hochberg); RW is at ceiling
# (change 0 for every stimulus and seed: not a test); PC median is descriptive.
#
#   python 13_runs_behavior_stats.py
#########################################################################
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mne.stats import fdr_correction

from config import category_colors, fb_timestep, fname

N_BOOT = 10000
CONDS = ["RW", "RL1", "RL2", "RL3"]
MODELS = ["PC held-out", "PC median"]
in_dir = Path(f"{fname.data_dir}/behavior/seeds")
# Runs: the original model and seeds 5, 15, 25, 35
RUNS = {"orig": in_dir / "item_preds_orig.csv",
        **{f"seed{s}": in_dir / f"item_preds_seed{s}.csv" for s in (5, 15, 25, 35)}}
rng = np.random.default_rng(0)


def load():
    out = {}
    for seed, f in RUNS.items():
        d = pd.read_csv(f)
        oof = d[d.model.str.startswith("PC nested")].copy()
        oof = oof[oof.condition.isin(["RL2", "RL3"])
                  & (oof.model.str[-1].astype(int) == oof.fold)]
        held = pd.concat([d[(d.model == "PC median") & d.condition.isin(["RW", "RL1"])], oof])
        held["model"] = "PC held-out"
        d = pd.concat([d[d.model == "PC median"], held])
        out[seed] = d.pivot_table(index=["model", "condition", "key"], columns="t",
                                  values="correct")
    return out


data = load()
seeds = list(data)
print("seeds:", seeds)

# ------------------------------------------------------------ per seed
rows = []
for s, d in data.items():
    for m in MODELS:
        for c in CONDS:
            x = d.loc[(m, c)]
            a0, a1 = x[0].to_numpy(), x[fb_timestep].to_numpy()
            b, cc = int(((a0 == 0) & (a1 == 1)).sum()), int(((a0 == 1) & (a1 == 0)).sum())
            diff = a1 - a0
            idx = rng.integers(0, len(diff), (N_BOOT, len(diff)))
            boot = diff[idx].mean(1) * 100
            rows.append(dict(seed=s, model=m, condition=c, n=len(diff),
                             acc_t0=a0.mean() * 100, acc_t25=a1.mean() * 100,
                             delta=diff.mean() * 100,
                             ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                             gained=b, lost=cc))
per_seed = pd.DataFrame(rows)
# ------------------------------------------------------------ across seeds
rows = []
for m in MODELS:
    for c in CONDS:
        # (seed, item) matrix of paired changes; items are the same across seeds
        D = np.array([(data[s].loc[(m, c)][fb_timestep] - data[s].loc[(m, c)][0])
                      .sort_index().to_numpy() for s in seeds])
        si = rng.integers(0, len(seeds), (N_BOOT, len(seeds)))
        ii = rng.integers(0, D.shape[1], (N_BOOT, D.shape[1]))
        boot = np.array([D[si[b]][:, ii[b]].mean() for b in range(N_BOOT)]) * 100
        ps = per_seed[(per_seed.model == m) & (per_seed.condition == c)]
        rows.append(dict(model=m, condition=c, n_seeds=len(seeds),
                         acc_t0=ps.acc_t0.mean(), acc_t25=ps.acc_t25.mean(),
                         delta=D.mean() * 100, delta_sd_seeds=ps.delta.std(ddof=1),
                         delta_min=ps.delta.min(), delta_max=ps.delta.max(),
                         ci_low=np.percentile(boot, 2.5), ci_high=np.percentile(boot, 97.5),
                         p_boot=min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean())),
                         seeds_positive=int((ps.delta > 0).sum()),
                         seeds_ci_above0=int((ps.ci_low > 0).sum())))
across = pd.DataFrame(rows)
FAMILY = (across.model == "PC held-out") & across.condition.isin(["RL1", "RL2", "RL3"])
across["p_boot_fdr"] = np.nan
across.loc[FAMILY, "p_boot_fdr"] = fdr_correction(across.loc[FAMILY, "p_boot"].to_numpy())[1]

per_seed.to_csv(in_dir / "behavior_per_seed.csv", index=False)
across.to_csv(in_dir / "behavior_across_seeds.csv", index=False)
pd.set_option("display.width", 250)
print(per_seed.round(3).to_string(index=False))
print(across.round(2).to_string(index=False))

# ------------------------------------------------------------ figure
# S2 Fig: the solid lines of Fig 3A and B repeated for every run, in the
# style of Fig 3 (broken y axis); original model thick, the four retrained seeds thin.
#   A  mean across the five outer folds of the nested cross-validation (as in Fig 3A:
#      fold f's nested hyperparameters; RL2/RL3 on the 27 stimuli of fold f, RW/RL1 on all)
#   B  median hyperparameters applied to all stimuli (as in Fig 3B)
#   C, D  RNN and GRU controls (07_, retrained on each run's backbone with the run's seed;
#      07_train_rnn_gru_controls.py), continuous axis because GRU accuracies overlap across conditions
from brokenaxes import brokenaxes
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


def curves_nested(f):
    d = pd.read_csv(f)
    acc = []
    for k_ in range(5):
        x = d[d.model == f"PC nested {k_}"]
        x = x[x.condition.isin(["RW", "RL1"]) | (x.fold == k_)]
        acc.append(x.groupby(["condition", "t"]).correct.mean().unstack() * 100)
    return sum(acc) / len(acc)


def curves_median(f):
    d = pd.read_csv(f)
    return d[d.model == "PC median"].groupby(["condition", "t"]).correct.mean().unstack() * 100


def curves_control(run, cell):
    f = RUNS[run] if run == "orig" else in_dir / f"item_preds_{run}_rnn_gru.csv"
    d = pd.read_csv(f)
    return d[d.model == cell].groupby(["condition", "t"]).correct.mean().unstack() * 100


panels = {"A": {r: curves_nested(f) for r, f in RUNS.items()},
          "B": {r: curves_median(f) for r, f in RUNS.items()}}
YL = ((5, 25), (44, 73), (76, 96), (97.5, 100.5))
lo = min(v.min().min() for p_ in panels.values() for v in p_.values())
assert lo >= YL[0][0], lo
fig = plt.figure(figsize=(14, 13.5))
gs = GridSpec(2, 2, figure=fig, wspace=0.45, hspace=0.35)
new_seeds = [r for r in RUNS if r != "orig"]
for j, (lab, cur) in enumerate(panels.items()):
    bax = brokenaxes(ylims=YL, hspace=0.2, fig=fig, subplot_spec=gs[j],
                     diag_color=plt.rcParams["axes.edgecolor"])
    for c, col in zip(CONDS, category_colors):
        for r in new_seeds:
            bax.plot(cur[r].columns, cur[r].loc[c].values, color=col, lw=1, alpha=0.55)
        bax.plot(cur["orig"].columns, cur["orig"].loc[c].values, color=col, lw=2.5)
    bax.vlines(fb_timestep, YL[0][0], YL[-1][1], colors="grey", linestyles="dashed", linewidth=2)
    bax.set_xlabel("Timesteps", fontsize=20, labelpad=25)
    bax.set_ylabel("Accuracy (%)", fontsize=20, labelpad=40)
    bax.tick_params(axis="both", labelsize=18)
    bax.axs[0].text(-0.2, 1.3, lab, transform=bax.axs[0].transAxes, fontsize=24, fontweight="bold")
    if j == 1:
        legend_ax = bax.axs[0]
for j, (lab, cell) in enumerate((("C", "RNN"), ("D", "GRU"))):
    ax = fig.add_subplot(gs[1, j])
    for c, col in zip(CONDS, category_colors):
        for r in new_seeds:
            y = curves_control(r, cell)
            ax.plot(y.columns, y.loc[c].values, color=col, lw=1, alpha=0.55)
        y = curves_control("orig", cell)
        ax.plot(y.columns, y.loc[c].values, color=col, lw=2.5)
    ax.axvline(fb_timestep, color="grey", ls="--", lw=2)
    ax.set_ylim(0, 102)
    ax.set_xlabel("Timesteps", fontsize=20)
    ax.set_ylabel("Accuracy (%)", fontsize=20)
    ax.tick_params(axis="both", labelsize=18)
    ax.spines[["right", "top"]].set_visible(False)
    ax.set_title(f"{cell} control", fontsize=20)
    ax.text(-0.2, 1.05, lab, transform=ax.transAxes, fontsize=24, fontweight="bold")
handles = [Line2D([], [], color=col, lw=2.5, label=c) for c, col in zip(CONDS, category_colors)]
handles += [Line2D([], [], color="none", label=""),
            Line2D([], [], color="0.3", lw=2.5, label="original"),
            Line2D([], [], color="0.3", lw=1, alpha=0.55, label="retrained seeds")]
legend_ax.legend(handles=handles, bbox_to_anchor=(1, 1), loc="upper left", fontsize=18, frameon=False)
for ext in ("pdf", "png"):
    fig.savefig(f"{fname.figures_dir}/behav/seed_accuracy.{ext}", bbox_inches="tight", dpi=300)
fig.savefig(f"{fname.figures_dir}/behav/seed_accuracy.tif", bbox_inches="tight", dpi=300,
            pil_kwargs={"compression": "tiff_lzw"})
for lab, cur in panels.items():
    print(lab, "t0 -> t25 per run:")
    print(pd.DataFrame({r: cur[r][0].round(1).astype(str) + " -> " + cur[r][fb_timestep].round(1).astype(str)
                        for r in RUNS}).to_string())
print("saved figure and tables to", in_dir)
