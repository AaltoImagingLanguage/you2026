"""S15 Fig: participant-level ridge-regression prediction scores (left vOT and ST).

The subject-level scores come from ``29_ridge_participant_level.py`` (PCoder x
state x time x participant). Curves: mean and 95% t confidence interval across
the 23 participants. Bars: one-tailed (w/ > w/o) paired temporal cluster
permutation tests across participants (5000 permutations, seed 50); BH-FDR
across the six ROI x PCoder cells on each cell's smallest cluster p. Solid bars
survive the correction, dashed bars are p < 0.05 uncorrected only. Layout,
colours and bar convention follow Fig 7 (``Fig7_ridge.py``).
The script also writes the time-resolved and whole-epoch CIs and classical
(pooled-SD) Cohen's d values, and a table figure of the whole-epoch effects.

Inputs:  ../data/behavior/ridge_subjinf_{vOT,ST}_RL23.pkl
Outputs: figures/ridge/ridge_subjinf_08style_vOT_ST_RL23.{pdf,png,tif} (S15 Fig)
         figures/ridge/ridge_subjinf_08style_cluster_results_RL23.csv
         figures/ridge/ridge_subjinf_{time_resolved,epoch}_ci_effect_sizes_RL23.csv
         figures/ridge/ridge_subjinf_supplementary_table_RL23.{pdf,png}

    python figures_tables/S15Fig_ridge_participant_level.py --hps RL23
"""

import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
import argparse
import csv
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import mne
import numpy as np

from config import fname

plt.style.use("default")  # this figure uses the matplotlib defaults, not the whitegrid style of config.py
from scipy.stats import t as student_t
from statsmodels.stats.multitest import multipletests


pa = argparse.ArgumentParser(description=__doc__)
pa.add_argument("--hps", default="RL23", choices=["RL1", "RL23"])
pa.add_argument("--rois", nargs="+", default=["vOT", "ST"], choices=["ST", "vOT"])
pa.add_argument("--data_dir", default=f"{fname.data_dir}/behavior")
pa.add_argument("--fig_dir", default="figures/ridge")
pa.add_argument("--n_perm", type=int, default=5000)
pa.add_argument("--seed", type=int, default=50)
pa.add_argument("--alpha", type=float, default=0.05)
args = pa.parse_args()

SUF = "_RL23" if args.hps == "RL23" else ""
OUTDIR = Path(args.fig_dir)
OUTDIR.mkdir(parents=True, exist_ok=True)
COLORS = ["lightblue", "lightcoral"]
LABELS = ["w/o feedback", "w/ feedback"]


def load_payload(roi):
    path = Path(args.data_dir) / f"ridge_subjinf_{roi}{SUF}.pkl"
    with open(path, "rb") as file:
        payload = pickle.load(file)
    subject_scores = np.asarray(payload["subj"], dtype=float)
    if subject_scores.ndim != 4 or subject_scores.shape[1] != 2:
        raise ValueError(f"Unexpected subject-level array shape in {path}: {subject_scores.shape}")
    return path, payload, subject_scores


def cluster_test(diff):
    """One-sided positive cluster test (tail=1)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="No clusters found")
        t_obs, clusters, pvals, _ = mne.stats.permutation_cluster_1samp_test(
            diff,
            n_permutations=args.n_perm,
            tail=1,
            seed=args.seed,
            verbose=False,
        )
    return t_obs, clusters, np.asarray(pvals, dtype=float)


loaded = {roi: load_payload(roi) for roi in args.rois}
reference = loaded[args.rois[0]][1]
times = np.asarray(reference["times"], dtype=float)
pcoders = list(reference["pcoders"])
for roi, (_, payload, scores) in loaded.items():
    if not np.array_equal(np.asarray(payload["times"], dtype=float), times):
        raise ValueError(f"Time coordinates differ for {roi}")
    if list(payload["pcoders"]) != pcoders:
        raise ValueError(f"PCoder coordinates differ for {roi}")
    if scores.shape[-1] != 23:
        raise ValueError(f"Expected 23 participants for {roi}, found {scores.shape[-1]}")


# Each cluster test controls family-wise error over time. BH correction is
# across the six ROI-by-PCoder cells (smallest cluster p per cell). Nominal
# clusters are dashed when they do not survive this cell-level correction.
cell_results = {}
raw_cell_p = []
cell_keys = []
for roi in args.rois:
    scores = loaded[roi][2]
    for pcoder in range(len(pcoders)):
        diff = (scores[pcoder, 1] - scores[pcoder, 0]).T  # participant x time
        t_obs, clusters, pvals = cluster_test(diff)
        key = (roi, pcoder)
        cell_results[key] = {"t_obs": t_obs, "clusters": clusters, "pvals": pvals}
        cell_keys.append(key)
        raw_cell_p.append(float(pvals.min()) if len(pvals) else 1.0)

reject, adjusted, _, _ = multipletests(raw_cell_p, alpha=args.alpha, method="fdr_bh")
for index, key in enumerate(cell_keys):
    cell_results[key].update(
        raw_p=float(raw_cell_p[index]),
        adjusted_p=float(adjusted[index]),
        reject=bool(reject[index]),
    )


def draw_two_dashes(ax, x_start, x_end, y):
    """Draw the two-dash marker of a nominal (uncorrected) cluster."""
    if x_end <= x_start:
        half_width = 6.0
        segments = ((x_start - half_width, x_start - 1.5),
                    (x_start + 1.5, x_start + half_width))
    else:
        gap = 0.2 * (x_end - x_start)
        dash = 0.4 * (x_end - x_start)
        segments = ((x_start, x_start + dash),
                    (x_start + dash + gap, x_end))
    for start, stop in segments:
        ax.plot([start, stop], [y, y], color="black", linewidth=4,
                solid_capstyle="butt")


def mean_ci(values, axis=-1):
    """Mean and two-sided 95% t confidence interval across participants."""
    values = np.asarray(values, dtype=float)
    n = values.shape[axis]
    mean = values.mean(axis=axis)
    standard_error = values.std(axis=axis, ddof=1) / np.sqrt(n)
    half_width = float(student_t.ppf(0.975, n - 1)) * standard_error
    return mean, mean - half_width, mean + half_width


def classical_cohens_d(w_off, w_on, axis=-1):
    """Classical pooled-SD Cohen's d for w/ minus w/o feedback.

    The mean paired contrast is standardized by the pooled raw within-state SD,
    not by the SD of the paired differences (d_z).
    """
    w_off = np.asarray(w_off, dtype=float)
    w_on = np.asarray(w_on, dtype=float)
    mean_difference = (w_on - w_off).mean(axis=axis)
    pooled_sd = np.sqrt(
        (w_off.var(axis=axis, ddof=1) + w_on.var(axis=axis, ddof=1)) / 2
    )
    return np.divide(
        mean_difference,
        pooled_sd,
        out=np.full_like(mean_difference, np.nan, dtype=float),
        where=pooled_sd > 0,
    )


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# Establish a common y-axis across both ROIs and all three PCoders.
n_subjects = loaded[args.rois[0]][2].shape[-1]
lower_values, upper_values = [], []
for roi in args.rois:
    scores = loaded[roi][2]
    for pcoder in range(len(pcoders)):
        for state in range(2):
            matrix = scores[pcoder, state]
            mean, ci_low, ci_high = mean_ci(matrix)
            lower_values.extend(ci_low.tolist())
            upper_values.extend(ci_high.tolist())
data_low, data_high = min(lower_values), max(upper_values)
span = max(data_high - data_low, 0.02)
y_low = data_low - 0.12 * span
y_high = data_high + 0.08 * span
bar_y = y_low + 0.035 * (y_high - y_low)


cluster_rows = []
time_effect_rows = []
epoch_effect_rows = []
fig, axes = plt.subplots(
    len(args.rois),
    len(pcoders),
    figsize=(12, 6.4),
    sharex=True,
    sharey=True,
    squeeze=False,
)
for roi_index, roi in enumerate(args.rois):
    source_path, payload, scores = loaded[roi]
    for pcoder, ax in enumerate(axes[roi_index]):
        for state in range(2):
            matrix = scores[pcoder, state]
            mean, ci_low, ci_high = mean_ci(matrix)
            ax.plot(times, mean, color=COLORS[state], linewidth=1.8, label=LABELS[state])
            ax.fill_between(times, ci_low, ci_high, color=COLORS[state], alpha=0.30,
                            linewidth=0)

        result = cell_results[(roi, pcoder)]
        nominal = np.flatnonzero(result["pvals"] < args.alpha)
        for cluster_index in nominal:
            cluster = result["clusters"][cluster_index][0]
            cluster_times = times[cluster]
            if result["reject"]:
                ax.plot(cluster_times, np.full(len(cluster_times), bar_y), color="black",
                        linewidth=4, solid_capstyle="butt")
            else:
                draw_two_dashes(ax, float(cluster_times[0]), float(cluster_times[-1]), bar_y)

        ax.axhline(0, color="#777777", linestyle="--", linewidth=0.8)
        ax.set_xlim(float(times.min()), float(times.max()))
        ax.set_ylim(y_low, y_high)
        if roi_index == len(args.rois) - 1:
            ax.set_xlabel("Time (ms)")
        if roi_index == 0:
            ax.set_title(f"PCoder {pcoder + 1}")
        ax.spines[["right", "top"]].set_visible(False)
        ax.grid(axis="y", color="#D9D9D9", linewidth=0.7, alpha=0.7)
        if pcoder == 0:
            ax.set_ylabel(f"{roi}\nPrediction score (correlation)")
        cluster_rows.append(
            {
                "roi": roi,
                "pcoder": pcoder + 1,
                "raw_min_cluster_p": result["raw_p"],
                "six_cell_bh_p": result["adjusted_p"],
                "reject": result["reject"],
                "n_nominal_clusters": len(nominal),
            }
        )

        # Time-resolved raw effects, CIs, and classical Cohen's d.
        w_off = scores[pcoder, 0]  # time x participant
        w_on = scores[pcoder, 1]
        off_mean, off_low, off_high = mean_ci(w_off)
        on_mean, on_low, on_high = mean_ci(w_on)
        difference = w_on - w_off
        diff_mean, diff_low, diff_high = mean_ci(difference)
        d_time = classical_cohens_d(w_off, w_on)
        for time_index, time_ms in enumerate(times):
            time_effect_rows.append(
                {
                    "roi": roi,
                    "pcoder": pcoder + 1,
                    "time_ms": float(time_ms),
                    "n_participants": n_subjects,
                    "without_feedback_mean_r": float(off_mean[time_index]),
                    "without_feedback_ci_low": float(off_low[time_index]),
                    "without_feedback_ci_high": float(off_high[time_index]),
                    "with_feedback_mean_r": float(on_mean[time_index]),
                    "with_feedback_ci_low": float(on_low[time_index]),
                    "with_feedback_ci_high": float(on_high[time_index]),
                    "delta_r": float(diff_mean[time_index]),
                    "delta_r_ci_low": float(diff_low[time_index]),
                    "delta_r_ci_high": float(diff_high[time_index]),
                    "classical_cohens_d": float(d_time[time_index]),
                }
            )

        # Window-independent summary: each participant is first averaged across
        # the complete analyzed epoch.
        epoch_off = w_off.mean(axis=0)
        epoch_on = w_on.mean(axis=0)
        epoch_difference = epoch_on - epoch_off
        epoch_off_mean, epoch_off_low, epoch_off_high = mean_ci(epoch_off)
        epoch_on_mean, epoch_on_low, epoch_on_high = mean_ci(epoch_on)
        epoch_diff_mean, epoch_diff_low, epoch_diff_high = mean_ci(epoch_difference)
        epoch_d = classical_cohens_d(epoch_off, epoch_on)
        epoch_effect_rows.append(
            {
                "roi": roi,
                "pcoder": pcoder + 1,
                "time_window_ms": "0-1000",
                "n_participants": n_subjects,
                "without_feedback_mean_r": float(epoch_off_mean),
                "without_feedback_ci_low": float(epoch_off_low),
                "without_feedback_ci_high": float(epoch_off_high),
                "with_feedback_mean_r": float(epoch_on_mean),
                "with_feedback_ci_low": float(epoch_on_low),
                "with_feedback_ci_high": float(epoch_on_high),
                "delta_r": float(epoch_diff_mean),
                "delta_r_ci_low": float(epoch_diff_low),
                "delta_r_ci_high": float(epoch_diff_high),
                "classical_cohens_d": float(epoch_d),
            }
        )

    print(f"loaded {source_path}")

axes[0, -1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.02), frameon=False)
fig.tight_layout(rect=(0.02, 0.04, 0.84, 0.98), w_pad=1.2, h_pad=1.5)

roi_tag = "_".join(args.rois)
stem = OUTDIR / f"ridge_subjinf_08style_{roi_tag}{SUF}"
for extension in ("pdf", "png", "tif"):
    save_args = {"bbox_inches": "tight", "pad_inches": 0.04}
    if extension in {"png", "tif"}:
        save_args.update(dpi=400, facecolor="white")
    if extension == "tif":
        save_args["pil_kwargs"] = {"compression": "tiff_lzw"}
    fig.savefig(stem.with_suffix(f".{extension}"), **save_args)
plt.close(fig)
print(f"saved -> {stem}.[pdf|png|tif]")


summary_path = OUTDIR / f"ridge_subjinf_08style_cluster_results{SUF}.csv"
with open(summary_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(cluster_rows[0]))
    writer.writeheader()
    writer.writerows(cluster_rows)
print(f"saved -> {summary_path}")

time_effect_path = OUTDIR / f"ridge_subjinf_time_resolved_ci_effect_sizes{SUF}.csv"
with open(time_effect_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(time_effect_rows[0]))
    writer.writeheader()
    writer.writerows(time_effect_rows)
print(f"saved -> {time_effect_path}")

epoch_effect_path = OUTDIR / f"ridge_subjinf_epoch_ci_effect_sizes{SUF}.csv"
with open(epoch_effect_path, "w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=list(epoch_effect_rows[0]))
    writer.writeheader()
    writer.writerows(epoch_effect_rows)
print(f"saved -> {epoch_effect_path}")


def format_interval(mean, low, high, decimals=4):
    """Format an estimate and its 95% CI for the supplementary table."""
    return f"{mean:.{decimals}f} [{low:.{decimals}f}, {high:.{decimals}f}]"


# Whole-epoch summary table (no data-driven time selection).
table_rows = []
for row in epoch_effect_rows:
    table_rows.append(
        [
            row["roi"],
            f'{row["pcoder"]}',
            format_interval(
                row["delta_r"],
                row["delta_r_ci_low"],
                row["delta_r_ci_high"],
            ),
            f'{row["classical_cohens_d"]:.2f}',
        ]
    )

table_columns = [
    "ROI",
    "PCoder",
    "Mean raw difference ($\\Delta r$)\n[95% CI]",
    "Classical Cohen's d",
]

table_fig, table_ax = plt.subplots(figsize=(8.6, 2.8))
table_ax.axis("off")
table_ax.set_position([0.0, 0.0, 1.0, 1.0])

summary_table = table_ax.table(
    cellText=table_rows,
    colLabels=table_columns,
    cellLoc="center",
    colLoc="center",
    colWidths=[0.12, 0.12, 0.52, 0.24],
    bbox=[0.01, 0.06, 0.98, 0.88],
)
summary_table.auto_set_font_size(False)
summary_table.set_fontsize(10)

n_table_rows = len(table_rows) + 1
n_table_cols = len(table_columns)
for (row_index, col_index), cell in summary_table.get_celld().items():
    cell.set_facecolor("white")
    cell.set_edgecolor("black")
    cell.set_linewidth(0)
    cell.PAD = 0.09
    if row_index == 0:
        cell.set_text_props(weight="bold")
        cell.visible_edges = "TB"
        cell.set_linewidth(1.0)
    elif row_index == n_table_rows - 1:
        cell.visible_edges = "B"
        cell.set_linewidth(1.0)
    else:
        cell.visible_edges = ""
    if col_index == 0:
        cell.get_text().set_ha("left")

# A thin midrule separates the two ROIs.
for col_index in range(n_table_cols):
    summary_table[(4, col_index)].visible_edges = "T"
    summary_table[(4, col_index)].set_linewidth(0.6)

table_stem = OUTDIR / f"ridge_subjinf_supplementary_table{SUF}"
table_fig.savefig(
    table_stem.with_suffix(".pdf"),
    bbox_inches="tight",
    pad_inches=0.05,
    facecolor="white",
)
table_fig.savefig(
    table_stem.with_suffix(".png"),
    dpi=400,
    bbox_inches="tight",
    pad_inches=0.05,
    facecolor="white",
)
plt.close(table_fig)
print(f"saved -> {table_stem}.[pdf|png]")
