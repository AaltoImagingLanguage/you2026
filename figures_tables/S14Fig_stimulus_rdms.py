########################################################################
# S14 Fig: stimulus-level (540 x 540) RDMs of the MEG data and the model.
#
# Rows 1-2: left vOT and ST MEG RDMs at the samples nearest 100, 300, 500, 700 and
# 900 ms; the MEG patterns are averaged across participants first, and the
# correlation-distance RDM over the 540 stimuli is computed at each single sample.
# Each ROI row has its own colour scale. Row 3: model RDMs of PCoders 1-3 without
# (t = 0) and with feedback (t = 25). The RDMs are descriptive.
#
# Inputs:  ../data/behavior/stim_item_meg_rdm_group_{vOT,ST}_RL23.npz and
#          ../data/behavior/stim_item_model_rdms_RL23.npz (included in the data release);
#          if missing, they are computed from ../data/meg_tc/{40,65}_ds100Hz_sti_ver.nc
#          (26_meg_vertex_time_courses.py, participant-level, not shared) and
#          ../data/behavior/pcoder_reps_conds_RL23.pkl
# Output:  figures/rsa/stim_item_rdms_combined_RL23.{png,pdf,tif}
#
#   python figures_tables/S14Fig_stimulus_rdms.py
########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from scipy.spatial.distance import pdist, squareform

from config import fb_timestep, fname, metric_rdm, time_len

pa = argparse.ArgumentParser(description=__doc__)
pa.add_argument("--hps", type=str, default="RL23", choices=["RL23"])
pa.add_argument("--fig_ext", nargs="+", default=["png", "pdf", "tif"], choices=["png", "pdf", "tif"])
pa.add_argument("--meg_rdm_times_ms", nargs="+", type=float, default=[100, 300, 500, 700, 900],
                help="times (ms) of the MEG RDMs")
args = pa.parse_args()
args.parcel_ind = 40
roi_id = args.parcel_ind
roi_name = {40: "vOT", 65: "ST"}[roi_id]
suf = "_RL23" if args.hps == "RL23" else ""
REPS = str(fname.pcoder_reps).replace(".pkl", "_RL23.pkl") if args.hps == "RL23" else fname.pcoder_reps
OUT = f"{fname.data_dir}/behavior/stim_rsa_unique_recurrent_{roi_name}{suf}.pkl"
RDM_OUT = f"{fname.data_dir}/behavior/stim_item_meg_rdm_group_{roi_name}{suf}.npz"
MODEL_RDM_OUT = f"{fname.data_dir}/behavior/stim_item_model_rdms{suf}.npz"
STATES = {"t0": 0, "t25": fb_timestep}          # feed-forward vs converged


# ---- stimuli, in the fixed order used everywhere (csv columns flattened) ------
csv = pd.read_csv(fname.stimuli_con_dir)
conds = csv.columns.tolist()
stimuli = [w for c in conds for w in csv[c].tolist()]        # 540 strings
N = len(stimuli)


def rdm(mat):
    """condensed correlation RDM over rows of `mat` (n_items, features)."""
    return pdist(mat, metric=metric_rdm)


def _save_model_rdm_archive(model_array, pcoder_names):
    """Atomically save model RDMs so concurrent ROI jobs cannot corrupt them."""
    output = Path(MODEL_RDM_OUT)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f"{output.stem}.{os.getpid()}.tmp.npz")
    np.savez_compressed(
        temporary,
        condensed_rdm=np.asarray(model_array, dtype=np.float32),
        pcoders=np.asarray([str(name) for name in pcoder_names]),
        states=np.asarray(list(STATES)),
        timesteps=np.asarray(list(STATES.values()), dtype=np.int16),
        stimuli=np.asarray(stimuli),
        conditions=np.asarray(conds),
        metric=np.asarray(metric_rdm),
    )
    os.replace(temporary, output)
    print("saved ->", output)


def _save_group_meg_rdm_archive(group_rdm, item_n, times_ms, output_path=None):
    """Save exact-time RDMs computed from participant-averaged MEG patterns."""
    output = Path(RDM_OUT if output_path is None else output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        condensed_rdm=np.asarray(group_rdm, dtype=np.float32),
        item_n=np.asarray(item_n, dtype=np.uint8),
        times_ms=np.asarray(times_ms, dtype=np.float32),
        time_aggregation=np.asarray("single_sample"),
        participant_aggregation=np.asarray("mean_pattern_then_rdm"),
        stimuli=np.asarray(stimuli),
        conditions=np.asarray(conds),
        metric=np.asarray(metric_rdm),
    )
    print("saved ->", output)


def load_or_compute_group_item_meg_rdms(target_roi_id=None):
    """Load or compute single-sample MEG RDMs of participant-averaged patterns."""
    target_roi_id = roi_id if target_roi_id is None else int(target_roi_id)
    target_roi_name = {40: "vOT", 65: "ST"}[target_roi_id]
    archive = Path(
        f"{fname.data_dir}/behavior/"
        f"stim_item_meg_rdm_group_{target_roi_name}{suf}.npz"
    )
    requested_times = np.asarray(args.meg_rdm_times_ms, dtype=float)
    if archive.exists():
        with np.load(archive, allow_pickle=False) as saved_rdm:
            saved_times = np.asarray(saved_rdm["times_ms"], dtype=float)
            time_method = (
                str(saved_rdm["time_aggregation"].item())
                if "time_aggregation" in saved_rdm.files else ""
            )
            participant_method = (
                str(saved_rdm["participant_aggregation"].item())
                if "participant_aggregation" in saved_rdm.files else ""
            )
            compatible = (
                time_method == "single_sample"
                and participant_method == "mean_pattern_then_rdm"
                and len(saved_times) == len(requested_times)
                and np.allclose(saved_times, requested_times, atol=0.51)
            )
            if compatible:
                result = {
                    "times": saved_times,
                    "group_meg_rdm": saved_rdm["condensed_rdm"],
                    "group_meg_rdm_item_n": saved_rdm["item_n"],
                    "conditions": saved_rdm["conditions"].tolist(),
                    "stimuli": saved_rdm["stimuli"].tolist(),
                    "roi": target_roi_name,
                }
                print("loaded ->", archive)
                return result
        print(f"{archive} has a different averaging/time definition; recomputing")

    print("computing exact-time participant-averaged item-level MEG RDMs")
    meg = xr.load_dataarray(
        fname.meg_tc_sti_ver(roi=target_roi_id, fs=100)
    ).sel(trial=stimuli)
    meg = meg.sel(time=slice(0, time_len))
    available_ms = np.asarray(meg.time.values, dtype=float) * 1000
    selected_indices = []
    for requested_time in requested_times:
        time_index = int(np.argmin(np.abs(available_ms - requested_time)))
        if time_index not in selected_indices:
            selected_indices.append(time_index)
    if len(selected_indices) != len(requested_times):
        raise ValueError("Requested MEG RDM times map to duplicate data samples.")

    # Average the MEG patterns across participants first, then compute a
    # correlation-distance RDM at each single selected sample (no temporal-window
    # averaging).
    mean_meg = meg.mean(dim="subject", skipna=True)
    group_rdms = []
    item_counts = []
    actual_times_ms = []
    for time_index in selected_indices:
        time_value = meg.time.values[time_index]
        patterns = (
            mean_meg.sel(time=time_value)
            .transpose("trial", "vertex")
            .data
        )
        group_rdms.append(pdist(patterns, metric=metric_rdm))
        sample = meg.sel(time=time_value).transpose("subject", "trial", "vertex").data
        item_counts.append(np.isfinite(sample).all(axis=2).sum(axis=0))
        actual_times_ms.append(float(time_value) * 1000)
        print(f"MEG RDM: {actual_times_ms[-1]:.0f} ms")

    group_rdm = np.asarray(group_rdms, dtype=np.float32)
    item_n = np.asarray(item_counts, dtype=np.uint8)
    times_ms = np.asarray(actual_times_ms, dtype=np.float32)
    _save_group_meg_rdm_archive(group_rdm, item_n, times_ms, archive)
    return {
        "times": times_ms,
        "group_meg_rdm": group_rdm,
        "group_meg_rdm_item_n": item_n,
        "conditions": conds,
        "stimuli": stimuli,
        "roi": target_roi_name,
    }


def load_or_compute_model_item_rdms():
    """Load compact model RDMs, computing them from saved representations once."""
    archive = Path(MODEL_RDM_OUT)
    if archive.exists():
        with np.load(archive, allow_pickle=False) as saved:
            result = {
                "condensed_rdm": saved["condensed_rdm"],
                "pcoders": saved["pcoders"].tolist(),
                "states": saved["states"].tolist(),
                "timesteps": saved["timesteps"].tolist(),
                "conditions": saved["conditions"].tolist(),
            }
        print("loaded ->", archive)
        return result

    print(f"{archive} not found; computing model RDMs from {REPS}")
    with open(REPS, "rb") as file:
        representation_dict = pickle.load(file)
    pcoder_names = list(representation_dict)
    state_rdms = [[] for _ in STATES]
    for pcoder_name in pcoder_names:
        representations = np.asarray(
            [representation for condition in conds
             for representation in representation_dict[pcoder_name][condition]]
        )
        for state_index, timestep in enumerate(STATES.values()):
            state_rdms[state_index].append(rdm(representations[:, timestep, :]))
        del representations
    model_array = np.asarray(state_rdms, dtype=np.float32)
    _save_model_rdm_archive(model_array, pcoder_names)
    return {
        "condensed_rdm": model_array,
        "pcoders": [str(name) for name in pcoder_names],
        "states": list(STATES),
        "timesteps": list(STATES.values()),
        "conditions": conds,
    }


def plot_combined_item_rdms(vot_payload, st_payload, model_payload=None):
    """Plot vOT/ST RDMs, optionally followed by the model RDMs."""
    condition_payload = model_payload if model_payload is not None else vot_payload
    condition_names = [
        str(name).replace("PW", "")
        for name in condition_payload.get("conditions", conds)
    ]
    condition_size = N // len(condition_names)
    centers = np.arange(len(condition_names)) * condition_size + (condition_size - 1) / 2
    boundaries = np.arange(1, len(condition_names)) * condition_size - 0.5

    # Compact three-row layout; font sizes chosen to stay legible at page width.
    if model_payload is None:
        fig = plt.figure(figsize=(12.0, 4.55))
        outer = fig.add_gridspec(
            2, 1, height_ratios=[1.0, 1.0],
            left=0.055, right=0.965, bottom=0.075, top=0.975, hspace=0.42,
        )
    else:
        fig = plt.figure(figsize=(12.0, 7.0))
        outer = fig.add_gridspec(
            3, 1, height_ratios=[1.0, 1.0, 1.08],
            left=0.055, right=0.965, bottom=0.065, top=0.975, hspace=0.46,
        )

    def draw_matrix(ax, condensed, norm):
        image = ax.imshow(
            squareform(condensed), cmap="viridis", norm=norm, origin="upper",
            interpolation="nearest", rasterized=True,
        )
        for boundary in boundaries:
            ax.axhline(boundary, color="white", lw=0.4, alpha=0.8)
            ax.axvline(boundary, color="white", lw=0.4, alpha=0.8)
        ax.set_xticks(centers)
        ax.set_yticks(centers)
        ax.tick_params(length=0, labelsize=9.5)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_visible(False)
        return image

    # Each neural ROI receives its own full-range color scale across the five
    # displayed time points. This preserves structure without clipping and
    # avoids implying that vOT and ST have identical distance distributions.
    for row_index, neural_payload in enumerate((vot_payload, st_payload)):
        row_grid = outer[row_index].subgridspec(
            1, 6, width_ratios=[1, 1, 1, 1, 1, 0.055], wspace=0.11
        )
        rdm_times = np.asarray(neural_payload["times"], dtype=float)
        requested = np.asarray(args.meg_rdm_times_ms, dtype=float)
        selected = [int(np.argmin(np.abs(rdm_times - time))) for time in requested]
        neural_rdms = np.asarray(neural_payload["group_meg_rdm"], dtype=float)[selected]
        finite = neural_rdms[np.isfinite(neural_rdms)]
        vmax = float(np.max(finite)) if finite.size else 1.0
        norm = Normalize(vmin=0.0, vmax=vmax if vmax > 0 else 1.0)
        row_axes = []
        image = None
        for panel_index, time_index in enumerate(selected):
            ax = fig.add_subplot(row_grid[0, panel_index])
            row_axes.append(ax)
            image = draw_matrix(
                ax,
                np.asarray(neural_payload["group_meg_rdm"])[time_index],
                norm,
            )
            ax.set_title(f"{rdm_times[time_index]:.0f} ms", fontsize=12, pad=5)
            ax.set_xticklabels(condition_names)
            if panel_index == 0:
                ax.set_yticklabels(condition_names)
                ax.text(
                    -0.34, 0.5, neural_payload["roi"],
                    transform=ax.transAxes, rotation=90, ha="center", va="center",
                    fontsize=13, fontweight="bold", clip_on=False,
                )
            else:
                ax.set_yticklabels([])
        colorbar_ax = fig.add_subplot(row_grid[0, 5])
        cbar = fig.colorbar(image, cax=colorbar_ax)
        cbar.ax.tick_params(labelsize=9.5)
        cbar.set_label("correlation distance", rotation=270,
                       labelpad=14, fontsize=10.5)

    def save_combined_figure(base):
        Path(base).parent.mkdir(parents=True, exist_ok=True)
        for ext in args.fig_ext:
            save_kwargs = {"bbox_inches": "tight", "dpi": 300}
            if ext == "tif":
                # A lower raster resolution plus lossless LZW compression keeps
                # the dense TIFF comfortably below typical 6-MB upload limits.
                save_kwargs["dpi"] = 250
                save_kwargs["pil_kwargs"] = {"compression": "tiff_lzw"}
            fig.savefig(f"{base}.{ext}", **save_kwargs)
        plt.close(fig)
        print("saved ->", f"{base}.[" + "|".join(args.fig_ext) + "]")

    if model_payload is None:
        meg_base = f"{fname.figures_dir}/rsa/stim_item_meg_rdms_vOT_ST{suf}"
        save_combined_figure(meg_base)
        return

    # Model row: each PCoder is one adjacent pair, w/o feedback then w/ feedback.
    model_grid = outer[2].subgridspec(
        1, 7, width_ratios=[1, 1, 1, 1, 1, 1, 0.055], wspace=0.13
    )
    model_rdms = np.asarray(model_payload["condensed_rdm"], dtype=float)
    model_finite = model_rdms[np.isfinite(model_rdms)]
    model_vmax = float(np.max(model_finite)) if model_finite.size else 1.0
    model_norm = Normalize(vmin=0.0, vmax=model_vmax if model_vmax > 0 else 1.0)
    state_names = list(model_payload["states"])
    state_order = [
        state_names.index("t0") if "t0" in state_names else 0,
        state_names.index("t25") if "t25" in state_names else 1,
    ]
    state_titles = ["w/o feedback", "w/ feedback"]
    model_axes = []
    image = None
    for pcoder_index in range(model_rdms.shape[1]):
        for state_position, state_index in enumerate(state_order):
            panel_index = pcoder_index * 2 + state_position
            ax = fig.add_subplot(model_grid[0, panel_index])
            model_axes.append(ax)
            image = draw_matrix(
                ax, model_rdms[state_index, pcoder_index], model_norm
            )
            ax.set_title(state_titles[state_position], fontsize=11.5, pad=5)
            ax.set_xticklabels(condition_names)
            if panel_index == 0:
                ax.set_yticklabels(condition_names)
                ax.text(
                    -0.34, 0.5, "Model", transform=ax.transAxes, rotation=90,
                    ha="center", va="center", fontsize=13, fontweight="bold",
                    clip_on=False,
                )
            else:
                ax.set_yticklabels([])

    model_colorbar_ax = fig.add_subplot(model_grid[0, 6])
    model_cbar = fig.colorbar(image, cax=model_colorbar_ax)
    model_cbar.ax.tick_params(labelsize=9.5)
    model_cbar.set_label("correlation distance", rotation=270,
                         labelpad=14, fontsize=10.5)

    # Position PCoder headings over their two state panels.
    fig.canvas.draw()
    for pcoder_index in range(model_rdms.shape[1]):
        left = model_axes[pcoder_index * 2].get_position()
        right = model_axes[pcoder_index * 2 + 1].get_position()
        fig.text(
            (left.x0 + right.x1) / 2,
            max(left.y1, right.y1) + 0.037,
            f"PCoder {pcoder_index + 1}",
            ha="center", va="bottom", fontsize=13, fontweight="bold",
        )

    combined_base = f"{fname.figures_dir}/rsa/stim_item_rdms_combined{suf}"
    save_combined_figure(combined_base)



plot_combined_item_rdms(
    load_or_compute_group_item_meg_rdms(40),
    load_or_compute_group_item_meg_rdms(65),
    load_or_compute_model_item_rdms(),
)
