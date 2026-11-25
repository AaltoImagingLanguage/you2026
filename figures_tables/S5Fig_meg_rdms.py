########################################################################
# S5 Fig A-B: participant-averaged MEG condition RDMs of left vOT (parcel 40) and
# left ST (parcel 65) at 100, 300, 500, 700 and 900 ms.
#
# These are the RDMs that enter the RSA (18_rsa_parcels.py): for each participant,
# the condition means of the morphed single-trial patterns in the 100-ms window
# starting at the labelled time, and the correlation distance between the four
# condition patterns. The RDMs are then averaged over participants; no RDM is
# averaged over time. The model RDMs (S5 Fig C) are plotted by 16_model_rdms.py.
#
# Input: ../data/dynamics/meg_rdms.npz (participant-wise condition RDMs of all parcels,
#        cached by 30_dynamics_rsa.py; in the data release), or, if absent,
#        ../data/meg_trial_patterns/parcel{40,65}.npz (17_meg_trial_patterns.py)
# Output: figures/rsa/meg_{vOT,ST}_rdms.pdf
#
#   python figures_tables/S5Fig_meg_rdms.py
########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from config import fname, rois
from rsa_utils import condition_means, condition_of_item, correlation_rdm, load_parcel
from utility import plot_meg_rdms

TIMES = [0.1, 0.3, 0.5, 0.7, 0.9]  # window starts, as in the ROI time courses (Fig 6)
VMAX = {"vOT": 0.45, "ST": 0.5}  # colour scale per region

cond = condition_of_item()
weights = np.zeros((4, cond.size))
weights[cond, np.arange(cond.size)] = 1

CACHE = Path(f"{fname.data_dir}/dynamics/meg_rdms.npz")
if CACHE.exists():
    cache = np.load(CACHE)


def parcel_rdms(roi_id):
    """(subject, time, 6) MEG condition RDMs of one parcel, and the window times."""
    if CACHE.exists():
        return cache["rdms"][roi_id], cache["times"]
    X0, valid, times = load_parcel(roi_id)
    return correlation_rdm(condition_means(X0, valid, weights)), times


for roi_name, roi_id in rois.items():
    meg_rdms, times = parcel_rdms(roi_id)
    rdms, names = [], []
    for t in TIMES:
        rdm = meg_rdms[:, np.argmin(np.abs(times - t))].mean(0)
        rdms.append(rdm)
        names.append(f"{t * 1000:.0f} ms")
        print(roi_name, names[-1], rdm.round(3))
    fig = plot_meg_rdms(rdms, names=names, n_rows=1, vmax=VMAX[roi_name],
                        colorbar_label="correlation distance")
    fig.set_size_inches(8.4, 2)  # narrower cells: less space between the square RDMs
    for ax in fig.axes[:-1]:  # RDM panels (the last axes is the colorbar)
        ax.title.set_fontsize(10)
        ax.tick_params(labelsize=7.5)
    fig.text(0.07, 0.9, roi_name, fontsize=11, fontweight="bold", ha="left", va="bottom")
    plt.savefig(fname.fig_meg_rdms(roi=roi_name), dpi=300, bbox_inches="tight", pad_inches=0.02)
    print("saved ->", fname.fig_meg_rdms(roi=roi_name))
