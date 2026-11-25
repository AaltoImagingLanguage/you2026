########################################################################
# Condition-wise RSA time courses for all 137 parcels (input of Figs 5-6 and Table 2).
#
# MEG patterns: condition means of the single-trial source patterns morphed to
# fsaverage (17_meg_trial_patterns.py), which equal the morphed evoked
# responses, averaged in 100-ms windows (20-ms steps). Correlation-distance
# RDMs (4 x 4 conditions); model RDMs from the condition-averaged PCoder
# representations at t = 0 and t = 25 (../data/behavior/pcoder_reps_conds_RL23.pkl,
# 16_model_rdms.py); Spearman RSA per participant.
#
# Outputs: ../data/rsa_tcs/rsa_tcs_parcel{p}_RL23.nc (pcoder x subject x time x
# feedback), read by 19_rsa_wholecortex.py, 21_rsa_controls.py,
# 22_effect_sizes_clusters.py, Tab2_effect_sizes_predefined_windows.py and
# model_brain_fdr.py. Existing output files are moved to
# ../data/rsa_tcs/unmorphed_backup/ (unless a copy is already there).
#
#   python 18_rsa_parcels.py --hps RL23
########################################################################
import argparse
import shutil
from pathlib import Path

import numpy as np
import xarray as xr

from config import fname, metric_rsa
from rsa_utils import (condition_means, condition_of_item, correlation_rdm,
                          cortical_labels, load_parcel, model_item_reps,
                          model_rdms, spearman)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hps", default="RL23", choices=["RL1", "RL23"])
args = parser.parse_args()


def rsa_path(roi):
    path = str(fname.rsa_tc(roi=roi))
    return path.replace(".nc", "_RL23.nc") if args.hps == "RL23" else path


backup = Path(rsa_path(0)).parent / "unmorphed_backup"
backup.mkdir(exist_ok=True)

labels = cortical_labels()
cond = condition_of_item()
weights = np.zeros((4, cond.size))
weights[cond, np.arange(cond.size)] = 1

model = model_rdms(model_item_reps(args.hps))
pcoders = sorted({key.split()[0] for key in model})
model_matrix = np.array([[model[f"{p} {s}"] for s in ("w/o", "w/")] for p in pcoders])

for parcel in range(len(labels)):
    X0, valid, times = load_parcel(parcel)
    meg_rdms = correlation_rdm(condition_means(X0, valid, weights))   # (sub, time, 6)
    rsa = np.stack([spearman(meg_rdms, m) for m in model_matrix])     # (pcoder, sub, time, 2)
    subjects = np.load(Path(fname.data_dir) / "meg_trial_patterns" / f"parcel{parcel}.npz")["subjects"]
    data = xr.DataArray(
        rsa,
        dims=["pcoder", "subject", "time", "feedback"],
        coords={"pcoder": range(len(pcoders)), "subject": subjects,
                "time": np.round(times, 2), "feedback": ["w/o", "w/"]},
        attrs={"description": f"RSA result for model and parcel {parcel} (morphed MEG patterns)",
               "metric_rsa": metric_rsa},
    )
    out = Path(rsa_path(parcel))
    if out.exists() and not (backup / out.name).exists():
        shutil.move(out, backup / out.name)
    data.to_netcdf(out)

whole = Path(rsa_path("whole_brain"))
if whole.exists():
    if (backup / whole.name).exists():
        whole.unlink()                 # a copy is already in the backup folder
    else:
        shutil.move(whole, backup / whole.name)
print(f"computed {len(labels)} parcels; earlier output files moved to {backup}")
