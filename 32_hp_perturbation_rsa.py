########################################################################
# Analysis of the hyperparameter perturbation (31_hp_perturbation.py; S4, S11 Figs).
#
# For each of the 306 sets (median hyperparameters, 5 fold-specific sets, 300 perturbed
# sets) and each stored timestep (t = 0, 25, 28): accuracy change per condition from
# t = 0, relative global prediction error E(t)/E(0), and the whole-cortex
# condition-wise RSA of PCoder 3 (mean over all 137 parcels and participants), at every
# 100-ms analysis window and averaged over the epoch, computed as in 30_dynamics_rsa.py
# (MEG RDMs of the main analysis, cached by 30_ in dynamics/meg_rdms.npz).
#
# Outputs: ../data/dynamics/hp_perturbation_per_set.csv, hp_perturbation_curves.npz
#
#   python 32_hp_perturbation_rsa.py
########################################################################
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from config import fname, time_windows
from rsa_utils import correlation_rdm, spearman, window_labels, window_mask

PERT = f"{fname.data_dir}/behavior/hp_perturbation"
OUT = Path(fname.data_dir) / "dynamics"
COND = ["RW", "RL1", "RL2", "RL3"]

parts = [np.load(f, allow_pickle=True) for f in sorted(glob.glob(f"{PERT}/part*.npz"))]
order = np.argsort(np.concatenate([p["set_index"] for p in parts]))
cat = {k: np.concatenate([p[k] for p in parts])[order]
       for k in ("set_index", "level", "name", "hps", "rho", "correct", "E_cond", "condmeans", "diverged")}
assert np.array_equal(cat["set_index"], np.arange(len(order))), "missing sets"
cond, steps = parts[0]["condition"], list(parts[0]["rsa_steps"])
n_sets = len(order)

rdms = np.array([[correlation_rdm(np.asarray(cat["condmeans"][s, i], float)[:, :, None])[0]
                  for i in range(len(steps))] for s in range(n_sets)])          # (set, step, 6)
assert np.allclose(rdms[:, 0], rdms[0, 0]), "t = 0 must not depend on the hyperparameters"

z = np.load(OUT / "meg_rdms.npz")
meg, times = z["rdms"], z["times"]
masks = [window_mask(times, w) for w in time_windows]
models = rdms.reshape(-1, 6)
rsa, tc = 0, 0
for parcel in range(meg.shape[0]):
    r = spearman(meg[parcel], models)                                            # (sub, time, k)
    w = np.stack([r[:, m].mean(1) for m in masks], 1)                            # (sub, window, k)
    rsa = rsa + w
    tc = tc + r
    print(f"parcel {parcel}", end="\r", flush=True)
rsa = (rsa / meg.shape[0]).reshape(meg.shape[1], len(time_windows), n_sets, len(steps))
# time-resolved cortex-average RSA at every 100-ms analysis window (no window averaging)
tc = (tc / meg.shape[0]).reshape(meg.shape[1], len(times), n_sets, len(steps))   # (sub, time, set, step)
drho_tc = (tc[..., 1:] - tc[..., :1]).mean(0).transpose(1, 2, 0)                 # (set, step>0, time)
rsa_tc = tc.mean(0).transpose(1, 2, 0)                                           # (set, step, time) raw RSA
rsa_tc_sem = (tc.std(0, ddof=1) / np.sqrt(tc.shape[0])).transpose(1, 2, 0)       # SEM over participants
epoch = (times >= 0) & (times <= 1.0)

acc = np.stack([cat["correct"][:, :, cond == c].mean(2) for c in range(4)], 2)  # (set, T, cond)
E = cat["E_cond"].mean(2)                                                        # (set, T)
rows = []
for s in range(n_sets):
    row = {"set": s, "level": cat["level"][s], "name": cat["name"][s], "rho": cat["rho"][s],
           "diverged": bool(cat["diverged"][s]), "t_min_E": int(np.argmin(E[s]))}
    row.update({f"hp{i}": v for i, v in enumerate(cat["hps"][s])})
    for t in steps[1:]:
        row[f"E_rel{t}"] = E[s, t] / E[s, 0]
        for c, lab in enumerate(COND):
            row[f"gain{t}_{lab}"] = acc[s, t, c] - acc[s, 0, c]
        row[f"drho{t}_epoch"] = drho_tc[s, steps.index(t) - 1, epoch].mean()
        for w, lab in enumerate(window_labels()):
            row[f"drho{t}_{lab}"] = (rsa[:, w, s, steps.index(t)] - rsa[:, w, s, 0]).mean()
    rows.append(row)
df = pd.DataFrame(rows)
df.to_csv(OUT / "hp_perturbation_per_set.csv", index=False)
np.savez_compressed(OUT / "hp_perturbation_curves.npz", acc=acc, E=E, rho=cat["rho"], name=cat["name"],
                    level=cat["level"], drho_tc=drho_tc, drho_steps=np.array(steps[1:]), meg_times=times,
                    rsa_tc=rsa_tc, rsa_tc_sem=rsa_tc_sem, rsa_steps=np.array(steps),
                    rsa_epoch=rsa_tc[..., epoch].mean(-1),
                    rsa_epoch_sem=tc[:, epoch].mean(1).std(0, ddof=1) / np.sqrt(tc.shape[0]))
print("saved ->", OUT / "hp_perturbation_per_set.csv")
