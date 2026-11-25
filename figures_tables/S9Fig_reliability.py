########################################################################
# S9 Fig: reliability and uncertainty of the condition-wise RSA, as time courses.
# One row of two panels. Whole cortex only (average of 137 parcels); every panel on the same time
# axis as Fig 5B and S8 Fig B: the START of each 100-ms analysis window (20-ms steps).
# (The noise ceiling of the MEG RDMs is shown in S8 Fig A, made by S6Fig_S8Fig_rsa_baselines.py.)
#   A  RSA difference with cross-validated squared Euclidean RDMs from independent
#      stimulus halves (mean, 95% t interval across participants). Bar: temporal cluster
#      with p < 0.05 in a two-tailed cluster-based permutation test across participants on
#      the cortex-averaged time course (5000 permutations, seed 42), as in Fig 5B; a single
#      robustness test of the main effect (own family, m = 1). Cluster limits are descriptive.
#   B  PCoder 3 RSA difference (w/ - w/o) with percentile 95% intervals from a crossed bootstrap
#      (1000 iterations) resampling stimuli only (grey band), participants only (blue band) or
#      both (dashed outline) (descriptive; same resamples in every parcel, so the cortex
#      average is valid).
# Inputs: ../data/rsa_baselines/parcels_RL23/parcel*.npz (24_), rsa_condition_baselines_RL23.nc (23_),
#         rsa_crossval_summary_RL23.csv (24_rsa_reliability_bootstrap_cv.py)
# Outputs: figures/rsa_baselines/S9_timecourses.{pdf,png}, figures/supporting_information/S9_Fig.tif (PDF in src/)
#   python figures_tables/S9Fig_reliability.py
########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from config import fname, time_windows
from si_utils import pdf_to_tif

D = Path(fname.data_dir) / "rsa_baselines"
FIG = Path("figures/rsa_baselines")
BOOT_MEASURES = ["w/o", "w/", "severity", "w/o | severity", "w/ | severity"]   # as in 31_
KINDS = {"participants": ("boot_subjects", "#0072B2"), "stimuli": ("boot_items", "#E69F00"),
         "participants + stimuli": ("boot_two_way", "0.25")}

files = [np.load(D / "parcels_RL23" / f"parcel{p}.npz") for p in range(137)]
t = files[0]["times"] * 1000                                   # window starts (ms)
stack = lambda key: np.stack([f[key] for f in files])         # noqa: E731

# ---------------------------------------------------------------- reliability (printed only)
rel = stack("reliability")                                     # (parcel, sub, t)
r_sub = np.tanh(np.arctanh(np.clip(rel, -0.999, 0.999)).mean(0))   # (sub, t), as 31_
z = np.arctanh(r_sub)
sb = lambda r: 2 * r / (1 + r)                                 # noqa: E731
rel_mean, rel_lo, rel_hi = (sb(np.tanh(z.mean(0) + k * stats.sem(z))) for k in (0, -1, 1))
nc = xr.load_dataset(D / "rsa_condition_baselines_RL23.nc")["noise_ceiling"].mean("parcel")
assert np.allclose(nc.time.values * 1000, t)

# ---------------------------------------------------------------- B: crossed-bootstrap intervals
m = {name: i for i, name in enumerate(BOOT_MEASURES)}
full = stack("full").mean(0)                                   # (sub, t, measure), cortex average
CONTRASTS = [("Raw", "w/", "w/o")]
boot = {kind: stack(key).mean(0) for kind, (key, _) in KINDS.items()}   # (B, t, measure)

# ---------------------------------------------------------------- A: cross-validated RDMs
cv = stack("cv_rsa")[:, 0].mean(0)                             # stimulus cv-Euclidean: (sub, t, cand)
d_cv = cv[..., 1] - cv[..., 0]
cv_ci = stats.t.interval(0.95, d_cv.shape[0] - 1, loc=d_cv.mean(0), scale=stats.sem(d_cv))
import mne  # noqa: E402
_, cv_clusters, cv_p, _ = mne.stats.permutation_cluster_1samp_test(
    d_cv, n_permutations=5000, tail=0, seed=42, out_type="mask", verbose=False)
for c, p in zip(cv_clusters, cv_p):
    print(f"A temporal cluster {t[c].min():.0f}-{t[c].max():.0f} ms, p = {p:.4f}, "
          f"mean delta in cluster {d_cv.mean(0)[c].mean():.3f}")
sig_clusters = [c for c, p in zip(cv_clusters, cv_p) if p < 0.05]

# ---------------------------------------------------------------- numbers quoted in the text
print("reliability (SB) by window start:", {int(a): round(float(b), 2) for a, b in zip(t[::5], rel_mean[::5])})
print("NC lower:", {int(a): round(float(b), 2) for a, b in zip(t[::5], nc.sel(bound="lower").values[::5])})
for name, a, b in CONTRASTS:
    obs = full[..., m[a]].mean(0) - full[..., m[b]].mean(0)
    for kind in KINDS:
        delta = boot[kind][..., m[a]] - boot[kind][..., m[b]]
        lo, hi = np.percentile(delta, [2.5, 97.5], axis=0)
        excl = t[(lo > 0) | (hi < 0)]
        print(f"B {name:24s} {kind:22s}: interval excludes 0 at {len(excl)}/{len(t)} starts"
              + (f" ({excl.min():.0f}-{excl.max():.0f} ms)" if len(excl) else ""))
print("A cv delta by start:", {int(a): round(float(b), 3) for a, b in zip(t[::5], d_cv.mean(0)[::5])})

# ---------------------------------------------------------------- figure
plt.rcParams.update({"font.size": 14})
fig, (axC, axB) = plt.subplots(1, 2, figsize=(13, 5.6), layout="constrained")

name, a, b = CONTRASTS[0]
# intervals by source, stimuli only (grey) and participants only (blue);
# and the joint interval as a dashed outline
for kind, color, alpha in [("stimuli", "0.45", 0.35), ("participants", "#0072B2", 0.35)]:
    delta = boot[kind][..., m[a]] - boot[kind][..., m[b]]
    lo, hi = np.percentile(delta, [2.5, 97.5], axis=0)
    axB.fill_between(t, lo, hi, color=color, alpha=alpha, lw=0, label=kind)
delta = boot["participants + stimuli"][..., m[a]] - boot["participants + stimuli"][..., m[b]]
lo, hi = np.percentile(delta, [2.5, 97.5], axis=0)
axB.plot(t, lo, color="k", lw=1.3, ls="--", label="participants + stimuli")
axB.plot(t, hi, color="k", lw=1.3, ls="--")
full_line, = axB.plot(t, full[..., m[a]].mean(0) - full[..., m[b]].mean(0), color="k", lw=2.5)
axB.set_ylabel("RSA difference (w/ − w/o)")
axB.set_title("Bootstrap over participants and stimuli", fontsize=15)
axB.add_artist(axB.legend(loc="upper left", frameon=False, fontsize=11, ncols=2, title_fontsize=11,
                           title="95% bootstrap interval, resampling:", alignment="left"))
axB.legend([full_line], ["full sample"], loc="lower left", frameon=False, fontsize=11)

axC.fill_between(t, cv_ci[0], cv_ci[1], color="0.3", alpha=0.25, lw=0, label="95% CI across participants")
axC.plot(t, d_cv.mean(0), color="k", lw=2.5, label="mean across participants")
y_bar = axC.get_ylim()[0] - 0.015
for i, c in enumerate(sig_clusters):
    axC.plot([t[c].min(), t[c].max()], [y_bar] * 2, color="k", lw=5, solid_capstyle="butt",
             label="temporal cluster, p < 0.05" if i == 0 else None)
axC.set_ylabel("RSA difference (w/ − w/o)")
axC.set_title("Cross-validated RDMs (stimulus halves)", fontsize=15)
axC.legend(loc="upper left", frameon=False, fontsize=11)

for ax, letter in zip((axC, axB), "AB"):
    ax.axhline(0, color="k", lw=0.8)
    ax.set_xlim(-10, 1010)
    ax.set_xlabel("Time (ms)")
    ax.spines[["right", "top"]].set_visible(False)
    ax.text(-0.16, 1.04, letter, transform=ax.transAxes, fontsize=22, fontweight="bold", va="bottom")

FIG.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG / "S9_timecourses.pdf")
fig.savefig(FIG / "S9_timecourses.png", dpi=110)
plt.close(fig)
pdf_to_tif(FIG / "S9_timecourses.pdf", "S9_Fig")
