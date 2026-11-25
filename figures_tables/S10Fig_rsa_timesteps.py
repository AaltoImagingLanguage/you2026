########################################################################
# S10 Fig, the dynamics RSA (PCoder 3, condition-wise RSA at
# timestep t minus t = 0).
#   A  descriptive: cortex-averaged Delta rho over t = 0..28 in each 200-ms
#      window (mean, 95% CI over participants); dotted lines = tested timesteps
#   B  statistics: whole-cortex cluster permutation tests against t = 0
#      (two-tailed), one BH family over all clusters of the distinct tests
#      t = 5, 10, 20, 25 (t = 15 = t = 10, identical tests);
#      each cell = smallest FDR-adjusted cluster p (log colour scale);
#      * = FDR p < .05 (all such clusters have Delta rho > 0)
# Inputs:  ../data/dynamics/rsa_trajectory.csv, ../data/dynamics/rsa_long_cluster_tests.csv
#          (30_dynamics_rsa.py)
# Outputs: figures/dynamics/dynamics_rsa.{pdf,png,tif},
#          ../data/dynamics/rsa_long_cluster_tests_distinct.csv
#
#   python figures_tables/S10Fig_rsa_timesteps.py
########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import mne
from matplotlib.colors import LogNorm

from config import fb_timestep, fname

D = Path(f"{fname.data_dir}/dynamics")
FIG = Path(f"{fname.figures_dir}/dynamics")
traj = pd.read_csv(D / "rsa_trajectory.csv")
cl_all = pd.read_csv(D / "rsa_long_cluster_tests.csv")
WINDOWS = list(dict.fromkeys(traj.window_ms))
# Identical tests are counted once. The PCoder 3 RDM rank
# order, hence every cluster, is identical at t = 10 and 15,
# so the BH family is formed over the distinct tests t = 5, 10, 20, 25 only.
key = ["window_ms", "p_cluster", "n_parcels_lh", "n_parcels_rh", "mean_delta_rho"]
same = lambda a, b: cl_all[cl_all.t == a][key].reset_index(drop=True).equals(
    cl_all[cl_all.t == b][key].reset_index(drop=True))
assert same(10, 15)
cl = cl_all[cl_all.t.isin([5, 10, 20, 25])].copy()
cl["p_fdr"] = mne.stats.fdr_correction(cl.p_cluster, alpha=0.05)[1]
cl.to_csv(D / "rsa_long_cluster_tests_distinct.csv", index=False)
print("family: m =", len(cl), "clusters from t = 5, 10, 20, 25")
SHOW = [5, 10, 15, 20, 25]          # t = 15 shown, same test as t = 10
TESTS = SHOW
COLORS = plt.cm.viridis(np.linspace(0, 0.9, len(WINDOWS)))
ALPHA = 0.05

fig, (ax, axb) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1.35, 1]})

# ---- A: descriptive trajectory ----
for w, lab in enumerate(WINDOWS):
    sub = traj[traj.window_ms == lab]
    ax.plot(sub.t, sub.delta_rho, color=COLORS[w], lw=1.5, label=f"{lab} ms")
    ax.fill_between(sub.t, sub.ci_low, sub.ci_high, color=COLORS[w], alpha=0.12, lw=0)
ax.grid(False)
for s in TESTS:
    ax.axvline(s, color="#888888", lw=0.8, ls=":", zorder=0)
ax.axhline(0, color="k", lw=0.6)
ax.axvline(fb_timestep, color="k", lw=0.8, ls="--")
ax.set_xlim(0, traj.t.max())
ax.set_xlabel("Timestep")
ax.set_ylabel(r"Cortex-averaged $\Delta\rho$ = RSA(t) $-$ RSA(0)")
ax.legend(frameon=False, fontsize=7, loc="upper left")
ax.spines[["right", "top"]].set_visible(False)

# ---- B: cluster tests ----
pmin = np.ones((len(WINDOWS), len(TESTS)))
for i, lab in enumerate(WINDOWS):
    for j, s in enumerate(TESTS):
        q = cl[(cl.window_ms == lab) & (cl.t == (10 if s == 15 else s))]
        if len(q):
            pmin[i, j] = q.p_fdr.min()
sig = pmin < ALPHA
norm = LogNorm(vmin=1e-3, vmax=1)
im = axb.imshow(np.clip(pmin, 1e-3, 1), cmap="Greys_r", norm=norm, aspect="auto")
for i in range(len(WINDOWS)):
    for j in range(len(TESTS)):
        txt = "< .001" if pmin[i, j] < 0.001 else (f"{pmin[i, j]:.3f}".lstrip("0") if pmin[i, j] < 1 else "1")
        axb.text(j, i, txt + ("*" if sig[i, j] else ""), ha="center", va="center", fontsize=8,
                 color="w" if pmin[i, j] < 0.03 else "k", weight="bold" if sig[i, j] else None)
cb = fig.colorbar(im, ax=axb, fraction=0.05, pad=0.03, ticks=[0.001, 0.01, 0.05, 0.1, 1])
cb.ax.set_yticklabels([".001", ".01", ".05", ".1", "1"])
cb.set_label("FDR-adjusted p", fontsize=8)
cb.ax.tick_params(labelsize=7)
axb.set_xticks(range(len(TESTS)))
axb.set_xticklabels(TESTS)
axb.set_yticks(range(len(WINDOWS)))
axb.set_yticklabels([f"{w} ms" for w in WINDOWS])
axb.set_xticks(np.arange(len(TESTS) + 1) - 0.5, minor=True)
axb.set_yticks(np.arange(len(WINDOWS) + 1) - 0.5, minor=True)
axb.grid(which="minor", color="w", lw=2)
axb.grid(which="major", visible=False)
axb.tick_params(which="minor", length=0)
axb.set_xlabel("Tested timestep")
axb.set_title("Smallest FDR-adjusted cluster p (vs t = 0)", fontsize=9)

for a_, lab in [(ax, "A"), (axb, "B")]:
    a_.text(-0.12 if a_ is ax else -0.3, 1.04, lab, transform=a_.transAxes, fontsize=13, fontweight="bold")
assert (cl[cl.p_fdr < ALPHA].mean_delta_rho > 0).all()
fig.tight_layout()
for ext in ("pdf", "png", "tif"):
    kw = {"pil_kwargs": {"compression": "tiff_lzw"}} if ext == "tif" else {}
    fig.savefig(FIG / f"dynamics_rsa.{ext}", bbox_inches="tight", dpi=300, **kw)
print("saved ->", FIG / "dynamics_rsa.pdf")
