########################################################################
# Table 2: effect sizes in the predefined 200-ms windows, independent of cluster
# selection. PCoder 3, with feedback (t = 25) minus without (t = 0), condition-wise
# RSA, one value per participant averaged over the window.
#   mean Delta rho [95% t-based CI] per 200-ms window of the whole-cortex test
#   d_z [95% CI, noncentral t]
#   averaged over the whole cortex (all 137 parcels); the same estimates for the post hoc
#   follow-up regions vOT and ST are in S4 Table A (S4Tab_effect_sizes_vOT_ST.py)
# Input:  figures/rsa/rsa_whole_brain_pcoderall_RL23_predefined_windows.csv
#         (19_rsa_wholecortex.py)
# Output: figures/effect_sizes/predefined_windows_whole_cortex.csv (source of Table 2)
#
#   python figures_tables/Tab2_effect_sizes_predefined_windows.py
########################################################################
import sys; from pathlib import Path; sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repository root: shared modules
from pathlib import Path

import pandas as pd

from config import fname

OUT = Path(f"{fname.figures_dir}/effect_sizes")
OUT.mkdir(exist_ok=True)
pre = pd.read_csv(f"{fname.figures_dir}/rsa/rsa_whole_brain_pcoderall_RL23_predefined_windows.csv")
R = pre[(pre.pcoder == 3) & (pre.region == "cortex")][
    ["window_ms", "mean_delta_rho", "ci95_low", "ci95_high", "d_z", "d_z_ci95_low", "d_z_ci95_high"]]
R.to_csv(OUT / "predefined_windows_whole_cortex.csv", index=False)

for r in R.itertuples():
    print(f"{r.window_ms} ms: delta_rho = {r.mean_delta_rho:.3f} [{r.ci95_low:.3f}, {r.ci95_high:.3f}], "
          f"d_z = {r.d_z:.2f} [{r.d_z_ci95_low:.2f}, {r.d_z_ci95_high:.2f}]")
print("saved ->", OUT / "predefined_windows_whole_cortex.csv")
