########################################################################
# CSV tables of the condition RDMs, their stability and the condition-label baselines.
#
#   A. Condition RDMs of PCoder 3 (w/o, w/) and the misspelling-severity and
#      lexicality RDMs.
#   B. Stability of the PCoder 3 condition RDM under the stimulus bootstrap
#      of 24_rsa_reliability_bootstrap_cv.py (same seed and draw order, so
#      these are the model RDMs that entered the crossed bootstrap).
#   C. Whole-cortex RSA of PCoder 3 and of the label RDMs with the MEG RDMs,
#      PCoder 3 raw and with severity or lexicality partialled out, and the
#      noise ceiling (from 23_rsa_condition_baselines.py).
#
# Inputs:  ../data/behavior/pcoder_reps_conds_RL23.pkl,
#          ../data/rsa_baselines/rsa_condition_baselines_summary_RL23.csv (23_)
# Outputs: ../data/rsa_baselines/table_{condition_rdms,model_rdm_stability,cortex_rsa}_RL23.csv
#          figures/rsa_baselines/S3_Table_RL23.md
# S6Fig_S8Fig_rsa_baselines.py reads table_condition_rdms_RL23.csv (S6 Fig); the other tables are descriptive and not reported.
#
#   python 25_rsa_baseline_tables.py
########################################################################
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from config import fname
from rsa_utils import (PAIRS, CONDITIONS, condition_of_item, label_rdms,
                          model_item_reps, model_rdms, spearman)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--hps", default="RL23", choices=["RL1", "RL23"])
parser.add_argument("--n-boot", type=int, default=1000)
parser.add_argument("--n-splits", type=int, default=100)
args = parser.parse_args()

OUT_DIR = Path(f"{fname.data_dir}/rsa_baselines")
TABLE_MD = Path(fname.figures_dir) / "rsa_baselines" / f"S3_Table_{args.hps}.md"
SEED = 42  # as in 24_rsa_reliability_bootstrap_cv.py
names = [c.replace("PW", "") for c in CONDITIONS]
pair_names = [f"{names[i]}-{names[j]}" for i, j in PAIRS]

item_reps = model_item_reps(args.hps)
item_reps = {"pcoder3": item_reps["pcoder3"]}
base = model_rdms(item_reps)
candidates = {"PCoder 3 w/o": base["pcoder3 w/o"], "PCoder 3 w/": base["pcoder3 w/"],
              **label_rdms()}

# ---------------------------------------------------------------- A. RDMs
table_a = pd.DataFrame(candidates, index=pair_names).T
table_a.to_csv(OUT_DIR / f"table_condition_rdms_{args.hps}.csv")

# ---------------------------------------------------------------- B. stability
# Reproduce the random stream of 24_rsa_reliability_bootstrap_cv.py: stimulus split-halves first,
# then the bootstrap item draws.
cond = condition_of_item()
n_items = cond.size
rng = np.random.default_rng(SEED)
for k in range(args.n_splits):
    for c in range(4):
        rng.permutation(np.flatnonzero(cond == c))
boot_weights = np.zeros((args.n_boot, 4, n_items), np.float32)
for b in range(args.n_boot):
    for c in range(4):
        drawn = rng.choice(np.flatnonzero(cond == c), size=(cond == c).sum(), replace=True)
        np.add.at(boot_weights[b, c], drawn, 1)
boot = [model_rdms(item_reps, w) for w in boot_weights]
rows = []
for state in ("w/o", "w/"):
    rdms = np.array([m[f"pcoder3 {state}"] for m in boot])
    rho = spearman(rdms, base[f"pcoder3 {state}"][None])[:, 0]
    order_kept = (rdms.argsort(1) == base[f"pcoder3 {state}"].argsort()).all(1)
    rows.append({"model": f"PCoder 3 {state}", "median rho": np.median(rho),
                 "2.5%": np.percentile(rho, 2.5), "97.5%": np.percentile(rho, 97.5),
                 "rank order unchanged (%)": 100 * order_kept.mean()})
table_b = pd.DataFrame(rows)
table_b.to_csv(OUT_DIR / f"table_model_rdm_stability_{args.hps}.csv", index=False)

# ---------------------------------------------------------------- C. cortex RSA
summary = pd.read_csv(OUT_DIR / f"rsa_condition_baselines_summary_{args.hps}.csv")
cortex = summary[summary.region == "cortex"]
rho = cortex[cortex.contrast.str.startswith("rho ")]
table_c = rho.pivot(index="contrast", columns="window_ms", values="mean")
table_c.index = table_c.index.str.replace("rho ", "").str.replace("pcoder3", "PCoder 3")
table_c = table_c.loc[[f"PCoder 3 {s}{c}" for c in ("", " | severity", " | lexicality")
                       for s in ("w/o", "w/")] + ["severity", "lexicality"]]
table_c.index = [n.replace(" | ", ", ") + " partialled out" if "|" in n else n
                 for n in table_c.index]
table_c = table_c.rename(index={"severity": "Severity RDM", "lexicality": "Lexicality RDM"})
nc = cortex[cortex.contrast == "noise ceiling"].set_index("window_ms")
table_c.loc["noise ceiling (lower-upper)"] = [
    f"{nc.loc[w, 'ci_low']:.2f}-{nc.loc[w, 'ci_high']:.2f}" for w in table_c.columns]
table_c = table_c[sorted(table_c.columns, key=lambda w: int(w.split("-")[0]))]
table_c.to_csv(OUT_DIR / f"table_cortex_rsa_{args.hps}.csv")

def to_markdown(df, index=True):
    """Pipe table without the optional tabulate dependency."""
    df = df.reset_index() if index else df
    lines = ["| " + " | ".join(map(str, df.columns)) + " |", "|" + "---|" * df.shape[1]]
    lines += ["| " + " | ".join(map(str, row)) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


with open(TABLE_MD, "w") as f:
    f.write("# Condition RDMs, model-RDM stability and whole-cortex RSA of PCoder 3\n\n")
    f.write("## A. Condition RDMs (correlation distance; severity in replaced letters; "
            "lexicality 1 = real vs misspelled)\n\n")
    f.write(to_markdown(table_a.round(3)) + "\n\n")
    f.write("## B. PCoder 3 RDM under the stimulus bootstrap "
            f"({args.n_boot} resamples; Spearman rho with the full-sample RDM)\n\n")
    f.write(to_markdown(table_b.round(2), index=False) + "\n\n")
    f.write("## C. RSA with the MEG RDMs, mean of 137 parcels (Spearman rho)\n\n")
    f.write(to_markdown(table_c.map(lambda v: v if isinstance(v, str) else f"{v:.2f}")) + "\n")
print(open(TABLE_MD).read())
print("saved ->", TABLE_MD)
