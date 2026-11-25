########################################################################
# Human catch-trial reaction-time analysis (S1 Table).
#
# Trial-level input: ../data/behavior/rt_trials.csv (included in the data release). With
# --log_dir, the table is first rebuilt from the Presentation logs of the MEG experiment
# (not shared; the same parsing produced ../data/behavior/performance.csv): for each
# participant and condition, the last 15 sentence-completion catch trials (sentence row
# "<cond>_<0|1>-...", outcome hit / incorrect / miss in "Stim Type"; RT = TTime of the
# following Response row, i.e. from sentence onset).
#
# Reported:
#   - trials per condition, correct / incorrect / missed responses, and the
#     number of usable (correct) RTs per participant and condition
#   - the main analysis: participant means of correct-trial RTs,
#     one-way repeated-measures ANOVA, F(3, 66)
#   - sphericity: Mauchly's test, Greenhouse-Geisser and Huynh-Feldt epsilon,
#     corrected degrees of freedom and p
#   - effect sizes: partial eta squared and generalized eta squared
#   - additional checks (not reported): participant medians; means after excluding RTs more than
#     2.5 SD from the participant's condition mean; a trial-level linear mixed
#     model of log RT (condition fixed, random participant intercept), tested
#     with a likelihood-ratio test against the intercept-only model
#   - the RW vs RL1 accuracy t test with d_z and its CI
# Outputs: ../data/behavior/rt_stats.csv (printed as well; input of S1Tab_human_rt.py).
#
#   python 11_human_rt_stats.py                       # from rt_trials.csv
#   python 11_human_rt_stats.py --log_dir LOG_DIR     # rebuild rt_trials.csv from the logs
########################################################################
import argparse
import re

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

from config import fname, subjects
from effect_sizes import dz_ci

parser = argparse.ArgumentParser()
parser.add_argument("--log_dir", default=None, help="folder of the Presentation logs (private)")
args = parser.parse_args()
TRIALS = f"{fname.data_dir}/behavior/rt_trials.csv"
CONDITIONS = ["RW", "RL1", "RL2", "RL3"]
N_CATCH = 15
TRIM_SD = 2.5


def read_subject(subject):
    log = pd.read_csv(f"{args.log_dir}/{subject}-flexwordrec.log", skiprows=3, sep="\t",
                      encoding="utf-8")[1:]
    log = log.iloc[:log[log["Code"] == "pause"].index[-1]]
    rows = []
    for ci, cond in enumerate(CONDITIONS, start=1):
        catch = log[log["Code"].str.match(re.compile(rf"({ci}_[0,1]-)"), na=False)]
        catch = catch.iloc[-N_CATCH:] if len(catch) > N_CATCH else catch
        for trial, idx in enumerate(catch.index):
            outcome = log["Stim Type"][idx]
            nxt = log.loc[idx + 1] if idx + 1 in log.index else None
            responded = nxt is not None and nxt["Event Type"] == "Response"
            rows.append({"subject": subject, "condition": cond, "trial": trial,
                         "outcome": outcome,
                         "rt_ms": nxt["TTime"] / 10 if responded else np.nan})
    return rows


def rm_anova(wide):
    """One-way repeated-measures ANOVA on (subject, condition) values, with
    Mauchly's test, GG/HF corrections, partial and generalized eta squared."""
    x = np.asarray(wide, float)
    n, k = x.shape
    gm = x.mean()
    ss_cond = n * ((x.mean(0) - gm) ** 2).sum()
    ss_subj = k * ((x.mean(1) - gm) ** 2).sum()
    ss_err = ((x - gm) ** 2).sum() - ss_cond - ss_subj
    df1, df2 = k - 1, (k - 1) * (n - 1)
    F = (ss_cond / df1) / (ss_err / df2)
    # sphericity on orthonormal contrasts
    C = np.linalg.qr(np.vstack([np.ones(k), np.eye(k)[:-1] - 1 / k]).T)[0][:, 1:]
    S = C.T @ np.cov(x.T) @ C
    p = k - 1
    W = np.linalg.det(S) / (np.trace(S) / p) ** p
    chi2 = -(n - 1 - (2 * p ** 2 + p + 2) / (6 * p)) * np.log(W)
    df_m = p * (p + 1) / 2 - 1
    eig = np.linalg.eigvalsh(S)
    gg = eig.sum() ** 2 / (p * (eig ** 2).sum())
    hf = min(1.0, (n * p * gg - 2) / (p * (n - 1 - p * gg)))
    return {"n": n, "F": F, "df1": df1, "df2": df2, "p": stats.f.sf(F, df1, df2),
            "mauchly_W": W, "mauchly_chi2": chi2, "mauchly_df": df_m,
            "mauchly_p": stats.chi2.sf(chi2, df_m),
            "eps_GG": gg, "df1_GG": df1 * gg, "df2_GG": df2 * gg,
            "p_GG": stats.f.sf(F, df1 * gg, df2 * gg),
            "eps_HF": hf, "p_HF": stats.f.sf(F, df1 * hf, df2 * hf),
            "eta_p2": ss_cond / (ss_cond + ss_err),
            "eta_G2": ss_cond / (ss_cond + ss_subj + ss_err)}


if args.log_dir:
    trials = pd.DataFrame([r for s in subjects for r in read_subject(s)])
    trials.to_csv(TRIALS, index=False)
trials = pd.read_csv(TRIALS)
correct = trials[(trials.outcome == "hit") & trials.rt_ms.notna()].copy()

# ---- trial counts ------------------------------------------------------------
counts = trials.groupby(["subject", "condition"]).outcome.value_counts().unstack(fill_value=0)
counts["trials"] = counts.sum(1)
summary = counts.groupby("condition").agg(["mean", "min", "max"]).reindex(CONDITIONS)
print("Catch trials per participant and condition (mean, min, max):")
print(summary.round(2).to_string())
print(f"hits without a response row: {((trials.outcome == 'hit') & trials.rt_ms.isna()).sum()}")
print("correct-trial RTs (ms): min %.0f, max %.0f" % (correct.rt_ms.min(), correct.rt_ms.max()))

# ---- the original analysis, reproduced ------------------------------------------
means = correct.groupby(["subject", "condition"]).rt_ms.mean().unstack()[CONDITIONS]
perf = pd.read_csv(fname.human_behav, index_col=0)
orig = perf.pivot(index="Subject", columns="type", values="rt")[CONDITIONS].loc[means.index]
assert np.allclose(means.values, orig.values, atol=1), "does not reproduce performance.csv"
acc = (trials.assign(hit=trials.outcome == "hit").groupby(["subject", "condition"]).hit.mean()
       .unstack()[CONDITIONS] * 100)
orig_acc = perf.pivot(index="Subject", columns="type", values="accuracy")[CONDITIONS].loc[acc.index]
assert np.allclose(acc.values, orig_acc.values, atol=0.1), "accuracy does not reproduce"
print("reproduces performance.csv (RT and accuracy)")

results = []
for name, wide in [
    ("mean RT, correct trials (as in the manuscript)", means),
    ("median RT, correct trials", correct.groupby(["subject", "condition"]).rt_ms.median().unstack()[CONDITIONS]),
]:
    results.append({"analysis": name, **rm_anova(wide), **{f"mean_{c}": wide[c].mean() for c in CONDITIONS}})

# trimmed: exclude RTs more than TRIM_SD SD from the participant's condition mean
g = correct.groupby(["subject", "condition"]).rt_ms
z = (correct.rt_ms - g.transform("mean")) / g.transform("std")
trimmed = correct[(z.abs() <= TRIM_SD) | z.isna()]
wide = trimmed.groupby(["subject", "condition"]).rt_ms.mean().unstack()[CONDITIONS]
results.append({"analysis": f"mean RT, correct trials, |z| <= {TRIM_SD}", **rm_anova(wide),
                **{f"mean_{c}": wide[c].mean() for c in CONDITIONS},
                "n_excluded": int(len(correct) - len(trimmed))})
res = pd.DataFrame(results)

# trial-level mixed model of log RT
correct["log_rt"] = np.log(correct.rt_ms)
full = smf.mixedlm("log_rt ~ C(condition, Treatment('RW'))", correct, groups="subject").fit(reml=False)
null = smf.mixedlm("log_rt ~ 1", correct, groups="subject").fit(reml=False)
lr = 2 * (full.llf - null.llf)
lmm = {"analysis": "trial-level LMM, log RT ~ condition + (1|participant)",
       "n_trials": len(correct), "LR_chi2": lr, "LR_df": 3, "p": stats.chi2.sf(lr, 3),
       **{f"beta_{c}": full.params[f"C(condition, Treatment('RW'))[T.{c}]"] for c in CONDITIONS[1:]}}
res = pd.concat([res, pd.DataFrame([lmm])], ignore_index=True)

# accuracy RW vs RL1
diff = (acc["RL1"] - acc["RW"]).values
t, p = stats.ttest_rel(acc["RL1"], acc["RW"])
d, lo, hi = dz_ci(diff)
res = pd.concat([res, pd.DataFrame([{"analysis": "accuracy RL1 - RW, paired t test", "n": len(diff),
                                     "t": t, "df": len(diff) - 1, "p": p, "d_z": d,
                                     "d_z_ci95_low": lo, "d_z_ci95_high": hi,
                                     "mean_RW": acc["RW"].mean(), "mean_RL1": acc["RL1"].mean()}])],
                ignore_index=True)
res.to_csv(f"{fname.data_dir}/behavior/rt_stats.csv", index=False)
with pd.option_context("display.width", 250, "display.max_columns", 40):
    print(res.T.to_string())
print("saved ->", f"{fname.data_dir}/behavior/rt_stats.csv")
