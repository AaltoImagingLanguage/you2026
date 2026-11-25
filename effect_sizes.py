"""Paired effect sizes used throughout the analyses (Table 2, S4 Table, S1 Table;
imported by 11_human_rt_stats.py, 19_rsa_wholecortex.py, 22_effect_sizes_clusters.py,
Tab2_effect_sizes_predefined_windows.py and 30_dynamics_rsa.py).

d_z = mean(D) / SD(D), the standardized mean of the within-participant
differences D (Lakens, 2013), with an exact 95% CI from the noncentral t
distribution: t = d_z sqrt(n) follows a noncentral t with n - 1 degrees of
freedom and noncentrality delta sqrt(n); the CI bounds are the values of
delta whose upper / lower tail probabilities equal 2.5% (Cumming & Finch, 2001).

Library module; `python effect_sizes.py` runs a self-check of the CI.
"""
import numpy as np
from scipy import optimize, stats


def dz(diff):
    diff = np.asarray(diff, float)
    return diff.mean() / diff.std(ddof=1)


def dz_ci(diff, level=0.95):
    """d_z and its exact CI for the paired differences `diff` (one per participant)."""
    diff = np.asarray(diff, float)
    n = diff.size
    d = dz(diff)
    t_obs, df, a = d * np.sqrt(n), n - 1, (1 - level) / 2

    def bound(p):  # ncp such that P(T >= t_obs | ncp) = p
        f = lambda ncp: stats.nct.sf(t_obs, df, ncp) - p  # noqa: E731
        lo, hi = t_obs - 10 - abs(t_obs), t_obs + 10 + abs(t_obs)
        return optimize.brentq(f, lo, hi)

    return d, bound(a) / np.sqrt(n), bound(1 - a) / np.sqrt(n)


def mean_ci(diff, level=0.95):
    """Mean of `diff` and its t-based CI."""
    diff = np.asarray(diff, float)
    n = diff.size
    half = stats.t.ppf(0.5 + level / 2, n - 1) * diff.std(ddof=1) / np.sqrt(n)
    return diff.mean(), diff.mean() - half, diff.mean() + half


def d_av(off, on):
    """d_av: mean difference / sqrt of the mean of the two variances (Lakens, 2013);
    for reference only."""
    off, on = np.asarray(off, float), np.asarray(on, float)
    return (on - off).mean() / np.sqrt((off.var(ddof=1) + on.var(ddof=1)) / 2)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    x = rng.normal(0.5, 1, 23)
    d, lo, hi = dz_ci(x)
    # check against the coverage definition
    n = x.size
    assert abs(stats.nct.sf(d * np.sqrt(n), n - 1, lo * np.sqrt(n)) - 0.025) < 1e-6
    assert abs(stats.nct.sf(d * np.sqrt(n), n - 1, hi * np.sqrt(n)) - 0.975) < 1e-6
    print(f"d_z = {d:.3f} [{lo:.3f}, {hi:.3f}]")
