from __future__ import annotations

from typing import Dict, Any, List, Tuple
import numpy as np
from scipy import stats


def compute_recovery_fraction(
    intact_return: float,
    damaged_return: float,
    repaired_return: float,
    eps: float = 1e-5,
) -> float:
    """Computes R = (R_R - R_D) / (R_I - R_D)."""
    denom = intact_return - damaged_return
    if abs(denom) < eps:
        return 0.0
    return float((repaired_return - damaged_return) / denom)


def paired_seed_analysis(
    evo_returns: List[float],
    control_returns: List[float],
    alpha: float = 0.05,
    n_bootstraps: int = 1000,
    seed: int = 42,
) -> Dict[str, Any]:
    """Computes paired statistical hypothesis tests and bootstrap confidence intervals."""
    arr_evo = np.asarray(evo_returns, dtype=np.float64)
    arr_ctrl = np.asarray(control_returns, dtype=np.float64)

    if len(arr_evo) != len(arr_ctrl):
        raise ValueError("Paired analysis requires equal-length lists from identical seeds")

    n = len(arr_evo)
    diffs = arr_evo - arr_ctrl
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

    # Effect size: Cohen's d for paired samples
    cohens_d = float(mean_diff / (std_diff + 1e-8))

    # Two-sided paired t-test
    if n >= 2 and std_diff > 1e-12:
        t_stat, p_val_t = stats.ttest_rel(arr_evo, arr_ctrl)
        t_stat = float(t_stat)
        p_val_t = float(p_val_t)
    else:
        t_stat, p_val_t = 0.0, 1.0

    # Wilcoxon signed-rank test
    if n >= 5 and np.any(diffs != 0):
        try:
            w_stat, p_val_w = stats.wilcoxon(diffs)
            w_stat = float(w_stat)
            p_val_w = float(p_val_w)
        except Exception:
            w_stat, p_val_w = 0.0, 1.0
    else:
        w_stat, p_val_w = 0.0, 1.0

    # 95% Bootstrap CI of the mean difference
    rng = np.random.default_rng(seed)
    boot_means = []
    for _ in range(n_bootstraps):
        boot_diffs = rng.choice(diffs, size=n, replace=True)
        boot_means.append(np.mean(boot_diffs))

    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))

    return {
        "n_seeds": n,
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "cohens_d": cohens_d,
        "t_statistic": t_stat,
        "p_value_ttest": p_val_t,
        "wilcoxon_stat": w_stat,
        "p_value_wilcoxon": p_val_w,
        "ci_95_lower": ci_lower,
        "ci_95_upper": ci_upper,
        "significant_at_005": bool(p_val_t < 0.05 and ci_lower > 0),
    }
