from __future__ import annotations

from typing import Dict, Any, List
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
    """Analyze paired returns with two-sided tests and a percentile bootstrap CI.

    The two legacy ``ci_95_*`` fields are retained for callers that expect them;
    ``ci_lower`` and ``ci_upper`` follow the requested ``alpha``.
    """
    arr_evo = np.asarray(evo_returns, dtype=np.float64)
    arr_ctrl = np.asarray(control_returns, dtype=np.float64)

    if arr_evo.ndim != 1 or arr_ctrl.ndim != 1:
        raise ValueError("Paired returns must be one-dimensional sequences")
    if arr_evo.size != arr_ctrl.size:
        raise ValueError("Paired analysis requires equal-length lists from identical seeds")
    if arr_evo.size == 0:
        raise ValueError("Paired analysis requires at least one seed")
    if not np.all(np.isfinite(arr_evo)) or not np.all(np.isfinite(arr_ctrl)):
        raise ValueError("Paired returns must contain only finite values")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    if (
        isinstance(n_bootstraps, bool)
        or not isinstance(n_bootstraps, (int, np.integer))
        or n_bootstraps < 1
    ):
        raise ValueError("n_bootstraps must be a positive integer")

    n = arr_evo.size
    diffs = arr_evo - arr_ctrl
    mean_diff = float(np.mean(diffs))
    std_diff = float(np.std(diffs, ddof=1)) if n > 1 else 0.0

    # Effect size: Cohen's d for paired samples
    cohens_d = float(mean_diff / std_diff) if std_diff > 1e-12 else None

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

    # Resample the paired differences, not the two arms independently.
    rng = np.random.default_rng(seed)
    boot_means = rng.choice(diffs, size=(n_bootstraps, n), replace=True).mean(axis=1)

    ci_lower = float(np.percentile(boot_means, 100.0 * (alpha / 2.0)))
    ci_upper = float(np.percentile(boot_means, 100.0 * (1.0 - alpha / 2.0)))
    ci_95_lower = float(np.percentile(boot_means, 2.5))
    ci_95_upper = float(np.percentile(boot_means, 97.5))

    return {
        "n_seeds": n,
        "alpha": float(alpha),
        "mean_diff": mean_diff,
        "std_diff": std_diff,
        "cohens_d": cohens_d,
        "t_statistic": t_stat,
        "p_value_ttest": p_val_t,
        "wilcoxon_stat": w_stat,
        "p_value_wilcoxon": p_val_w,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "ci_95_lower": ci_95_lower,
        "ci_95_upper": ci_95_upper,
        "significant_at_alpha": bool(p_val_t < alpha),
        "significant_at_005": bool(p_val_t < 0.05),
    }


def holm_bonferroni(p_values: Dict[str, float]) -> Dict[str, float]:
    """Return Holm-adjusted p-values while preserving input key order."""
    if not p_values:
        return {}

    normalized = {}
    for name, value in p_values.items():
        p_value = float(value)
        if not np.isfinite(p_value) or not 0.0 <= p_value <= 1.0:
            raise ValueError(f"p-value for {name!r} must be finite and between 0 and 1")
        normalized[name] = p_value

    ordered = sorted(normalized.items(), key=lambda item: item[1])
    adjusted = {}
    running_max = 0.0
    count = len(ordered)
    for index, (name, p_value) in enumerate(ordered):
        running_max = max(running_max, min(1.0, (count - index) * p_value))
        adjusted[name] = running_max

    return {name: adjusted[name] for name in normalized}
