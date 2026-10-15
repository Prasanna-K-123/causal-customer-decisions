"""Known-propensity policy values, paired contrasts and budget decisions."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def augmented_scores(y: np.ndarray, arm: np.ndarray, mu: np.ndarray, propensity: np.ndarray | None = None) -> np.ndarray:
    y, arm, mu = np.asarray(y, float), np.asarray(arm, int), np.asarray(mu, float)
    if mu.shape != (len(y), 3) or arm.shape != y.shape or not np.isin(arm, [0, 1, 2]).all():
        raise ValueError("Expected aligned outcomes, three arms and n-by-3 predictions.")
    p = np.full_like(mu, 1 / 3) if propensity is None else np.broadcast_to(np.asarray(propensity, float), mu.shape)
    if not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Propensities must be positive and sum to one for every customer.")
    if not np.isfinite(y).all() or not np.isfinite(mu).all():
        raise ValueError("Nonfinite outcome or prediction.")
    result = mu.copy()
    row = np.arange(len(y))
    result[row, arm] += (y - mu[row, arm]) / p[row, arm]
    return result


def policy_scores(gamma: np.ndarray, actions: np.ndarray, margin: float = 1.0, cost: float = 0.0) -> np.ndarray:
    actions = np.asarray(actions, int)
    if len(actions) != len(gamma) or not np.isin(actions, [0, 1, 2]).all():
        raise ValueError("Invalid action vector.")
    if not 0 <= margin <= 1 or cost < 0:
        raise ValueError("Margin must lie in [0,1] and contact cost must be nonnegative.")
    return margin * gamma[np.arange(len(actions)), actions] - cost * (actions != 0)


def summarize(values: np.ndarray, confidence: float = 0.95) -> dict:
    values = np.asarray(values, float)
    if len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Need at least two finite scores.")
    estimate = float(values.mean())
    se = float(values.std(ddof=1) / np.sqrt(len(values)))
    critical = float(norm.ppf((1 + confidence) / 2))
    return {"estimate": estimate, "standard_error": se, "lower": estimate - critical * se, "upper": estimate + critical * se, "n": len(values)}


def paired_bootstrap(values: np.ndarray, seed: int, repetitions: int = 2000) -> dict:
    """Paired scores preserve covariance between policies on the same customer."""
    values = np.asarray(values, float)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for start in range(0, repetitions, 50):
        count = min(50, repetitions - start)
        ix = rng.integers(0, len(values), size=(count, len(values)))
        means[start:start + count] = values[ix].mean(axis=1)
    low, high = np.quantile(means, [0.025, 0.975])
    return {"lower": float(low), "upper": float(high), "repetitions": repetitions, "seed": seed, "type": "paired customer percentile bootstrap; fixed trained policies"}


def allocate(predicted_means: np.ndarray, margin: float, cost: float, capacity: float, tie_key: np.ndarray) -> np.ndarray:
    """Assign the best positive-value treatment, with a hard batch contact cap."""
    if not 0 <= capacity <= 1 or not 0 <= margin <= 1 or cost < 0:
        raise ValueError("Invalid scenario assumptions.")
    means = np.asarray(predicted_means, float)
    if means.ndim != 2 or means.shape[1] != 3 or not np.isfinite(means).all():
        raise ValueError("Expected finite n-by-3 predicted means or effects.")
    if len(tie_key) != len(means):
        raise ValueError("Tie keys must align with customers.")
    gains = margin * (means[:, 1:] - means[:, [0]]) - cost
    best = np.argmax(gains, axis=1) + 1
    value = gains[np.arange(len(means)), best - 1]
    eligible = np.flatnonzero(value > 0)
    order = eligible[np.lexsort((np.asarray(tie_key)[eligible], -value[eligible]))]
    selected = order[:int(np.floor(capacity * len(means)))]
    actions = np.zeros(len(means), dtype=int)
    actions[selected] = best[selected]
    return actions


def uniform_policy(arm: int, n: int, capacity: float, priority: np.ndarray) -> np.ndarray:
    result = np.zeros(n, dtype=int)
    if arm:
        selected = np.argsort(priority, kind="stable")[:int(np.floor(capacity * n))]
        result[selected] = arm
    return result
