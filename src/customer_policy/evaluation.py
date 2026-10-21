"""Known-propensity policy values, paired contrasts and budget decisions."""
from __future__ import annotations
import numpy as np
from scipy.stats import norm


def augmented_scores(y: np.ndarray, arm: np.ndarray, mu: np.ndarray, propensity: np.ndarray | None = None) -> np.ndarray:
    y = _finite_array(y, "Outcomes")
    mu = _finite_array(mu, "Predictions")
    if y.ndim != 1:
        raise ValueError("Outcomes must be a one-dimensional vector.")
    arm = _arm_vector(arm, len(y), "Treatment")
    if mu.shape != (len(y), 3) or arm.shape != y.shape or not np.isin(arm, [0, 1, 2]).all():
        raise ValueError("Expected aligned outcomes, three arms and n-by-3 predictions.")
    p = (np.full_like(mu, 1 / 3) if propensity is None
         else np.broadcast_to(_finite_array(propensity, "Propensities"), mu.shape))
    if not np.isfinite(p).all() or (p <= 0).any() or not np.allclose(p.sum(axis=1), 1):
        raise ValueError("Propensities must be positive and sum to one for every customer.")
    if not np.isfinite(y).all() or not np.isfinite(mu).all():
        raise ValueError("Nonfinite outcome or prediction.")
    result = mu.copy()
    row = np.arange(len(y))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result[row, arm] += (y - mu[row, arm]) / p[row, arm]
    if not np.isfinite(result).all():
        raise ValueError("Augmented-score arithmetic exceeds finite precision.")
    return result


def policy_scores(gamma: np.ndarray, actions: np.ndarray, margin: float = 1.0, cost: float = 0.0) -> np.ndarray:
    gamma = _finite_array(gamma, "Augmented scores")
    if gamma.ndim != 2 or gamma.shape[1] != 3:
        raise ValueError("Expected an n-by-3 augmented-score matrix.")
    actions = _arm_vector(actions, len(gamma), "Policy action")
    margin = _scenario_scalar(margin, "Margin", upper=1.0)
    cost = _scenario_scalar(cost, "Contact cost")
    with np.errstate(over="ignore", invalid="ignore"):
        result = margin * gamma[np.arange(len(actions)), actions] - cost * (actions != 0)
    if not np.isfinite(result).all():
        raise ValueError("Policy-value arithmetic exceeds finite precision.")
    return result


def summarize(values: np.ndarray, confidence: float = 0.95) -> dict:
    values = _score_vector(values)
    confidence = _scenario_scalar(confidence, "Confidence", upper=1.0)
    if not 0 < confidence < 1:
        raise ValueError("Confidence must lie strictly between zero and one.")
    with np.errstate(over="ignore", invalid="ignore"):
        estimate = float(values.mean())
        se = float(values.std(ddof=1) / np.sqrt(len(values)))
    if not np.isfinite([estimate, se]).all():
        raise ValueError("Summary arithmetic exceeds finite precision.")
    critical = float(norm.ppf((1 + confidence) / 2))
    lower, upper = estimate - critical * se, estimate + critical * se
    if not np.isfinite([critical, lower, upper]).all():
        raise ValueError("Confidence interval exceeds finite numerical precision.")
    return {"estimate": estimate, "standard_error": se, "lower": lower, "upper": upper, "n": len(values)}


def paired_bootstrap(values: np.ndarray, seed: int, repetitions: int = 2000) -> dict:
    """Paired scores preserve covariance between policies on the same customer."""
    values = _score_vector(values)
    repetitions = _integer_count(repetitions, "Bootstrap repetitions", minimum=2)
    rng = np.random.default_rng(seed)
    means = np.empty(repetitions)
    for start in range(0, repetitions, 50):
        count = min(50, repetitions - start)
        ix = rng.integers(0, len(values), size=(count, len(values)))
        with np.errstate(over="ignore", invalid="ignore"):
            means[start:start + count] = values[ix].mean(axis=1)
    if not np.isfinite(means).all():
        raise ValueError("Bootstrap arithmetic exceeds finite precision.")
    low, high = np.quantile(means, [0.025, 0.975])
    return {"lower": float(low), "upper": float(high), "repetitions": repetitions, "seed": seed, "type": "paired customer percentile bootstrap; fixed trained policies"}


def allocate(predicted_means: np.ndarray, margin: float, cost: float, capacity: float, tie_key: np.ndarray) -> np.ndarray:
    """Assign the best positive-value treatment, with a hard batch contact cap."""
    capacity = _scenario_scalar(capacity, "Capacity", upper=1.0)
    margin = _scenario_scalar(margin, "Margin", upper=1.0)
    cost = _scenario_scalar(cost, "Contact cost")
    means = _finite_array(predicted_means, "Predictions")
    if means.ndim != 2 or means.shape[1] != 3 or not np.isfinite(means).all():
        raise ValueError("Expected finite n-by-3 predicted means or effects.")
    tie_key = _tie_vector(tie_key, len(means))
    with np.errstate(over="ignore", invalid="ignore"):
        gains = margin * (means[:, 1:] - means[:, [0]]) - cost
    if not np.isfinite(gains).all():
        raise ValueError("Allocation arithmetic exceeds finite precision.")
    best = np.argmax(gains, axis=1) + 1
    value = gains[np.arange(len(means)), best - 1]
    eligible = np.flatnonzero(value > 0)
    order = eligible[np.lexsort((np.asarray(tie_key)[eligible], -value[eligible]))]
    selected = order[:int(np.floor(capacity * len(means)))]
    actions = np.zeros(len(means), dtype=int)
    actions[selected] = best[selected]
    return actions


def uniform_policy(arm: int, n: int, capacity: float, priority: np.ndarray) -> np.ndarray:
    n = _integer_count(n, "Batch size", minimum=0)
    arm_input = np.asarray(arm)
    if arm_input.ndim != 0:
        raise ValueError("Uniform treatment must be a scalar arm label.")
    arm = int(_arm_vector(arm_input.reshape(1), 1, "Uniform treatment")[0])
    capacity = _scenario_scalar(capacity, "Capacity", upper=1.0)
    priority = _finite_array(priority, "Priorities")
    if priority.shape != (n,):
        raise ValueError("Priorities must be one-dimensional and match the batch.")
    result = np.zeros(n, dtype=int)
    if arm:
        selected = np.argsort(priority, kind="stable")[:int(np.floor(capacity * n))]
        result[selected] = arm
    return result


def _finite_array(value, name: str) -> np.ndarray:
    """Convert numerical values without silently discarding complex components."""
    try:
        if np.iscomplexobj(value):
            raise ValueError(f"{name} must be real and finite.")
        result = np.asarray(value, dtype=float)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real finite numbers.") from error
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain real finite numbers.")
    return result


def _arm_vector(value, n: int, name: str) -> np.ndarray:
    """Validate the original labels before converting to an integer index."""
    labels = np.asarray(value)
    if (labels.shape != (n,) or labels.dtype.kind not in "iuf"
            or not np.isfinite(labels).all()
            or not np.isin(labels, [0, 1, 2]).all()):
        raise ValueError(f"{name} labels must be an aligned vector of 0, 1 or 2.")
    return labels.astype(np.intp, copy=False)


def _scenario_scalar(value, name: str, upper: float | None = None) -> float:
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a finite real scalar.")
    result = float(raw)
    if not np.isfinite(result) or result < 0 or (upper is not None and result > upper):
        raise ValueError(f"{name} is outside its finite admissible range.")
    return result


def _score_vector(value) -> np.ndarray:
    result = _finite_array(value, "Scores")
    if result.ndim != 1 or len(result) < 2:
        raise ValueError("Need a one-dimensional vector of at least two finite scores.")
    return result


def _integer_count(value, name: str, minimum: int) -> int:
    if (isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer)) or value < minimum):
        raise ValueError(f"{name} must be an integer of at least {minimum}.")
    return int(value)


def _tie_vector(value, n: int) -> np.ndarray:
    keys = np.asarray(value)
    if keys.shape != (n,) or keys.dtype.kind not in "iufUS":
        raise ValueError("Tie keys must be an aligned real-number or string vector.")
    if keys.dtype.kind in "iuf" and not np.isfinite(keys).all():
        raise ValueError("Numerical tie keys must be finite.")
    if len(np.unique(keys)) != n:
        raise ValueError("Tie keys must be unique to resolve equal gains independently of row order.")
    return keys
