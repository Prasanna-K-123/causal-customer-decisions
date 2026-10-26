"""Pure statistical operations with explicit release-versus-causal boundaries."""
from __future__ import annotations
import hashlib
import numpy as np
import pandas as pd
from scipy.special import expit

FEATURES = [f"f{i}" for i in range(12)]
COLUMNS = FEATURES + ["treatment", "conversion", "visit", "exposure"]
CATEGORICAL = ["f1", "f3", "f4", "f5", "f6", "f8", "f9", "f11"]

def features(frame):
    missing = set(FEATURES) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing permitted feature columns: {sorted(missing)}")
    x = frame.loc[:, FEATURES].to_numpy(dtype=np.float64, copy=True)
    if not np.isfinite(x).all():
        raise ValueError("Features must be finite.")
    x[x == 0] = 0.0
    return x

def validate_frame(frame):
    if list(frame.columns) != COLUMNS:
        raise ValueError("Source column order/schema differs from the declared version.")
    x = features(frame)
    for col in COLUMNS[12:]:
        a = frame[col].to_numpy()
        if not np.isfinite(a).all() or not np.isin(a, [0, 1]).all():
            raise ValueError(f"{col} must contain finite binary numbers before conversion.")
    return x

def fingerprint(x):
    x = np.asarray(x, dtype=np.float64).copy()
    if x.ndim != 2 or x.shape[1] != 12 or not np.isfinite(x).all():
        raise ValueError("Fingerprint requires twelve finite float64 features.")
    x[x == 0] = 0.0
    return pd.util.hash_pandas_object(pd.DataFrame(x, columns=FEATURES),
                                     index=False).to_numpy(dtype=np.uint64)

def mix64(values, salt):
    z = np.asarray(values, dtype=np.uint64) ^ np.uint64(salt)
    with np.errstate(over="ignore"):
        z = z + np.uint64(0x9E3779B97F4A7C15)
        z = (z ^ (z >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        z = (z ^ (z >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        return z ^ (z >> np.uint64(31))

def split_ids(fp, salt=2026102602):
    b = mix64(fp, salt) % np.uint64(10000)
    return np.where(b < 6000, 0, np.where(b < 8000, 1, 2)).astype(np.int8)

def digest_array(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(a.tobytes()).hexdigest()

def assert_group_disjoint(fp, partition):
    frame = pd.DataFrame({"fp": fp, "partition": partition})
    if (frame.groupby("fp", sort=False).partition.nunique() > 1).any():
        raise ValueError("A feature-fingerprint group crosses partitions.")

def _validate_binary(a, name):
    a = np.asarray(a)
    if not np.isfinite(a).all() or not np.isin(a, [0, 1]).all():
        raise ValueError(f"{name} is not binary.")
    return a.astype(float)

def augmented_contrast(y, t, m0, m1, e):
    y, t = _validate_binary(y, "outcome"), _validate_binary(t, "treatment")
    m0, m1, e = (np.asarray(a, dtype=float) for a in (m0, m1, e))
    if not (y.shape == t.shape == m0.shape == m1.shape == e.shape):
        raise ValueError("Augmented-score inputs must have equal shapes.")
    if not all(np.isfinite(a).all() for a in (m0, m1, e)):
        raise ValueError("Nonfinite nuisance prediction.")
    if ((e <= 0) | (e >= 1)).any():
        raise ValueError("Propensities must be strictly inside (0,1).")
    return m1 - m0 + t*(y-m1)/e - (1-t)*(y-m0)/(1-e)

def inverse_weighted_contrast(y, t, e):
    return augmented_contrast(y, t, np.zeros_like(y, dtype=float),
                              np.zeros_like(y, dtype=float), e)

def budget_weights(score, fp, budget, positive_only=False):
    """Group-consistent fractional allocation at the capacity boundary."""
    score, fp = np.asarray(score, float), np.asarray(fp, np.uint64)
    if score.ndim != 1 or score.shape != fp.shape or len(score) == 0:
        raise ValueError("Scores and groups must be nonempty equal vectors.")
    if not np.isfinite(score).all() or not np.isfinite(budget) or not 0 <= budget <= 1:
        raise ValueError("Finite scores and a capacity in [0,1] are required.")
    g, inverse, counts = np.unique(fp, return_inverse=True, return_counts=True)
    order_first = np.unique(fp, return_index=True)[1]
    group_score = score[order_first]
    if not np.allclose(score, group_score[inverse], rtol=0, atol=1e-12):
        raise ValueError("Identical feature groups received different ranking scores.")
    order = np.lexsort((g, -group_score))
    before = np.cumsum(counts[order]) - counts[order]
    allocated = np.clip(budget*len(score)-before, 0, counts[order]) / counts[order]
    if positive_only:
        allocated[group_score[order] <= 0] = 0
    group_weight = np.zeros(len(g))
    group_weight[order] = allocated
    return group_weight[inverse]

def integrated_contribution(score, fp, psi, budgets):
    contributions = np.stack([(budget_weights(score, fp, b)-b)*psi for b in budgets])
    return np.trapezoid(contributions, x=np.asarray(budgets), axis=0)

def clustered_summary(values, fp):
    v, fp = np.asarray(values, float), np.asarray(fp, np.uint64)
    if v.ndim != 1 or v.shape != fp.shape or len(v) < 2 or not np.isfinite(v).all():
        raise ValueError("Cluster summary requires finite equal-length vectors.")
    _, inv = np.unique(fp, return_inverse=True)
    g = int(inv.max()+1)
    if g < 2:
        raise ValueError("At least two fingerprint clusters required.")
    mean = float(v.mean())
    sums = np.bincount(inv, weights=v-mean)
    variance = g/(g-1)*float(sums@sums)/len(v)**2
    se = float(np.sqrt(variance))
    return {"estimate":mean, "se":se, "ci95":[mean-1.959963984540054*se,
            mean+1.959963984540054*se], "rows":len(v), "fingerprint_groups":g}

def multiplier_interval(values, fp, draws=1000, seed=20261026):
    """Gaussian cluster multiplier; group aggregation avoids row-by-draw matrices."""
    v = np.asarray(values, float)
    _, inv = np.unique(fp, return_inverse=True)
    g = int(inv.max()+1)
    if g < 2:
        raise ValueError("At least two clusters required.")
    sums = np.bincount(inv, weights=v-v.mean()) * np.sqrt(g/(g-1))/len(v)
    rng = np.random.default_rng(seed)
    perturbations = []
    for start in range(0, draws, 10):
        count = min(10, draws-start)
        perturbations.extend((rng.standard_normal((count,g)) @ sums).tolist())
    limits = np.quantile(float(v.mean())+np.asarray(perturbations), [.025,.975])
    return {"draws":draws, "seed":seed, "ci95":limits.tolist(),
            "method":"Gaussian fingerprint-cluster multiplier; fixed fits and batch ranks"}

def original_response(selected_response, retention_ratio):
    s = np.asarray(selected_response, float)
    r = np.asarray(retention_ratio, float)
    if not np.isfinite(s).all() or ((s < 0)|(s > 1)).any():
        raise ValueError("Selected response must be a finite probability.")
    if not np.isfinite(r).all() or (r <= 0).any():
        raise ValueError("Retention ratio must be positive and finite.")
    return s/(r*(1-s)+s)

def selected_response(original, retention_ratio):
    p = np.asarray(original, float)
    r = np.asarray(retention_ratio, float)
    if not np.isfinite(p).all() or ((p < 0)|(p > 1)).any():
        raise ValueError("Original response must be a finite probability.")
    if not np.isfinite(r).all() or (r <= 0).any():
        raise ValueError("Retention ratio must be positive and finite.")
    return r*p/(1-p+r*p)

def synthetic_truth(x):
    p0 = expit(-3+.7*x[:,0]-.4*x[:,1])
    tau = .035*np.tanh(2*x[:,0])-.015*x[:,2]
    return p0, np.clip(p0+tau,.001,.95)
