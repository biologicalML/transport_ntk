"""Simulate Van der Pol dataset: 100 conditions sampled from a smooth radial
distribution in (mu, cloud_center_x, cloud_center_y, cloud_sigma) space.

Output: vanderpol.h5ad          AnnData (N×2) with obs["condition"]
        conditions.csv          list of treated condition IDs (cond0..cond99)
        condition_params.csv    per-condition parameter values
        moa                     (mu, cx, cy, sigma) embedding (4 × 101 CSV)

Run from data/vanderpol/:
    python simulate_data.py
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

DATA_DIR = Path(".")

rng = np.random.default_rng(12)

n_per_cond = 1000
dt = 0.01
steps = 300
n_total = 100

params = {
    "mu":    {"center": 2.0, "low": 0.5, "high": 3.0},
    "cx":    {"center": 0.0, "low": -1.5, "high": 1.5},
    "cy":    {"center": 0.0, "low": -1.5, "high": 1.5},
    "sigma": {"center": 0.5, "low": 0.2, "high": 1.0},
}

half_ranges = np.array([max(v["center"] - v["low"], v["high"] - v["center"]) for v in params.values()])
centers = np.array([v["center"] for v in params.values()])
lows = np.array([v["low"] for v in params.values()])
highs = np.array([v["high"] for v in params.values()])

# Radius ~ Exp(2.5) capped at 1: most conditions cluster near center,
# with a heavy tail reaching the parameter edges.
rate = 2.5
radii = np.minimum(rng.exponential(scale=1.0 / rate, size=n_total), 1.0)
directions = rng.normal(size=(n_total, 4))
directions /= np.linalg.norm(directions, axis=1, keepdims=True)

order = np.argsort(radii)
radii = radii[order]
directions = directions[order]

offsets = radii[:, None] * directions * half_ranges[None, :]
sampled = np.clip(centers[None, :] + offsets, lows[None, :], highs[None, :])
all_mus, all_cxs, all_cys, all_sigmas = sampled.T


def integrate(z0, mu):
    z = z0.copy()
    for _ in range(steps):
        x, y = z[:, 0:1], z[:, 1:2]
        z = z + dt * np.concatenate([y, mu * (1.0 - x**2) * y - x], axis=1)
    return z


rows, labels, condition_order = [], [], []

z_ctrl = rng.normal(size=(n_per_cond, 2))
rows.append(z_ctrl)
labels.extend(["control"] * n_per_cond)
condition_order.append("control")

param_records = []
for idx, (mu, cx, cy, sigma) in enumerate(zip(all_mus, all_cxs, all_cys, all_sigmas)):
    name = f"cond{idx}"
    z0 = rng.normal(size=(n_per_cond, 2)) * float(sigma) + np.array([float(cx), float(cy)])
    rows.append(integrate(z0, float(mu)))
    labels.extend([name] * n_per_cond)
    condition_order.append(name)
    param_records.append({"condition": name, "mu": float(mu), "cx": float(cx),
                           "cy": float(cy), "sigma": float(sigma)})

X = np.vstack(rows).astype(np.float32)
obs = pd.DataFrame({"condition": pd.Categorical(labels, categories=condition_order)})
adata = ad.AnnData(X=X, obs=obs)
adata.var_names = ["y1", "y2"]

adata.write_h5ad(DATA_DIR / "vanderpol.h5ad")
pd.Series([f"cond{k}" for k in range(100)]).to_csv(
    DATA_DIR / "conditions.csv", index=False, header=False
)
pd.DataFrame(param_records).to_csv(DATA_DIR / "condition_params.csv", index=False)

# Moa: (mu, cx, cy, sigma) as 4-D embedding; control = zeros
embed_cols = {"control": [0.0, 0.0, 0.0, 0.0]}
for rec in param_records:
    embed_cols[rec["condition"]] = [rec["mu"], rec["cx"], rec["cy"], rec["sigma"]]
pd.DataFrame(embed_cols, index=[0, 1, 2, 3]).to_csv(DATA_DIR / "moa")

print(
    f"Saved vanderpol.h5ad: {adata.n_obs} rows, {len(condition_order)} conditions "
    f"(1 control + {n_total} treated)"
)
