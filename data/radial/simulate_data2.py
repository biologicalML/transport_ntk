"""Simulate radial dataset: 100 conditions with (alpha, beta) radial ODE parameters.

50 conditions are tightly clustered near (alpha=1.0, beta=2.5); 50 are spread
across a 10x5 grid of the original ring ranges.

Output: ring2.h5ad              AnnData (N×2) with obs["condition"]
        conditions2.csv         list of treated condition IDs (cond0..cond99)
        condition_params2.csv   per-condition (alpha, beta) values
        moa                     (alpha, beta) embedding (2 × 101 CSV)

Run from data/ring/:
    python simulate_data2.py
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

DATA_DIR = Path(".")

rng = np.random.default_rng(12)

n_per_cond = 1000
dt = 0.01
steps = 100

n_cluster = 50
n_spread = 50

cluster_center_alpha = 1.0
cluster_center_beta = 2.5
cluster_jitter = 0.02
cluster_alphas = cluster_center_alpha + rng.uniform(-cluster_jitter, cluster_jitter, n_cluster)
cluster_betas = cluster_center_beta + rng.uniform(-cluster_jitter, cluster_jitter, n_cluster)

spread_alphas_vals = np.linspace(0.1, 2.0, 10)
spread_betas_vals = np.linspace(0.5, 5.0, 5)
spread_pairs = [(float(a), float(b)) for a in spread_alphas_vals for b in spread_betas_vals]


def integrate(z0, alpha, beta):
    z = z0.copy()
    for _ in range(steps):
        r2 = np.sum(z**2, axis=1, keepdims=True)
        z = z + dt * (alpha - beta * r2) * z
    return z


rows, labels, condition_order = [], [], []

z_ctrl = rng.normal(size=(n_per_cond, 2))
rows.append(z_ctrl)
labels.extend(["control"] * n_per_cond)
condition_order.append("control")

param_records = []
idx = 0

for a, b in zip(cluster_alphas, cluster_betas):
    name = f"cond{idx}"
    z0 = rng.normal(size=(n_per_cond, 2))
    rows.append(integrate(z0, float(a), float(b)))
    labels.extend([name] * n_per_cond)
    condition_order.append(name)
    param_records.append({"condition": name, "alpha": float(a), "beta": float(b)})
    idx += 1

for a, b in spread_pairs:
    name = f"cond{idx}"
    z0 = rng.normal(size=(n_per_cond, 2))
    rows.append(integrate(z0, a, b))
    labels.extend([name] * n_per_cond)
    condition_order.append(name)
    param_records.append({"condition": name, "alpha": a, "beta": b})
    idx += 1

X = np.vstack(rows).astype(np.float32)
obs = pd.DataFrame({"condition": pd.Categorical(labels, categories=condition_order)})
adata = ad.AnnData(X=X, obs=obs)
adata.var_names = ["y1", "y2"]

adata.write_h5ad(DATA_DIR / "ring2.h5ad")
pd.Series([f"cond{k}" for k in range(100)]).to_csv(
    DATA_DIR / "conditions2.csv", index=False, header=False
)
pd.DataFrame(param_records).to_csv(DATA_DIR / "condition_params2.csv", index=False)

# Moa: (alpha, beta) as 2-D embedding; control = (0, 0)
embed_cols = {"control": [0.0, 0.0]}
for rec in param_records:
    embed_cols[rec["condition"]] = [rec["alpha"], rec["beta"]]
pd.DataFrame(embed_cols, index=[0, 1]).to_csv(DATA_DIR / "moa")

print(
    f"Saved ring2.h5ad: {adata.n_obs} rows, {len(condition_order)} conditions "
    f"(1 control + {n_cluster} clustered + {n_spread} spread)"
)
