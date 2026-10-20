"""Simulate 1D histogram dataset: 1000 conditions, each a mixture of 3 Gaussians.
Constraint: weighted mean sum_i(mu_i * prop_i) = 0 for every condition.
Condition variable: (mu1, std1, prop1, mu2, std2, prop2, mu3, std3, prop3) - 9 dims.
Control: standard normal N(0,1).

Output: sim_mixture_a5.h5ad   AnnData (N×1) with obs["condition"]
        condition_info         9×1000 condition parameter CSV
        moa                    symlink → condition_info
        sim-conditions.csv     list of non-control condition IDs

Run from data/sim_mixture_a5/:
    python simulate.py
"""

from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd

DATA_DIR = Path(".")

rng = np.random.default_rng(42)

N_CONDITIONS = 1000
N_SAMPLES_PER_COND = 600
N_CTRL = 1000
N_COMPONENTS = 3
WEIGHTED_MEAN = 0.0

mus = rng.uniform(-5, 5, size=(N_CONDITIONS, N_COMPONENTS))
stds = rng.uniform(1.0, 1.5, size=(N_CONDITIONS, N_COMPONENTS))
props = rng.dirichlet([1 / 4] * N_COMPONENTS, size=N_CONDITIONS)

# Enforce weighted-mean constraint: shift mus so sum_i mu_i * prop_i == WEIGHTED_MEAN
current_wmean = np.sum(mus * props, axis=1, keepdims=True)
mus = mus - current_wmean + WEIGHTED_MEAN
assert np.allclose(np.sum(mus * props, axis=1), WEIGHTED_MEAN), "Weighted mean constraint violated"

# Sort components by mu per condition for identifiability
order = np.argsort(mus, axis=1)
mus = np.take_along_axis(mus, order, axis=1)
stds = np.take_along_axis(stds, order, axis=1)
props = np.take_along_axis(props, order, axis=1)

cond_ids = [f"cond{i:04d}" for i in range(N_CONDITIONS)]

# Moa: 9-dim condition parameters (features × conditions)
condition_info = pd.DataFrame(
    {
        cid: [
            mus[i, 0], stds[i, 0], props[i, 0],
            mus[i, 1], stds[i, 1], props[i, 1],
            mus[i, 2], stds[i, 2], props[i, 2],
        ]
        for i, cid in enumerate(cond_ids)
    },
    index=["mu1", "std1", "prop1", "mu2", "std2", "prop2", "mu3", "std3", "prop3"],
)
condition_info.to_csv(DATA_DIR / "condition_info")

moa_link = DATA_DIR / "moa"
if moa_link.is_symlink():
    moa_link.unlink()
moa_link.symlink_to("condition_info")


def sample_gmm(mus_k, stds_k, props_k, n, rng):
    counts = rng.multinomial(n, props_k)
    parts = [rng.normal(mu, std, cnt) for mu, std, cnt in zip(mus_k, stds_k, counts) if cnt > 0]
    return np.concatenate(parts)


ctrl_samples = rng.normal(0, 1, N_CTRL)
all_X, all_cond = [], []

all_X.append(ctrl_samples.reshape(-1, 1))
all_cond.extend(["CTRL"] * N_CTRL)

for i, cid in enumerate(cond_ids):
    samples = sample_gmm(mus[i], stds[i], props[i], N_SAMPLES_PER_COND, rng)
    all_X.append(samples.reshape(-1, 1))
    all_cond.extend([cid] * N_SAMPLES_PER_COND)

X = np.vstack(all_X).astype(np.float32)
obs = pd.DataFrame({"condition": all_cond})
obs.index = obs.index.astype(str)

adata = ad.AnnData(X=X, obs=obs)
adata.write(DATA_DIR / "sim_mixture_a5.h5ad")
pd.Series(cond_ids).to_csv(DATA_DIR / "sim-conditions.csv", index=False, header=False)
print(f"Wrote sim_mixture_a5.h5ad: {adata.shape}, {adata.obs['condition'].nunique()} conditions")
