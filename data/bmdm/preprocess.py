"""Preprocess BMDM dataset.

Input:  main_bulk.h5ad          OPS bulk feature matrix (conditions × features)
        main_pseq_singlecell.h5ad   scRNA-seq perturbation data

Steps:
  1. Train EBMoCAVI on main_pseq_singlecell.h5ad (see tutorial_ebmocavi_macs.ipynb).
     Extract adata.obsm['salient_rep'] and write to experimental_outputs_filtered.h5ad.
  2. Run this script from data/bmdm/ to filter conditions and derive moa.

Output: bmdm.h5ad            filtered AnnData
        conditions-sub.csv   condition list
        moa                  OPS bulk feature matrix (features × conditions)

Cell filtering:
  - Keep conditions with >= 128 cells.
  - Keep conditions with Wasserstein distance from control > median WD.

Run from data/bmdm/:
    python preprocess.py
"""

import pickle as pkl
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from cmonge.metrics import wasserstein_distance
from tqdm.auto import tqdm

DATA_DIR = Path(".")

SAMPLES_H5AD = DATA_DIR / "experimental_outputs_filtered.h5ad"
INPUT_H5AD = DATA_DIR / "inputs.h5ad"
OUTPUT_H5AD = DATA_DIR / "bmdm.h5ad"
OUTPUT_CONDITIONS = DATA_DIR / "conditions.csv"
OUTPUT_MOA = DATA_DIR / "moa"
WD_CACHE = DATA_DIR / "wasserstein_distances.pkl"

COND_COL = "perturb_DE"
CONTROL_LABEL = "Control"
MIN_CELLS = 128
CTRL_SUBSAMPLE = 512


def build_moa(input_path: Path, conditions: list, out_path: Path) -> None:
    in_features = ad.read_h5ad(input_path)
    conds = [c for c in conditions if c != CONTROL_LABEL and c in in_features.obs_names]
    emb = in_features[conds].to_df().T       # features × conditions
    emb[CONTROL_LABEL] = 0.0
    emb.to_csv(out_path)
    print(f"Wrote moa: {emb.shape[0]} features × {emb.shape[1]} conditions → {out_path.name}")


def main():
    print(f"Loading {INPUT_H5AD.name}")
    adata = ad.read_h5ad(INPUT_H5AD)
    print(f"Input shape: {adata.shape}")

    counts = adata.obs[COND_COL].value_counts()
    eligible = counts[counts >= MIN_CELLS].index
    adata_sub = adata[adata.obs[COND_COL].isin(eligible)].copy()
    print(f"After cell-count filter (>={MIN_CELLS}): {adata_sub.shape}")

    cells_control = adata_sub[adata_sub.obs[COND_COL] == CONTROL_LABEL][:CTRL_SUBSAMPLE]
    wd_dict = {}
    for cond in tqdm(adata_sub.obs[COND_COL].unique(), desc="Computing WD"):
        cells = adata[adata.obs[COND_COL] == cond]
        wd_dict[cond] = wasserstein_distance(cells.X, cells_control.X, epsilon=None)
    with open(WD_CACHE, "wb") as f:
        pkl.dump(wd_dict, f)

    sub_conds = adata_sub.obs[COND_COL].unique()
    q50 = np.quantile([wd_dict[c] for c in sub_conds], 0.5)
    print(f"WD median: {q50:.4f}")
    keep = [c for c in sub_conds if wd_dict.get(c, 0) > q50] + [CONTROL_LABEL]
    adata_sub = adata_sub[adata_sub.obs[COND_COL].isin(keep)].copy()
    print(f"After WD filter: {adata_sub.shape}, {adata_sub.obs[COND_COL].nunique()} conditions")

    adata_sub.write(OUTPUT_H5AD)
    conditions = adata_sub.obs[COND_COL].unique().tolist()
    pd.Series(conditions).to_csv(OUTPUT_CONDITIONS, index=False, header=False)
    print(f"Wrote {OUTPUT_H5AD.name} and {OUTPUT_CONDITIONS.name}")

    build_moa(INPUT_OPS, conditions, OUTPUT_MOA)


if __name__ == "__main__":
    main()
