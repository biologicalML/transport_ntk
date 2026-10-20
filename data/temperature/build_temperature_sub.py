"""Build the temperature dataset (top-1000 stations by observation count).

Input:  {2023,2024,2025}/*.csv   NOAA GSOD year directories in the working directory

Output: temperature-sub.h5ad          AnnData with TEMP values + CTRL distribution
        temperature-sub_scaled.h5ad   same, with sc.pp.scale applied
        temperature-sub-conditions.csv list of station ids
        moa                           station-info table (lat/lon/elev + name-PCA8, scaled)

Run from data/weather/:
    python build_temperature_sub.py
"""

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA

DATA_DIR = Path(".")
YEARS = ["2023", "2024", "2025"]
N_STATIONS = 1000
SEED = 42

KEEP_COLS = ["STATION", "NAME", "LATITUDE", "LONGITUDE", "ELEVATION", "TEMP"]


def load_one(path):
    return pd.read_csv(path, usecols=KEEP_COLS, low_memory=False)


def main():
    print("Listing CSV paths...")
    paths = [str(DATA_DIR / year / fname)
             for year in YEARS for fname in os.listdir(DATA_DIR / year)]
    print(f"  {len(paths)} files total")

    print("Loading CSVs in parallel...")
    tbls = Parallel(n_jobs=-1, verbose=1)(delayed(load_one)(p) for p in paths)
    tbl = pd.concat(tbls, ignore_index=True)
    del tbls

    tbl.loc[tbl["TEMP"] > 999, "TEMP"] = np.nan
    tbl = tbl.dropna(subset=["TEMP", "LONGITUDE", "LATITUDE", "ELEVATION"]).reset_index(drop=True)
    print(f"  {len(tbl)} rows after dropping missing values")

    tbl["station"] = "s" + tbl["STATION"].astype(str)

    # Deduplicate: keep stations with exactly one unique name
    names_tbl = tbl[["station", "NAME"]].drop_duplicates()
    dup_stations = names_tbl.groupby("station").size()[lambda x: x > 1].index
    dup_names = names_tbl.groupby("NAME").size()[lambda x: x > 1].index
    names_ok = names_tbl.loc[
        ~names_tbl["station"].isin(dup_stations)
        & ~names_tbl["NAME"].isin(dup_names)
        & names_tbl["NAME"].notna()
    ].copy()
    id_to_name = names_ok.set_index("station")["NAME"].to_dict()

    tbl_sel = tbl.loc[tbl["station"].isin(id_to_name)].copy()
    top1000 = tbl_sel["station"].value_counts().nlargest(N_STATIONS).index
    tbl_sel = tbl_sel.loc[tbl_sel["station"].isin(top1000)].copy()
    print(f"  {len(tbl_sel)} rows for top-{N_STATIONS} stations")

    rng = np.random.default_rng(SEED)
    ctrl_samples = rng.standard_normal(1000) * tbl_sel["TEMP"].std() + tbl_sel["TEMP"].mean()

    adata = ad.AnnData(
        X=tbl_sel["TEMP"].values.reshape(-1, 1).astype(np.float32),
        obs=tbl_sel[["STATION", "LONGITUDE", "LATITUDE", "ELEVATION"]].reset_index(drop=True),
    )
    adata.obs["station"] = "s" + adata.obs["STATION"].astype(str)

    ctrl_obs = pd.DataFrame({
        "STATION": ["-1"] * 1000, "LONGITUDE": [np.nan] * 1000,
        "LATITUDE": [np.nan] * 1000, "ELEVATION": [np.nan] * 1000,
        "station": ["CTRL"] * 1000,
    })
    adata_wctrl = ad.concat(
        [adata, ad.AnnData(X=ctrl_samples.reshape(-1, 1).astype(np.float32), obs=ctrl_obs)],
        axis=0,
    )
    adata_wctrl.obs["STATION"] = adata_wctrl.obs["STATION"].astype(str)
    adata_wctrl.obs = adata_wctrl.obs.reset_index(drop=True)

    adata_wctrl.write(DATA_DIR / "temperature-sub.h5ad")
    pd.Series(adata_wctrl.obs["station"].unique()).to_csv(
        DATA_DIR / "temperature-sub-conditions.csv", index=False, header=False
    )
    scaled = adata_wctrl.copy()
    sc.pp.scale(scaled)
    scaled.write(DATA_DIR / "temperature-sub_scaled.h5ad")
    print("Wrote temperature-sub.h5ad, temperature-sub_scaled.h5ad, conditions")

    # Moa: lat/lon/elev (z-scored) concatenated with station name PCA-8
    info = (
        adata_wctrl.obs[["station", "LONGITUDE", "LATITUDE", "ELEVATION"]]
        .drop_duplicates("station").set_index("station").fillna(0.0)
    )
    info_scaled = (info - info.mean()) / info.std()

    station_ids = [s for s in info.index if s != "CTRL"]
    print("  encoding station names with SentenceTransformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    name_embs = model.encode(
        [id_to_name[s] for s in station_ids], show_progress_bar=True, convert_to_numpy=True
    )
    name_pca = PCA(n_components=8, random_state=SEED).fit_transform(name_embs)

    name_df = pd.DataFrame(name_pca, index=station_ids, columns=[f"name_pc{i}" for i in range(8)])
    name_df.loc["CTRL"] = 0.0

    moa = pd.concat([info_scaled, name_df], axis=1)
    moa.T.to_csv(DATA_DIR / "moa")
    print(f"Wrote moa: {moa.shape[0]} stations × {moa.shape[1]} features")


if __name__ == "__main__":
    main()
