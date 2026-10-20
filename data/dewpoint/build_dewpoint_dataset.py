"""Build the dewpoint dataset (1000 randomly-selected stations, >=512 measurements each).

Input:  {2023,2024,2025}/*.csv   NOAA GSOD year directories in the working directory

Output: dewpoint.h5ad            AnnData with DEWP values + CTRL distribution
        dewpoint_scaled.h5ad     same, with sc.pp.scale applied
        dewpoint-conditions.csv  list of station ids (including CTRL)
        moa                      station-info table (lat/lon/elev, z-scored)

Run from data/weather/dewpoint/:
    python build_dewpoint_dataset.py
"""

import os
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc
from joblib import Parallel, delayed

DATA_DIR = Path(".")
WEATHER_DIR = DATA_DIR.parent      # data/weather/ contains the raw year dirs
YEARS = ["2023", "2024", "2025"]
DEWP_MIN_OBS = 512
N_STATIONS = 1000
SEED = 42

KEEP_COLS = ["STATION", "LATITUDE", "LONGITUDE", "ELEVATION", "DEWP"]


def load_one(path):
    return pd.read_csv(path, usecols=KEEP_COLS)


def main():
    print("Listing CSV paths...")
    paths = [str(WEATHER_DIR / year / fname)
             for year in YEARS for fname in os.listdir(WEATHER_DIR / year)]
    print(f"  {len(paths)} files total")

    print("Loading CSVs in parallel...")
    tbls = Parallel(n_jobs=-1, verbose=1)(delayed(load_one)(p) for p in paths)
    tbl = pd.concat(tbls, ignore_index=True)
    del tbls

    tbl.loc[tbl["DEWP"] > 999, "DEWP"] = np.nan
    tbl = tbl.dropna(subset=["DEWP", "LONGITUDE", "LATITUDE", "ELEVATION"]).reset_index(drop=True)
    print(f"  {len(tbl)} rows after dropping missing values")

    tbl["station"] = "s" + tbl["STATION"].astype(str)

    obs_counts = tbl.groupby("station").size()
    eligible = obs_counts.loc[obs_counts >= DEWP_MIN_OBS].index.tolist()
    print(f"  {len(eligible)} stations with >={DEWP_MIN_OBS} DEWP obs")

    if len(eligible) < N_STATIONS:
        raise RuntimeError(f"only {len(eligible)} eligible stations; cannot sample {N_STATIONS}")

    rng = np.random.default_rng(SEED)
    selected_set = set(rng.choice(np.asarray(eligible), size=N_STATIONS, replace=False).tolist())
    tbl_sub = tbl.loc[tbl["station"].isin(selected_set)].copy()
    print(f"  {len(tbl_sub)} rows for {N_STATIONS} selected stations")

    ctrl_samples = rng.standard_normal(1000) * tbl_sub["DEWP"].std() + tbl_sub["DEWP"].mean()

    adata = ad.AnnData(
        X=tbl_sub["DEWP"].values.reshape(-1, 1).astype(np.float32),
        obs=tbl_sub[["STATION", "LONGITUDE", "LATITUDE", "ELEVATION"]].reset_index(drop=True),
    )
    adata.obs["station"] = "s" + adata.obs["STATION"].astype(str)

    ctrl_obs = pd.DataFrame({
        "STATION": ["-1"] * 1000, "LONGITUDE": [0] * 1000,
        "LATITUDE": [0] * 1000, "ELEVATION": [0] * 1000,
        "station": ["CTRL"] * 1000,
    })
    adata_wctrl = ad.concat(
        [adata, ad.AnnData(X=ctrl_samples.reshape(-1, 1).astype(np.float32), obs=ctrl_obs)],
        axis=0,
    )
    adata_wctrl.obs["STATION"] = adata_wctrl.obs["STATION"].astype(str)
    adata_wctrl.obs = adata_wctrl.obs.reset_index(drop=True)

    adata_wctrl.write(DATA_DIR / "dewpoint.h5ad")
    pd.Series(adata_wctrl.obs["station"].unique()).to_csv(
        DATA_DIR / "dewpoint-conditions.csv", index=False, header=False
    )

    # Moa: lat/lon/elev z-scored per station (features × stations)
    info = (
        adata_wctrl.obs[["station", "LONGITUDE", "LATITUDE", "ELEVATION"]]
        .drop_duplicates()
        .groupby("station", observed=False)
        .mean()
    )
    info_scaled = (info - info.mean(axis=0)) / info.std(axis=0)
    info_scaled.T.to_csv(DATA_DIR / "moa")
    print(f"Wrote moa: {info_scaled.shape[1]} features × {info_scaled.shape[0]} stations")

    adata_scaled = adata_wctrl.copy()
    sc.pp.scale(adata_scaled)
    adata_scaled.write(DATA_DIR / "dewpoint_scaled.h5ad")
    print("Wrote dewpoint.h5ad, dewpoint_scaled.h5ad, conditions, moa")


if __name__ == "__main__":
    main()
