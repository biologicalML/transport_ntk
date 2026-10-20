"""Step 3: Build a PCA-8 reduced CLINC150 dataset ready for training.

Input:  clinc150_samples_scaled.h5ad  (from step 2)
        clinc150-conditions.csv       (from step 2)
        moa                           (from step 2, copied unchanged)

Output (written to pca8/):
    clinc150pca8_samples.h5ad   16000 x 8, whitened PCA of sample embeddings
    clinc150pca8-conditions.csv one condition per line (copied from step 2)
    moa                         condition embedding (copied unchanged from step 2)
    clinc150pca8_variance_explained.txt

The moa is copied unchanged: the condition-space embedding (SentenceTransformer
label encodings) is independent of the sample-space PCA reduction.

Run from data/Clinc150/:
    python step3_build_pca8.py [--n-components 8]
"""

import argparse
import shutil
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import PCA

DATA_DIR = Path(".")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-components", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    K = args.n_components
    dset_name = f"clinc150pca{K}"
    out_dir = DATA_DIR / f"pca{K}"
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading clinc150_samples_scaled.h5ad")
    adata = ad.read_h5ad(DATA_DIR / "clinc150_samples_scaled.h5ad")
    X = np.asarray(adata.X, dtype=np.float64)
    logger.info(f"X shape: {X.shape}")

    logger.info(f"Fitting whitened PCA to {K} dims")
    pca = PCA(n_components=K, whiten=True, random_state=args.seed)
    Z = pca.fit_transform(X).astype(np.float32)
    evr = pca.explained_variance_ratio_
    logger.info(f"Cumulative variance explained: {evr.sum() * 100:.1f}%")

    new_adata = ad.AnnData(
        X=Z,
        obs=adata.obs.reset_index(drop=True),
        var=pd.DataFrame(index=[f"pc{i+1}" for i in range(K)]),
    )
    new_adata.write_h5ad(out_dir / f"{dset_name}_samples.h5ad")

    shutil.copyfile(DATA_DIR / "clinc150-conditions.csv", out_dir / f"{dset_name}-conditions.csv")
    shutil.copyfile(DATA_DIR / "moa", out_dir / "moa")
    logger.info("Copied conditions and moa")

    summary = out_dir / f"{dset_name}_variance_explained.txt"
    with open(summary, "w") as f:
        f.write(f"Clinc150 PCA-{K} build\n")
        f.write("Sample PCA (whitened)\nPC\tvar_ratio\tcumulative\n")
        for i, e in enumerate(evr, start=1):
            f.write(f"{i}\t{e:.6f}\t{evr[:i].sum():.6f}\n")
    logger.info(f"Done. Output in {out_dir}/")


if __name__ == "__main__":
    main()
