"""Step 2: Build CLINC150 AnnData and condition embeddings (moa).

Inputs:
    samples_embedded.pkl    from step1_encode_samples.py
    HuggingFace datasets:
        cmaldona/All-Generalization-OOD-CLINC150 (train split) -- sample texts + labels
        DeepPavlov/clinc150 (intents parquet)                   -- canonical label names

Outputs: clinc150_samples.h5ad           AnnData (16000 x 64): 15000 real + 1000 CTRL
         clinc150_samples_scaled.h5ad    same, with X / 0.1 applied
         clinc150-conditions.csv         150 label names + CTRL
         moa                             64-dim SentenceTransformer label embeddings (z-scored)

Moa derivation:
    Each label is encoded with SentenceTransformer "all-MiniLM-L6-v2" (truncated to 64 dims),
    then globally z-scored. Control gets a zero vector.
    Format: rows = embedding dimensions, columns = condition names.

Run from data/Clinc150/:
    python step2_format_labels.py
"""

import pickle as pkl
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(".")
TARGET_DIM = 64
SEED = 42


def main():
    with open(DATA_DIR / "samples_embedded.pkl", "rb") as f:
        emb_data = pkl.load(f)
    sample_embeddings = emb_data["embeddings"].numpy().astype(np.float32)
    print(f"Sample embeddings: {sample_embeddings.shape}")

    samples_df = pd.read_csv(
        "hf://datasets/cmaldona/All-Generalization-OOD-CLINC150/train_clinc150.csv"
    )
    samples_df["labels"] = samples_df["labels"].str.replace(" ", "_")

    label_df = pd.read_parquet(
        "hf://datasets/DeepPavlov/clinc150/intents/intents-00000-of-00001.parquet"
    )
    label_df = label_df.set_index("id")["name"].str.replace(" ", "_")

    # Moa: encode each label with SentenceTransformer, z-score, zero vector for CTRL
    print("Encoding label names with SentenceTransformer...")
    model = SentenceTransformer("all-MiniLM-L6-v2")
    label_embeddings = model.encode(
        label_df.tolist(), batch_size=64, show_progress_bar=True,
        convert_to_numpy=True, truncate_dim=TARGET_DIM,
    )
    label_embeddings_scaled = label_embeddings / label_embeddings.std()
    moa_df = pd.DataFrame(
        np.concatenate([label_embeddings_scaled, np.zeros((1, TARGET_DIM))], axis=0).T,
        columns=label_df.tolist() + ["CTRL"],
    )
    moa_df.to_csv(DATA_DIR / "moa")
    print(f"Wrote moa: {TARGET_DIM} dims × {moa_df.shape[1]} conditions")

    # Build sample AnnData (exclude OOD rows)
    adata = ad.AnnData(
        X=sample_embeddings,
        obs=pd.DataFrame({"label": samples_df["labels"].values}),
    )
    adata = adata[adata.obs["label"] != "ood"].copy()
    print(f"Samples after OOD exclusion: {adata.shape}")

    # CTRL: Gaussian noise matching real embedding statistics
    np.random.seed(SEED)
    ctrl_X = (np.random.randn(1000, adata.shape[1]) * np.std(adata.X, axis=0)
              + np.mean(adata.X, axis=0))
    adata_ctrl = ad.AnnData(
        X=ctrl_X.astype(np.float32),
        obs=pd.DataFrame({"label": ["CTRL"] * 1000}),
    )

    adata_full = ad.concat([adata, adata_ctrl], axis=0)
    adata_full.obs["label"] = adata_full.obs["label"].str.replace(" ", "_")
    adata_full.obs = adata_full.obs.reset_index(drop=True)

    adata_full.write(DATA_DIR / "clinc150_samples.h5ad")
    print(f"Wrote clinc150_samples.h5ad: {adata_full.shape}")

    adata_scaled = adata_full.copy()
    adata_scaled.X = adata_scaled.X / 0.1
    adata_scaled.write(DATA_DIR / "clinc150_samples_scaled.h5ad")
    print("Wrote clinc150_samples_scaled.h5ad")

    all_conditions = label_df.tolist() + ["CTRL"]
    pd.Series(all_conditions).to_csv(DATA_DIR / "clinc150-conditions.csv", index=False, header=False)
    print(f"Wrote clinc150-conditions.csv: {len(all_conditions)} conditions")


if __name__ == "__main__":
    main()
