#!/usr/bin/env python
"""Build the K562 dataset's condition embedding from MCF7 scVI cells.

For every drug-dose condition kept in the K562 subset, take the mean of the
MCF7 scVI cell embeddings in that condition. Save the result as a CSV in the
``ModeOfActionEmbedding`` format (rows = latent dims, columns = drug-dose).

Also writes a finalized K562 h5ad with the canonical ``drug-dose`` column
attached (via the hvg drug_df mapping used by select_cond.py) so it can be
loaded by the existing sciplex pipeline.
"""

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd


def attach_drug_dose(adata: ad.AnnData, drug_df: pd.DataFrame) -> ad.AnnData:
    """Add the canonical 'drug-dose' column via the hvg mapping (mirrors select_cond.py)."""
    obs = adata.obs.copy()
    obs["product"] = obs["product_dose"].astype(str).str.split("_").str[0]
    obs["dose"] = obs["dose"].astype(float)
    merged = obs.merge(
        drug_df,
        left_on=["product", "dose"],
        right_on=["product_name", "dose"],
        how="left",
    )
    merged.index = obs.index
    n_null = merged["drug-dose"].isnull().sum()
    print(f"  null drug-dose after merge: {n_null} / {len(merged)}")
    adata = adata.copy()
    adata.obs = merged
    return adata


def main(args):
    print(f"Loading MCF7 scVI embedded data from {args.mcf7_embedded}...")
    mcf7 = ad.read_h5ad(args.mcf7_embedded)
    print(mcf7)

    print(f"Loading K562 scVI embedded data from {args.k562_embedded}...")
    k562 = ad.read_h5ad(args.k562_embedded)
    print(k562)

    print(f"Loading hvg drug_df mapping from {args.hvg_file}...")
    hvg = ad.read_h5ad(args.hvg_file)
    drug_df = hvg.obs[
        ["product_name", "drug_code", "drug", "dose", "drug-dose", "drug_code-dose"]
    ].drop_duplicates()
    drug_df["dose"] = drug_df["dose"].astype(float)
    print(f"  drug_df rows: {len(drug_df)}")

    print("Attaching drug-dose to MCF7...")
    mcf7 = attach_drug_dose(mcf7, drug_df)
    print("Attaching drug-dose to K562...")
    k562 = attach_drug_dose(k562, drug_df)

    # Map control: hvg drug-dose mapping doesn't cover the vehicle/control row,
    # so fill it explicitly with "control-0" to match the existing sciplex-sub
    # convention.
    is_control_mcf7 = mcf7.obs["drug_dose_name"] == args.control_key
    mcf7.obs.loc[is_control_mcf7, "drug-dose"] = args.control_label
    is_control_k562 = k562.obs["drug_dose_name"] == args.control_key
    k562.obs.loc[is_control_k562, "drug-dose"] = args.control_label

    n_null_mcf7 = mcf7.obs["drug-dose"].isnull().sum()
    n_null_k562 = k562.obs["drug-dose"].isnull().sum()
    print(f"MCF7 null drug-dose after control fill: {n_null_mcf7}")
    print(f"K562 null drug-dose after control fill: {n_null_k562}")
    if n_null_mcf7 or n_null_k562:
        # Drop unmapped cells; this should not happen for the eligible
        # conditions but we guard against it explicitly.
        mcf7 = mcf7[~mcf7.obs["drug-dose"].isnull()].copy()
        k562 = k562[~k562.obs["drug-dose"].isnull()].copy()
        print(f"After dropping unmapped: MCF7 {mcf7.n_obs}, K562 {k562.n_obs}")

    # Restrict to drug-dose conditions present in both, so the embedding CSV
    # exactly covers the K562 dataset's conditions.
    k562_conds = set(k562.obs["drug-dose"].unique())
    mcf7_conds = set(mcf7.obs["drug-dose"].unique())
    shared = sorted(k562_conds & mcf7_conds)
    print(
        f"K562 drug-dose conds: {len(k562_conds)}, MCF7: {len(mcf7_conds)}, "
        f"shared: {len(shared)}"
    )
    k562 = k562[k562.obs["drug-dose"].isin(shared)].copy()
    mcf7 = mcf7[mcf7.obs["drug-dose"].isin(shared)].copy()

    # Compute mean MCF7 cell embedding per condition.
    print("Computing mean MCF7 cell embedding per drug-dose...")
    X = np.asarray(mcf7.X)
    df = pd.DataFrame(X, index=mcf7.obs["drug-dose"].values)
    mean_emb = df.groupby(level=0).mean()  # rows = drug-dose, cols = latent dim
    mean_emb = mean_emb.loc[shared]  # consistent ordering
    print(f"Mean embedding shape: {mean_emb.shape}")

    # Format expected by ModeOfActionEmbedding: rows = latent dims, columns =
    # drug-dose names. The loader does pd.read_csv then drops "Unnamed: 0", so
    # we write with index=True and an unnamed index.
    out_df = mean_emb.T  # rows = latent dim index, cols = drug-dose
    out_df.index = range(out_df.shape[0])
    embed_path = os.path.join(args.output_dir, args.embed_name)
    print(f"Writing condition embedding CSV to {embed_path}")
    out_df.to_csv(embed_path)

    # Also save the similarity placeholder (Wasserstein-like) — not used by
    # the embedding loader at inference time, but kept so the file layout
    # matches sciplex-sub. We use pairwise euclidean distance between the mean
    # embeddings as a stand-in.
    from scipy.spatial.distance import squareform, pdist

    sim = pd.DataFrame(
        squareform(pdist(mean_emb.values, metric="euclidean")),
        index=mean_emb.index,
        columns=mean_emb.index,
    )
    sim_path = os.path.join(args.output_dir, "similarity")
    print(f"Writing similarity CSV to {sim_path}")
    sim.to_csv(sim_path)

    # Save final K562 h5ad
    final_path = os.path.join(args.output_dir, "scvi_embedded_sub.h5ad")
    print(f"Writing finalized K562 h5ad to {final_path}")
    k562.write(final_path)

    print(
        f"\nDone. Conditions: {len(shared)} | "
        f"K562 cells: {k562.n_obs} | latent dim: {mean_emb.shape[1]}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--k562_embedded",
        type=str,
        default="scvi_embedded.h5ad",
        help="Path to K562 scVI embedded h5ad (output of scvi_embed.py on K562 subset)",
    )
    parser.add_argument(
        "--mcf7_embedded",
        type=str,
        default="mcf7_embed/scvi_embedded.h5ad",
        help="Path to MCF7 scVI embedded h5ad",
    )
    parser.add_argument(
        "--hvg_file",
        type=str,
        default="../../cmg_datasets/scrna-sciplex3/hvg.h5ad",
        help="HVG h5ad providing the product_name+dose -> drug-dose mapping",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output directory (sciplex-k562/)",
    )
    parser.add_argument(
        "--embed_name",
        type=str,
        default="moa",
        help="CSV filename for the condition embedding (matches ModeOfActionEmbedding 'name' arg)",
    )
    parser.add_argument(
        "--control_key",
        type=str,
        default="control_1.0",
        help="Control drug_dose_name in the raw data",
    )
    parser.add_argument(
        "--control_label",
        type=str,
        default="control-0",
        help="Canonical drug-dose label for control (matches sciplex-sub convention)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
