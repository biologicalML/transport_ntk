#!/usr/bin/env python
"""Prepare K562 and matched MCF7 subsets of sciplex3 for scVI embedding.

The K562 subset is restricted to drug-dose conditions that
  (a) have at least ``--min_cells_k562`` K562 cells, and
  (b) are also present in MCF7 (so a mean MCF7 cell embedding can be
      computed for use as the K562 condition embedding).

The MCF7 subset is restricted to the same set of conditions.
"""

import argparse
import os

import anndata as ad
import numpy as np
import pandas as pd


def main(args):
    print(f"Loading {args.input_file}...")
    adata = ad.read_h5ad(args.input_file)
    print(adata)

    k562_mask = adata.obs["cell_type"] == "K562"
    mcf7_mask = adata.obs["cell_type"] == "MCF7"
    print(f"K562 cells: {k562_mask.sum()}, MCF7 cells: {mcf7_mask.sum()}")

    k562_counts = adata.obs.loc[k562_mask, "drug_dose_name"].value_counts()
    mcf7_conds = set(adata.obs.loc[mcf7_mask, "drug_dose_name"].unique())

    eligible = k562_counts[k562_counts >= args.min_cells_k562]
    selected = sorted([c for c in eligible.index if c in mcf7_conds])
    print(
        f"K562 conds with >= {args.min_cells_k562} cells: {len(eligible)}; "
        f"of which present in MCF7: {len(selected)}"
    )

    # Always keep the control condition if it exists.
    if args.control_key and args.control_key not in selected:
        if args.control_key in mcf7_conds and args.control_key in k562_counts.index:
            selected.append(args.control_key)
            print(f"Force-added control '{args.control_key}'")
        else:
            print(f"WARNING: control '{args.control_key}' missing from K562 or MCF7")

    selected_set = set(selected)
    print(f"Final selected condition count: {len(selected)}")

    cond_file = os.path.join(args.output_dir, "selected_conditions.csv")
    pd.Series(sorted(selected_set), name="drug_dose_name").to_csv(
        cond_file, index=False
    )
    print(f"Wrote selected conditions to {cond_file}")

    cond_mask = adata.obs["drug_dose_name"].isin(selected_set)

    k562_sub = adata[k562_mask & cond_mask].copy()
    mcf7_sub = adata[mcf7_mask & cond_mask].copy()
    print(f"K562 subset: {k562_sub}")
    print(f"MCF7 subset: {mcf7_sub}")

    k562_path = os.path.join(args.output_dir, "k562_input.h5ad")
    mcf7_path = os.path.join(args.output_dir, "mcf7_embed", "mcf7_input.h5ad")
    os.makedirs(os.path.dirname(mcf7_path), exist_ok=True)

    print(f"Writing {k562_path}")
    k562_sub.write(k562_path)
    print(f"Writing {mcf7_path}")
    mcf7_sub.write(mcf7_path)

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input_file",
        type=str,
        default="../sciplex3_biolord.h5ad",
        help="Input h5ad (full sciplex3 biolord file)",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=".",
        help="Output dir (the sciplex-k562/ directory)",
    )
    parser.add_argument(
        "--min_cells_k562",
        type=int,
        default=256,
        help="Minimum K562 cells per drug-dose condition",
    )
    parser.add_argument(
        "--control_key",
        type=str,
        default="control_1.0",
        help="Control drug_dose_name to always keep",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    main(args)
