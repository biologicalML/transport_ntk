from typing import Sequence, Any
import argparse
import jax.numpy as jnp
import anndata as ad
import jax
import numpy as np

def commutation_matrix(m, k):
        """K_mk: vec(X^T) = K_mk @ vec(X) for X of shape (m, k)"""
        K = jnp.zeros((m * k, m * k))
        for i in range(m):
            for j in range(k):
                K = K.at[j * m + i, i * k + j].set(1.)
        return K

def adata_conditional_subset(
    adata: ad.AnnData,
    condition_list: Sequence[Any],
    max_cells_per_condition: int = 256,
    validation_split: float = 0.2,
    prng_key=None,
):
    ctrl_cells = adata.obs_names[adata.obs["is_control"]].tolist()
    subset_train_cells = []
    subset_val_cells = []
    for cond in condition_list:
        cond_cells = adata.obs_names[adata.obs["condition"] == cond].values
        max_cells = min(len(cond_cells), max_cells_per_condition)
        n_val_cells = int(max_cells * validation_split)
        n_train_cells = max_cells - n_val_cells
        if len(cond_cells) > max_cells_per_condition:
            key, subkey = jax.random.split(prng_key)
            cond_cells_idx = jax.random.choice(
                key=subkey,
                a=np.arange(len(cond_cells)),
                shape=(max_cells_per_condition,),
                replace=False,
            )
            cond_cells = cond_cells[np.array(cond_cells_idx)]
        subset_train_cells.extend(cond_cells[:n_train_cells])
        subset_val_cells.extend(cond_cells[n_train_cells:])
    validation_cells = ctrl_cells + subset_val_cells
    training_cells = ctrl_cells + subset_train_cells
    return adata[training_cells].copy(), adata[validation_cells].copy()


def kernel_distance(kernel_matrix):
    diag = jnp.diag(kernel_matrix)
    n = kernel_matrix.shape[0]
    kernel_dist = jnp.zeros((n, n))
    diag_tiled = jnp.tile(diag, (n, 1))
    kernel_sqdist = jnp.clip(diag_tiled + diag_tiled.T - 2 * kernel_matrix, 0, None)
    kernel_dist = jnp.sqrt(kernel_sqdist)
    return kernel_dist

def matrix_sqrt(C):
    eigenvalues, Q = jnp.linalg.eigh(
        C
    )  # use eigh for symmetric matrices (faster, more stable)
    eigenvalues = jnp.clip(eigenvalues, 0, None)  # guard against numerical negatives
    return Q @ jnp.diag(jnp.sqrt(eigenvalues)) @ Q.T


def get_parser():
    parser = argparse.ArgumentParser(description="Active Learning Training Script")
    parser.add_argument(
        "--prefix",
        type=str,
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="random",
        # One parser serves all three trainers, so this list is the union of
        # what they accept: `meanvar_ntk`, `quartile_ntk`, `balsa` and
        # `balsa_emd` are quantile-only, `ntk_sizematched` is used by the
        # Monge and flow-matching trainers. A trainer raises on a name it does
        # not implement.
        choices=[
            # No kernel: `random` reuses the condition order across seeds,
            # `random_fixed` reseeds per (run, fold, iteration).
            "random",
            "random_fixed",
            # The transport NTK: tangent kernel of the conditional map ("tNTK").
            "tntk",
            # The hidden-feature Gram ("NTK"). `ntk_sizematched` sizes the
            # mean-prediction network's trunk to the model's; `ntk` keeps the
            # default two-layer trunk. Pair either with
            # --batch-selection-strategy minvar.
            "ntk",
            "ntk_sizematched",
            # Score ablations of the NTK kernel, used by the quantile models.
            "meanvar_ntk",
            "quartile_ntk",
            # BALSA (arXiv:2501.01248): MC-dropout conditional-flow
            # disagreement. "balsa" is the PSD quantile-deviation Gram,
            # "balsa_emd" the diagonal of the paper's EMD score -- pair that
            # one with --batch-selection-strategy uncertainty.
            "balsa",
            "balsa_emd",
        ],
        help="Active learning strategy",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="train/configs/k562.yml",
        help="Path to config file",
    )
    parser.add_argument(
        "--model-config-id", type=str, default="s", help="Model config identifier"
    )
    parser.add_argument(
        "--batch-size", type=int, default=5, help="Batch size for active learning"
    )
    parser.add_argument(
        "--init-batch-size", type=int, default=20, help="Batch size for active learning"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-03, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--split-seed", type=int, default=42, help="Random seed for train/val split"
    )
    parser.add_argument(
        "--init-seed",
        type=int,
        default=42,
        help="Random seed for iter 0 sample selection",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=None,
        help="Max # of AL iterations to run",
    )
    parser.add_argument(
        "--random-al-seed",
        type=int,
        default=42,
        help="Random seed for random sample selections",
    )
    parser.add_argument(
        "--rerun", action="store_true", help="Whether to rerun existing checkpoints"
    )
    parser.add_argument(
        "--reeval",
        action="store_true",
        help="Whether to re-evaluate existing checkpoints",
    )
    parser.add_argument("--data", type=str, default="sciplex-sub")
    parser.add_argument(
        "--batch-selection-strategy",
        type=str,
        # Every run in the paper uses minvar, except BALSA-EMD (uncertainty).
        default="minvar",
        choices=[
            # Greedy k-center on the kernel distance; the default rule.
            "coreset",
            # Highest predictive disagreement; the rule BALSA-EMD is acquired with.
            "uncertainty",
            # Largest-cluster max-distance (LCMD) on the kernel distance.
            "lcmd",
            # k-means++ seeding: sample proportional to squared kernel distance
            # to the nearest selected point.
            "kmeanspp",
            # Total variance minimization: greedily add the candidate that most
            # reduces the summed GP posterior variance over the pool. The rule
            # the NTK and tNTK kernels are compared under.
            "minvar",
        ],
        help="Batch selection strategy for active learning",
    )
    parser.add_argument(
        "--no-transform-kernel", 
        action="store_false", 
        dest="transform_kernel",
        help="Whether to apply GP posterior transformation to kernel matrix before batch selection"
    )

    return parser
