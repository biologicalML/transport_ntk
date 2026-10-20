from typing import Literal
from loguru import logger
import random
import jax
import jax.random
import jax.numpy as jnp
import numpy as np
from transformation import GPPosteriorTransformation
from utils import kernel_distance


# ---------------------------------------------------------------------------
# Kernel-free (random) acquisition
# ---------------------------------------------------------------------------
# Strategies that pick without a model kernel, so they need neither a kernel
# matrix nor the condition embedding. All four trainers import this list rather
# than testing `strategy != "random"`, so adding a kernel-free strategy does not
# need a sweep of every guard.
NO_KERNEL_STRATEGIES = ("random", "random_fixed")

# Strategies that may own the shared iter-0 checkpoint, preferred first. Iter 0
# is strategy-independent (its batch is always drawn with strategy="random",
# seeded by init_seed), so every arm loads one shared iter 0 instead of
# retraining it -- that keeps GPU non-determinism from drifting the starting
# point across strategies. New runs source it from `random_fixed`, the corrected
# uniform baseline that is now the random arm of the grid; the legacy `random`
# arm stays a fallback donor so prefixes whose only iter 0 on disk is named
# ..._random_..._iter0 keep resolving. A run whose own strategy is listed here
# computes its own iter 0 rather than symlinking to the other donor.
ITER0_DONOR_STRATEGIES = ("random_fixed", "random")


def acquisition_seed_for(init_seed: int, fold_idx: int, al_iter: int) -> int:
    """Reproducible per-(run, fold, iteration) seed for the `random_fixed` arm.

    SeedSequence mixes the three indices properly, so consecutive seeds do not
    produce correlated draws -- the failure mode that makes the legacy `random`
    baseline pick the same narrow window of the condition pool in every run.
    """
    return int(
        np.random.SeedSequence([int(init_seed), int(fold_idx), int(al_iter)])
        .generate_state(1)[0]
    )


def sample_random_batch(
    unseen_conditions,
    batch_size,
    strategy="random",
    random_seed=42,
    acquisition_seed=None,
):
    """Draw a batch without a kernel. Shared by all four AL trainers.

    strategy="random" -- LEGACY, kept bit-identical for comparability with
        existing runs. The AL loop does not pass `random_seed` for al_iter > 0,
        so this reseeds to the default (42) on every iteration of every run.
        `random.sample` then picks a fixed *index* into `unseen_conditions`,
        whose order is the fixed per-fold pool order, and the run's initial batch
        only shifts that index by the number of its removals falling before it.
        Across 50 (fold, seed) runs this yields ~12-17 distinct conditions, all
        inside a ~6-wide window of the pool order -- and because that window is a
        contiguous block of a fixed list, it inherits whatever bias the region
        has.

    strategy="random_fixed" -- the corrected uniform baseline. `acquisition_seed`
        is derived per (init_seed, fold, al_iter) by the caller via
        `acquisition_seed_for`, so every run and iteration draws independently
        while staying reproducible.
    """
    k = min(batch_size, len(unseen_conditions))
    if strategy == "random_fixed":
        rng = np.random.default_rng(
            acquisition_seed if acquisition_seed is not None else random_seed
        )
        return [
            unseen_conditions[i]
            for i in rng.choice(len(unseen_conditions), size=k, replace=False)
        ]
    random.seed(random_seed)
    return random.sample(unseen_conditions, k=k)


def gp_posterior_kernel(kernel, cond_index, sigma=1.0):
    kernel_transformer = GPPosteriorTransformation(kernel, cond_index, sigma)
    return kernel_transformer.get_kernel_matrix(
        idxs_1=list(range(kernel.shape[0])), idxs_2=list(range(kernel.shape[0]))
    )



def get_gp_posterior_diag_update(kernel, cond_index, candidate_update_index, sigma=1.0):
    '''
    Calculate the diagonal update in the GP posterior kernel after conditioning on cond_index plus one of candidate_update_index.

    Runs in numpy float64, like `GPPosteriorTransformation` and
    the greedy rules. The gradient kernels here are dominated by a near
    rank-one component (on real runs the top eigenvalue holds ~93% of the trace),
    so `s_star` is a difference of nearly equal large numbers: conditioning on ~10
    conditions takes the posterior variance from O(1e3) down to O(1). In float32
    that cancellation cost ~0.2% relative error in the returned scores — enough to
    reorder near-ties among candidates.

    Note `sigma` enters as the noise *variance* (K_SS + sigma I), matching
    the greedy rules; `GPPosteriorTransformation` takes the same argument as
    a standard deviation and squares it. Both agree at the default sigma=1.0.
    '''
    kernel = np.asarray(kernel, dtype=np.float64)
    cond_index = np.asarray(cond_index, dtype=np.int64).reshape(-1)
    candidate_update_index = np.asarray(candidate_update_index, dtype=np.int64).reshape(-1)

    A = kernel[np.ix_(cond_index, cond_index)] + sigma * np.eye(len(cond_index)) # K_{SS} + sigma I
    bn = kernel[np.ix_(cond_index, candidate_update_index)] # (n_cond, n_candidates), column j is k_Sj
    # Solve rather than invert: same result, better conditioned.
    A_inv_bn = np.linalg.solve(A, bn) if len(cond_index) else bn
    b_s = kernel[cond_index, :] # (n_cond, n_target)

    b_star = kernel[:, candidate_update_index] # each row is for each target index
    bn_star = np.diag(kernel)[candidate_update_index] # (n_candidates,)

    s_star = bn_star + sigma - np.sum(bn * A_inv_bn, axis=0) # (n_candidates,)

    A_inv_bn_bs_allcands = b_s.T @ A_inv_bn # (n_target, n_candidates)

    quad_term = A_inv_bn_bs_allcands**2 / s_star
    cross_term = -2 * b_star / s_star * A_inv_bn_bs_allcands
    res_term = b_star ** 2 / s_star

    diag_update = - (quad_term + cross_term + res_term)
    assert diag_update.shape == (kernel.shape[0], len(candidate_update_index))
    return jnp.asarray(diag_update.sum(axis=0).squeeze())
    


def select_batch_coreset(
    kernel, seen_index, batch_size, transform_kernel=True, random_seed=42, sigma=1.0
):
    """
    seen_idx: condition indices the model is trained on so far
    sigma: GP posterior noise / regularization strength (K_SS + sigma I)
    """
    selected_index = []
    if len(seen_index) > 0:
        if transform_kernel:
            kernel_ = gp_posterior_kernel(kernel, seen_index, sigma)
        else:
            kernel_ = kernel
        kernel_dist = kernel_distance(kernel_)
        kernel_dist = kernel_dist.at[seen_index, :].set(-jnp.inf)
    for i in range(batch_size):
        seen_and_selected_index = jnp.array(seen_index + selected_index)
        # if len(seen_and_selected_index) == 0:
        #     max_idx = jnp.argmax(jnp.diag(kernel_)).item()
        #     kernel_dist = kernel_dist.at[max_idx, :].set(-jnp.inf)
        #     selected_index.append(max_idx)
        # else:
        if transform_kernel:
            kernel_ = gp_posterior_kernel(kernel, seen_and_selected_index, sigma)
        else:
            kernel_ = kernel
        kernel_dist = kernel_distance(kernel_)
        kernel_dist = kernel_dist.at[seen_index, :].set(-jnp.inf)
        dist_to_seen = kernel_dist[:, seen_and_selected_index].min(
            axis=1
        )  # Min distance to any of the seen samples
        max_min_dist_idx = jnp.argmax(dist_to_seen).item()
        kernel_dist = kernel_dist.at[max_min_dist_idx, :].set(-jnp.inf)
        selected_index.append(max_min_dist_idx)
    return selected_index


def select_batch_uncertainty(
    kernel, seen_index, batch_size, random_seed=42, transform_kernel=True, sigma=1.0
):
    """
    seen_idx: condition indices the model is trained on so far
    sigma: GP posterior noise / regularization strength (K_SS + sigma I)
    """
    if transform_kernel:
        kernel = gp_posterior_kernel(kernel, seen_index, sigma)
    if len(seen_index) > 0:
        kernel = kernel.at[seen_index, :].set(-jnp.inf)
    max_idx = jnp.argsort(jnp.diag(kernel))[::-1][:batch_size]
    logger.debug(f"Kernel diagonal values: {jnp.diag(kernel)}")
    logger.debug(f"Uncertainty selected indices: {max_idx}")
    return max_idx










def select_batch_lcmd(
    kernel, seen_index, batch_size, transform_kernel=True, random_seed=42, sigma=1.0
):
    selected_idxes = []
    for i in range(batch_size):
        seen_and_selected_index = seen_index + selected_idxes
        if transform_kernel:
            kernel_ = gp_posterior_kernel(kernel, seen_and_selected_index, sigma)
        else:
            kernel_ = kernel
        kernel_dist = kernel_distance(kernel_)
        # Assign each sample to its closest seen sample
        assigned_seen = jnp.argmin(kernel_dist[:, seen_and_selected_index], axis=1)
        # get cluster size with is the sum of squared distance to the assigned seen samples
        cluster_size = jnp.zeros((len(seen_and_selected_index),))
        dist_to_cluster_centers = jnp.zeros((kernel_dist.shape[0],))

        for i in range(len(seen_and_selected_index)):
            if (assigned_seen == i).sum() == 0:
                continue
            cluster_size = cluster_size.at[i].set(
                jnp.sum(
                    kernel_dist[assigned_seen == i, seen_and_selected_index[i]] ** 2
                )
            )
            dist_to_cluster_centers = dist_to_cluster_centers.at[
                assigned_seen == i
            ].set(kernel_dist[assigned_seen == i, seen_and_selected_index[i]])

        # Max distance sample is selected, prevent seen index from being selected again
        for idx in seen_and_selected_index:
            dist_to_cluster_centers = dist_to_cluster_centers.at[idx].set(-jnp.inf)
        # Select samples from the largest clusters
        largest_cluster_idx = jnp.argmax(cluster_size)
        candidate_idxes = jnp.where(assigned_seen == largest_cluster_idx)[0]
        next_sample = candidate_idxes[
            jnp.argmax(dist_to_cluster_centers[candidate_idxes])
        ].item()
        selected_idxes.append(next_sample)

    return selected_idxes


def select_batch_kmeanspp(
    kernel, seen_index, batch_size, transform_kernel=False, random_seed=42, sigma=1.0
):
    eps = 1e-8
    selected_idxes = []
    key = jax.random.PRNGKey(random_seed)
    for i in range(batch_size):
        seen_and_selected_index = seen_index + selected_idxes
        if transform_kernel:
            kernel_ = gp_posterior_kernel(kernel, seen_and_selected_index, sigma)
        else:
            kernel_ = kernel
        kernel_dist = kernel_distance(kernel_)
        min_dist = jnp.min((kernel_dist**2)[:, seen_and_selected_index], axis=1)
        p = (min_dist + eps) / (min_dist.sum() + eps * len(min_dist))
        for idx in seen_and_selected_index:
            p = p.at[idx].set(0)
        key, subkey = jax.random.split(key)
        selected_idx = jax.random.choice(subkey, a=kernel_dist.shape[0], p=p).item()
        selected_idxes.append(selected_idx)
    return selected_idxes


def select_batch_minvar(
    kernel, seen_index, batch_size, sigma=1.0, **kwargs
):
    selected_idxes = []
    for _ in range(batch_size):
        candidate_idx = jnp.array(list(set(range(kernel.shape[0])) - set(seen_index+selected_idxes)))
        selected_idx = candidate_idx[jnp.argmin(get_gp_posterior_diag_update(kernel, seen_index + selected_idxes, candidate_idx, sigma)).item()].item()
        selected_idxes.append(selected_idx)
    return selected_idxes


def select_batch(
    trained_conditions,
    all_conditions,
    batch_size,
    kernel_matrix,
    batch_selection_strategy: Literal[
        "coreset", "uncertainty", "lcmd", "kmeanspp", "minvar"
    ] = "uncertainty",
    transform_kernel=True,
    random_seed=42,
    sigma=1.0,
):
    trained_conditions_indices = [all_conditions.index(c) for c in trained_conditions]
    select_batch = {
        "coreset": select_batch_coreset,
        "uncertainty": select_batch_uncertainty,
        "lcmd": select_batch_lcmd,
        "kmeanspp": select_batch_kmeanspp,
        "minvar": select_batch_minvar,
    }[batch_selection_strategy]

    idx = select_batch(
        kernel_matrix,
        seen_index=trained_conditions_indices,
        batch_size=batch_size,
        random_seed=random_seed,
        transform_kernel=transform_kernel,
        sigma=sigma,
    )

    logger.info(
        f"Selected indices: {idx}, seen index:{trained_conditions_indices} using strategy: {batch_selection_strategy} (sigma={sigma})"
    )
    return [all_conditions[i] for i in idx]
