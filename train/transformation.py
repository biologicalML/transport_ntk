''' Kernel transformation to GP posteriors. Adapted from https://github.com/dholzmueller/bmdal_reg/blob/main/bmdal_reg/bmdal/feature_maps.py '''

import jax.numpy as jnp
import numpy as np
from loguru import logger


def robust_cholesky(matrix: np.ndarray) -> np.ndarray:
    """
    Cholesky decomposition with jitter fallback. Runs in numpy float64.
    """
    matrix = np.asarray(matrix, dtype=np.float64)
    eps = 1e-5 * np.trace(matrix) / matrix.shape[-1] + 1e-30
    for i in range(15):
        try:
            return np.linalg.cholesky(matrix)
        except np.linalg.LinAlgError:
            logger.warning('Increasing jitter for Cholesky decomposition')
            matrix = matrix + eps * np.eye(matrix.shape[-1], dtype=np.float64)
            eps *= 2
    raise RuntimeError('Could not Cholesky decompose the matrix')


class GPPosteriorTransformation:
    def __init__(self, prior_kernel, cond_idx, sigma):
        '''
        prior_kernel: (n_conds, n_conds) kernel matrix for the prior GP
        cond_idx: indices of the data GP will be conditioned on
        sigma: noise level for the GP prior

        All internal math runs in numpy float64 — JAX's global dtype stays float32
        so this module is independent of the jax_enable_x64 flag (which breaks
        other JAX-based libraries like ott/sinkhorn).
        '''
        prior_kernel = np.asarray(prior_kernel, dtype=np.float64)
        
        # Symmetrize — upstream float32 matmuls can leave small asymmetry.
        prior_kernel = 0.5 * (prior_kernel + prior_kernel.T)
        # NOTE: do not clip the entries to be non-negative. Every kernel here is a
        # Gram matrix of gradient/feature vectors, so a negative off-diagonal just
        # means two conditions have anti-aligned sensitivities — a legitimate
        # entry. Clipping it to 0 destroys positive semi-definiteness (and with it
        # the GP posterior this class computes). Only the *diagonal* is guaranteed
        # non-negative, and that is what `scale` reads below.
        scale = float(np.max(np.diag(prior_kernel)))
        print("Using scale:", scale)
        self.scale = scale if scale > 0 else 1.0
        self.prior_kernel = prior_kernel / self.scale
        self.cond_idx = np.asarray(cond_idx, dtype=np.int64)
        self.sigma = float(sigma)
        mat = self.prior_kernel[np.ix_(self.cond_idx, self.cond_idx)]
        sigma_normalized = (self.sigma ** 2) / self.scale
        eye = np.eye(mat.shape[0], dtype=np.float64)
        L = robust_cholesky(mat + sigma_normalized * eye)
        L_inv = np.linalg.inv(L)
        self.inv_mat = L_inv.T @ L_inv

    def get_kernel_matrix(self, idxs_1, idxs_2) -> jnp.ndarray:
        idxs_1 = np.asarray(idxs_1, dtype=np.int64)
        idxs_2 = np.asarray(idxs_2, dtype=np.int64)
        prior_mat = self.prior_kernel[np.ix_(idxs_1, idxs_2)]
        mat_1 = self.prior_kernel[np.ix_(idxs_1, self.cond_idx)]
        mat_2 = self.prior_kernel[np.ix_(idxs_2, self.cond_idx)]
        result = prior_mat - mat_1 @ self.inv_mat @ mat_2.T
        # Symmetrize when the output is square over the same index set.
        if idxs_1.shape == idxs_2.shape and np.array_equal(idxs_1, idxs_2):
            result = 0.5 * (result + result.T)
        result = result * self.scale
        return jnp.asarray(result, dtype=jnp.float32)
