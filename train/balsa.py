"""BALSA: Bayesian Active Learning by distribution disagreement (EMD variant).

Reference
---------
Werner & Schmidt-Thieme, *Bayesian Active Learning By Distribution Disagreement*
(arXiv:2501.01248), reference implementation at
https://github.com/wernerth94/Bayesian-Active-Learning-By-Distribution-Disagreement

What BALSA does in the original paper
-------------------------------------
A conditional density model ``p(y | x)`` is fit as a *dropout MLP encoder* followed
by a *neural spline flow head* (``classifiers/normalizing_flow.py:EncNSF``): the
encoder maps the input to a context vector, and the flow is conditioned on that
context. MC-dropout is then used as a cheap Bayesian posterior over parameters:
``k`` forward passes with active dropout give ``k`` different predictive
distributions ``p_hat_i(y | x)`` for the *same* input. The acquisition score is how
much these disagree, measured (BALSA-EMD, ``agents/balsa.py:BALSA_EMD``) as the sum
of Earth-Mover distances between samples of consecutive dropout trials:

    BALSA_EMD(x) = sum_{i=1}^{k-1} EMD(y'_i, y'_{i+1}),   y'_i ~ p_hat_i(. | x)

and the top-``b`` scoring pool points are queried.

How it maps onto ACPL
---------------------
Here a "point" is a *condition*: the model input is the condition embedding
``z_c = cond_arr_dict[c]`` and the label is the whole cell population of that
condition, ``Y_c = target_dict[c]`` of shape ``(n_cells, data_dim)``. So the
conditional density model is ``p(y | z_c)`` — exactly the object the flow head
models — fit by maximum likelihood on the cells of the currently labelled
(``train_conditions``) set, and BALSA scores every condition in
``all_conditions``. Unlike the paper's 1-D regression targets, ``y`` is
multivariate, so the EMD of the paper's score is generalised in two ways
(``emd_mode``): ``"marginal"`` (per-dimension 1-D Wasserstein-1, summed over
dimensions — reduces exactly to the paper's score when ``data_dim == 1``) or
``"sinkhorn"`` (entropic OT between the two ``d``-dimensional point clouds).

Kernel matrices, not just scores
--------------------------------
ACPL selects *batches* from a kernel matrix (``acpl.selection.select_batch``), so
``get_balsa_kernel_matrix`` offers two ways to express the disagreement:

``kernel_mode="quantile"`` (default)
    Per-condition features are the *quantile deviations* of the dropout ensemble:
    for trial ``i``, ``Q_i(c)`` are the per-dimension sample quantiles of
    ``y'_i ~ p_hat_i(.|z_c)``, and ``phi_c = concat_i (Q_i(c) - Qbar(c))``, scaled so
    that ``K_cc = mean_i W2^2(p_hat_i(.|z_c), pbar(.|z_c))`` in the 1-D-marginal
    (quantile-space) sense. This is the L2/W2 analogue of BALSA-EMD's L1/W1
    disagreement, and being a Gram matrix ``K = Phi Phi^T`` it is PSD, so *every*
    batch-selection strategy (``minvar``, ``coreset``, ``lcmd``, ``kmeanspp``,
    ``typiclust2``, ``uncertainty``) is meaningful on it: the diagonal is the BALSA
    uncertainty, the off-diagonals say which conditions the ensemble disagrees
    about *in the same way*.

``kernel_mode="emd"``
    ``K = diag(BALSA_EMD(c))`` from the literal paper score. Combined with
    ``batch_selection_strategy="uncertainty"`` this reproduces the paper's
    top-``b`` acquisition (the GP-posterior transform in ``select_batch`` leaves the
    diagonal of unlabelled conditions untouched for a diagonal kernel).

Both modes return the trained flow plus per-condition scores/features in the
``aux`` dict, so the two can be compared without refitting.

Everything is flax/optax/JAX to match the rest of ``acpl`` (``ntk.py``); the
conditional rational-quadratic-spline flow is implemented here because ``flowjax``
is not installed in the ``cmonge`` environment.
"""

from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict, List, Optional, Sequence, Tuple

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import optax
from loguru import logger
from tqdm.auto import tqdm

# ---------------------------------------------------------------------------
# Rational-quadratic spline transform (Durkan et al., 2019, "Neural Spline Flows")
# ---------------------------------------------------------------------------

_MIN_BIN_SIZE = 1e-3  # minimum relative bin width/height
_MIN_DERIVATIVE = 1e-3


def _spline_params(raw: jnp.ndarray, knots: int, interval: float):
    """Turn unconstrained conditioner outputs into monotone spline knots.

    Args:
        raw: ``(..., 3 * knots - 1)`` unconstrained parameters.
        knots: number of spline bins.
        interval: spline is defined on ``[-interval, interval]`` (identity outside).

    Returns:
        ``(cum_widths, cum_heights, derivatives)`` with shapes ``(..., knots + 1)``.
    """
    k = knots
    w_raw, h_raw, d_raw = raw[..., :k], raw[..., k : 2 * k], raw[..., 2 * k :]

    def _cumulative(logits):
        frac = jax.nn.softmax(logits, axis=-1)
        frac = _MIN_BIN_SIZE + (1.0 - _MIN_BIN_SIZE * k) * frac
        sizes = 2.0 * interval * frac
        cum = jnp.cumsum(sizes, axis=-1)
        cum = jnp.concatenate([jnp.zeros_like(cum[..., :1]), cum], axis=-1) - interval
        # kill accumulated float error at the right edge
        return cum.at[..., -1].set(interval)

    cum_widths = _cumulative(w_raw)
    cum_heights = _cumulative(h_raw)

    # interior derivatives are free & positive, boundary derivatives are 1 so the
    # spline glues continuously onto the identity tails
    derivatives = jax.nn.softplus(d_raw) + _MIN_DERIVATIVE  # (..., knots - 1)
    ones = jnp.ones_like(derivatives[..., :1])
    derivatives = jnp.concatenate([ones, derivatives, ones], axis=-1)
    return cum_widths, cum_heights, derivatives


def _gather(arr: jnp.ndarray, idx: jnp.ndarray) -> jnp.ndarray:
    return jnp.take_along_axis(arr, idx[..., None], axis=-1)[..., 0]


def rational_quadratic_spline(
    x: jnp.ndarray,
    raw: jnp.ndarray,
    knots: int = 8,
    interval: float = 4.0,
    inverse: bool = False,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Monotone rational-quadratic spline, elementwise over ``x``.

    Args:
        x: values to transform, shape ``(...)``.
        raw: ``(..., 3 * knots - 1)`` unconstrained spline parameters.
        knots: number of bins.
        interval: transform is the identity outside ``[-interval, interval]``.
        inverse: apply the inverse map instead of the forward map.

    Returns:
        ``(y, log_det)`` where ``log_det`` is the log absolute derivative of the
        applied direction (so it can simply be summed into a log-density).
    """
    cum_w, cum_h, derivs = _spline_params(raw, knots, interval)

    inside = (x > -interval) & (x < interval)
    x_safe = jnp.clip(x, -interval + 1e-6, interval - 1e-6)

    edges = cum_h if inverse else cum_w
    idx = jnp.sum(edges[..., :-1] <= x_safe[..., None], axis=-1) - 1
    idx = jnp.clip(idx, 0, knots - 1)

    x_k, x_k1 = _gather(cum_w, idx), _gather(cum_w, idx + 1)
    y_k, y_k1 = _gather(cum_h, idx), _gather(cum_h, idx + 1)
    d_k, d_k1 = _gather(derivs, idx), _gather(derivs, idx + 1)

    width = x_k1 - x_k
    height = y_k1 - y_k
    s = height / width  # average slope in the bin
    d_sum = d_k1 + d_k - 2.0 * s

    if inverse:
        dy = x_safe - y_k
        a = height * (s - d_k) + dy * d_sum
        b = height * d_k - dy * d_sum
        c = -s * dy
        disc = jnp.clip(b**2 - 4.0 * a * c, 0.0, None)
        xi = jnp.clip(2.0 * c / (-b - jnp.sqrt(disc)), 0.0, 1.0)
        out = x_k + xi * width
    else:
        xi = (x_safe - x_k) / width
        numerator = height * (s * xi**2 + d_k * xi * (1.0 - xi))
        denominator = s + d_sum * xi * (1.0 - xi)
        out = y_k + numerator / denominator

    # forward derivative dy/dx evaluated at xi (Durkan et al., eq. 5)
    denominator = s + d_sum * xi * (1.0 - xi)
    dydx_num = s**2 * (d_k1 * xi**2 + 2.0 * s * xi * (1.0 - xi) + d_k * (1.0 - xi) ** 2)
    log_det = jnp.log(jnp.clip(dydx_num, 1e-30, None)) - 2.0 * jnp.log(
        jnp.clip(denominator, 1e-30, None)
    )
    if inverse:
        log_det = -log_det

    out = jnp.where(inside, out, x)
    log_det = jnp.where(inside, log_det, 0.0)
    return out, log_det


# ---------------------------------------------------------------------------
# Model: dropout MLP encoder + conditional coupling spline flow
# ---------------------------------------------------------------------------


class DropoutEncoder(nn.Module):
    """MLP encoder with dropout after every hidden layer (paper's Bayesian trunk).

    Mirrors ``EncNSF._encode`` of the reference implementation (``Linear`` +
    ``Dropout`` per hidden layer) with the ReLU activations the paper describes.
    Dropout is what makes ``k`` forward passes ``k`` posterior samples.
    """

    hidden_dims: Sequence[int]
    dropout_rate: float = 0.05

    @nn.compact
    def __call__(self, x: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        for dim in self.hidden_dims:
            x = nn.Dense(dim)(x)
            x = nn.relu(x)
            x = nn.Dropout(rate=self.dropout_rate, deterministic=deterministic)(x)
        return x


class ConditionerMLP(nn.Module):
    """Plain MLP mapping (masked input, context) -> raw spline parameters."""

    width: int
    depth: int
    out_dim: int

    @nn.compact
    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for _ in range(self.depth):
            x = nn.relu(nn.Dense(self.width)(x))
        # zero-init the last layer -> the flow starts at (near) identity, which
        # makes early training of the spline stack stable
        return nn.Dense(self.out_dim, kernel_init=nn.initializers.zeros)(x)


class ConditionalSplineFlow(nn.Module):
    """Conditional RQS coupling flow with an MC-dropout context encoder.

    ``context()`` runs the (stochastic) encoder; ``log_prob_given_context()`` and
    ``sample_given_context()`` run the deterministic flow head. Keeping them apart
    is what lets us draw *one* dropout mask per trial and then many samples from
    the resulting predictive distribution, exactly like the reference agent.

    All methods are batched over a leading axis.
    """

    data_dim: int
    hidden_dims: Sequence[int] = (32, 32)
    dropout_rate: float = 0.05
    flow_layers: int = 4
    knots: int = 8
    interval: float = 4.0
    nn_width: int = 64
    nn_depth: int = 2

    def setup(self):
        self.encoder = DropoutEncoder(
            hidden_dims=tuple(self.hidden_dims), dropout_rate=self.dropout_rate
        )
        n_params = 3 * self.knots - 1
        self.conditioners = [
            ConditionerMLP(
                width=self.nn_width,
                depth=self.nn_depth,
                out_dim=self.data_dim * n_params,
            )
            for _ in range(self.flow_layers)
        ]

    # --- plumbing -------------------------------------------------------
    def _mask(self, layer: int) -> jnp.ndarray:
        """1 = passed through unchanged (and used to condition), 0 = transformed."""
        if self.data_dim == 1:
            # no room to split: the single dimension is transformed conditioned on
            # the context only
            return jnp.zeros((1,))
        return (jnp.arange(self.data_dim) % 2 == layer % 2).astype(jnp.float32)

    def _coupling(self, x, h, layer: int, inverse: bool):
        mask = self._mask(layer)
        raw = self.conditioners[layer](jnp.concatenate([x * mask, h], axis=-1))
        raw = raw.reshape(x.shape[:-1] + (self.data_dim, 3 * self.knots - 1))
        y, log_det = rational_quadratic_spline(
            x, raw, knots=self.knots, interval=self.interval, inverse=inverse
        )
        y = jnp.where(mask > 0, x, y)
        log_det = jnp.sum(jnp.where(mask > 0, 0.0, log_det), axis=-1)
        return y, log_det

    def context(self, z: jnp.ndarray, deterministic: bool = True) -> jnp.ndarray:
        """Condition embedding -> flow context (stochastic when dropout is on)."""
        return self.encoder(z, deterministic=deterministic)

    def log_prob_given_context(self, y: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """log p(y | h) — runs the flow in the normalising (y -> base) direction."""
        log_det = jnp.zeros(y.shape[:-1])
        u = y
        for layer in reversed(range(self.flow_layers)):
            u, ld = self._coupling(u, h, layer, inverse=True)
            log_det = log_det + ld
        base = -0.5 * jnp.sum(u**2 + jnp.log(2.0 * jnp.pi), axis=-1)
        return base + log_det

    def forward_given_context(self, u: jnp.ndarray, h: jnp.ndarray) -> jnp.ndarray:
        """Base sample -> data sample (generative direction)."""
        y = u
        for layer in range(self.flow_layers):
            y, _ = self._coupling(y, h, layer, inverse=False)
        return y

    def sample_given_context(
        self, key: jax.Array, h: jnp.ndarray, n_samples: int
    ) -> jnp.ndarray:
        """Draw ``n_samples`` from ``p(. | h)`` for a single context ``h``."""
        u = jr.normal(key, (n_samples, self.data_dim))
        h_tiled = jnp.broadcast_to(h, (n_samples, h.shape[-1]))
        return self.forward_given_context(u, h_tiled)

    def __call__(
        self, y: jnp.ndarray, z: jnp.ndarray, deterministic: bool = True
    ) -> jnp.ndarray:
        """log p(y | z) with the dropout encoder in front (training objective)."""
        return self.log_prob_given_context(y, self.context(z, deterministic))


@dataclass
class BalsaFlow:
    """A fitted BALSA density model plus the standardisation it was fit under."""

    model: ConditionalSplineFlow
    params: Any
    y_mean: jnp.ndarray
    y_std: jnp.ndarray
    z_mean: jnp.ndarray
    z_std: jnp.ndarray
    losses: Dict[str, List[float]] = field(default_factory=dict)

    def standardize_z(self, z: jnp.ndarray) -> jnp.ndarray:
        return (z - self.z_mean) / self.z_std

    def unstandardize_y(self, y: jnp.ndarray) -> jnp.ndarray:
        return y * self.y_std + self.y_mean


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------


def stack_condition_embeddings(
    cond_arr_dict: Dict[str, jnp.ndarray], conditions: Sequence[str]
) -> Dict[str, jnp.ndarray]:
    """Flatten ``cond_arr_dict`` entries to one embedding vector per condition.

    Same convention as ``ntk.get_ntk_matrix``: entries may be ``(emb_dim,)`` or
    ``(n_contexts, emb_dim)``, in which case the first context is used.
    """
    cond_dict = {}
    for c in conditions:
        arr = jnp.asarray(cond_arr_dict[c])
        assert arr.ndim <= 2, arr.ndim
        cond_dict[c] = arr[0, :] if arr.ndim == 2 else arr
    return cond_dict


def get_target_cells(
    datamodule, condition: str, split_type: str = "train"
) -> jnp.ndarray:
    """Target cells of ``condition`` for one split, as ``(n_cells, n_features)``.

    Cell-population counterpart of ``ntk.get_target_mean`` (same loader fields and
    the same optional dimensionality reduction).
    """
    loader = datamodule.loaders[condition]
    cells = {
        "train": loader.target_train_cells,
        "valid": loader.target_valid_cells,
        "test": loader.target_test_cells,
    }[split_type]
    x = jnp.asarray(loader.target_adata[cells].X)
    if getattr(loader, "reduction", None):
        x = loader.encoder(x)
    return x


def collect_condition_targets(
    datamodule,
    conditions: Sequence[str],
    split_type: str = "train",
    max_cells: Optional[int] = None,
    random_seed: int = 0,
) -> Dict[str, jnp.ndarray]:
    """``{condition: (n_cells, n_features)}`` for the requested split.

    Conditions whose split is empty are skipped (they simply get no score).
    """
    rng = np.random.default_rng(random_seed)
    out = {}
    for c in tqdm(conditions, desc=f"Collecting {split_type} cells"):
        x = get_target_cells(datamodule, c, split_type=split_type)
        if x.shape[0] == 0:
            logger.warning(f"collect_condition_targets: no {split_type} cells for {c}")
            continue
        if max_cells is not None and x.shape[0] > max_cells:
            idx = rng.choice(x.shape[0], size=max_cells, replace=False)
            x = x[jnp.asarray(idx)]
        out[c] = x
    return out


def _split_train_val(
    target_dict: Dict[str, jnp.ndarray],
    conditions: Sequence[str],
    val_prop: float,
    rng: np.random.Generator,
):
    """Per-condition cell-level split, used when no validation cells are given."""
    train, val = {}, {}
    for c in conditions:
        x = target_dict[c]
        n = x.shape[0]
        n_val = int(round(val_prop * n))
        if n < 4 or n_val < 1:
            train[c], val[c] = x, x
            continue
        perm = jnp.asarray(rng.permutation(n))
        val[c] = x[perm[:n_val]]
        train[c] = x[perm[n_val:]]
    return train, val


def _flatten_pairs(
    cond_dict: Dict[str, jnp.ndarray],
    target_dict: Dict[str, jnp.ndarray],
    conditions: Sequence[str],
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Stack all (cell, condition-embedding) pairs of ``conditions``."""
    ys, zs = [], []
    for c in conditions:
        y = jnp.asarray(target_dict[c])
        ys.append(y)
        zs.append(jnp.broadcast_to(cond_dict[c], (y.shape[0], cond_dict[c].shape[0])))
    return jnp.concatenate(ys, axis=0), jnp.concatenate(zs, axis=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------


def fit_balsa_flow(
    cond_dict: Dict[str, jnp.ndarray],
    target_dict: Dict[str, jnp.ndarray],
    train_conditions: Sequence[str],
    val_target_dict: Optional[Dict[str, jnp.ndarray]] = None,
    hidden_dims: Sequence[int] = (32, 32),
    dropout_rate: float = 0.05,
    flow_layers: int = 4,
    knots: int = 8,
    interval: float = 4.0,
    nn_width: int = 64,
    nn_depth: int = 2,
    lr: float = 1e-3,
    wd: float = 3e-4,
    batch_size: int = 256,
    num_epochs: int = 200,
    max_steps_per_epoch: int = 50,
    patience: int = 20,
    val_prop: float = 0.1,
    random_seed: int = 0,
) -> BalsaFlow:
    """Maximum-likelihood fit of ``p(y | z_c)`` on the labelled conditions.

    Trains with dropout active (so the fitted weights are the MC-dropout
    posterior's mode) and early-stops on the held-out negative log-likelihood
    evaluated with dropout *off*.

    Args:
        cond_dict: ``{condition: (emb_dim,)}`` condition embeddings.
        target_dict: ``{condition: (n_cells, data_dim)}`` training cells.
        train_conditions: conditions the model is allowed to see (labelled set).
        val_target_dict: optional held-out cells per condition for early stopping;
            when ``None``, ``val_prop`` of each condition's cells are held out.
        hidden_dims: encoder trunk (the dropout/Bayesian part of the model).
        dropout_rate: MC-dropout rate; 0.05 is the paper's best-performing value.
        flow_layers, knots, interval, nn_width, nn_depth: flow-head architecture.
        lr, wd, batch_size, num_epochs, max_steps_per_epoch, patience: optimisation.
        random_seed: seeds init, batching and the dropout masks used in training.

    Returns:
        A :class:`BalsaFlow` holding the best parameters and the standardisation.
    """
    train_conditions = list(train_conditions)
    if len(train_conditions) == 0:
        raise ValueError("fit_balsa_flow: no training conditions given")
    missing = [c for c in train_conditions if c not in target_dict]
    if missing:
        raise ValueError(f"fit_balsa_flow: no target cells for conditions {missing}")

    rng = np.random.default_rng(random_seed)
    if val_target_dict is None:
        fit_targets, val_targets = _split_train_val(
            target_dict, train_conditions, val_prop, rng
        )
    else:
        fit_targets = target_dict
        val_targets = {
            c: val_target_dict.get(c, target_dict[c]) for c in train_conditions
        }

    y_train, z_train = _flatten_pairs(cond_dict, fit_targets, train_conditions)
    y_val, z_val = _flatten_pairs(cond_dict, val_targets, train_conditions)

    # standardisation: the spline is only expressive inside [-interval, interval]
    y_mean = y_train.mean(axis=0)
    y_std = jnp.maximum(y_train.std(axis=0), 1e-6)
    z_mean = z_train.mean(axis=0)
    z_std = jnp.maximum(z_train.std(axis=0), 1e-6)
    y_train = (y_train - y_mean) / y_std
    z_train = (z_train - z_mean) / z_std
    y_val = (y_val - y_mean) / y_std
    z_val = (z_val - z_mean) / z_std

    data_dim = y_train.shape[-1]
    model = ConditionalSplineFlow(
        data_dim=data_dim,
        hidden_dims=tuple(hidden_dims),
        dropout_rate=dropout_rate,
        flow_layers=flow_layers,
        knots=knots,
        interval=interval,
        nn_width=nn_width,
        nn_depth=nn_depth,
    )

    key = jr.PRNGKey(random_seed)
    key, init_key, drop_key = jr.split(key, 3)
    variables = model.init(
        {"params": init_key, "dropout": drop_key},
        y_train[:1],
        z_train[:1],
        deterministic=True,
    )
    params = variables["params"]

    tx_factory = getattr(optax, "nadamw", optax.adamw)  # paper optimises with NAdam
    tx = tx_factory(learning_rate=lr, weight_decay=wd)
    opt_state = tx.init(params)

    def nll(params, y, z, deterministic, dropout_key):
        log_p = model.apply(
            {"params": params},
            y,
            z,
            deterministic=deterministic,
            rngs={"dropout": dropout_key},
        )
        return -jnp.mean(log_p)

    @jax.jit
    def train_step(params, opt_state, y, z, dropout_key):
        loss, grads = jax.value_and_grad(nll)(params, y, z, False, dropout_key)
        updates, opt_state = tx.update(grads, opt_state, params)
        return optax.apply_updates(params, updates), opt_state, loss

    @jax.jit
    def val_loss_fn(params, dropout_key):
        return nll(params, y_val, z_val, True, dropout_key)

    n_train = y_train.shape[0]
    steps_per_epoch = int(
        max(1, min(max_steps_per_epoch, np.ceil(n_train / batch_size)))
    )
    bsz = int(min(batch_size, n_train))

    best_val = float("inf")
    best_params = params
    no_improve = 0
    losses = {"train": [], "val": []}

    for epoch in range(num_epochs):
        epoch_loss = 0.0
        for _ in range(steps_per_epoch):
            key, batch_key, dropout_key = jr.split(key, 3)
            idx = jr.choice(batch_key, n_train, shape=(bsz,), replace=False)
            params, opt_state, loss = train_step(
                params, opt_state, y_train[idx], z_train[idx], dropout_key
            )
            epoch_loss += float(loss)
        epoch_loss /= steps_per_epoch

        key, val_key = jr.split(key)
        val = float(val_loss_fn(params, val_key))
        losses["train"].append(epoch_loss)
        losses["val"].append(val)

        if not np.isfinite(val):
            logger.warning(f"fit_balsa_flow: non-finite val NLL at epoch {epoch + 1}")
            break
        if val < best_val - 1e-5:
            best_val, best_params, no_improve = val, params, 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(
                    f"fit_balsa_flow: early stop at epoch {epoch + 1}, "
                    f"best val NLL={best_val:.4f}"
                )
                break
        if (epoch + 1) % 25 == 0:
            logger.debug(
                f"fit_balsa_flow epoch {epoch + 1}/{num_epochs}: "
                f"train NLL={epoch_loss:.4f} val NLL={val:.4f}"
            )

    logger.info(
        f"fit_balsa_flow: fit on {n_train} cells from {len(train_conditions)} "
        f"conditions (data_dim={data_dim}), best val NLL={best_val:.4f}"
    )
    return BalsaFlow(
        model=model,
        params=best_params,
        y_mean=y_mean,
        y_std=y_std,
        z_mean=z_mean,
        z_std=z_std,
        losses=losses,
    )


# ---------------------------------------------------------------------------
# MC-dropout predictive samples
# ---------------------------------------------------------------------------


def mc_dropout_samples(
    bf: BalsaFlow,
    z: jnp.ndarray,
    key: jax.Array,
    dropout_trials: int = 25,
    n_samples: int = 64,
    shared_base_noise: bool = False,
) -> jnp.ndarray:
    """Predictive samples from ``dropout_trials`` MC-dropout posterior draws.

    Args:
        bf: fitted model.
        z: *raw* (unstandardised) condition embedding, shape ``(emb_dim,)``.
        key: PRNG key.
        dropout_trials: ``k``, number of dropout forward passes (reference: 25).
        n_samples: samples drawn per trial (reference: 64).
        shared_base_noise: reuse the same base-distribution noise for every trial.
            Off by default (the reference draws fresh samples per trial); turning it
            on removes sampling noise from the disagreement estimate, leaving only
            the parameter disagreement, at the cost of deviating from the paper.
            Worth switching on (or raising ``n_samples``) when the Monte-Carlo floor
            matters: with independent draws, two *identical* distributions still
            score ``E|sorted diff| > 0``, a floor that grows with each condition's
            own spread and so mildly favours wide-distribution conditions. On the
            synthetic check in this module's development that floor was roughly half
            the signal at ``n_samples=64, data_dim=4``.

    Returns:
        ``(dropout_trials, n_samples, data_dim)`` samples in the original data scale.
    """
    return mc_dropout_samples_batched(
        bf,
        jnp.asarray(z)[None, :],
        key,
        dropout_trials=dropout_trials,
        n_samples=n_samples,
        shared_base_noise=shared_base_noise,
    )[:, 0]


def make_ensemble_sampler(bf: BalsaFlow, n_samples: int):
    """Build the jitted ``(z_std, ctx_keys, sample_keys) -> samples`` sampler.

    Returned separately so a scoring sweep compiles once and reuses the same
    executable for every chunk of conditions.
    """
    model, params = bf.model, bf.params

    def _sample_ensemble(z_std, ctx_keys, sample_keys):
        # one encoder pass per dropout trial over *all* conditions at once; flax
        # draws an independent dropout mask per row, which is exactly what the
        # reference implementation gets from a single batched torch forward pass
        contexts = jax.vmap(
            lambda dk: model.apply(
                {"params": params},
                z_std,
                False,  # deterministic=False -> dropout active
                method=ConditionalSplineFlow.context,
                rngs={"dropout": dk},
            )
        )(
            ctx_keys
        )  # (k, n_cond, h_dim)

        def per_trial(contexts_t, sample_key):
            keys = jr.split(sample_key, contexts_t.shape[0])
            return jax.vmap(
                lambda h, k: model.apply(
                    {"params": params},
                    k,
                    h,
                    n_samples,
                    method=ConditionalSplineFlow.sample_given_context,
                )
            )(contexts_t, keys)

        return jax.vmap(per_trial)(contexts, sample_keys)

    return jax.jit(_sample_ensemble)


def mc_dropout_samples_batched(
    bf: BalsaFlow,
    z: jnp.ndarray,
    key: jax.Array,
    dropout_trials: int = 25,
    n_samples: int = 64,
    shared_base_noise: bool = False,
    sampler=None,
) -> jnp.ndarray:
    """:func:`mc_dropout_samples` for many conditions at once.

    Args:
        bf: fitted model.
        z: ``(n_cond, emb_dim)`` *raw* (unstandardised) condition embeddings.
        key: PRNG key.
        dropout_trials, n_samples, shared_base_noise: see
            :func:`mc_dropout_samples`.
        sampler: optional pre-built :func:`make_ensemble_sampler` closure to reuse
            (avoids recompiling per chunk).

    Returns:
        ``(dropout_trials, n_cond, n_samples, data_dim)`` in the original data scale.
    """
    z_std = bf.standardize_z(jnp.asarray(z))
    ctx_key, noise_key = jr.split(key)
    ctx_keys = jr.split(ctx_key, dropout_trials)
    if shared_base_noise:
        # identical sampling key per trial -> the same base noise is pushed through
        # every posterior sample, so only parameter disagreement remains
        sample_keys = jnp.broadcast_to(
            noise_key, (dropout_trials,) + jnp.shape(noise_key)
        )
    else:
        sample_keys = jr.split(noise_key, dropout_trials)

    if sampler is None:
        sampler = make_ensemble_sampler(bf, n_samples)
    return bf.unstandardize_y(sampler(z_std, ctx_keys, sample_keys))


# ---------------------------------------------------------------------------
# Distances between predictive distributions
# ---------------------------------------------------------------------------


def emd_marginal(a: jnp.ndarray, b: jnp.ndarray, reduce: str = "sum") -> jnp.ndarray:
    """Per-dimension 1-D Wasserstein-1 distance between two sample sets.

    ``a``, ``b``: ``(n_a, d)`` / ``(n_b, d)``. For equal sample sizes this is
    ``mean |sort(a) - sort(b)|`` per dimension, i.e. exactly what
    ``scipy.stats.wasserstein_distance`` computes on equal-weight samples (the
    reference implementation's distance), and reduces to it when ``d == 1``.
    Unequal sizes are handled by comparing quantile functions on a common grid.
    """
    a, b = jnp.asarray(a), jnp.asarray(b)
    if a.shape[0] == b.shape[0]:
        per_dim = jnp.mean(jnp.abs(jnp.sort(a, axis=0) - jnp.sort(b, axis=0)), axis=0)
    else:
        n = max(a.shape[0], b.shape[0])
        levels = (jnp.arange(n) + 0.5) / n
        qa = jnp.quantile(a, levels, axis=0)
        qb = jnp.quantile(b, levels, axis=0)
        per_dim = jnp.mean(jnp.abs(qa - qb), axis=0)
    return jnp.sum(per_dim) if reduce == "sum" else jnp.mean(per_dim)


def emd_sinkhorn(
    a: jnp.ndarray, b: jnp.ndarray, epsilon: Optional[float] = None
) -> jnp.ndarray:
    """Entropic-OT (debiased Sinkhorn divergence) between two ``d``-dim clouds.

    Uses a Euclidean ground cost so the value approximates a genuine
    (multivariate) Earth-Mover distance rather than a squared-Euclidean OT cost.
    """
    from ott.geometry import costs, pointcloud
    from ott.tools.sinkhorn_divergence import sinkhorn_divergence

    geom_kwargs = {"cost_fn": costs.Euclidean()}
    if epsilon is not None:
        geom_kwargs["epsilon"] = epsilon
    out = sinkhorn_divergence(
        pointcloud.PointCloud,
        jnp.asarray(a),
        jnp.asarray(b),
        solve_kwargs={"threshold": 1e-3, "max_iterations": 2000},
        **geom_kwargs,
    )
    return jnp.clip(out[0], 0.0, None)


def _consecutive_emd(
    samples: jnp.ndarray,
    emd_mode: str = "marginal",
    reduce: str = "sum",
    emd_epsilon: Optional[float] = None,
) -> float:
    """``sum_{i=1}^{k-1} EMD(y'_i, y'_{i+1})`` — the BALSA-EMD acquisition score."""
    total = 0.0
    for i in range(samples.shape[0] - 1):
        if emd_mode == "marginal":
            d = emd_marginal(samples[i], samples[i + 1], reduce=reduce)
        elif emd_mode == "sinkhorn":
            d = emd_sinkhorn(samples[i], samples[i + 1], epsilon=emd_epsilon)
        else:
            raise ValueError(f"Unknown emd_mode: {emd_mode}")
        total += float(d)
    return total


def quantile_features(
    samples: jnp.ndarray, n_quantiles: int = 32, center: bool = True
) -> jnp.ndarray:
    """Quantile-deviation feature vector of an MC-dropout ensemble.

    ``samples``: ``(k, n_samples, d)``. Per trial we take ``n_quantiles``
    per-dimension sample quantiles ``Q_i`` (a discretised inverse CDF), subtract the
    ensemble mean ``Qbar``, and concatenate. With the ``1/sqrt(k * n_quantiles)``
    scaling used here, the squared norm of the result is

        ``mean_i (1/m) sum_{levels, dims} (Q_i - Qbar)^2  =  mean_i W2^2(p_i, pbar)``

    in the 1-D-marginal sense — the L2/W2 analogue of the L1/W1 BALSA-EMD score,
    but expressed as an inner-product feature so that Gram matrices built from it
    are PSD.
    """
    k = samples.shape[0]
    levels = (jnp.arange(n_quantiles) + 0.5) / n_quantiles
    q = jnp.stack(
        [jnp.quantile(samples[i], levels, axis=0) for i in range(k)]
    )  # (k, n_quantiles, d)
    if center:
        q = q - q.mean(axis=0, keepdims=True)
    return q.reshape(-1) / jnp.sqrt(k * n_quantiles)


@jax.jit
def _marginal_emd_scores_batched(samples: jnp.ndarray) -> jnp.ndarray:
    """Per-dimension BALSA-EMD scores for a batch of conditions.

    ``samples``: ``(k, n_cond, n_samples, d)``. Returns ``(n_cond, d)``, the
    per-dimension ``sum_{i} W1(y'_i, y'_{i+1})``; the caller sums or averages over
    dimensions. Identical arithmetic to :func:`emd_marginal` looped over
    consecutive trials, just vectorised over conditions and trials.
    """
    sorted_samples = jnp.sort(samples, axis=2)
    per_pair = jnp.mean(
        jnp.abs(sorted_samples[1:] - sorted_samples[:-1]), axis=2
    )  # (k-1, n_cond, d)
    return jnp.sum(per_pair, axis=0)


@partial(jax.jit, static_argnums=1)
def _quantile_features_batched(
    samples: jnp.ndarray, n_quantiles: int = 32
) -> jnp.ndarray:
    """:func:`quantile_features` for a batch: ``(k, n_cond, n, d) -> (n_cond, k*m*d)``."""
    k = samples.shape[0]
    levels = (jnp.arange(n_quantiles) + 0.5) / n_quantiles
    q = jnp.quantile(samples, levels, axis=2)  # (m, k, n_cond, d)
    q = q - q.mean(axis=1, keepdims=True)
    q = jnp.transpose(q, (2, 1, 0, 3))  # (n_cond, k, m, d)
    return q.reshape(q.shape[0], -1) / jnp.sqrt(k * n_quantiles)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def get_balsa_scores(
    bf: BalsaFlow,
    cond_dict: Dict[str, jnp.ndarray],
    conditions: Sequence[str],
    dropout_trials: int = 25,
    n_samples: int = 64,
    n_quantiles: int = 32,
    emd_mode: str = "marginal",
    emd_reduce: str = "sum",
    emd_epsilon: Optional[float] = None,
    shared_base_noise: bool = False,
    random_seed: int = 0,
    compute_emd: bool = True,
    chunk_size: int = 64,
) -> Tuple[Dict[str, float], Dict[str, jnp.ndarray]]:
    """BALSA-EMD scores and quantile features for every condition.

    Conditions are processed in chunks of ``chunk_size`` (one batched encoder pass
    per dropout trial per chunk), which is what keeps a pool of hundreds of
    conditions cheap. ``chunk_size`` therefore changes how PRNG keys are consumed:
    results are reproducible for a given ``(random_seed, chunk_size)`` pair, and only
    Monte-Carlo-equivalent — not bit-identical — across different chunk sizes.

    Returns:
        ``(scores, features)``: ``scores[c]`` is ``sum_i EMD(y'_i, y'_{i+1})`` over
        consecutive MC-dropout trials (``0.0`` for all conditions if
        ``compute_emd=False``), and ``features[c]`` is the quantile-deviation
        feature vector of the same ensemble.
    """
    conditions = list(conditions)
    if emd_mode not in ("marginal", "sinkhorn"):
        raise ValueError(f"Unknown emd_mode: {emd_mode}")
    z_all = jnp.stack([jnp.asarray(cond_dict[c]) for c in conditions])

    key = jr.PRNGKey(random_seed)
    sampler = make_ensemble_sampler(bf, n_samples)
    scores, features = {}, {}

    for start in tqdm(
        range(0, len(conditions), chunk_size), desc="BALSA scoring", unit="chunk"
    ):
        chunk = conditions[start : start + chunk_size]
        key, chunk_key = jr.split(key)
        samples = mc_dropout_samples_batched(
            bf,
            z_all[start : start + chunk_size],
            chunk_key,
            dropout_trials=dropout_trials,
            n_samples=n_samples,
            shared_base_noise=shared_base_noise,
            sampler=sampler,
        )  # (k, n_chunk, n_samples, data_dim)

        feats = _quantile_features_batched(samples, n_quantiles)
        for i, c in enumerate(chunk):
            features[c] = feats[i]

        if not compute_emd:
            for c in chunk:
                scores[c] = 0.0
        elif emd_mode == "marginal":
            per_dim = _marginal_emd_scores_batched(samples)  # (n_chunk, data_dim)
            reduced = (
                jnp.sum(per_dim, axis=-1)
                if emd_reduce == "sum"
                else jnp.mean(per_dim, axis=-1)
            )
            for i, c in enumerate(chunk):
                scores[c] = float(reduced[i])
        else:
            # multivariate OT has no batched closed form here: one solve per pair
            for i, c in enumerate(chunk):
                scores[c] = _consecutive_emd(
                    samples[:, i],
                    emd_mode="sinkhorn",
                    reduce=emd_reduce,
                    emd_epsilon=emd_epsilon,
                )
    return scores, features


def get_balsa_kernel_matrix(
    cond_arr_dict,
    target_dict,
    val_target_dict,
    train_conditions,
    all_conditions,
    hidden_dims=[32, 32],
    kernel_mode: str = "quantile",
    dropout_rate: float = 0.05,
    dropout_trials: int = 25,
    n_samples: int = 64,
    n_quantiles: int = 32,
    emd_mode: str = "marginal",
    emd_reduce: str = "sum",
    emd_epsilon: Optional[float] = None,
    shared_base_noise: bool = False,
    flow_layers: int = 4,
    knots: int = 8,
    interval: float = 4.0,
    nn_width: int = 64,
    nn_depth: int = 2,
    lr: float = 1e-3,
    wd: float = 3e-4,
    batch_size: int = 256,
    num_epochs: int = 200,
    max_steps_per_epoch: int = 50,
    patience: int = 20,
    random_seed: int = 0,
    chunk_size: int = 64,
) -> Tuple[jnp.ndarray, Dict[str, Any]]:
    """BALSA-EMD kernel matrix over ``all_conditions``.

    Fits ``p(y | z_c)`` (dropout MLP encoder + conditional neural spline flow) on
    the cells of ``train_conditions`` only, then measures MC-dropout distribution
    disagreement at every condition in ``all_conditions``.

    Args:
        cond_arr_dict: ``{condition: embedding}`` (``(emb_dim,)`` or
            ``(n_contexts, emb_dim)``), as passed to the other ``acpl`` kernels.
        target_dict: ``{condition: (n_cells, data_dim)}`` cells used for fitting.
            Only ``train_conditions`` entries are read, so unlabelled conditions'
            cells are never touched. See :func:`collect_condition_targets`.
        val_target_dict: optional ``{condition: (n_cells, data_dim)}`` validation
            cells for early stopping; ``None`` holds out 10% of each condition.
        train_conditions: currently labelled conditions.
        all_conditions: conditions to score, in the order the kernel rows/columns
            must follow (``acpl.selection.select_batch`` indexes by this order).
        hidden_dims: dropout encoder trunk, the model's Bayesian part.
        kernel_mode: ``"quantile"`` for the PSD quantile-deviation Gram matrix
            (usable with every batch-selection strategy) or ``"emd"`` for
            ``diag(BALSA_EMD)``, which reproduces the paper's top-``b`` acquisition
            when combined with ``batch_selection_strategy="uncertainty"``.
        dropout_rate: MC-dropout rate (paper's best value: 0.05).
        dropout_trials: number of posterior samples ``k`` (reference default: 25).
        n_samples: predictive samples per trial (reference default: 64).
        n_quantiles: quantile-grid resolution for ``kernel_mode="quantile"``.
        emd_mode: ``"marginal"`` (per-dimension 1-D W1 summed over dimensions,
            matching the reference's ``scipy.stats.wasserstein_distance``) or
            ``"sinkhorn"`` (multivariate entropic OT with a Euclidean cost).
        emd_reduce: ``"sum"`` or ``"mean"`` over dimensions for ``"marginal"``.
        emd_epsilon: entropic regularisation for ``"sinkhorn"`` (``None`` = OTT's
            relative default).
        shared_base_noise: see :func:`mc_dropout_samples`.
        flow_layers, knots, interval, nn_width, nn_depth: flow-head architecture.
        lr, wd, batch_size, num_epochs, max_steps_per_epoch, patience: optimisation.
        random_seed: seeds fitting, dropout masks and predictive sampling.

    Returns:
        ``(kernel_matrix, aux)`` where ``kernel_matrix`` is
        ``(len(all_conditions), len(all_conditions))`` and ``aux`` holds the fitted
        model (``"balsa_flow"``), ``"scores"``, ``"features"``, ``"losses"`` and the
        resolved settings, so both kernel modes can be inspected without refitting.
    """
    all_conditions = list(all_conditions)
    train_conditions = list(train_conditions)
    if kernel_mode not in ("quantile", "emd"):
        raise ValueError(f"Unknown kernel_mode: {kernel_mode}")

    cond_dict = stack_condition_embeddings(cond_arr_dict, all_conditions)

    bf = fit_balsa_flow(
        cond_dict=cond_dict,
        target_dict=target_dict,
        train_conditions=train_conditions,
        val_target_dict=val_target_dict,
        hidden_dims=hidden_dims,
        dropout_rate=dropout_rate,
        flow_layers=flow_layers,
        knots=knots,
        interval=interval,
        nn_width=nn_width,
        nn_depth=nn_depth,
        lr=lr,
        wd=wd,
        batch_size=batch_size,
        num_epochs=num_epochs,
        max_steps_per_epoch=max_steps_per_epoch,
        patience=patience,
        random_seed=random_seed,
    )

    scores, features = get_balsa_scores(
        bf,
        cond_dict,
        all_conditions,
        dropout_trials=dropout_trials,
        n_samples=n_samples,
        n_quantiles=n_quantiles,
        emd_mode=emd_mode,
        emd_reduce=emd_reduce,
        emd_epsilon=emd_epsilon,
        shared_base_noise=shared_base_noise,
        random_seed=random_seed,
        # the EMD score is only needed for its own kernel mode; skip the (possibly
        # Sinkhorn-priced) computation when the quantile Gram is what we return
        compute_emd=kernel_mode == "emd" or emd_mode == "marginal",
        chunk_size=chunk_size,
    )

    if kernel_mode == "quantile":
        phi = jnp.stack([features[c] for c in all_conditions])
        kernel_mat = phi @ phi.T
    else:
        kernel_mat = jnp.diag(jnp.asarray([scores[c] for c in all_conditions]))

    diag = jnp.diag(kernel_mat)
    logger.info(
        f"get_balsa_kernel_matrix[{kernel_mode}]: {len(all_conditions)} conditions, "
        f"k={dropout_trials} dropout trials, disagreement diag "
        f"min={float(diag.min()):.4g} median={float(jnp.median(diag)):.4g} "
        f"max={float(diag.max()):.4g}"
    )

    aux = {
        "balsa_flow": bf,
        "scores": scores,
        "features": features,
        "losses": bf.losses,
        "kernel_mode": kernel_mode,
        "emd_mode": emd_mode,
        "dropout_trials": dropout_trials,
        "dropout_rate": dropout_rate,
        "n_samples": n_samples,
    }
    return kernel_mat, aux
