from typing import Literal, Tuple, Dict, Optional
import hashlib
import json
import os
import pandas as pd
import pickle as pkl
from loguru import logger
from sklearn.model_selection import KFold
from tqdm.auto import tqdm
import numpy as np
import jax
import jax.numpy as jnp
import jax.random as jr
import optax
import equinox as eqx

# Local imports
from cmonge.models.embedding import BaseEmbedding, EmbeddingFactory
from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.utils import load_config as load_config_util
from ntk import (
    get_ntk_matrix,
    get_meanvar_ntk_matrix,
    get_quartile_ntk_matrix,
    evaluate_mean_predictor,
)
from balsa import get_balsa_kernel_matrix
from selection import (
    NO_KERNEL_STRATEGIES,
    acquisition_seed_for,
    sample_random_batch,
    select_batch,
)
from utils import get_parser, kernel_distance
from consts import condition_map, control_map
from sklearn.model_selection import train_test_split
from scipy.stats import Mixture, Normal
import sys

# ---- progress / logging controls -------------------------------------------
# By default the script prints only high-level progress (one line per fold and
# per AL iteration, with the running test metric). Set AL_VERBOSE=1 to restore
# the detailed per-step logs and the tqdm progress bars.
VERBOSE = bool(os.environ.get("AL_VERBOSE"))

# Anchor all output to the repo root so the location is independent of the
# current working directory (jobs cd into src/acpl before running, which used
# to scatter results into src/acpl/models instead of the canonical
# <repo>/models that collect_results_quantile.py reads from).
REPO_ROOT = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)


def _progress_filter(record):
    """Keep our own progress logs; silence chatty INFO/DEBUG from the cmonge /
    cellflow libraries (per-condition datamodule setup spam). Warnings+ pass."""
    if VERBOSE:
        return True
    name = record["name"] or ""
    if name.startswith(("cmonge", "cellflow", "cfp")):
        return record["level"].no >= 30  # WARNING and above only
    return True


def _configure_logging():
    """Compact single-line progress logging; hide DEBUG detail unless AL_VERBOSE=1."""
    logger.remove()
    logger.add(
        sys.stderr,
        level="DEBUG" if VERBOSE else "INFO",
        format="<green>{time:HH:mm:ss}</green> | {message}",
        colorize=True,
        filter=_progress_filter,
    )


def _mean_test_w2(trainer):
    """Mean test-split w2 distance from a trainer's evaluation metrics (NaN if absent)."""
    try:
        ev = trainer.metrics.get("evaluation", {})
        vals = [
            v["w2_distance"]
            for k, v in ev.items()
            if k.startswith("test-") and v.get("w2_distance") is not None
        ]
        return float(np.mean(vals)) if vals else float("nan")
    except Exception:
        return float("nan")

# `n_qs` -- the number of quantile levels the model predicts -- is resolved below
# as `config.get("n_qs", n_bins)`, so a config with no explicit `n_qs` inherits its
# `n_bins`. w2_distance is a mean over those levels, so two runs at different n_qs
# are on different measurement scales and must never be averaged together.
#
# The config id is part of the output path (`Quantile:{model_config_id}{prefix}`),
# and it is the ONLY thing that separates them on disk. Changing a config's n_qs
# without changing its id therefore writes the new scale into the old directory:
# that is how sim_mixture_a5 `Quantile:dv10lr1e-02wd1e-03_ib10` ended up holding
# seeds 0-9 at 32 and seeds 10-49 at 16. Put the level count in the id instead --
# add a new entry rather than editing an existing one.
model_configs = {
    "s": dict(dim_hidden=[64, 64, 32], n_bins=32, nn_width=64, nn_depth=2),
    "d16": dict(n_bins=32, nn_width=32, nn_depth=3, n_qs=16),
    "d32": dict(n_bins=32, nn_width=32, nn_depth=3, n_qs=32),
    "l": dict(dim_hidden=[128, 128, 128, 32], n_bins=32, nn_width=128, nn_depth=3),
}

# Deprecated. Everything under `models/quantile-al/*/Quantile:dv10*` was produced
# by this config back when it was named `d`, at n_qs=16; the alias keeps those
# directories resumable and re-evaluable. New work must name the level count
# (`d16` / `d32`) so it lands in its own directory. Do not repoint this at `d32`
# -- that would write 32-level runs into the 16-level directories.
model_configs["d"] = model_configs["d16"]


class ConditionalQuantile(eqx.Module):
    """Conditional Quantile Predictor for active learning.

    This class uses an MLP to predict quantile values for each 1 dimensional data
    given a conditioning variable.
    """

    mlp: eqx.nn.MLP
    final_layer_min: eqx.nn.Linear
    final_layer_deltas: eqx.nn.MLP
    dropout: eqx.nn.Dropout
    cond_dim: int
    dropout_rate: float
    n_qs: int
    infinite_data_gaussian: bool

    def __init__(
        self,
        key: jax.Array,
        cond_dim: int,
        nn_width: int = 50,
        nn_depth: int = 1,
        n_qs: int = 64,
        dropout_rate: float = 0.0,
        infinite_data_gaussian: bool = False,
    ):
        """Initialize conditional quantile predictor.

        Args:
            key: JAX random key
            cond_dim: Dimension of conditioning variable
            n_qs: Number of quantiles (includes Q_0)
            nn_width: Width of MLP hidden layers
            nn_depth: Number of MLP hidden layers
            dropout_rate: Dropout probability applied after each hidden layer (0 = no dropout)
        """
        key, key_min, key_deltas = jr.split(key, 3)
        self.cond_dim = cond_dim
        self.dropout_rate = dropout_rate
        self.n_qs = n_qs
        # MLP: condition -> logits for all dimensions
        # Output: [n_bins] logits
        self.mlp = eqx.nn.MLP(
            in_size=cond_dim,
            out_size=nn_width,
            width_size=nn_width,
            depth=nn_depth,
            activation=jax.nn.relu,
            key=key,
        )
        self.final_layer_min = eqx.nn.Linear(nn_width, 1, key=key_min)
        self.final_layer_deltas = eqx.nn.MLP(in_size=nn_width, out_size=n_qs - 1, width_size=0, depth=0, final_activation=jax.nn.softplus, key=key_deltas)
        self.dropout = eqx.nn.Dropout(p=dropout_rate)
        self.infinite_data_gaussian = infinite_data_gaussian

    def get_hidden(self, condition, mlp_weights=None, mlp_biases=None, key=None, inference=True):
        """Forward pass through all layers except the last.

        Args:
            condition: Input condition vector
            mlp_weights: List of weight matrices for each hidden layer. If None, uses
                         self.mlp layers. Pass explicitly to enable jax.jacobian/grad
                         through these parameters.
            mlp_biases: List of bias vectors for each hidden layer. If None, uses
                        self.mlp layers.
            key: JAX random key for dropout (required when inference=False)
            inference: If True, dropout is disabled (default). Always True during
                       evaluation, sampling, and kernel computation.
        """
        layers = self.mlp.layers[:-1]
        n_layers = len(layers)
        if mlp_weights is None:
            mlp_weights = [layer.weight for layer in layers]
        if mlp_biases is None:
            mlp_biases = [layer.bias for layer in layers]
        keys = (
            jr.split(key, n_layers)
            if (key is not None and not inference)
            else [None] * n_layers
        )
        x = condition
        for i, (w, b) in enumerate(zip(mlp_weights, mlp_biases)):
            x = w @ x + b
            x = jax.nn.relu(x)
            x = self.dropout(x, key=keys[i], inference=inference)
        return x

    def pred_from_params(self, min_weights, min_bias, deltas_weights, deltas_bias, h):
        min_pred = min_weights @ h + min_bias
        deltas_pred = jax.nn.softplus(deltas_weights @ h + deltas_bias)
        return jnp.concatenate([min_pred, min_pred + jnp.cumsum(deltas_pred)])  # Ensure monotonicity

    def predict(
        self,
        condition: jnp.ndarray,
        key=None,
        inference=True,
        return_jacobian=False,
    ) -> jnp.ndarray:
        """Predict quantile values for given condition.

        Args:
            condition: Conditioning variable [cond_dim] or [batch, cond_dim]
            q: Quantile index [n_qs] or [batch, n_qs]
            key: JAX random key for dropout (required when inference=False)
            inference: If True (default), dropout is disabled. Set to False during training.

        Returns:
            Quantile values [data_dim, n_qs] or [batch, data_dim, n_qs]
        """
        z = self.get_hidden(condition, key=key, inference=inference)
        if return_jacobian:
            return z
        return self.pred_from_params(
            self.final_layer_min.weight, self.final_layer_min.bias, self.final_layer_deltas.layers[0].weight, self.final_layer_deltas.layers[0].bias, z
        )


class ConditionalQuantileTrainer:
    """Trainer for conditional quantile prediction."""

    embedding_factory: Dict[str, BaseEmbedding] = EmbeddingFactory

    def __init__(
        self,
        cond_dim: int,
        config: dict,
        logger_path: str,
        datamodule: ConditionalDataModule,
        condition_batch_size=32,
    ):
        """Initialize trainer.

        Args:
            data_dim: Dimension of data
            cond_dim: Dimension of conditioning variables
            config: Configuration dictionary
            logger_path: Path to save training logs
        """
        self.cond_dim = cond_dim
        self.config = config
        self.logger_path = logger_path
        self.metrics = {}
        self.model = None
        self.datamodule = datamodule
        self.condition_batch_size = condition_batch_size
        self.infinite_data_gaussian = config.get("infinite_data_gaussian", False)

    def setup_model(self, key: jax.Array):
        """Setup model."""
        embed_module = self.embedding_factory[self.config["embedding"]["name"]]
        self.embedding_module = embed_module(
            datamodule=self.datamodule, **self.config["embedding"]
        )  # Calculates embedding

        self.model = ConditionalQuantile(
            key=key,
            cond_dim=self.cond_dim,
            n_qs=self.config.get("n_qs", 16),
            nn_width=self.config.get("nn_width", 50),
            nn_depth=self.config.get("nn_depth", 1),
            dropout_rate=self.config.get("dropout_rate", 0.0),
            infinite_data_gaussian=self.infinite_data_gaussian,
        )
        return self.model

    def generate_batch(
        self,
        datamodule: ConditionalDataModule,
        split_type: str,
        condition: Optional[str] = None,
        train: bool = True,
    ) -> Dict[str, jnp.ndarray]:
        """Generate a batch of condition and samples."""
        if condition is None:
            condition = datamodule.sample_condition(split_type)
        loaders = datamodule.loaders[condition]
        loader_source, loader_target = loaders.get_loaders_by_type(split_type)
        embeddings, n_contexts = self.embedding_module(condition=condition)
        # Get raw data
        source_data = next(loader_source)
        target_data = next(loader_target)

        return (
            {
                "source": source_data,
                "target": target_data,
                "condition": embeddings,
                "num_contexts": 1,  # n_contexts,
                "train": train,
            },
            condition,
        )

    def _compute_target_quantiles(self, data: jnp.ndarray,) -> jnp.ndarray:
        data_sorted = jnp.sort(data, axis=0)
        quantiles = jnp.linspace(1 / (2*self.model.n_qs), 1 - 1 / (2*self.model.n_qs), self.model.n_qs)
        quantile_values = jnp.quantile(data_sorted, quantiles, axis=0, method="linear")
        return quantiles, quantile_values

    def _compute_theoretical_quantiles(self, condition: jnp.ndarray, n_qs: int = None) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Compute theoretical quantiles for infinite-data Gaussian setting.

        ``n_qs`` defaults to ``self.model.n_qs`` but may be passed explicitly so
        callers without a built model (e.g. the data-prep trainer in
        ``prepare_data_grad``) can still use this.
        """
        if n_qs is None:
            n_qs = self.model.n_qs
        mu1, std1, prop1, mu2, std2, prop2, mu3, std3, prop3 = condition.tolist()
        components = [Normal(mu=mu1, sigma=std1), Normal(mu=mu2, sigma=std2), Normal(mu=mu3, sigma=std3)]
        mixture = Mixture(weights=[prop1, prop2, prop3], components=components)
        quantiles = jnp.linspace(1 / (2*n_qs), 1 - 1 / (2*n_qs), n_qs)
        quantile_values = mixture.icdf(quantiles)
        return quantiles, quantile_values

    def train(self):
        """Train the quantile predictor.

        Returns:
            Trained model, logs, final step
        """
        logger.debug("Training conditional quantile predictor...")

        # Collect all training data from the datamodule
        target_quantiles = []  # Pre-computed quantiles
        val_quantiles = []
        target_quantile_values = []
        val_quantile_values = []
        condition_data = []
        val_condition_data = []
        for condition in tqdm(
            self.datamodule.train_conditions, desc="Loading training data",
            disable=not VERBOSE,
        ):
            train_batch, _ = self.generate_batch(
                self.datamodule, "train", condition=condition, train=True
            )
            if self.infinite_data_gaussian:
                target_q, target_qvalues = self._compute_theoretical_quantiles(train_batch["condition"][0, :])
            else:
                target_q, target_qvalues = self._compute_target_quantiles(
                    train_batch["target"], 
                )
            cond_emb = train_batch["condition"][0, :]
            if jnp.any(jnp.isnan(cond_emb)):
                logger.warning(
                    f"NaN in condition embedding for {condition}, ignoring condition"
                )
                continue
            target_quantiles.append(target_q)
            target_quantile_values.append(target_qvalues)
            condition_data.append(cond_emb)

        for val_cond in (self.datamodule.valid_conditions):
            valid_batch, _ = self.generate_batch(
                self.datamodule, "valid", condition=val_cond, train=False
            )
            cond_emb = valid_batch["condition"][0, :]
            # Match the training objective: theoretical mixture quantiles in the
            # infinite-data regime, empirical sample quantiles otherwise. Keeping
            # this consistent with train() avoids a mismatched early-stopping signal.
            if self.infinite_data_gaussian:
                val_q, val_qvalues = self._compute_theoretical_quantiles(cond_emb)
            else:
                val_q, val_qvalues = self._compute_target_quantiles(valid_batch["target"])
            val_quantiles.append(val_q)
            val_quantile_values.append(val_qvalues)
            val_condition_data.append(cond_emb)

        # Stack data
        condition_data = jnp.stack(condition_data)  # [n_conditions, cond_dim]
        if len(val_condition_data) > 0:
            val_condition_data = jnp.stack(val_condition_data)
        else:
            pass
        target_quantile_values = jnp.stack(
            target_quantile_values
        )  # [n_conditions, n_samples]
        val_quantile_values = jnp.stack(
            val_quantile_values
        )  # [n_conditions, n_samples]
        target_quantiles = jnp.stack(target_quantiles)
        val_quantiles = jnp.stack(val_quantiles)

        logger.debug(
            f"Training on {len(condition_data)} conditions with {self.model.n_qs} quantiles each"
        )

        # Setup optimizer
        lr = self.config.get("lr", 1e-5)
        weight_decay = self.config.get("weight_decay", 0.0)
        if weight_decay > 0.0:
            optimizer = optax.adamw(lr, weight_decay=weight_decay)
        else:
            optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(self.model, eqx.is_inexact_array))

        # Training loop
        train_losses = []
        val_losses = []
        best_val_loss = float("inf")
        best_model_path = self.logger_path.replace("_log.pkl", "_best_model.eqx")
        patience_counter = 0

        n_epochs = self.config.get("num_epochs", 3000)
        min_epochs = self.config.get("min_epochs", 200)
        patience = self.config.get("patience", 10)

        loss_fn = _loss_fn
        train_step = (
            lambda model, opt_state, cond, target_q, target_qvals, key: _train_step(
                model, opt_state, cond, target_q, target_qvals, key, optimizer
            )
        )

        key = jr.PRNGKey(self.config.get("seed", 0))

        for epoch in tqdm(range(n_epochs), desc="Training", disable=not VERBOSE):
            # Shuffle training data
            key, subkey = jr.split(key)
            perm = jr.permutation(subkey, len(condition_data))
            train_cond_shuffled = condition_data[perm]
            train_quantiles_values_shuffled = target_quantile_values[perm]
            train_quantiles_shuffled = target_quantiles[perm]

            # Train epoch
            epoch_losses = []
            for i in range(0, len(condition_data), self.condition_batch_size):
                batch_cond = train_cond_shuffled[i : i + self.condition_batch_size]
                batch_q = train_quantiles_shuffled[i : i + self.condition_batch_size]
                batch_qvals = train_quantiles_values_shuffled[
                    i : i + self.condition_batch_size
                ]

                key, subkey = jr.split(key)
                self.model, opt_state, batch_loss = train_step(
                    self.model, opt_state, batch_cond, batch_q, batch_qvals, subkey
                )
                epoch_losses.append(float(batch_loss))

            train_loss = np.mean(epoch_losses)
            train_losses.append(train_loss)

            # Validation
            val_loss = float(
                loss_fn(self.model, val_condition_data, val_quantiles, val_quantile_values)
            )
            val_losses.append(val_loss)

            # Early stopping
            if epoch >= min_epochs:
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    eqx.tree_serialise_leaves(best_model_path, self.model)
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= patience:
                    logger.info(f"Early stopping at epoch {epoch}")
                    break

        if os.path.exists(best_model_path):
            self.model = eqx.tree_deserialise_leaves(best_model_path, self.model)
            os.remove(best_model_path)

        logs = {
            "train_loss": np.array(train_losses),
            "val_loss": np.array(val_losses),
        }

        # Save logs
        with open(self.logger_path, "wb") as f:
            pkl.dump(logs, f)

        return self.model, logs, len(train_losses)

    def _compute_qvalues(self, data: jnp.ndarray) -> jnp.ndarray:
        """Compute empirical quantile values for data.

        Args:
            data: [n_samples]

        Returns:
            Quantile values [n_qs]
        """
        data_sorted = jnp.sort(data, axis=0)
        n_samples = data.shape[0]
        quantiles = jnp.linspace(1 / (2 * self.model.n_qs), 1 - 1 / (2 * self.model.n_qs), self.model.n_qs)
        quantile_values = jnp.quantile(data_sorted, quantiles, axis=0, method="linear")
        return quantile_values

    def evaluate(
        self,
        datamodule: ConditionalDataModule,
        splits=["test"],
        n_samples=1000,
        **kwargs,
    ):
        """Evaluate the trained model.

        Args:
            datamodule: ConditionalDataModule from cmonge
            splits: Which splits to evaluate on
            n_samples: Number of samples for evaluation (not currently used)

        Returns:
            Dictionary of evaluation metrics
        """
        logger.debug(f"Evaluating on splits: {splits}")
        results = {}

        for split in splits:
            # Get conditions for this split
            if split == "train":
                split_conditions = datamodule.train_conditions
            elif split == "valid":
                split_conditions = datamodule.valid_conditions
            elif split == "test":
                split_conditions = datamodule.test_conditions
            else:
                logger.warning(f"Unknown split: {split}, skipping")
                continue

            for condition in tqdm(split_conditions, desc=f"Evaluating {split}", disable=not VERBOSE):
                loaders = datamodule.loaders[condition]
                loader_source, loader_target = loaders.get_loaders_by_type(split)

                embeddings, _ = self.embedding_module(condition=condition)
                cond_embedding = embeddings[0]
                target_batch = next(loader_target)

                # Get predicted quantiles for visualization
                pred_qvalues = self.model.predict(cond_embedding, key=None, inference=True)
                
                # Compute Wasserstein distance between target and predicted quantiles.
                # In the infinite-data regime the model is trained on the theoretical
                # mixture quantiles, so score it against those too (otherwise the metric
                # compares against finite-sample quantiles and reports a spurious gap).
                if self.infinite_data_gaussian:
                    _, target_qvalues = self._compute_theoretical_quantiles(cond_embedding)
                else:
                    target_qvalues = self._compute_qvalues(target_batch)

                # Align shapes before differencing. _compute_qvalues returns (n_qs, 1)
                # for the (n_samples, 1) target batch, while pred is (n_qs,); subtracting
                # them directly broadcasts to (n_qs, n_qs) and massively inflates the
                # reported distance. Flatten both to (n_qs,) so we compare like for like.
                pred_qvalues = jnp.reshape(pred_qvalues, (-1,))
                target_qvalues = jnp.reshape(target_qvalues, (-1,))

                # For each dimension, compute q and then L1 distance
                w1_dist = float(
                    jnp.sum(jnp.abs(target_qvalues - pred_qvalues))
                    * (1 / self.model.n_qs)
                )
                w2_dist = float(jnp.sqrt(jnp.mean((target_qvalues - pred_qvalues) ** 2))
                )

                results[f"{split}-{condition}"] = {
                    "target": target_batch,
                    "predicted_qvals": pred_qvalues,  # [data_dim, n_qs]
                    "target_qvals": target_qvalues,  # [data_dim, n_qs]
                    "w1_distance": w1_dist,
                    "w2_distance": w2_dist,
                }

        # Accumulate results in metrics dictionary
        if "evaluation" not in self.metrics:
            self.metrics["evaluation"] = {}
        self.metrics["evaluation"].update(results)

        return results


@eqx.filter_jit
def _loss_fn(model, cond, target_qs, target_qvals):
    pred_qvals = jax.vmap(lambda c: model.predict(c), in_axes=0)(
        cond,
    )
    pred_qvals = pred_qvals.reshape(
        target_qs.shape[0], target_qs.shape[1]
    )  # [n_conditions, n_qs, data_dim]
    wasserstein2_loss = jnp.sqrt(jnp.mean((pred_qvals - target_qvals.squeeze()) ** 2))
    return wasserstein2_loss


@eqx.filter_jit
def _loss_fn_train(model, cond, target_qs, target_qvals, key):
    keys = jr.split(key, cond.shape[0])
    pred_qvals = jax.vmap(
        lambda c, k: model.predict(c, key=k, inference=False), in_axes=0
    )(cond, keys)
    pred_qvals = pred_qvals.reshape(
        target_qs.shape[0], target_qs.shape[1]
    )  # [n_conditions, n_qs]

    wasserstein2_loss = jnp.sqrt(jnp.mean((pred_qvals - target_qvals.squeeze()) ** 2))
    return wasserstein2_loss


@eqx.filter_jit
def _train_step(model, opt_state, cond, target_qs, target_qvals, key, optimizer):
    loss, grads = eqx.filter_value_and_grad(_loss_fn_train)(
        model, cond, target_qs, target_qvals, key
    )
    updates, opt_state = optimizer.update(grads, opt_state, model)
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def _theoretical_mixture_mean(condition):
    """Theoretical mean of the 3-component Gaussian mixture for a condition.

    Mirrors ``_compute_theoretical_quantiles``: build the mixture from the
    condition params and read off the (population) mean. By construction this
    equals the enforced weighted-mean constraint (0 for sim_mixture*), so it
    carries no per-condition signal -- which is the whole point in the
    infinite-data regime.
    """
    mu1, std1, prop1, mu2, std2, prop2, mu3, std3, prop3 = jnp.asarray(condition).tolist()
    components = [Normal(mu=mu1, sigma=std1), Normal(mu=mu2, sigma=std2), Normal(mu=mu3, sigma=std3)]
    mixture = Mixture(weights=[prop1, prop2, prop3], components=components)
    return jnp.asarray([mixture.mean()])


def _mixture_from_condition(condition):
    mu1, std1, prop1, mu2, std2, prop2, mu3, std3, prop3 = jnp.asarray(condition).tolist()
    components = [Normal(mu=mu1, sigma=std1), Normal(mu=mu2, sigma=std2), Normal(mu=mu3, sigma=std3)]
    return Mixture(weights=[prop1, prop2, prop3], components=components)


def _theoretical_mixture_meanvar(condition):
    """Theoretical (mean, variance) of the 3-component mixture, shape (2,).

    Unlike ``_theoretical_mixture_mean`` the variance does carry per-condition
    signal in the sim_mixture* family (the mean is pinned to 0 by construction).
    """
    mixture = _mixture_from_condition(condition)
    return jnp.asarray([mixture.mean(), mixture.variance()])


def _theoretical_mixture_quartiles(condition, levels=(0.25, 0.5, 0.75)):
    """Theoretical quartiles of the 3-component mixture, shape (len(levels),).

    Mirrors the empirical branch in ``prepare_data_grad`` (sample quartiles), so
    the target dimension matches whatever ``get_quartile_ntk_matrix`` predicts.
    """
    mixture = _mixture_from_condition(condition)
    return jnp.asarray(mixture.icdf(list(levels))).ravel()


# ---- BALSA (distribution-disagreement) settings ----------------------------
# BALSA is the only strategy here that consumes whole cell populations instead of
# per-condition summary statistics, so it gets its own caps/defaults.
BALSA_STRATEGIES = ("balsa", "balsa_emd")
BALSA_MAX_CELLS = 4096  # per-condition cap on cells handed to the density model
BALSA_INF_N_CELLS = 1000  # cells drawn from the mixture per condition (inf-data)
BALSA_INF_N_VALID_CELLS = 250
BALSA_DROPOUT_RATE = 0.05  # the paper's best-performing MC-dropout rate


def _cond_seed(condition):
    """Stable per-condition seed (``hash()`` is salted per process, so avoid it)."""
    return int(hashlib.md5(str(condition).encode()).hexdigest()[:8], 16)


def _sample_mixture_cells(condition, n_cells, rng):
    """Draw ``n_cells`` samples from a condition's 3-component Gaussian mixture.

    Same parameterisation as ``_mixture_from_condition``, but sampled with numpy
    after renormalising the weights in float64. ``scipy.stats.Mixture.sample``
    cannot be used here: it forwards the weights to ``rng.multinomial``, whose
    ``sum(pvals[:-1]) > 1.0`` check has no tolerance, so float32 embeddings whose
    proportions sum to 1 + 1e-7 (common in sim_mixture_a5) raise ValueError -- even
    though the same weights are fine for ``Mixture.icdf``, which is what the model's
    own inf-data targets come from.
    """
    mu1, std1, p1, mu2, std2, p2, mu3, std3, p3 = jnp.asarray(condition).tolist()
    mus = np.asarray([mu1, mu2, mu3], dtype=np.float64)
    stds = np.asarray([std1, std2, std3], dtype=np.float64)
    weights = np.asarray([p1, p2, p3], dtype=np.float64)
    weights = weights / weights.sum()
    counts = rng.multinomial(n_cells, weights)
    cells = np.concatenate(
        [rng.normal(m, s, size=int(c)) for m, s, c in zip(mus, stds, counts)]
    )
    rng.shuffle(cells)
    return jnp.asarray(cells, dtype=jnp.float32).reshape(-1, 1)


def _subsample_cells(x, condition, max_cells=BALSA_MAX_CELLS):
    """Cap a condition's cell population, deterministically per condition."""
    if x.shape[0] <= max_cells:
        return x
    rng = np.random.default_rng(_cond_seed(condition))
    return x[jnp.asarray(rng.choice(x.shape[0], size=max_cells, replace=False))]


def prepare_data_grad(trainer, datamodule, conditions, inf_data=False, strategy="ntk"):
    """Prepare data for gradient-based kernel computation.

    Args:
        trainer: ConditionalQuantileTrainer with embedding_module
        datamodule: ConditionalDataModule instance
        conditions: List of condition strings
        inf_data: If True, use the theoretical mixture mean (from the condition
            params) instead of the empirical sample mean, so the mean targets
            match the infinite-data regime the model is trained/scored on.
        strategy: which kernel the stats are being prepared for. The ``*_ntk``
            strategies want a *summary* per condition (mean / quartiles /
            mean-variance), whereas ``balsa*`` needs the raw sample itself, since it
            fits a full conditional density model -- see the branches below.

    Returns:
        x0: Source samples [n_samples, data_dim]
        cond_arr_dict: {condition: embedding (1, emb_dim)}
        stat_train_data: {condition: statistics of train-split target cells}, or the
            train-split samples themselves ``(n_cells, data_dim)`` for ``balsa*``
        stat_valid_data: {condition: statistics of valid-split target cells}, or the
            valid-split samples themselves for ``balsa*``
        quantile_target_data: {condition: target quantiles (data_dim, n_qs)} at the
            model's quantile levels, used as y_z for the fisher-style score
            kernels ("grad_score"/"mean_score"). Theoretical (mixture icdf) in the
            inf_data regime, empirical (sample quantiles) otherwise -- never None.
    """
    x0 = None
    cond_arr_dict = {}
    stat_train_data = {}
    stat_valid_data = {}
    quantile_target_data = {}

    # Quantile levels match _compute_target_quantiles / the model's prediction grid.
    n_qs = getattr(getattr(trainer, "model", None), "n_qs", None) or trainer.config.get("n_qs", 32)
    q_levels = jnp.linspace(1 / (2 * n_qs), 1 - 1 / (2 * n_qs), n_qs)

    for c in tqdm(conditions, desc="Preparing condition data", disable=not VERBOSE):
        loader = datamodule.loaders[c]
        loader_source, _ = loader.get_loaders_by_type("train")

        embeddings, _ = trainer.embedding_module(condition=c)

        if x0 is None:
            x0 = next(loader_source)

        cond_arr_dict[c] = embeddings

        if inf_data:
            # Theoretical mean and quantiles straight from the mixture params --
            # no finite sample needed (the inf-data regime has none to rely on).
            if strategy in BALSA_STRATEGIES:
                # No finite sample exists in the inf-data regime, so draw the cells
                # BALSA's density model needs straight from the mixture. Seeded per
                # condition so the same condition always yields the same draw.
                _rng = np.random.default_rng(_cond_seed(c))
                stat_train_data[c] = _sample_mixture_cells(
                    embeddings[0, :], BALSA_INF_N_CELLS, _rng
                )
                stat_valid_data[c] = _sample_mixture_cells(
                    embeddings[0, :], BALSA_INF_N_VALID_CELLS, _rng
                )
            elif strategy == "ntk":
                theo_mean = _theoretical_mixture_mean(embeddings[0, :])
                stat_train_data[c] = theo_mean
                stat_valid_data[c] = theo_mean
            elif strategy == "quartile_ntk":
                theo_quartiles = _theoretical_mixture_quartiles(embeddings[0, :])
                stat_train_data[c] = theo_quartiles
                stat_valid_data[c] = theo_quartiles
            elif strategy == "meanvar_ntk":
                theo_meanvar = _theoretical_mixture_meanvar(embeddings[0, :])
                stat_train_data[c] = theo_meanvar
                stat_valid_data[c] = theo_meanvar
            _, qvals = trainer._compute_theoretical_quantiles(embeddings[0, :], n_qs=n_qs)
            quantile_target_data[c] = jnp.asarray(qvals).reshape(1, -1)  # (1, n_qs)
        else:
            train_x = jnp.asarray(loader.target_adata[loader.target_train_cells].X)
            valid_x = jnp.asarray(loader.target_adata[loader.target_valid_cells].X)
            if getattr(loader, "reduction", None):
                train_x = loader.encoder(train_x)
                valid_x = loader.encoder(valid_x)
            if strategy in BALSA_STRATEGIES:
                # BALSA fits p(y | z_c) on the cells themselves, so hand over the
                # (optionally subsampled) populations rather than a summary.
                stat_train_data[c] = _subsample_cells(train_x, c)
                stat_valid_data[c] = (
                    _subsample_cells(valid_x, c) if len(valid_x) > 0
                    else stat_train_data[c]
                )
            elif strategy == "ntk":
                stat_train_data[c] = train_x.mean(axis=0)
                stat_valid_data[c] = valid_x.mean(axis=0) if len(valid_x) > 0 else train_x.mean(axis=0)
            elif strategy == "quartile_ntk":
                # Flatten to 1-D (n_levels * data_dim,) so the target shape matches
                # the predictor output; for data_dim=1 this is just the 3 quartiles.
                _quartiles = lambda x: jnp.quantile(x, jnp.array([0.25, 0.5, 0.75]), axis=0).ravel()
                stat_train_data[c] = _quartiles(train_x)
                stat_valid_data[c] = _quartiles(valid_x) if len(valid_x) > 0 else _quartiles(train_x)
            elif strategy == "meanvar_ntk":
                _meanvar = lambda x: jnp.concatenate([jnp.mean(x, axis=0), jnp.var(x, axis=0)])
                stat_train_data[c] = _meanvar(train_x)
                stat_valid_data[c] = _meanvar(valid_x) if len(valid_x) > 0 else _meanvar(train_x)
            # Empirical target quantiles y_z: (n_qs, data_dim) -> (data_dim, n_qs).
            quantile_target_data[c] = jnp.quantile(train_x, q_levels, axis=0).T

    return x0, cond_arr_dict, stat_train_data, stat_valid_data, quantile_target_data


def get_tntk_kernel_matrix(
    state,
    cond_arr_dict,
):
    """Compute gradient-based kernel matrix.

    This computes the Jacobian of the flow transformation with respect
    to the conditioning variable, similar to the Monge approach.

    NotImplemented: Jacobian computation through flow layers.
    """
    conditions = list(cond_arr_dict.keys())
    cond_z = {}
    for condition in conditions:
        z = state.get_hidden(cond_arr_dict[condition][0, :], key=None, inference=True)

        Jmw, Jmb, Jdw, Jdb = jax.jacobian(state.pred_from_params, argnums=[0, 1, 2, 3])(state.final_layer_min.weight, state.final_layer_min.bias, state.final_layer_deltas.layers[0].weight, state.final_layer_deltas.layers[0].bias, z)
        Jm = jnp.concatenate([Jmw, Jmb[...,None]], axis=-1)
        Jd = jnp.concatenate([Jdw, Jdb[...,None]], axis=-1)
        J = jnp.concatenate([Jm, Jd], axis=-2)
        cond_z[condition] = J.ravel()

    cond_tensor = jnp.stack(
        [cond_z[condition] for condition in conditions], axis=0
    )  # [n_conditions, n_bins * hidden_dim]
    kernel_matrix = cond_tensor @ cond_tensor.T / state.n_qs

    return kernel_matrix, None




def get_kernels_quantile(
    cond_arr_dict,
    state,
    kernel_type: Literal["tntk"],
    key=None,
    sketch_dim=512,
    quantile_target_dict=None,
):
    """Get kernel matrices for active learning.

    Args:
        cond_arr_dict: Dictionary mapping conditions to embeddings
        state: Model state
        kernel_type: Type of kernel to compute

    Returns:
        kernel_matrix, kernel_distance_matrix, cond_z
    """
    if kernel_type == "tntk":
        kernel_matrix, cond_z = get_tntk_kernel_matrix(state, cond_arr_dict)
        kernel_distance_matrix = kernel_distance(kernel_matrix)
    else:
        raise ValueError(f"Unknown kernel type {kernel_type}")

    return kernel_matrix, kernel_distance_matrix, cond_z


def get_batch_quantile(
    all_conditions,
    unseen_conditions,
    trained_conditions,
    strategy,
    batch_size,
    state=None,
    cond_arr_dict=None,
    x0=None,
    random_seed=42,
    batch_selection_strategy: Literal[
        "coreset", "uncertainty", "kmeanspp", "lcmd", "typiclust2"
    ] = "coreset",
    tmp_save_path=None,
    stat_dict=None,
    stat_valid_dict=None,
    quantile_target_dict=None,
    hidden_dims=None,
    balsa_dropout_rate=BALSA_DROPOUT_RATE,
    balsa_dropout_trials=25,
    balsa_n_samples=64,
    acquisition_seed=None,
):
    """Select next batch for active learning.

    Similar to get_batch_cmonge but for histogram predictors.

    The ``balsa``/``balsa_emd`` strategies expect ``stat_dict``/``stat_valid_dict`` to
    hold whole cell populations per condition (what ``prepare_data_grad`` produces
    for those strategies) rather than summary statistics.
    """
    if strategy in NO_KERNEL_STRATEGIES:
        # `random` is the legacy frozen-seed baseline, `random_fixed` the
        # corrected uniform one -- see acpl.selection.sample_random_batch.
        batch = sample_random_batch(
            unseen_conditions,
            batch_size,
            strategy=strategy,
            random_seed=random_seed,
            acquisition_seed=acquisition_seed,
        )
        kernel_mat = None
        kernel_distance_mat = None
        cond_z = None
    else:
        if strategy in ("ntk", "meanvar_ntk", "quartile_ntk"):
            # mean_dict already holds the theoretical (inf-data) or empirical mean,
            # built upstream in prepare_data_grad, so no inf_data flag is needed here.
            # "ntk" uses the hidden-feature Gram (pure input geometry); "ntk_score"
            # uses the residual-weighted score embedding (y - f) . d f/d theta.
            if strategy == "ntk":
                kernel_mat, mean_pred_state = get_ntk_matrix(cond_arr_dict, stat_dict, stat_valid_dict, trained_conditions, all_conditions, hidden_dims=hidden_dims)
            elif strategy == "quartile_ntk":
                kernel_mat, mean_pred_state = get_quartile_ntk_matrix(cond_arr_dict, stat_dict, stat_valid_dict, trained_conditions, all_conditions, hidden_dims=hidden_dims)
            elif strategy == "meanvar_ntk":
                kernel_mat, mean_pred_state = get_meanvar_ntk_matrix(cond_arr_dict, stat_dict, stat_valid_dict, trained_conditions, all_conditions, hidden_dims=hidden_dims)
            kernel_distance_mat = None
            cond_z = None

            # Evaluate mean predictor on unseen (unlabeled) conditions and log results
            cond_dict = {
                c: (cond_arr_dict[c][0, :] if cond_arr_dict[c].ndim > 1 else cond_arr_dict[c])
                for c in unseen_conditions
            }
            mean_pred_results = evaluate_mean_predictor(
                mean_pred_state, unseen_conditions, cond_dict, stat_dict
            )
            mean_w2 = float(np.mean([r["mse"] ** 0.5 for r in mean_pred_results]))
            mean_r2 = float(np.mean([r["r2"] for r in mean_pred_results]))
            mean_pearson = float(np.mean([r["pearson_r"] for r in mean_pred_results]))
            logger.debug(
                f"MeanPredictor on unseen conditions ({len(unseen_conditions)}): "
                f"avg RMSE={mean_w2:.4f}  avg R2={mean_r2:.4f}  avg Pearson={mean_pearson:.4f}"
            )
            if tmp_save_path:
                with open(f"{tmp_save_path}/mean_predictor_eval.pkl", "wb") as _f:
                    pkl.dump({"results": mean_pred_results, "summary": {"avg_rmse": mean_w2, "avg_r2": mean_r2, "avg_pearson": mean_pearson}}, _f)
        elif strategy in BALSA_STRATEGIES:
            # BALSA: fit a conditional normalizing flow with a dropout encoder on the
            # labelled conditions' cells, then score every condition by how much the
            # MC-dropout ensemble of predictive distributions disagrees with itself.
            # "balsa" returns the PSD quantile-deviation Gram (usable with any batch
            # selection strategy); "balsa_emd" returns diag(sum_i EMD(y'_i, y'_i+1)),
            # the paper's literal score, which reproduces its top-b acquisition when
            # paired with --batch-selection-strategy uncertainty.
            kernel_mat, balsa_aux = get_balsa_kernel_matrix(
                cond_arr_dict,
                stat_dict,
                stat_valid_dict,
                trained_conditions,
                all_conditions,
                hidden_dims=hidden_dims,
                kernel_mode="emd" if strategy == "balsa_emd" else "quantile",
                dropout_rate=balsa_dropout_rate,
                dropout_trials=balsa_dropout_trials,
                n_samples=balsa_n_samples,
                random_seed=random_seed,
            )
            kernel_distance_mat = None
            cond_z = None

            _mean = lambda s: float(np.mean(s)) if len(s) else float("nan")
            _unseen_mean = _mean([balsa_aux["scores"][c] for c in unseen_conditions])
            _seen_mean = _mean([balsa_aux["scores"][c] for c in trained_conditions])
            logger.debug(
                f"BALSA ({strategy}, k={balsa_dropout_trials}, "
                f"dropout={balsa_dropout_rate}): mean EMD disagreement "
                f"unseen={_unseen_mean:.4f} seen={_seen_mean:.4f}"
            )
            if tmp_save_path:
                with open(f"{tmp_save_path}/balsa_eval.pkl", "wb") as _f:
                    pkl.dump(
                        {
                            "scores": balsa_aux["scores"],
                            "losses": balsa_aux["losses"],
                            "kernel_mode": balsa_aux["kernel_mode"],
                            "emd_mode": balsa_aux["emd_mode"],
                            "dropout_rate": balsa_aux["dropout_rate"],
                            "dropout_trials": balsa_aux["dropout_trials"],
                            "seen_conditions": trained_conditions,
                        },
                        _f,
                    )
        else:
            kernel_mat, kernel_distance_mat, cond_z = get_kernels_quantile(
                cond_arr_dict, state, kernel_type=strategy, key=jr.PRNGKey(random_seed),
                quantile_target_dict=quantile_target_dict,
            )

            # Check for NaN/Inf
            if jnp.any(jnp.isnan(kernel_mat)) or jnp.any(jnp.isinf(kernel_mat)):
                logger.error("Kernel matrix contains NaN or Inf values")
                # Save for debugging
                if tmp_save_path:
                    with open(f"{tmp_save_path}/failed_kernel_nan_inf.pkl", "wb") as f:
                        pkl.dump(
                            {
                                "kernel_matrix": kernel_mat,
                                "kernel_distance_matrix": kernel_distance_mat,
                                "conditions": all_conditions,
                            },
                            f,
                        )
                raise ValueError("Kernel matrix contains NaN or Inf values")

        batch = select_batch(
            trained_conditions=trained_conditions,
            all_conditions=all_conditions,
            batch_size=min(batch_size, len(unseen_conditions)),
            kernel_matrix=kernel_mat,
            batch_selection_strategy=batch_selection_strategy,
            random_seed=random_seed,
        )

    return batch, kernel_mat, kernel_distance_mat, cond_z


def run_single_iteration(
    config_path: str,
    train_conditions: list,
    test_conditions: list,
    strategy: str = "random",
    lr: float = 1e-3,
    output_dir: str = None,
    run_id: str = "test_run",
    evaluate: bool = True,
    checkpoint_dir: str = None,
    rerun: bool = False,
    reeval: bool = False,
    dim_hidden: list = None,
    n_bins: int = 100,
    nn_width: int = 50,
    nn_depth: int = 1,
    dropout_rate: float = 0.0,
    weight_decay: float = 0.0,
    validate_with_conditions: bool = False,
    validate_with_folds: bool = False,
    validate_conditions_seed: int = 0,
    n_qs: int = None,
    inf_data: bool = False,
):
    """Run a single training iteration.

    Args:
        config_path: Path to config file
        train_conditions: Conditions to train on
        test_conditions: Conditions to test on
        strategy: AL strategy
        lr: Learning rate
        output_dir: Output directory
        run_id: Run identifier
        evaluate: Whether to evaluate
        checkpoint_dir: Checkpoint directory
        rerun: Force retraining
        reeval: Force re-evaluation
        dim_hidden: Hidden dimensions (not used)
        n_bins: Number of histogram bins
        nn_width: Width of MLP networks
        nn_depth: Depth of MLP networks

    Returns:
        Dictionary with trainer, model, logs, etc.
    """
    if output_dir is None:
        output_dir = "/tmp/acpl_hist"
    os.makedirs(output_dir, exist_ok=True)

    if checkpoint_dir is None:
        checkpoint_dir = f"{output_dir}/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    # Load config (reuse from cmonge)
    train_config = load_config_util(config_path)

    if validate_with_conditions:
        if validate_with_folds:
            folds = list(KFold(n_splits=5, shuffle=True, random_state=0).split(train_conditions))
            train_idx, valid_idx = folds[validate_conditions_seed]
        else:
            train_idx, valid_idx = train_test_split(
                list(range(len(train_conditions))),
                test_size=5,
                random_state=validate_conditions_seed,
            )
        train_conditions_arr = [train_conditions[i] for i in train_idx]
        valid_conditions = [train_conditions[i] for i in valid_idx]
        train_conditions = train_conditions_arr
        train_config.condition.valid_conditions = valid_conditions
        train_config.condition.ood = valid_conditions
        train_config.condition.split = [1.0, 0.0, 0.0]

    train_config.condition.conditions = train_conditions
    if validate_with_conditions:
        # Include valid_conditions so splitter loads them; ood check assigns split [0,1,0]
        train_config.condition.conditions = train_conditions + valid_conditions
    train_config.model.optim.lr = lr
    train_config.data.batch_size = 500
    logger.debug(f"Train config: {train_config}")

    # Setup test config
    test_config = load_config_util(config_path)
    test_config.condition.conditions = test_conditions
    test_config.data.split = [0.0, 0.0, 1.0]
    test_config.condition.split = [0.0, 0.0, 1.0]
    test_config.data.batch_size = 500

    # Create data modules (reuse ConditionalDataModule from cmonge!)
    train_datamodule = ConditionalDataModule(train_config.data, train_config.condition)
    train_datamodule.name = run_id
    test_datamodule = ConditionalDataModule(test_config.data, test_config.condition)

    logger.debug(f"Training on {len(train_conditions)} conditions: {train_conditions}")
    logger.debug(f"Testing on {len(test_conditions)} conditions: {test_conditions}")

    # Get condition dimension from config or embedding
    cond_dim = train_config.model.mlp["dim_cond"]  # Default condition embedding size
    if not cond_dim:
        raise ValueError(
            f"Condition embedding dimension (dim_cond) must be specified in config.model.mlp.dim_cond. Current config: {train_config.model.mlp}"
        )

    # Create trainer with histogram-specific config
    logger_path = f"{output_dir}/{run_id}_log.pkl"
    hist_config = {
        "lr": lr,
        "n_bins": train_config.model.get("n_bins", 32),
        "nn_width": nn_width,
        "nn_depth": nn_depth,
        "dropout_rate": dropout_rate,
        "weight_decay": weight_decay,
        "num_epochs": 10000,  # train_config.model.get("num_train_iters", 1000),
        "min_epochs": 1000,
        "patience": 10,
        "batch_size": min(
            train_config.data.batch_size, 32
        ),  # Smaller batch for histogram training
        "val_prop": 0.1,
        "seed": train_config.data.seed,
        "max_samples_per_condition": 1000,
        "infinite_data_gaussian": inf_data,
    }
    hist_config.update(train_config.model)
    if n_qs is not None:
        hist_config["n_qs"] = n_qs
    logger.debug(f"Histogram config: {hist_config}")

    trainer = ConditionalQuantileTrainer(
        cond_dim=cond_dim,
        config=hist_config,
        logger_path=logger_path,
        datamodule=train_datamodule,
    )

    # Setup model
    key = jr.PRNGKey(hist_config["seed"])
    model = trainer.setup_model(key)

    # Train or load checkpoint
    if not os.path.exists(f"{checkpoint_dir}/model.eqx") or rerun:
        logger.debug("Starting training...")
        model, logs, log_step = trainer.train()
        logger.debug(f"Training completed at step {log_step}")

        # Save checkpoint using Equinox serialization
        checkpoint_path = f"{checkpoint_dir}/model.eqx"
        eqx.tree_serialise_leaves(checkpoint_path, model)
        logger.debug(f"Checkpoint saved to {checkpoint_path}")
    else:
        try:
            # Load checkpoint using Equinox deserialization
            checkpoint_path = f"{checkpoint_dir}/model.eqx"
            # Create model structure first (template)
            template_model = trainer.setup_model(key)
            # Load parameters into template
            model = eqx.tree_deserialise_leaves(checkpoint_path, template_model)
            trainer.model = model
            logger.debug(f"Loaded existing checkpoint from {checkpoint_path}")
            logs = None
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, retraining...")
            model, logs, log_step = trainer.train()
            checkpoint_path = f"{checkpoint_dir}/model.eqx"
            eqx.tree_serialise_leaves(checkpoint_path, model)

    result = {
        "trainer": trainer,
        "model": model,
        "train_datamodule": train_datamodule,
        "test_datamodule": test_datamodule,
        "logs": logs,
        "config": train_config,
        "eval_results": None,
    }

    # Evaluate
    eval_path = logger_path.replace("_log.pkl", "_trainer_metrics.pkl")
    # Retraining (rerun) invalidates any existing metrics, so force re-eval too.
    if evaluate and (not os.path.exists(eval_path) or reeval or rerun):
        logger.debug("Running evaluation on test set...")
        trainer.evaluate(test_datamodule, splits=["test"])
        logger.debug("Running evaluation on train/valid sets...")
        trainer.evaluate(train_datamodule, splits=["train", "valid"])

        # Save all evaluation results to a single pkl file
        with open(eval_path, "wb") as f:
            pkl.dump(trainer.metrics, f)
        logger.debug(f"Saved evaluation results to {eval_path}")

    logger.debug(f"Results saved to {output_dir}")
    return result


def run_al_loop(
    config_path: str,
    data: str,
    strategy: str = "random",
    batch_selection_strategy: str = "kmeanspp",
    batch_size: int = 20,
    init_batch_size: int = 20,
    init_seed: int = 0,
    split_seed: int = 0,
    max_iter: int = 3,
    lr: float = 1e-4,
    prefix: str = "al_hist",
    rerun: bool = False,
    reeval: bool = False,
    n_bins: int = 32,
    nn_width: int = 50,
    nn_depth: int = 1,
    dropout_rate: float = 0.0,
    weight_decay: float = 0.0,
    validate_with_conditions: bool = False,
    validate_with_folds: bool = False,
    validate_conditions_seed: int = 0,
    n_qs: int = None,
    inf_data: bool = False,
):
    """Run the full active learning loop with conditional histogram prediction.

    Args:
        config_path: Path to config YAML
        data: Dataset name
        strategy: AL strategy ('random', 'tntk')
        batch_selection_strategy: Batch selection strategy
        batch_size: Batch size for AL iterations
        init_batch_size: Initial batch size
        init_seed: Random seed for initialization
        split_seed: Random seed for K-fold splits
        max_iter: Maximum AL iterations
        lr: Learning rate
        prefix: Prefix for output directory
        rerun: Force retraining
        reeval: Force re-evaluation
        n_bins: Number of histogram bins
        nn_width: int = 50,
        nn_depth: int = 1,
    """
    # Load conditions
    conditions = pd.read_csv(condition_map[data], index_col=None, header=None)[
        0
    ].tolist()
    if control_map[data] in conditions:
        conditions.remove(control_map[data])

    logger.info(
        f"=== {data} | strategy={strategy} | seed={init_seed} | "
        f"batch={batch_size} | {len(conditions)} conditions | {max_iter} AL iters ==="
    )

    if batch_size > init_batch_size and init_batch_size != 1:
        init_batch_size = batch_size

    kf = KFold(n_splits=5, random_state=split_seed, shuffle=True)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(conditions)):
        output_dir = f"{REPO_ROOT}/models/quantile-al/{data}/{prefix}/"
        os.makedirs(output_dir, exist_ok=True)

        bss_suffix = (
            f"_{batch_selection_strategy}"
            if batch_selection_strategy != "coreset"
            else ""
        )
        exp_id = f"al0_{strategy}{bss_suffix}_size{batch_size}_fold{fold_idx}_seed{init_seed}"

        # Check if already completed
        last_iter_log = f"{output_dir}/{exp_id}_iter{max_iter}_log.pkl"
        if os.path.exists(last_iter_log) and not rerun and not reeval:
            logger.debug(f"Fold {fold_idx} already completed, skipping...")
            continue

        # Log split
        log_path = f"{output_dir}/{exp_id}.json"
        with open(log_path, "w") as f:
            json.dump(
                {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}, f
            )

        shared_train_val_conditions = [conditions[i] for i in train_idx]
        heldout_test_conditions = [conditions[i] for i in test_idx]

        # Prepare condition embeddings for non-random strategies
        if strategy not in NO_KERNEL_STRATEGIES:
            logger.debug("Preparing condition embeddings for active learning...")
            # Create temporary datamodule to get embeddings
            next_config = load_config_util(config_path)
            next_config.condition.conditions = shared_train_val_conditions
            next_config.model.embedding.checkpoint = True
            next_datamodule = ConditionalDataModule(
                next_config.data, next_config.condition
            )

            cond_dim = next_config.model.get("cond_dim", 64)

            logger_path = f"{output_dir}/temp_embedding_log"
            temp_trainer = ConditionalQuantileTrainer(
                cond_dim=cond_dim,
                config={"seed": 0, "n_bins": next_config.model.get("n_bins", 100)},
                logger_path=logger_path,
                datamodule=next_datamodule,
            )

            # Setup embedding module

            embedding_type = next_config.model.embedding.get("type", "moa")
            embedding_class = EmbeddingFactory.get(embedding_type)

            if embedding_class is not None:
                temp_trainer.embedding_module = embedding_class(
                    datamodule=next_datamodule,
                    **next_config.model.embedding,
                )
            else:
                logger.warning("No embedding module available, using random strategy")
                strategy = "random"

            if strategy not in NO_KERNEL_STRATEGIES:
                x0, cond_arr_dict, stat_dict, stat_valid_dict, quantile_target_dict = prepare_data_grad(
                    temp_trainer, next_datamodule, shared_train_val_conditions, inf_data=inf_data,
                    strategy=strategy,
                )
            else:
                x0, cond_arr_dict, stat_dict, stat_valid_dict, quantile_target_dict = None, None, None, None, None
        else:
            cond_arr_dict = None
            x0 = None
            stat_dict = None
            stat_valid_dict = None
            quantile_target_dict = None

        # AL loop
        unlabeled_conditions = shared_train_val_conditions.copy()
        train_conditions = []
        net = None

        # Resume from the last completed checkpoint if available
        last_completed_iter = -1
        for _i in range(max_iter + 1):
            _run_id = f"{exp_id}_iter{_i}"
            if (
                os.path.exists(f"{output_dir}/{_run_id}/model.eqx")
                and os.path.exists(f"{output_dir}/{_run_id}_kernels.pkl")
                and not rerun
            ):
                last_completed_iter = _i
            else:
                break

        if last_completed_iter >= 0:
            logger.debug(
                f"Fold {fold_idx}: found completed iters 0-{last_completed_iter}, reconstructing state..."
            )
            for _i in range(last_completed_iter + 1):
                _run_id = f"{exp_id}_iter{_i}"
                with open(f"{output_dir}/{_run_id}_kernels.pkl", "rb") as f:
                    _kernels_data = pkl.load(f)
                _batch = _kernels_data["selected_conditions"]
                for _x in _batch:
                    unlabeled_conditions.remove(_x)
                train_conditions = train_conditions + _batch
                if reeval:
                    # Eval-only pass: load this iter's existing checkpoint and
                    # rewrite trainer_metrics.pkl. Reuses the saved selections
                    # (kernels.pkl), so no batch re-selection occurs.
                    logger.debug(
                        f"  Re-evaluating fold {fold_idx} iter {_i} on existing checkpoint..."
                    )
                    _re = run_single_iteration(
                        config_path=config_path,
                        train_conditions=train_conditions,
                        test_conditions=heldout_test_conditions,
                        strategy=strategy,
                        lr=lr,
                        output_dir=output_dir,
                        run_id=_run_id,
                        evaluate=True,
                        checkpoint_dir=f"{output_dir}/{_run_id}",
                        rerun=False,
                        reeval=True,
                        n_bins=n_bins,
                        nn_width=nn_width,
                        nn_depth=nn_depth,
                        dropout_rate=dropout_rate,
                        weight_decay=weight_decay,
                        validate_with_conditions=validate_with_conditions,
                        validate_with_folds=validate_with_folds,
                        validate_conditions_seed=validate_conditions_seed,
                        n_qs=n_qs,
                        inf_data=inf_data,
                    )
                    del _re
                    jax.clear_caches()
            logger.debug(
                f"  Restored train_conditions ({len(train_conditions)}), "
                f"unlabeled_conditions ({len(unlabeled_conditions)})"
            )
            _resume_run_id = f"{exp_id}_iter{last_completed_iter}"
            logger.debug(
                f"  Loading net from checkpoint: {output_dir}/{_resume_run_id}/model.eqx"
            )
            _resume_result = run_single_iteration(
                config_path=config_path,
                train_conditions=train_conditions,
                test_conditions=heldout_test_conditions,
                strategy=strategy,
                lr=lr,
                output_dir=output_dir,
                run_id=_resume_run_id,
                evaluate=False,
                checkpoint_dir=f"{output_dir}/{_resume_run_id}",
                rerun=False,
                n_bins=n_bins,
                nn_width=nn_width,
                nn_depth=nn_depth,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                validate_with_conditions=validate_with_conditions,
                validate_with_folds=validate_with_folds,
                validate_conditions_seed=validate_conditions_seed,
                n_qs=n_qs,
                inf_data=inf_data
            )
            net = _resume_result["model"]
            del _resume_result
            logger.debug(f"  Net loaded successfully from iter {last_completed_iter}")

        for al_iter in range(max_iter + 1):
            if al_iter <= last_completed_iter and not rerun:
                logger.debug(
                    f"  Skipping fold {fold_idx} iter {al_iter} (already completed)"
                )
                continue
            if len(unlabeled_conditions) == 0:
                break

            logger.info(f"  fold {fold_idx+1}/5 · iter {al_iter}/{max_iter} · training…")
            run_id = f"{exp_id}_iter{al_iter}"
            checkpoint_dir = f"{output_dir}/{run_id}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Select batch
            if al_iter == 0:
                batch, kernel_mat, kernel_distance_mat, cond_z = get_batch_quantile(
                    all_conditions=shared_train_val_conditions,
                    unseen_conditions=unlabeled_conditions,
                    trained_conditions=[],
                    strategy="random",
                    batch_size=init_batch_size,
                    random_seed=init_seed,
                    tmp_save_path=checkpoint_dir,
                )
            else:
                batch, kernel_mat, kernel_distance_mat, cond_z = get_batch_quantile(
                    all_conditions=shared_train_val_conditions,
                    unseen_conditions=unlabeled_conditions,
                    trained_conditions=train_conditions,
                    strategy=strategy,
                    batch_size=batch_size,
                    state=net,
                    cond_arr_dict=cond_arr_dict,
                    x0=x0,
                    batch_selection_strategy=batch_selection_strategy,
                    tmp_save_path=checkpoint_dir,
                    stat_dict=stat_dict,
                    stat_valid_dict=stat_valid_dict,
                    quantile_target_dict=quantile_target_dict,
                    # Match NTK MeanPredictor hidden dims to the quantile
                    # distribution predicting model (width nn_width, depth nn_depth).
                    # BALSA reuses the same list for its dropout encoder trunk.
                    hidden_dims=[nn_width] * nn_depth + [n_qs],
                    # Mirrors BALD_Point.inject_config: raise the model's dropout to
                    # at least the paper's 0.05 so MC dropout has something to sample.
                    balsa_dropout_rate=max(dropout_rate, BALSA_DROPOUT_RATE),
                    acquisition_seed=acquisition_seed_for(
                        init_seed, fold_idx, al_iter
                    ),
                )

            # Save kernel info
            with open(f"{output_dir}/{run_id}_kernels.pkl", "wb") as f:
                pkl.dump(
                    {
                        "x0": x0,
                        # "cond_arr_dict": cond_arr_dict,
                        # "cond_z": cond_z,
                        "seen_conditions": train_conditions,
                        "selected_conditions": batch,
                        "kernel_matrix": kernel_mat,
                        "kernel_distance_matrix": kernel_distance_mat,
                    },
                    f,
                )

            # Update conditions
            for x in batch:
                unlabeled_conditions.remove(x)
            train_conditions = train_conditions + batch

            # Run single iteration
            result = run_single_iteration(
                config_path=config_path,
                train_conditions=train_conditions,
                test_conditions=heldout_test_conditions,
                strategy=strategy,
                lr=lr,
                output_dir=output_dir,
                run_id=run_id,
                evaluate=True,
                checkpoint_dir=checkpoint_dir,
                rerun=rerun,
                reeval=reeval,
                n_bins=n_bins,
                nn_width=nn_width,
                nn_depth=nn_depth,
                dropout_rate=dropout_rate,
                weight_decay=weight_decay,
                validate_with_conditions=validate_with_conditions,
                validate_with_folds=validate_with_folds,
                validate_conditions_seed=validate_conditions_seed,
                n_qs=n_qs,
                inf_data=inf_data,
            )

            net = result["model"]
            mean_w2 = _mean_test_w2(result["trainer"])
            del result
            jax.clear_caches()

            logger.info(
                f"  ✓ fold {fold_idx+1}/5 · iter {al_iter}/{max_iter} · "
                f"train={len(train_conditions)} conds · test w2={mean_w2:.4f}"
            )


if __name__ == "__main__":
    _configure_logging()
    parser = get_parser()
    parser.add_argument(
        "--n-bins",
        type=int,
        default=32,
        help="Number of histogram bins per dimension",
    )
    parser.add_argument(
        "--dropout-rate",
        type=float,
        default=0.3,
        help="Dropout rate applied after each hidden layer (0 = disabled)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0,
        help="AdamW weight decay coefficient (0 = use plain Adam without decay)",
    )
    parser.add_argument(
        "--validate-with-conditions",
        action="store_true",
        help="Whether to hold out some conditions for validation instead of random split",
    )
    parser.add_argument(
        "--validate-conditions-seed",
        type=int,
        default=0,
        help="Random seed for selecting validation conditions if --validate-with-conditions is set",
    )
    parser.add_argument(
        "--inf-data",
        action="store_true",
        help="Whether to use infinite data (3-component mixture Gaussian)",
    )
    args = parser.parse_args()

    # Get model config if available
    config = model_configs[args.model_config_id]
    n_bins = config.get("n_bins", 32)
    nn_width = config.get("nn_width", 50)
    nn_depth = config.get("nn_depth", 1)
    if args.max_iter is None:
        args.max_iter = 5
    run_al_loop(
        config_path=args.config,
        data=args.data,
        strategy=args.strategy,
        batch_selection_strategy=args.batch_selection_strategy,
        batch_size=args.batch_size,
        init_batch_size=args.init_batch_size,
        init_seed=args.init_seed,
        split_seed=args.split_seed,
        max_iter=args.max_iter,
        lr=args.lr,
        prefix=args.prefix,
        rerun=args.rerun,
        reeval=args.reeval,
        n_bins=n_bins,
        nn_width=nn_width,
        nn_depth=nn_depth,
        dropout_rate=args.dropout_rate,
        weight_decay=args.weight_decay,
        validate_with_conditions=args.validate_with_conditions,
        validate_with_folds=False,
        validate_conditions_seed=args.validate_conditions_seed,
        n_qs=config.get("n_qs", n_bins),
        inf_data=args.inf_data,
    )
