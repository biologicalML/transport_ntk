from typing import Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn
import optax
from flax.training import train_state
from loguru import logger




class MeanPredictor(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int

    def setup(self):
        self.dense_layers = [nn.Dense(dim) for dim in self.hidden_dims]
        self.norm_layers = [nn.LayerNorm() for _ in self.hidden_dims]
        self.output_layer = nn.Dense(self.output_dim)

    def get_hidden(self, x):
        for dense, norm in zip(self.dense_layers, self.norm_layers):
            x = dense(x)
            x = norm(x)
            x = nn.relu(x)
        return x

    def __call__(self, x):
        x = self.get_hidden(x)
        return self.output_layer(x)

class OneDPredictor(nn.Module):
    hidden_dims: Sequence[int]
    output_dim: int

    def setup(self):
        self.dense_layers = [nn.Dense(dim) for dim in self.hidden_dims]
        self.norm_layers = [nn.LayerNorm() for _ in self.hidden_dims]
        self.output_layer = nn.Dense(self.output_dim)

    def get_hidden(self, x):
        for dense, norm in zip(self.dense_layers, self.norm_layers):
            x = dense(x)
            x = norm(x)
            x = nn.relu(x)
        return x

    def __call__(self, x):
        x = self.get_hidden(x)
        return self.output_layer(x)



def get_target_mean(datamodule, condition, split_type="train", inf_data = False):
    loader = datamodule.loaders[condition]
    cells = {
        "train": loader.target_train_cells,
        "valid": loader.target_valid_cells,
        "test": loader.target_test_cells,
    }[split_type]
    x = jnp.asarray(loader.target_adata[cells].X)
    if getattr(loader, "reduction", None):
        x = loader.encoder(x)
    return x.mean(axis=0)  # (n_features,)




def train_mean_predictor(
    model,
    train_conditions,
    emb_dict,
    mean_dict,
    val_mean_dict,
    lr=1e-3,
    wd=1e-3,
    num_epochs=200,
    patience=20,
    init_key=None,
):
    if init_key is None:
        init_key = jr.PRNGKey(0)

    embs = jnp.stack([emb_dict[c] for c in train_conditions])    # (n_train, emb_dim)
    targets = jnp.stack([mean_dict[c] for c in train_conditions])  # (n_train, n_feat)
    val_targets = jnp.stack([val_mean_dict.get(c, mean_dict[c]) for c in train_conditions])  # (n_train, n_feat)

    params = model.init(init_key, embs[0])["params"]
    state = train_state.TrainState.create(
        apply_fn=model.apply, params=params, tx=optax.adamw(lr, weight_decay=wd)
    )

    @jax.jit
    def step(state, embs, targets):
        def loss_fn(params):
            preds = jax.vmap(lambda e: state.apply_fn({"params": params}, e))(embs)
            return jnp.mean((preds - targets) ** 2)

        loss, grads = jax.value_and_grad(loss_fn)(state.params)
        return state.apply_gradients(grads=grads), loss

    best_loss = float("inf")
    best_params = state.params
    no_improve = 0

    for epoch in range(num_epochs):
        state, loss = step(state, embs, targets)
        _, val_loss = step(state, embs, val_targets)
        val = float(val_loss)
        if val < best_loss - 1e-7:
            best_loss = val
            best_params = state.params
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch + 1}, loss={best_loss:.6f}")
                break
        if (epoch + 1) % 50 == 0:
            logger.debug(f"Epoch {epoch + 1}/{num_epochs}: loss={val:.6f}")

    return state.replace(params=best_params)



def evaluate_mean_predictor(state, conditions, emb_dict, target_mean_dict):
    results = []
    for c in conditions:
        pred = state.apply_fn({"params": state.params}, emb_dict[c])
        true = target_mean_dict[c]
        mse = float(jnp.mean((pred - true) ** 2))
        ss_res = float(jnp.sum((pred - true) ** 2))
        ss_tot = float(jnp.sum((true - true.mean()) ** 2))
        r2 = 1.0 - ss_res / (ss_tot + 1e-8)
        pred_c = pred - pred.mean()
        true_c = true - true.mean()
        pearson_r = float(
            jnp.sum(pred_c * true_c)
            / (jnp.sqrt(jnp.sum(pred_c**2) * jnp.sum(true_c**2)) + 1e-8)
        )
        results.append(
            {
                "condition": c,
                "mse": mse,
                "r2": r2,
                "pearson_r": pearson_r,
                "pred_mean": jnp.array(pred),
                "true_mean": jnp.array(true),
            }
        )
    return results

def get_ntk_matrix(cond_arr_dict, mean_dict, val_mean_dict, train_conditions, all_conditions, hidden_dims=[32, 32]):
    # 1. Train model
    model = MeanPredictor(hidden_dims=hidden_dims, output_dim=mean_dict[all_conditions[0]].shape[0])
    cond_dict = {}
    for c in all_conditions:
        assert cond_arr_dict[c].ndim <= 2, cond_arr_dict[c].ndim
        if cond_arr_dict[c].ndim == 2:
            cond_dict[c] = cond_arr_dict[c][0,:]
        else:
            cond_dict[c] = cond_arr_dict[c]
    d = mean_dict[all_conditions[0]].shape[0]

    state = train_mean_predictor(model, train_conditions, cond_dict, mean_dict, val_mean_dict)

    # 2. Kernel = dot product of hidden representations over all conditions
    hidden_vecs = jnp.stack([
        model.apply({"params": state.params}, cond_dict[c], method=model.get_hidden)
        for c in all_conditions
    ])
    kernel_mat = d * hidden_vecs @ hidden_vecs.T + d
    return kernel_mat, state

def get_quartile_ntk_matrix(cond_arr_dict, quartile_dict, val_quartile_dict, train_conditions, all_conditions, hidden_dims=[32, 32]):
    # 1. Train model
    # output_dim = number of quartile targets (3 per data dim)
    model = OneDPredictor(hidden_dims=hidden_dims, output_dim=quartile_dict[all_conditions[0]].shape[0])
    cond_dict = {}
    for c in all_conditions:
        assert cond_arr_dict[c].ndim <= 2, cond_arr_dict[c].ndim
        if cond_arr_dict[c].ndim == 2:
            cond_dict[c] = cond_arr_dict[c][0,:]
        else:
            cond_dict[c] = cond_arr_dict[c]
    d = quartile_dict[all_conditions[0]].shape[0]

    state = train_mean_predictor(model, train_conditions, cond_dict, quartile_dict, val_quartile_dict)

    # 2. Kernel = dot product of hidden representations over all conditions
    hidden_vecs = jnp.stack([
        model.apply({"params": state.params}, cond_dict[c], method=model.get_hidden)
        for c in all_conditions
    ])
    kernel_mat = d * hidden_vecs @ hidden_vecs.T + d
    return kernel_mat, state

def get_meanvar_ntk_matrix(cond_arr_dict, meanvar_dict, val_meanvar_dict, train_conditions, all_conditions, hidden_dims=[32, 32]):
    # 1. Train model
    # output_dim = mean + variance per data dim
    model = OneDPredictor(hidden_dims=hidden_dims, output_dim=meanvar_dict[all_conditions[0]].shape[0])
    cond_dict = {}
    for c in all_conditions:
        assert cond_arr_dict[c].ndim <= 2, cond_arr_dict[c].ndim
        if cond_arr_dict[c].ndim == 2:
            cond_dict[c] = cond_arr_dict[c][0,:]
        else:
            cond_dict[c] = cond_arr_dict[c]
    d = meanvar_dict[all_conditions[0]].shape[0]

    state = train_mean_predictor(model, train_conditions, cond_dict, meanvar_dict, val_meanvar_dict)

    # 2. Kernel = dot product of hidden representations over all conditions
    hidden_vecs = jnp.stack([
        model.apply({"params": state.params}, cond_dict[c], method=model.get_hidden)
        for c in all_conditions
    ])
    kernel_mat = d * hidden_vecs @ hidden_vecs.T + d
    return kernel_mat, state










