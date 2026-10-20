from typing import Literal, Sequence
import json
import os
import pandas as pd
import pickle as pkl
from loguru import logger
from cmonge.datasets.conditional_loader import ConditionalDataModule
from cmonge.trainers.conditional_monge_trainer import ConditionalMongeTrainer
from cmonge.utils import load_config
from ntk import get_ntk_matrix
from selection import (
    NO_KERNEL_STRATEGIES,
    acquisition_seed_for,
    sample_random_batch,
    select_batch,
)
from utils import get_parser
from consts import condition_map, control_map
from save_restore import save_checkpoint_with_architecture
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm
from flax.core import unfreeze
import gc
import time
import jax
import jax.numpy as jnp



model_configs = {
    "s": dict(dim_hidden=[64, 64, 32], dim_data_map=[32, 32], dim_cond_map=[32, 32]),
    "l5": dict(
        dim_hidden=[512, 512, 512, 64],
        dim_data_map=[256, 256],
        dim_cond_map=[256, 256],
    ),
}


def get_tntk_kernel_matrix(state, x0, cond_arr_dict, num_contexts=1):
    conditions = list(cond_arr_dict.keys())
    cond_z = {}
    for c in tqdm(conditions):
        cond_z[c] = state.apply_fn(
            {"params": state.params},
            x=x0,
            num_contexts=num_contexts,
            c=cond_arr_dict[c],
            train=False,
            return_jacobian=True,
        )
    all_cond_mat = jnp.stack([jnp.ravel(cond_z[c]) for c in conditions])
    n_samples = cond_z[c].shape[0]
    assert all_cond_mat.shape == (len(conditions), n_samples * cond_z[c].shape[1])
    # scale by output dim d and add constant d so kernel signal matches fixed GP-posterior sigma
    d = x0.shape[1]
    kernel_matrix = (all_cond_mat @ all_cond_mat.T) / n_samples * d + d
    assert kernel_matrix.shape == (len(conditions), len(conditions))
    return kernel_matrix, cond_z












def get_last_layer_key(params):
    """Return the key of the last Dense layer in params, robust to architecture."""

    flat = unfreeze(params)
    dense_keys = [k for k in flat.keys() if k.startswith("Dense_")]
    if not dense_keys:
        raise ValueError(f"No Dense_ keys found in params: {list(flat.keys())}")
    return max(dense_keys, key=lambda k: int(k.split("_")[-1]))




def generate_batch(trainer, datamodule, split_type, condition):
    loaders = datamodule.loaders[condition]
    loader_source, loader_target = loaders.get_loaders_by_type(split_type)
    embeddings, n_contexts = trainer.embedding_module(condition=condition)
    return {
        "source": next(loader_source),
        "target": next(loader_target),
        "condition": embeddings,
        "num_contexts": n_contexts,
    }


def prepare_data_grad(trainer, datamodule, conditions, get_val_mean=False):
    """Obtain condition embedding & source distribution.

    Uses every available source (control) cell from the datamodule's loader
    (all splits, not just train) instead of sampling a single batch. The
    condition embedding is broadcast to match x0's size.
    """
    x0 = None
    cond_arr_dict = {}
    mean_dict = {}
    val_mean_dict = {}
    for c in tqdm(conditions):
        c_batch = generate_batch(trainer, datamodule, "train", c)
        if x0 is None:
            loader = datamodule.loaders[c]
            control_x = loader.control_adata.X
            if getattr(loader, "reduction", None):
                control_x = loader.encoder(control_x)
            x0 = jnp.asarray(control_x)
            # The kernels are MC averages over these cells, so record how many the
            # run actually got -- it is capped by the control cells in the h5ad, not
            # by any batch size (sciplex-k562: 512; ebmocavi2: 25952).
            logger.info(
                f"Kernel x0: using all {x0.shape[0]} control cells "
                f"(dim {x0.shape[1]}) from condition {c}'s loader"
            )
        emb = c_batch["condition"]
        cond_arr_dict[c] = jnp.broadcast_to(emb[:1], (x0.shape[0], emb.shape[1]))
        mean_dict[c] = jnp.mean(c_batch["target"], axis=0)
    if get_val_mean:
        for c in tqdm(conditions):
            c_batch = generate_batch(trainer, datamodule, "val", c)
            emb = c_batch["condition"]
            val_mean_dict[c] = jnp.mean(c_batch["target"], axis=0)
    return x0, cond_arr_dict, mean_dict, val_mean_dict


def get_kernels_cmonge(
    cond_arr_dict,
    state,
    x0,
    kernel_type: Literal["tntk"],
    conditions=None,
):
    """Kernel over the conditions in `cond_arr_dict`.

    Every kernel builder below orders its rows by `cond_arr_dict.keys()`. Pass
    `conditions` (the caller's `all_conditions`) to pin that order — the kernel
    row `i` must be the condition `all_conditions[i]`, since that is how
    `selection.select_batch` maps names to indices.
    """
    if conditions is not None:
        conditions = list(dict.fromkeys(conditions))
        missing = [c for c in conditions if c not in cond_arr_dict]
        if missing:
            raise KeyError(
                f"cond_arr_dict is missing {len(missing)} condition(s) needed for "
                f"the kernel, e.g. {missing[:5]}"
            )
        cond_arr_dict = {c: cond_arr_dict[c] for c in conditions}
    get_kernel_method = {
        "tntk": get_tntk_kernel_matrix,
    }[kernel_type]
    kernel_matrix, cond_z = get_kernel_method(state, x0, cond_arr_dict)
    return kernel_matrix, cond_z


def get_batch_cmonge(
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
        "coreset", "uncertainty", "kmeanspp", "lcmd", "typiclust2", "minvar"
    ] = "coreset",
    tmp_save_path=None,
    transform_kernel=True,
    mean_dict=None,
    val_mean_dict=None,
    dim_hidden=None,
    ntk_match_cmonge=False,
    acquisition_seed=None,
):
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
    elif strategy in ("ntk", "ntk_sizematched"):
        # `ntk_sizematched` is a distinct strategy (saved under its own name) that
        # is identical to `ntk` except the MeanPredictor's hidden trunk is sized to
        # match the cmonge model. The `ntk_match_cmonge` flag forces the same
        # behaviour for the plain `ntk` strategy.
        if strategy == "ntk_sizematched" or ntk_match_cmonge:
            # Match the NTK MeanPredictor's hidden trunk to the cmonge model's
            # so the two networks have the same depth/width. The cmonge potential
            # MLP processes its features through `dim_hidden` (see
            # cmonge.models.nn.ConditionalMLP), so we reuse that list verbatim.
            if dim_hidden is None:
                raise ValueError(
                    "ntk_match_cmonge=True requires dim_hidden (the cmonge "
                    "model's hidden dims) to be passed to get_batch_cmonge."
                )
            hidden_dims = list(dim_hidden)
        else:
            d = state.params[get_last_layer_key(state.params)]["kernel"].shape[0]
            hidden_dims = [d, d]
        kernel_mat, _ = get_ntk_matrix(
            cond_arr_dict,
            mean_dict,
            val_mean_dict,
            trained_conditions,
            all_conditions,
            hidden_dims=hidden_dims,
        )
        cond_z = None
    else:
        _t_kernel = time.perf_counter()
        kernel_mat, cond_z = get_kernels_cmonge(
            cond_arr_dict, state, x0, kernel_type=strategy, conditions=all_conditions
        )
        try:
            jax.block_until_ready(kernel_mat)
        except Exception:
            pass
        logger.info(f"[TIMING] kernel_s={time.perf_counter() - _t_kernel:.4f}")
        if jnp.any(jnp.isnan(kernel_mat)) or jnp.any(jnp.isinf(kernel_mat)):
            with open(f"{tmp_save_path}/failed_kernel_nan_inf.pkl", "wb") as f:
                pkl.dump(
                    {
                        "seen_conditions": trained_conditions,
                        "unseen_conditions": unseen_conditions,
                        "all_conditions": all_conditions,
                        "kernel_matrix": kernel_mat,
                        "state": state,
                        "x0": x0,
                        "cond_z": cond_z,
                        "cond_arr_dict": cond_arr_dict,
                    },
                    f,
                )
            raise ValueError("Kernel matrix contains NaN or Inf values")
    try:
        if strategy not in NO_KERNEL_STRATEGIES:
            # select_batch indexes the kernel by `all_conditions.index(name)`, so a
            # kernel that is not exactly len(all_conditions) rows means the rows and
            # the names have drifted apart -- fail here rather than select on a
            # mis-indexed kernel.
            if kernel_mat.shape[0] != len(all_conditions):
                raise ValueError(
                    f"kernel has {kernel_mat.shape[0]} rows but all_conditions has "
                    f"{len(all_conditions)} entries; kernel rows must be ordered by "
                    "all_conditions"
                )
            _t_sel = time.perf_counter()
            batch = select_batch(
                trained_conditions=trained_conditions,
                all_conditions=all_conditions,
                batch_size=min(batch_size, len(unseen_conditions)),
                kernel_matrix=kernel_mat,
                batch_selection_strategy=batch_selection_strategy,
                random_seed=random_seed,
                transform_kernel=transform_kernel,
            )
            logger.info(f"[TIMING] selection_s={time.perf_counter() - _t_sel:.4f}")
        return batch, kernel_mat, cond_z
    except Exception as e:
        logger.error(f"Error selecting batch: {e}")
        logger.info(
            f"Saving kernel at {tmp_save_path}/failed_kernel_{strategy}_{batch_selection_strategy}.pkl and exiting"
        )
        kernel_logger_path = (
            f"{tmp_save_path}/failed_kernel_{strategy}_{batch_selection_strategy}.pkl"
        )
        with open(kernel_logger_path, "wb") as f:
            pkl.dump(
                {
                    "seen_conditions": trained_conditions,
                    "unseen_conditions": unseen_conditions,
                    "all_conditions": all_conditions,
                    "kernel_matrix": kernel_mat,
                    "cond_z": cond_z,
                },
                f,
            )
        raise e


def run_single_iteration(
    config_path: str,
    train_conditions: list,
    test_conditions: list,
    strategy: str = "random",
    lr: float = 1e-4,
    reg: float = None,
    cond_dropout: float = 0.3,
    grad_acc: int = 1,
    output_dir: str = None,
    run_id: str = "test_run",
    evaluate: bool = True,
    log_transport: bool = True,
    n_eval_samples: int = 1,
    checkpoint_dir: str = None,
    rerun: bool = False,
    reeval: bool = False,
    dim_hidden: list = None,
    dim_data_map: list = None,
    dim_cond_map: list = None,
    only_save_transport=False,
    num_train_iter=200,
    save_config=False,
    log_freq=1,
    patience: int = 10,
    lr_patience: int = 5,
    validate_with_conditions: bool = False,
    validate_with_folds: bool = False,
    validate_conditions_seed: int = 0,
    no_early_stopping: bool = False,
    weight_decay: float = None,
):
    """
    Run a single training iteration for testing predictive fit.

    This is useful for:
    - Testing model architecture changes
    - Debugging training issues
    - Evaluating model performance on specific train/test splits
    - Quick experiments without full active learning loop
    - Each iteration in the AL loop

    Args:
        config_path: Path to the config YAML file
        train_conditions: List of conditions to train on
        test_conditions: List of conditions to evaluate on
        strategy: Strategy type ('random', 'tntk', 'normal')
        lr: Learning rate
        reg: Regularization parameter for Monge Gap
        cond_dropout: Condition dropout rate
        grad_acc: Number of gradient accumulation steps
        output_dir: Directory to save outputs (default: /tmp/acpl_test)
        run_id: Identifier for this run
        evaluate: Whether to run evaluation after training
        log_transport: Whether to log transport samples during evaluation
        n_eval_samples: Number of samples for evaluation
        checkpoint_dir: Directory for checkpoints (default: output_dir/run_id)
        rerun: Force retraining even if checkpoint exists
        reeval: Force re-evaluation even if log exists
        dim_hidden: Hidden layer dimensions (default: [64, 64, 64, 32])
        dim_data_map: Data map dimensions (default: [32, 32])
        dim_cond_map: Condition map dimensions (default: [32, 32])
        validate_with_conditions: Whether to validate using conditions instead of samples
        validate_conditions_seed: Random seed for selecting conditions for validation
    Returns:
        dict with keys:
            - 'trainer': The trained trainer object
            - 'model': The trained model state
            - 'train_datamodule': Training data module
            - 'test_datamodule': Test data module
            - 'logs': Training logs (if available)
            - 'eval_results': Evaluation results (if evaluate=True)
            - 'config': The training config
    """

    if output_dir is None:
        output_dir = "/tmp/acpl_test"
    os.makedirs(output_dir, exist_ok=True)

    if checkpoint_dir is None:
        checkpoint_dir = f"{output_dir}/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    if dim_hidden is None:
        dim_hidden = [256, 256, 256, 256, 64]
    if dim_data_map is None:
        dim_data_map = [128, 128, 128]
    if dim_cond_map is None:
        dim_cond_map = [128, 128, 128]

    # Setup training config
    trainer, train_datamodule, test_datamodule, train_config, logger_path = (
        setup_trainer(
            config_path,
            train_conditions,
            test_conditions,
            dim_hidden,
            dim_data_map,
            dim_cond_map,
            cond_dropout,
            grad_acc,
            reg,
            lr,
            num_train_iter,
            output_dir,
            run_id,
            validate_with_conditions=validate_with_conditions,
            validate_with_folds=validate_with_folds,
            validate_conditions_seed=validate_conditions_seed,
            weight_decay=weight_decay,
        )
    )

    net = None
    logs = None

    # Train or load checkpoint
    if not os.path.exists(checkpoint_dir) or rerun:
        logger.info("Starting training...")
        net, logs, log_step = trainer.train(
            train_datamodule,
            log_freq=log_freq,
            loss_patience=patience,
            lr_patience=lr_patience,
            restore_best=not no_early_stopping,
        )
        logger.info(f"Training completed at step {log_step}")
        if save_config:
            with open(f"{checkpoint_dir}/train_config.pkl", "wb") as f:
                pkl.dump(train_config, f)

        trainer.save_checkpoint(path=str(checkpoint_dir))
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    else:
        try:
            trainer = ConditionalMongeTrainer.load_checkpoint(
                jobid=1,
                logger_path=logger_path,
                config=train_config.model,
                datamodule=train_datamodule,
                ckpt_path=checkpoint_dir,
            )
            net = trainer.state_neural_net
            logger.info(f"Loaded existing checkpoint from {checkpoint_dir}")

        except Exception as e:
            logger.warning(
                f"Failed to load checkpoint from {checkpoint_dir}: {e}, retraining..."
            )
            net, logs, log_step = trainer.train(
                train_datamodule,
                log_freq=log_freq,
                loss_patience=patience,
                lr_patience=lr_patience,
                restore_best=not no_early_stopping,
            )
            save_checkpoint_with_architecture(
                checkpoint_dir,
                trainer,
                train_config,
                train_conditions,
                metadata={
                    "strategy": strategy,
                    "lr": lr,
                    "cond_dropout": cond_dropout,
                    "grad_acc": grad_acc,
                    "reg": reg,
                },
            )

    result = {
        "trainer": trainer,
        "model": net,
        "train_datamodule": train_datamodule,
        "test_datamodule": test_datamodule,
        "logs": logs,
        "config": train_config,
        "eval_results": None,
    }

    # Evaluate
    if only_save_transport:
        logger.info("Only saving transport samples without evaluation.")
        trainer.evaluate(
            train_datamodule,
            n_samples=n_eval_samples,
            log_transport=True,
            do_log_metrics=False,
            splits=["train", "valid"],
        )
        trainer.evaluate(
            test_datamodule,
            n_samples=n_eval_samples,
            log_transport=True,
            do_log_metrics=False,
            splits=["test"],
        )
    elif evaluate and (not os.path.exists(logger_path) or reeval):
        logger.info("Running evaluation on test set...")
        trainer.evaluate(
            test_datamodule, n_samples=n_eval_samples, log_transport=log_transport
        )
        logger.info("Running evaluation on train/valid sets...")
        trainer.evaluate(
            train_datamodule,
            n_samples=n_eval_samples,
            log_transport=log_transport,
            splits=["train", "valid"],
        )
        result = {
            "trainer": trainer,
            "model": net,
            "train_datamodule": train_datamodule,
            "test_datamodule": test_datamodule,
            "logs": logs,
            "config": train_config,
            "eval_results": None,
        }

    logger.info(f"Results saved to {output_dir}")
    return result


def setup_condition(
    config_path: str,
    conditions: Sequence[str],
    checkpoint_prefix: str,
    train_conditions: Sequence[str],
    dim_hidden: list,
    dim_data_map: list,
    dim_cond_map: list,
    output_dir: str,
    moa: bool = False,
    strategy: str = "random",
    data: str = "",
):
    # Prepare condition embeddings for non-random strategies
    if strategy not in NO_KERNEL_STRATEGIES or moa:
        # Prepare condition embedding
        next_config = load_config(config_path)

        if strategy == "normal":
            next_config.data.name = "sciplex_normal"

        if moa:
            next_config.condition.conditions = conditions
            next_config.model.embedding.checkpoint = False
            next_config.model.embedding.model_dir = (
                f"./data/{data}/"
            )
            checkpoint_dir = f"{checkpoint_prefix}/moa"
            next_config.data.split = [1.0, 0, 0]
            next_config.condition.split = [1.0, 0, 0]
        else:
            next_config.condition.conditions = train_conditions
            next_config.model.embedding.checkpoint = True
            next_config.data.batch_size = 4096
        next_config.model.mlp.dim_hidden = dim_hidden
        next_config.model.mlp.dim_data_map = dim_data_map
        next_config.model.mlp.dim_cond_map = dim_cond_map
        next_datamodule = ConditionalDataModule(next_config.data, next_config.condition)
        logger_path = f"{checkpoint_prefix}/log"
        next_trainer = ConditionalMongeTrainer(
            jobid=1,
            logger_path=f"{output_dir}/log",
            config=next_config.model,
            datamodule=next_datamodule,
        )
        # Order matters: `select_batch` maps condition names to kernel row indices
        # via the `all_conditions` list it is given (the caller's `train_conditions`
        # / `conditions`), so cond_arr_dict — and hence the kernel — must be built
        # in that same order. `next_datamodule.conditions` is
        # `list(set(config.conditions))` (cmonge/datasets/conditional_loader.py), a
        # hash-randomised permutation that differs on every process, so using it
        # here silently permuted the kernel rows relative to `all_conditions`.
        kernel_conditions = list(dict.fromkeys(conditions if moa else train_conditions))
        x0, cond_arr_dict, mean_dict, val_mean_dict = prepare_data_grad(
            next_trainer, next_datamodule, kernel_conditions
        )
    else:
        cond_arr_dict = None
        x0 = None
        mean_dict = None
        val_mean_dict = None
    return cond_arr_dict, x0, mean_dict, val_mean_dict


def setup_trainer(
    config_path: str,
    train_conditions: Sequence[str],
    test_conditions: Sequence[str],
    dim_hidden: list,
    dim_data_map: list,
    dim_cond_map: list,
    cond_dropout: float,
    grad_acc: int,
    reg: float,
    lr: float,
    num_train_iter: int,
    output_dir: str,
    run_id: str,
    validate_with_conditions: bool = False,
    validate_with_folds: bool = False,
    validate_conditions_seed: int = 0,
    weight_decay: float = None,
):
    from cmonge.utils import load_config as load_config_util

    if validate_with_conditions:
        logger.debug("Splitting train conditions into train/valid with KFold...")
        if validate_with_folds:
            folds = list(
                KFold(n_splits=5, shuffle=True, random_state=0).split(train_conditions)
            )
            train_idx, valid_idx = folds[validate_conditions_seed]
        else:
            train_idx, valid_idx = train_test_split(
                list(range(len(train_conditions))),
                test_size=5,
                random_state=validate_conditions_seed,
            )
        valid_conditions = [train_conditions[i] for i in valid_idx]

    train_config = load_config_util(config_path)
    train_config.model.mlp.dim_hidden = dim_hidden
    train_config.model.mlp.dim_data_map = dim_data_map
    train_config.model.mlp.dim_cond_map = dim_cond_map
    train_config.model.mlp.cond_dropout_rate = cond_dropout
    train_config.model.mlp.layer_norm = True
    train_config.condition.conditions = train_conditions

    train_config.model.embedding.checkpoint = True

    if grad_acc is not None:
        train_config.model.optim.grad_acc_steps = grad_acc

    train_config.model.regularizer.kwargs.epsilon_regularizer = reg
    train_config.model.optim.lr = lr
    if weight_decay is not None:
        train_config.model.optim.kwargs.weight_decay = weight_decay
    train_config.model.num_train_iters = num_train_iter
    if validate_with_conditions:
        train_config.condition.valid_conditions = valid_conditions
        train_config.condition.ood = valid_conditions
        train_config.condition.split = [1.0, 0.0, 0.0]
        # Include valid_conditions in conditions so splitter loads them;
        # the ood check in splitter assigns them split [0,1,0] (valid only, not trained on)
        train_config.condition.conditions = train_conditions + valid_conditions

    # Setup test config
    test_config = load_config_util(config_path)
    test_config.condition.conditions = test_conditions
    test_config.data.split = [0.0, 0.0, 1.0]
    test_config.condition.split = [0.0, 0.0, 1.0]

    # Create data modules
    train_datamodule = ConditionalDataModule(
        train_config.data,
        train_config.condition,
    )
    train_datamodule.name = run_id
    test_datamodule = ConditionalDataModule(test_config.data, test_config.condition)

    logger.info(f"Training on {len(train_conditions)} conditions: {train_conditions}")
    logger.info(f"Testing on {len(test_conditions)} conditions: {test_conditions}")

    # Create trainer
    logger_path = f"{output_dir}/{run_id}_log.pkl"

    trainer = ConditionalMongeTrainer(
        jobid=1,
        logger_path=logger_path,
        config=train_config.model,
        datamodule=train_datamodule,
    )
    return trainer, train_datamodule, test_datamodule, train_config, logger_path


def run_al_loop(
    config_path: str,
    data: str,
    strategy: str = "random",
    batch_selection_strategy: str = "kmeanspp",
    batch_size: int = 20,
    init_batch_size: int = 20,
    init_seed: int = 0,
    split_seed: int = 0,
    max_iter: int = None,
    lr: float = 1e-4,
    reg: float = None,
    cond_dropout: float = 0.0,
    grad_acc: int = None,
    prefix: str = "al_smonge",
    rerun: bool = False,
    reeval: bool = False,
    moa: bool = False,
    fold_filter: list = None,
    dim_hidden: list = None,
    dim_data_map: list = None,
    dim_cond_map: list = None,
    patience: int = 10,
    lr_patience: int = 5,
    validate_with_conditions: bool = False,
    validate_conditions_seed: int = 0,
    num_train_iter: int = None,
    no_early_stopping: bool = False,
    transform_kernel: bool = True,
    ntk_match_cmonge: bool = False,
):
    """
    Run the full active learning loop.

    Args:
        config_path: Path to config YAML
        data: Dataset name
        strategy: AL strategy ('random', 'tntk', 'normal')
        batch_selection_strategy: Batch selection strategy
        batch_size: Batch size for AL iterations
        init_batch_size: Initial batch size
        init_seed: Random seed for initialization
        split_seed: Random seed for K-fold splits
        max_iter: Maximum AL iterations
        lr: Learning rate
        reg: Regularization parameter
        cond_dropout: Condition dropout rate
        grad_acc: Gradient accumulation steps
        prefix: Prefix for output directory
        rerun: Force retraining
        reeval: Force re-evaluation
        moa: Whether to use MOA embeddings
        fold_filter: List of fold indices to run (None = all)
        validate_with_conditions: Whether to hold out some conditions for validation instead of random split
        validate_with_folds: Whether to use k-fold cross-validation for validation
        validate_conditions_seed: Random seed for selecting validation conditions
        num_train_iter: Override number of training epochs (None = use config default)
        no_early_stopping: Disable early stopping (train for all epochs)
        transform_kernel: Whether to apply GP posterior transformation to kernel matrix
        ntk_match_cmonge: For strategy='ntk', size the NTK MeanPredictor's hidden
            trunk to match the cmonge model's `dim_hidden` instead of the default
            two-layer [d, d] (d = cmonge last-layer input dim).
    """
    if max_iter is None:
        if batch_size == 1:
            max_iter = 5
        else:
            max_iter = 3

    conditions = pd.read_csv(condition_map[data], index_col=None, header=None)[
        0
    ].tolist()
    if control_map[data] in conditions:
        conditions.remove(control_map[data])

    logger.info(f"Using active learning strategy: {strategy}")
    logger.info(f"Experiment: Training model on {len(conditions)} conditions")

    if batch_size > init_batch_size and init_batch_size != 1:
        init_batch_size = batch_size

    kf = KFold(n_splits=5, random_state=split_seed, shuffle=True)

    for fold_idx, (train_idx, test_idx) in enumerate(kf.split(conditions)):
        if fold_filter is not None and fold_idx not in fold_filter:
            continue
        output_dir = f"./models/cmonge-al/{data}/{prefix}/"
        os.makedirs(output_dir, exist_ok=True)
        bss_suffix = (
            f"_{batch_selection_strategy}"
            if batch_selection_strategy != "coreset"
            else ""
        )
        exp_id = f"al{'gp' if transform_kernel else '0'}_{strategy}{bss_suffix}_size{batch_size}_fold{fold_idx}_seed{init_seed}"
        if not no_early_stopping:
            exp_id += f"_pat{patience}_lrpat{lr_patience}"
        last_iter_log = f"{output_dir}/{exp_id}_iter{max_iter}_log.pkl"
        if os.path.exists(last_iter_log) and not rerun:
            logger.info(f"Fold {fold_idx} already completed, skipping...")
            continue

        # Log split
        log_path = f"{output_dir}/{exp_id}.json"
        cond_dict = {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}

        shared_train_val_conditions = [conditions[i] for i in train_idx]
        heldout_test_conditions = [conditions[i] for i in test_idx]

        with open(log_path, "w") as f:
            json.dump(cond_dict, f)

        cond_arr_dict, x0, mean_dict, val_mean_dict = setup_condition(
            config_path=config_path,
            conditions=conditions,
            checkpoint_prefix=prefix,
            train_conditions=shared_train_val_conditions,
            dim_hidden=dim_hidden,
            dim_data_map=dim_data_map,
            dim_cond_map=dim_cond_map,
            output_dir=output_dir,
            moa=moa,
            strategy=strategy,
            data=data,
        )

        # AL loop
        unlabeled_conditions = shared_train_val_conditions.copy()
        train_conditions = []
        net = None
        for al_iter in range(max_iter + 1):
            if len(unlabeled_conditions) == 0:
                break

            logger.info(f"Running fold {fold_idx} iter {al_iter}")
            run_id = f"{exp_id}_iter{al_iter}"
            checkpoint_dir = f"{output_dir}/{run_id}"
            os.makedirs(checkpoint_dir, exist_ok=True)

            # Select batch
            if al_iter == 0:
                batch, kernel_mat, cond_z = get_batch_cmonge(
                    all_conditions=shared_train_val_conditions,
                    unseen_conditions=unlabeled_conditions,
                    trained_conditions=[],
                    strategy="random",
                    batch_size=init_batch_size,
                    random_seed=init_seed,
                    tmp_save_path=checkpoint_dir,
                    transform_kernel=transform_kernel,
                )
            else:
                batch, kernel_mat, cond_z = get_batch_cmonge(
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
                    transform_kernel=transform_kernel,
                    mean_dict=mean_dict,
                    val_mean_dict=val_mean_dict,
                    dim_hidden=dim_hidden,
                    ntk_match_cmonge=ntk_match_cmonge,
                    acquisition_seed=acquisition_seed_for(
                        init_seed, fold_idx, al_iter
                    ),
                )

            # Save kernel info (skip large arrays x0/cond_arr_dict/cond_z to save disk)
            with open(f"{output_dir}/{run_id}_kernels.pkl", "wb") as f:
                pkl.dump(
                    {
                        "seen_conditions": train_conditions,
                        "selected_conditions": batch,
                        "kernel_matrix": kernel_mat,
                    },
                    f,
                )

            # Free GPU memory before training: the previous net and the
            # kernel arrays are not needed during training (a fresh trainer
            # and state are built inside run_single_iteration).
            del kernel_mat, cond_z
            net = None
            jax.clear_caches()
            gc.collect()

            # Update conditions
            try:
                for x in batch:
                    unlabeled_conditions.remove(x)
            except ValueError as e:
                logger.error(f"Selected batch: {batch}")
                logger.error(
                    f"Unlabeled conditions before removal: {unlabeled_conditions}"
                )
                raise e
            train_conditions = train_conditions + batch

            # Run single iteration
            _t_fit = time.perf_counter()
            result = run_single_iteration(
                config_path=config_path,
                train_conditions=train_conditions,
                test_conditions=heldout_test_conditions,
                strategy=strategy,
                lr=lr,
                reg=reg,
                cond_dropout=cond_dropout,
                grad_acc=grad_acc,
                output_dir=output_dir,
                run_id=run_id,
                evaluate=True,
                log_transport=True,
                n_eval_samples=1,
                checkpoint_dir=checkpoint_dir,
                rerun=rerun,
                reeval=reeval,
                dim_hidden=dim_hidden,
                dim_data_map=dim_data_map,
                dim_cond_map=dim_cond_map,
                save_config=(fold_idx == 0) & (init_seed == 0) & (al_iter == 0),
                log_freq=1,
                patience=999999 if no_early_stopping else patience,
                lr_patience=999999 if no_early_stopping else lr_patience,
                validate_with_conditions=validate_with_conditions,
                validate_with_folds=False,
                validate_conditions_seed=validate_conditions_seed,
                num_train_iter=num_train_iter if num_train_iter is not None else 200,
                no_early_stopping=no_early_stopping,
            )

            logger.info(f"[TIMING] fit_s={time.perf_counter() - _t_fit:.4f}")
            net = result["model"]

            if moa:
                exit(0)


if __name__ == "__main__":
    # Mirrors the TPU branch in al_train_cflow.py. Every kernel strategy here
    # ends in a float32 Gram computed on device -- get_tntk_kernel_matrix's
    # einsum, the ntk/ntk_sizematched variants and
    # get_projected_grad_kernel_matrix's `all_cond_mat @ all_cond_mat.T`. On TPU
    # a float32 matmul defaults to bfloat16 passes, which would quietly degrade
    # those Grams and therefore the batch `select_batch` picks from them;
    # downstream selection is already numpy float64 (acpl/selection.py) and
    # cannot recover precision the Gram never had. That would make a TPU arm
    # non-comparable to runs produced on GPU. CPU and GPU are unaffected -- their DEFAULT precision is already
    # float32, so this branch is a no-op for every non-TPU run and cannot
    # perturb results produced on the cluster.
    if jax.devices()[0].platform == "tpu":
        jax.config.update("jax_default_matmul_precision", "highest")
        logger.info("TPU backend: jax_default_matmul_precision=highest")
    # The nvidia-smi poll only feeds the keep-alive sizing controller, and it
    # reports device-wide utilisation -- meaningless (and one subprocess per
    parser = get_parser()
    parser.add_argument(
        "--reg", type=float, default=None, help="Regularization parameter for Monge Gap"
    )
    parser.add_argument(
        "--moa", action="store_true", help="Whether to only learn MOA embeddings"
    )
    parser.add_argument(
        "--grad-acc",
        type=int,
        default=None,
        help="Number of gradient accumulation steps",
    )
    parser.add_argument(
        "--cond-dropout",
        type=float,
        default=0.0,
        help="Condition dropout rate",
    )
    parser.add_argument(
        "--fold",
        type=int,
        nargs="+",
        default=None,
        help="Specific fold(s) to run (default: all)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience (number of steps without improvement)",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=10,
        help="LR reduction patience (number of steps without improvement before reducing LR)",
    )
    parser.add_argument(
        "--validate-with-conditions",
        action="store_true",
        help="Whether to validate using conditions instead of samples",
    )
    parser.add_argument(
        "--validate-conditions-seed",
        type=int,
        default=0,
        help="Random seed for selecting conditions for validation split (if --validate-with-conditions is set)",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=None,
        help="Override number of training epochs (default: use config value)",
    )
    parser.add_argument(
        "--no-early-stopping",
        action="store_true",
        help="Disable early stopping (train for all epochs)",
    )
    parser.add_argument(
        "--ntk-match-cmonge",
        action="store_true",
        help="For strategy=ntk, make the NTK MeanPredictor hidden trunk the same "
        "size as the cmonge model (use its dim_hidden) instead of the default [d, d]",
    )
    args = parser.parse_args()

    # def _dummy_matmul_loop(stop_event, size=4096):
    #     gpu = jax.devices('gpu')[0]
    #     a = jax.device_put(jnp.ones((size, size)), gpu)
    #     while not stop_event.is_set():
    #         jax.block_until_ready(a @ a)

    # stop_dummy = threading.Event()
    # dummy_thread = threading.Thread(target=_dummy_matmul_loop, args=(stop_dummy,), daemon=True)
    # dummy_thread.start()

    # try:
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
        reg=args.reg,
        cond_dropout=args.cond_dropout,
        grad_acc=args.grad_acc,
        prefix=args.prefix,
        rerun=args.rerun,
        reeval=args.reeval,
        moa=args.moa,
        fold_filter=args.fold,
        patience=args.patience,
        lr_patience=args.lr_patience,
        dim_hidden=model_configs[args.model_config_id]["dim_hidden"],
        dim_data_map=model_configs[args.model_config_id]["dim_data_map"],
        dim_cond_map=model_configs[args.model_config_id]["dim_cond_map"],
        validate_with_conditions=args.validate_with_conditions,
        validate_conditions_seed=args.validate_conditions_seed,
        num_train_iter=args.num_epochs,
        no_early_stopping=args.no_early_stopping,
        transform_kernel=args.transform_kernel,
        ntk_match_cmonge=args.ntk_match_cmonge,
    )
    # finally:
    #     stop_dummy.set()
    #     dummy_thread.join()

    # Every result is written by this point. Exit without running CPython's
    # finalisers: tearing down the CUDA client while JAX still holds device
    # buffers can abort the process (SIGABRT) after a successful run.
    # al_train_cflow.py ends the same way.
    os._exit(0)
