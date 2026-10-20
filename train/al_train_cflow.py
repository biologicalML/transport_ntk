from typing import Literal, Tuple, List, Dict
import json
import gc
import os
import time
import pickle as pkl
import functools
import numpy as np
import gc
import jax
import jax.random
import jax.numpy as jnp

import flax.linen as nn
from flax.training import train_state
import optax

from loguru import logger
import pandas as pd
import anndata as ad
from ntk import get_ntk_matrix
from selection import (
    ITER0_DONOR_STRATEGIES,
    NO_KERNEL_STRATEGIES,
    acquisition_seed_for,
    sample_random_batch,
    select_batch,
)
from utils import (
    get_parser,
    adata_conditional_subset,
)
from consts import condition_map, control_map
from sklearn.model_selection import KFold, train_test_split
from tqdm.auto import tqdm
from cmonge.utils import load_config


import diffrax
# PIDController's own default norm; diffrax 0.7.0 does not re-export it.
import cellflow
from cellflow.utils import match_linear, labeled_match_linear
from cellflow.model import CellFlow
from cellflow.data._dataloader import TrainSampler
from cellflow.training import CheckpointCallback
import cloudpickle
import shutil

import os
import time

from jax.profiler import save_device_memory_profile

save_device_memory_profile("memory_profile.pb")


# --------------------------------------------------------------------------
# .running lock lifetime management.
#
# The per-fold lock used to be released only by a `finally:` block. Python does
# not run `finally:` when the default SIGTERM disposition kills the process, so
# every scancel / TIMEOUT / GPU-util-watchdog kill orphaned its lock, and the
# next submission then skipped that fold forever. Two defences:
#   1. a SIGTERM/SIGINT/SIGHUP handler that unlinks held locks before dying;
#   2. an acquire path that reclaims a lock whose owning Slurm job is gone,
#      which also covers SIGKILL, OOM-kill and node faults (uncatchable).
# --------------------------------------------------------------------------
import atexit
import re as _re
import signal

_ACTIVE_LOCKS = set()


def _release_active_locks():
    """Unlink every lock this process currently holds. Never raises."""
    for path in list(_ACTIVE_LOCKS):
        try:
            os.remove(path)
        except OSError:
            pass
        _ACTIVE_LOCKS.discard(path)


def _lock_signal_handler(signum, frame):
    if _ACTIVE_LOCKS:
        logger.warning(
            f"Signal {signum} received; releasing {len(_ACTIVE_LOCKS)} "
            f".running lock(s) before exit: {sorted(_ACTIVE_LOCKS)}"
        )
    _release_active_locks()
    # Restore the default disposition and re-raise so the job's recorded exit
    # state still reflects the signal rather than a clean exit.
    try:
        signal.signal(signum, signal.SIG_DFL)
    except (ValueError, OSError):
        pass
    os.kill(os.getpid(), signum)


def _install_lock_signal_handlers():
    for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        try:
            signal.signal(sig, _lock_signal_handler)
        except (ValueError, OSError):
            # Not on the main thread, or platform lacks the signal.
            pass
    atexit.register(_release_active_locks)


def _owner_process_is_alive(pid):
    """True / False if determinable, None if we cannot tell.

    None is the conservative answer: a lock whose owner cannot be proved dead
    is left alone. The lock file records the pid of the process that took it,
    so a run killed without cleanup can be reclaimed by the next one instead
    of blocking that fold forever.
    """
    if not pid or pid in ("n/a", "<unreadable>"):
        return None
    try:
        os.kill(int(pid), 0)
    except ValueError:
        return None
    except ProcessLookupError:
        return False
    except PermissionError:
        # Exists, owned by another user.
        return True
    return True


def _acquire_fold_lock(lock_path, fold, ignore_lock):
    """Return an open fd for `lock_path`, or None if another run owns it."""
    if ignore_lock and os.path.exists(lock_path):
        logger.warning(f"--ignore-lock: removing stale lock {lock_path}")
        try:
            os.remove(lock_path)
        except OSError:
            pass

    for _attempt in range(3):
        try:
            fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                with open(lock_path) as _f:
                    owner = _f.read().strip()
            except OSError:
                owner = "<unreadable>"
            match = _re.search(r"pid=(\S+)", owner)
            owner_pid = match.group(1) if match else None
            alive = _owner_process_is_alive(owner_pid)
            if alive is False:
                logger.warning(
                    f"Fold {fold}: reclaiming stale lock {lock_path} -- owning "
                    f"process {owner_pid} is no longer running ({owner})."
                )
                try:
                    os.remove(lock_path)
                except OSError:
                    pass
                continue  # retry the O_EXCL create
            reason = (
                "owner still running"
                if alive
                else "could not determine whether the owner is alive"
            )
            logger.warning(
                f"Fold {fold} is already being processed ({reason}; lock held by "
                f"{owner} at {lock_path}), skipping. Pass --ignore-lock to override."
            )
            return None
        else:
            _ACTIVE_LOCKS.add(lock_path)
            return fd

    logger.warning(
        f"Fold {fold}: gave up reclaiming {lock_path} after repeated races, skipping."
    )
    return None


_install_lock_signal_handlers()




def prepare_data_cond(
    train_sampler, conditions, random_seed, sample_size=4096*2, valid_fraction=0.2
) -> Tuple[
    np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray], dict[str, np.ndarray]
]:
    """Obtain condition embeddings, source cells, and per-condition target cell means.

    Returns:
        x0: source cell array (sample_size, n_features)
        cond_arr_dict: {condition: embedding (1, emb_dim)} for all conditions
        mean_train_data: {condition: mean of train-split target cells}
        mean_valid_data: {condition: mean of valid-split target cells}
    """
    rng = np.random.default_rng(random_seed)
    source_dist_idx = rng.integers(0, train_sampler.n_source_dists)
    train_sampler.batch_size = sample_size
    # Get source cells
    source_cells_mask = train_sampler._data.split_covariates_mask == source_dist_idx
    source_batch_idcs = train_sampler._sample_from_mask(rng, source_cells_mask)
    x0 = train_sampler._data.cell_data[source_batch_idcs]

    cond_arr_dict = {}
    mean_train_data = {}
    mean_valid_data = {}
    cond_emb = train_sampler._data.condition_data["condition"]
    for cidx, (pert_idx, c) in enumerate(
        train_sampler._data.perturbation_idx_to_covariates.items()
    ):
        cond_name = c[0]
        cond_arr_dict[cond_name] = cond_emb[[cidx], ...]

        # All target cells for this condition
        target_idcs = np.where(
            train_sampler._data.perturbation_covariates_mask == pert_idx
        )[0]
        target_cells = train_sampler._data.cell_data[target_idcs]

        # Deterministic train/valid split
        n_total = len(target_cells)
        n_valid = max(1, int(n_total * valid_fraction))
        shuffled = rng.permutation(n_total)
        mean_valid_data[cond_name] = target_cells[shuffled[:n_valid]].mean(0)
        mean_train_data[cond_name] = target_cells[shuffled[n_valid:]].mean(0)

    if not all([c in cond_arr_dict for c in conditions]):
        missing_conds = [c for c in conditions if c not in cond_arr_dict]
        raise ValueError(f"Conditions {missing_conds} not found in condition data.")
    return x0, cond_arr_dict, mean_train_data, mean_valid_data




def _compute_trajectory_grads(apply_fn, params, x0, cond_arr):
    """Integrate Jacobians along the ODE trajectory using the same Tsit5 adaptive steps.

    Augments the ODE state with a running integral of the Jacobian so the
    quadrature uses exactly the same step sizes and weights as the trajectory.
    All samples are integrated in a single batched solve so each RHS evaluation
    is one large apply_fn call (good GPU utilization), sharing one adaptive schedule.

    Returns:
        grads: (n_samples, n_out_dim) -- ∫₀¹ grad(x(t), t) dt per sample
    """
    n_samples = x0.shape[0]
    probe = jax.eval_shape(
        lambda: apply_fn(
            {"params": params},
            t=jnp.zeros((n_samples, 1)),
            x_t=x0,
            cond={"condition": cond_arr},
            train=False,
            encoder_noise=0.0,
            return_jacobian=True,
        )
    )
    n_out_dim = probe[1].shape[-1]

    def augmented_vf(t, state, args):
        x, _ = state
        v_t, grad = apply_fn(
            {"params": params},
            t=jnp.full((x.shape[0], 1), t),
            x_t=x,
            cond={"condition": args},
            train=False,
            encoder_noise=0.0,
            return_jacobian=True,
        )
        return v_t, grad

    y0 = (x0, jnp.zeros((n_samples, n_out_dim)))
    result = diffrax.diffeqsolve(
        diffrax.ODETerm(augmented_vf),
        solver=diffrax.Tsit5(),
        t0=0.0,
        t1=1.0,
        dt0=None,
        y0=y0,
        args=cond_arr,
        saveat=diffrax.SaveAt(t1=True),
        stepsize_controller=diffrax.PIDController(rtol=1e-05, atol=1e-05),
        max_steps=16**5,
    )
    y, grad_integral = result.ys  # (1, n_samples, n_out_dim)
    return y[0], grad_integral[0]  # (n_samples, n_out_dim)


def get_tntk_kernel_matrix(
    state: train_state.TrainState,
    x0: np.ndarray,
    conditions: List[str],
    cond_arr_dict: Dict[str, np.ndarray],
):
    @jax.jit
    def compute_trajectory_grads(cond_arr):
        return _compute_trajectory_grads(state.apply_fn, state.params, x0, cond_arr)[1]

    cond_stack = jnp.stack([cond_arr_dict[c] for c in conditions])
    grads = jax.vmap(compute_trajectory_grads)(cond_stack)
    # scale by output dim d and add constant d so kernel signal matches fixed GP-posterior sigma
    d = x0.shape[1]
    kernel_matrix = jnp.einsum("isdeh,jsdeh->ij", grads, grads) / x0.shape[0] * d + d

    return kernel_matrix, None




















def get_kernels_cflow(
    all_conditions: List[str],
    cond_arr_dict: Dict[str, np.ndarray],
    state: train_state.TrainState,
    x0: np.ndarray,
    kernel_type: Literal["tntk"],
    res_weight: float = 1.0,
    mean_dict: Dict[str, np.ndarray] = None,
    val_mean_dict: Dict[str, np.ndarray] = None,
    trained_conditions: List[str] = None,
    dim_hidden: List[int] = None,
):

    get_kernel_matrix = {
        "tntk": get_tntk_kernel_matrix,
    }[kernel_type]
    kernel_mat, cond_z = get_kernel_matrix(state, x0, all_conditions, cond_arr_dict)
    return kernel_mat


def _remove_epoch_ckpt(path: str) -> None:
    """Delete a ``{base}-epoch{N}`` directory and its ``_es_state.pkl`` sidecar."""
    try:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        sidecar = f"{path}_es_state.pkl"
        if os.path.exists(sidecar):
            os.remove(sidecar)
    except OSError:
        pass


class EpochCheckpointCallback(CheckpointCallback):
    """Periodic full-model checkpoint for crash-resume.

    Writes at each validation step:
      - ``{base_path}-epoch{N}/`` via ``cf.save()`` (full pickled CellFlow)
      - ``{base_path}-epoch{N}_es_state.pkl`` (early-stopping state from
        ``trainer._es_state``: valid_loss, waited, waited_lr, lr_reductions,
        best_params, training_logs, last_epoch)

    Training logs live inside ``_es_state.pkl`` and are re-seeded on resume;
    the callback deliberately does NOT touch the top-level
    ``{run_id}_training_logs.pkl`` mid-training, because the outer main loop
    treats that file as the "fold completed" sentinel.

    Only the last ``keep_last`` epoch directories are retained. On successful
    ``on_train_end`` the intermediate checkpoints are purged (the caller then
    writes the final checkpoint via ``cf.save(checkpoint_dir)``).
    """

    def __init__(self, base_path: str, keep_last: int = 2):
        self.base_path = base_path.rstrip("/")
        self.keep_last = max(1, int(keep_last))
        self._model = None
        self._saved_epochs: list[int] = []

    def on_train_begin(self, train_conditions, model):
        self._model = model

    def on_log_iteration(self, state, epoch_id: int | None = None):
        if self._model is None or epoch_id is None:
            return
        ckpt_path = f"{self.base_path}-epoch{epoch_id}"
        try:
            os.makedirs(ckpt_path, exist_ok=True)
            self._model.save(ckpt_path, overwrite=True)
            es_state = getattr(self._model.trainer, "_es_state", None)
            if es_state is not None:
                with open(f"{ckpt_path}_es_state.pkl", "wb") as f:
                    cloudpickle.dump(es_state, f)
        except Exception as e:
            logger.warning(
                f"EpochCheckpointCallback: failed to save at epoch {epoch_id}: {e}"
            )
            return
        self._saved_epochs.append(epoch_id)
        logger.info(
            f"EpochCheckpointCallback: saved {ckpt_path} "
            f"(kept epochs: {self._saved_epochs})"
        )
        while len(self._saved_epochs) > self.keep_last:
            old = self._saved_epochs.pop(0)
            _remove_epoch_ckpt(f"{self.base_path}-epoch{old}")

    def on_train_end(self, model):
        # Purge *all* ``-epoch{N}`` dirs + sidecars for this checkpoint,
        # including ones left over from a previous crashed run we resumed from.
        import glob as _glob

        for p in _glob.glob(f"{self.base_path}-epoch*"):
            try:
                if os.path.isdir(p):
                    shutil.rmtree(p, ignore_errors=True)
                elif os.path.isfile(p):
                    os.remove(p)
            except OSError:
                pass
        self._saved_epochs = []


def run_single_iteration(
    adata: ad.AnnData,
    train_conditions: list,
    test_conditions: list,
    config,
    model_config: dict,
    n_time_samples_train: int = 128,
    n_epochs: int = 1000,
    n_conds_per_batch: int = 1,
    output_dir: str = None,
    run_id: str = "test_run",
    checkpoint_dir: str = None,
    rerun: bool = False,
    solver_class: Literal["otfm", "conical"] = "otfm",
    early_stopping_patience: int = None,
    min_epochs: int = 100,
    valid_freq=1000,
    validate_with_conditions: bool = False,
    validate_with_folds: bool = False,
    validate_conditions_seed: int = 0,
    resume: bool = False,
    lr_patience: int | None = None,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
):
    """
    Run a single training iteration.

    Args:
        adata: AnnData object with all data
        train_conditions: List of conditions to train on
        test_conditions: List of conditions to evaluate on
        config: Configuration object
        model_config: Model configuration dictionary
        n_time_samples_train: Number of time samples for training
        n_epochs: Number of epochs to train
        n_conds_per_batch: Number of conditions per batch
        output_dir: Directory to save outputs
        run_id: Identifier for this run
        checkpoint_dir: Directory for checkpoints (default: output_dir/run_id)
        rerun: Force retraining even if checkpoint exists
        validate_with_conditions: Whether to hold out some conditions for validation instead of random split
        validate_with_folds: Whether to use k-fold cross-validation for validation
        validate_conditions_seed: Random seed for selecting validation conditions

    Returns:
        dict with keys:
            - 'cf': The CellFlow object
            - 'net': The trained model state
            - 'logs': Training logs
    """
    if output_dir is None:
        output_dir = "/tmp/cellflow_test"
    os.makedirs(output_dir, exist_ok=True)

    if checkpoint_dir is None:
        checkpoint_dir = f"{output_dir}/{run_id}"
    os.makedirs(checkpoint_dir, exist_ok=True)

    logger.info(f"Training on {len(train_conditions)} conditions: {train_conditions}")
    logger.info(f"Testing on {len(test_conditions)} conditions: {test_conditions}")

    key = jax.random.PRNGKey(0)
    d = pd.read_csv(f"{config.model.embedding.model_dir}/moa", index_col=0).to_dict(
        orient="list"
    )
    adata.uns["condition_embeddings"] = {k: np.array(v) for k, v in d.items()}
    logger.info(
        f"condition embeddings: {list(adata.uns['condition_embeddings'].keys())[:5]}"
    )
    # Sample train and test data
    if validate_with_conditions:
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
        actual_train_conditions = [train_conditions[i] for i in train_idx]
        val_conditions = [train_conditions[i] for i in valid_idx]
        adata_train, adata_valid = adata_conditional_subset(
            adata, actual_train_conditions, prng_key=key
        )
        adata_valid_cond, _ = adata_conditional_subset(
            adata, val_conditions, validation_split=0, prng_key=key
        )
        n_val_conditions = len(val_conditions)
    else:
        adata_train, adata_valid = adata_conditional_subset(
            adata, train_conditions, prng_key=key
        )
        n_val_conditions = len(train_conditions)
    adata_test, _ = adata_conditional_subset(
        adata, test_conditions, validation_split=0, prng_key=key
    )

    logger.debug(f"# train conditions: {len(adata_train.obs.condition.unique())}")
    if validate_with_conditions:
        logger.debug(
            f"# valid conditions: {len(adata_valid_cond.obs.condition.unique())}"
        )
    logger.debug(f"# test conditions: {len(adata_test.obs.condition.unique())}")

    # Create CellFlow object
    cf = CellFlow(adata_train, solver=solver_class)
    cf.prepare_data(
        sample_rep="X",
        control_key="is_control",
        perturbation_covariates={"condition": ("condition",)},
        perturbation_covariate_reps={"condition": "condition_embeddings"},
        max_combination_length=1,
    )

    # Prepare validation data
    if validate_with_conditions:
        cf.prepare_validation_data(
            adata_valid_cond,
            name="valid_cond",
            n_conditions_on_log_iteration=n_val_conditions,
            n_conditions_on_train_end=n_val_conditions,
            predict_kwargs={"max_steps": 16**5},
        )
    cf.prepare_validation_data(
        adata_valid,
        name="valid",
        n_conditions_on_log_iteration=None,
        n_conditions_on_train_end=None,
        predict_kwargs={"max_steps": 16**5},
    )
    cf.prepare_validation_data(
        adata_train,
        name="train",
        n_conditions_on_log_iteration=0,
        n_conditions_on_train_end=None,
        predict_kwargs={"max_steps": 16**5},
    )
    cf.prepare_validation_data(
        adata_test,
        name="test",
        n_conditions_on_log_iteration=0,
        n_conditions_on_train_end=None,
        predict_kwargs={"max_steps": 16**5},
    )

    # Prepare model
    _n_conds_per_batch = min(n_conds_per_batch, len(train_conditions))
    cf.prepare_model(**model_config)
    n_time_sample_per_step = n_time_samples_train if solver_class == "conical" else 1
    # Calculate training parameters
    num_iterations = int(
        n_time_samples_train
        / n_time_sample_per_step
        * len(train_conditions)
        * n_epochs
        // _n_conds_per_batch
    )
    logger.debug(
        f"solver_class: {solver_class}, n_time_sample_per_step: {n_time_sample_per_step}"
    )
    # Honour the caller's valid_freq. This used to be pinned to 1000 here, which
    # silently shadowed both the parameter and --valid-freq, so the flag was dead.
    # The default is still 1000 (= epoch_length * 100 → n_epochs/100 validation
    # checks), so runs that do not pass the flag are unchanged. It is worth
    # turning down: on array 15491193 each validation block cost 20-35s against
    # ~35s for the 1000 training steps it followed, i.e. ~40% of wall clock spent
    # in a host-bound callback that also drags GPU utilisation to the reaper's floor.
    log_path = f"{output_dir}/{run_id}_training_logs.pkl"

    metrics_callback = cellflow.training.Metrics(
        metrics=["r_squared", "mmd", "sinkhorn_div", "e_distance"],
        log_metrics=["sinkhorn_div"],
        call_loader_keys=(
            ["valid", "valid_cond", "test"]
            if validate_with_conditions
            else ["valid", "test"]
        ),
    )
    epoch_ckpt_callback = EpochCheckpointCallback(base_path=checkpoint_dir, keep_last=2)
    callbacks = [metrics_callback, epoch_ckpt_callback]

    net = None
    logs = None

    # Detect partial checkpoint for resume (crashed mid-training)
    import re as _re

    _parent_dir = os.path.dirname(checkpoint_dir)
    _base_name = os.path.basename(checkpoint_dir)
    _epoch_pattern = _re.compile(rf"^{_re.escape(_base_name)}-epoch(\d+)$")
    resume_candidates: list[tuple[int, str]] = []
    if resume and os.path.exists(_parent_dir):
        for name in os.listdir(_parent_dir):
            match = _epoch_pattern.match(name)
            if match:
                resume_candidates.append(
                    (int(match.group(1)), os.path.join(_parent_dir, name))
                )
    resume_candidates.sort(key=lambda x: x[0], reverse=True)

    resume_ckpt_dir = None
    resume_epoch = -1
    resume_cf = None
    for _cand_epoch, _cand_dir in resume_candidates:
        try:
            resume_cf = CellFlow.load(_cand_dir)
            resume_ckpt_dir = _cand_dir
            resume_epoch = _cand_epoch
            break
        except (EOFError, FileNotFoundError, pkl.UnpicklingError) as e:
            logger.warning(
                f"Failed to load checkpoint at epoch {_cand_epoch} "
                f"({type(e).__name__}: {e}); trying next candidate"
            )
    if resume and resume_candidates and resume_ckpt_dir is None:
        logger.warning(
            f"All {len(resume_candidates)} partial checkpoint(s) failed to load; "
            f"rerunning from iteration start"
        )

    # Train or load checkpoint
    if resume_ckpt_dir is not None:
        logger.info(
            f"Resuming from partial checkpoint at epoch {resume_epoch}: {resume_ckpt_dir}"
        )
        cf = resume_cf
        es_sidecar = f"{resume_ckpt_dir}_es_state.pkl"
        resume_state = None
        if os.path.exists(es_sidecar):
            try:
                with open(es_sidecar, "rb") as f:
                    resume_state = cloudpickle.load(f)
                logger.info(
                    f"Loaded early-stopping state from {es_sidecar} "
                    f"(waited={resume_state.get('waited')}, "
                    f"valid_loss={resume_state.get('valid_loss')})"
                )
            except Exception as e:
                logger.warning(f"Failed to load ES state from {es_sidecar}: {e}")
        cf.train(
            num_iterations=num_iterations,
            batch_size=1024,
            callbacks=callbacks,
            valid_freq=valid_freq,
            n_conditions_per_batch=_n_conds_per_batch,
            out_of_core_dataloading=True,
            profile=False,
            epoch_length=n_time_samples_train
            / n_time_sample_per_step
            * len(train_conditions)
            // _n_conds_per_batch,
            early_stopping=early_stopping_patience is not None,
            min_epochs=min_epochs,
            early_stopping_patience=early_stopping_patience,
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            min_lr=min_lr,
            resume_state=resume_state,
            initial_epoch_offset=resume_epoch,
        )
        with open(log_path, "wb") as f:
            pkl.dump(cf.trainer.training_logs, f)
        net = cf.solver.vf_state_inference
        cf.save(checkpoint_dir, overwrite=True)
        logger.info(f"Resumed training completed and saved to {checkpoint_dir}")
    elif not (os.path.exists(checkpoint_dir) and os.path.exists(log_path)) or rerun:
        logger.info("Starting training...")
        cf.train(
            num_iterations=num_iterations,
            batch_size=1024,
            callbacks=callbacks,
            valid_freq=valid_freq,
            n_conditions_per_batch=_n_conds_per_batch,
            out_of_core_dataloading=True,
            profile=False,
            epoch_length=n_time_samples_train
            / n_time_sample_per_step
            * len(train_conditions)
            // _n_conds_per_batch,
            early_stopping=early_stopping_patience is not None,
            min_epochs=min_epochs,
            early_stopping_patience=early_stopping_patience,  # 20 for imagenet
            lr_patience=lr_patience,
            lr_factor=lr_factor,
            min_lr=min_lr,
        )
        with open(log_path, "wb") as f:
            pkl.dump(cf.trainer.training_logs, f)
        net = (
            cf.solver.vf_state_inference
        )  # calls vf_state that is upated every time vf_state_fn step is taken
        cf.save(checkpoint_dir, overwrite=True)
        logger.info(f"Training completed and saved to {checkpoint_dir}")
    else:
        try:
            # Find checkpoint_dir with highest epoch number
            import re

            parent_dir = os.path.dirname(checkpoint_dir)
            base_name = os.path.basename(checkpoint_dir)
            pattern = re.compile(rf"^{re.escape(base_name)}-epoch(\d+)$")

            if os.path.exists(f"{checkpoint_dir.rstrip('/')}-final"):
                best_epoch = 200
                best_ckpt_dir = f"{checkpoint_dir.rstrip('/')}-final"
            else:
                best_epoch = -1
                best_ckpt_dir = checkpoint_dir

                if os.path.exists(parent_dir):
                    for name in os.listdir(parent_dir):
                        match = pattern.match(name)
                        if match:
                            epoch = int(match.group(1))
                            if epoch > best_epoch:
                                best_epoch = epoch
                                best_ckpt_dir = os.path.join(parent_dir, name)

            if best_epoch >= 0:
                logger.info(f"Found checkpoint at epoch {best_epoch}: {best_ckpt_dir}")

            cf = CellFlow.load(best_ckpt_dir)
            net = cf.solver.vf_state_inference
            logger.info(f"Loaded existing checkpoint from {best_ckpt_dir}")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, retraining...")
            cf.train(
                num_iterations=num_iterations,
                batch_size=1024,
                callbacks=callbacks,
                valid_freq=valid_freq,
                n_conditions_per_batch=_n_conds_per_batch,
                out_of_core_dataloading=True,
                epoch_length=n_time_samples_train
                / n_time_sample_per_step
                * len(train_conditions)
                // _n_conds_per_batch,
                early_stopping=True,
                min_epochs=min_epochs,
                early_stopping_patience=early_stopping_patience,  # 20 for imagenet
                lr_patience=lr_patience,
                lr_factor=lr_factor,
                min_lr=min_lr,
            )
            with open(log_path, "wb") as f:
                pkl.dump(cf.trainer.training_logs, f)
            net = cf.solver.vf_state_inference
            cf.save(checkpoint_dir, overwrite=True)

    # Clean up
    jax.clear_caches()
    gc.collect()

    return {
        "cf": cf,
        "net": net,
        "logs": logs,
    }


def get_batch_cflow(
    all_conditions: List[str],
    unseen_conditions: List[str],
    trained_conditions: List[str],
    strategy: Literal[
        "random", "random_fixed", "tntk",
    ],
    batch_size: int,
    state=None,
    cond_arr_dict=None,
    x0=None,
    random_seed=42,
    batch_selection_strategy: Literal[
        "coreset", "uncertainty", "kmeanspp", "lcmd", "typiclust2"
    ] = "coreset",
    transform_kernel=True,
    mean_dict=None,
    val_mean_dict=None,
    dim_hidden=None,
    res_weight=1.0,
    acquisition_seed=None,
):
    # Started before the dispatch, not inside one branch: every strategy except
    # NO_KERNEL_STRATEGIES logs kernel_s below, so assigning this only in the
    # `else` arm made `ntk_sizematched` and `x` die with UnboundLocalError the
    # moment they reached that log line (observed 2026-08-07, array 15450084:
    # 27 nsb + 27 xb tasks lost, zero failures in grad/mean/random_fixed).
    # Hoisting it also means those two finally get a kernel timing at all.
    _t_kernel = time.perf_counter()
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
    elif strategy in ("ntk", "ntk_sizematched"):
        # `ntk_sizematched` is a distinct strategy (saved under its own name) that
        # sizes the NTK MeanPredictor's hidden trunk to match the cflow model
        # (its `dim_hidden`, used as the flow's decoder_dims). Plain `ntk` keeps
        # the default two-layer [d, d] (d = output-layer input dim).
        cond_dict = {c: cond_arr_dict[c].squeeze() for c in all_conditions}
        if strategy == "ntk_sizematched":
            if dim_hidden is None:
                raise ValueError(
                    "strategy='ntk_sizematched' requires dim_hidden (the cflow "
                    "model's hidden dims) to be passed to get_batch_cflow."
                )
            hidden_dims = list(dim_hidden)
        else:
            d = state.params["output_layer"]["kernel"].shape[0]
            hidden_dims = [d, d]
        kernel_mat, _ = get_ntk_matrix(
            cond_dict,
            mean_dict,
            val_mean_dict,
            trained_conditions,
            all_conditions,
            hidden_dims=hidden_dims,
        )
    else:
        _t_kernel = time.perf_counter()
        kernel_mat = get_kernels_cflow(
            all_conditions,
            cond_arr_dict,
            state,
            x0,
            kernel_type=strategy,
            res_weight=res_weight,
            mean_dict=mean_dict,
            val_mean_dict=val_mean_dict,
            trained_conditions=trained_conditions,
            dim_hidden=dim_hidden,
        )
    if strategy not in NO_KERNEL_STRATEGIES:
        try:
            jax.block_until_ready(kernel_mat)
        except Exception:
            pass
        logger.info(f"[TIMING] kernel_s={time.perf_counter() - _t_kernel:.4f}")
        _t_sel = time.perf_counter()
        batch = select_batch(
            trained_conditions=trained_conditions,
            all_conditions=all_conditions,
            batch_size=min(batch_size, len(unseen_conditions)),
            kernel_matrix=kernel_mat,
            batch_selection_strategy=batch_selection_strategy,
            random_seed=random_seed,
        )
        logger.info(f"[TIMING] selection_s={time.perf_counter() - _t_sel:.4f}")
    logger.debug(f"Selected batch: {batch}")
    return batch, kernel_mat


model_configs = {
    "s": dict(dim_hidden=[64, 64, 32], dim_data_map=[32, 32], dim_cond_map=[32, 32]),
    "l": dict(
        dim_hidden=[128, 128, 128, 32], dim_data_map=[64, 64], dim_cond_map=[64, 64]
    ),
}

if __name__ == "__main__":
    run_in_gpu = False

    _platform = jax.devices()[0].platform
    if _platform == "gpu":
        run_in_gpu = True
    elif _platform == "tpu":
        # Every kernel here ends in a float32 Gram computed on device --
        # get_tntk_kernel_matrix's einsum, get_projected_grad_kernel_matrix's
        # `all_cond_mat @ all_cond_mat.T`, and the weighted/ntk variants. On TPU
        # a float32 matmul defaults to bfloat16 passes, which would quietly
        # degrade those Grams and therefore the batch `select_batch` picks from
        # them; downstream selection is already numpy float64
        # (acpl/selection.py) and cannot recover precision the Gram never had.
        # CPU and GPU are unaffected: their DEFAULT precision is already
        # float32, so this branch is a no-op for every non-TPU run and cannot
        # perturb results produced on the cluster.
        jax.config.update("jax_default_matmul_precision", "highest")
        logger.info("TPU backend: jax_default_matmul_precision=highest")
    parser = get_parser()
    parser.add_argument(
        "--cpu", action="store_true", help="Run training on CPU instead of GPU"
    )
    parser.add_argument(
        "--n-conds", default=1, type=int, help="number of conditions per batch"
    )
    parser.add_argument(
        "--conical-path",
        action="store_true",
        help="Ignore source samples and use conical path",
    )
    parser.add_argument(
        "--cond-dropout", default=0.0, type=float, help="Condition dropout rate"
    )
    parser.add_argument(
        "--sigma",
        default=0.1,
        type=float,
        help="Standard deviation for conical path noise",
    )
    parser.add_argument(
        "--valid-freq",
        default=1000,
        type=int,
        help="Frequency of validation steps",
    )
    parser.add_argument(
        "--early-stopping-patience",
        default=None,
        type=int,
        help="Number of epochs with no improvement after which training will be stopped. If not set, early stopping is disabled.",
    )
    parser.add_argument(
        "--weight-decay",
        default=0.1,
        type=float,
        help=(
            "Decoupled weight decay for AdamW. 0 selects plain Adam, which is "
            "what every run before 2026-08-25 used: the optimizer was built as "
            "optax.adam and the `optim.kwargs.weight_decay: 1.0e-5` in the "
            "dataset YAMLs was never read by this trainer (that block is "
            "consumed by the cmonge trainers, which do call optax.adamw)."
        ),
    )
    parser.add_argument(
        "--condition-mode",
        default="stochastic",
        choices=["stochastic", "deterministic"],
        help=(
            "How the condition embedding is produced. 'stochastic' (the "
            "hardcoded behaviour before 2026-08-29) DRAWS the embedding from a "
            "Gaussian on every forward pass; 'deterministic' uses its mean. "
            "The 2026-08-28 fit sweep found the sampling -- not the KL weight -- "
            "is what blurs the pushforward: over 20 paired selections, "
            "deterministic improved the training-condition fit by 35%% on "
            "ring2stdn (t=-5.8, 18/20) and 76%% on vanderpolstdn (t=-5.4, "
            "19/20), while merely lowering --regularization to 0.01 did nothing "
            "(-0.6%% and +0.1%%)."
        ),
    )
    parser.add_argument(
        "--regularization",
        default=1.0,
        type=float,
        help=(
            "KL weight in stochastic mode, L2 weight in deterministic mode "
            "(CellFlow-al/.../model/_cellflow.py:325). CellFlow ASSERTS >0 for "
            "stochastic mode, so 0.0 is only valid with "
            "--condition-mode deterministic."
        ),
    )
    parser.add_argument(
        "--lr-patience",
        default=None,
        type=int,
        help="Validation checks with no improvement after which LR is reduced. If not set, adaptive LR is disabled.",
    )
    parser.add_argument(
        "--lr-factor",
        default=0.5,
        type=float,
        help="Multiplicative factor applied to LR on plateau.",
    )
    parser.add_argument(
        "--min-lr",
        default=1e-6,
        type=float,
        help="Lower bound on LR for adaptive LR reduction.",
    )
    parser.add_argument(
        "--n-epochs", default=1000, type=int, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--min-epochs",
        default=100,
        type=int,
        help="Minimum number of epochs before early stopping can trigger",
    )
    parser.add_argument(
        "--fold",
        default=None,
        type=int,
        help="If set, only run the specified fold index (0-4). Otherwise run all folds sequentially.",
    )
    parser.add_argument(
        "--ignore-lock",
        action="store_true",
        help="Ignore existing .running lock files (use to recover from stale locks after a crash).",
    )
    parser.add_argument(
        "--iter0-only",
        action="store_true",
        help="Train only iter 0 then exit. Used to seed the donor-strategy "
        "(random_fixed, or legacy random) iter 0 checkpoints that other "
        "strategies symlink to.",
    )
    parser.add_argument(
        "--iter0-prefix",
        default=None,
        help="Alternate --prefix to source the shared iter 0 checkpoint from. "
        "Iter 0 is strategy/kernel-independent, so e.g. v3 mean/grad runs can "
        "reuse the random_fixed (or legacy random) iter 0 under a v2 prefix. "
        "Falls back to --prefix when unset or when the alt copy is missing.",
    )
    args = parser.parse_args()
    if args.fold is not None and not (0 <= args.fold < 5):
        parser.error("--fold must be in [0, 5)")
    n_conds_per_batch = args.n_conds
    max_iter = args.max_iter if args.max_iter is not None else 3
    n_epochs = args.n_epochs
    n_time_samples_train = 100
    n_time_samples_kernel_inference = 100
    config = load_config(args.config)
    # TODO: Load cmonge config and translate to the following.

    if args.conical_path:
        match_fn = None
    else:
        match_fn = functools.partial(
            labeled_match_linear if n_conds_per_batch > 1 else match_linear,
            epsilon=0.05,
            tau_a=1.0,
            tau_b=1.0,
        )

    adata = ad.read_h5ad(config.data.file_path)
    adata.obs["condition"] = adata.obs[config.data.drug_col]
    adata.obs["is_control"] = adata.obs["condition"] == control_map[args.data]
    logger.info(f"# control cells = {adata.obs['is_control'].sum()}")

    control_cells = adata[adata.obs["is_control"]].X
    control_std = jnp.std(jnp.array(control_cells), axis=0).min()
    logger.info(f"Control cells min std: {control_std}")
    n_time_sample_per_step = n_time_samples_train if args.conical_path else 1
    model_config = dict(
        condition_mode=args.condition_mode,
        regularization=args.regularization,
        condition_embedding_dim=model_configs[args.model_config_id]["dim_cond_map"][-1],
        cond_output_dropout=args.cond_dropout,
        condition_encoder_kwargs={},
        pooling="mean",
        pool_sample_covariates=False,
        time_freqs=model_configs[args.model_config_id]["dim_hidden"][-1],
        time_encoder_dims=model_configs[args.model_config_id]["dim_hidden"][-2:],
        time_encoder_dropout=0.0,
        hidden_dims=model_configs[args.model_config_id]["dim_data_map"],
        hidden_dropout=0.0,
        conditioning="concatenation",
        decoder_dims=model_configs[args.model_config_id]["dim_hidden"],
        vf_act_fn=nn.silu,
        vf_kwargs=None,
        probability_path=(
            {"conical": args.sigma} if args.conical_path else {"constant_noise": 0.0}
        ),
        match_fn=match_fn,
        # adamw when weight decay is on, adam when it is off. NOT `adamw(0.0)`
        # unconditionally: optax's adamw adds a decay term to the update even at
        # 0.0, and keeping the exact adam code path at wd=0 means reruns of the
        # pre-2026-08-25 sweeps stay bit-comparable with what is already on disk.
        # inject_hyperparams still wraps it so --lr-patience can keep writing
        # state.hyperparams['learning_rate'] on plateau.
        optimizer=(
            optax.inject_hyperparams(optax.adamw)(
                learning_rate=args.lr, weight_decay=args.weight_decay
            )
            if args.weight_decay
            else optax.inject_hyperparams(optax.adam)(learning_rate=args.lr)
        ),
        solver_kwargs=(
            {}
            # {"n_time_per_sample": n_time_samples_train} if args.conical_path else {}
        ),
        layers_after_pool={
            "layer_type": "mlp",
            "dims": model_configs[args.model_config_id]["dim_cond_map"],
        },
        layer_norm_before_concatenation=False,
        linear_projection_before_concatenation=False,
        sample_multiple_conditions_per_batch=n_conds_per_batch > 1,
    )
    d = pd.read_csv(f"{config.model.embedding.model_dir}/moa", index_col=0).to_dict(
        orient="list"
    )
    adata.uns["condition_embeddings"] = {k: np.array(v) for k, v in d.items()}
    logger.info(
        f"condition embeddings: {list(adata.uns['condition_embeddings'].keys())[:5]}"
    )

    logger.info(f"Using active learning strategy: {args.strategy}")
    conditions = pd.read_csv(condition_map[args.data], index_col=None, header=None)[
        0
    ].tolist()
    conditions.remove(control_map[args.data])
    logger.info(f"Experiment: Training model on {conditions}")

    if args.init_batch_size != 20:
        init_batch_label = f"_ib{args.init_batch_size}"
    else:
        init_batch_label = ""

    if args.batch_size > args.init_batch_size:
        args.init_batch_size = args.batch_size

    key = jax.random.PRNGKey(args.split_seed)
    kf = KFold(n_splits=5, random_state=args.split_seed, shuffle=True)

    for i, (train_idx, test_idx) in enumerate(kf.split(conditions)):
        if args.fold is not None and i != args.fold:
            continue
        prefix = (
            f"./models/cellflow-al/models/{args.data}/{args.prefix}/"
        )

        lr_patience_label = (
            f"_lrp{args.lr_patience}" if args.lr_patience is not None else ""
        )
        # Weight decay MUST be in the run id. Without it a wd=0.1 run writes to
        # the same checkpoint/log path as the wd=0 run of the same cell and
        # silently overwrites it -- two different optimizers, one directory, and
        # nothing in the collected CSV to tell them apart. Empty at wd=0 so every
        # pre-2026-08-25 path is unchanged (same reason lr_patience_label is
        # empty when --lr-patience is unset). Formatted with %g so 0.1 -> "wd0.1"
        # rather than "wd0.1000000000000000055".
        wd_label = f"_wd{args.weight_decay:g}" if args.weight_decay else ""
        # Condition-encoder label. Empty in the default stochastic/1.0 case, so
        # every path written before 2026-08-29 is byte-identical and nothing on
        # disk is orphaned. A deterministic run MUST NOT share a directory with a
        # stochastic one: they differ by 35-76% on the training-condition fit,
        # which is far larger than the seed spread, so an unlabelled mix would
        # look like enormous run-to-run noise.
        if args.condition_mode == "deterministic":
            cond_label = "_det" if args.regularization == 0.0 else f"_det{args.regularization:g}"
        else:
            cond_label = "" if args.regularization == 1.0 else f"_reg{args.regularization:g}"
        exp_id = f"{'gp_' if args.transform_kernel else ''}{args.model_config_id}_{args.strategy}{'_' + args.batch_selection_strategy if args.batch_selection_strategy != 'coreset' else ''}_size{args.batch_size}_fold{i}_seed{args.init_seed}{lr_patience_label}{wd_label}{cond_label}"
        # v2: with validation with conditions instead of samples

        # Check if last iteration already exists - if so, skip this fold
        last_iter_log = f"{prefix}/{exp_id}_iter{max_iter}_training_logs.pkl"
        if os.path.exists(last_iter_log) and not args.rerun:
            logger.info(
                f"Fold {i} already completed (found {last_iter_log}), skipping..."
            )
            continue

        # Acquire a per-fold lock to prevent concurrent runs from clobbering
        # each other's intermediate checkpoints/logs.
        os.makedirs(prefix, exist_ok=True)
        lock_path = f"{prefix}/{exp_id}.running"
        fd = _acquire_fold_lock(lock_path, i, args.ignore_lock)
        if fd is None:
            continue
        with os.fdopen(fd, "w") as _f:
            import datetime, socket

            _f.write(
                f"pid={os.getpid()} host={socket.gethostname()} "
                f"pid={os.getpid()} "
                f"started={datetime.datetime.now().isoformat()}\n"
            )

        try:
            # Log split
            log_path = f"{prefix}/{exp_id}.json"
            os.makedirs(os.path.dirname(log_path), exist_ok=True)
            with open(log_path, "w") as f:
                json.dump(
                    {"train_idx": train_idx.tolist(), "test_idx": test_idx.tolist()}, f
                )
            shared_train_val_conditions = [conditions[i] for i in train_idx]
            heldout_test_conditions = [conditions[i] for i in test_idx]

            # Sample test cells
            adata_test, _ = adata_conditional_subset(
                adata, heldout_test_conditions, validation_split=0, prng_key=key
            )

            if args.strategy not in NO_KERNEL_STRATEGIES:
                cf_all = CellFlow(
                    adata, solver="otfm" if not args.conical_path else "conical"
                )
                cf_all.prepare_data(
                    sample_rep="X",
                    control_key="is_control",
                    perturbation_covariates={"condition": ("condition",)},
                    perturbation_covariate_reps={"condition": "condition_embeddings"},
                    max_combination_length=1,
                )
                train_sampler = TrainSampler(
                    data=cf_all.train_data,
                    batch_size=args.batch_size,
                )
                logger.debug(
                    f"ID_CF_ALL:{cf_all.train_data.perturbation_covariates_mask}"
                )
                logger.debug(
                    f"ID_CF_ALL:{cf_all.train_data.perturbation_idx_to_covariates}"
                )
                logger.debug(f"ID_CF_ALL:{cf_all.train_data.perturbation_idx_to_id}")
                logger.debug(f"ID_CF_ALL:{cf_all.train_data.condition_data}")
                x0, cond_arr_dict, mean_dict, val_mean_dict = prepare_data_cond(
                    train_sampler,
                    shared_train_val_conditions,
                    random_seed=args.split_seed,
                    sample_size=4096,
                )
                if args.conical_path:
                    # For conical path, we want to ignore the source distribution and use a standard normal x0 for all conditions
                    key, subkey = jax.random.split(key)
                    x0 = (
                        jax.random.normal(subkey, shape=(2**12, x0.shape[1]))
                        * args.sigma
                    )
            else:
                x0 = None
                cond_arr_dict = None
                mean_dict = None
                val_mean_dict = None
            unlabeled_conditions = shared_train_val_conditions.copy()
            al_iter = 0
            train_conditions = []

            net = None

            while len(unlabeled_conditions) > 0 and al_iter <= max_iter:
                # Single active learning iteration
                # Initial batch
                logger.info(f"Running fold {i} iter {al_iter}")
                run_id = f"{exp_id}_iter{al_iter}"
                checkpoint_dir = f"{prefix}/{run_id}"
                os.makedirs(checkpoint_dir, exist_ok=True)

                kernel_logger_path = f"{prefix}/{run_id}_kernels.pkl"
                reused_batch_from_pickle = False
                batch = None
                kernel_mat = None
                if os.path.exists(kernel_logger_path):
                    try:
                        with open(kernel_logger_path, "rb") as f:
                            _saved_kernels = pkl.load(f)
                        batch = list(_saved_kernels["selected_conditions"])
                        kernel_mat = _saved_kernels.get("kernel_matrix")
                        reused_batch_from_pickle = True
                        logger.info(
                            f"Reusing selected batch for iter {al_iter} from "
                            f"{kernel_logger_path}: {batch}"
                        )
                    except (EOFError, pkl.UnpicklingError, KeyError) as e:
                        logger.warning(
                            f"Failed to load {kernel_logger_path} "
                            f"({type(e).__name__}: {e}); re-running selection"
                        )
                        batch = None
                        kernel_mat = None

                if not reused_batch_from_pickle:
                    if al_iter == 0:
                        batch, kernel_mat = get_batch_cflow(
                            all_conditions=shared_train_val_conditions,
                            unseen_conditions=unlabeled_conditions,
                            trained_conditions=[],
                            strategy="random",
                            batch_size=args.init_batch_size,
                            random_seed=args.init_seed,
                            transform_kernel=args.transform_kernel,
                        )
                    else:
                        res_weight = 1.0
                        batch, kernel_mat = get_batch_cflow(
                            all_conditions=shared_train_val_conditions,
                            unseen_conditions=unlabeled_conditions,
                            trained_conditions=train_conditions,
                            strategy=args.strategy,
                            batch_size=args.batch_size,
                            state=net,
                            cond_arr_dict=cond_arr_dict,
                            x0=x0,
                            batch_selection_strategy=args.batch_selection_strategy,
                            transform_kernel=args.transform_kernel,
                            mean_dict=mean_dict,
                            val_mean_dict=val_mean_dict,
                            dim_hidden=model_configs[args.model_config_id]["dim_hidden"],
                            res_weight=res_weight,
                            acquisition_seed=acquisition_seed_for(
                                args.init_seed, i, al_iter
                            ),
                        )
                logger.info(f"Unlabeled conditions:{unlabeled_conditions}")
                for x in batch:
                    try:
                        unlabeled_conditions.remove(x)
                    except ValueError as e:
                        logger.info(
                            f"{x} among selected batch {batch} not in Unlabeled conditions:{unlabeled_conditions}"
                        )
                        raise e
                if not reused_batch_from_pickle:
                    with open(kernel_logger_path, "wb") as f:
                        pkl.dump(
                            {
                                "seen_conditions": train_conditions,
                                "selected_conditions": batch,
                                "kernel_matrix": kernel_mat,
                            },
                            f,
                        )
                train_conditions += batch
                assert len(train_conditions) == len(set(train_conditions))

                # Iter 0 is strategy-independent (random init batch, seeded by
                # init_seed), so every arm loads one shared iter 0 rather than
                # training its own -- that keeps GPU non-determinism from
                # drifting the starting point across strategies. New runs source
                # it from `random_fixed`, with the legacy `random` arm as
                # fallback donor for prefixes whose only iter 0 on disk predates
                # it (see ITER0_DONOR_STRATEGIES). A run that already has its own
                # iter 0 on disk keeps it; when no donor copy exists at all, the
                # donor arms train it (that is what --iter0-only submits) and
                # every other strategy stops rather than silently forking a
                # second starting point.
                # Tests for checkpoint CONTENT, not for the directory:
                # os.makedirs(checkpoint_dir, exist_ok=True) runs unconditionally
                # ~100 lines above, so a bare os.path.exists(checkpoint_dir) is
                # always True and made this entire branch unreachable -- every
                # arm silently retrained its own iter 0 and the --iter0-prefix
                # donor was never loaded. Reaching this branch WITH the empty
                # directory present is the intended path: the body below rmdir()s
                # exactly that directory before symlinking the donor's checkpoint
                # into place.
                if al_iter == 0 and not (
                    (os.path.isdir(checkpoint_dir) and os.listdir(checkpoint_dir))
                    or os.path.exists(f"{checkpoint_dir.rstrip('/')}-final")
                ):
                    # Iter 0 is strategy/kernel-independent, so it may be sourced
                    # from an alternate prefix (e.g. v3 runs reusing the v2
                    # random iter 0). When --iter0-prefix is set, load from there
                    # first; fall back to this run's own prefix only when the alt
                    # copy is absent (e.g. a different model config under v2).
                    cand_prefix_dirs = []
                    if args.iter0_prefix:
                        cand_prefix_dirs.append(
                            f"./models/cellflow-al/models/"
                            f"{args.data}/{args.iter0_prefix}/"
                        )
                    cand_prefix_dirs.append(prefix)
                    cand = cand_run_id = None
                    cand_log = cand_ckpt = cand_ckpt_final = None
                    searched = []
                    for cand_strategy in ITER0_DONOR_STRATEGIES:
                        # Iter 0 is independent of the ACQUISITION RULE as well
                        # as of the kernel: the initial batch comes from
                        # sample_random_batch(--init-seed) and no selection runs
                        # at iteration 0. Interpolating THIS run's rule into the
                        # donor id therefore asks for a donor that cannot exist
                        # -- an arm with a non-default rule looked for
                        # gp_l_random_fixed_<rule>_..., while the donor the
                        # NTK arm wrote is gp_l_random_fixed_minvar_.... Every
                        # task of array 15588397 aborted at fold 0 on exactly
                        # that. This run's own rule is still tried FIRST, so a
                        # minvar run resolves to the donor it always did; any other
                        # rule found on disk is accepted only as a fallback.
                        c_head = (
                            f"{'gp_' if args.transform_kernel else ''}"
                            f"{args.model_config_id}_{cand_strategy}"
                        )
                        c_tail = (
                            f"_size{args.batch_size}"
                            f"_fold{i}_seed{args.init_seed}"
                            f"{lr_patience_label}"
                            # Donor must share the optimizer. iter0 IS the model
                            # every later iteration starts from, so pairing a
                            # wd=0.1 main run with a wd=0 donor would silently
                            # mix optimizers within one AL trajectory. Matching
                            # the label means a new weight decay needs its own
                            # iter0 -- deliberate, not an oversight.
                            f"{wd_label}"
                            # Donor must share the condition encoder for the same
                            # reason it must share the optimizer: iter0 IS the
                            # model every later iteration starts from, so a
                            # deterministic main run resuming from a stochastic
                            # donor would silently mix the two inside one AL
                            # trajectory. A new condition mode therefore needs its
                            # own iter0 -- deliberate, not an oversight.
                            f"{cond_label}"
                        )
                        c_own = (
                            f"_{args.batch_selection_strategy}"
                            if args.batch_selection_strategy != "coreset"
                            else ""
                        )
                        c_exp_ids = [f"{c_head}{c_own}{c_tail}"]
                        for cand_dir in cand_prefix_dirs:
                            try:
                                entries = os.listdir(cand_dir)
                            except OSError:
                                continue
                            for entry in entries:
                                for suf in ("_iter0-final", "_iter0"):
                                    if not entry.endswith(suf):
                                        continue
                                    stem = entry[: -len(suf)]
                                    if not (
                                        stem.startswith(c_head)
                                        and stem.endswith(c_tail)
                                    ):
                                        break
                                    # The rule sits between head and tail. No
                                    # acquisition rule contains "_", so this
                                    # also stops the `random` donor from
                                    # matching a `random_fixed` directory
                                    # (mid would be "_fixed_minvar").
                                    mid = stem[len(c_head) : len(stem) - len(c_tail)]
                                    if mid and not (
                                        mid.startswith("_") and "_" not in mid[1:]
                                    ):
                                        break
                                    if stem not in c_exp_ids:
                                        c_exp_ids.append(stem)
                                    break
                        for c_exp_id in c_exp_ids:
                            c_run_id = f"{c_exp_id}_iter0"
                            for cand_dir in cand_prefix_dirs:
                                c_log = f"{cand_dir}/{c_run_id}_training_logs.pkl"
                                c_ckpt = f"{cand_dir}/{c_run_id}"
                                c_ckpt_final = f"{c_ckpt}-final"
                                searched.append(c_ckpt)
                                if os.path.exists(c_log) and (
                                    os.path.exists(c_ckpt_final)
                                    or os.path.exists(c_ckpt)
                                ):
                                    cand, cand_run_id = cand_strategy, c_run_id
                                    cand_log, cand_ckpt, cand_ckpt_final = (
                                        c_log,
                                        c_ckpt,
                                        c_ckpt_final,
                                    )
                                    break
                            if cand_log is not None:
                                break
                        if cand_log is not None:
                            break
                    if cand_log is not None:
                        own_log = f"{prefix}/{run_id}_training_logs.pkl"
                        own_ckpt_final = f"{checkpoint_dir.rstrip('/')}-final"
                        if not os.path.exists(own_log):
                            os.symlink(cand_log, own_log)
                        src_ckpt = (
                            cand_ckpt_final
                            if os.path.exists(cand_ckpt_final)
                            else cand_ckpt
                        )
                        dst_ckpt = (
                            own_ckpt_final
                            if src_ckpt == cand_ckpt_final
                            else checkpoint_dir
                        )
                        if (
                            dst_ckpt == checkpoint_dir
                            and os.path.isdir(checkpoint_dir)
                            and not os.listdir(checkpoint_dir)
                        ):
                            os.rmdir(checkpoint_dir)
                        if not os.path.exists(dst_ckpt):
                            os.symlink(src_ckpt, dst_ckpt)
                        logger.info(f"Reusing iter 0 from {cand} run: {cand_run_id}")
                    elif args.strategy in ITER0_DONOR_STRATEGIES:
                        logger.info(
                            f"No shared iter 0 on disk yet; this '{args.strategy}' "
                            f"donor run trains it (looked for: {', '.join(searched)})"
                        )
                    else:
                        donors = "/".join(ITER0_DONOR_STRATEGIES)
                        raise FileNotFoundError(
                            f"Iter 0 must be loaded from a '{donors}' strategy run, "
                            f"but no completed one was found (training_logs.pkl + "
                            f"ckpt/-final). Looked for: {', '.join(searched)}. Run "
                            f"the '{ITER0_DONOR_STRATEGIES[0]}' strategy first, or "
                            f"change this behavior."
                        )

                # Run single iteration

                # if run_in_gpu:
                _t_fit = time.perf_counter()
                result = run_single_iteration(
                    adata=adata,
                    train_conditions=train_conditions,
                    test_conditions=heldout_test_conditions,
                    config=config,
                    model_config=model_config,
                    n_time_samples_train=n_time_samples_train,
                    n_epochs=n_epochs,
                    n_conds_per_batch=n_conds_per_batch,
                    output_dir=prefix,
                    run_id=run_id,
                    checkpoint_dir=checkpoint_dir,
                    rerun=args.rerun,
                    solver_class="conical" if args.conical_path else "otfm",
                    early_stopping_patience=args.early_stopping_patience,
                    min_epochs=args.min_epochs,
                    valid_freq=args.valid_freq,
                    resume=args.ignore_lock,
                    lr_patience=args.lr_patience,
                    lr_factor=args.lr_factor,
                    min_lr=args.min_lr,
                )
                logger.info(f"[TIMING] fit_s={time.perf_counter() - _t_fit:.4f}")

                jax.clear_caches()
                gc.collect()

                net = result["net"]
                al_iter += 1

                if args.iter0_only:
                    logger.info("--iter0-only: completed iter 0, stopping")
                    break
        finally:
            try:
                os.remove(lock_path)
            except OSError:
                pass
            _ACTIVE_LOCKS.discard(lock_path)
    jax.clear_caches()
    os._exit(0)
