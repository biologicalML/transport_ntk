import pandas as pd
from loguru import logger
import jax
import os
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import yaml
from yaml import CLoader as Loader
from dotmap import DotMap
import numpy as np
import pickle as pkl
import jaxlib

matplotlib.rcParams["pdf.fonttype"] = 42
results_prefix = "results/cmonge"

# An "arm" is a (kernel, selection rule) pair. `ntk` is the size-matched NTK
# kernel (`ntk_sizematched`) under `minvar`; a name that is not
# listed is taken to be a kernel collected under whatever
# --batch-selection-strategy says.
#
#   name          kernel              selection rule
ARM_SPECS = {
    "ntk": ("ntk_sizematched", "minvar"),
}
# Arms whose runs carry no kernel at all.
NO_KERNEL_STRATEGIES = ("random", "random_fixed")
# The camera-ready row set.
DEFAULT_STRATEGIES = ["random_fixed", "ntk", "tntk"]


def resolve_arm(strategy, batch_selection_strategy="minvar"):
    """(kernel, selection-rule suffix, run-id tag) for one arm name."""
    if strategy in ARM_SPECS:
        kernel, bss = ARM_SPECS[strategy]
        transform = True
    else:
        kernel, bss = strategy, batch_selection_strategy
        transform = bss == "minvar" and strategy not in NO_KERNEL_STRATEGIES
    # Runs that transform the kernel into a GP posterior are written under the
    # `algp_` tag, the rest under `al0_`.
    al_tag = "algp" if transform else "al0"
    # The run id carries the rule as a suffix only when it is not the default.
    return kernel, (f"{bss}_" if bss else ""), al_tag


def load_config(path, use_cache=True):
    """Load config with optional pickle caching for faster subsequent loads.

    Supports multiple file formats:
    - Direct .pkl file: loads directly from pickle
    - YAML file ending with '_log' or '/log'
    - Will check for .pkl cache automatically

    Args:
        path: Path to log file (can be .pkl, _log, or /log)
        use_cache: If True, cache parsed logs as pickle files
    """
    # Determine actual file paths to check
    possible_paths = []

    # Check if path already ends with .pkl
    if path.endswith(".pkl"):
        possible_paths.append((path, "pkl"))
    else:
        # Try .pkl version first (fastest)
        possible_paths.append((path + ".pkl", "pkl"))
        possible_paths.append((path.rstrip("/") + ".pkl", "pkl"))

        # Then try YAML paths
        possible_paths.append((path, "yaml"))
        if not path.endswith("_log") and not path.endswith("/log"):
            possible_paths.append((path + "_log", "yaml"))
            possible_paths.append((path + "/log", "yaml"))

    # Try to load from the first existing file
    yaml_path = None
    for file_path, file_type in possible_paths:
        if os.path.exists(file_path):
            if file_type == "pkl":
                # Load directly from pickle
                try:
                    with open(file_path, "rb") as f:
                        return pkl.load(f)
                except Exception as e:
                    print(f"Warning: Failed to load pickle {file_path}: {e}")
                    # Continue to try other paths
                    continue
            else:
                # Found YAML file
                yaml_path = file_path
                break

    if yaml_path is None:
        raise FileNotFoundError(
            f"Could not find log file at {path} or any variant (.pkl, _log, /log)"
        )

    # Parse YAML file (slow)
    with open(yaml_path, "r") as file:
        yaml_data = yaml.load(file, Loader=Loader)

    result = DotMap(yaml_data)

    # Save to cache for next time
    if use_cache:
        cache_path = yaml_path + ".pkl"
        try:
            with open(cache_path, "wb") as f:
                pkl.dump(result, f, protocol=pkl.HIGHEST_PROTOCOL)
        except Exception as e:
            print(f"Warning: Failed to save cache {cache_path}: {e}")

    return result


def extract_required_fields(log, max_conds_with_particles=3):
    """Extract only required fields from log to save memory.

    Returns a minimal log containing only:
    - experiments[-1]["test-set"] with mean_statistics and point_clouds
    - experiments[0]["ott-logs"]["eval"]["fitting_loss"] (if exists)
    """
    minimal_log = {"experiments": []}

    if len(log["experiments"]) == 0:
        return log

    # Extract first experiment's validation loss if it exists
    first_exp = {}
    for exp in log["experiments"]:
        if "ott-logs" in exp:
            ott = exp["ott-logs"]
            ott_minimal = {}
            if (
                "eval" in ott
                and "fitting_loss" in ott["eval"]
                and len(ott["eval"]["fitting_loss"]) > 0
            ):
                ott_minimal["eval"] = {"fitting_loss": ott["eval"]["fitting_loss"]}
            if (
                "eval_epoch" in ott
                and "fitting_loss" in ott["eval_epoch"]
                and len(ott["eval_epoch"]["fitting_loss"]) > 0
            ):
                ott_minimal["eval_epoch"] = {
                    "fitting_loss": ott["eval_epoch"]["fitting_loss"]
                }
            if ott_minimal:
                first_exp["ott-logs"] = ott_minimal
                break

    # Extract last experiment's test results
    last_exp = {}
    exp_idx = -1 if len(log["experiments"]) > 0 else 0

    for split in ["train-set", "valid-set", "test-set"]:
        if split in log["experiments"][exp_idx]:
            last_exp[split] = {}
            # Keep to first n conditions for point clouds
            particle_data = {}
            for cond in log["experiments"][exp_idx][split].keys():
                cond_data = {}

                # Keep mean_statistics
                if "mean_statistics" in log["experiments"][exp_idx][split][cond]:
                    cond_data["mean_statistics"] = log["experiments"][exp_idx][split][
                        cond
                    ]["mean_statistics"]

                # Keep point_clouds (but limit to first 3 examples to save memory)
                if (
                    "point_clouds" in log["experiments"][exp_idx][split][cond]
                    and len(particle_data) < max_conds_with_particles
                ):
                    pc = log["experiments"][exp_idx][split][cond]["point_clouds"]
                    if pc is not None and "source" in pc and len(pc["source"]) > 0:
                        particle_data[cond] = {
                            "source": np.concatenate(
                                [np.asarray(x) for x in pc["source"]]
                            ),
                            "target": np.concatenate(
                                [np.asarray(x) for x in pc["target"]]
                            ),
                            "transport": np.concatenate(
                                [np.asarray(x) for x in pc["transport"]]
                            ),
                        }

                last_exp[split][cond] = cond_data
                last_exp[split]["particle"] = particle_data

    # Build minimal log structure
    minimal_log["experiments"] = [first_exp, last_exp] if first_exp else [last_exp]

    return DotMap(minimal_log)


def load_all_logs(
    strategies,
    batch_size=100,
    prefix="default",
    max_iter=3,
    data="k562",
    log_path_format="./models/cmonge-al/{data}/{prefix}/{al_tag}_{strategy}_{batch_selection_strategy}size{batch_size}_fold{fold}_seed{seed}_iter{al_iter}",
    incomplete=False,
    patience=30,
    lr_patience=10,
    batch_selection_strategy="minvar",
):
    """Load all logs for given parameters, keeping only required fields.

    Returns:
        List of dictionaries with 'log', 'strategy', 'fold', 'seed', 'al_iter', 'log_path'
    """
    all_logs = []
    missing = 0
    debug = False
    fold_seed = 10 if not debug else 1
    if debug:
        strategies = ["random"]
    
    for strategy in strategies:
        kernel, bss, al_tag = resolve_arm(strategy, batch_selection_strategy)
        prefix_ = prefix
        for cv_fold in range(5):
            for seed in range(fold_seed):
                for al_iter in range(max_iter + 1):
                    
                    log_path_prefix = log_path_format.format(
                        data=data,
                        prefix=prefix_,
                        strategy=kernel,
                        batch_selection_strategy=bss,
                        al_iter=al_iter,
                        batch_size=batch_size,
                        fold=cv_fold,
                        seed=seed,
                        patience=patience,
                        lr_patience=lr_patience,
                        al_tag=al_tag,
                    )

                    log = None
                    log_path = None
                    if not os.path.exists(log_path_prefix + "_log.pkl") and al_tag == "al0":
                        log_path_prefix = log_path_format.format(
                            data=data,
                            prefix=prefix_,
                            strategy=kernel,
                        batch_selection_strategy=bss,
                            al_iter=al_iter,
                            batch_size=batch_size,
                            fold=cv_fold,
                            seed=seed,
                            patience=patience,
                            lr_patience=lr_patience,
                            al_tag="algp",
                        )
                    if os.path.exists(log_path_prefix + "_log.pkl"):
                        log_path = log_path_prefix + "_log.pkl"
                        try:
                            with open(log_path, "rb") as f:
                                full_log = pkl.load(f)
                        except EOFError:
                            logger.warning(
                                f"EOFError loading {log_path}. Falling back to reading log"
                            )
                            try:
                                full_log = load_config(
                                    log_path_prefix + "_log", use_cache=False
                                )
                                with open(log_path, "wb") as f:
                                    pkl.dump(full_log, f)
                            except FileNotFoundError as e:
                                logger.warning(f"Log file not found: {e}")
                                missing += 1
                                if not incomplete:
                                    raise ValueError(
                                        f"Missing log for {log_path_prefix}"
                                    )
                                continue
                        log = extract_required_fields(full_log)
                        del full_log  # Free memory immediately

                    if log is not None:
                        all_logs.append(
                            {
                                "log": log,
                                "strategy": strategy,
                                "fold": cv_fold,
                                "seed": seed,
                                "al_iter": al_iter,
                                "log_path": log_path,
                                "batch_size": batch_size,
                            }
                        )
                        logger.info(f"Done loading {log_path_prefix}.")

                    else:
                        missing += 1
                        logger.warning(f"Missing log for {log_path_prefix}.")
                        if not incomplete:
                            raise ValueError(f"Missing log for {log_path_prefix}")

    return all_logs, missing


def get_results(
    strategies,
    batch_size=100,
    prefix="default",
    max_iter=3,
    data="k562",
    log_path_format="./models/cmonge-al/{data}/{prefix}/{al_tag}_{strategy}_{batch_selection_strategy}size{batch_size}_fold{fold}_seed{seed}_iter{al_iter}",
    incomplete=False,
    all_logs=None,
    patience=30,
    lr_patience=10,
):
    """Get results from logs. If all_logs is provided, uses pre-loaded logs."""
    if all_logs is None:
        all_logs = load_all_logs(
            strategies,
            batch_size,
            prefix,
            max_iter,
            data,
            log_path_format,
            incomplete,
            patience=patience,
            lr_patience=lr_patience,
        )

    dfs = []
    logger.debug(f"Processing {len(all_logs)} logs to extract results...")
    for log_info in all_logs:
        log = log_info["log"]
        strategy = log_info["strategy"]
        cv_fold = log_info["fold"]
        seed = log_info["seed"]
        al_iter = log_info["al_iter"]
        batch_size = log_info["batch_size"]

        rows = []
        if len(log.experiments) == 1:
            cond_res = log.experiments[0]["test-set"]
        else:
            cond_res = log.experiments[-1]["test-set"]
        for test_cond in cond_res.keys():
            data_dict = cond_res[test_cond]["mean_statistics"]
            data_dict["cond"] = test_cond
            rows.append(pd.Series(data_dict))
        df = pd.concat(rows, axis=1).T
        df["batch_size"] = batch_size
        df["strategy"] = strategy
        df["fold"] = cv_fold
        df["seed"] = seed
        df["al_iter"] = al_iter
        dfs.append(df)
    metrics_df = pd.concat(dfs)
    return metrics_df


# Add up or down arrow to indicate better or worse performance
metrics_direction = {
    "sinkhorn div": "↓",
}


def rel_agg(metrics_df, filter_strategies=None):
    id_cols0 = ["batch_size", "strategy", "fold", "seed", "cond"]
    metric_cols = [f"mean_{m}" for m in metrics_direction.keys()]
    metric_cols_rel = [f"mean_{m}_rel" for m in metrics_direction.keys()]

    # 1. Apply filter that requires all strategies to be present in each (batch_size, fold, al_iter) combination
    if filter_strategies is not None:
        metrics_df = metrics_df.loc[metrics_df.strategy.isin(filter_strategies)]
    n_strategies = len(metrics_df[["strategy", "al_iter"]].drop_duplicates())

    # Count unique (strategy, al_iter) pairs per group — keep only groups that have all of them
    group_cols = ["batch_size", "fold", "seed"]
    strategy_counts = (
        metrics_df.dropna()
        .groupby(group_cols)[["strategy", "al_iter"]]
        .apply(lambda x: len(x.drop_duplicates()))
    )
    print(strategy_counts)

    # Keep only groups where all strategies are present
    valid_groups = strategy_counts[strategy_counts == n_strategies].index
    metrics_df = metrics_df.set_index(group_cols).loc[valid_groups].reset_index()
    num_exp = len(metrics_df[["fold", "seed"]].drop_duplicates())

    # 2. Calculate relative performance compared to first strategy in each group
    res = (
        metrics_df[metric_cols]
        - metrics_df.groupby(id_cols0)[metric_cols].transform("first")
        # / metrics_df.groupby(id_cols0)[metric_cols].transform("first")
    )  # * 100
    res = res.add_suffix("_rel")
    metrics_df = pd.concat([metrics_df, res], axis=1)
    id_cols = ["batch_size", "strategy", "fold", "al_iter", "cond"]
    metrics_df_agg = (
        metrics_df[id_cols + metric_cols + metric_cols_rel]
        .groupby(id_cols)
        .mean()
        .reset_index()
    )

    # Print out per strategy counts of valid folds and seeds
    for strategy in metrics_df["strategy"].unique():
        print(
            f"{strategy}: {metrics_df[metrics_df['strategy'] == strategy][['fold', 'seed']].drop_duplicates().groupby('fold').agg(list)}"
        )
        break
        # .sort_values(['fold', 'seed'])
    # Also return a per seed×fold aggregation (averaged across conds but keeping seed)
    id_cols_sf = ["batch_size", "strategy", "fold", "seed", "al_iter"]
    metrics_df_agg_sf = (
        metrics_df[id_cols_sf + metric_cols + metric_cols_rel]
        .groupby(id_cols_sf)
        .mean()
        .reset_index()
    )
    return metrics_df_agg, metrics_df_agg_sf, num_exp


def get_validation_loss(log_info, init_batch_size: int):
    log = log_info["log"]
    strategy = log_info["strategy"]
    cv_fold = log_info["fold"]
    seed = log_info["seed"]
    al_iter = log_info["al_iter"]
    batch_size = log_info["batch_size"]

    ott_logs = log["experiments"][0]["ott-logs"]
    # Prefer eval_epoch (one value per step, averaged across conditions)
    if (
        "eval_epoch" in ott_logs
        and "fitting_loss" in ott_logs["eval_epoch"]
        and len(ott_logs["eval_epoch"]["fitting_loss"]) > 0
    ):
        values = ott_logs["eval_epoch"]["fitting_loss"]
    else:
        # Fallback: window-average the per-condition eval values
        window_size = batch_size * (init_batch_size // batch_size + al_iter)
        raw_values = ott_logs["eval"]["fitting_loss"]
        values = [
            np.mean(raw_values[i * window_size : (i + 1) * window_size])
            for i in range(len(raw_values) // window_size)
        ]
    metrics_df = (
        pd.Series(values, name="sinkhorn_div").to_frame().reset_index(names=["epoch"])
    )
    metrics_df["batch_size"] = batch_size
    metrics_df["strategy"] = strategy
    metrics_df["fold"] = cv_fold
    metrics_df["seed"] = seed
    metrics_df["al_iter"] = al_iter
    return metrics_df


def get_validation_losses(
    strategies,
    batch_size=100,
    prefix="al_s",
    max_iter=5,
    data="k562",
    incomplete=False,
    log_path_format="./models/cmonge-al/{data}/{prefix}/{al_tag}_{strategy}_{batch_selection_strategy}size{batch_size}_fold{fold}_seed{seed}_iter{al_iter}_log",
    all_logs=None,
    patience=30,
    lr_patience=10,
):
    """Get validation loss from logs. If all_logs is provided, uses pre-loaded logs."""
    if all_logs is None:
        all_logs = load_all_logs(
            strategies,
            batch_size,
            prefix,
            max_iter,
            data,
            log_path_format,
            incomplete,
            patience=patience,
            lr_patience=lr_patience,
        )

    init_batch = max(batch_size, 20)
    if "_ib" in prefix:
        init_batch = int(prefix.split("_ib")[-1])
    dfs = []
    for log_info in all_logs:
        try:
            metrics_df = get_validation_loss(log_info, init_batch)
            dfs.append(metrics_df)
        except (KeyError, IndexError):
            logger.warning(
                f"No ott-logs for {log_info.get('log_path', 'unknown')}, skipping."
            )
    if not dfs:
        logger.warning("No validation loss data found in any logs.")
        return pd.DataFrame(
            columns=[
                "sinkhorn_div",
                "epoch",
                "batch_size",
                "strategy",
                "fold",
                "seed",
                "al_iter",
            ]
        )
    metrics_df = pd.concat(dfs).reset_index(drop=True)
    return metrics_df


def convert_to_numeric(df: pd.DataFrame):
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].apply(
            lambda x: (
                np.array(x).item()
                if isinstance(
                    x,
                    (
                        np.ndarray,
                        jax.Array,
                        jaxlib._jax.ArrayImpl,
                    ),
                )
                else x
            )
        )
    for col in df.columns:
        if col not in df.select_dtypes(include=[np.number]).columns.tolist():
            df[col] = df[col].apply(
                lambda x: (
                    np.array(x)
                    if isinstance(
                        x,
                        (
                            jax.Array,
                            jaxlib._jax.ArrayImpl,
                        ),
                    )
                    else x
                )
            )
    return df


def main():
    parser = ArgumentParser()
    parser.add_argument("data", type=str, help="Dataset name")
    parser.add_argument("batch_size", type=str, help="Batch size")
    parser.add_argument(
        "--prefix", type=str, default="al_smonge", help="Prefix for log path"
    )
    parser.add_argument(
        "--max_iter", type=int, default=3, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--incomplete", action="store_true", help="Allow incomplete experiments"
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=30,
        help="Early stopping patience used in training",
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=10,
        help="LR reduction patience used in training",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated arm names (see ARM_SPECS at the top of this file)",
    )
    parser.add_argument(
        "--batch-selection-strategy",
        type=str,
        default="minvar",
        help="Selection rule for arms that are not listed in ARM_SPECS",
    )
    args = parser.parse_args()
    strategies = [s for s in args.strategies.split(",") if s]
    os.makedirs(f"{results_prefix}/{args.data}/{args.prefix}", exist_ok=True)
    # argparse hands this over as a string; every downstream comparison is numeric.
    args.batch_size = int(args.batch_size)

    if args.batch_size == 1:
        args.max_iter = 10
    all_logs, missing = load_all_logs(
        strategies,
        batch_size=args.batch_size,
        prefix=args.prefix,
        max_iter=args.max_iter,
        data=args.data,
        incomplete=args.incomplete,
        patience=args.patience,
        lr_patience=args.lr_patience,
        batch_selection_strategy=args.batch_selection_strategy,
    )

    val_error_df = get_validation_losses(
        strategies,
        batch_size=args.batch_size,
        prefix=args.prefix,
        max_iter=args.max_iter,
        data=args.data,
        incomplete=args.incomplete,
        all_logs=all_logs,
        patience=args.patience,
        lr_patience=args.lr_patience,
    )
    if len(val_error_df) > 0:
        n_iter = args.max_iter + 1
        al_iter_palette = dict(
            zip(range(n_iter), sns.color_palette("tab10", n_iter))
        )
        al_iter_hue_order = list(range(n_iter))
        fig, ax = plt.subplots(1, len(strategies), figsize=(15, 5))
        if len(strategies) == 1:
            ax = [ax]
        for i, strategy in enumerate(strategies):
            pdf = val_error_df.loc[val_error_df["strategy"] == strategy].copy()
            pdf = convert_to_numeric(pdf)
            pdf["run_id"] = pdf["fold"].astype(str) + "_" + pdf["seed"].astype(str)
            if len(pdf) > 0:
                sns.lineplot(
                    pdf,
                    x="epoch",
                    y="sinkhorn_div",
                    hue="al_iter",
                    ax=ax[i],
                    estimator=None,
                    units="run_id",
                    palette=al_iter_palette,
                    hue_order=al_iter_hue_order,
                    alpha=0.3,
                )
            n_folds = len(pdf["fold"].unique())
            n_seeds = len(pdf["seed"].unique())
            ax[i].set_title(f"{strategy} (F{n_folds}×S{n_seeds})")

        plt.savefig(
            f"{results_prefix}/{args.data}/{args.prefix}/val_loss_plot_{args.data}_size{args.batch_size}.png"
        )
    metrics_df = get_results(
        strategies,
        batch_size=args.batch_size,
        prefix=args.prefix,
        max_iter=args.max_iter,
        data=args.data,
        incomplete=args.incomplete,
        all_logs=all_logs,
        patience=args.patience,
        lr_patience=args.lr_patience,
    )

    metrics_df.to_csv(
        f"{results_prefix}/{args.data}/{args.prefix}/metrics_{args.data}_size{args.batch_size}.csv",
        index=False,
    )

if __name__ == "__main__":
    main()
