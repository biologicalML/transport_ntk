from typing import Dict, List
import pandas as pd
import pickle as pkl
import os
import matplotlib
from argparse import ArgumentParser
from loguru import logger


matplotlib.rcParams["pdf.fonttype"] = 42


# An "arm" is a (kernel, selection rule) pair, and the rule is part of the run
# id. A name that is not listed here is a kernel collected under
# --batch-selection-strategy.
#
#   name          kernel              selection rule
ARM_SPECS = {
    "ntk": ("ntk_sizematched", "minvar"),
}
NO_KERNEL_STRATEGIES = ("random", "random_fixed")
DEFAULT_STRATEGIES = ["random_fixed", "ntk", "tntk"]


def resolve_arm(strategy, batch_selection_strategy="minvar"):
    """(kernel, selection-rule suffix) for one arm name."""
    if strategy in ARM_SPECS:
        kernel, bss = ARM_SPECS[strategy]
    else:
        kernel, bss = strategy, batch_selection_strategy
    return kernel, (f"{bss}_" if bss else "")


def load_all_logs(
    strategies,
    batch_size=100,
    prefix="default",
    max_iter=3,
    data="radial",
    incomplete=False,
    model_config_id="s",
    lr_patience_label="",
    log_path_format="models/cellflow-al/models/{data}/{prefix}/gp_{model_config_id}_{strategy}_{batch_selection_strategy}size{batch_size}_fold{fold}_seed{seed}{lr_patience_label}_iter{al_iter}_training_logs.pkl",
    batch_selection_strategy="minvar",
):
    """Load all logs for given parameters."""
    all_logs = []
    n_missing = 0
    for strategy in strategies:
        kernel, bss = resolve_arm(strategy, batch_selection_strategy)
        for seed in range(10):
            for fold in range(5):
                for al_iter in range(max_iter + 1):
                    log_path = log_path_format.format(
                        data=data,
                        prefix=prefix,
                        strategy=kernel,
                        batch_selection_strategy=bss,
                        al_iter=al_iter,
                        batch_size=batch_size,
                        seed=seed,
                        fold=fold,
                        model_config_id=model_config_id,
                        lr_patience_label=lr_patience_label,
                    )

                    log = None
                    if os.path.exists(log_path):
                        with open(log_path, "rb") as f:
                            log = pkl.load(f)

                    if log is not None:
                        all_logs.append(
                            {
                                "log": log,
                                "strategy": strategy,
                                "fold": fold,
                                "seed": seed,
                                "al_iter": al_iter,
                                "log_path": log_path,
                                "batch_size": batch_size,
                            }
                        )
                    elif not incomplete:
                        raise ValueError(f"Missing log for {log_path}")
                    else:
                        n_missing += 1
                        logger.info(f"Missing log for {log_path}")

    return all_logs, n_missing


def get_results(
    strategies,
    batch_size=100,
    prefix="al_s",
    max_iter=3,
    data="spherepair-sub",
    incomplete=False,
    model_config_id="l",
    lr_patience_label="",
    log_path_format="models/cellflow-al/models/{data}/{prefix}/gp_{model_config_id}_{strategy}_{batch_selection_strategy}size{batch_size}_fold{fold}_seed{seed}{lr_patience_label}_iter{al_iter}_training_logs.pkl",
    all_logs=None,
):
    """Get results from logs. If all_logs is provided, uses pre-loaded logs."""
    if all_logs is None:
        all_logs, missing = load_all_logs(
            strategies,
            batch_size,
            prefix,
            max_iter,
            data,
            incomplete,
            log_path_format=log_path_format,
            model_config_id=model_config_id,
            lr_patience_label=lr_patience_label,
        )
    else:
        missing = 0

    dfs = []
    for log_info in all_logs:
        log = log_info["log"]
        strategy = log_info["strategy"]
        fold = log_info["fold"]
        seed = log_info["seed"]
        al_iter = log_info["al_iter"]
        batch_size = log_info["batch_size"]
        log_path = log_info["log_path"]

        metrics_series = []
        for metric in ["sinkhorn_div"]:
            try:
                metric_series = pd.Series(log[f"test_{metric}"][-1], name=metric)
            except KeyError:
                if not incomplete:
                    raise ValueError(f"Missing metric {metric} in log {log_path}")
                else:
                    print(f"Missing metric {metric} in log {log_path}")
                    missing += 1
            metrics_series.append(metric_series)
        metrics_df = pd.concat(metrics_series, axis=1)
        metrics_df["cond"] = metrics_df.index.map(
            lambda t: t[0] if isinstance(t, tuple) else t
        )
        metrics_df.reset_index(drop=True, inplace=True)
        metrics_df["batch_size"] = batch_size
        metrics_df["fold"] = fold
        metrics_df["strategy"] = strategy
        metrics_df["seed"] = seed
        metrics_df["al_iter"] = al_iter
        dfs.append(metrics_df)
    metrics_df = pd.concat(dfs)
    return metrics_df, missing


def get_validation_loss(log_info):
    log = log_info["log"]
    strategy = log_info["strategy"]
    cv_fold = log_info["fold"]
    seed = log_info["seed"]
    al_iter = log_info["al_iter"]
    batch_size = log_info["batch_size"]
    metric_dict = {}
    for metric in log.keys():
        if not metric.startswith("valid_") or metric.startswith(
            "train_"
        ):  # Backwards compatibility. Only one of two should have values
            continue
        if metric.endswith("_mean"):
            metrics_df = (
                pd.Series(log[metric], name=metric)
                .to_frame()
                .sort_index()
                .reset_index(names=["epoch"])
            )
        else:
            metrics_df = (
                pd.DataFrame(log[metric])
                .T.sort_index()
                .melt(ignore_index=False, var_name="cond")
                .reset_index(names=["epoch"])
            )
        metrics_df["batch_size"] = batch_size
        metrics_df["strategy"] = strategy
        metrics_df["fold"] = cv_fold
        metrics_df["seed"] = seed
        metrics_df["al_iter"] = al_iter
        metric_dict[metric] = metrics_df
    return metric_dict


def get_validation_losses(
    strategies,
    batch_size=100,
    prefix="al_s",
    max_iter=3,
    data="spherepair-sub",
    incomplete=False,
    log_path_format="models/cellflow-al/models/{data}/{prefix}/gp_{model_config_id}_{strategy}_{batch_selection_strategy}size{batch_size}_fold{fold}_seed{seed}{lr_patience_label}_iter{al_iter}_training_logs.pkl",
    all_logs=None,
    model_config_id="l",
    lr_patience_label="",
):
    """Get validation loss from logs. If all_logs is provided, uses pre-loaded logs."""
    if all_logs is None:
        all_logs, _ = load_all_logs(
            strategies,
            batch_size,
            prefix,
            max_iter,
            data,
            incomplete,
            log_path_format=log_path_format,
            model_config_id=model_config_id,
            lr_patience_label=lr_patience_label,
        )

    metric_dict: Dict[str, List[pd.DataFrame]] = {}
    missing = 0
    for i, log_info in enumerate(all_logs):
        this_metric_dict = get_validation_loss(log_info)
        for metric, metrics_df in this_metric_dict.items():
            if i == 0:
                logger.debug(f"{metric}: {metrics_df.head()}")
            if metric in metric_dict:
                metric_dict[metric].append(metrics_df)
            else:
                metric_dict[metric] = [metrics_df]
    metrics_dict = {
        k: pd.concat(v).reset_index(drop=True) for k, v in metric_dict.items()
    }

    return metrics_dict


# Add up or down arrow to indicate better or worse performance
metrics_direction = {
    "sinkhorn_div": "↓",
}


def rel_agg(metrics_df: pd.DataFrame):
    id_cols0 = ["batch_size", "strategy", "fold", "seed", "cond"]
    metric_cols = [m for m in metrics_direction.keys() if m in metrics_df.columns]
    metric_cols_rel = [
        f"{m}_rel" for m in metrics_direction.keys() if m in metrics_df.columns
    ]
    res = metrics_df[metric_cols] - metrics_df.groupby(id_cols0)[metric_cols].transform(
        "first"
    )
    res = res.add_suffix("_rel")
    metrics_df = pd.concat([metrics_df, res], axis=1)
    id_cols = ["batch_size", "strategy", "seed", "al_iter"]
    # Apply filter that requires all strategies to be present in each (batch_size, fold, al_iter) combination
    strategies = metrics_df[["strategy", "al_iter"]].drop_duplicates()
    n_strategies = len(strategies)
    # Count how many strategies exist for each (batch_size, fold, seed, al_iter, cond) group
    group_cols = ["batch_size", "fold", "seed", "cond"]
    strategy_counts = metrics_df.groupby(group_cols)[["strategy", "al_iter"]].size()

    # Keep only groups where all strategies are present
    valid_groups = strategy_counts[strategy_counts == n_strategies].index
    metrics_df = metrics_df.set_index(group_cols).loc[valid_groups].reset_index()
    metrics_df_agg_sf = metrics_df[["batch_size", "strategy", "seed", "al_iter", "fold"] + metric_cols + metric_cols_rel].groupby(["batch_size", "strategy", "fold", "seed", "al_iter"]).mean().reset_index() # remove outlier
    metrics_df_agg = (
        metrics_df_agg_sf[id_cols + metric_cols_rel].groupby(id_cols).mean().reset_index()
    )
    # Print out per strategy counts of valid folds and seeds
    for strategy in metrics_df["strategy"].unique():
        print(
            f"{strategy}: {metrics_df[metrics_df['strategy'] == strategy][['fold', 'seed']].drop_duplicates().groupby('fold').agg(list)}"
        )
        break
        # .sort_values(['fold', 'seed'])
    return metrics_df_agg, metrics_df_agg_sf, len(metrics_df[["fold", "seed"]].drop_duplicates())


def main():
    parser = ArgumentParser()
    parser.add_argument("data", type=str, help="Dataset name")
    parser.add_argument("batch_size", type=str, help="Batch size")
    parser.add_argument(
        "--prefix", type=str, default="al_s", help="Prefix for log path"
    )
    parser.add_argument(
        "--max_iter", type=int, default=3, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--incomplete", action="store_true", help="Allow incomplete experiments"
    )
    parser.add_argument(
        "--model-config-id", type=str, default="s", help="Model config ID"
    )
    parser.add_argument(
        "--lr-patience",
        type=int,
        default=None,
        help="lr_patience used during training; inserted as '_lrp{lr_patience}' between seed and iter in path",
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
    lr_patience_label = (
        f"_lrp{args.lr_patience}" if args.lr_patience is not None else ""
    )

    strategies = [s for s in args.strategies.split(",") if s]
    os.makedirs(f"results/cflow/{args.data}/{args.prefix}", exist_ok=True)
    args.batch_size = int(args.batch_size)

        # Load all logs once
    all_logs, missing_logs = load_all_logs(
        strategies,
        batch_size=args.batch_size,
        prefix=args.prefix,
        max_iter=args.max_iter,
        data=args.data,
        incomplete=args.incomplete,
        model_config_id=args.model_config_id,
        lr_patience_label=lr_patience_label,
        batch_selection_strategy=args.batch_selection_strategy,
    )

    metrics_df, missing = get_results(
        strategies,
        batch_size=args.batch_size,
        prefix=args.prefix,
        max_iter=args.max_iter,
        data=args.data,
        incomplete=args.incomplete,
        all_logs=all_logs,
        model_config_id=args.model_config_id,
        lr_patience_label=lr_patience_label,
    )
    missing += missing_logs

    metrics_df.to_csv(
        f"results/cflow/{args.data}/{args.prefix}/metrics_{args.data}_size{args.batch_size}.csv",
        index=False,
    )
    metrics_df_agg, metrics_df_agg_sf, num_exp = rel_agg(metrics_df)

    metrics_df.to_csv(
        f"results/cflow/{args.data}/{args.prefix}/val_metrics_{args.data}_size{args.batch_size}.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
