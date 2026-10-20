import os
import pickle as pkl
from argparse import ArgumentParser

import numpy as np
import pandas as pd
from loguru import logger

from collect_results_cmonge import convert_to_numeric


def compute_relative_metrics(metrics_df):
    """Compute relative change in w2 distance compared to first iteration.

    Args:
        metrics_df: DataFrame with w2_distance column

    Returns:
        DataFrame with added w2_distance_rel column
    """
    # Group by all identifying columns except al_iter
    id_cols = [
        "batch_size",
        "strategy",
        "fold",
        "seed",
        "cond",
        "split",
    ]
    # Compute relative change: current - first (al_iter=0)
    metrics_df = metrics_df.sort_values("al_iter")

    metrics_df_first = metrics_df.loc[metrics_df.al_iter == 0].rename(
        columns={"w2_distance": "w2_distance_first"}
    )
    metrics_df = metrics_df.merge(
        metrics_df_first[id_cols + ["w2_distance_first"]], on=id_cols, how="left"
    )

    metrics_df["w2_distance_rel"] = (
        metrics_df["w2_distance"] - metrics_df["w2_distance_first"]
    )

    return metrics_df


def compute_summary_tables(metrics_df, save_path=None, include_iter0_in_overall=False):
    """Summarize relative w2 decrease per (strategy, batch_size).

    Two aggregation schemes:
      - foldseed: statistics taken directly across every (fold, seed) pair
      - seed:     first average across folds per seed, then statistics across seeds

    Two metrics:
      - per-iteration: mean/std of w2_distance_rel at each al_iter
      - overall:       for each (fold, seed) take mean of w2_distance_rel across
                       iterations, then aggregate

    Args:
        metrics_df: DataFrame with w2_distance_rel (test split, conditions kept).
        save_path:  If given, write *_per_iter.csv and *_overall.csv alongside it.
        include_iter0_in_overall: iter 0 is always 0 by construction; excluded by default.

    Returns:
        (per_iter_df, overall_df)
    """
    df = metrics_df.loc[metrics_df["split"] == "test"].copy()
    id_cols = ["strategy", "batch_size"]

    # Collapse conditions -> one value per (id, al_iter, fold, seed)
    per_fs = (
        df.groupby(id_cols + ["al_iter", "fold", "seed"])["w2_distance_rel"]
        .mean()
        .reset_index()
    )

    # -------- Metric 1: per-iteration --------
    per_iter_foldseed = (
        per_fs.groupby(id_cols + ["al_iter"])["w2_distance_rel"]
        .agg(mean_foldseed="mean", std_foldseed="std", n_foldseed="count")
        .reset_index()
    )

    # Mean across folds per seed, then stats across seeds
    per_seed_iter = (
        per_fs.groupby(id_cols + ["al_iter", "seed"])["w2_distance_rel"]
        .mean()
        .reset_index()
    )
    per_iter_seed = (
        per_seed_iter.groupby(id_cols + ["al_iter"])["w2_distance_rel"]
        .agg(mean_seed="mean", std_seed="std", n_seed="count")
        .reset_index()
    )

    per_iter = per_iter_foldseed.merge(per_iter_seed, on=id_cols + ["al_iter"])

    # -------- Metric 2: mean-over-iterations per (fold, seed) --------
    per_fs_for_overall = (
        per_fs if include_iter0_in_overall else per_fs.loc[per_fs["al_iter"] > 0]
    )

    mean_over_iter_fs = (
        per_fs_for_overall.groupby(id_cols + ["fold", "seed"])["w2_distance_rel"]
        .mean()
        .reset_index()
    )

    overall_foldseed = (
        mean_over_iter_fs.groupby(id_cols)["w2_distance_rel"]
        .agg(mean_foldseed="mean", std_foldseed="std", n_foldseed="count")
        .reset_index()
    )

    mean_over_iter_seed = (
        mean_over_iter_fs.groupby(id_cols + ["seed"])["w2_distance_rel"]
        .mean()
        .reset_index()
    )
    overall_seed = (
        mean_over_iter_seed.groupby(id_cols)["w2_distance_rel"]
        .agg(mean_seed="mean", std_seed="std", n_seed="count")
        .reset_index()
    )

    overall = overall_foldseed.merge(overall_seed, on=id_cols)

    per_iter = per_iter.sort_values(id_cols + ["al_iter"]).reset_index(drop=True)
    overall = overall.sort_values(id_cols).reset_index(drop=True)

    if save_path is not None:
        per_iter_path = save_path.replace(".csv", "_per_iter.csv")
        overall_path = save_path.replace(".csv", "_overall.csv")
        per_iter.to_csv(per_iter_path, index=False)
        overall.to_csv(overall_path, index=False)
        logger.info(f"Saved per-iteration summary to {per_iter_path}")
        logger.info(f"Saved overall summary to {overall_path}")

    return per_iter, overall



# An "arm" is a (kernel, selection rule) pair. Every arm is acquired with
# `minvar` except BALSA-EMD, which is acquired with `uncertainty`.
#
#   name          (kernel,     selection rule)
ARM_SPECS = {
    "balsa_emd": ("balsa_emd", "uncertainty"),
}
DEFAULT_STRATEGIES = [
    "random_fixed", "ntk", "meanvar_ntk", "quartile_ntk", "balsa_emd", "tntk",
]


def arm_kernel(strategy):
    """The kernel an arm is built on (the name itself if it is not an arm)."""
    return ARM_SPECS[strategy][0] if strategy in ARM_SPECS else strategy


def build_run_id(strategy, batch_size, fold, seed, al_iter,
                 batch_selection_strategy="minvar"):
    """Run id written by al_train_quantile.py for one (arm, fold, seed, iter).

    The trainer only ever knew about kernels, so the id carries the kernel and
    the rule -- `al0_ntk_minvar_...` for the NTK arm, `al0_balsa_emd_uncertainty_...`
    for BALSA-EMD. The rule is omitted only when it is `coreset`, which the
    trainer treats as the unnamed default; the camera-ready quantile runs use
    `minvar`, so their ids carry `_minvar`.
    """
    bss = ARM_SPECS.get(strategy, (None, batch_selection_strategy))[1]
    bss_suffix = "" if bss == "coreset" else f"_{bss}"
    return (
        f"al0_{arm_kernel(strategy)}{bss_suffix}"
        f"_size{batch_size}_fold{fold}_seed{seed}_iter{al_iter}"
    )


def load_quantile_eval(eval_path, incomplete=False):
    if not os.path.exists(eval_path):
        if incomplete:
            return None
        raise FileNotFoundError(f"Missing eval file: {eval_path}")
    with open(eval_path, "rb") as f:
        return pkl.load(f)


def extract_quantile_metrics(
    eval_dict, strategy, batch_size, fold, seed, al_iter
):
    rows = []
    evaluation = eval_dict.get("evaluation", {}) if isinstance(eval_dict, dict) else {}
    for key, result in evaluation.items():
        if "-" not in key:
            continue
        split, cond = key.split("-", 1)
        row = {
            "split": split,
            "cond": cond,
            "w2_distance": result.get("w2_distance", np.nan),
            "strategy": strategy,
            "batch_size": batch_size,
            "fold": fold,
            "seed": seed,
            "al_iter": al_iter,
        }
        rows.append(row)
    return rows


def load_all_quantile_logs(
    strategies,
    batch_size,
    prefix,
    max_iter,
    data,
    log_path_format,
    incomplete=False,
    n_folds=5,
    n_seeds=10,
):
    all_rows = []
    missing = 0
    for strategy in strategies:
        for fold in range(n_folds):
            for seed in range(n_seeds):
                for al_iter in range(max_iter + 1):
                    run_id = build_run_id(
                        strategy,
                        batch_size,
                        fold,
                        seed,
                        al_iter,
                    )
                    eval_path = log_path_format.format(
                        data=data,
                        prefix=prefix,
                        run_id=run_id,
                    )
                    eval_dict = load_quantile_eval(eval_path, incomplete=incomplete)
                    if eval_dict is None:
                        missing += 1
                        continue
                    rows = extract_quantile_metrics(
                        eval_dict,
                        strategy,
                        batch_size,
                        fold,
                        seed,
                        al_iter,
                    )
                    all_rows.extend(rows)
                    logger.info(f"Loaded {eval_path}")

    return all_rows, missing


def get_results(
    strategies,
    batch_size=100,
    prefix="al_quantile",
    max_iter=3,
    data="sim_mixture",
    log_path_format="./models/quantile-al/{data}/{prefix}/{run_id}_trainer_metrics.pkl",
    incomplete=False,
    n_folds=5,
    n_seeds=10,
):
    rows, missing = load_all_quantile_logs(
        strategies=strategies,
        batch_size=batch_size,
        prefix=prefix,
        max_iter=max_iter,
        data=data,
        log_path_format=log_path_format,
        incomplete=incomplete,
        n_folds=n_folds,
        n_seeds=n_seeds,
    )

    if len(rows) == 0:
        raise ValueError("No quantile evaluation logs found.")

    metrics_df = pd.DataFrame(rows)
    metrics_df = convert_to_numeric(metrics_df)
    return metrics_df, missing


def main():
    parser = ArgumentParser()
    parser.add_argument("data", type=str, help="Dataset name")
    parser.add_argument(
        "batch_size", type=str, help="Batch size (or comma-separated list)"
    )
    parser.add_argument(
        "--prefix", type=str, default="al_quantile", help="Prefix for log path"
    )
    parser.add_argument(
        "--max-iter", type=int, default=5, help="Maximum number of iterations"
    )
    parser.add_argument(
        "--incomplete", action="store_true", help="Allow incomplete experiments"
    )
    parser.add_argument("--n-folds", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--n-seeds", type=int, default=10, help="Number of seeds")
    parser.add_argument(
        "--strategies",
        type=str,
        default=",".join(DEFAULT_STRATEGIES),
        help="Comma-separated list of strategies",
    )
    args = parser.parse_args()

    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]

    os.makedirs(
        f"results/quantile/{args.data}/{args.prefix}", exist_ok=True
    )

    args.batch_size = int(args.batch_size)

    metrics_df, missing = get_results(
        strategies=strategies,
        batch_size=args.batch_size,
        prefix=args.prefix,
        max_iter=args.max_iter,
        data=args.data,
        incomplete=args.incomplete,
        n_folds=args.n_folds,
        n_seeds=args.n_seeds,
    )

    out_csv = (
        f"./results/quantile/{args.data}/{args.prefix}/metrics_"
        f"{args.data}_size{args.batch_size}"
        f"{'' if missing == 0 else '_partial'}.csv"
    )
    metrics_df.to_csv(out_csv, index=False)
    logger.info(f"Saved metrics to {out_csv}")

if __name__ == "__main__":
    main()
