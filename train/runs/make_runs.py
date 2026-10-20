"""Expand `grid.py` into one command list per dataset.

Each line is a single self-contained training run (one arm, one seed, all five
folds and every AL iteration in sequence) that writes under
`models/<family>-al/<dataset>/<prefix>/`. Run them however your site prefers --
sequentially, with GNU parallel, or as a job array; nothing here assumes a
scheduler.

  python train/runs/make_runs.py                 # writes train/runs/*.txt
  python train/runs/make_runs.py --arms tntk     # just one arm
  bash train/runs/k562.txt                       # run them one after another
"""
import argparse
from pathlib import Path

import grid

HERE = Path(__file__).resolve().parent


def cmonge_cmds(dataset, arms, seeds):
    mc, lr = grid.CMONGE["datasets"][dataset]
    for b0, b in grid.CMONGE["cells"]:
        prefix = grid.CMONGE["prefix"](b0)
        for arm in arms:
            kernel, bss, _ = grid.ARMS[arm]
            for seed in seeds:
                yield (
                    f"python train/al_train_cmonge.py --strategy={kernel}"
                    f" --batch-selection-strategy={bss} --batch-size {b}"
                    f" --init-batch-size={b0} --init-seed {seed}"
                    f" --config=train/configs/{dataset}.yml --data={dataset}"
                    f" --prefix={prefix} --model-config-id={mc} --lr {lr}"
                    f" --cond-dropout=0.0 --num-epochs {grid.CMONGE['num_epochs']}"
                    f" --no-early-stopping --grad-acc 1"
                    f" --max-iter {grid.CMONGE['max_iter']}"
                )


def cflow_cmds(dataset, arms, seeds):
    mc, lr, epochs, esp, sigma, cdrop = grid.CFLOW["datasets"][dataset]
    lrp = grid.CFLOW["lr_patience"].get(dataset)
    lrp_flag = f" --lr-patience {lrp}" if lrp is not None else ""
    for b0, b in grid.CFLOW["cells"]:
        prefix = grid.CFLOW["prefix"](b0)
        for arm in arms:
            kernel, bss, _ = grid.ARMS[arm]
            for seed in seeds:
                yield (
                    f"python train/al_train_cflow.py --strategy={kernel}"
                    f" --batch-selection-strategy={bss} --batch-size {b}"
                    f" --init-batch-size={b0} --init-seed {seed}"
                    f" --config=train/configs/{dataset}.yml --data={dataset}"
                    f" --prefix={prefix} --iter0-prefix={prefix}"
                    f" --n-conds={grid.CFLOW['n_conds']} --conical-path"
                    f" --model-config-id={mc} --lr {lr} --sigma {sigma}"
                    f" --cond-dropout {cdrop} --n-epochs {epochs}"
                    f" --early-stopping-patience {esp}{lrp_flag}"
                    f" --max-iter={grid.CFLOW['max_iter']}"
                )


def quantile_cmds(dataset, arms, seeds):
    for b0, b in grid.QUANTILE["cells"]:
        mc, wd = grid.QUANTILE["datasets"][dataset][b0]
        prefix = grid.QUANTILE["prefix"](b0, mc, wd)
        for arm in arms:
            # The quantile trainer takes the kernel and the rule separately.
            # Every quantile arm is acquired with `minvar` except BALSA-EMD, which
            # is pinned to `uncertainty` (its score is a disagreement, not a
            # kernel). The rule is part of the run id unless it is `coreset`,
            # which the trainer treats as the unnamed default.
            kernel = arm
            bss = {"balsa_emd": "uncertainty"}.get(arm, "minvar")
            for seed in seeds:
                yield (
                    f"python train/al_train_quantile.py --strategy={kernel}"
                    f" --batch-selection-strategy={bss} --batch-size {b}"
                    f" --init-batch-size={b0} --init-seed {seed}"
                    f" --config=train/configs/{dataset}.yml --data={dataset}"
                    f" --prefix={prefix} --model-config-id={mc}"
                    f" --dropout-rate={grid.QUANTILE['dropout_rate']}"
                    f" --weight-decay={wd} --lr={grid.QUANTILE['lr']}"
                    f" --max-iter={grid.QUANTILE['max_iter']}"
                )


FAMILIES = {
    "cmonge": (grid.CMONGE["datasets"], cmonge_cmds, grid.MAIN_ARMS),
    "cflow": (grid.CFLOW["datasets"], cflow_cmds, grid.MAIN_ARMS),
    "quantile": (grid.QUANTILE["datasets"], quantile_cmds, grid.QUANTILE["arms"]),
}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arms", nargs="+", default=None,
                    help="Subset of arms; default is each family's paper row set")
    ap.add_argument("--seeds", nargs="+", type=int, default=grid.SEEDS)
    ap.add_argument("--out-dir", default=str(HERE))
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    total = 0
    for family, (datasets, builder, default_arms) in FAMILIES.items():
        arms = args.arms or default_arms
        for dataset in datasets:
            cmds = list(builder(dataset, arms, args.seeds))
            path = out / f"{dataset}.txt"
            path.write_text(
                f"# {family}: {dataset}, {len(cmds)} runs "
                f"({len(arms)} arms x {len(args.seeds)} seeds x "
                f"{len(cmds) // max(len(arms) * len(args.seeds), 1)} budget cells).\n"
                "# Each line runs all 5 folds and every AL iteration in sequence.\n"
                + "\n".join(cmds) + "\n"
            )
            total += len(cmds)
            print(f"{path}  ({len(cmds)} runs)")
    print(f"{total} runs in total")


if __name__ == "__main__":
    main()
