"""Run the 2D-Gaussian AL study for the configuration in
train/configs/configs_2d_gaussian.yml (`both_complex_b2`, the main-text figure)."""

import torch
import time
import os
import json
import argparse
import yaml
from dataclasses import asdict
from experiment import ExperimentConfig, run_experiment

RESULTS_DIR = "./results"
STRATEGIES = ["mean", "transport", "random"]


def load_configs(path="train/configs/configs_2d_gaussian.yml"):
    with open(path) as f:
        raw = yaml.safe_load(f)
    return [ExperimentConfig(**entry) for entry in raw]


def run_config(config, n_seeds, strategies):
    out_dir = os.path.join(RESULTS_DIR, config.name)
    os.makedirs(out_dir, exist_ok=True)

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    all_results = {s: [] for s in strategies}
    all_indices = {s: [] for s in strategies}

    t0 = time.time()
    for seed in range(n_seeds):
        print(f"\n[{config.name}] Seed {seed+1}/{n_seeds} ({time.time()-t0:.0f}s)")
        for strategy in strategies:
            res, idx = run_experiment(strategy, seed=seed, config=config)
            all_results[strategy].append(res)
            all_indices[strategy].append(idx)

    torch.save({"results": all_results, "indices": all_indices},
               os.path.join(out_dir, "results.pt"))
    print(f"\n[{config.name}] Done in {time.time()-t0:.0f}s → {out_dir}/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=30)
    parser.add_argument("--configs", nargs="*", help="Run specific configs by name")
    parser.add_argument("--config-file", default="train/configs/configs_2d_gaussian.yml")
    args = parser.parse_args()

    configs = load_configs(args.config_file)
    if args.configs:
        configs = [c for c in configs if c.name in args.configs]

    for config in configs:
        run_config(config, args.seeds, STRATEGIES)
