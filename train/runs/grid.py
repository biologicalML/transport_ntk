"""The camera-ready experiment grid, as data.

`make_runs.py` turns this into the per-dataset command lists in this directory.
`eval/build_summaries.py` reads the same table to find the result directories,
so the commands and the summariser cannot disagree about a prefix.

Every value here is the one used for the camera-ready runs. Where a setting is
per-dataset it is written out per dataset rather than hidden behind a default.
"""

SEEDS = list(range(10))      # --init-seed
FOLDS = 5                    # cross-validation folds, run inside one process

# Arms. An arm is a (kernel, selection rule) pair; every kernel arm is acquired
# with total variance minimization (`minvar`). "NTK" and "tNTK" are the display
# names used in the paper.
#
#   arm            kernel             selection rule   paper label
ARMS = {
    "random_fixed": ("random_fixed", "minvar", "Random"),
    "ntk": ("ntk_sizematched", "minvar", "NTK"),
    "tntk": ("tntk", "minvar", "tNTK"),
}
MAIN_ARMS = ["random_fixed", "ntk", "tntk"]

# --- conditional Monge (Sinkhorn divergence tables) ------------------------
CMONGE = {
    "datasets": {
        # dataset: (model-config-id, learning rate)
        "k562": ("l5", 1e-4),
        "bmdm": ("l5", 1e-3),
    },
    # (initial budget B0, per-iteration batch B)
    "cells": [(10, 1), (10, 5), (20, 5), (20, 10), (30, 5), (30, 10)],
    "max_iter": 5,
    "num_epochs": 100,
    "cond_dropout": 0.0,
    # Output prefix per B0. The trainer writes
    # models/cmonge-al/<dataset>/<prefix>/ and the collector reads it back.
    "prefix": lambda b0: f"cameraready_ib{b0}",
}

# --- conditional flow matching --------------------------------------------
CFLOW = {
    "datasets": {
        # dataset: (model-config-id, lr, n_epochs, early-stopping patience,
        #           sigma, cond dropout)
        "radial": ("l", 5e-6, 30_000, 5, 1.0, 0.0),
        "vanderpol": ("l", 5e-5, 30_000, 5, 1.0, 0.0),
        "clinc150": ("l", 5e-5, 300_000, 30, 1.0, 0.5),
    },
    # --lr-patience per dataset; absent = not passed. The trainer then tags the
    # run id with `_lrp<N>`, so the collector needs the same `--lr-patience`.
    "lr_patience": {"clinc150": 5},
    "cells": [(5, 1), (5, 5), (10, 5)],
    "max_iter": 3,
    "n_conds": 128,
    "conical_path": True,
    "prefix": lambda b0: f"cameraready_ib{b0}_c",
}

# --- conditional quantile networks (W2 tables) -----------------------------
QUANTILE = {
    "datasets": {
        # dataset: {B0: (model-config-id, weight decay)}
        "sim_mixture": {5: ("l", 1e-4), 10: ("d32", 1e-3)},
        "temperature": {5: ("s", 1e-3), 10: ("d32", 1e-2)},
        "dewpoint": {5: ("s", 1e-4), 10: ("s", 1e-1)},
    },
    "cells": [(5, 1), (5, 5), (10, 1)],
    "max_iter": 5,
    "lr": 1e-2,
    "dropout_rate": 0.0,
    # The quantile arms. All are acquired with `minvar` except BALSA-EMD, which is
    # pinned to `uncertainty`: the NTK kernel, its two score ablations,
    # BALSA-EMD, and the tangent kernel.
    "arms": ["random_fixed", "ntk", "meanvar_ntk", "quartile_ntk",
             "balsa_emd", "tntk"],
    "prefix": lambda b0, mc, wd: f"Quantile:{mc}cameraready_ib{b0}_wd{wd}",
}
