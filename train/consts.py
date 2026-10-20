"""Dataset registry for the eight datasets shipped with this package.

Paths are relative to the package root, so every entry resolves when a trainer
is run from there (``python train/al_train_cmonge.py ... --config
train/configs/k562.yml``). The names are the ones used throughout the paper;
the mapping to the directory each dataset came from is in docs/DATASETS.md.
"""

config_prefix = "./train/configs"

config_path = {
    # conditional Monge (Sinkhorn divergence tables)
    "k562": f"{config_prefix}/k562.yml",
    "bmdm": f"{config_prefix}/bmdm.yml",
    # conditional flow matching
    "radial": f"{config_prefix}/radial.yml",
    "vanderpol": f"{config_prefix}/vanderpol.yml",
    "clinc150": f"{config_prefix}/clinc150.yml",
    # conditional quantile networks (W2 tables)
    "sim_mixture": f"{config_prefix}/sim_mixture.yml",
    "temperature": f"{config_prefix}/temperature.yml",
    "dewpoint": f"{config_prefix}/dewpoint.yml",
}

# Short ids used in model/output directory names.
data_id = {
    "k562": "k5",
    "bmdm": "bm",
    "radial": "ra",
    "vanderpol": "vp",
    "clinc150": "cl",
    "sim_mixture": "sm",
    "temperature": "tm",
    "dewpoint": "dw",
}

# One row per condition; the AL loop selects among these.
condition_map = {
    "k562": "./data/k562/sub_drug-dose.csv",
    "bmdm": "./data/bmdm/conditions-sub.csv",
    "radial": "./data/radial/conditions2.csv",
    "vanderpol": "./data/vanderpol/conditions.csv",
    "clinc150": "./data/clinc150/clinc150pca8-conditions.csv",
    "sim_mixture": "./data/sim_mixture/sim-conditions.csv",
    "temperature": "./data/temperature/temperature-sub-conditions.csv",
    "dewpoint": "./data/dewpoint/dewpoint-conditions.csv",
}

# The source distribution every conditional map transports from.
control_map = {
    "k562": "control-0",
    "bmdm": "Control",
    "radial": "control",
    "vanderpol": "control",
    "clinc150": "CTRL",
    "sim_mixture": "CTRL",
    "temperature": "CTRL",
    "dewpoint": "CTRL",
}

# Learning rate used for the camera-ready runs where it differs from the
# trainer default (1e-3). K562 is the only one.
dataset_lr = {
    "k562": 1e-4,
}
