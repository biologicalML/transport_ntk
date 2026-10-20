# Datasets

Eight datasets, all shipped in `data/<name>/`. Each directory holds the cells
(`<name>.h5ad`), the condition list the AL loop selects from, the condition
embedding (`moa`) the conditional models consume, and the preprocessing script
that produced them.

| name | cells × dims | conditions file | control | source |
|---|---|---|---|---|
| `k562` | 64,971 × 10 | `sub_drug-dose.csv` | `control-0` | sci-Plex K562, scVI embedding of the drug–dose screen |
| `bmdm` | 73,161 × 10 | `conditions-sub.csv` | `Control` | BMDM Perturb-seq, 10 principal components of the salient representation |
| `radial` | 101,000 × 2 | `conditions2.csv` | `control` | simulated radial flow (`simulate_data2.py`) |
| `vanderpol` | 101,000 × 2 | `conditions.csv` | `control` | simulated Van der Pol oscillator (`simulate_data.py`) |
| `clinc150` | 16,000 × 8 | `clinc150pca8-conditions.csv` | `CTRL` | CLINC-150 intents, sentence embeddings reduced to 8 PCs |
| `sim_mixture` | 601,000 × 1 | `sim-conditions.csv` | `CTRL` | simulated Gaussian mixture (`simulate.py`) |
| `temperature` | 970,212 × 1 | `temperature-sub-conditions.csv` | `CTRL` | station temperature records, scaled |
| `dewpoint` | 912,482 × 1 | `dewpoint-conditions.csv` | `CTRL` | station dew-point records, scaled |

The two Monge datasets are 10-dimensional: `k562` is the scVI embedding and
`bmdm` the 10-PC subset of the 50-dimensional salient representation. The
Sinkhorn divergences in the paper are therefore computed in 10 dimensions for
both rows of that table.

`conditions.csv` in a directory is the short smoke-test list; the file named in
the table is the one the camera-ready runs used, and the one
`train/consts.py` points at.

Each dataset's `moa` file is the condition embedding: a table with one row per
condition. `train/configs/<name>.yml` points `embedding.model_dir` at the
dataset directory, so the loader finds it without any path edits.

## Regenerating a dataset

The preprocessing scripts are shipped next to the data they produced
(`preprocess.py`, `prepare_k562_data.py`, `simulate.py`, `simulate_data.py`,
`step1_encode_samples.py` … `step3_build_pca8.py`,
`build_temperature_sub.py`, `build_dewpoint_dataset.py`). The simulated
datasets regenerate from their script alone; the three real ones need their
raw source, which is public but too large to ship.
