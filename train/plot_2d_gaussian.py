"""Main-text 2D-Gaussian figure, from a finished `al_2d_gaussian.py` run.

Five panels: (A) the ground-truth mean field, (B) its covariance ellipses with
the acquired points overlaid, and (C-E) squared W2 against the labelled-set
size, split into total / mean / covariance error. The camera-ready figure is
configuration `both_complex_b2`, whose run ships in
`results/gaussian_2d/both_complex_b2/`.

Usage:
  python train/plot_2d_gaussian.py                       # both_complex_b2
"""
import argparse
import os
import sys

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse, Patch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from experiment import (  # noqa: E402
    load_config, load_results, get_distribution_parameters, make_grids, get_device,
)

try:  # optional house style; the figure is identical without it
    import mplscience
    mplscience.set_style()
except ImportError:
    pass

ap = argparse.ArgumentParser(description=__doc__)
ap.add_argument("--config", default="both_complex_b2",
                help="Configuration name, as in train/configs/configs_2d_gaussian.yml")
ap.add_argument("--results-dir", default="results/gaussian_2d",
                help="Directory holding <config>/results.pt")
ap.add_argument("--config-file", default="train/configs/configs_2d_gaussian.yml")
ap.add_argument("--out-dir", default="results/figures")
args = ap.parse_args()

OUTPUT_DIR = args.out_dir
os.makedirs(OUTPUT_DIR, exist_ok=True)

CONFIG_NAME = args.config
config = load_config(CONFIG_NAME, config_file=args.config_file)
device = get_device(config)
all_results, all_indices = load_results(CONFIG_NAME, results_dir=args.results_dir)
print(f"Loaded '{CONFIG_NAME}': {len(all_results['random'])} seeds")


# Ground truth on a 12x12 grid (matching plot_ground_truth)
n_viz = 12
g_viz = torch.linspace(0.05, 0.95, n_viz, device=device)
x_viz = torch.stack(torch.meshgrid(g_viz, g_viz, indexing='ij'), dim=-1).reshape(-1, 2)
mu_viz, L_viz = get_distribution_parameters(x_viz, config)

mu_np = mu_viz.cpu().numpy()
L_np = L_viz.cpu().numpy()
x_np = x_viz.cpu().numpy()

# Covariance matrices for ellipses
Sigma_np = np.einsum('nij,nkj->nik', L_np, L_np)

# Anisotropy ratio for coloring
eigvals = np.linalg.eigvalsh(Sigma_np)
aniso_ratio = eigvals[:, 1] / (eigvals[:, 0] + 1e-10)

# Pool grid (for acquired points overlay)
x_pool, _, _ = make_grids(config, device)
x_pool_np = x_pool.cpu().numpy()

# Ellipse scale matching plot_ground_truth
cell_size = 1.0 / (n_viz - 1)
scale_ell = 0.6 / n_viz

# ── Main paper figure: 1x5 horizontal ───────────────────────────
# (a) Mean field  (b) Covariance  (c) W2² total  (d) W2² mean  (e) W2² cov

from matplotlib.patches import Patch

colors = {'random': 'black', 'mean': 'C0', 'transport': 'C3'}
labels = {'random': 'Random', 'mean': 'NTK', 'transport': ' tNTK'}
sizes = [config.n_init + r * config.batch_size for r in range(config.n_rounds + 1)]
n_repeats = len(all_results['random'])

# ── Layout: 5 plots ─────────────────────────────────────────────
fig, axes = plt.subplots(1, 5, figsize=(15, 3.5))
ax_mu, ax_cov, ax_w2, ax_w2_mean, ax_w2_cov = axes

LABEL_FS = 16
TICK_FS = 13
LEGEND_FS = 13
PANEL_FS = 20

LEG_KW = dict(loc='upper center', bbox_to_anchor=(0.5, -0.28),
              frameon=False, fontsize=LEGEND_FS)


# ════════════════════════════════════════════════════════════════
# (a) Mean field — purple arrows
# ════════════════════════════════════════════════════════════════
mean_scale = 2.5
ax_mu.quiver(x_np[:, 0], x_np[:, 1], mean_scale * mu_np[:, 0], mean_scale * mu_np[:, 1],
             scale=15, width=0.005, color='purple', label=r'Mean($x$)')
ax_mu.set_xlim(-0.05, 1.05)
ax_mu.set_ylim(-0.05, 1.05)
ax_mu.set_xlabel(r'$x_1$', fontsize=LABEL_FS)
ax_mu.set_ylabel(r'$x_2$', fontsize=LABEL_FS)
ax_mu.tick_params(labelsize=TICK_FS)
ax_mu.legend(**LEG_KW)


# ════════════════════════════════════════════════════════════════
# (b) Covariance ellipses
# ════════════════════════════════════════════════════════════════
for i in range(len(x_np)):
    evals, evecs = np.linalg.eigh(Sigma_np[i])
    angle = np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1]))
    w = scale_ell * np.sqrt(evals[0] / evals.max())
    h = scale_ell * np.sqrt(evals[1] / evals.max())
    ell = Ellipse(xy=(x_np[i, 0], x_np[i, 1]), width=w, height=h, angle=angle,
                  facecolor='purple', edgecolor='k', lw=0.8)
    ax_cov.add_patch(ell)

ax_cov.legend(handles=[Patch(facecolor='purple', edgecolor='k', label=r'Cov($x$)')],
              **LEG_KW)
ax_cov.set_xlim(-0.05, 1.05)
ax_cov.set_ylim(-0.05, 1.05)
ax_cov.set_xlabel(r'$x_1$', fontsize=LABEL_FS)
ax_cov.set_ylabel(r'$x_2$', fontsize=LABEL_FS)
ax_cov.tick_params(labelsize=TICK_FS)


# ════════════════════════════════════════════════════════════════
# (c, d, e) W2² error curves — total, mean, covariance
# ════════════════════════════════════════════════════════════════
err_axes = [ax_w2, ax_w2_mean, ax_w2_cov]
err_keys = ['total', 'mean', 'covariance']
err_ylabels = [r'$\log\,(W_2^2)$',
               r'$\log\,({W_2^2}_{\mathrm{mean}})$',
               r'$\log\,({W_2^2}_{\mathrm{cov}})$']

for ax, key, ylabel in zip(err_axes, err_keys, err_ylabels):
    for s in ['transport', 'mean', 'random']:
        vals = np.array([[r[i][key] for i in range(config.n_rounds + 1)]
                         for r in all_results[s]])
        mu = vals.mean(axis=0)
        se = vals.std(axis=0) / np.sqrt(n_repeats)
        ax.plot(sizes, np.log(mu), color=colors[s], lw=2.5, label=labels[s], marker='o')
        ax.fill_between(sizes, np.log(np.maximum(mu - se, 1e-12)),
                        np.log(mu + se), color=colors[s], alpha=0.15)
    ax.set_xticks(sizes)
    ax.set_xlabel('Training set size', fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS)
    ax.tick_params(labelsize=TICK_FS)

# Capture handles for the standalone legend PDF; no legend on panel C.
err_handles, err_labels = ax_w2.get_legend_handles_labels()


# ════════════════════════════════════════════════════════════════
# Add Panel Labels (A–E)
# ════════════════════════════════════════════════════════════════
panel_labels = [r'$\mathbf{A.}$', r'$\mathbf{B.}$', r'$\mathbf{C.}$',
                r'$\mathbf{D.}$', r'$\mathbf{E.}$']

for ax, label in zip(axes, panel_labels):
    ax.text(-0.5, 0.95, label, transform=ax.transAxes,
            fontsize=PANEL_FS, va='bottom', ha='left')

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'fig_gaussian_2d_main.pdf'), dpi=300, bbox_inches='tight')
print(f"Saved to {OUTPUT_DIR}/fig_gaussian_2d_main.pdf")


# ════════════════════════════════════════════════════════════════
# Standalone legend PDF for panels C/D/E
# ════════════════════════════════════════════════════════════════
fig_leg = plt.figure(figsize=(4.5, 0.4))
fig_leg.legend(err_handles, err_labels, loc='center', ncol=3,
               frameon=False, fontsize=LEGEND_FS)
fig_leg.savefig(os.path.join(OUTPUT_DIR, 'fig_gaussian_2d_main_legend.pdf'),
                dpi=300, bbox_inches='tight')
print(f"Saved to {OUTPUT_DIR}/fig_gaussian_2d_main_legend.pdf")