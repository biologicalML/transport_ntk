"""
2D Gaussian Transport Kernel Active Learning Experiment.

Ground truth: conditional 2D Gaussians N(μ(x), Σ(x)) over x ∈ [0,1]².
  - Mean varies near x₁≈0 (exponential decay), flat elsewhere.
  - Covariance is isotropic except near x₁≈1, where it becomes anisotropic
    with rotation angle depending on x₂.

Strategies:
  - Transport kernel AL (Eq. 21–22 from the paper)
  - Mean-only NTK AL (BMDAL baseline)
  - Random

Evaluation: closed-form Bures–Wasserstein W₂² decomposed into mean + covariance.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
from dataclasses import dataclass
import os, yaml
from tqdm import tqdm

# ════════════════════════════════════════════════════════════════
# CONFIG
# ════════════════════════════════════════════════════════════════

@dataclass
class ExperimentConfig:
    name: str = "default"
    # Grid
    n_pool: int = 200
    n_test: int = 100
    n_init: int = 5
    # AL
    batch_size: int = 4
    n_rounds: int = 5
    # Kernel
    lam: float = 1e-3
    # Training
    lr: float = 1e-3
    n_train_steps: int = 2000
    hidden_dim: int = 64
    # Ground truth
    alpha: float = 2.0
    beta: float = 2.0
    gamma: float = 5.0
    stretch: float = 1.0
    sigma_base: float = 0.3
    # Device
    device: str = "mps" if torch.backends.mps.is_available() else "cpu"


def get_device(config: ExperimentConfig) -> torch.device:
    return torch.device(config.device)


def load_config(name, config_file="train/configs/configs_2d_gaussian.yml"):
    """Load a named config from YAML."""
    with open(config_file) as f:
        raw = yaml.safe_load(f)
    for entry in raw:
        if entry["name"] == name:
            return ExperimentConfig(**entry)
    raise ValueError(f"Config '{name}' not found in {config_file}")


def load_results(name, results_dir="./results"):
    """Load saved results for a config name."""
    path = os.path.join(results_dir, name, "results.pt")
    data = torch.load(path, weights_only=False)
    return data["results"], data["indices"]


# ════════════════════════════════════════════════════════════════
# GROUND TRUTH
# ════════════════════════════════════════════════════════════════

def get_distribution_parameters(x: torch.Tensor, config: ExperimentConfig):
    """
    Ground truth conditional Gaussian parameters.

    Args:
        x: (N, 2) input locations in [0,1]²
    Returns:
        mu: (N, 2) mean vectors
        L:  (N, 2, 2) lower-triangular Cholesky factors
    """
    x1, x2 = x[:, 0], x[:, 1]
    N = x.shape[0]

    # Mean: varies near x₁≈0
    mu = torch.stack([config.alpha * torch.exp(-config.beta * x1) * x2, 
                      config.alpha * torch.exp(-config.beta * x1) * (1 - x2)], dim=1)

    # Covariance: anisotropic near x₁≈1
    w = torch.exp(-config.gamma * (1 - x1))  # weight near x₁=1
    phi = np.pi * x2  # rotation angle

    lam1 = config.sigma_base + config.stretch * w
    lam2 = torch.full_like(lam1, config.sigma_base)

    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)

    # R @ diag(lam1, lam2) → columns of the "square root"
    # Then Σ = R diag(lam²) R^T, Cholesky of which we compute explicitly.
    # Simpler: let A = R @ diag(lam1, lam2), then Σ = A A^T.
    # Compute Cholesky of Σ = A A^T directly.
    A = torch.zeros(N, 2, 2, device=x.device, dtype=x.dtype)
    A[:, 0, 0] = cos_phi * lam1
    A[:, 0, 1] = -sin_phi * lam2
    A[:, 1, 0] = sin_phi * lam1
    A[:, 1, 1] = cos_phi * lam2

    Sigma = A @ A.transpose(1, 2)
    L = torch.linalg.cholesky(Sigma)

    assert (L[:, 0, 0] > 0).all() and (L[:, 1, 1] > 0).all()

    return mu, L


# ════════════════════════════════════════════════════════════════
# GRIDS
# ════════════════════════════════════════════════════════════════

def make_grids(config: ExperimentConfig, device: torch.device):
    """
    Returns:
        x_pool: (n_pool, 2) candidate grid
        x_test: (n_test, 2) test grid (offset from pool)
        init_indices: list of indices into x_pool for initial training set
    """
    # Pool grid
    n_side_pool = int(np.round(np.sqrt(config.n_pool)))
    gp = torch.linspace(0.0, 1.0, n_side_pool, device=device)
    grid_p = torch.stack(torch.meshgrid(gp, gp, indexing='ij'), dim=-1)
    x_pool = grid_p.reshape(-1, 2)

    # Test grid (offset by half a cell)
    n_side_test = int(np.round(np.sqrt(config.n_test)))
    half = 0.5 / n_side_pool
    gt = torch.linspace(half, 1.0 - half, n_side_test, device=device)
    grid_t = torch.stack(torch.meshgrid(gt, gt, indexing='ij'), dim=-1)
    x_test = grid_t.reshape(-1, 2)

    # Initial points: 4 corners + center, snapped to nearest pool point
    corners = torch.tensor([
        [0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0], [0.5, 0.5]
    ], device=device)
    init_indices = []
    for c in corners:
        dists = torch.sum((x_pool - c.unsqueeze(0)) ** 2, dim=1)
        init_indices.append(dists.argmin().item())
    # Deduplicate while preserving order
    seen = set()
    init_indices = [i for i in init_indices if not (i in seen or seen.add(i))]
    # include some random points if n_init > 5
    if config.n_init > len(init_indices):
        remaining = list(set(range(x_pool.shape[0])) - set(init_indices))
        rng = np.random.RandomState(42)
        extra = list(rng.choice(remaining, size=config.n_init - len(init_indices), replace=False))
        init_indices.extend(extra)

    return x_pool, x_test, init_indices


# ════════════════════════════════════════════════════════════════
# MODEL
# ════════════════════════════════════════════════════════════════

class MeanNet(nn.Module):
    """Predicts the conditional mean μ(x) ∈ R²."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        return self.head(self.backbone(x))

    def get_feature_embeddings(self, x):
        """Backbone features h(x), shape (N, hidden_dim). Detached from head."""
        with torch.no_grad():
            return self.backbone(x)

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class CovarianceNet(nn.Module):
    """Predicts the Cholesky factor L(x) of the conditional covariance Σ(x)."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.head = nn.Linear(hidden_dim, 3)  # l11, l21, l22

    def forward(self, x):
        """Returns L of shape (N, 2, 2) lower-triangular with positive diagonal."""
        raw = self.head(self.backbone(x))
        N = x.shape[0]
        L = torch.zeros(N, 2, 2, device=x.device, dtype=x.dtype)
        L[:, 0, 0] = F.softplus(raw[:, 0])
        L[:, 1, 0] = raw[:, 1]
        L[:, 1, 1] = F.softplus(raw[:, 2])
        return L

    def get_feature_embeddings(self, x):
        """Backbone features g(x), shape (N, hidden_dim). Detached from head."""
        with torch.no_grad():
            return self.backbone(x)

    def get_pre_activations(self, x):
        """Raw Cholesky head output (N, 3) before softplus."""
        with torch.no_grad():
            return self.head(self.backbone(x))

    def reset_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)


class GaussianTransportNet(nn.Module):
    """Wrapper combining MeanNet and CovarianceNet."""

    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.mean_net = MeanNet(hidden_dim)
        self.cov_net = CovarianceNet(hidden_dim)

    def forward(self, x):
        """Returns (mu, L)."""
        return self.mean_net(x), self.cov_net(x)

    def reset_weights(self):
        self.mean_net.reset_weights()
        self.cov_net.reset_weights()


# ════════════════════════════════════════════════════════════════
# TRAINING
# ════════════════════════════════════════════════════════════════

def kl_loss(mu_pred, L_pred, mu_true, L_true):
    """
    KL(N(mu_true, Σ_true) || N(mu_pred, Σ_pred)), mean over batch.
    Uses Cholesky factors for numerical stability.
    """
    # Solve L_pred^{-1} L_true and L_pred^{-1} (mu_pred - mu_true)
    # torch.linalg.solve_triangular: solve L @ X = B
    LpinvLt = torch.linalg.solve_triangular(L_pred, L_true, upper=False)
    diff = (mu_pred - mu_true).unsqueeze(-1)
    Lpinv_diff = torch.linalg.solve_triangular(L_pred, diff, upper=False)

    d = 2
    trace_term = (LpinvLt ** 2).sum(dim=(-2, -1))
    mahal_term = (Lpinv_diff ** 2).sum(dim=(-2, -1))
    logdet_pred = 2 * L_pred.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)
    logdet_true = 2 * L_true.diagonal(dim1=-2, dim2=-1).log().sum(dim=-1)

    kl = 0.5 * (trace_term + mahal_term - d + logdet_pred - logdet_true)
    return kl.mean()


def train_model(model, x_train, mu_train, L_train, config: ExperimentConfig):
    """Train from scratch. Returns loss curve."""
    model.reset_weights()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)
    losses = []
    for step in tqdm(range(config.n_train_steps), desc="Training", leave=False):
        optimizer.zero_grad()
        mu_pred, L_pred = model(x_train)
        loss = kl_loss(mu_pred, L_pred, mu_train, L_train)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    return losses


# ════════════════════════════════════════════════════════════════
# KERNELS
# ════════════════════════════════════════════════════════════════

def compute_transport_kernel(model, x):
    """
    Closed-form transport sensitivity kernel (Eq. closed-kernel).

    k(x, x') = 2·h(x)^T h(x')
             + (1 + σ'(a¹(x))σ'(a¹(x')) + σ'(a³(x))σ'(a³(x'))) · (g(x)^T g(x') + 1)

    Args:
        model: trained GaussianTransportNet
        x: (N, 2) input locations
    Returns:
        K: (N, N) kernel matrix
    """
    h = model.mean_net.get_feature_embeddings(x)     # (N, D)
    g = model.cov_net.get_feature_embeddings(x)      # (N, D)
    raw = model.cov_net.get_pre_activations(x)       # (N, 3)

    s1 = torch.sigmoid(raw[:, 0])              # σ'(a¹)
    s3 = torch.sigmoid(raw[:, 2])              # σ'(a³)

    K_mean = 2.0 * (h @ h.t()) + 2
    cov_scalar = 1.0 + torch.outer(s1, s1) + torch.outer(s3, s3)
    K_cov = cov_scalar * (g @ g.t() + 1.0)

    return K_mean + K_cov


def compute_mean_kernel(model, x):
    """
    Mean-only last-layer NTK: k(xi, xj) = d * h(xi)^T h(xj).
    """
    h = model.mean_net.get_feature_embeddings(x)  # (N, hidden_dim)
    d = 2
    return d * (h @ h.t())


# ════════════════════════════════════════════════════════════════
# ACQUISITION
# ════════════════════════════════════════════════════════════════

def select_batch_cond_max_var(K, pool_indices, train_indices, batch_size, lam):
    """
    Greedy max posterior variance batch selection.

    Sequentially picks the pool point with highest posterior variance,
    adds it to the selected set, and repeats.

    K: (M, M) kernel over pool points, precomputed.
    Returns: list of newly selected pool indices.
    """
    selected = list(train_indices)
    remaining = sorted(set(pool_indices) - set(selected))

    for _ in range(batch_size):
        if not remaining:
            break

        S_idx = torch.tensor(selected, dtype=torch.long, device=K.device)
        R_idx = torch.tensor(remaining, dtype=torch.long, device=K.device)

        K_SS = K[S_idx][:, S_idx] + lam * torch.eye(len(S_idx), device=K.device)
        K_RS = K[R_idx][:, S_idx]
        k_diag = K[R_idx, R_idx]

        L_chol = torch.linalg.cholesky(K_SS)
        alpha = torch.linalg.solve_triangular(L_chol, K_RS.t(), upper=False)
        var = k_diag - (alpha ** 2).sum(dim=0)

        best = var.argmax().item()
        selected.append(remaining[best])
        remaining.pop(best)

    return selected[len(train_indices):]


def select_batch_min_total_var(K, pool_indices, train_indices, batch_size, lam):
    """
    Greedy min total variance batch selection.

    Sequentially picks the pool point with highest posterior variance,
    adds it to the selected set, and repeats.

    K: (M, M) kernel over pool points, precomputed.
    Returns: list of newly selected pool indices.
    """
    selected = list(train_indices)
    S_idx = torch.tensor(selected, dtype=torch.long, device=K.device)
    K_S = K[:, S_idx]
    Kp = K[S_idx,:][:,S_idx] + lam * torch.eye(len(S_idx), device=K.device)
        
    k_diag = K.diag()

    L_chol = torch.linalg.cholesky(Kp)
    L_inv = torch.linalg.inv(L_chol)
    inv_mat = L_inv.T @ L_inv
    K_post = K - K_S @ inv_mat @ K_S.t()
    K_curr = K_post

    remaining = sorted(set(pool_indices) - set(selected))

    for _ in range(batch_size):
        if not remaining:
            break

        
        R_idx = torch.tensor(remaining, dtype=torch.long, device=K.device)

        col_norms_sq = torch.sum(K_curr * K_curr, axis=0)
        diag_plus_sigma = torch.diag(K_curr) + 1
        scores = (col_norms_sq / diag_plus_sigma)[R_idx]
        best_local = torch.argmax(scores).item()
        j_star = R_idx[best_local]
        # update matrix
        col = K_curr[:, j_star]
        denom = diag_plus_sigma[j_star]
        K_curr = K_curr - torch.outer(col, col) / denom
        selected.append(j_star.item())
        remaining.pop(best_local)

    return selected[len(train_indices):]

# ════════════════════════════════════════════════════════════════
# EVALUATION
# ════════════════════════════════════════════════════════════════

def bures_wasserstein_sq(mu1, L1, mu2, L2):
    """
    W₂² between Gaussians via eigendecomposition for the matrix square root.

    W₂² = ||μ₁ - μ₂||² + Tr(Σ₁) + Tr(Σ₂) - 2·Tr(sqrtm(Σ₁^{1/2} Σ₂ Σ₁^{1/2}))

    Returns: (total, mean_component, cov_component), each shape (N,).
    """
    # Move to CPU for eigendecomposition (not supported on MPS)
    mu1, mu2 = mu1.cpu(), mu2.cpu()
    L1, L2 = L1.cpu(), L2.cpu()

    mean_sq = ((mu1 - mu2) ** 2).sum(dim=-1)

    Sigma1 = L1 @ L1.transpose(-2, -1)
    Sigma2 = L2 @ L2.transpose(-2, -1)

    tr1 = Sigma1.diagonal(dim1=-2, dim2=-1).sum(dim=-1)
    tr2 = Sigma2.diagonal(dim1=-2, dim2=-1).sum(dim=-1)

    # Σ₁^{1/2} via eigendecomposition
    eigvals1, eigvecs1 = torch.linalg.eigh(Sigma1)
    sqrt1 = eigvecs1 @ torch.diag_embed(eigvals1.sqrt()) @ eigvecs1.transpose(-2, -1)

    # M = Σ₁^{1/2} Σ₂ Σ₁^{1/2}, then Tr(sqrtm(M))
    M = sqrt1 @ Sigma2 @ sqrt1
    eigvals_M = torch.linalg.eigvalsh(M)
    cross = 2 * eigvals_M.clamp(min=0.0).sqrt().sum(dim=-1)

    cov_sq = tr1 + tr2 - cross
    total = mean_sq + cov_sq

    return total, mean_sq, cov_sq


def evaluate(model, x_test, mu_test, L_test):
    """Evaluate model on test set. Returns dict with average errors."""
    model.eval()
    with torch.no_grad():
        mu_pred, L_pred = model(x_test)
    model.train()

    total, mean_err, cov_err = bures_wasserstein_sq(mu_pred, L_pred, mu_test, L_test)
    return {
        'total': total.mean().item(),
        'mean': mean_err.mean().item(),
        'covariance': cov_err.mean().item(),
    }


# ════════════════════════════════════════════════════════════════
# EXPERIMENT LOOP
# ════════════════════════════════════════════════════════════════

def run_experiment(strategy: str, seed: int, config: ExperimentConfig):
    """
    Full AL loop for one strategy and seed.

    strategy: 'transport', 'mean', or 'random'
    Returns: {round_idx: {'total', 'mean', 'covariance'}}
    """
    device = get_device(config)
    torch.manual_seed(seed)
    np.random.seed(seed)

    x_pool, x_test, init_indices = make_grids(config, device)
    mu_pool, L_pool = get_distribution_parameters(x_pool, config)
    mu_test, L_test = get_distribution_parameters(x_test, config)

    n_pool = x_pool.shape[0]
    train_indices = list(init_indices)
    model = GaussianTransportNet(config.hidden_dim).to(device)

    results = {}
    n_eval_rounds = config.n_rounds + 1  # evaluate before first acquisition + after each

    for r in range(n_eval_rounds):
        print(f"Round {r}/{config.n_rounds} - Strategy: {strategy} - Seed: {seed}")
        # Train
        x_tr = x_pool[train_indices]
        mu_tr = mu_pool[train_indices]
        L_tr = L_pool[train_indices]
        train_model(model, x_tr, mu_tr, L_tr, config)

        # Evaluate
        results[r] = evaluate(model, x_test, mu_test, L_test)
        print(f"  [{strategy}] seed={seed} round={r} n_train={len(train_indices)} "
              f"W2²={results[r]['total']:.4f} "
              f"(mean={results[r]['mean']:.4f}, cov={results[r]['covariance']:.4f})")

        # Acquire (except after last round)
        if r < config.n_rounds:
            if strategy == 'random':
                remaining = list(set(range(n_pool)) - set(train_indices))
                rng = np.random.RandomState(seed * 100 + r)
                new = list(rng.choice(remaining, size=config.batch_size, replace=False))
            else:
                # Compute kernel
                if strategy == 'transport':
                    K = compute_transport_kernel(model, x_pool)
                elif strategy == 'mean':
                    K = compute_mean_kernel(model, x_pool)

                new = select_batch_min_total_var(
                    K,
                    pool_indices=list(range(n_pool)),
                    train_indices=train_indices,
                    batch_size=config.batch_size,
                    lam=config.lam,
                )

            train_indices = train_indices + new

    return results, train_indices


# ════════════════════════════════════════════════════════════════
# PLOTTING
# ════════════════════════════════════════════════════════════════

COLORS = {'random': 'gray', 'mean': 'C3', 'transport': 'C0'}
LABELS = {'random': 'Random', 'mean': 'Mean-only NTK', 'transport': 'Transport Kernel'}


def plot_ground_truth(config: ExperimentConfig, save_path=None, mean_scaling=1.0, std_scaling=1.0):
    """Visualize mean field (quiver) and covariance ellipses."""
    device = torch.device('cpu')
    n_side = 12
    g = torch.linspace(0.05, 0.95, n_side)
    grid = torch.stack(torch.meshgrid(g, g, indexing='ij'), dim=-1).reshape(-1, 2)
    mu, L = get_distribution_parameters(grid, config)
    mu = mean_scaling * mu.numpy()
    Sigma = (L @ L.transpose(-2, -1)).numpy()
    grid_np = grid.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    # (a) Mean field as quiver
    ax = axes[0]
    ax.quiver(grid_np[:, 0], grid_np[:, 1], mu[:, 0], mu[:, 1],
              scale=15, width=0.005, color='C0')
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r'$x_1$', fontsize=13)
    ax.set_ylabel(r'$x_2$', fontsize=13)
    ax.set_title(r'(a) Mean $\mu(x)$', fontsize=13)
    ax.set_aspect('equal')

    # (b) Covariance ellipses — color by anisotropy ratio
    ax = axes[1]
    # Compute anisotropy for colormap
    aniso = np.zeros(len(grid_np))
    for i in range(len(grid_np)):
        eigvals = np.linalg.eigvalsh(Sigma[i])
        aniso[i] = eigvals.max() / eigvals.min()
    cmap = plt.cm.coolwarm
    norm = plt.Normalize(vmin=1.0, vmax=aniso.max())

    scale = std_scaling * 0.6 / n_side  # scale ellipses to grid spacing
    for i in range(len(grid_np)):
        eigvals, eigvecs = np.linalg.eigh(Sigma[i])
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        w, h = scale * 2 * np.sqrt(eigvals) / np.sqrt(eigvals.max())
        # Preserve aspect ratio: normalize by max eigenvalue so largest axis = scale
        w = scale * np.sqrt(eigvals[0] / eigvals.max())
        h = scale * np.sqrt(eigvals[1] / eigvals.max())
        color = cmap(norm(aniso[i]))
        ell = Ellipse(xy=grid_np[i], width=w, height=h, angle=angle,
                      facecolor=color, alpha=0.5, edgecolor='k', lw=0.4)
        ax.add_patch(ell)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(r'$x_1$', fontsize=13)
    ax.set_ylabel(r'$x_2$', fontsize=13)
    ax.set_title(r'(b) Covariance $\Sigma(x)$ ellipses', fontsize=13)
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    plt.colorbar(sm, ax=ax, label='Anisotropy ratio')
    ax.set_aspect('equal')

    # (c) Heatmap of Tr(Σ)
    ax = axes[2]
    n_heat = 40
    gh = torch.linspace(0.0, 1.0, n_heat)
    grid_h = torch.stack(torch.meshgrid(gh, gh, indexing='ij'), dim=-1).reshape(-1, 2)
    _, Lh = get_distribution_parameters(grid_h, config)
    tr_sigma = (Lh ** 2).sum(dim=(-2, -1)).numpy().reshape(n_heat, n_heat)
    ax.imshow(tr_sigma.T, origin='lower', extent=[0, 1, 0, 1],
              cmap='viridis', aspect='equal')
    ax.set_xlabel(r'$x_1$', fontsize=13)
    ax.set_ylabel(r'$x_2$', fontsize=13)
    ax.set_title(r'(c) $\mathrm{Tr}(\Sigma(x))$', fontsize=13)
    plt.colorbar(ax.images[0], ax=ax)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig


def _draw_ellipses(ax, grid_np, Sigma, n_side, title, cmap_range=None):
    """Helper: draw covariance ellipses color-coded by anisotropy."""
    aniso = np.array([np.linalg.eigvalsh(S).max() / np.linalg.eigvalsh(S).min()
                      for S in Sigma])
    cmap = plt.cm.coolwarm
    vmin, vmax = (1.0, aniso.max()) if cmap_range is None else cmap_range
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    scale = 0.6 / n_side
    for i in range(len(grid_np)):
        eigvals, eigvecs = np.linalg.eigh(Sigma[i])
        angle = np.degrees(np.arctan2(eigvecs[1, 1], eigvecs[0, 1]))
        w = scale * np.sqrt(eigvals[0] / eigvals.max())
        h = scale * np.sqrt(eigvals[1] / eigvals.max())
        color = cmap(norm(aniso[i]))
        ell = Ellipse(xy=grid_np[i], width=w, height=h, angle=angle,
                      facecolor=color, alpha=0.5, edgecolor='k', lw=0.4)
        ax.add_patch(ell)
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05)
    ax.set_aspect('equal')
    ax.set_title(title, fontsize=13)
    return norm


def plot_prediction_vs_truth(model, config: ExperimentConfig, save_path=None):
    """
    2x2 figure: top row = ground truth (mean, cov), bottom row = model prediction.
    """
    device = next(model.parameters()).device
    n_side = 12
    g = torch.linspace(0.05, 0.95, n_side)
    grid = torch.stack(torch.meshgrid(g, g, indexing='ij'), dim=-1).reshape(-1, 2)
    grid_dev = grid.to(device)

    # Ground truth
    mu_true, L_true = get_distribution_parameters(grid, config)
    mu_true_np = mu_true.numpy()
    Sigma_true = (L_true @ L_true.transpose(-2, -1)).numpy()

    # Prediction
    model.eval()
    with torch.no_grad():
        mu_pred, L_pred = model(grid_dev)
    model.train()
    mu_pred_np = mu_pred.cpu().numpy()
    Sigma_pred = (L_pred @ L_pred.transpose(-2, -1)).cpu().numpy()
    grid_np = grid.numpy()

    # Shared colormap range for ellipses
    aniso_true = np.array([np.linalg.eigvalsh(S).max() / np.linalg.eigvalsh(S).min()
                           for S in Sigma_true])
    aniso_pred = np.array([np.linalg.eigvalsh(S).max() / np.linalg.eigvalsh(S).min()
                           for S in Sigma_pred])
    cmap_range = (1.0, max(aniso_true.max(), aniso_pred.max()))

    # Shared quiver scale
    mag_true = np.sqrt((mu_true_np ** 2).sum(axis=1)).max()
    mag_pred = np.sqrt((mu_pred_np ** 2).sum(axis=1)).max()
    qscale = 15 * max(mag_true, mag_pred) / max(mag_true, 1e-8)

    fig, axes = plt.subplots(2, 2, figsize=(6, 6))

    # Top left: true mean
    ax = axes[0, 0]
    ax.quiver(grid_np[:, 0], grid_np[:, 1], mu_true_np[:, 0], mu_true_np[:, 1],
              scale=qscale, width=0.005, color='C0')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal')
    ax.set_title(r'Ground truth $\mu(x)$', fontsize=13)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')

    # Top right: true covariance
    norm = _draw_ellipses(axes[0, 1], grid_np, Sigma_true, n_side,
                          r'Ground truth $\Sigma(x)$', cmap_range)
    axes[0, 1].set_xlabel(r'$x_1$'); axes[0, 1].set_ylabel(r'$x_2$')

    # Bottom left: predicted mean
    ax = axes[1, 0]
    ax.quiver(grid_np[:, 0], grid_np[:, 1], mu_pred_np[:, 0], mu_pred_np[:, 1],
              scale=qscale, width=0.005, color='C3')
    ax.set_xlim(-0.05, 1.05); ax.set_ylim(-0.05, 1.05); ax.set_aspect('equal')
    ax.set_title(r'Predicted $\hat\mu(x)$', fontsize=13)
    ax.set_xlabel(r'$x_1$'); ax.set_ylabel(r'$x_2$')

    # Bottom right: predicted covariance
    _draw_ellipses(axes[1, 1], grid_np, Sigma_pred, n_side,
                   r'Predicted $\hat\Sigma(x)$', cmap_range)
    axes[1, 1].set_xlabel(r'$x_1$'); axes[1, 1].set_ylabel(r'$x_2$')
    sm = plt.cm.ScalarMappable(cmap=plt.cm.coolwarm, norm=norm)
    # plt.colorbar(sm, ax=axes[:, 1].ravel().tolist(), label='Anisotropy ratio', shrink=0.8)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig


def plot_results(all_results: dict, config: ExperimentConfig, save_path=None):
    """
    Three-panel figure: total, mean, covariance error vs training set size.

    all_results: {strategy: [per_seed_result_dict, ...]}
    """
    sizes = [config.n_init + r * config.batch_size for r in range(config.n_rounds + 1)]

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    metrics = [('total', r'$W_2^2$ total'), ('mean', 'Mean error'), ('covariance', 'Covariance error')]

    for ax, (key, title) in zip(axes, metrics):
        for strategy in ['transport', 'mean', 'random']:
            if strategy not in all_results:
                continue
            vals = np.array([
                [res[r][key] for r in range(config.n_rounds + 1)]
                for res in all_results[strategy]
            ])
            mu = vals.mean(axis=0)
            se = vals.std(axis=0) / np.sqrt(vals.shape[0])
            ax.plot(sizes, mu, color=COLORS[strategy], lw=2.5, label=LABELS[strategy])
            ax.fill_between(sizes, mu - 2 * se, mu + 2 * se,
                            color=COLORS[strategy], alpha=0.15)
        ax.set_xlabel('Training set size', fontsize=12)
        ax.set_ylabel(title, fontsize=12)
        ax.set_title(title, fontsize=13)
        # y log
        ax.set_yscale('log')
        ax.legend(fontsize=10)
        ax.grid(alpha=0.3)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig


def plot_acquired_points(all_train_indices: dict, config: ExperimentConfig, save_path=None):
    """
    Show where each strategy acquires points on [0,1]².

    all_train_indices: {strategy: [list_of_train_indices_per_seed, ...]}
    """
    device = torch.device('cpu')
    x_pool, _, init_indices = make_grids(config, device)
    x_pool_np = x_pool.numpy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

    for ax, strategy in zip(axes, ['transport', 'mean', 'random']):
        if strategy not in all_train_indices:
            continue
        # Aggregate acquired points (excluding init) across seeds
        acq = []
        for indices in all_train_indices[strategy]:
            acq.extend([i for i in indices if i not in init_indices])
        acq_pts = x_pool_np[acq]

        ax.scatter(acq_pts[:, 0], acq_pts[:, 1],
                   c=COLORS[strategy], s=15, alpha=0.4, edgecolors='none')
        # Mark init points
        init_pts = x_pool_np[init_indices]
        ax.scatter(init_pts[:, 0], init_pts[:, 1],
                   c='black', s=60, marker='x', lw=2, label='Init')

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel(r'$x_1$', fontsize=12)
        ax.set_ylabel(r'$x_2$', fontsize=12)
        ax.set_title(f'{LABELS[strategy]}', fontsize=13)
        ax.set_aspect('equal')
        ax.legend(fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    return fig
