# Transport Neural Tangent Kernel (tNTK)

Transport Neural Tangent Kernel (tNTK) for active learning of generative models (Ryu et al., NeurIPS 2026).

Many experiments measure a whole *distribution* per condition: a population of
cells per perturbation, a set of sentence embeddings per intent, a spread of
temperatures per location. A conditional generative model learns to push a
source distribution onto each condition's target. When every condition is
expensive to measure, the question is **which conditions to measure next**.

This repository implements the answer from the paper: score candidate
conditions with the **transport neural tangent kernel (tNTK)**, the tangent
kernel of the model's conditional transport map, and acquire them by total
variance minimization under that kernel.

## Main idea: the transport NTK

A conditional pushforward model $T_\theta(x, c)$ maps a source sample
$x \sim \mu$ to a sample of the target distribution $\nu_c$ of condition $c$.
Kernel-based active learning normally uses the NTK of a network that predicts
one summary of each condition, typically its mean, so two conditions look
similar when their predicted means are similar. The tNTK instead compares the
whole transport map, averaged over the source distribution:

$$
k_\theta^\mathrm{tNTK}(x, x') = \int \operatorname{Tr}\left[D_\theta T_{\theta}(x)(y)^\top D_\theta T_{\theta}(x')(y)\right]d\nu(y).
$$

It needs one forward pass per condition and no extra training. Each model
family supplies its own $\phi$:

| family | feature $\phi(y, x)$ | code |
|---|---|---|
| conditional Monge | last hidden layer of the Monge map, on every source cell | [`get_grad_kernel_matrix`](train/al_train_cmonge.py#L40) |
| conditional flow matching | velocity-field features integrated along the ODE trajectory, $\int_0^1 \phi(x_t, t, c)\,dt$, solved jointly with the flow (Tsit5) | [`_compute_trajectory_grads`](train/al_train_cflow.py#L241), [`get_grad_kernel_matrix`](train/al_train_cflow.py#L296) |
| conditional quantile network | Jacobian of all predicted quantiles w.r.t. the output-layer parameters (the source is the set of quantile levels) | [`get_grad_kernel_matrix`](train/al_train_quantile.py#L826) |


## Repository layout

```
train/      trainers, dataset configs, and the run lists for the full grid
eval/       per-family collectors + the table / panel / significance builder
data/       the eight datasets, their condition lists and embeddings
extern/     the two bundled libraries the trainers import (git submodules)
results/    the camera-ready tables, panels and the 2D-Gaussian figure run
```

`extern/` holds forks of [CellFlow](https://github.com/theislab/CellFlow)
([jykr/CellFlow-al](https://github.com/jykr/CellFlow-al)) and
[conditional Monge](https://github.com/AI4SCR/conditional-monge-gap) v0.1.2
([jykr/cmonge-al](https://github.com/jykr/cmonge-al)), both on branch
`transport_ntk`, with the hooks the kernels need: hidden features via
`return_jacobian=True`, plus active-learning training options.

## Install

```bash
git clone --recurse-submodules https://github.com/biologicalML/transport_ntk.git
cd transport_ntk   # or, in an existing clone: git submodule update --init
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e extern/cmonge-al -e extern/CellFlow-al
```

`requirements.txt` is the exact environment the camera-ready runs used: 186
pinned packages, including CUDA 12 builds of JAX and PyTorch. On a machine
without a GPU, install CPU builds instead (`pip install jax==0.8.1
jaxlib==0.8.1 torch==2.11.0`); every trainer runs on CPU, only slower. Every
command below is run from this directory.

## The three model families

| family | trainer | datasets | metric |
|---|---|---|---|
| conditional Monge | `train/al_train_cmonge.py` | `k562`, `bmdm` | Sinkhorn divergence |
| conditional flow matching | `train/al_train_cflow.py` | `radial`, `vanderpol`, `clinc150` | Sinkhorn divergence |
| conditional quantile network | `train/al_train_quantile.py` | `sim_mixture`, `temperature`, `dewpoint` | $W_2$ distance |

The 2D-Gaussian study is separate and self-contained:
`train/al_2d_gaussian.py` with `train/experiment.py`.

## Citation

Please cite the following work if you use this repository.
```
@INPROCEEDINGS{Ryu2026-tntk,
  title     = "Active Learning of Conditional Generative Models via the
               Transport Neural Tangent Kernel",
  author    = "Ryu, Jayoung and Cho, Kyunghyun and Lopez, Romain",
  booktitle = "The Fortieth Annual Conference on NeurIPS",
  year      =  2026
}
```
