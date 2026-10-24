"""Figures for Part 1. Each function saves to images/ and returns the path."""
import os

import matplotlib.pyplot as plt
import numpy as np

from em_gmm import LABELS, e_step, gaussian_pdf

IMG_DIR = "images"
C1, C2, INK, MUTED = "#2a78b5", "#d1662b", "#222222", "#8a8a8a"
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False,
                     "axes.spines.right": False, "font.size": 10})


def _save(fig, name):
    os.makedirs(IMG_DIR, exist_ok=True)
    path = os.path.join(IMG_DIR, name)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    return path


def _grid(x, pad=3):
    return np.linspace(x.min() - pad, x.max() + pad, 600)


def plot_raw(x, name="01_unlabeled_heights.png"):
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.hist(x, bins=40, density=True, color=MUTED, alpha=0.7)
    ax.axvline(x.mean(), color=INK, ls="--", lw=1)
    ax.text(x.mean(), ax.get_ylim()[1] * 0.95, f"  global mean {x.mean():.2f} in", va="top")
    ax.set(title="What EM sees: 1,139 unlabeled heights", xlabel="height (in)", ylabel="density")
    return _save(fig, name)


def plot_mean_split(x, y, ms, name="02_mean_split_vs_truth.png"):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
    bins = np.linspace(x.min(), x.max(), 40)
    ax = axes[0]
    ax.hist(x[x < ms["cut"]], bins=bins, color=C1, alpha=0.8, label=f"pile 1 (n={ms['n'][0]})")
    ax.hist(x[x >= ms["cut"]], bins=bins, color=C2, alpha=0.8, label=f"pile 2 (n={ms['n'][1]})")
    ax.axvline(ms["cut"], color=INK, ls="--")
    ax.set(title="Naive: hard cut at the global mean", xlabel="height (in)", ylabel="count")
    ax.legend(frameon=False)
    ax = axes[1]
    ax.hist(x[y == 0], bins=bins, color=C1, alpha=0.6, label=f"children (n={(y == 0).sum()})")
    ax.hist(x[y == 1], bins=bins, color=C2, alpha=0.6, label=f"fathers (n={(y == 1).sum()})")
    ax.axvline(ms["cut"], color=INK, ls="--")
    ax.set(title="Truth (hidden from EM): the groups overlap", xlabel="height (in)")
    ax.legend(frameon=False)
    return _save(fig, name)


def plot_fit(x, params, name="03_em_fit.png"):
    g = _grid(x)
    comp = params.pi * gaussian_pdf(g[:, None], params.mu, params.var)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist(x, bins=40, density=True, color=MUTED, alpha=0.45, label="data")
    for k, c in enumerate((C1, C2)):
        ax.plot(g, comp[:, k], color=c, lw=2,
                label=f"{LABELS[k]}: μ={params.mu[k]:.2f}, σ²={params.var[k]:.2f}, π={params.pi[k]:.2f}")
    ax.plot(g, comp.sum(1), color=INK, lw=1.4, ls="--", label="mixture p(x)")
    ax.set(title="Fitted 2-component Gaussian mixture", xlabel="height (in)", ylabel="density")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, name)


def plot_convergence(history, name="04_log_likelihood.png"):
    it = [h["iteration"] for h in history]
    ll = [h["log_likelihood"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    axes[0].plot(it, ll, color=INK, lw=1.6)
    axes[0].set(title="Log-likelihood never decreases", xlabel="iteration", ylabel="log-likelihood")
    axes[1].semilogy(it[1:], np.abs(np.diff(ll)) + 1e-16, color=C2, lw=1.6)
    axes[1].axhline(1e-6, color=MUTED, ls=":")
    axes[1].text(it[-1], 1e-6, "tol = 1e-6 ", ha="right", va="bottom", color=MUTED)
    axes[1].set(title="Stop when the change becomes negligible", xlabel="iteration", ylabel="|ΔLL|")
    return _save(fig, name)


def plot_posterior(x, params, test_in=None, name="05_posterior_curve.png"):
    g = _grid(x)
    post = e_step(g, params)
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.plot(g, post[:, 0], color=C1, lw=2, label=f"P({LABELS[0]} | x)")
    ax.plot(g, post[:, 1], color=C2, lw=2, label=f"P({LABELS[1]} | x)")
    ax.axhline(0.5, color=MUTED, ls=":")
    if test_in is not None:
        p = e_step(np.array([test_in]), params)[0]
        ax.axvline(test_in, color=INK, ls="--", lw=1)
        ax.scatter([test_in] * 2, p, color=[C1, C2], zorder=3)
        ax.text(test_in, 0.55, f"  test = {test_in:.2f} in", color=INK)
    ax.set(title="Soft assignment: posterior probability across heights",
           xlabel="height (in)", ylabel="posterior", ylim=(-0.03, 1.03))
    ax.legend(frameon=False, loc="center right")
    return _save(fig, name)


def plot_hidden_structure(df, params, name="06_what_em_found.png"):
    """Children split by sex vs the fitted components."""
    g = np.linspace(55, 82, 600)
    comp = params.pi * gaussian_pdf(g[:, None], params.mu, params.var)
    fam = df.drop_duplicates("family")
    bins = np.linspace(55, 82, 45)
    n = len(df) + len(fam)
    w = lambda a: np.full(len(a), 1 / (n * (bins[1] - bins[0])))
    groups = [(df[df.gender == "female"].childHeight, "#7fb3d8", "daughters"),
              (df[df.gender == "male"].childHeight, "#9cc79a", "sons"),
              (fam.father, "#f0a679", "fathers")]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.hist([a for a, _, _ in groups], bins=bins, weights=[w(a) for a, _, _ in groups],
            stacked=True, color=[c for _, c, _ in groups], label=[l for _, _, l in groups])
    for k, c in enumerate((C1, C2)):
        ax.plot(g, comp[:, k], color=c, lw=2.2, label=f"EM component {k + 1}")
    ax.set(title="What EM actually separated: daughters vs. (sons + fathers)",
           xlabel="height (in)", ylabel="density")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, name)
