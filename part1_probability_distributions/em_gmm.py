"""
Part 1 - Expectation-Maximization for a 2-component 1-D Gaussian Mixture,
implemented from scratch (NumPy only for array math; no sklearn / scipy fitting).

Population pair: Galton (1886) FATHERS vs CHILDREN heights, in inches.
Component 1 = Children (shorter mean), Component 2 = Fathers (taller mean).

Model
-----
    p(x) = pi_1 * N(x | mu_1, var_1) + pi_2 * N(x | mu_2, var_2)

E-step  : responsibility  r_ik = pi_k N(x_i|mu_k,var_k) / sum_j pi_j N(x_i|mu_j,var_j)
M-step  : N_k   = sum_i r_ik
          mu_k  = sum_i r_ik x_i / N_k
          var_k = sum_i r_ik (x_i - mu_k)^2 / N_k
          pi_k  = N_k / N
Stop    : when |LL_t - LL_{t-1}| < tol, LL = sum_i log sum_k pi_k N(x_i|mu_k,var_k)
"""
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

LABELS = ("Child", "Father")
CM_PER_INCH = 2.54
VAR_FLOOR = 1e-6  # stops a component collapsing onto a single point


# ---------------------------------------------------------------- data
def load_heights(path="data/GaltonFamilies.csv", parent="father"):
    """Return (heights, true_labels). Labels are kept ONLY for evaluation,
    never passed to EM. Each parent is counted once (one row per family),
    because the raw file repeats the parents on every child's row."""
    df = pd.read_csv(path)
    children = df["childHeight"].to_numpy(float)
    parents = df.drop_duplicates("family")[parent].to_numpy(float)
    x = np.concatenate([children, parents])
    y = np.concatenate([np.zeros(len(children), int), np.ones(len(parents), int)])
    return x, y


def to_inches(height, unit="in"):
    return height / CM_PER_INCH if unit == "cm" else float(height)


# ---------------------------------------------------------------- params
@dataclass
class GMMParams:
    mu: np.ndarray
    var: np.ndarray
    pi: np.ndarray

    def as_row(self):
        return dict(mu1=self.mu[0], mu2=self.mu[1], var1=self.var[0],
                    var2=self.var[1], pi1=self.pi[0], pi2=self.pi[1])


def init_params(x, method="quantile"):
    """Starting guess. Order is fixed so component 1 is the shorter one.
    quantile  : means at the 25th/75th percentile, shared global variance, equal weights
    mean_split: hard split at the global mean (the naive method, used only as a start)"""
    if method == "quantile":
        mu = np.percentile(x, [25, 75])
        var = np.full(2, x.var())
        pi = np.array([0.5, 0.5])
    elif method == "mean_split":
        lo, hi = x[x < x.mean()], x[x >= x.mean()]
        mu = np.array([lo.mean(), hi.mean()])
        var = np.array([lo.var(), hi.var()])
        pi = np.array([len(lo), len(hi)]) / len(x)
    else:
        raise ValueError(f"unknown init method: {method}")
    return GMMParams(mu.astype(float), var.astype(float), pi.astype(float))


# ---------------------------------------------------------------- core math
def gaussian_pdf(x, mu, var):
    """N(x | mu, var). Broadcasts: x shape (n,1) against mu/var shape (k,) -> (n,k)."""
    return np.exp(-((x - mu) ** 2) / (2 * var)) / np.sqrt(2 * np.pi * var)


def weighted_densities(x, p):
    """pi_k * N(x_i | mu_k, var_k), shape (n, k)."""
    return p.pi * gaussian_pdf(np.atleast_1d(x)[:, None], p.mu, p.var)


def log_likelihood(x, p):
    return float(np.sum(np.log(weighted_densities(x, p).sum(axis=1))))


def e_step(x, p):
    """Soft assignment: each point gets a probability of belonging to each component."""
    wd = weighted_densities(x, p)
    return wd / wd.sum(axis=1, keepdims=True)


def m_step(x, r):
    """Re-estimate parameters using responsibilities as fractional counts."""
    n_k = r.sum(axis=0)
    mu = (r * x[:, None]).sum(axis=0) / n_k
    var = (r * (x[:, None] - mu) ** 2).sum(axis=0) / n_k
    return GMMParams(mu, np.maximum(var, VAR_FLOOR), n_k / len(x))


# ---------------------------------------------------------------- fit
@dataclass
class EMResult:
    params: GMMParams
    history: list = field(default_factory=list)  # one dict per iteration
    converged: bool = False

    def table(self, n_rows=None):
        df = pd.DataFrame(self.history).set_index("iteration")
        return df if n_rows is None else df.head(n_rows)


def fit_em(x, init="quantile", tol=1e-6, max_iter=500):
    p = init_params(x, init) if isinstance(init, str) else init
    history = [dict(iteration=0, **p.as_row(), log_likelihood=log_likelihood(x, p))]
    for it in range(1, max_iter + 1):
        r = e_step(x, p)
        p = m_step(x, r)
        ll = log_likelihood(x, p)
        history.append(dict(iteration=it, **p.as_row(), log_likelihood=ll))
        if abs(ll - history[-2]["log_likelihood"]) < tol:
            return EMResult(p, history, converged=True)
    return EMResult(p, history, converged=False)


# ---------------------------------------------------------------- use
def classify(height, params, unit="in"):
    """Posterior P(component | height) via Bayes' rule = the E-step for one point."""
    h = to_inches(height, unit)
    post = e_step(np.array([h]), params)[0]
    return {f"P({LABELS[k]})": float(post[k]) for k in range(2)}


def print_classification(height, params, unit="in"):
    h = to_inches(height, unit)
    post = classify(height, params, unit)
    wd = weighted_densities(np.array([h]), params)[0]
    print(f"Test height: {height} {unit}  ->  {h:.2f} in")
    for k, lab in enumerate(LABELS):
        print(f"  pi_{k+1}*N(x|mu_{k+1},var_{k+1}) [{lab:6}] = {wd[k]:.6e}")
    print(f"  evidence p(x)                      = {wd.sum():.6e}")
    for lab, v in post.items():
        print(f"  {lab:10} = {v:.6f}  ({v*100:.2f}%)")
    print(f"  -> classified as: {max(post, key=post.get)[2:-1]}")
    return post


def mean_split(x):
    """The naive method the brief asks about: hard cut at the global mean."""
    cut = x.mean()
    lo, hi = x[x < cut], x[x >= cut]
    return dict(cut=cut, mu=(lo.mean(), hi.mean()), sd=(lo.std(), hi.std()),
                n=(len(lo), len(hi)))


def accuracy(pred, y):
    return float((pred == y).mean())
