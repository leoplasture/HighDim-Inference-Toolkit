from __future__ import annotations

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

# Allow running as `python scripts/make_correlation_sweep_figure.py` without installing.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from highdim_inference_toolkit.debiased_lasso import DebiasedLasso
from highdim_inference_toolkit.utils import generate_high_dim_data


def _center(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return X - X.mean(axis=0), y - y.mean()


def _one_run(
    *,
    n: int,
    p: int,
    s: int,
    rho: float,
    beta_strength: float,
    sigma: float,
    lam: float,
    j: int,
    seed: int,
) -> tuple[bool, float]:
    X, y, beta = generate_high_dim_data(
        n=n,
        p=p,
        s=s,
        beta_strength=beta_strength,
        covariance="toeplitz",
        rho=rho,
        seed=seed,
        sigma=sigma,
    )
    Xc, yc = _center(X, y)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        m = DebiasedLasso(lambda_param=lam, lambda_debias=lam).fit(Xc, yc)
    lo, hi = m.confidence_interval(j=j, alpha=0.05)
    return (lo <= beta[j] <= hi), float(hi - lo)


def main() -> None:
    # This is a sensitivity demo: how correlation (Toeplitz rho) affects inference quality.
    p = 120
    s = 8
    beta_strength = 2.5
    sigma = 1.0
    j = 0
    lam = 0.08

    # Sweep correlation.
    rhos = [0.0, 0.3, 0.6, 0.9]
    n_values = [120, 200, 350]
    n_sims = 60
    base_seed = 2026

    coverage = {n: [] for n in n_values}
    length = {n: [] for n in n_values}

    for n in n_values:
        for rho in rhos:
            covered = 0
            lens: list[float] = []
            for t_idx in range(n_sims):
                c, L = _one_run(
                    n=n,
                    p=p,
                    s=s,
                    rho=rho,
                    beta_strength=beta_strength,
                    sigma=sigma,
                    lam=lam,
                    j=j,
                    seed=base_seed + 10_000 * n + 100 * int(rho * 10) + t_idx,
                )
                covered += int(c)
                lens.append(L)
            coverage[n].append(covered / n_sims)
            length[n].append(float(np.mean(lens)))

    out_dir = _REPO_ROOT / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ci_coverage_vs_correlation.png"

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    for n in n_values:
        ax[0].plot(rhos, coverage[n], marker="o", label=f"n={n}")
    ax[0].axhline(0.95, linestyle="--", linewidth=1, color="gray")
    ax[0].set_title("Debiased Lasso: coverage vs correlation")
    ax[0].set_xlabel("Toeplitz correlation ρ")
    ax[0].set_ylabel("empirical coverage")
    ax[0].set_ylim(0.0, 1.0)
    ax[0].legend(fontsize=9)

    for n in n_values:
        ax[1].plot(rhos, length[n], marker="o", label=f"n={n}")
    ax[1].set_title("Debiased Lasso: CI length vs correlation")
    ax[1].set_xlabel("Toeplitz correlation ρ")
    ax[1].set_ylabel("avg CI length")
    ax[1].legend(fontsize=9)

    fig.suptitle(
        "Sensitivity to feature correlation"
        f" | p={p}, s={s}, sigma={sigma}, beta={beta_strength}, lambda={lam}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
