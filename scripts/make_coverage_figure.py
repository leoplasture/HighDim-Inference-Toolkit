from __future__ import annotations

from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np

# Allow running as `python scripts/make_coverage_figure.py` without installing the package.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from highdim_inference_toolkit.debiased_lasso import DebiasedLasso
from highdim_inference_toolkit.utils import generate_high_dim_data


def _one_run(
    *,
    n: int,
    p: int,
    s: int,
    beta_strength: float,
    j: int,
    sigma: float,
    lambda_param: float,
    lambda_debias: float,
    seed: int,
) -> tuple[bool, float]:
    X, y, beta_true = generate_high_dim_data(
        n=n,
        p=p,
        s=s,
        beta_strength=beta_strength,
        seed=seed,
        sigma=sigma,
    )

    j_use = int(j)
    if not (0 <= j_use < beta_true.shape[0]):
        raise IndexError("j out of range")
    if beta_true[j_use] == 0.0:
        nz = np.flatnonzero(beta_true)
        if nz.size > 0:
            j_use = int(nz[0])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model = DebiasedLasso(
            lambda_param=lambda_param, lambda_debias=lambda_debias
        ).fit(X, y)
    lo, hi = model.confidence_interval(j=j_use, alpha=0.05)
    covered = bool(lo <= beta_true[j_use] <= hi)
    return covered, float(hi - lo)


def main() -> None:
    # Keep this demo lightweight and deterministic.
    p = 80
    s = 5
    beta_strength = 3.0
    sigma = 1.0
    j = 0  # If j is not a signal coordinate in a run, we fall back to a signal index.
    # Penalties chosen for a quick, stable demo (not tuned).
    lambda_param = 0.08
    lambda_debias = 0.08

    n_values = [120, 200, 350]
    n_sims = 60
    base_seed = 2026

    coverages: list[float] = []
    lengths: list[float] = []
    for n in n_values:
        covered = 0
        lens: list[float] = []
        for t in range(n_sims):
            c, L = _one_run(
                n=n,
                p=p,
                s=s,
                beta_strength=beta_strength,
                j=j,
                sigma=sigma,
                lambda_param=lambda_param,
                lambda_debias=lambda_debias,
                seed=base_seed + 1000 * n + t,
            )
            covered += int(c)
            lens.append(L)
        coverages.append(covered / n_sims)
        lengths.append(float(np.mean(lens)))

    out_dir = Path(__file__).resolve().parents[1] / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "debiased_lasso_coverage.png"

    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].plot(n_values, coverages, marker="o")
    ax[0].axhline(0.95, linestyle="--", linewidth=1)
    ax[0].set_title("Empirical coverage (95% CI)")
    ax[0].set_xlabel("n")
    ax[0].set_ylabel("coverage")
    ax[0].set_ylim(0.0, 1.0)

    ax[1].plot(n_values, lengths, marker="o")
    ax[1].set_title("Average CI length")
    ax[1].set_xlabel("n")
    ax[1].set_ylabel("length")

    fig.suptitle(
        "Debiased Lasso Wald CI demo"
        f" (p={p}, s={s}, sigma={sigma}, lambda={lambda_param})",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
