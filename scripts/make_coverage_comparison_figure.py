from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t

# Allow running as `python scripts/make_coverage_comparison_figure.py` without installing.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from highdim_inference_toolkit.debiased_lasso import DebiasedLasso
from highdim_inference_toolkit.lasso import LassoCD


def _generate_with_fixed_support(
    *,
    n: int,
    p: int,
    support: list[int],
    beta_strength: float,
    sigma: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    beta = np.zeros(p, dtype=float)
    beta[support] = beta_strength
    y = X @ beta + rng.normal(0.0, sigma, size=n)
    return X, y, beta


def _center(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Xc = X - X.mean(axis=0)
    yc = y - y.mean()
    return Xc, yc


def _ols_ci(
    X: np.ndarray,
    y: np.ndarray,
    j_in_subset: int,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """OLS CI for coefficient index j_in_subset in design X (no intercept).

    Uses Student-t with df = n - d.
    """
    n, d = X.shape
    if n <= d + 2:
        raise ValueError("Need n > d + 2 for OLS CI")

    XtX = X.T @ X
    inv = np.linalg.inv(XtX)
    beta_hat = inv @ (X.T @ y)
    resid = y - X @ beta_hat
    sigma2 = float((resid @ resid) / (n - d))
    se = float(np.sqrt(max(sigma2 * inv[j_in_subset, j_in_subset], 0.0)))
    crit = float(t.ppf(1.0 - alpha / 2.0, df=n - d))
    b = float(beta_hat[j_in_subset])
    return (b - crit * se, b + crit * se)


@dataclass
class Summary:
    coverage: float
    avg_length: float
    selection_rate: float | None = None


def _run_one(
    *,
    n: int,
    p: int,
    s: int,
    beta_strength: float,
    sigma: float,
    lam: float,
    seed: int,
    j: int,
) -> tuple[bool, float, bool, float, bool, float, bool]:
    # Fix support so that j is always a signal coordinate.
    support = [j] + list(range(1, s)) if j != 0 else list(range(0, s))
    support = sorted(set(support))
    X, y, beta = _generate_with_fixed_support(
        n=n,
        p=p,
        support=support,
        beta_strength=beta_strength,
        sigma=sigma,
        seed=seed,
    )
    Xc, yc = _center(X, y)

    # Debiased Lasso
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        db = DebiasedLasso(lambda_param=lam, lambda_debias=lam).fit(Xc, yc)
    lo_db, hi_db = db.confidence_interval(j=j, alpha=0.05)
    covered_db = bool(lo_db <= beta[j] <= hi_db)
    len_db = float(hi_db - lo_db)

    # Oracle OLS on true support
    Xs = Xc[:, support]
    j_pos = support.index(j)
    lo_or, hi_or = _ols_ci(Xs, yc, j_pos, alpha=0.05)
    covered_or = bool(lo_or <= beta[j] <= hi_or)
    len_or = float(hi_or - lo_or)

    # Post-Lasso OLS (selection on full X, then OLS on selected subset)
    lasso = LassoCD().fit(Xc, yc, lambda_param=lam, max_iter=5000, tol=1e-6)
    coef = np.asarray(lasso.coef_)
    selected = np.where(np.abs(coef) > 1e-8)[0].tolist()
    if len(selected) == 0:
        selected = np.argsort(-np.abs(coef))[: max(1, s)].tolist()
    selected = sorted(set(selected))
    selected_j = j in selected
    if selected_j and len(selected) < n - 3:
        Xsel = Xc[:, selected]
        lo_ps, hi_ps = _ols_ci(Xsel, yc, selected.index(j), alpha=0.05)
        covered_ps = bool(lo_ps <= beta[j] <= hi_ps)
        len_ps = float(hi_ps - lo_ps)
    else:
        covered_ps = False
        len_ps = float("nan")

    return covered_db, len_db, covered_or, len_or, covered_ps, len_ps, selected_j


def main() -> None:
    p = 120
    s = 8
    beta_strength = 2.5
    sigma = 1.0
    j = 0
    lam = 0.08

    n_values = [120, 200, 350]
    n_sims = 80
    base_seed = 2026

    db_cov, db_len = [], []
    or_cov, or_len = [], []
    ps_cov, ps_len, ps_sel = [], [], []

    for n in n_values:
        c_db = 0
        c_or = 0
        c_ps = 0
        lens_db: list[float] = []
        lens_or: list[float] = []
        lens_ps: list[float] = []
        sel = 0
        sel_ci = 0

        for t_idx in range(n_sims):
            (
                covered_db,
                len_db_i,
                covered_or,
                len_or_i,
                covered_ps_i,
                len_ps_i,
                selected_j,
            ) = _run_one(
                n=n,
                p=p,
                s=s,
                beta_strength=beta_strength,
                sigma=sigma,
                lam=lam,
                seed=base_seed + 1000 * n + t_idx,
                j=j,
            )
            c_db += int(covered_db)
            c_or += int(covered_or)
            lens_db.append(len_db_i)
            lens_or.append(len_or_i)

            sel += int(selected_j)
            if np.isfinite(len_ps_i):
                sel_ci += 1
                c_ps += int(covered_ps_i)
                lens_ps.append(len_ps_i)

        db_cov.append(c_db / n_sims)
        or_cov.append(c_or / n_sims)
        db_len.append(float(np.mean(lens_db)))
        or_len.append(float(np.mean(lens_or)))

        ps_sel.append(sel / n_sims)
        if sel_ci > 0:
            ps_cov.append(c_ps / sel_ci)
            ps_len.append(float(np.mean(lens_ps)))
        else:
            ps_cov.append(float("nan"))
            ps_len.append(float("nan"))

    out_dir = _REPO_ROOT / "assets"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "ci_coverage_comparison.png"

    fig, ax = plt.subplots(1, 2, figsize=(11, 4))

    ax[0].plot(n_values, db_cov, marker="o", label="Debiased Lasso")
    ax[0].plot(n_values, or_cov, marker="o", label="Oracle OLS (true support)")
    ax[0].plot(n_values, ps_cov, marker="o", label="Post-Lasso OLS (selected)")
    ax[0].axhline(0.95, linestyle="--", linewidth=1, color="gray")
    ax[0].set_title("Empirical coverage (95% CI)")
    ax[0].set_xlabel("n")
    ax[0].set_ylabel("coverage")
    ax[0].set_ylim(0.0, 1.0)
    ax[0].legend(fontsize=9)

    ax[1].plot(n_values, db_len, marker="o", label="Debiased Lasso")
    ax[1].plot(n_values, or_len, marker="o", label="Oracle OLS")
    ax[1].plot(n_values, ps_len, marker="o", label="Post-Lasso OLS")
    ax[1].set_title("Average CI length")
    ax[1].set_xlabel("n")
    ax[1].set_ylabel("length")
    ax[1].legend(fontsize=9)

    fig.suptitle(
        "CI comparison (j is a true signal)"
        f" | p={p}, s={s}, sigma={sigma}, beta={beta_strength}, lambda={lam}",
        fontsize=11,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    print(f"Wrote {out_path}")
    print("Post-Lasso selection rate (j selected):", list(zip(n_values, ps_sel)))


if __name__ == "__main__":
    main()
