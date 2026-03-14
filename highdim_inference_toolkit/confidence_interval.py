from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import numpy as np
from scipy.stats import norm


@dataclass
class HighDimCI:
    """Confidence interval helpers for high-dimensional inference."""

    @staticmethod
    def normal_approx(
        debiased_coef: float, std_err: float, alpha: float = 0.05
    ) -> tuple[float, float]:
        if std_err < 0:
            raise ValueError("std_err must be >= 0")
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        z = float(norm.ppf(1.0 - alpha / 2.0))
        return (debiased_coef - z * std_err, debiased_coef + z * std_err)

    @staticmethod
    def bootstrap_ci(
        X: np.ndarray,
        y: np.ndarray,
        estimator_factory: Callable[[], Any],
        j: int,
        n_bootstraps: int = 1000,
        alpha: float = 0.05,
        seed: int | None = None,
    ) -> tuple[float, float]:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        n = X.shape[0]
        rng = np.random.default_rng(seed)
        stats = []
        for _ in range(n_bootstraps):
            idx = rng.integers(0, n, size=n)
            est = estimator_factory()
            est.fit(X[idx], y[idx])
            if hasattr(est, "debiased_coef"):
                stats.append(float(est.debiased_coef(j)))
            else:
                stats.append(float(np.asarray(est.coef_)[j]))

        lo = float(np.quantile(stats, alpha / 2.0))
        hi = float(np.quantile(stats, 1.0 - alpha / 2.0))
        return (lo, hi)

    @staticmethod
    def coverage_simulation(
        true_beta: np.ndarray,
        X: np.ndarray,
        estimator_factory: Callable[[], Any],
        j: int,
        n_simulations: int = 200,
        alpha: float = 0.05,
        sigma: float = 1.0,
        seed: int | None = None,
    ) -> dict[str, float]:
        """Monte Carlo coverage/length for an estimator's CI method.

        The estimator created by estimator_factory must implement:
        - fit(X, y)
        - confidence_interval(j, alpha)
        """
        rng = np.random.default_rng(seed)
        true_beta = np.asarray(true_beta, dtype=float).reshape(-1)
        X = np.asarray(X, dtype=float)
        n = X.shape[0]
        covered = 0
        lengths: list[float] = []
        for _ in range(n_simulations):
            y = X @ true_beta + rng.normal(0.0, sigma, size=n)
            est = estimator_factory()
            est.fit(X, y)
            lo, hi = est.confidence_interval(j=j, alpha=alpha)
            lengths.append(float(hi - lo))
            covered += int(lo <= true_beta[j] <= hi)

        return {
            "coverage": float(covered) / float(n_simulations),
            "avg_length": float(np.mean(lengths)) if lengths else float("nan"),
        }
