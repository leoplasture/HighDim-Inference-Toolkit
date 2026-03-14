from __future__ import annotations

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.stats import norm

from .lasso import LassoCD


def _estimate_sigma(residual: np.ndarray) -> float:
    residual = np.asarray(residual, dtype=float).reshape(-1)
    sigma2 = float(np.mean(residual**2))
    return float(np.sqrt(max(sigma2, 0.0)))


@dataclass
class DebiasedLasso:
    """Debiased Lasso for inference in high-dimensional linear regression.

    Uses a standard bias-correction step:
       beta_db,j = beta_lasso,j + theta_j^T X^T (y - X beta_lasso) / n
    where theta_j is estimated via nodewise Lasso.
    """

    lambda_param: float
    lambda_debias: float
    lasso_max_iter: int = 2000
    lasso_tol: float = 1e-4

    beta_lasso_: np.ndarray | None = None
    intercept_: float = 0.0
    sigma_hat_: float | None = None
    _thetas: dict[int, np.ndarray] | None = None
    _X: np.ndarray | None = None
    _y: np.ndarray | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DebiasedLasso":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)
        if X.ndim != 2:
            raise ValueError("X must be 2D")
        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y have incompatible shapes")

        lasso = LassoCD().fit(
            X,
            y,
            lambda_param=self.lambda_param,
            max_iter=self.lasso_max_iter,
            tol=self.lasso_tol,
        )
        self.beta_lasso_ = np.asarray(lasso.coef_, dtype=float)
        self.intercept_ = float(lasso.intercept_)
        residual = y - (self.intercept_ + X @ self.beta_lasso_)
        self.sigma_hat_ = _estimate_sigma(residual)

        s_hat = int(np.sum(np.abs(self.beta_lasso_) > 0))
        if p > 1 and n < (s_hat * np.log(p)) ** 2:
            warnings.warn(
                "Sample size condition may be violated (n < (s*log(p))^2); CIs may be invalid.",
                RuntimeWarning,
            )

        self._thetas = {}
        self._X = X
        self._y = y
        return self

    def _theta(self, j: int) -> np.ndarray:
        if self._thetas is None or self._X is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        if j in self._thetas:
            return self._thetas[j]

        X = self._X
        n, p = X.shape
        if not (0 <= j < p):
            raise IndexError("j out of range")

        mask = np.ones(p, dtype=bool)
        mask[j] = False
        X_minus = X[:, mask]
        x_j = X[:, j]

        node_lasso = LassoCD().fit(
            X_minus,
            x_j,
            lambda_param=self.lambda_debias,
            max_iter=self.lasso_max_iter,
            tol=self.lasso_tol,
        )
        gamma = np.asarray(node_lasso.coef_, dtype=float)
        r = x_j - (node_lasso.intercept_ + X_minus @ gamma)

        tau2 = float(np.mean(r**2))
        tau2 = max(tau2, 1e-12)

        theta = np.zeros(p, dtype=float)
        theta[j] = 1.0
        theta[mask] = -gamma
        theta = theta / tau2

        self._thetas[j] = theta
        return theta

    def debiased_coef(self, j: int) -> float:
        if self.beta_lasso_ is None or self._X is None or self._y is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = self._X
        y = self._y
        n = X.shape[0]

        theta = self._theta(j)
        residual = y - (self.intercept_ + X @ self.beta_lasso_)
        score = (X.T @ residual) / n
        return float(self.beta_lasso_[j] + theta @ score)

    def _std_err(self, j: int) -> float:
        if self.sigma_hat_ is None or self._X is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = self._X
        n = X.shape[0]
        theta = self._theta(j)
        Sigma_hat = (X.T @ X) / n
        var = (self.sigma_hat_**2) * float(theta @ Sigma_hat @ theta) / n
        return float(np.sqrt(max(var, 0.0)))

    def confidence_interval(self, j: int, alpha: float = 0.05) -> tuple[float, float]:
        if not (0.0 < alpha < 1.0):
            raise ValueError("alpha must be in (0, 1)")
        est = self.debiased_coef(j)
        se = self._std_err(j)
        z = float(norm.ppf(1.0 - alpha / 2.0))
        return (est - z * se, est + z * se)

    def hypothesis_test(self, j: int, null_value: float = 0.0) -> tuple[float, float]:
        est = self.debiased_coef(j)
        se = self._std_err(j)
        if se == 0.0:
            return (
                np.inf if est != null_value else 0.0,
                0.0 if est != null_value else 1.0,
            )
        z = (est - null_value) / se
        p = 2.0 * float(1.0 - norm.cdf(abs(z)))
        return (float(z), p)
