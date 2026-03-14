from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _soft_threshold(value: float, threshold: float) -> float:
    if value > threshold:
        return value - threshold
    if value < -threshold:
        return value + threshold
    return 0.0


@dataclass
class LassoCD:
    """Lasso regression via coordinate descent (NumPy-only).

    Minimizes:
        (1 / 2n) * ||y - X beta||_2^2 + lambda_param * ||beta||_1

    Notes:
        - Handles intercept by centering internally.
        - Standardizes features to unit RMS for stable updates.
    """

    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    _x_mean: np.ndarray | None = None
    _x_scale: np.ndarray | None = None
    _y_mean: float = 0.0

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        lambda_param: float,
        max_iter: int = 1000,
        tol: float = 1e-4,
    ) -> "LassoCD":
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float).reshape(-1)

        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")
        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y have incompatible shapes")
        if lambda_param < 0:
            raise ValueError("lambda_param must be >= 0")
        if n == 0 or p == 0:
            raise ValueError("X must have non-zero dimensions")

        self._x_mean = X.mean(axis=0)
        X_centered = X - self._x_mean
        self._x_scale = np.sqrt((X_centered**2).mean(axis=0))
        self._x_scale[self._x_scale == 0.0] = 1.0
        Xs = X_centered / self._x_scale

        self._y_mean = float(y.mean())
        ys = y - self._y_mean

        beta = np.zeros(p, dtype=float)
        r = ys.copy()
        z = (Xs**2).mean(axis=0)
        z[z == 0.0] = 1.0

        for _ in range(max_iter):
            beta_prev = beta.copy()
            for j in range(p):
                r_j = r + Xs[:, j] * beta[j]
                rho = float((Xs[:, j] * r_j).mean())
                beta_j_new = _soft_threshold(rho, lambda_param) / z[j]
                if beta_j_new != beta[j]:
                    r = r_j - Xs[:, j] * beta_j_new
                    beta[j] = beta_j_new

            if np.max(np.abs(beta - beta_prev)) < tol:
                break

        self.coef_ = beta / self._x_scale
        self.intercept_ = self._y_mean - float(self._x_mean @ self.coef_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_
