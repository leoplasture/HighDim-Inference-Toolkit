from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .lasso import LassoCD


@dataclass
class TransLasso:
    """Trans-Lasso (two-step) for transfer learning in high-dimensional regression."""

    lambda_w: float
    lambda_delta: float
    n_auxiliary_datasets: int | None = None

    coef_: np.ndarray | None = None
    intercept_: float = 0.0

    _w_target_: np.ndarray | None = None
    _w_aux_list_: list[np.ndarray] | None = None

    def fit(
        self,
        X_target: np.ndarray,
        y_target: np.ndarray,
        X_auxiliary_list: list[np.ndarray] | None = None,
        y_auxiliary_list: list[np.ndarray] | None = None,
        max_iter: int = 2000,
        tol: float = 1e-4,
    ) -> "TransLasso":
        X_target = np.asarray(X_target, dtype=float)
        y_target = np.asarray(y_target, dtype=float).reshape(-1)
        X_auxiliary_list = X_auxiliary_list or []
        y_auxiliary_list = y_auxiliary_list or []
        if len(X_auxiliary_list) != len(y_auxiliary_list):
            raise ValueError("Auxiliary X/y lists must have the same length")

        Xs = [X_target] + [np.asarray(x, dtype=float) for x in X_auxiliary_list]
        ys = [y_target] + [
            np.asarray(y, dtype=float).reshape(-1) for y in y_auxiliary_list
        ]

        X_pool = np.vstack(Xs)
        y_pool = np.concatenate(ys, axis=0)

        w_pooled = LassoCD().fit(
            X_pool, y_pool, lambda_param=self.lambda_w, max_iter=max_iter, tol=tol
        )

        w_target = LassoCD().fit(
            X_target, y_target, lambda_param=self.lambda_w, max_iter=max_iter, tol=tol
        )
        self._w_target_ = np.asarray(w_target.coef_, dtype=float)
        self._w_aux_list_ = []
        for Xa, ya in zip(X_auxiliary_list, y_auxiliary_list):
            wk = LassoCD().fit(
                Xa, ya, lambda_param=self.lambda_w, max_iter=max_iter, tol=tol
            )
            self._w_aux_list_.append(np.asarray(wk.coef_, dtype=float))

        r_target = y_target - w_pooled.predict(X_target)
        delta = LassoCD().fit(
            X_target,
            r_target,
            lambda_param=self.lambda_delta,
            max_iter=max_iter,
            tol=tol,
        )

        self.coef_ = np.asarray(w_pooled.coef_, dtype=float) + np.asarray(
            delta.coef_, dtype=float
        )
        self.intercept_ = float(w_pooled.intercept_ + delta.intercept_)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model is not fitted. Call fit() first.")
        X = np.asarray(X, dtype=float)
        return self.intercept_ + X @ self.coef_

    def get_informative_set(self, threshold: float) -> list[int]:
        if self._w_target_ is None or self._w_aux_list_ is None:
            raise RuntimeError(
                "Call fit() with auxiliary datasets before get_informative_set()."
            )
        if threshold <= 0:
            raise ValueError("threshold must be > 0")

        p = self._w_target_.shape[0]
        if 0.0 < threshold < 1.0:
            max_diff = int(np.floor(threshold * p))
        else:
            max_diff = int(threshold)

        eps = 1e-3
        informative: list[int] = []
        for k, w_aux in enumerate(self._w_aux_list_):
            diff_count = int(np.sum(np.abs(w_aux - self._w_target_) > eps))
            if diff_count <= max_diff:
                informative.append(k)
        return informative
