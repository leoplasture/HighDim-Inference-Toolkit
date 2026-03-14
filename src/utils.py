"""Utility helpers used across the toolkit.

The functions in this module are intentionally lightweight (NumPy-only) and are
used by examples and unit tests for:

- feature/response standardization
- synthetic data generation for high-dimensional regression and transfer learning
- simple evaluation metrics (estimation/prediction/support recovery)

This file mirrors the public utilities in the installed package.
"""

from __future__ import annotations

import numpy as np


def standardize(
    X: np.ndarray, y: np.ndarray | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None, float | None]:
    """Standardize a design matrix (and optionally center a response).

    This helper performs column-wise centering and scaling of ``X``:

    - ``x_mean[j] = mean(X[:, j])``
    - ``x_scale[j] = sqrt(mean((X[:, j] - x_mean[j])**2))``

    Columns with zero empirical variance are left unscaled by setting their
    scale to 1.0.

    If ``y`` is provided, it is reshaped to 1D and centered.

    Args:
        X: Design matrix of shape ``(n, p)``.
        y: Optional response vector of shape ``(n,)`` (or any array-like that
            flattens to length ``n``).

    Returns:
        A 5-tuple ``(Xs, x_mean, x_scale, ys, y_mean)`` where:

        - ``Xs``: Standardized design matrix of shape ``(n, p)``.
        - ``x_mean``: Column means of ``X`` with shape ``(p,)``.
        - ``x_scale``: Column scales of ``X`` with shape ``(p,)``.
        - ``ys``: Centered response (shape ``(n,)``) or ``None`` if ``y`` is
          ``None``.
        - ``y_mean``: Mean of ``y`` as a float, or ``None`` if ``y`` is ``None``.
    """
    X = np.asarray(X, dtype=float)
    x_mean = X.mean(axis=0)
    Xc = X - x_mean
    x_scale = np.sqrt((Xc**2).mean(axis=0))
    x_scale[x_scale == 0.0] = 1.0
    Xs = Xc / x_scale
    if y is None:
        return Xs, x_mean, x_scale, None, None
    y = np.asarray(y, dtype=float).reshape(-1)
    y_mean = float(y.mean())
    ys = y - y_mean
    return Xs, x_mean, x_scale, ys, y_mean


def generate_high_dim_data(
    n: int,
    p: int,
    s: int,
    beta_strength: float = 1.0,
    covariance: str = "identity",
    rho: float = 0.5,
    seed: int | None = None,
    sigma: float = 1.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Generate synthetic sparse linear-regression data.

    Constructs a sparse coefficient vector ``beta`` with ``s`` non-zeros (each
    equal to ``±beta_strength``), draws a design matrix ``X`` with either
    independent standard normal columns or a Toeplitz correlation structure, and
    returns responses from the model:

    $$y = X\beta + \varepsilon, \quad \varepsilon \sim \mathcal{N}(0, \sigma^2 I_n).$$

    Args:
        n: Number of samples.
        p: Number of features.
        s: Sparsity level (number of non-zeros in ``beta``). Must satisfy
            ``0 <= s <= p``.
        beta_strength: Magnitude of each non-zero coefficient.
        covariance: Covariance structure for ``X``. Supported values:
            ``"identity"`` (i.i.d. standard normal) and ``"toeplitz"``.
        rho: Correlation parameter for Toeplitz covariance. Must satisfy
            ``0 <= rho < 1`` when ``covariance="toeplitz"``.
        seed: Optional seed for reproducibility.
        sigma: Noise standard deviation.

    Returns:
        ``(X, y, beta)`` where:

        - ``X`` has shape ``(n, p)``
        - ``y`` has shape ``(n,)``
        - ``beta`` has shape ``(p,)``

    Raises:
        ValueError: If ``s`` is outside ``[0, p]``, if ``rho`` is invalid for the
            Toeplitz option, or if an unknown covariance type is requested.
    """
    if not (0 <= s <= p):
        raise ValueError("s must satisfy 0 <= s <= p")
    rng = np.random.default_rng(seed)

    if covariance == "identity":
        X = rng.normal(size=(n, p))
    elif covariance == "toeplitz":
        if not (0.0 <= rho < 1.0):
            raise ValueError("rho must be in [0, 1) for toeplitz covariance")
        idx = np.arange(p)
        Sigma = rho ** np.abs(idx[:, None] - idx[None, :])
        L = np.linalg.cholesky(Sigma + 1e-10 * np.eye(p))
        X = rng.normal(size=(n, p)) @ L.T
    else:
        raise ValueError("Unknown covariance type")

    beta = np.zeros(p, dtype=float)
    if s > 0:
        support = rng.choice(p, size=s, replace=False)
        signs = rng.choice([-1.0, 1.0], size=s)
        beta[support] = beta_strength * signs

    y = X @ beta + rng.normal(0.0, sigma, size=n)
    return X, y, beta


def generate_transfer_learning_data(
    n_target: int,
    n_aux: int,
    p: int,
    s: int,
    h: int,
    K_auxiliary: int,
    seed: int | None = None,
    sigma: float = 1.0,
) -> tuple[
    np.ndarray,
    np.ndarray,
    list[np.ndarray],
    list[np.ndarray],
    np.ndarray,
    list[np.ndarray],
]:
    r"""Generate a simple transfer-learning regression benchmark.

    The target task follows a sparse linear model with coefficient vector
    ``beta_true`` having ``s`` non-zero entries.

    Each auxiliary task ``k`` has coefficient

    $$w^{(k)} = \beta - \delta^{(k)},$$

    where the contrast vector ``delta^(k)`` has ``h`` non-zeros (each equal to
    ``±1``).

    Args:
            n_target: Number of samples in the target dataset.
            n_aux: Number of samples per auxiliary dataset.
            p: Number of features.
            s: Sparsity level for ``beta_true``.
            h: Sparsity level for each contrast vector ``delta^(k)``.
            K_auxiliary: Number of auxiliary datasets/tasks.
            seed: Optional seed for reproducibility.
            sigma: Noise standard deviation (used for target and auxiliaries).

    Returns:
            A 6-tuple ``(X_target, y_target, X_aux_list, y_aux_list, beta_true, w_list)``:

            - ``X_target``: Target design matrix, shape ``(n_target, p)``.
            - ``y_target``: Target response vector, shape ``(n_target,)``.
            - ``X_aux_list``: List of auxiliary design matrices (length ``K_auxiliary``),
                each of shape ``(n_aux, p)``.
            - ``y_aux_list``: List of auxiliary response vectors (length ``K_auxiliary``),
                each of shape ``(n_aux,)``.
            - ``beta_true``: Target coefficient vector, shape ``(p,)``.
            - ``w_list``: List of auxiliary coefficient vectors (length ``K_auxiliary``),
                each of shape ``(p,)``.

    Notes:
            The target dataset is generated via :func:`generate_high_dim_data` using the
            provided ``seed``. The auxiliary datasets share the same random generator
            stream initialized with ``seed`` for the contrasts and auxiliary designs.
    """
    rng = np.random.default_rng(seed)
    X_target, y_target, beta_true = generate_high_dim_data(
        n=n_target, p=p, s=s, beta_strength=1.0, seed=seed, sigma=sigma
    )

    w_list: list[np.ndarray] = []
    X_aux_list: list[np.ndarray] = []
    y_aux_list: list[np.ndarray] = []

    for k in range(K_auxiliary):
        delta = np.zeros(p, dtype=float)
        if h > 0:
            idx = rng.choice(p, size=min(h, p), replace=False)
            delta[idx] = rng.choice([-1.0, 1.0], size=idx.shape[0])
        w = beta_true - delta
        Xk = rng.normal(size=(n_aux, p))
        yk = Xk @ w + rng.normal(0.0, sigma, size=n_aux)
        w_list.append(w)
        X_aux_list.append(Xk)
        y_aux_list.append(yk)

    return X_target, y_target, X_aux_list, y_aux_list, beta_true, w_list


def estimation_error(
    beta_hat: np.ndarray, beta_true: np.ndarray, metric: str = "l2"
) -> float:
    """Compute a norm-based estimation error between coefficients.

    Args:
        beta_hat: Estimated coefficient vector.
        beta_true: Ground-truth coefficient vector.
        metric: Which norm to use. Supported values:
            ``"l2"`` (Euclidean), ``"l1"`` (Manhattan), ``"linf"`` (max-abs).

    Returns:
        The selected norm of ``beta_hat - beta_true`` as a float.

    Raises:
        ValueError: If ``metric`` is not one of ``{"l2", "l1", "linf"}``.
    """
    beta_hat = np.asarray(beta_hat, dtype=float)
    beta_true = np.asarray(beta_true, dtype=float)
    diff = beta_hat - beta_true
    if metric == "l2":
        return float(np.linalg.norm(diff))
    if metric == "l1":
        return float(np.linalg.norm(diff, ord=1))
    if metric == "linf":
        return float(np.max(np.abs(diff)))
    raise ValueError("Unknown metric")


def prediction_error(
    X_test: np.ndarray, y_test: np.ndarray, beta_hat: np.ndarray, intercept: float = 0.0
) -> float:
    r"""Compute mean squared prediction error on held-out data.

    Computes

    $$\frac{1}{n}\sum_{i=1}^n (y_i - (\text{intercept} + x_i^\top \hat\beta))^2.$$

    Args:
        X_test: Test design matrix of shape ``(n, p)``.
        y_test: Test response vector of shape ``(n,)`` (or array-like that flattens to ``n``).
        beta_hat: Estimated coefficient vector of shape ``(p,)``.
        intercept: Optional intercept term added to predictions.

    Returns:
        Mean squared error as a float.
    """
    X_test = np.asarray(X_test, dtype=float)
    y_test = np.asarray(y_test, dtype=float).reshape(-1)
    beta_hat = np.asarray(beta_hat, dtype=float)
    pred = intercept + X_test @ beta_hat
    return float(np.mean((y_test - pred) ** 2))


def support_recovery(
    beta_hat: np.ndarray, beta_true: np.ndarray, threshold: float = 1e-6
) -> dict[str, float]:
    """Compute precision/recall for support recovery.

    The *support* of a coefficient vector is the index set of entries whose
    absolute value exceeds ``threshold``.

    Args:
        beta_hat: Estimated coefficient vector.
        beta_true: Ground-truth coefficient vector.
        threshold: Absolute-value cutoff used to define the support.

    Returns:
        Dictionary with keys ``"precision"`` and ``"recall"``.

        If the predicted support is empty, precision is defined as 1.0.
        If the true support is empty, recall is defined as 1.0.
    """
    beta_hat = np.asarray(beta_hat, dtype=float)
    beta_true = np.asarray(beta_true, dtype=float)
    supp_hat = set(np.where(np.abs(beta_hat) > threshold)[0].tolist())
    supp_true = set(np.where(np.abs(beta_true) > threshold)[0].tolist())
    tp = len(supp_hat & supp_true)
    fp = len(supp_hat - supp_true)
    fn = len(supp_true - supp_hat)
    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    return {"precision": float(precision), "recall": float(recall)}
