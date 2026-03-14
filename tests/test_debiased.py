import numpy as np
import pytest
import warnings

from highdim_inference_toolkit.debiased_lasso import DebiasedLasso
from highdim_inference_toolkit.utils import generate_high_dim_data


def test_debiased_lasso_ci_shapes_and_finiteness():
    X, y, beta_true = generate_high_dim_data(
        n=250, p=60, s=5, beta_strength=3.0, seed=0, sigma=1.0
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model = DebiasedLasso(lambda_param=0.08, lambda_debias=0.08).fit(X, y)

    j = int(np.flatnonzero(beta_true)[0]) if np.any(beta_true != 0) else 0
    est = model.debiased_coef(j)
    lo, hi = model.confidence_interval(j=j, alpha=0.05)
    z, p = model.hypothesis_test(j=j, null_value=0.0)

    assert np.isfinite(est)
    assert np.isfinite(lo) and np.isfinite(hi)
    assert lo < hi
    assert np.isfinite(z)
    assert 0.0 <= p <= 1.0


def test_debiased_lasso_detects_strong_signal():
    X, y, beta_true = generate_high_dim_data(
        n=600, p=80, s=1, beta_strength=10.0, seed=1, sigma=1.0
    )
    j = int(np.flatnonzero(beta_true)[0])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        model = DebiasedLasso(lambda_param=0.05, lambda_debias=0.05).fit(X, y)
    _z, p = model.hypothesis_test(j=j, null_value=0.0)
    assert p < 0.05
