import numpy as np
import pytest
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score

from highdim_inference_toolkit.lasso import LassoCD
from highdim_inference_toolkit.utils import generate_high_dim_data


@pytest.fixture(scope="module")
def sparse_problem():
    X, y, beta = generate_high_dim_data(
        n=120, p=60, s=8, beta_strength=2.0, seed=123, sigma=0.5
    )
    idx = np.arange(X.shape[0])
    train = idx[:100]
    test = idx[100:]
    return X[train], y[train], X[test], y[test], beta


def test_lasso_runs_and_shapes(sparse_problem):
    Xtr, ytr, _, _, _ = sparse_problem
    model = LassoCD().fit(Xtr, ytr, lambda_param=0.05, max_iter=5000, tol=1e-6)
    assert model.coef_ is not None
    assert model.coef_.shape == (Xtr.shape[1],)


def test_prediction_accuracy_r2(sparse_problem):
    Xtr, ytr, Xte, yte, _ = sparse_problem
    model = LassoCD().fit(Xtr, ytr, lambda_param=0.05, max_iter=5000, tol=1e-6)
    r2 = r2_score(yte, model.predict(Xte))
    assert r2 > 0.6, f"Expected reasonable prediction accuracy, got R2={r2:.3f}"


def test_comparison_with_sklearn(sparse_problem):
    Xtr, ytr, _, _, _ = sparse_problem
    lam = 0.05

    ours = LassoCD().fit(Xtr, ytr, lambda_param=lam, max_iter=8000, tol=1e-6)
    sk = Lasso(alpha=lam, fit_intercept=True, max_iter=10000, tol=1e-6)
    sk.fit(Xtr, ytr)

    corr = float(np.corrcoef(ours.coef_, sk.coef_)[0, 1])
    assert corr > 0.7, f"Expected similar coefficient pattern; corr={corr:.3f}"


def test_edge_case_n_less_than_p():
    X, y, _ = generate_high_dim_data(
        n=40, p=100, s=5, beta_strength=1.5, seed=7, sigma=1.0
    )
    model = LassoCD().fit(X, y, lambda_param=0.1, max_iter=2000, tol=1e-4)
    assert np.isfinite(model.intercept_)
    assert np.all(np.isfinite(model.coef_))


def test_all_zero_beta_reasonable():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 30))
    y = rng.normal(size=80)
    model = LassoCD().fit(X, y, lambda_param=0.5, max_iter=3000, tol=1e-6)
    assert np.linalg.norm(model.coef_, ord=1) < 5.0
