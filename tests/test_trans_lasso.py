import numpy as np

from highdim_inference_toolkit.trans_lasso import TransLasso
from highdim_inference_toolkit.utils import generate_transfer_learning_data


def test_trans_lasso_fit_predict_shapes():
    (
        X_target,
        y_target,
        X_aux_list,
        y_aux_list,
        _beta_true,
        _w_list,
    ) = generate_transfer_learning_data(
        n_target=120,
        n_aux=120,
        p=40,
        s=5,
        h=3,
        K_auxiliary=2,
        seed=0,
        sigma=1.0,
    )

    model = TransLasso(lambda_w=0.08, lambda_delta=0.08).fit(
        X_target, y_target, X_auxiliary_list=X_aux_list, y_auxiliary_list=y_aux_list
    )
    pred = model.predict(X_target[:10])
    assert pred.shape == (10,)
    assert model.coef_ is not None
    assert model.coef_.shape == (X_target.shape[1],)


def test_trans_lasso_informative_set_returns_valid_indices():
    (
        X_target,
        y_target,
        X_aux_list,
        y_aux_list,
        _beta_true,
        _w_list,
    ) = generate_transfer_learning_data(
        n_target=80,
        n_aux=80,
        p=30,
        s=4,
        h=2,
        K_auxiliary=3,
        seed=2,
        sigma=1.0,
    )
    model = TransLasso(lambda_w=0.1, lambda_delta=0.1).fit(
        X_target, y_target, X_auxiliary_list=X_aux_list, y_auxiliary_list=y_aux_list
    )
    informative = model.get_informative_set(threshold=0.9)
    assert isinstance(informative, list)
    assert all(isinstance(k, int) for k in informative)
    assert all(0 <= k < len(X_aux_list) for k in informative)
