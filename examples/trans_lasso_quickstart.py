import numpy as np

from highdim_inference_toolkit.trans_lasso import TransLasso
from highdim_inference_toolkit.utils import generate_transfer_learning_data


def main() -> None:
    (
        X_target,
        y_target,
        X_aux_list,
        y_aux_list,
        beta_true,
        _w_list,
    ) = generate_transfer_learning_data(
        n_target=150,
        n_aux=150,
        p=60,
        s=6,
        h=3,
        K_auxiliary=2,
        seed=0,
        sigma=1.0,
    )

    model = TransLasso(lambda_w=0.08, lambda_delta=0.08).fit(
        X_target,
        y_target,
        X_auxiliary_list=X_aux_list,
        y_auxiliary_list=y_aux_list,
    )

    j = int(np.flatnonzero(beta_true)[0])
    print("Example coordinate j =", j)
    print("beta_true[j] =", float(beta_true[j]))
    print("beta_hat[j]  =", float(model.coef_[j]))
    print("informative auxiliary idx:", model.get_informative_set(threshold=0.9))


if __name__ == "__main__":
    main()
