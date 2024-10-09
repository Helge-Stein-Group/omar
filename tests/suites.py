import numpy as np
from numba.core.typing.builtins import Print


def suite_data_matrix(x, basis_mean, fit_matrix) -> None:
    x1 = x[np.argmin(np.abs(x[:, 0] - 1)), 0]
    x08 = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    ref_data_matrix = np.column_stack([
        np.maximum(0, x[:, 0] - x1),
        np.maximum(0, x[:, 0] - x1) * np.maximum(0, x[:, 1] - x08),
    ])

    ref_basis_mean = ref_data_matrix.mean(axis=0)
    ref_data_matrix -= ref_basis_mean

    assert np.allclose(ref_basis_mean, basis_mean)
    assert np.allclose(ref_data_matrix, fit_matrix)

def suite_fit(coefficients, fit_matrix, y, y_true, y_mean) -> None:
    ref_coefficients = np.linalg.lstsq(fit_matrix, y, rcond=None)[0]

    y_pred_model = fit_matrix @ coefficients + y_mean
    mse_model = np.mean((y_pred_model - y_true) ** 2)
    y_pred_np = fit_matrix @ ref_coefficients + y_mean
    mse_np = np.mean((y_pred_np - y_true) ** 2)

    assert np.allclose(mse_model, mse_np, 0.01)