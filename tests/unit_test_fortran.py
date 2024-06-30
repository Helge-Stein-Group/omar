import fortran.data_matrix
import numpy as np
import utils

print(fortran.data_matrix.__doc__)


def test_data_matrix() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(100, 2)

    fit_matrix = fortran.data_matrix.omars_data_matrix(x, 0, nbases, covariates, nodes, hinges, where)

    x1 = x[np.argmin(np.abs(x[:, 0] - 1)), 0]
    x08 = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    ref_data_matrix = np.column_stack([
        np.ones(x.shape[0], dtype=float),
        np.maximum(0, x[:, 0] - x1),
        np.maximum(0, x[:, 0] - x1) * np.maximum(0, x[:, 1] - x08),
    ])

    assert np.allclose(ref_data_matrix, fit_matrix)
