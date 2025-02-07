import numpy as np

from jaxtyping import Float

from omar import OMAR

N_SAMPLES = 100
DIM = 2


def generate_data(n_samples: int = N_SAMPLES, dim: int = DIM) \
        -> tuple[
            Float[np.ndarray, "{n_samples} {dim}"],
            Float[np.ndarray, "{n_samples}"],
            Float[np.ndarray, "{n_samples}"]
        ]:
    x = np.random.normal(2, 1, size=(n_samples, dim))
    y_true = (x[:, 0] +
              np.maximum(0, (x[:, 0] - 1)) +
              np.maximum(0, (x[:, 0] - 1)) * x[:, 1] +
              np.maximum(0, (x[:, 0] - 1)) * np.maximum(0, (x[:, 1] - 0.8)))
    y = y_true + 0.12 * np.random.normal(size=n_samples)
    return x, y, y_true


def reference_model(x: Float[np.ndarray, "n_samples"]) -> OMAR:
    model = OMAR()

    x1 = x[np.argmin(np.abs(x[:, 0] - 1)), 0]
    x08 = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]

    model.nbases = 5
    model.mask[1, 1:3] = True
    model.mask[1:3, 3:5] = [True, True]
    model.truncated[1, 2:5] = True
    model.truncated[2, 4] = True
    model.cov[1, 1:5] = 0
    model.cov[2, 3:5] = 1
    model.root[1, 2:5] = x1
    model.root[2, 4] = x08

    model.coefficients = np.array([1, 1, 1, 1, 1])

    return model


def reference_data_matrix(x: Float[np.ndarray, "n_samples dim"]) \
        -> tuple[Float[np.ndarray, "{n_samples} 4"], Float[np.ndarray, "4"]]:
    x1 = x[np.argmin(np.abs(x[:, 0] - 1)), 0]
    x08 = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    ref_data_matrix = np.column_stack([
        x[:, 0],
        np.maximum(0, x[:, 0] - x1),
        np.maximum(0, x[:, 0] - x1) * x[:, 1],
        np.maximum(0, x[:, 0] - x1) * np.maximum(0, x[:, 1] - x08),
    ])

    ref_data_matrix_mean = ref_data_matrix.mean(axis=0)
    ref_data_matrix -= ref_data_matrix_mean

    return ref_data_matrix, ref_data_matrix_mean


def reference_covariance_matrix(ref_data_matrix: Float[np.ndarray, "n_samples 4"]) -> Float[np.ndarray, "4 4"]:
    ref_cov_matrix = ref_data_matrix.T @ ref_data_matrix + np.eye(ref_data_matrix.shape[1]) * 1e-8

    return ref_cov_matrix


def reference_rhs(y: Float[np.ndarray, "n_samples"], ref_data_matrix: Float[np.ndarray, "n_samples 4"]) -> Float[
    np.ndarray, "n_samples"]:
    ref_rhs = ref_data_matrix.T @ (y - y.mean())

    return ref_rhs
