import numpy as np

import regression


def generate_data(n_samples: int, dim: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.random.normal(2, 1, size=(n_samples, dim))
    zero = np.zeros(n_samples)
    y_true = (np.maximum(zero, (x[:, 0] - 1)) +
              np.maximum(zero, (x[:, 0] - 1)) * np.maximum(0, (x[:, 1] - 0.8)))
    y = y_true + 0.12 * np.random.normal(size=n_samples)
    return x, y, y_true


def data_generation_model(n_samples: int, dim: int) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, regression.OMARS]:
    x, y, y_true = generate_data(n_samples, dim)

    reference_model = regression.OMARS()
    x1 = x[np.argmin(np.abs(x[:, 0] - 1)), 0]
    x08 = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    reference_model.nbases = 3
    reference_model.covariates[1, 1] = 0
    reference_model.covariates[1:3, 2] = [0, 1]
    reference_model.nodes[1, 1] = x1
    reference_model.nodes[1:3, 2] = [x1, x08]
    reference_model.hinges[1, 1] = True
    reference_model.hinges[1:3, 2] = [True, True]
    reference_model.where[1, 1] = True
    reference_model.where[1:3, 2] = [True, True]
    reference_model.fit(x, y)
    return x, y, y_true, reference_model
