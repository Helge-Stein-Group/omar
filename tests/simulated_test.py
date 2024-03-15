import numpy as np

from omars.regression import fit
from omars.utils import sigmoid


def data_generation_model(n_samples: int, dim: int) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    x = np.random.normal(size=(n_samples, dim))
    zero = np.zeros(n_samples)
    y_true = (np.maximum(zero, (x[:, 0] - 1)) +
              np.maximum(zero, (x[:, 0] - 1)) * np.maximum(0, (x[:, 1] - 0.8)))
    y = y_true + 0.12 * np.random.normal(size=n_samples)
    return x, y, y_true


def evaluate_prediction(y_pred: np.ndarray, y_true: np.ndarray, y: np.ndarray) -> float:
    mse_0 = np.mean((np.mean(y) - y_true) ** 2)
    mse = np.mean((y_pred - y_true) ** 2)

    r2 = 1 - mse / mse_0

    return r2


def omars_test(x: np.ndarray, y: np.ndarray, y_true: np.ndarray, name: str) -> float:
    model = fit(x, y, 10)
    y_pred = model(x)

    r2 = evaluate_prediction(y_pred, y_true, y)

    print(f"{name}: R2: {r2}")
    print(f"Number of basis functions: {len(model)}")
    return r2


def test_scenario1():
    n_samples = 100
    dim = 2

    x, y, y_true = data_generation_model(n_samples, dim)
    r2 = omars_test(x, y, y_true, "Scenario 1")

    assert r2 > 0.9


def test_scenario2():
    n_samples = 100
    dim = 20

    x, y, y_true = data_generation_model(n_samples, dim)
    r2 = omars_test(x, y, y_true, "Scenario 2")

    assert r2 > 0.9


def test_scenario3():
    n_samples = 100
    dim = 10

    x = np.random.normal(size=(n_samples, dim))
    l1 = x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3] + x[:, 4]
    l2 = x[:, 5] - x[:, 6] + x[:, 7] - x[:, 8] + x[:, 9]
    y = sigmoid(l1) + sigmoid(l2) + 0.12 * np.random.normal(size=n_samples)
    r2 = omars_test(x, y, sigmoid(l1) + sigmoid(l2), "Scenario 3")

    assert r2 < 0.8
