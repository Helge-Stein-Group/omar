import numpy as np

from src.omars import omars
from src.utils import sigmoid


def data_generation_model1(n_samples: int, dim: int) -> (np.ndarray, np.ndarray):
    X = np.random.normal(size=(n_samples, dim))
    y_true = np.max([0, (X[:, 0] - 1)]) + np.max([0, (X[:, 0] - 1)]) * np.max([0, (X[:, 1] - 0.8)])
    y = y_true + 0.12 * np.random.normal(size=n_samples)
    return X, y, y_true


def evaluate_prediction(y_pred: np.ndarray, y_true: np.ndarray, y: np.ndarray) -> float:
    mse_0 = np.mean((np.mean(y) - y_true) ** 2)
    mse = np.mean((y_pred - y_true) ** 2)

    r2 = 1 - mse / mse_0

    print(f"R2: {r2}")

    return r2


def test_scenario1():
    n_samples = 100
    dim = 2

    X, y, y_true = data_generation_model1(n_samples, dim)

    y_pred = omars(X, y)

    r2 = evaluate_prediction(y_pred, y_true, y)

    assert r2 > 0.9


def test_scenario2():
    n_samples = 100
    dim = 20

    X, y, y_true = data_generation_model1(n_samples, dim)

    y_pred = omars(X, y)

    r2 = evaluate_prediction(y_pred, y_true, y)

    assert r2 > 0.9


def test_scenario3():
    n_samples = 100
    dim = 10

    X = np.random.normal(size=(n_samples, dim))
    l1 = X[:, 0] + X[:, 1] + X[:, 2] + X[:, 3] + X[:, 4]
    l2 = X[:, 5] - X[:, 6] + X[:, 7] - X[:, 8] + X[:, 9]
    y = sigmoid(l1) + sigmoid(l2) + 0.12 * np.random.normal(size=n_samples)

    y_pred = omars(X, y)

    r2 = evaluate_prediction(y_pred, y, y)

    assert r2 > 0.7
