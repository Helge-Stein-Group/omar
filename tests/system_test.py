import numpy as np

import regression
import utils


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def evaluate_prediction(y_pred: np.ndarray, y_true: np.ndarray, y: np.ndarray) -> float:
    mse_0 = np.mean((np.mean(y) - y_true) ** 2)
    mse = np.mean((y_pred - y_true) ** 2)

    r2 = 1 - mse / mse_0

    return r2


def omars_test(x: np.ndarray, y: np.ndarray, y_true: np.ndarray) -> tuple[
    float, regression.OMARS]:

    model = regression.OMARS()
    model.find_bases(x, y)
    y_pred = model(x)

    r2 = evaluate_prediction(y_pred, y_true, y)

    return r2, model


def test_scenario1() -> None:
    n_samples = 100
    dim = 2

    x, y, y_true, reference_model = utils.data_generation_model(n_samples, dim)
    r2, model = omars_test(x, y, y_true)

    r2_ref = evaluate_prediction(reference_model(x), y_true, y)
    print("R2 reference: ", r2_ref)
    print("R2: ", r2)
    print(model)
    assert r2 > 0.9


def test_scenario2() -> None:
    n_samples = 100
    dim = 20

    x, y, y_true, reference_model = utils.data_generation_model(n_samples, dim)
    r2, model = omars_test(x, y, y_true)

    r2_ref = evaluate_prediction(reference_model(x), y_true, y)
    print("R2 reference: ", r2_ref)
    print("R2: ", r2)
    print(model)
    assert r2 > 0.9


def test_scenario3() -> None:
    n_samples = 100
    dim = 10

    x = np.random.normal(size=(n_samples, dim))
    l1 = x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3] + x[:, 4]
    l2 = x[:, 5] - x[:, 6] + x[:, 7] - x[:, 8] + x[:, 9]
    y = sigmoid(l1) + sigmoid(l2) + 0.12 * np.random.normal(size=n_samples)
    r2, model = omars_test(x, y, sigmoid(l1) + sigmoid(l2))

    print(model)
    assert r2 < 0.8
