import numpy as np
from datetime import datetime

import regression


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


def data_generation_model(n_samples: int, dim: int) \
        -> tuple[np.ndarray, np.ndarray, np.ndarray, regression.Model]:
    x = np.random.normal(2, 1, size=(n_samples, dim))
    zero = np.zeros(n_samples)
    y_true = (np.maximum(zero, (x[:, 0] - 1)) +
              np.maximum(zero, (x[:, 0] - 1)) * np.maximum(0, (x[:, 1] - 0.8)))
    y = y_true + 0.12 * np.random.normal(size=n_samples)

    reference_model = regression.Model()
    x1 = x[np.argmin(np.abs(x[:,0] - 1)), 0]
    x08 = x[np.argmin(np.abs(x[:,1] - 0.8)), 1]
    reference_model.basis = [
        regression.Basis(),
        regression.Basis(v=[0], t=np.array([x1]), hinge=[True]),
        regression.Basis(v=[0, 1], t=np.array([x1, x08]), hinge=[True, True]),
    ]
    reference_model.fit(x, y)
    print("Coefficients: ", reference_model.coefficients)
    return x, y, y_true, reference_model


def evaluate_prediction(y_pred: np.ndarray, y_true: np.ndarray, y: np.ndarray) -> float:
    # Undoing the standardization
    #y_pred = y_pred * np.std(y) + np.mean(y)

    mse_0 = np.mean((np.mean(y) - y_true) ** 2)
    mse = np.mean((y_pred - y_true) ** 2)

    r2 = 1 - mse / mse_0

    return r2


def omars_test(x: np.ndarray, y: np.ndarray, y_true: np.ndarray, name: str) -> float:
    model = regression.fit(x, y, 10)
    y_pred = model(x)

    r2 = evaluate_prediction(y_pred, y_true, y)

    print(f"{name}: R2: {r2}")
    print(f"Number of basis functions: {len(model)}")
    return r2, model


def test_scenario1():
    n_samples = 100
    dim = 2

    x, y, y_true, reference_model = data_generation_model(n_samples, dim)
    r2, model = omars_test(x, y, y_true, "Scenario 1")
    print("LOF: ", model.gcv)

    r2_ref = evaluate_prediction(reference_model(x), y_true, y)
    print("Reference R2: ", r2_ref)
    print("Reference LOF: ", reference_model.gcv)
    assert r2 > 0.9
    print(model)


def test_scenario2():
    n_samples = 100
    dim = 20

    x, y, y_true, reference_model = data_generation_model(n_samples, dim)
    r2, model = omars_test(x, y, y_true, "Scenario 2")
    r2_ref = evaluate_prediction(reference_model(x), y_true, y)
    print(r2_ref)
    assert r2 > 0.9


def test_scenario3():
    n_samples = 100
    dim = 10

    x = np.random.normal(size=(n_samples, dim))
    l1 = x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3] + x[:, 4]
    l2 = x[:, 5] - x[:, 6] + x[:, 7] - x[:, 8] + x[:, 9]
    y = sigmoid(l1) + sigmoid(l2) + 0.12 * np.random.normal(size=n_samples)
    r2, model = omars_test(x, y, sigmoid(l1) + sigmoid(l2), "Scenario 3")

    assert r2 < 0.8


def test_speed():
    for n in range(2, 4):
        n_samples = 10 ** n
        dim = 4
        x, y, y_true, reference_model = data_generation_model(n_samples, dim)
        for i in range(6):
            regression.method = i
            start = datetime.now()
            omars_test(x, y, y_true, f"Speed test (Method: {i}): {n_samples}")
            end = datetime.now()
            print(f"Speed test (Method: {i}): {n_samples} - Time elapsed: {end - start}")
