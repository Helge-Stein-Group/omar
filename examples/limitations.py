# This file aims to demonstrate the limitations of the modelling capabilities of omar.
# Underscoring the expectations on the function to be modelled.


import numpy as np
import omar
import omar.tests.utils as utils
from jaxtyping import Float


def sigmoid(x: Float[np.ndarray, "N d"]) -> Float[np.ndarray, "N d"]:
    """
    Applies the sigmoid function element-wise.

    Args:
        x: Input array.

    Returns:
        Array with the sigmoid function applied element-wise
    """
    return 1 / (1 + np.exp(-x))


def evaluate_prediction(y_pred: Float[np.ndarray, "N"],
                        y_true: Float[np.ndarray, "N"],
                        y: Float[np.ndarray, "N d"]) -> float:
    """
    Evaluates the prediction using R^2 score.

    Args:
        y_pred: Predicted values.
        y_true: True values.
        y: Original values.

    Returns:
        R^2 score.
    """
    mse_0 = np.mean((np.mean(y) - y_true) ** 2)
    mse = np.mean((y_pred - y_true) ** 2)
    r2 = 1 - mse / mse_0
    return r2


def train_and_evaluate(x: np.ndarray, y: np.ndarray, y_true: np.ndarray, description: str, ref: bool = True):
    """
    Trains the omar model and evaluates its performance.

    Args:
        x: Input features.
        y: Target values.
        y_true: True values.
        description: Description of the data.
        ref: Whether to use the reference
    """
    print(f"Training omar model on {description} data...")
    model = omar.OMAR()
    model.find_bases(x, y)

    if ref:
        ref_model = utils.reference_model(x)
        ref_model.y_mean = np.mean(y)
        ref_model._fit(x, y)
        r2_ref = evaluate_prediction(ref_model(x), y_true, y)
        print(f"R2 reference: {r2_ref}")

    r2 = evaluate_prediction(model(x), y_true, y)
    print(f"R2: {r2}")
    print(model)


print(
    "Generating initial data with 100 samples and 2 dimensions, which has a purely linear relationship, dependent on all dimensions.")
x, y, y_true = utils.generate_data()
train_and_evaluate(x, y, y_true, "initial")

print(
    "Increasing the data dimensionality to 20 does barely change the model, as it manages to identify the relevant dimensions.")
x, y, y_true = utils.generate_data(100, 20)
train_and_evaluate(x, y, y_true, "higher dimensional")

print("However once we have purely nonlinear data, e.g. here a additive sigmoid function, the approach struggles.")
n_samples = 100
dim = 10
x = np.random.normal(size=(n_samples, dim))
l1 = x[:, 0] + x[:, 1] + x[:, 2] + x[:, 3] + x[:, 4]
l2 = x[:, 5] - x[:, 6] + x[:, 7] - x[:, 8] + x[:, 9]
y_true = sigmoid(l1) + sigmoid(l2)
y = y_true + 0.12 * np.random.normal(size=n_samples)
train_and_evaluate(x, y, y_true, "nonlinear", False)
