import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from omars import BasisFunctionDescription, basis_function

mpl.use("Qt5Agg")


def inspect_fit_3D(
        X: np.ndarray,
        y: np.ndarray,
        model_functions: list[BasisFunctionDescription],
        coefficients: np.ndarray,
        title: str,
) -> None:
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert X.shape[1] == 2
    assert len(model_functions) == coefficients.size
    assert isinstance(title, str)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    data_x = X[:, 0]
    data_y = X[:, 1]
    data_z = y
    ax.scatter(data_x, data_y, data_z, c='r', marker='o')

    x_func = np.linspace(min(data_x), max(data_x), 100)
    y_func = np.linspace(min(data_y), max(data_y), 100)
    x_func, y_func = np.meshgrid(x_func, y_func)
    z_func = predict(np.c_[x_func.ravel(), y_func.ravel()], coefficients, model_functions)

    ax.plot_surface(x_func, y_func, z_func.reshape(x_func.shape), alpha=0.5, cmap="coolwarm")

    ax.invert_zaxis()

    eps = 0.1
    ax.set_ylim(min(data_y) - eps, max(data_y) + eps)
    ax.set_xlim(min(data_x) - eps, max(data_x) + eps)
    ax.set_zlim(min(data_z) - eps, max(data_z) + eps)

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    ax.set_title(title)

    plt.show()


if __name__ == "__main__":
    n_samples = 100
    ndim = 2

    X = np.random.normal(size=(n_samples, ndim))
    zero = np.zeros(n_samples)
    y_true = np.maximum(zero, (X[:, 0] - 1)) + np.maximum(zero, (X[:, 0] - 1)) * np.maximum(0, (X[:, 1] - 0.8))
    y = y_true + 0 * np.random.normal(size=n_samples)

    from omars import omars, predict

    coefficients, model_functions = omars(X, y, 10)
    y_pred = predict(X, coefficients, model_functions)
    print(model_functions)
    inspect_fit_3D(X, y, model_functions, coefficients, "Full model")
    for i in range(len(model_functions)):
        print(model_functions[i])
        print(coefficients[i])
        inspect_fit_3D(X, y_pred, [model_functions[i]], coefficients[i:i+1], str(i))
    inspect_fit_3D(X, y, model_functions, coefficients, "Full model")
