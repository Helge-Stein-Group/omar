import matplotlib.pyplot as plt
import numpy as np

from omars import OMARS

def inspect_fit(
        x: np.ndarray,
        y: np.ndarray,
        model: OMARS,
        title: str,
        fit: bool = True
) -> None:
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    assert x.shape[1] == 2
    assert isinstance(model, OMARS)
    assert isinstance(title, str)
    if fit:
        model._fit(x,y)

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    data_x = x[:, 0]
    data_y = x[:, 1]
    data_z = y
    ax.scatter(data_x, data_y, data_z, c='r', marker='o')

    x_func = np.linspace(min(data_x), max(data_x), 100)
    y_func = np.linspace(min(data_y), max(data_y), 100)
    x_func, y_func = np.meshgrid(x_func, y_func)
    z_func = model(np.c_[x_func.ravel(), y_func.ravel()])

    ax.plot_surface(x_func, y_func, z_func.reshape(x_func.shape), alpha=0.5,
                    cmap="coolwarm")

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
    from omars import OMARS
    from tests.utils import generate_data

    x, y, y_true = generate_data()

    model = OMARS()
    model.find_bases(x, y)

    print(model)
    inspect_fit(x, y, model, "Full model")
    for j, i in enumerate(model._active_base_indices()):
        print(model[i])
        if i !=0:
            print(model.coefficients[j-1])
            inspect_fit(x, y, model[i], str(i))
        else:
            inspect_fit(x, y, model[i], str(i), False)