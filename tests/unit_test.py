import numpy as np

import regression
import utils


def test_fit():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    model.fit(x, y)

    result = np.linalg.lstsq(model.fit_matrix, y, rcond=None)
    coefficients = result[0]

    assert np.allclose(coefficients[0], model.coefficients[0], 0.05, 0.1)
    assert np.allclose(coefficients[1], model.coefficients[1], 0.05, 0.05)
    assert np.allclose(coefficients[2], model.coefficients[2], 0.05, 0.05)


def update_case(model, x, y, func):
    tri = model.fit(x, y)
    u = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    for c in [0.5, 0.4, 0.3]:
        t = x[np.argmin(np.abs(x[:, 1] - c)), 1]
        model.basis[-1].t[-1] = t
        tri = func(x, y, tri, u, t, 1, model.fit_matrix[:, 1])
        u = t
    return model, tri


def test_update_fit_matrix():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_fit_matrix = model.fit_matrix.copy()

    def update_func(x, y, tri, u, t, v, selected_fit):
        model.update_initialisation(x, u, t, v, selected_fit)
        model.update_fit_matrix()
        return None

    model, tri = update_case(model, x, y, update_func)
    updated_fit_matrix = model.fit_matrix.copy()

    model.calculate_fit_matrix(x)
    full_fit_matrix = model.fit_matrix.copy()

    assert np.allclose(updated_fit_matrix, full_fit_matrix)


def test_update_covariance_matrix():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_covariance = model.covariance_matrix.copy()

    def update_func(x, y, tri, u, t, v, selected_fit):
        model.update_initialisation(x, u, t, v, selected_fit)
        model.update_fit_matrix()
        covariance_addition = model.update_covariance_matrix()
        return None

    model, tri = update_case(model, x, y, update_func)
    updated_covariance = model.covariance_matrix.copy()

    model.calculate_fit_matrix(x)
    model.calculate_covariance_matrix()
    full_covariance = model.covariance_matrix.copy()
    assert np.allclose(updated_covariance[:-1, :-1], full_covariance[:-1, :-1])
    assert np.allclose(updated_covariance[-1, :-1], full_covariance[-1, :-1])
    assert np.allclose(updated_covariance, full_covariance)


def test_update_right_hand_side():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_right_hand_side = model.right_hand_side.copy()

    def update_func(x, y, tri, u, t, v, selected_fit):
        model.update_initialisation(x, u, t, v, selected_fit)
        model.update_right_hand_side(y)
        return None

    model, tri = update_case(model, x, y, update_func)
    updated_right_hand_side = model.right_hand_side.copy()

    model.calculate_fit_matrix(x)
    model.calculate_right_hand_side(y)
    full_right_hand_side = model.right_hand_side.copy()

    assert np.allclose(updated_right_hand_side, full_right_hand_side)


def test_decompose():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_covariance = model.covariance_matrix.copy()

    u = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    t = x[np.argmin(np.abs(x[:, 1] - 0.5)), 1]
    model.basis[-1].t[-1] = t
    model.update_initialisation(x, u, t, 1, model.fit_matrix[:, 1])
    model.update_fit_matrix()
    covariance_addition = model.update_covariance_matrix()

    updated_covariance = model.covariance_matrix.copy()

    eigenvalues, eigenvectors = model.decompose_addition(covariance_addition)
    reconstructed_covariance = former_covariance + eigenvalues[0] * np.outer(
        eigenvectors[0], eigenvectors[0])
    reconstructed_covariance += eigenvalues[1] * np.outer(eigenvectors[1],
                                                          eigenvectors[1])

    assert np.allclose(reconstructed_covariance, updated_covariance)


def test_update_cholesky():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_cholesky = model.fit(x, y)

    def update_func(x, y, tri, u, t, v, selected_fit):
        model.update_initialisation(x, u, t, v, selected_fit)
        model.update_fit_matrix()
        covariance_addition = model.update_covariance_matrix()
        if covariance_addition.any():
            eigenvalues, eigenvectors = model.decompose_addition(covariance_addition)
            tri = regression.update_cholesky(tri, eigenvectors, eigenvalues)
        return tri

    model, updated_cholesky = update_case(model, x, y, update_func)

    full_cholesky = model.fit(x, y)
    assert np.allclose(np.tril(updated_cholesky), np.tril(full_cholesky))


def test_update_fit():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_tri = model.fit(x, y)
    former_coefficients = model.coefficients.copy()

    model, updated_tri = update_case(model, x, y, model.update_fit)
    updated_rhs = model.right_hand_side.copy()
    updated_coefficients = model.coefficients.copy()
    updated_gcv = model.gcv

    full_tri = model.fit(x, y)
    full_rhs = model.right_hand_side.copy()
    full_coefficients = model.coefficients.copy()
    full_gcv = model.gcv

    assert np.allclose(np.tril(updated_tri), np.tril(full_tri))
    assert np.allclose(updated_rhs, full_rhs)
    assert np.allclose(updated_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(updated_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(updated_coefficients[2], full_coefficients[2], 0.05, 0.05)
    assert np.allclose(updated_gcv, full_gcv)
