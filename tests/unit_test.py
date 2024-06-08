from copy import deepcopy

import numpy as np

import regression
import utils


def test_data_matrix():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    x1 = x[np.argmin(np.abs(x[:, 0] - 1)), 0]
    x08 = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    ref_data_matrix = np.column_stack([
        np.ones(x.shape[0], dtype=float),
        np.maximum(0, x[:, 0] - x1),
        np.maximum(0, x[:, 0] - x1) * np.maximum(0, x[:, 1] - x08),
    ])

    assert np.allclose(ref_data_matrix, model.fit_matrix)


def test_fit():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    result = np.linalg.lstsq(model.fit_matrix, y, rcond=None)
    coefficients = result[0]

    assert np.allclose(coefficients[0], model.coefficients[0], 0.05, 0.1)
    assert np.allclose(coefficients[1], model.coefficients[1], 0.05, 0.05)
    assert np.allclose(coefficients[2], model.coefficients[2], 0.05, 0.05)


def update_case(model, x, y, func):
    chol = model.fit(x, y)
    old_node = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    for c in [0.5, 0.4, 0.3]:
        new_node = x[np.argmin(np.abs(x[:, 1] - c)), 1]
        nodes = model.nodes[:, model.nbases - 1]
        nodes[np.sum(model.where[:, model.nbases - 1])] = new_node
        model.update_basis(
            model.covariates[:, model.nbases - 1],
            nodes,
            model.hinges[:, model.nbases - 1],
            model.where[:, model.nbases - 1],
        )
        chol = func(x, y, chol, old_node, 1)
        old_node = new_node
    return model, chol


def extend_case(model, x, y, func):
    for c in [0.5, 0.4, 0.3]:
        new_node = x[np.argmin(np.abs(x[:, 1] - c)), 1]

        additional_covariates = np.tile(model.covariates[:, model.nbases - 1], (2, 1)).T
        additional_nodes = np.tile(model.nodes[:, model.nbases - 1], (2, 1)).T
        additional_hinges = np.tile(model.hinges[:, model.nbases - 1], (2, 1)).T
        additional_where = np.tile(model.where[:, model.nbases - 1], (2, 1)).T

        parent_depth = np.sum(model.where[:, model.nbases - 1])

        additional_covariates[parent_depth, 0] = 1
        additional_nodes[parent_depth, 0] = 0.0
        additional_hinges[parent_depth, 0] = False
        additional_where[parent_depth, 0] = True

        additional_covariates[parent_depth, 1] = 1
        additional_nodes[parent_depth, 1] = new_node
        additional_hinges[parent_depth, 1] = True
        additional_where[parent_depth, 1] = True
        chol = func(x, y, 2)

    return model, chol


def shrink_case(model, x, y, func):
    for i in range(3):
        removal_slice = slice(model.nbases - 1, model.nbases)
        model.remove_basis(removal_slice)
        chol = func(x, y, i + 1)
    return model, chol


def test_update_fit_matrix():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_fit_matrix = model.fit_matrix.copy()

    def update_func(x, y, chol, old_node, parent_idx):
        model.update_init(x, old_node, parent_idx)
        model.update_fit_matrix()
        return None

    model, chol = update_case(model, x, y, update_func)
    updated_fit_matrix = model.fit_matrix.copy()

    model.calculate_fit_matrix(x)
    full_fit_matrix = model.fit_matrix.copy()

    assert np.allclose(updated_fit_matrix, full_fit_matrix)


def test_extend_fit_matrix():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_fit_matrix = model.fit_matrix.copy()

    def extend_func(x, y, i):
        model.extend_fit_matrix(x, i)
        return None

    model, tri = extend_case(model, x, y, extend_func)
    extended_fit_matrix = model.fit_matrix.copy()

    model.calculate_fit_matrix(x)
    full_fit_matrix = model.fit_matrix.copy()

    assert np.allclose(extended_fit_matrix, full_fit_matrix)


def test_update_covariance_matrix():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_covariance = model.covariance_matrix.copy()

    def update_func(x, y, tri, u, t, v, selected_fit):
        model.update_init(x, u, t, v, selected_fit)
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


def test_extend_covariance_matrix():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_covariance = model.covariance_matrix.copy()

    def extend_func(x, y, i):
        model.extend_fit_matrix(x, i)
        model.extend_covariance_matrix(i)
        return None

    model, tri = extend_case(model, x, y, extend_func)
    extended_covariance = model.covariance_matrix.copy()
    extended_fixed_mean = model.fixed_mean.copy()
    extended_candidate_mean = model.candidate_mean.copy()

    model.calculate_fit_matrix(x)
    model.calculate_covariance_matrix()
    full_covariance = model.covariance_matrix.copy()
    full_fixed_mean = model.fixed_mean.copy()
    full_candidate_mean = model.candidate_mean.copy()

    assert np.allclose(extended_covariance, full_covariance)
    assert np.allclose(extended_fixed_mean, full_fixed_mean)
    assert np.allclose(extended_candidate_mean, full_candidate_mean)


def test_update_right_hand_side():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_right_hand_side = model.right_hand_side.copy()

    def update_func(x, y, tri, u, t, v, selected_fit):
        model.update_init(x, u, t, v, selected_fit)
        model.update_right_hand_side(y)
        return None

    model, tri = update_case(model, x, y, update_func)
    updated_right_hand_side = model.right_hand_side.copy()

    model.calculate_fit_matrix(x)
    model.calculate_right_hand_side(y)
    full_right_hand_side = model.right_hand_side.copy()

    assert np.allclose(updated_right_hand_side, full_right_hand_side)


def test_extend_right_hand_side():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_right_hand_side = model.right_hand_side.copy()

    def extend_func(x, y, i):
        model.extend_fit_matrix(x, i)
        model.extend_covariance_matrix(i)
        model.extend_right_hand_side(y, i)
        return None

    model, tri = extend_case(model, x, y, extend_func)
    extended_right_hand_side = model.right_hand_side.copy()

    model.calculate_fit_matrix(x)
    model.calculate_covariance_matrix()
    model.calculate_right_hand_side(y)
    full_right_hand_side = model.right_hand_side.copy()

    assert np.allclose(extended_right_hand_side, full_right_hand_side)


def test_decompose():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_covariance = model.covariance_matrix.copy()

    u = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    t = x[np.argmin(np.abs(x[:, 1] - 0.5)), 1]
    model.basis[-1].t[-1] = t
    model.update_init(x, u, t, 1, model.fit_matrix[:, 1])
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
        model.update_init(x, u, t, v, selected_fit)
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
    updated_gcv = model.lof

    full_tri = model.fit(x, y)
    full_rhs = model.right_hand_side.copy()
    full_coefficients = model.coefficients.copy()
    full_gcv = model.lof

    assert np.allclose(np.tril(updated_tri), np.tril(full_tri))
    assert np.allclose(updated_rhs, full_rhs)
    assert np.allclose(updated_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(updated_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(updated_coefficients[2], full_coefficients[2], 0.05, 0.05)
    assert np.allclose(updated_gcv, full_gcv)


def test_extend_fit():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_tri = model.fit(x, y)
    former_coefficients = model.coefficients.copy()

    model, tri = extend_case(model, x, y, model.extend_fit)
    updated_rhs = model.right_hand_side.copy()
    updated_coefficients = model.coefficients.copy()
    updated_gcv = model.lof

    full_tri = model.fit(x, y)
    full_rhs = model.right_hand_side.copy()
    full_coefficients = model.coefficients.copy()
    full_gcv = model.lof

    assert np.allclose(np.tril(tri), np.tril(full_tri))
    assert np.allclose(updated_rhs, full_rhs)
    assert np.allclose(updated_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(updated_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(updated_coefficients[2], full_coefficients[2], 0.05, 0.05)
    assert np.allclose(updated_gcv, full_gcv)


def test_shrink_fit():
    x, y, y_true, model = utils.data_generation_model(100, 2)
    for i in range(7):
        basis_add = deepcopy(model.basis[0])
        basis_add.t = np.array(np.random.choice(x[:, 0], 1))
        basis_add.v = np.array([0])
        basis_add.hinge = np.array(np.random.choice([True, False], 1))
        model.add([basis_add])

    former_tri = model.fit(x, y)
    former_coefficients = model.coefficients.copy()

    model, tri = shrink_case(model, x, y, model.shrink_fit)
    updated_rhs = model.right_hand_side.copy()
    updated_coefficients = model.coefficients.copy()
    updated_gcv = model.lof

    full_tri = model.fit(x, y)
    full_rhs = model.right_hand_side.copy()
    full_coefficients = model.coefficients.copy()
    full_gcv = model.lof

    assert np.allclose(np.tril(tri), np.tril(full_tri))
    assert np.allclose(updated_rhs, full_rhs)
    assert np.allclose(updated_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(updated_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(updated_coefficients[2], full_coefficients[2], 0.05, 0.05)
    assert np.allclose(updated_gcv, full_gcv)
