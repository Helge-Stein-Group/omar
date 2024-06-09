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

        model.add_basis(additional_covariates,
                        additional_nodes,
                        additional_hinges,
                        additional_where)
        chol = func(x, y, 2)

    return model, chol


def shrink_case(model, x, y, func):
    for i in range(3):
        removal_slice = slice(model.nbases - 1, model.nbases)
        model.remove_basis(removal_slice)
        chol = func(x, y, removal_slice)
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

    model, chol = extend_case(model, x, y, extend_func)
    extended_fit_matrix = model.fit_matrix.copy()

    model.calculate_fit_matrix(x)
    full_fit_matrix = model.fit_matrix.copy()

    assert np.allclose(extended_fit_matrix, full_fit_matrix)


def test_update_covariance_matrix():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_covariance = model.covariance_matrix.copy()

    def update_func(x, y, chol, old_node, parent_idx):
        model.update_init(x, old_node, parent_idx)
        model.update_fit_matrix()
        covariance_addition = model.update_covariance_matrix()
        return None

    model, chol = update_case(model, x, y, update_func)
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

    model, chol = extend_case(model, x, y, extend_func)
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

    def update_func(x, y, chol, old_node, parent_idx):
        model.update_init(x, old_node, parent_idx)
        model.update_right_hand_side(y)
        return None

    model, chol = update_case(model, x, y, update_func)
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

    model, chol = extend_case(model, x, y, extend_func)
    extended_right_hand_side = model.right_hand_side.copy()

    model.calculate_fit_matrix(x)
    model.calculate_covariance_matrix()
    model.calculate_right_hand_side(y)
    full_right_hand_side = model.right_hand_side.copy()

    assert np.allclose(extended_right_hand_side, full_right_hand_side)


def test_decompose():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_covariance = model.covariance_matrix.copy()

    old_node = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    new_node = x[np.argmin(np.abs(x[:, 1] - 0.5)), 1]
    model.nodes[2, 2] = new_node
    model.update_init(x, old_node, 1)
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

    def update_func(x, y, chol, old_node, parent_idx):
        model.update_init(x, old_node, parent_idx)
        model.update_fit_matrix()
        covariance_addition = model.update_covariance_matrix()
        if covariance_addition.any():
            eigenvalues, eigenvectors = model.decompose_addition(covariance_addition)
            chol = regression.update_cholesky(chol, eigenvectors, eigenvalues)
        return chol

    model, updated_cholesky = update_case(model, x, y, update_func)

    full_cholesky = model.fit(x, y)
    assert np.allclose(np.tril(updated_cholesky), np.tril(full_cholesky))


def test_update_fit():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_chol = model.fit(x, y)
    former_coefficients = model.coefficients.copy()

    model, updated_chol = update_case(model, x, y, model.update_fit)
    updated_rhs = model.right_hand_side.copy()
    updated_coefficients = model.coefficients.copy()
    updated_lof = model.lof

    full_chol = model.fit(x, y)
    full_rhs = model.right_hand_side.copy()
    full_coefficients = model.coefficients.copy()
    full_lof = model.lof

    assert np.allclose(np.tril(updated_chol), np.tril(full_chol))
    assert np.allclose(updated_rhs, full_rhs)
    assert np.allclose(updated_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(updated_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(updated_coefficients[2], full_coefficients[2], 0.05, 0.05)
    assert np.allclose(updated_lof, full_lof, 0.01)


def test_extend_fit():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_tri = model.fit(x, y)
    former_coefficients = model.coefficients.copy()

    model, chol = extend_case(model, x, y, model.extend_fit)
    extended_rhs = model.right_hand_side.copy()
    extended_coefficients = model.coefficients.copy()
    extended_lof = model.lof

    full_chol = model.fit(x, y)
    full_rhs = model.right_hand_side.copy()
    full_coefficients = model.coefficients.copy()
    full_lof = model.lof

    assert np.allclose(np.tril(chol), np.tril(full_chol))
    assert np.allclose(extended_rhs, full_rhs)
    assert np.allclose(extended_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(extended_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(extended_coefficients[2], full_coefficients[2], 0.05, 0.05)
    assert np.allclose(extended_lof, full_lof, 0.01)


def test_shrink_fit():
    x, y, y_true, model = utils.data_generation_model(100, 2)

    additional_covariates = np.tile(model.covariates[:, 1], (7, 1)).T
    additional_nodes = np.tile(model.nodes[:, 1], (7, 1)).T
    additional_hinges = np.tile(model.hinges[:, 1], (7, 1)).T
    additional_where = np.tile(model.where[:, 1], (7, 1)).T

    additional_nodes[2, :] = np.random.choice(x[:, 0], 7)

    model.add_basis(additional_covariates,
                    additional_nodes,
                    additional_hinges,
                    additional_where)

    former_chol = model.fit(x, y)
    former_coefficients = model.coefficients.copy()

    model, chol = shrink_case(model, x, y, model.shrink_fit)
    shrunk_rhs = model.right_hand_side.copy()
    shrunk_coefficients = model.coefficients.copy()
    shrunk_lof = model.lof

    full_tri = model.fit(x, y)
    full_rhs = model.right_hand_side.copy()
    full_coefficients = model.coefficients.copy()
    full_gcv = model.lof

    assert np.allclose(np.tril(chol), np.tril(full_tri))
    assert np.allclose(shrunk_rhs, full_rhs)
    assert np.allclose(shrunk_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(shrunk_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(shrunk_coefficients[2], full_coefficients[2], 0.05, 0.05)
    assert np.allclose(shrunk_lof, full_gcv, 0.01)


def test_forward_pass():
    x, y, y_true, ref_model = utils.data_generation_model(100, 2)

    model = regression.OMARS()
    model.forward_pass(x, y_true)

    expected_node_1 = x[np.argmin(np.abs(x[:, 0] - 1)), 0]
    expected_node_2 = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]

    potential_bases_1 = np.where(
        np.sum(np.isin(model.nodes, expected_node_1), axis=0) > 0)[0]
    potential_bases_2 = np.where(
        np.sum(np.isin(model.nodes, expected_node_2), axis=0) > 0)[0]

    potential_bases_12 = np.intersect1d(potential_bases_1, potential_bases_2)

    match1 = False
    match2 = False

    for pidx in potential_bases_1:
        if (np.allclose(model.nodes[:, pidx], ref_model.nodes[:, 1]) and
                np.allclose(model.covariates[:, pidx], ref_model.covariates[:, 1]) and
                np.allclose(model.hinges[:, pidx], ref_model.hinges[:, 1]) and
                np.allclose(model.where[:, pidx], ref_model.where[:, 1])):
            match1 = True

    for pidx in potential_bases_12:
        if (np.allclose(model.nodes[:, pidx], ref_model.nodes[:, 2]) and
                np.allclose(model.covariates[:, pidx], ref_model.covariates[:, 2]) and
                np.allclose(model.hinges[:, pidx], ref_model.hinges[:, 2]) and
                np.allclose(model.where[:, pidx], ref_model.where[:, 2])):
            match2 = True

    print("Expected node 1: ", expected_node_1)
    print("Expected node 2: ", expected_node_2)
    print(model)
    assert match1
    assert match2


def test_backward_pass():
    x, y, y_true, ref_model = utils.data_generation_model(100, 2)

    test_model = deepcopy(ref_model)
    test_model.backward_pass(x, y_true)

    print(test_model)
    assert ref_model == test_model
