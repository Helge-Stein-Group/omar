from copy import deepcopy

import numpy as np

import regression
import utils


def test_data_matrix() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)

    x1 = model.nodes[1, 1]
    x08 = model.nodes[2, 2]
    ref_data_matrix = np.column_stack([
        np.maximum(0, x[:, 0] - x1),
        np.maximum(0, x[:, 0] - x1) * np.maximum(0, x[:, 1] - x08),
    ])
    ref_basis_mean = ref_data_matrix.mean(axis=0)
    ref_data_matrix -= ref_basis_mean

    assert np.allclose(ref_basis_mean, model.basis_mean)
    assert np.allclose(ref_data_matrix, model.fit_matrix)


def test_fit() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)
    model._fit(x, y)

    coefficients = np.linalg.lstsq(model.fit_matrix, y, rcond=None)[0]

    y_pred_model = model(x)
    mse_model = np.mean((y_pred_model - y_true) ** 2)
    y_pred_np = model.fit_matrix @ coefficients + model.y_mean
    mse_np = np.mean((y_pred_np - y_true) ** 2)

    assert np.allclose(mse_model, mse_np, 0.01)


def update_case(model: regression.OMARS, x: np.ndarray, y: np.ndarray,
                func: callable) -> tuple[regression.OMARS, np.ndarray]:
    chol = model._fit(x, y)
    assert model.covariates[
               np.sum(model.where[:, model.nbases - 1]), model.nbases - 1] == 1
    old_node = model.nodes[np.sum(model.where[:, model.nbases - 1]), model.nbases - 1]
    for new_node in sorted([value for value in x[:, 1] if value < old_node],
                           reverse=True)[:3]:
        nodes = model.nodes[:, model.nbases - 1]
        nodes[np.sum(model.where[:, model.nbases - 1])] = new_node
        model._update_basis(
            model.covariates[:, model.nbases - 1],
            nodes,
            model.hinges[:, model.nbases - 1],
            model.where[:, model.nbases - 1],
        )
        chol = func(x, y, chol, old_node, model.nbases - 2)
        old_node = new_node
    return model, chol


def extend_case(model: regression.OMARS, x: np.ndarray, y: np.ndarray,
                func: callable) -> tuple[
    regression.OMARS, np.ndarray]:
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

        model._add_basis(additional_covariates,
                         additional_nodes,
                         additional_hinges,
                         additional_where)
        chol = func(x, y, 2)

    return model, chol


def shrink_case(model: regression.OMARS, x: np.ndarray, y: np.ndarray,
                func: callable) -> tuple[
    regression.OMARS, np.ndarray]:
    for i in range(3):
        model._remove_basis(model.nbases - 1)
        chol = func(x, y, model.nbases)
    return model, chol


def test_update_fit_matrix() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_fit_matrix = model.fit_matrix.copy()

    def update_func(x, y, chol, old_node, parent_idx):
        model._update_init(x, old_node, parent_idx)
        model._update_fit_matrix()
        return None

    model, chol = update_case(model, x, y, update_func)
    updated_fit_matrix = model.fit_matrix.copy()

    model._calculate_fit_matrix(x)
    full_fit_matrix = model.fit_matrix.copy()

    assert np.allclose(updated_fit_matrix, full_fit_matrix)


def test_extend_fit_matrix() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_fit_matrix = model.fit_matrix.copy()

    def extend_func(x, y, i):
        model._extend_fit_matrix(x, i)
        return None

    model, chol = extend_case(model, x, y, extend_func)
    extended_fit_matrix = model.fit_matrix.copy()

    model._calculate_fit_matrix(x)
    full_fit_matrix = model.fit_matrix.copy()

    assert np.allclose(extended_fit_matrix, full_fit_matrix)


def test_update_covariance_matrix() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_covariance = model.covariance_matrix.copy()

    def update_func(x, y, chol, old_node, parent_idx):
        model._update_init(x, old_node, parent_idx)
        model._update_fit_matrix()
        covariance_addition = model._update_covariance_matrix()
        return None

    model, chol = update_case(model, x, y, update_func)
    updated_covariance = model.covariance_matrix.copy()

    model._calculate_fit_matrix(x)
    model._calculate_covariance_matrix()
    full_covariance = model.covariance_matrix.copy()
    assert np.allclose(updated_covariance[:-1, :-1], full_covariance[:-1, :-1])
    assert np.allclose(updated_covariance[-1, :-1], full_covariance[-1, :-1])
    assert np.allclose(updated_covariance, full_covariance)


def test_extend_covariance_matrix() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_covariance = model.covariance_matrix.copy()

    def extend_func(x, y, i):
        model._extend_fit_matrix(x, i)
        model._extend_covariance_matrix(i)
        return None

    model, chol = extend_case(model, x, y, extend_func)
    extended_covariance = model.covariance_matrix.copy()

    model._calculate_fit_matrix(x)
    model._calculate_covariance_matrix()
    full_covariance = model.covariance_matrix.copy()

    assert np.allclose(extended_covariance, full_covariance)


def test_update_right_hand_side() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_right_hand_side = model.right_hand_side.copy()

    def update_func(x, y, chol, old_node, parent_idx):
        model._update_init(x, old_node, parent_idx)
        model._update_fit_matrix()
        model._update_right_hand_side(y)
        return None

    model, chol = update_case(model, x, y, update_func)
    updated_right_hand_side = model.right_hand_side.copy()

    model._calculate_fit_matrix(x)
    model._calculate_right_hand_side(y)
    full_right_hand_side = model.right_hand_side.copy()

    assert np.allclose(updated_right_hand_side, full_right_hand_side)


def test_extend_right_hand_side() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_right_hand_side = model.right_hand_side.copy()

    def extend_func(x, y, i):
        model._extend_fit_matrix(x, i)
        model._extend_covariance_matrix(i)
        model._extend_right_hand_side(y, i)
        return None

    model, chol = extend_case(model, x, y, extend_func)
    extended_right_hand_side = model.right_hand_side.copy()

    model._calculate_fit_matrix(x)
    model._calculate_covariance_matrix()
    model._calculate_right_hand_side(y)
    full_right_hand_side = model.right_hand_side.copy()

    assert np.allclose(extended_right_hand_side, full_right_hand_side)


def test_decompose() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_covariance = model.covariance_matrix.copy()

    old_node = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    new_node = x[np.argmin(np.abs(x[:, 1] - 0.5)), 1]
    model.nodes[2, 2] = new_node
    model._update_init(x, old_node, 1)
    model._update_fit_matrix()
    covariance_addition = model._update_covariance_matrix()

    updated_covariance = model.covariance_matrix.copy()

    eigenvalues, eigenvectors = model._decompose_addition(covariance_addition)
    reconstructed_covariance = former_covariance + eigenvalues[0] * np.outer(
        eigenvectors[0], eigenvectors[0])
    reconstructed_covariance += eigenvalues[1] * np.outer(eigenvectors[1],
                                                          eigenvectors[1])

    assert np.allclose(reconstructed_covariance, updated_covariance)


def test_update_cholesky() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_cholesky = model._fit(x, y)

    def update_func(x, y, chol, old_node, parent_idx):
        model._update_init(x, old_node, parent_idx)
        model._update_fit_matrix()
        covariance_addition = model._update_covariance_matrix()
        if covariance_addition.any():
            eigenvalues, eigenvectors = model._decompose_addition(covariance_addition)
            chol = regression.update_cholesky(chol, eigenvectors, eigenvalues)
        return chol

    model, updated_cholesky = update_case(model, x, y, update_func)

    full_cholesky = model._fit(x, y)
    assert np.allclose(np.tril(updated_cholesky), np.tril(full_cholesky))


def test_update_fit() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_chol = model._fit(x, y)
    former_coefficients = model.coefficients.copy()

    model, updated_chol = update_case(model, x, y, model._update_fit)
    updated_rhs = model.right_hand_side.copy()
    updated_coefficients = model.coefficients.copy()
    updated_lof = model.lof

    full_chol = model._fit(x, y)
    full_rhs = model.right_hand_side.copy()
    full_coefficients = model.coefficients.copy()
    full_lof = model.lof

    assert np.allclose(np.tril(updated_chol), np.tril(full_chol))
    assert np.allclose(updated_rhs, full_rhs)
    assert np.allclose(updated_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(updated_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(updated_lof, full_lof, 0.01)


def test_extend_fit() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)

    former_tri = model._fit(x, y)
    former_coefficients = model.coefficients.copy()

    model, chol = extend_case(model, x, y, model._extend_fit)
    extended_rhs = model.right_hand_side.copy()
    extended_coefficients = model.coefficients.copy()
    extended_lof = model.lof

    full_chol = model._fit(x, y)
    full_rhs = model.right_hand_side.copy()
    full_coefficients = model.coefficients.copy()
    full_lof = model.lof

    assert np.allclose(np.tril(chol), np.tril(full_chol))
    assert np.allclose(extended_rhs, full_rhs)
    assert np.allclose(extended_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(extended_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(extended_coefficients[2], full_coefficients[2], 0.05, 0.05)
    assert np.allclose(extended_lof, full_lof, 0.01)


def test_shrink_fit() -> None:
    x, y, y_true, model = utils.data_generation_model(100, 2)

    additional_covariates = np.tile(model.covariates[:, 1], (7, 1)).T
    additional_nodes = np.tile(model.nodes[:, 1], (7, 1)).T
    additional_hinges = np.tile(model.hinges[:, 1], (7, 1)).T
    additional_where = np.tile(model.where[:, 1], (7, 1)).T

    additional_nodes[2, :] = np.random.choice(x[:, 0], 7)

    model._add_basis(additional_covariates,
                     additional_nodes,
                     additional_hinges,
                     additional_where)

    former_chol = model._fit(x, y)
    former_coefficients = model.coefficients.copy()

    model, chol = shrink_case(model, x, y, model._shrink_fit)
    shrunk_rhs = model.right_hand_side.copy()
    shrunk_coefficients = model.coefficients.copy()
    shrunk_lof = model.lof

    full_tri = model._fit(x, y)
    full_rhs = model.right_hand_side.copy()
    full_coefficients = model.coefficients.copy()
    full_gcv = model.lof

    assert np.allclose(np.tril(chol), np.tril(full_tri))
    assert np.allclose(shrunk_rhs, full_rhs)
    assert np.allclose(shrunk_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(shrunk_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(shrunk_coefficients[2], full_coefficients[2], 0.05, 0.05)
    assert np.allclose(shrunk_lof, full_gcv, 0.01)


def test_expand_bases() -> None:
    x, y, y_true, ref_model = utils.data_generation_model(100, 2)

    model = regression.OMARS()
    model._expand_bases(x, y)

    expected_node_1 = ref_model.nodes[1, 1]
    expected_node_2 = ref_model.nodes[2, 2]

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


def test_prune_bases() -> None:
    x, y, y_true, ref_model = utils.data_generation_model(100, 2)

    test_model = deepcopy(ref_model)
    test_model._prune_bases(x, y_true)

    print(test_model)
    assert ref_model == test_model
