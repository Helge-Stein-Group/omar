import numpy as np

import regression_noop
import utils


def test_data_matrix() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    basis_indices = regression_noop.active_base_indices(where)
    fit_matrix, basis_mean = regression_noop.data_matrix(x, basis_indices, covariates,
                                                         nodes, hinges, where)

    x1 = x[np.argmin(np.abs(x[:, 0] - 1)), 0]
    x08 = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    ref_data_matrix = np.column_stack([
        np.maximum(0, x[:, 0] - x1),
        np.maximum(0, x[:, 0] - x1) * np.maximum(0, x[:, 1] - x08),
    ])

    ref_basis_mean = ref_data_matrix.mean(axis=0)
    ref_data_matrix -= ref_basis_mean

    assert np.allclose(ref_basis_mean, basis_mean)
    assert np.allclose(ref_data_matrix, fit_matrix)


def test_fit() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    results = regression_noop.fit(x, y, nbases, covariates, nodes, hinges, where, 3)
    coefficients = results[1]

    ref_coefficients = np.linalg.lstsq(results[2], y, rcond=None)[0]

    y_pred_model = results[2] @ coefficients + results[-1]
    mse_model = np.mean((y_pred_model - y_true) ** 2)
    y_pred_np = results[2] @ ref_coefficients + results[-1]
    mse_np = np.mean((y_pred_np - y_true) ** 2)

    assert np.allclose(mse_model, mse_np, 0.01)


def update_case(nbases: int,
                covariates: np.ndarray,
                nodes: np.ndarray,
                hinges: np.ndarray,
                where: np.ndarray, x: np.ndarray, y: np.ndarray, func: callable):
    fit_results = regression_noop.fit(x, y, nbases, covariates, nodes, hinges, where, 3)
    old_node = nodes[np.sum(where[:, nbases - 1]), nbases - 1]
    for new_node in sorted([value for value in x[:, 1] if value < old_node],
                           reverse=True)[:3]:
        nodes[np.sum(where[:, nbases - 1]), nbases - 1] = new_node

        fit_results = func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                           old_node, 1)
        old_node = new_node
    return nbases, covariates, nodes, hinges, where, fit_results


def extend_case(nbases: int,
                covariates: np.ndarray,
                nodes: np.ndarray,
                hinges: np.ndarray,
                where: np.ndarray, x: np.ndarray, y: np.ndarray, func: callable):
    fit_results = regression_noop.fit(x, y, nbases, covariates, nodes, hinges, where, 3)
    for c in [0.5, 0.4, 0.3]:
        new_node = x[np.argmin(np.abs(x[:, 1] - c)), 1]

        additional_covariates = np.tile(covariates[:, nbases - 1], (2, 1)).T
        additional_nodes = np.tile(nodes[:, nbases - 1], (2, 1)).T
        additional_hinges = np.tile(hinges[:, nbases - 1], (2, 1)).T
        additional_where = np.tile(where[:, nbases - 1], (2, 1)).T

        parent_depth = np.sum(where[:, nbases - 1])

        additional_covariates[parent_depth, 0] = 1
        additional_nodes[parent_depth, 0] = 0.0
        additional_hinges[parent_depth, 0] = False
        additional_where[parent_depth, 0] = True

        additional_covariates[parent_depth, 1] = 1
        additional_nodes[parent_depth, 1] = new_node
        additional_hinges[parent_depth, 1] = True
        additional_where[parent_depth, 1] = True

        nbases = regression_noop.add_bases(nbases, additional_covariates,
                                           additional_nodes, additional_hinges,
                                           additional_where, covariates, nodes, hinges,
                                           where)
        fit_results = func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                           2)

    return nbases, covariates, nodes, hinges, where, fit_results


def shrink_case(nbases: int,
                covariates: np.ndarray,
                nodes: np.ndarray,
                hinges: np.ndarray,
                where: np.ndarray, x: np.ndarray, y: np.ndarray, func: callable):
    fit_results = regression_noop.fit(x, y, nbases, covariates, nodes, hinges, where, 3)
    for i in range(6):
        nbases = regression_noop.remove_bases(nbases, where, nbases - 1)
        fit_results = func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                           nbases)
    return nbases, covariates, nodes, hinges, where, fit_results


def test_update_fit_matrix() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    def update_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    old_node, parent_idx):
        init_results = regression_noop.update_init(x, old_node, parent_idx, nbases,
                                                   covariates, nodes, where,
                                                   *fit_results[2:4])
        regression_noop.update_fit_matrix(*fit_results[2:4], *init_results)
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = update_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        update_func)
    updated_basis_mean = fit_results[3]
    updated_fit_matrix = fit_results[2]

    full_fit_matrix, full_basis_mean = regression_noop.calculate_fit_matrix(x,
                                                                            covariates,
                                                                            nodes,
                                                                            hinges,
                                                                            where)

    assert np.allclose(updated_basis_mean, full_basis_mean)
    assert np.allclose(updated_fit_matrix, full_fit_matrix)


def test_extend_fit_matrix() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    def extend_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    nadditions):
        fit_results = list(fit_results)
        fit_results[2], fit_results[3] = regression_noop.extend_fit_matrix(x,
                                                                           nadditions,
                                                                           *fit_results[
                                                                            2:4],
                                                                           covariates,
                                                                           nodes,
                                                                           hinges,
                                                                           where)
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = extend_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        extend_func)
    extended_basis_mean = fit_results[3]
    extended_fit_matrix = fit_results[2]

    full_fit_matrix, full_basis_mean = regression_noop.calculate_fit_matrix(x,
                                                                            covariates,
                                                                            nodes,
                                                                            hinges,
                                                                            where)

    assert np.allclose(extended_basis_mean, full_basis_mean)
    assert np.allclose(extended_fit_matrix, full_fit_matrix)


def test_update_covariance_matrix() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    def update_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    old_node, parent_idx):
        init_results = regression_noop.update_init(x, old_node, parent_idx, nbases,
                                                   covariates, nodes, where,
                                                   fit_results[2], fit_results[3])
        fit_results = list(fit_results)
        regression_noop.update_fit_matrix(*fit_results[2:4], *init_results)
        covariance_addition = regression_noop.update_covariance_matrix(fit_results[4],
                                                                       init_results[0],
                                                                       fit_results[2])
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = update_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        update_func)
    updated_covariance = fit_results[4]

    full_fit_matrix, full_basis_mean = regression_noop.calculate_fit_matrix(x,
                                                                            covariates,
                                                                            nodes,
                                                                            hinges,
                                                                            where)
    full_covariance = regression_noop.calculate_covariance_matrix(full_fit_matrix)

    assert np.allclose(updated_covariance[:-1, :-1], full_covariance[:-1, :-1])
    assert np.allclose(updated_covariance[-1, :-1], full_covariance[-1, :-1])
    assert np.allclose(updated_covariance, full_covariance)


def test_extend_covariance_matrix() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    def extend_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    nadditions):
        fit_results = list(fit_results)
        fit_results[2], fit_results[3] = regression_noop.extend_fit_matrix(x,
                                                                           nadditions,
                                                                           *fit_results[
                                                                            2:4],
                                                                           covariates,
                                                                           nodes,
                                                                           hinges,
                                                                           where)
        fit_results[4] = regression_noop.extend_covariance_matrix(fit_results[4],
                                                                  nadditions,
                                                                  fit_results[2])

        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = extend_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        extend_func)
    extended_covariance = fit_results[4]

    full_fit_matrix, full_basis_mean = regression_noop.calculate_fit_matrix(x,
                                                                            covariates,
                                                                            nodes,
                                                                            hinges,
                                                                            where)
    full_covariance = regression_noop.calculate_covariance_matrix(full_fit_matrix)

    assert np.allclose(extended_covariance, full_covariance)


def test_update_right_hand_side() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    def update_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    old_node, parent_idx):
        init_results = regression_noop.update_init(x, old_node, parent_idx, nbases,
                                                   covariates, nodes, where,
                                                   fit_results[2], fit_results[3])
        regression_noop.update_right_hand_side(fit_results[6], y, y.mean(),
                                               init_results[0])
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = update_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        update_func)
    updated_right_hand_side = fit_results[6]

    full_fit_matrix, full_basis_mean = regression_noop.calculate_fit_matrix(x,
                                                                            covariates,
                                                                            nodes,
                                                                            hinges,
                                                                            where)
    full_right_hand_side, _ = regression_noop.calculate_right_hand_side(y,
                                                                        full_fit_matrix)

    assert np.allclose(updated_right_hand_side, full_right_hand_side)


def test_extend_right_hand_side() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    def extend_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    nadditions):
        fit_results = list(fit_results)
        fit_results[2], fit_results[3] = regression_noop.extend_fit_matrix(x,
                                                                           nadditions,
                                                                           *fit_results[
                                                                            2:4],
                                                                           covariates,
                                                                           nodes,
                                                                           hinges,
                                                                           where)
        fit_results[6], _ = regression_noop.extend_right_hand_side(fit_results[6], y,
                                                                   fit_results[2],
                                                                   y.mean(),
                                                                   nadditions)
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = extend_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        extend_func)
    extended_right_hand_side = fit_results[6]

    full_fit_matrix, full_basis_mean = regression_noop.calculate_fit_matrix(x,
                                                                            covariates,
                                                                            nodes,
                                                                            hinges,
                                                                            where)
    full_right_hand_side, _ = regression_noop.calculate_right_hand_side(y,
                                                                        full_fit_matrix)

    assert np.allclose(extended_right_hand_side, full_right_hand_side)


def test_decompose() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    fit_results = regression_noop.fit(x, y, nbases, covariates, nodes, hinges, where, 3)

    former_covariance = fit_results[4].copy()

    old_node = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    new_node = x[np.argmin(np.abs(x[:, 1] - 0.5)), 1]
    nodes[2, 2] = new_node
    init_results = regression_noop.update_init(x, old_node, 1, nbases, covariates,
                                               nodes, where, *fit_results[2:4])

    regression_noop.update_fit_matrix(*fit_results[2:4], *init_results)

    covariance_addition = regression_noop.update_covariance_matrix(fit_results[4],
                                                                   init_results[0],
                                                                   fit_results[2])

    updated_covariance = fit_results[4]

    eigenvalues, eigenvectors = regression_noop.decompose_addition(covariance_addition)
    reconstructed_covariance = former_covariance + eigenvalues[0] * np.outer(
        eigenvectors[0], eigenvectors[0])
    reconstructed_covariance += eigenvalues[1] * np.outer(eigenvectors[1],
                                                          eigenvectors[1])

    assert np.allclose(reconstructed_covariance, updated_covariance)


def test_update_cholesky() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    def update_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    old_node, parent_idx):
        init_results = regression_noop.update_init(x, old_node, parent_idx, nbases,
                                                   covariates, nodes, where,
                                                   *fit_results[2:4])
        fit_results = list(fit_results)
        regression_noop.update_fit_matrix(*fit_results[2:4], *init_results)
        covariance_addition = regression_noop.update_covariance_matrix(fit_results[4],
                                                                       init_results[0],
                                                                       fit_results[2])
        if covariance_addition.any():
            eigenvalues, eigenvectors = regression_noop.decompose_addition(
                covariance_addition)
            fit_results[5] = regression_noop.update_cholesky(fit_results[5],
                                                             eigenvectors, eigenvalues)
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = update_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        update_func)
    updated_cholesky = fit_results[5]

    full_fit_results = regression_noop.fit(x, y, nbases, covariates, nodes, hinges,
                                           where, 3)
    full_cholesky = full_fit_results[5]
    assert np.allclose(np.tril(updated_cholesky), np.tril(full_cholesky))


def test_update_fit() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    def update_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    old_node, parent_idx):
        fit_results = list(fit_results)
        fit_results[:-1] = regression_noop.update_fit(x, y, nbases, covariates, nodes,
                                                      where, 3, *fit_results[2:5],
                                                      fit_results[6],
                                                      fit_results[-1], old_node,
                                                      parent_idx, fit_results[5])
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = update_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        update_func)
    updated_chol = fit_results[5]
    updated_rhs = fit_results[6]
    updated_coefficients = fit_results[1]
    updated_lof = fit_results[0]

    full_fit_results = regression_noop.fit(x, y, nbases, covariates, nodes, hinges,
                                           where, 3)
    full_chol = full_fit_results[5]
    full_rhs = full_fit_results[6]
    full_coefficients = full_fit_results[1]
    full_lof = full_fit_results[0]

    assert np.allclose(np.tril(updated_chol), np.tril(full_chol))
    assert np.allclose(updated_rhs, full_rhs)
    assert np.allclose(updated_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(updated_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(updated_lof, full_lof, 0.01)


def test_extend_fit() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    def extend_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    nadditions):
        fit_results = list(fit_results)
        fit_results[:-1] = regression_noop.extend_fit(x, y, nbases, covariates, nodes,
                                                      hinges, where, 3, nadditions,
                                                      *fit_results[2:5],
                                                      fit_results[6], fit_results[-1])
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = extend_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        extend_func)
    chol = fit_results[5]
    extended_rhs = fit_results[6]
    extended_coefficients = fit_results[1]
    extended_lof = fit_results[0]

    full_fit_results = regression_noop.fit(x, y, nbases, covariates, nodes, hinges,
                                           where, 3)
    full_chol = full_fit_results[5]
    full_rhs = full_fit_results[6]
    full_coefficients = full_fit_results[1]
    full_lof = full_fit_results[0]

    assert np.allclose(np.tril(chol), np.tril(full_chol))
    assert np.allclose(extended_rhs, full_rhs)
    assert np.allclose(extended_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(extended_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(extended_lof, full_lof, 0.01)


def test_shrink_fit() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.data_generation_model_noop(
        100, 2)

    additional_covariates = np.tile(covariates[:, 1], (7, 1)).T
    additional_nodes = np.tile(nodes[:, 1], (7, 1)).T
    additional_hinges = np.tile(hinges[:, 1], (7, 1)).T
    additional_where = np.tile(where[:, 1], (7, 1)).T

    additional_nodes[2, :] = np.random.choice(x[:, 0], 7)

    nbases = regression_noop.add_bases(nbases,
                                       additional_covariates,
                                       additional_nodes,
                                       additional_hinges,
                                       additional_where, covariates, nodes, hinges,
                                       where)

    def shrink_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    removal_idx):
        fit_results = list(fit_results)
        fit_results[:-1] = regression_noop.shrink_fit(y, fit_results[-1], nbases, 3,
                                                      removal_idx, *fit_results[2:5],
                                                      fit_results[6])
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = shrink_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        shrink_func)
    chol = fit_results[5]
    shrunk_rhs = fit_results[6]
    shrunk_coefficients = fit_results[1]
    shrunk_lof = fit_results[0]

    full_fit_results = regression_noop.fit(x, y, nbases, covariates, nodes, hinges,
                                           where, 3)
    full_chol = full_fit_results[5]
    full_rhs = full_fit_results[6]
    full_coefficients = full_fit_results[1]
    full_lof = full_fit_results[0]

    assert np.allclose(np.tril(chol), np.tril(full_chol))
    assert np.allclose(shrunk_rhs, full_rhs)
    assert np.allclose(shrunk_coefficients[0], full_coefficients[0], 0.05, 0.1)
    assert np.allclose(shrunk_coefficients[1], full_coefficients[1], 0.05, 0.05)
    assert np.allclose(shrunk_lof, full_lof, 0.01)

# May fail but should be somewhat close
def test_expand_bases() -> None:
    x, y, y_true, nbases, ref_covariates, ref_nodes, ref_hinges, ref_where = utils.data_generation_model_noop(
        100, 2)

    expansion_results = regression_noop.expand_bases(x, y, 11, 3, 11, 0)
    nbases = expansion_results[0]
    covariates = expansion_results[1]
    nodes = expansion_results[2]
    hinges = expansion_results[3]
    where = expansion_results[4]

    expected_node_1 = ref_nodes[1, 1]
    expected_node_2 = ref_nodes[2, 2]

    potential_bases_1 = np.where(
        np.sum(np.isin(nodes, expected_node_1), axis=0) > 0)[0]
    potential_bases_2 = np.where(
        np.sum(np.isin(nodes, expected_node_2), axis=0) > 0)[0]

    potential_bases_12 = np.intersect1d(potential_bases_1, potential_bases_2)

    match1 = False
    match2 = False

    for pidx in potential_bases_1:
        if (np.allclose(nodes[:, pidx], ref_nodes[:, 1]) and
                np.allclose(covariates[:, pidx], ref_covariates[:, 1]) and
                np.allclose(hinges[:, pidx], ref_hinges[:, 1]) and
                np.allclose(where[:, pidx], ref_where[:, 1])):
            match1 = True

    for pidx in potential_bases_12:
        if (np.allclose(nodes[:, pidx], ref_nodes[:, 2]) and
                np.allclose(covariates[:, pidx], ref_covariates[:, 2]) and
                np.allclose(hinges[:, pidx], ref_hinges[:, 2]) and
                np.allclose(where[:, pidx], ref_where[:, 2])):
            match2 = True

    print("Expected node 1: ", expected_node_1)
    print("Expected node 2: ", expected_node_2)

    model = regression_noop.OMARS(nbases, covariates, nodes, hinges, where,
                                  np.zeros(11), y.mean())
    print(model)
    assert match1
    assert match2


def test_prune_bases() -> None:
    x, y, y_true, ref_nbases, ref_covariates, ref_nodes, ref_hinges, ref_where = utils.data_generation_model_noop(
        100,
        2)

    nbases = ref_nbases
    covariates = ref_covariates.copy()
    nodes = ref_nodes.copy()
    hinges = ref_hinges.copy()
    where = ref_where.copy()

    fit_results = regression_noop.fit(x, y, nbases, covariates, nodes, hinges, where, 3)

    nbases, where, coefficients = regression_noop.prune_bases(x, y, nbases, covariates,
                                                              nodes, hinges, where,
                                                              fit_results[0],
                                                              *fit_results[2:5],
                                                              *fit_results[6:8],
                                                              3)

    ref_model = regression_noop.OMARS(ref_nbases, ref_covariates, ref_nodes, ref_hinges,
                                      ref_where, fit_results[1], y.mean())
    test_model = regression_noop.OMARS(nbases, covariates, nodes, hinges, where,
                                       coefficients, y.mean())
    assert ref_model == test_model
