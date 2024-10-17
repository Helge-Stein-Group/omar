import fortran.omars as fortran
import utils
from suites import *


def test_data_matrix() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(100, 2)

    basis_indices = fortran.omars.active_base_indices(where, nbases)
    fit_matrix, basis_mean = fortran.omars.data_matrix(x, basis_indices, covariates,
                                                       nodes, hinges, where)

    suite_data_matrix(x, basis_mean, fit_matrix)


def test_fit() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    results = fortran.omars.fit(x, y, nbases, covariates, nodes, hinges, where, 3)

    suite_fit(results[1], results[2], y, y_true, results[-1])


def update_case(nbases: int,
                covariates: np.ndarray,
                nodes: np.ndarray,
                hinges: np.ndarray,
                where: np.ndarray, x: np.ndarray, y: np.ndarray, func: callable):
    fit_results = fortran.omars.fit(x, y, nbases, covariates, nodes, hinges, where, 3)
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
    fit_results = fortran.omars.fit(x, y, nbases, covariates, nodes, hinges, where, 3)
    for c in [0.5, 0.4, 0.3]:
        new_node = x[np.argmin(np.abs(x[:, 1] - c)), 1]

        addition_slice = slice(nbases, nbases + 2)
        covariates[:, addition_slice] = np.tile(covariates[:, nbases - 1], (2, 1)).T
        nodes[:, addition_slice] = np.tile(nodes[:, nbases - 1], (2, 1)).T
        hinges[:, addition_slice] = np.tile(hinges[:, nbases - 1], (2, 1)).T
        where[:, addition_slice] = np.tile(where[:, nbases - 1], (2, 1)).T

        nbases += 2

        parent_depth = np.sum(where[:, nbases - 1])

        covariates[parent_depth, addition_slice] = 1
        nodes[parent_depth, nbases] = 0.0
        hinges[parent_depth, nbases] = False
        where[parent_depth, addition_slice] = True

        nodes[parent_depth, nbases + 1] = new_node
        hinges[parent_depth, nbases + 1] = True

        fit_results = func(x, y, nbases, covariates, nodes, hinges, where, fit_results, 2)

    return nbases, covariates, nodes, hinges, where, fit_results


def shrink_case(nbases: int,
                covariates: np.ndarray,
                nodes: np.ndarray,
                hinges: np.ndarray,
                where: np.ndarray, x: np.ndarray, y: np.ndarray, func: callable):
    fit_results = fortran.omars.fit(x, y, nbases, covariates, nodes, hinges, where, 3)
    for i in range(6):
        where[:, nbases - 1] = False
        nbases -= 1
        fit_results = func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                           nbases)
    return nbases, covariates, nodes, hinges, where, fit_results

def test_update_fit_matrix() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    def update_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    old_node, parent_idx):
        init_results = fortran.omars.update_init(x, old_node, parent_idx, nbases,
                                                    covariates, nodes, where,
                                                    *fit_results[2:4])

        fortran.omars.update_fit_matrix(*fit_results[2:4], *init_results)
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = update_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        update_func)
    updated_basis_mean = fit_results[3]
    updated_fit_matrix = fit_results[2]

    full_fit_matrix, full_basis_mean = fortran.omars.calculate_fit_matrix(x, nbases,
                                                                             covariates,
                                                                             nodes,
                                                                             hinges,
                                                                             where)
    assert np.allclose(updated_basis_mean, full_basis_mean)
    assert np.allclose(updated_fit_matrix, full_fit_matrix)

def test_extend_fit_matrix() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    def extend_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    nadditions):
        fit_results = list(fit_results)
        fit_results[2], fit_results[3] = fortran.omars.extend_fit_matrix(x,
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

    full_fit_matrix, full_basis_mean = fortran.omars.calculate_fit_matrix(x,nbases,
                                                                             covariates,
                                                                             nodes,
                                                                             hinges,
                                                                             where)

    assert np.allclose(extended_basis_mean, full_basis_mean)
    assert np.allclose(extended_fit_matrix, full_fit_matrix)

def test_update_covariance_matrix() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    def update_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    old_node, parent_idx):
        init_results = fortran.omars.update_init(x, old_node, parent_idx, nbases,
                                                    covariates, nodes, where,
                                                    fit_results[2], fit_results[3])
        fit_results = list(fit_results)
        fortran.omars.update_fit_matrix(*fit_results[2:4], *init_results)
        covariance_addition = fortran.omars.update_covariance_matrix(fit_results[4],
                                                                        init_results[0],
                                                                        fit_results[2])
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = update_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        update_func)
    updated_covariance = fit_results[4]

    full_fit_matrix, full_basis_mean = fortran.omars.calculate_fit_matrix(x, nbases,
                                                                             covariates,
                                                                             nodes,
                                                                             hinges,
                                                                             where)
    full_covariance = fortran.omars.calculate_covariance_matrix(full_fit_matrix)

    assert np.allclose(updated_covariance[:-1, :-1], full_covariance[:-1, :-1])
    assert np.allclose(updated_covariance[-1, :-1], full_covariance[-1, :-1])
    assert np.allclose(updated_covariance, full_covariance)

def test_extend_covariance_matrix() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    def extend_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    nadditions):
        fit_results = list(fit_results)
        fit_results[2], fit_results[3] = fortran.omars.extend_fit_matrix(x,
                                                                            nadditions,
                                                                            *fit_results[
                                                                             2:4],
                                                                            covariates,
                                                                            nodes,
                                                                            hinges,
                                                                            where)
        fit_results[4] = fortran.omars.extend_covariance_matrix(fit_results[4],
                                                                   nadditions,
                                                                   fit_results[2])

        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = extend_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        extend_func)
    extended_covariance = fit_results[4]

    full_fit_matrix, full_basis_mean = fortran.omars.calculate_fit_matrix(x,nbases,
                                                                             covariates,
                                                                             nodes,
                                                                             hinges,
                                                                             where)
    full_covariance = fortran.omars.calculate_covariance_matrix(full_fit_matrix)

    assert np.allclose(extended_covariance, full_covariance)

def test_update_right_hand_side() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    def update_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    old_node, parent_idx):
        init_results = fortran.omars.update_init(x, old_node, parent_idx, nbases,
                                                    covariates, nodes, where,
                                                    fit_results[2], fit_results[3])
        fortran.omars.update_right_hand_side(fit_results[6], y, y.mean(),
                                                init_results[0])
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = update_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        update_func)
    updated_right_hand_side = fit_results[6]

    full_fit_matrix, full_basis_mean = fortran.omars.calculate_fit_matrix(x,nbases,
                                                                             covariates,
                                                                             nodes,
                                                                             hinges,
                                                                             where)
    full_right_hand_side, _ = fortran.omars.calculate_right_hand_side(y,
                                                                         full_fit_matrix)

    assert np.allclose(updated_right_hand_side, full_right_hand_side)

def test_extend_right_hand_side() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    def extend_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    nadditions):
        fit_results = list(fit_results)
        fit_results[2], fit_results[3] = fortran.omars.extend_fit_matrix(x,
                                                                            nadditions,
                                                                            *fit_results[
                                                                             2:4],
                                                                            covariates,
                                                                            nodes,
                                                                            hinges,
                                                                            where)
        fit_results[6] = fortran.omars.extend_right_hand_side(fit_results[6], y,
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

    full_fit_matrix, full_basis_mean = fortran.omars.calculate_fit_matrix(x,nbases,
                                                                             covariates,
                                                                             nodes,
                                                                             hinges,
                                                                             where)
    full_right_hand_side, _ = fortran.omars.calculate_right_hand_side(y,
                                                                         full_fit_matrix)

    assert np.allclose(extended_right_hand_side, full_right_hand_side)

def test_decompose() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    fit_results = fortran.omars.fit(x, y, nbases, covariates, nodes, hinges, where, 3)

    former_covariance = fit_results[4].copy()

    old_node = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
    new_node = x[np.argmin(np.abs(x[:, 1] - 0.5)), 1]
    nodes[2, 2] = new_node
    init_results = fortran.omars.update_init(x, old_node, 1, nbases, covariates,
                                                nodes, where, *fit_results[2:4])

    fortran.omars.update_fit_matrix(*fit_results[2:4], *init_results)

    covariance_addition = fortran.omars.update_covariance_matrix(fit_results[4],
                                                                    init_results[0],
                                                                    fit_results[2])

    updated_covariance = fit_results[4]

    eigenvalues, eigenvectors = fortran.omars.decompose_addition(covariance_addition)
    reconstructed_covariance = former_covariance + eigenvalues[0] * np.outer(
        eigenvectors[0], eigenvectors[0])
    reconstructed_covariance += eigenvalues[1] * np.outer(eigenvectors[1],
                                                          eigenvectors[1])

    assert np.allclose(reconstructed_covariance, updated_covariance)

def test_update_cholesky() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    def update_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    old_node, parent_idx):
        init_results = fortran.omars.update_init(x, old_node, parent_idx, nbases,
                                                    covariates, nodes, where,
                                                    *fit_results[2:4])
        fit_results = list(fit_results)
        fortran.omars.update_fit_matrix(*fit_results[2:4], *init_results)
        covariance_addition = fortran.omars.update_covariance_matrix(fit_results[4],
                                                                        init_results[0],
                                                                        fit_results[2])
        if covariance_addition.any():
            eigenvalues, eigenvectors = fortran.omars.decompose_addition(
                covariance_addition)
            fortran.omars.update_cholesky(fit_results[5], eigenvectors, eigenvalues)
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = update_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        update_func)
    updated_cholesky = fit_results[5]

    full_fit_results = fortran.omars.fit(x, y, nbases, covariates, nodes, hinges,
                                            where, 3)
    full_cholesky = full_fit_results[5]
    assert np.allclose(np.tril(updated_cholesky), np.tril(full_cholesky))