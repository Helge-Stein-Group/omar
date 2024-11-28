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

    results = fortran.omars.fit(x, y, y.mean(),nbases, covariates, nodes, hinges, where, 3)

    suite_fit(results[1], results[2], y, y_true, y.mean())


def update_case(nbases: int,
                covariates: np.ndarray,
                nodes: np.ndarray,
                hinges: np.ndarray,
                where: np.ndarray, x: np.ndarray, y: np.ndarray, func: callable):
    fit_results = fortran.omars.fit(x, y, y.mean(), nbases, covariates, nodes, hinges, where, 3)
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
    fit_results = fortran.omars.fit(x, y, y.mean(),nbases, covariates, nodes, hinges, where, 3)
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
    fit_results = fortran.omars.fit(x, y, y.mean(),nbases, covariates, nodes, hinges, where, 3)
    for i in range(6):
        where[:, nbases - 1] = False
        nbases -= 1
        fit_results = func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                           nbases - 1)
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

    full_fit_matrix, full_basis_mean = fortran.omars.calculate_fit_matrix(x, nbases,
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

    full_fit_matrix, full_basis_mean = fortran.omars.calculate_fit_matrix(x, nbases,
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

    full_fit_matrix, full_basis_mean = fortran.omars.calculate_fit_matrix(x, nbases,
                                                                          covariates,
                                                                          nodes,
                                                                          hinges,
                                                                          where)
    full_right_hand_side = fortran.omars.calculate_right_hand_side(y, y.mean(), full_fit_matrix)

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
        fit_results[6] = fortran.omars.extend_right_hand_side(fit_results[6], y, y.mean(),
                                                              fit_results[2],

                                                              nadditions)
        return fit_results

    nbases, covariates, nodes, hinges, where, fit_results = extend_case(nbases,
                                                                        covariates,
                                                                        nodes, hinges,
                                                                        where, x, y,
                                                                        extend_func)
    extended_right_hand_side = fit_results[6]

    full_fit_matrix, full_basis_mean = fortran.omars.calculate_fit_matrix(x, nbases,
                                                                          covariates,
                                                                          nodes,
                                                                          hinges,
                                                                          where)
    full_right_hand_side = fortran.omars.calculate_right_hand_side(y, y.mean(),
                                                                   full_fit_matrix)

    assert np.allclose(extended_right_hand_side, full_right_hand_side)


def test_decompose() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    fit_results = fortran.omars.fit(x, y, y.mean(),nbases, covariates, nodes, hinges, where, 3)

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

    full_fit_results = fortran.omars.fit(x, y, y.mean(), nbases, covariates, nodes, hinges,
                                         where, 3)
    full_cholesky = full_fit_results[5]
    assert np.allclose(np.tril(updated_cholesky), np.tril(full_cholesky))


def test_update_fit() -> None:
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    def update_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    old_node, parent_idx):
        fit_results = list(fit_results)
        fit_results[:2] = fortran.omars.update_fit(x, y, y.mean(),nbases, covariates, nodes,
                                                   where, 3, *fit_results[2:5],
                                                   fit_results[6], old_node,
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

    full_fit_results = fortran.omars.fit(x, y,y.mean(), nbases, covariates, nodes, hinges,
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
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    def extend_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    nadditions):
        fit_results = list(fit_results)

        temp_fit, temp_mean = fortran.omars.extend_fit_matrix(x, nadditions, *fit_results[2:4],
                                                              covariates, nodes, hinges, where)
        temp_cov = fortran.omars.extend_covariance_matrix(fit_results[4], nadditions,
                                                          temp_fit)
        temp_rhs = fortran.omars.extend_right_hand_side(fit_results[6], y, y.mean(), temp_fit,
                                                        nadditions)
        fit_results[:-1] = fortran.omars.extend_fit(x, y, y.mean(), nbases, covariates, nodes,
                                                    hinges, where, 3, nadditions,
                                                    *fit_results[2:5],
                                                    fit_results[6])
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

    full_fit_results = fortran.omars.fit(x, y,y.mean(), nbases, covariates, nodes, hinges,
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


def test_shrink_fit() -> None: # TODO deprecated since we are no longer  returning shrunk fit_results
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(
        100, 2)

    additional_covariates = np.tile(covariates[:, 1], (7, 1)).T
    additional_nodes = np.tile(nodes[:, 1], (7, 1)).T
    additional_hinges = np.tile(hinges[:, 1], (7, 1)).T
    additional_where = np.tile(where[:, 1], (7, 1)).T

    additional_nodes[2, :] = np.random.choice(x[:, 0], 7)

    nbases += 7
    covariates[:, nbases - 7:nbases] = additional_covariates
    nodes[:, nbases - 7:nbases] = additional_nodes
    hinges[:, nbases - 7:nbases] = additional_hinges
    where[:, nbases - 7:nbases] = additional_where

    def shrink_func(x, y, nbases, covariates, nodes, hinges, where, fit_results,
                    removal_idx):
        fit_results = list(fit_results)
        fit_results[:-1] = fortran.omars.shrink_fit(y, y.mean(), nbases, 3,
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

    full_fit_results = fortran.omars.fit(x, y,y.mean(), nbases, covariates, nodes, hinges,
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

def test_expand_bases() -> None:
    x, y, y_true, nbases, ref_covariates, ref_nodes, ref_hinges, ref_where = utils.generate_data_and_splines(
        100, 2)
    from regression import OMARS
    test_model = OMARS()
    test_model._expand_bases(x, y)
    print(test_model)
    print(test_model.lof)
    expansion_results = fortran.omars.expand_bases(x, y, y.mean(), 11, 3, 11, 0.0)
    print(f"LOF: {expansion_results[5]}")
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

    import regression_numba
    model = regression_numba.OMARS(nbases, covariates, nodes, hinges, where, np.zeros(11), y.mean())

    print(model)
    assert match1
    assert match2

def test_prune_bases() -> None:
    x, y, y_true, ref_nbases, ref_covariates, ref_nodes, ref_hinges, ref_where = utils.generate_data_and_splines(
        100,
        2)

    nbases = ref_nbases
    covariates = ref_covariates.copy()
    nodes = ref_nodes.copy()
    hinges = ref_hinges.copy()
    where = ref_where.copy()

    lof, coefficients_prev, fit_matrix, basis_mean, covariance_matrix, chol, right_hand_side = fortran.omars.fit(
        x, y, y.mean(), nbases, covariates, nodes, hinges, where, 3
    )

    # Call the prune_bases subroutine with the correct arguments
    coefficients, where = fortran.omars.prune_bases(
        x, y, y.mean(), nbases, covariates, nodes, hinges, where,
        lof, fit_matrix, basis_mean, covariance_matrix, right_hand_side, 3
    )
    coefficients = coefficients[:nbases]

    import regression_numba
    ref_model = regression_numba.OMARS(ref_nbases, ref_covariates, ref_nodes,
                                       ref_hinges,
                                       ref_where, coefficients_prev, y.mean())
    test_model = regression_numba.OMARS(nbases, covariates, nodes, hinges, where,
                                        coefficients, y.mean())
    assert ref_model == test_model

def test_find_bases() -> None:
    x, y, y_true = utils.generate_data(100, 2)

    from regression import OMARS

    from regression import OMARS as OMARS_numba

    model_numba = OMARS_numba(11, 5, 0, 3)
    model_numba.find_bases(x, y)

    model_ref = OMARS()
    model_ref.find_bases(x, y)

    nbases, covariates, nodes, hinges, where, coefficients = fortran.omars.find_bases(x, y, y.mean(), 11, 5, 0, 3)
    # TODO coefficients have the wrong size only the nbases first values are valid
    coefficients = coefficients[:nbases - 1]
    hinges = hinges.astype(bool)
    where = where.astype(bool)
    model = OMARS()
    model.nbases = nbases
    model.coefficients = coefficients
    model.covariates = covariates
    model.nodes = nodes
    model.hinges = hinges
    model.where = where
    model.y_mean = y.mean()
    model._calculate_fit_matrix(x)
    model._generalised_cross_validation(y)

    print(model)
    print(model_ref)
    print(model_numba)
    print(model.lof)
    print(model_ref.lof)
    print(model_numba.lof)