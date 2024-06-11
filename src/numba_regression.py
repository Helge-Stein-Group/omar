import numpy as np
from scipy.linalg import cho_factor, cho_solve
from numba import njit, int32, float32, bool_


@njit(cache=True, fastmath=True)
def update_cholesky(chol: np.ndarray,
                    update_vectors: list[np.ndarray],
                    multipliers: list[float32]) -> np.ndarray:
    """
    Update the Cholesky decomposition by rank-1 matrices.
    Args:
        chol: Cholesky decomposition of the original matrix. [nbases x nbases]
        update_vectors: List of update vectors. List of [nbases x 1].
        multipliers: List of multipliers. List of [1]

    Returns:
        Updated Cholesky decomposition.

    Notes: Algortihm according to [1] Oswin Krause. Christian Igel. A More Efficient Rank-one Covariance Matrix Update
    for Evolution Strategies. 2015 ACM Conference. https://christian-igel.github.io/paper/AMERCMAUfES.pdf.
    Adapted for computation speed and parallelization.
    """

    assert chol.shape[0] == chol.shape[1]
    assert chol.shape[0] == len(update_vectors[0])
    assert len(update_vectors) == len(multipliers)

    for update_vec, multiplier in zip(update_vectors, multipliers):
        diag = np.diag(chol).copy()
        chol = chol / diag
        diag **= 2

        u = np.zeros((chol.shape[0], update_vec.shape[0]))
        u[0, :] = update_vec
        u[0, 1:] -= update_vec[0] * chol[1:, 0]
        b = np.ones(chol.shape[0])
        for i in range(1, chol.shape[0]):
            u[i, :] = u[i - 1, :]
            u[i, i + 1:] -= u[i - 1, i] * chol[i + 1:, i]
            b[i] = b[i - 1] + multiplier * u[i - 1, i - 1] ** 2 / diag[i - 1]

        for i in range(chol.shape[0]):
            chol[i, i] = np.sqrt(diag[i] + multiplier / b[i] * u[i, i] ** 2)
            chol[i + 1:, i] *= chol[i, i]
            chol[i + 1:, i] += multiplier / b[i] * u[i, i] * u[i, i + 1:] / chol[i, i]

    return chol


def data_matrix(x: np.ndarray,
                basis_slice: slice,
                nodes: np.ndarray,
                covariates: np.ndarray,
                hinges: np.ndarray,
                where: np.ndarray) -> np.ndarray:
    result = -nodes[:, basis_slice] + x[:, covariates[:, basis_slice]]
    np.maximum(0, result, where=hinges[:, basis_slice], out=result)

    return result.prod(axis=1, where=where[:, basis_slice])


def print_bases(nbases: int32,
                max_nbases: int32,
                nodes: np.ndarray,
                covariates: np.ndarray,
                hinges: np.ndarray,
                where: np.ndarray
                ) -> str:
    desc = "Basis functions: \n"
    for basis_idx in range(nbases):
        for func_idx in range(max_nbases):
            if where[func_idx, basis_idx]:
                cov = covariates[func_idx, basis_idx]
                node = nodes[func_idx, basis_idx]
                hinge = hinges[func_idx, basis_idx]
                desc += f"(x[{cov} - {node}]){u'\u208A' if hinge else ''}"
        desc += "\n"
    return desc


def predict(x: np.ndarray,
            nbases: int32,
            coefficients: np.ndarray,
            nodes: np.ndarray,
            covariates: np.ndarray,
            hinges: np.ndarray,
            where: np.ndarray
            ) -> np.ndarray:
    return data_matrix(x, slice(nbases),
                       nodes,
                       covariates,
                       hinges,
                       where) @ coefficients


def add_basis(nbases,
              new_covariates: np.ndarray,
              new_nodes: np.ndarray,
              new_hinges: np.ndarray,
              new_where: np.ndarray,
              covariates: np.ndarray,
              nodes: np.ndarray,
              hinges: np.ndarray,
              where: np.ndarray):
    addition_slice = slice(nbases, nbases + covariates.shape[1])
    covariates[:, addition_slice] = new_covariates
    nodes[:, addition_slice] = new_nodes
    hinges[:, addition_slice] = new_hinges
    where[:, addition_slice] = new_where
    nbases += new_covariates.shape[1]

    return nbases


def update_basis(nbases,
                 new_covariates: np.ndarray,
                 new_nodes: np.ndarray,
                 new_hinges: np.ndarray,
                 new_where: np.ndarray,
                 covariates: np.ndarray,
                 nodes: np.ndarray,
                 hinges: np.ndarray,
                 where: np.ndarray) -> None:
    covariates[:, nbases - 1] = new_covariates
    nodes[:, nbases - 1] = new_nodes
    hinges[:, nbases - 1] = new_hinges
    where[:, nbases - 1] = new_where


def remove_basis(
        nbases,
        where: np.ndarray,
        removal_slice: slice):
    where[:, removal_slice] = False
    nbases -= removal_slice.stop - removal_slice.start

    return nbases


def calculate_means(fit_matrix: np.ndarray,
                    collapsed_fit: np.ndarray):
    fixed_mean = collapsed_fit[:-1] / fit_matrix.shape[0]
    candidate_mean = collapsed_fit[-1] / fit_matrix.shape[0]

    return fixed_mean, candidate_mean


def extend_means(fixed_mean, candidate_mean, fit_matrix: np.ndarray,
                 collapsed_fit: np.ndarray, nadditions: int32):
    fixed_mean = np.append(fixed_mean, candidate_mean)
    fixed_mean = np.append(fixed_mean,
                           collapsed_fit[:-1] / fit_matrix.shape[0])
    candidate_mean = collapsed_fit[-1] / fit_matrix.shape[0]
    return fixed_mean, candidate_mean


def shrink_means(fixed_mean, candidate_mean,
                 removal_slice: slice):
    joint_mean = np.append(fixed_mean, candidate_mean)
    reduced_mean = np.delete(joint_mean, removal_slice)

    fixed_mean = reduced_mean[:-1]
    candidate_mean = reduced_mean[-1]

    return fixed_mean, candidate_mean


def update_init(
        nbases: int32,
        covariates: np.ndarray,
        nodes: np.ndarray,
        where: np.ndarray,
        fit_matrix: np.ndarray,
        candidate_mean: float32,
        x: np.ndarray,
        old_node: float32,
        parent_idx: int32):
    prod_idx = np.sum(where[:, nbases - 1])
    new_node = nodes[prod_idx, nbases - 1]
    covariate = covariates[prod_idx, nbases - 1]

    indices = x[:, covariate] > new_node
    update = np.where(x[indices, covariate] >= old_node,
                      old_node - new_node,
                      x[indices, covariate] - new_node)
    update *= fit_matrix[indices, parent_idx]

    update_mean = np.sum(update) / len(x)
    candidate_mean += update_mean

    return indices, update, update_mean, candidate_mean


def calculate_fit_matrix(x: np.ndarray, nbases: int32,
                         nodes, covariates, hinges, where) -> np.ndarray:
    fit_matrix = data_matrix(x, slice(nbases), nodes, covariates, hinges, where)

    return fit_matrix


def extend_fit_matrix(x: np.ndarray, nadditions: int32, nbases: int,
                      fit_matrix: np.ndarray,
                      covariates, nodes, hinges, where):
    ext_slice = slice(nbases - nadditions, nbases)
    fit_matrix = np.column_stack((
        fit_matrix,
        data_matrix(x, ext_slice, nodes, covariates, hinges, where)
    ))
    return fit_matrix


def update_fit_matrix(fit_matrix: np.ndarray, indices, update):
    fit_matrix[indices, -1] += update
    return fit_matrix


def calculate_covariance_matrix(fit_matrix: np.ndarray):
    covariance_matrix = fit_matrix.T @ fit_matrix
    collapsed_fit = np.sum(fit_matrix, axis=0)
    fixed_mean, candidate_mean = calculate_means(fit_matrix, collapsed_fit)
    covariance_matrix -= np.outer(collapsed_fit, collapsed_fit) / \
                         fit_matrix.shape[0]

    covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-8
    return fixed_mean, candidate_mean, covariance_matrix


def extend_covariance_matrix(nadditions: int32,
                             fit_matrix,
                             covariance_matrix: np.ndarray,
                             fixed_mean,
                             candidate_mean):
    covariance_extension = fit_matrix.T @ fit_matrix[:, -nadditions:]
    collapsed_fit = np.sum(fit_matrix[:, -nadditions:], axis=0)
    fixed_mean, candidate_mean = extend_means(fixed_mean, candidate_mean, fit_matrix, collapsed_fit, nadditions)
    full_fit = np.append(fixed_mean, candidate_mean) * \
               fit_matrix.shape[0]
    covariance_extension -= np.outer(full_fit, collapsed_fit) / \
                            fit_matrix.shape[0]

    covariance_matrix = np.column_stack((covariance_matrix,
                                         covariance_extension[:-nadditions]))
    covariance_matrix = np.row_stack((covariance_matrix,
                                      covariance_extension.T))
    for j in range(1, nadditions + 1):
        covariance_matrix[-j, -j] += 1e-8

    return fixed_mean, candidate_mean, covariance_matrix


def update_covariance_matrix(covariance_matrix,
                             update,
                             fit_matrix,
                             indices,
                             fixed_mean,
                             update_mean,
                             candidate_mean) -> np.ndarray:
    covariance_addition = np.zeros_like(covariance_matrix[-1, :])
    covariance_addition[:-1] += np.tensordot(update,
                                             fit_matrix[indices,
                                             :-1] - fixed_mean,
                                             axes=[[0], [0]])
    covariance_addition[-1] += np.tensordot(
        fit_matrix[indices, -1] - update,
        update - update_mean,
        axes=[[0], [0]]
    )
    covariance_addition[-1] += np.tensordot(
        update,
        fit_matrix[indices, -1] - candidate_mean,
        axes=[[0], [0]]
    )

    covariance_matrix[-1, :-1] += covariance_addition[:-1]
    covariance_matrix[:, -1] += covariance_addition

    return covariance_addition


def decompose_addition(covariance_addition: np.ndarray) -> tuple[list[float32],
list[np.ndarray]]:
    eigenvalue_intermediate = np.sqrt(
        covariance_addition[-1] ** 2 + 4 * np.sum(covariance_addition[:-1] ** 2))
    eigenvalues = [
        (covariance_addition[-1] + eigenvalue_intermediate) / 2,
        (covariance_addition[-1] - eigenvalue_intermediate) / 2,
    ]
    eigenvectors = [
        np.array([*(covariance_addition[:-1] / eigenvalues[0]), 1]),
        np.array([*(covariance_addition[:-1] / eigenvalues[1]), 1]),
    ]
    eigenvectors[0] /= np.linalg.norm(eigenvectors[0])
    eigenvectors[1] /= np.linalg.norm(eigenvectors[1])

    return eigenvalues, eigenvectors


def calculate_right_hand_side(fit_matrix, y: np.ndarray):
    y_mean = np.mean(y)
    right_hand_side = fit_matrix.T @ (y - y_mean)
    return y_mean, right_hand_side


def extend_right_hand_side(right_hand_side, fit_matrix, y_mean, y: np.ndarray, nadditions: int32):
    right_hand_side = np.append(right_hand_side,
                                fit_matrix[:, -nadditions:].T @ (
                                        y - y_mean))
    return right_hand_side


def update_right_hand_side(right_hand_side, y: np.ndarray, indices, y_mean, update) -> None:
    right_hand_side[-1] += np.sum(
        update * (y[indices] - y_mean))


def generalised_cross_validation(fit_matrix, coefficients, nbases, smoothness, y: np.ndarray):
    y_pred = fit_matrix @ coefficients
    mse = np.sum((y - y_pred) ** 2)  # rhs instead of y?

    c_m = nbases + 1 + smoothness * (nbases - 1)
    lof = mse / len(y) / (1 - c_m / len(y)) ** 2
    return lof


def fit(x: np.ndarray, y: np.ndarray,
        nbases, nodes, covariates, hinges, where, smoothness):
    fit_matrix = calculate_fit_matrix(x, nbases, nodes, covariates, hinges, where)
    fixed_mean, candidate_mean, covariance_matrix = calculate_covariance_matrix(fit_matrix)
    y_mean, right_hand_side = calculate_right_hand_side(fit_matrix, y)

    chol, lower = cho_factor(covariance_matrix, lower=True)

    coefficients = cho_solve((chol, lower), right_hand_side)

    lof = generalised_cross_validation(fit_matrix, coefficients, nbases, smoothness, y)

    return (np.tril(chol),
            coefficients,
            fit_matrix,
            fixed_mean,
            candidate_mean,
            y_mean,
            covariance_matrix,
            right_hand_side, lof)


def extend_fit(x: np.ndarray, y: np.ndarray, nadditions: int32,
               nbases, fit_matrix, covariates, nodes, hinges, where, covariance_matrix, right_hand_side, smoothness,
               y_mean, fixed_mean, candidate_mean):
    fit_matrix = extend_fit_matrix(x, nadditions, nbases, fit_matrix, covariates, nodes, hinges, where)
    fixed_mean, candidate_mean, covariance_matrix = extend_covariance_matrix(nadditions, fit_matrix, covariance_matrix,
                                                                             fixed_mean, candidate_mean)
    right_hand_side = extend_right_hand_side(right_hand_side, fit_matrix, y_mean, y, nadditions)

    chol, lower = cho_factor(covariance_matrix, lower=True)

    coefficients = cho_solve((chol, lower), right_hand_side)

    lof = generalised_cross_validation(fit_matrix, coefficients, nbases, smoothness, y)

    return np.tril(
        chol), coefficients, fit_matrix, fixed_mean, candidate_mean, covariance_matrix, right_hand_side, lof


def update_fit(x: np.ndarray, y: np.ndarray, chol: np.ndarray,
               old_node: float32,
               parent_idx: int32,
               nbases,
               covariates,
               nodes,
               where,
               candidate_mean, covariance_matrix, fixed_mean, right_hand_side, y_mean, smoothness, fit_matrix):
    indices, update, update_mean, candidate_mean = update_init(nbases, covariates, nodes, where, fit_matrix,
                                                               candidate_mean, x, old_node, parent_idx)
    update_fit_matrix(fit_matrix, indices, update)
    covariance_addition = update_covariance_matrix(covariance_matrix, update, fit_matrix, indices, fixed_mean,
                                                   update_mean, candidate_mean)

    if covariance_addition.any():
        eigenvalues, eigenvectors = decompose_addition(covariance_addition)
        chol = update_cholesky(chol, eigenvectors, eigenvalues)

    update_right_hand_side(right_hand_side, y, indices, y_mean, update)

    coefficients = cho_solve((chol, True), right_hand_side)

    lof = generalised_cross_validation(fit_matrix, coefficients, nbases, smoothness, y)

    return chol, coefficients, fit_matrix, fixed_mean, candidate_mean, covariance_matrix, right_hand_side, lof


def shrink_fit(x: np.ndarray, y: np.ndarray,
               removal_slice: slice,
               fit_matrix, covariance_matrix, fixed_mean, candidate_mean, right_hand_side, nbases,
               smoothness):
    fit_matrix = np.delete(fit_matrix, removal_slice, axis=1)
    covariance_matrix = np.delete(covariance_matrix, removal_slice,
                                  axis=0)
    covariance_matrix = np.delete(covariance_matrix, removal_slice,
                                  axis=1)
    fixed_mean, candidate_mean = shrink_means(fixed_mean, candidate_mean, removal_slice)

    right_hand_side = np.delete(right_hand_side, removal_slice)

    chol, lower = cho_factor(covariance_matrix, lower=True)

    coefficients = cho_solve((chol, True), right_hand_side)

    lof = generalised_cross_validation(fit_matrix, coefficients, nbases, smoothness, y)

    return chol, coefficients, fit_matrix, fixed_mean, candidate_mean, covariance_matrix, right_hand_side, lof


def expand_bases(x: np.ndarray, y: np.ndarray, nbases, nodes, covariates, hinges, where, smoothness,
                 max_nbases, max_ncandidates, aging_factor):
    all_covariates = set(range(x.shape[1]))
    chol, coefficients, fit_matrix, fixed_mean, candidate_mean, y_mean, covariance_matrix, right_hand_side, lof = fit(x,
                                                                                                                      y,
                                                                                                                      nbases,
                                                                                                                      nodes,
                                                                                                                      covariates,
                                                                                                                      hinges,
                                                                                                                      where,
                                                                                                                      smoothness)
    candidate_queue = {i: 1. for i in range(nbases)}
    for _ in range((max_nbases - nbases) // 2):
        best_lof = np.inf
        best_covariate = None
        best_node = None
        best_hinge = None
        best_where = None
        for parent_idx in sorted(candidate_queue, key=candidate_queue.get)[
                          :-max_ncandidates - 1:-1]:
            eligible_covariates = all_covariates - set(
                covariates[where[:, parent_idx], parent_idx])
            basis_lof = np.inf
            parent_depth = np.sum(where[:, parent_idx])
            for cov in eligible_covariates:
                eligible_knots = x[
                    np.where(fit_matrix[:, parent_idx] > 0)[0], cov]
                eligible_knots[::-1].sort()
                additional_covariates = np.tile(covariates[:, parent_idx],
                                                (2, 1)).T
                additional_nodes = np.tile(nodes[:, parent_idx],
                                           (2, 1)).T  # not copied, issue?
                additional_hinges = np.tile(hinges[:, parent_idx], (2, 1)).T
                additional_where = np.tile(where[:, parent_idx], (2, 1)).T

                additional_covariates[parent_depth + 1, :] = cov
                additional_nodes[parent_depth + 1, 0] = 0.0
                additional_nodes[parent_depth + 1, 1] = eligible_knots[0]
                additional_hinges[parent_depth + 1, 0] = False
                additional_hinges[parent_depth + 1, 1] = True
                additional_where[parent_depth + 1, :] = True

                nbases = add_basis(
                    nbases,
                    additional_covariates,
                    additional_nodes,
                    additional_hinges,
                    additional_where,
                    covariates, nodes, hinges, where
                )
                chol, coefficients, fix_matrix, fixed_mean, candidate_mean, covariance_matrix, right_hand_side, lof = extend_fit(
                    x, y, 2, nbases, fit_matrix, covariates, nodes, hinges, where, covariance_matrix,
                    right_hand_side, smoothness, y_mean, fixed_mean, candidate_mean)
                old_node = eligible_knots[0]

                for new_node in eligible_knots[1:]:
                    updated_nodes = additional_nodes[:, 1]
                    updated_nodes[parent_depth + 1] = new_node
                    update_basis(
                        nbases,
                        additional_covariates[:, 1],
                        updated_nodes,
                        additional_hinges[:, 1],
                        additional_where[:, 1],
                        covariates,
                        nodes,
                        hinges,
                        where
                    )
                    chol, coefficients, fit_matrix, fixed_mean, candidate_mean, covariance_matrix, right_hand_side, lof = update_fit(
                        x, y, chol, old_node, parent_idx, nbases, covariates, nodes, where, candidate_mean,
                        covariance_matrix, fixed_mean, right_hand_side, y_mean, smoothness, fit_matrix)
                    old_node = new_node
                    if lof < basis_lof:
                        basis_lof = lof
                    if lof < best_lof:
                        best_lof = lof
                        addition_slice = slice(nbases - 2, nbases)
                        best_covariate = covariates[:, addition_slice].copy()
                        best_node = nodes[:, addition_slice].copy()
                        best_hinge = hinges[:, addition_slice].copy()
                        best_where = where[:, addition_slice].copy()
                removal_slice = slice(nbases - 2, nbases)
                nbases = remove_basis(nbases, where, removal_slice)
                chol, coefficients, fit_matrix, fixed_mean, candidate_mean, covariance_matrix, right_hand_side, lof = shrink_fit(
                    x, y, removal_slice, fit_matrix, covariance_matrix, fixed_mean, candidate_mean, right_hand_side,
                    nbases, smoothness)
            candidate_queue[parent_idx] = best_lof - basis_lof
        for unselected_idx in sorted(candidate_queue, key=candidate_queue.get)[
                              max_ncandidates:]:
            candidate_queue[unselected_idx] += aging_factor
        if best_covariate is not None:
            nbases = add_basis(nbases, best_covariate, best_node, best_hinge, best_where, covariates, nodes, hinges,
                               where)
            chol, coefficients, fit_matrix, fixed_mean, candidate_mean, covariance_matrix, right_hand_side, lof = extend_fit(
                x, y, 2, nbases, fit_matrix, covariates, nodes, hinges, where, covariance_matrix, right_hand_side,
                smoothness, y_mean, fixed_mean, candidate_mean)
            for i in range(2):
                candidate_queue[len(candidate_queue)] = 0

    return nbases, covariates, nodes, hinges, where, coefficients, lof, fit_matrix, covariance_matrix, right_hand_side, fixed_mean, candidate_mean, y_mean


def prune_bases(x: np.ndarray, y: np.ndarray, nbases, covariates, nodes, hinges, where, coefficients, lof, fit_matrix,
                covariance_matrix, right_hand_side, fixed_mean, candidate_mean, y_mean, smoothness):
    best_nbases = nbases
    best_nodes = nodes.copy()
    best_covariates = covariates.copy()
    best_hinges = hinges.copy()
    best_where = where.copy()

    best_trimmed_nbases = nbases
    best_trimmed_nodes = nodes.copy()
    best_trimmed_covariates = covariates.copy()
    best_trimmed_hinges = hinges.copy()
    best_trimmed_where = where.copy()

    best_lof = lof

    while nbases > 1:
        best_trimmed_lof = np.inf

        previous_nbases = nbases
        previous_nodes = nodes.copy()
        previous_covariates = covariates.copy()
        previous_hinges = hinges.copy()
        previous_where = where.copy()
        previous_fit = fit_matrix.copy()
        previous_covariance = covariance_matrix.copy()
        previous_right_hand_side = right_hand_side.copy()
        previous_fixed_mean = fixed_mean.copy()
        previous_candidate_mean = candidate_mean

        # First basis function (constant 1) cannot be excluded
        for basis_idx in range(1, nbases):
            removal_slice = slice(basis_idx, basis_idx + 1)
            nbases = remove_basis(nbases, where, removal_slice)
            chol, coefficients, fit_matrix, fixed_mean, candidate_mean, covariance_matrix, right_hand_side, lof = shrink_fit(
                x, y, removal_slice, fit_matrix, covariance_matrix, fixed_mean, candidate_mean, right_hand_side,
                nbases, smoothness)
            if lof < best_trimmed_lof:
                best_trimmed_lof = lof
                best_trimmed_nbases = nbases
                best_trimmed_nodes = nodes.copy()
                best_trimmed_covariates = covariates.copy()
                best_trimmed_hinges = hinges.copy()
                best_trimmed_where = where.copy()
            if lof < best_lof:
                best_lof = lof
                best_nbases = nbases
                best_nodes = nodes.copy()
                best_covariates = covariates.copy()
                best_hinges = hinges.copy()
                best_where = where.copy()
            nbases = previous_nbases
            nodes = previous_nodes.copy()
            covariates = previous_covariates.copy()
            hinges = previous_hinges.copy()
            previous_where = previous_where.copy()
            fit_matrix = previous_fit.copy()
            covariance_matrix = previous_covariance.copy()
            right_hand_side = previous_right_hand_side.copy()
            fixed_mean = previous_fixed_mean.copy()
            candidate_mean = previous_candidate_mean

        nbases = best_trimmed_nbases
        nodes = best_trimmed_nodes.copy()
        covariates = best_trimmed_covariates.copy()
        hinges = best_trimmed_hinges.copy()
        where = best_trimmed_where.copy()
        chol, coefficients, fit_matrix, fixed_mean, candidate_mean, y_mean, covariance_matrix, right_hand_side, lof = fit(
            x,
            y,
            nbases,
            nodes,
            covariates,
            hinges,
            where,
            smoothness)
    nbases = best_nbases
    nodes = best_nodes.copy()
    covariates = best_covariates.copy()
    hinges = best_hinges.copy()
    where = best_where.copy()
    chol, coefficients, fit_matrix, fixed_mean, candidate_mean, y_mean, covariance_matrix, right_hand_side, lof = fit(
        x,
        y,
        nbases,
        nodes,
        covariates,
        hinges,
        where,
        smoothness)
    return nbases, covariates, nodes, hinges, where, coefficients


def find_bases(x: np.ndarray, y: np.ndarray, max_nbases: int32, max_ncandidates: int32, aging_factor: float32,
               covariates, nodes, hinges, where, smoothness, nbases):
    nbases, covariates, nodes, hinges, where, coefficients, lof, fit_matrix, covariance_matrix, right_hand_side, fixed_mean, candidate_mean, y_mean = expand_bases(
        x, y, nbases, nodes, covariates, hinges, where, smoothness, max_nbases, max_ncandidates, aging_factor)
    nbases, covariates, nodes, hinges, where, coefficients = prune_bases(x, y, nbases, covariates, nodes, hinges, where,
                                                                         coefficients, lof, fit_matrix,
                                                                         covariance_matrix,
                                                                         right_hand_side, fixed_mean, candidate_mean,
                                                                         y_mean, smoothness)
    return nbases, covariates, nodes, hinges, where, coefficients


if __name__ == "__main__":
    max_nbases = 11
    max_ncandidates = 5
    aging_factor = 0.
    smoothness = 3

    # Bases
    nbases = 1
    covariates = np.zeros((max_nbases, max_nbases), dtype=int32)
    nodes = np.zeros((max_nbases, max_nbases), dtype=float32)
    hinges = np.zeros((max_nbases, max_nbases), dtype=bool_)
    where = np.zeros((max_nbases, max_nbases), dtype=bool_)

    from tests.utils import data_generation_model

    x, y, y_true, reference_model = data_generation_model(100, 2)

    nbases, covariates, nodes, hinges, where, coefficients = find_bases(x, y, max_nbases, max_ncandidates, aging_factor,
                                                                            covariates, nodes, hinges, where, smoothness,
                                                                            nbases)
    print_bases(nbases, max_nbases, nodes, covariates, hinges, where)
    print("Coefficients: ", coefficients)
