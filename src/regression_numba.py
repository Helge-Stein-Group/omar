from typing import Self

import numpy as np
from numba import njit, prange


@njit(cache=True, error_model="numpy", fastmath=True, parallel=True)
def data_matrix(x: np.ndarray,
                basis_start: int,
                basis_end: int,
                covariates: np.ndarray,
                nodes: np.ndarray,
                hinges: np.ndarray,
                where: np.ndarray) -> np.ndarray:
    """
    Calculate the data matrix for the given data and basis, which is the
    evaluation of the basis for the data points.

    Args:
        x: Data points. [n x d]
        basis_start: First basis to evaluate.
        basis_end: Last basis to evaluate.
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        hinges: Hinges of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]

    Returns:
        Data matrix.
    """
    n_samples = x.shape[0]
    n_basis = basis_end - basis_start
    result = np.ones((n_samples, n_basis))
    for i in prange(n_basis):
        basis_idx = i + basis_start
        for func_idx in range(where.shape[0]):
            if where[func_idx, basis_idx]:
                intermediate_result = -nodes[func_idx, basis_idx] + x[:, covariates[func_idx, basis_idx]]
                if hinges[func_idx, basis_idx]:
                    intermediate_result = np.maximum(0, intermediate_result)
                result[:, i] *= intermediate_result
    return result


@njit(cache=True, error_model="numpy", fastmath=True, parallel=False)
def update_cholesky(chol: np.ndarray,
                    update_vectors: list[np.ndarray],
                    multipliers: list[float]) -> np.ndarray:
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

        for i in prange(chol.shape[0]):
            chol[i, i] = np.sqrt(diag[i] + multiplier / b[i] * u[i, i] ** 2)
            chol[i + 1:, i] *= chol[i, i]
            chol[i + 1:, i] += multiplier / b[i] * u[i, i] * u[i, i + 1:] / chol[i, i]

    return chol


@njit(cache=True, error_model="numpy", fastmath=True)
def add_bases(nbases,
              new_covariates: np.ndarray,
              new_nodes: np.ndarray,
              new_hinges: np.ndarray,
              new_where: np.ndarray,
              covariates: np.ndarray,
              nodes: np.ndarray,
              hinges: np.ndarray,
              where: np.ndarray) -> int:
    """
    Add a basis to the model.

    Args:
        nbases: Number of basis functions.
        new_covariates: Covariates of the basis. [max_nbases x nadditions]
        new_nodes: Nodes of the basis. [max_nbases x nadditions]
        new_hinges: Hinges of the basis. [max_nbases x nadditions]
        new_where: Signals product length of basis. [max_nbases x nadditions]
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        hinges: Hinges of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]

    Returns:
        New number of basis functions.
    """
    addition_slice = slice(nbases, nbases + new_covariates.shape[1])
    covariates[:, addition_slice] = new_covariates
    nodes[:, addition_slice] = new_nodes
    hinges[:, addition_slice] = new_hinges
    where[:, addition_slice] = new_where
    nbases += new_covariates.shape[1]

    return nbases


@njit(cache=True, error_model="numpy", fastmath=True)
def update_basis(nbases,
                 new_covariates: np.ndarray,
                 new_nodes: np.ndarray,
                 new_hinges: np.ndarray,
                 new_where: np.ndarray,
                 covariates: np.ndarray,
                 nodes: np.ndarray,
                 hinges: np.ndarray,
                 where: np.ndarray) -> None:
    """
    Update the latest basis.

    Args:
        nbases: Number of basis functions.
        new_covariates: Covariates of the basis. [max_nbases x nadditions]
        new_nodes: Nodes of the basis. [max_nbases x nadditions]
        new_hinges: Hinges of the basis. [max_nbases x nadditions]
        new_where: Signals product length of basis. [max_nbases x nadditions]
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        hinges: Hinges of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]
    """
    covariates[:, nbases - 1] = new_covariates
    nodes[:, nbases - 1] = new_nodes
    hinges[:, nbases - 1] = new_hinges
    where[:, nbases - 1] = new_where


@njit(cache=True, error_model="numpy", fastmath=True)
def remove_bases(
        nbases,
        where: np.ndarray,
        removal_slice: slice) -> int:
    """
    Remove basis from the model.

    Args:
        nbases: Number of basis functions.
        where: Signals product length of basis. [max_nbases x max_nbases]
        removal_slice: Slice of the basis to be removed.

    Returns:
        New number of basis functions.

    Notes:
        Only the where attributes is turned off, the other attributes (e.g. covariates) are deliberately not
        deleted to avoid recalculation in the expansion step.
    """
    where[:, removal_slice] = False
    nbases -= removal_slice.stop - removal_slice.start

    return nbases


@njit(cache=True, error_model="numpy", fastmath=True)
def calculate_means(fit_matrix: np.ndarray,
                    collapsed_fit: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Calculate and store the means of the fixed and candidate basis.

    Args:
        fit_matrix: Evaluation of the bases on the data. [n x nbases]
        collapsed_fit: Sum of the fit matrix. [nbases]

    Returns:
        Mean of the fixed and candidate basis.
    """
    fixed_mean = collapsed_fit[:-1] / fit_matrix.shape[0]
    candidate_mean = collapsed_fit[-1] / fit_matrix.shape[0]

    return fixed_mean, candidate_mean


@njit(cache=True, error_model="numpy", fastmath=True)
def extend_means(fit_matrix: np.ndarray,
                 collapsed_fit: np.ndarray,
                 fixed_mean: np.ndarray,
                 candidate_mean: float) -> tuple[np.ndarray, float]:
    """
    Extend the means of the fixed and candidate basis
    by accounting for newly added bases.

    Args:
        fit_matrix: Evaluation of the bases on the data. [n x nbases]
        collapsed_fit: Sum of the fit matrix. [nadditions]
        fixed_mean: Mean of the previous fixed basis. [nbases-1-nadditions]
        candidate_mean: Mean of the previous candidate basis.

    Returns:
        Fixed and candidate mean of the extended basis.
    """
    fixed_mean = np.append(fixed_mean, candidate_mean)
    fixed_mean = np.append(fixed_mean,
                           collapsed_fit[:-1] / fit_matrix.shape[0])
    candidate_mean = collapsed_fit[-1] / fit_matrix.shape[0]
    return fixed_mean, candidate_mean


@njit(cache=True, error_model="numpy", fastmath=True)
def shrink_means(fixed_mean: np.ndarray,
                 candidate_mean: float,
                 removal_slice: slice) -> tuple[np.ndarray, float]:
    """
    Shrink the means of the fixed and candidate basis 
    accounting for removed bases.

    Args:
        fixed_mean: Mean of the previous fixed basis. [nbases-1+nremovals]
        candidate_mean: Mean of the previous candidate basis.
        removal_slice: Slice of the basis to be removed.

    Returns:
        Fixed and candidate mean of the shrunk basis.
    """
    joint_mean = np.append(fixed_mean, candidate_mean)
    reduced_mean = np.delete(joint_mean, removal_slice)

    fixed_mean = reduced_mean[:-1]
    candidate_mean = reduced_mean[-1]

    return fixed_mean, candidate_mean


@njit(cache=True, error_model="numpy", fastmath=True)
def update_init(
        x: np.ndarray,
        old_node: float,
        parent_idx: int,
        nbases: int,
        covariates: np.ndarray,
        nodes: np.ndarray,
        where: np.ndarray,
        fit_matrix: np.ndarray,
        candidate_mean: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Initialize the update of the fit by precomputing the necessary update values, 
    that allow for a fast update of the cholesky decomposition and therefore a 
    faster least-squares fit.

    Args:
        x: Data points. [n x d]
        old_node: Previous node of the basis function.
        parent_idx: Index of the parent basis function.
        nbases: Number of basis functions.
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]
        fit_matrix: Evaluation of the bases on the data. [n x nbases]
        candidate_mean: Mean of the candidate basis.

    Returns:
        Indices of the data points that need to be updated.
        Update values of the data points.
        Mean of the updates.
        Mean of the candidate basis.
    """
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


@njit(cache=True, error_model="numpy", fastmath=True)
def calculate_fit_matrix(x: np.ndarray,
                         nbases: int,
                         covariates: np.ndarray,
                         nodes: np.ndarray,
                         hinges: np.ndarray,
                         where) -> np.ndarray:
    """
    Calculate the data matrix for the given data.

    Args:
        x: Data points. [n x d]
        nbases: Number of basis functions.
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        hinges: Hinges of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]

    Returns:
        Fit matrix. [n x nbases]
    """
    fit_matrix = data_matrix(x, 0, nbases, covariates, nodes, hinges, where)

    return fit_matrix


@njit(cache=True, error_model="numpy", fastmath=True)
def extend_fit_matrix(x: np.ndarray,
                      nadditions: int,
                      nbases: int,
                      fit_matrix: np.ndarray,
                      covariates: np.ndarray,
                      nodes: np.ndarray,
                      hinges: np.ndarray,
                      where: np.ndarray) -> np.ndarray:
    """
    Extend the data matrix by evaluating the newly added basis functions for the data points.

    Args:
        x: Data points. [n x d]
        nadditions: Number of newly added basis functions.
        nbases: Number of basis functions.
        fit_matrix: Evaluation of the bases on the data. [n x nbases-nadditions]
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        hinges: Hinges of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]

    Returns:
        Extended fit matrix. [n x nbases]
    """
    fit_matrix = np.column_stack((
        fit_matrix,
        data_matrix(x, nbases - nadditions, nbases, covariates, nodes, hinges, where)
    ))
    return fit_matrix


@njit(cache=True, error_model="numpy", fastmath=True)
def update_fit_matrix(fit_matrix: np.ndarray,
                      indices: np.ndarray,
                      update: np.ndarray) -> None:
    """
    Update the data matrix by adding the update attribute to the last column.

    Args:
        fit_matrix: Evaluation of the bases on the data. [n x nbases]
        indices: Indices of the data points that need to be updated.
        update: Update values of the data points.
    """
    fit_matrix[indices, -1] += update


@njit(cache=True, error_model="numpy", fastmath=True)
def calculate_covariance_matrix(fit_matrix: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, float]:
    """
    Calculate the covariance matrix of the data matrix,
    which is the chosen formulation of the least-squares problem.

    Args:
        fit_matrix: Evaluation of the bases on the data. [n x nbases]

    Returns:
        Covariance matrix.
        Mean of the fixed basis.
        Mean of the candidate basis.
    """
    covariance_matrix = fit_matrix.T @ fit_matrix
    collapsed_fit = np.sum(fit_matrix, axis=0)
    fixed_mean, candidate_mean = calculate_means(fit_matrix, collapsed_fit)
    covariance_matrix -= np.outer(collapsed_fit, collapsed_fit) / \
                         fit_matrix.shape[0]

    covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-8
    return covariance_matrix, fixed_mean, candidate_mean,


@njit(cache=True, error_model="numpy", fastmath=True)
def extend_covariance_matrix(covariance_matrix: np.ndarray,
                             nadditions: int,
                             fit_matrix: np.ndarray,
                             fixed_mean: np.ndarray,
                             candidate_mean: float) -> tuple[
    np.ndarray, np.ndarray, float]:
    """
    Extend the covariance matrix by accounting for the newly added basis functions, expects and
    extended fit matrix.

    Args:
        covariance_matrix: Covariance matrix to extend. [nbases-nadditions x nbases-nadditions]
        nadditions: Number of newly added basis functions.
        fit_matrix: Evaluation of the bases on the data. [n x nbases]
        fixed_mean: Mean of the fixed basis. [nbases-1-nadditions]
        candidate_mean: Mean of the candidate basis.

    Returns:
        Extended covariance matrix. [nbases x nbases]
        Mean of the fixed basis. [nbases]
        Mean of the candidate basis.

    Notes:
        Also updates the means of the fixed and candidate basis. Done here to avoid
        recalculation of the sum.
    """
    covariance_extension = fit_matrix.T @ fit_matrix[:, -nadditions:]
    collapsed_fit = np.sum(fit_matrix[:, -nadditions:], axis=0)
    fixed_mean, candidate_mean = extend_means(fit_matrix,
                                              collapsed_fit,
                                              fixed_mean,
                                              candidate_mean)
    full_fit = np.append(fixed_mean, candidate_mean) * fit_matrix.shape[0]
    covariance_extension -= np.outer(full_fit, collapsed_fit) / fit_matrix.shape[0]

    covariance_matrix = np.column_stack((covariance_matrix,
                                         covariance_extension[:-nadditions]))
    covariance_matrix = np.row_stack((covariance_matrix,
                                      covariance_extension.T))
    for j in range(1, nadditions + 1):
        covariance_matrix[-j, -j] += 1e-8

    return covariance_matrix, fixed_mean, candidate_mean


@njit(cache=True, error_model="numpy", fastmath=True)
def update_covariance_matrix(covariance_matrix: np.ndarray,
                             update: np.ndarray,
                             fit_matrix: np.ndarray,
                             indices: np.ndarray,
                             fixed_mean: np.ndarray,
                             update_mean: np.ndarray,
                             candidate_mean: float) -> np.ndarray:
    """
    Update the covariance matrix.
    Expects update initialisation and an updated fit matrix.

    Args:
        covariance_matrix: Covariance matrix to update. [nbases x nbases]
        update: Update values of the data points.
        fit_matrix: Evaluation of the bases on the data. [n x nbases]
        indices: Indices of the data points that need to be updated.
        fixed_mean: Mean of the fixed basis. [nbases-1]
        update_mean: Mean of the updates.
        candidate_mean: Mean of the candidate basis.

    Returns:
        Covariance addition: Vector that was used to update the covariance matrix.
    """
    covariance_addition = np.zeros(covariance_matrix.shape[0], dtype=float)
    covariance_addition[:-1] += np.sum(update[np.newaxis, ...] @ (fit_matrix[indices, :-1] - fixed_mean), axis=0)
    covariance_addition[-1] += (fit_matrix[indices, -1] - update) @ (update - update_mean)
    covariance_addition[-1] += update @ (fit_matrix[indices, -1] - candidate_mean)

    covariance_matrix[-1, :-1] += covariance_addition[:-1]
    covariance_matrix[:, -1] += covariance_addition

    return covariance_addition


@njit(cache=True, error_model="numpy", fastmath=True, parallel=False)
def decompose_addition(covariance_addition: np.ndarray) \
        -> tuple[list[float], list[np.ndarray]]:
    """
    Decompose the addition to the covariance matrix,
    which was done by adding to the last row and column of the matrix,
    into eigenvalues and eigenvectors to perform 2 rank-1 updates.

    Args:
        covariance_addition: Addition to the covariance matrix.
        (the same vector is applied to the row and column) [nbases]

    Returns:
        Eigenvalues and eigenvectors of the addition.
    """
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


@njit(cache=True, error_model="numpy", fastmath=True)
def calculate_right_hand_side(y: np.ndarray,
                              fit_matrix: np.ndarray) -> tuple[np.ndarray, float]:
    """
   Calculate the right hand side of the least-squares problem.

   Args:
       y: Target values. [n]
       fit_matrix: Evaluation of the bases on the data. [n x nbases]

   Returns:
       Right hand side of the least-squares problem. [nbases]
       Mean of the target values.
   """
    y_mean = np.mean(y)
    right_hand_side = fit_matrix.T @ (y - y_mean)
    return right_hand_side, y_mean


@njit(cache=True, error_model="numpy", fastmath=True)
def extend_right_hand_side(right_hand_side: np.ndarray,
                           y: np.ndarray,
                           fit_matrix: np.ndarray,
                           y_mean: float,
                           nadditions: int):
    """
    Extend the right hand side of the least-squares problem
    by accounting for the newly added basis functions.

    Args:
        right_hand_side: Right hand side of the least-squares problem. [nbases-nadditions]
        y: Target values. [n]
        fit_matrix: Evaluation of the bases on the data. [n x nbases]
        y_mean: Mean of the target values.
        nadditions: Number of newly added basis functions.

    Returns:
        Extended right hand side of the least-squares problem. [nbases]
    """
    right_hand_side = np.append(right_hand_side,
                                fit_matrix[:, -nadditions:].T @ (
                                        y - y_mean))
    return right_hand_side


@njit(cache=True, error_model="numpy", fastmath=True)
def update_right_hand_side(right_hand_side: np.ndarray,
                           y: np.ndarray,
                           y_mean: float,
                           indices: np.ndarray,
                           update: np.ndarray) -> None:
    """
    Update the right hand side according to the basis update.

    Args:
        right_hand_side: Right hand side of the least-squares problem. [nbases]
        y: Target values. [n]
        y_mean: Mean of the target values.
        indices: Indices of the data points that need to be updated.
        update: Update values of the data points.
    """
    right_hand_side[-1] += np.sum(
        update * (y[indices] - y_mean))


@njit(cache=True, error_model="numpy", fastmath=True)
def generalised_cross_validation(y: np.ndarray,
                                 fit_matrix: np.ndarray,
                                 coefficients: np.ndarray,
                                 nbases: int,
                                 smoothness: int) -> float:
    """
    Calculate the generalised cross validation criterion, the lack of fit criterion.

    Args:
        y: Target values. [n]
        fit_matrix: Evaluation of the bases on the data. [n x nbases]
        coefficients: Coefficients of the basis. [nbases]
        nbases: Number of basis functions.
        smoothness: Cost for each basis optimization.

    Returns:
        Generalised cross validation criterion.
    """
    y_pred = fit_matrix @ coefficients
    mse = np.sum((y - y_pred) ** 2)

    c_m = nbases + 1 + smoothness * (nbases - 1)
    lof = mse / len(y) / (1 - c_m / len(y)) ** 2
    return lof


@njit(cache=True, error_model="numpy", fastmath=True)
def cho_solve_numpy(chol: np.ndarray,
                    right_hand_side: np.ndarray) -> np.ndarray:
    """
    Solve the linear system Ax=b given the cholesky decomposition of A. Replacement for scipy.linalg.cho_solve

    Args:
        chol: Cholesky decomposition of A.
        right_hand_side: Right hand side b.

    Returns:
        Solution x.
    """
    # Forward substitution
    y = np.zeros_like(right_hand_side)
    for i in range(chol.shape[0]):
        y[i] = (right_hand_side[i] - chol[i, :i] @ y[:i]) / chol[i, i]

    # Backward substitution
    x = np.zeros_like(right_hand_side)
    for i in range(chol.shape[0] - 1, -1, -1):
        x[i] = (y[i] - chol[i + 1:, i] @ x[i + 1:]) / chol[i, i]

    return x


@njit(cache=True, error_model="numpy", fastmath=True)
def fit(x: np.ndarray,
        y: np.ndarray,
        nbases: int,
        covariates: np.ndarray,
        nodes: np.ndarray,
        hinges: np.ndarray,
        where: np.ndarray,
        smoothness: int) -> tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """
    Calculate the least-squares fit of the current model from scratch.

    Args:
        x: Data points. [n x d]
        y: Target values. [n]
        nbases: Number of basis functions.
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        hinges: Hinges of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]
        smoothness: Cost for each basis optimization.

    Returns:
        Generalised cross validation criterion.
        Coefficients of the basis. [nbases]
        Fit matrix. [n x nbases]
        Covariance matrix. [nbases x nbases]
        Cholesky decomposition of the covariance matrix. [nbases x nbases]
        Right hand side of the least-squares problem. [nbases]
        Mean of the fixed basis.
        Mean of the candidate basis.
        Mean of the target values.
    """
    fit_matrix = calculate_fit_matrix(x, nbases, covariates, nodes, hinges, where)
    covariance_matrix, fixed_mean, candidate_mean = calculate_covariance_matrix(
        fit_matrix)
    right_hand_side, y_mean = calculate_right_hand_side(y, fit_matrix)

    chol = np.linalg.cholesky(covariance_matrix)

    coefficients = cho_solve_numpy(chol, right_hand_side)

    lof = generalised_cross_validation(y, fit_matrix, coefficients, nbases, smoothness)

    return (lof,
            coefficients,
            fit_matrix,
            covariance_matrix,
            np.tril(chol),
            right_hand_side,
            fixed_mean,
            candidate_mean,
            y_mean)


@njit(cache=True, error_model="numpy", fastmath=True)
def extend_fit(x: np.ndarray,
               y: np.ndarray,
               nbases: int,
               covariates: np.ndarray,
               nodes: np.ndarray,
               hinges: np.ndarray,
               where: np.ndarray,
               smoothness: int,
               nadditions: int,
               fit_matrix: np.ndarray,
               covariance_matrix: np.ndarray,
               right_hand_side: np.ndarray,
               y_mean: float,
               fixed_mean: np.ndarray,
               candidate_mean: float) -> tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Extend the least-squares fit. Expects the correct fit matrix,
    covariance matrix and right hand side of the smaller model.

    Args:
        x: Data points. [n x d]
        y: Target values. [n]
        nbases: Number of basis functions.
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        hinges: Hinges of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]
        smoothness: Cost for each basis optimization.
        nadditions: Number of newly added basis functions.
        fit_matrix: Evaluation of the bases on the data. [n x nbases-nadditions]
        covariance_matrix: Covariance matrix of the smaller model. [nbases-nadditions x nbases-nadditions]
        right_hand_side: Right hand side of the smaller model. [nbases-nadditions]
        y_mean: Mean of the target values.
        fixed_mean: Mean of the fixed basis. [nbases-1-nadditions]
        candidate_mean: Mean of the previous candidate basis.

    Returns:
        Generalised cross validation criterion.
        Coefficients of the basis. [nbases]
        Fit matrix. [n x nbases]
        Covariance matrix. [nbases x nbases]
        Cholesky decomposition of the covariance matrix. [nbases x nbases]
        Right hand side of the least-squares problem. [nbases]
        Mean of the fixed basis. [nbases-1]
        Mean of the candidate basis.
    """
    fit_matrix = extend_fit_matrix(x, nadditions, nbases, fit_matrix, covariates, nodes,
                                   hinges, where)
    covariance_matrix, fixed_mean, candidate_mean = extend_covariance_matrix(
        covariance_matrix,
        nadditions,
        fit_matrix,
        fixed_mean,
        candidate_mean)
    right_hand_side = extend_right_hand_side(right_hand_side, y, fit_matrix, y_mean,
                                             nadditions)

    chol = np.linalg.cholesky(covariance_matrix)

    coefficients = cho_solve_numpy(chol, right_hand_side)

    lof = generalised_cross_validation(y, fit_matrix, coefficients, nbases, smoothness)

    return (lof,
            coefficients,
            fit_matrix,
            covariance_matrix,
            np.tril(chol),
            right_hand_side,
            fixed_mean,
            candidate_mean)


@njit(cache=True, error_model="numpy", fastmath=True)
def update_fit(x: np.ndarray,
               y: np.ndarray,
               nbases: int,
               covariates: np.ndarray,
               nodes: np.ndarray,
               where: np.ndarray,
               smoothness: int,
               fit_matrix: np.ndarray,
               covariance_matrix: np.ndarray,
               right_hand_side: np.ndarray,
               y_mean: float,
               fixed_mean: np.ndarray,
               candidate_mean: float,
               old_node: float,
               parent_idx: int,
               chol: np.ndarray) -> tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Update the least-squares fit. Expects the correct fit matrix, covariance matrix
    and right hand side of the previous model. The only change in the bases occurred
    in the node of the last basis function, which is smaller than the old node.

    Args:
        x: Data points. [n x d]
        y: Target values. [n]
        nbases: Number of basis functions.
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]
        smoothness: Cost for each basis optimization.
        fit_matrix: Evaluation of the bases on the data. [n x nbases]
        covariance_matrix: Covariance matrix of the model. [nbases x nbases]
        right_hand_side: Right hand side of the model. [nbases]
        y_mean: Mean of the target values.
        fixed_mean: Mean of the fixed basis. [nbases-1]
        candidate_mean: Mean of the candidate basis.
        old_node: Previous node of the basis function.
        parent_idx: Index of the parent basis function.
        chol: Cholesky decomposition of the covariance matrix. [nbases x nbases]

    Returns:
        Generalised cross validation criterion.
        Coefficients of the basis. [nbases]
        Fit matrix. [n x nbases]
        Covariance matrix. [nbases x nbases]
        Cholesky decomposition of the covariance matrix. [nbases x nbases]
        Right hand side of the least-squares problem. [nbases]
        Mean of the fixed basis. [nbases-1]
        Mean of the candidate basis.
    """
    indices, update, update_mean, candidate_mean = update_init(x, old_node, parent_idx,
                                                               nbases, covariates,
                                                               nodes, where, fit_matrix,
                                                               candidate_mean)
    update_fit_matrix(fit_matrix, indices, update)
    covariance_addition = update_covariance_matrix(covariance_matrix, update,
                                                   fit_matrix, indices, fixed_mean,
                                                   update_mean, candidate_mean)

    if covariance_addition.any():
        eigenvalues, eigenvectors = decompose_addition(covariance_addition)
        chol = update_cholesky(chol, eigenvectors, eigenvalues)

    update_right_hand_side(right_hand_side, y, y_mean, indices, update)

    coefficients = cho_solve_numpy(chol, right_hand_side)

    lof = generalised_cross_validation(y, fit_matrix, coefficients, nbases, smoothness)

    return (lof,
            coefficients,
            fit_matrix,
            covariance_matrix,
            np.tril(chol),
            right_hand_side,
            fixed_mean,
            candidate_mean)


@njit(cache=True, error_model="numpy", fastmath=True)
def delete(array: np.ndarray, removal_slice: slice, axis: int) -> np.ndarray:
    """
    Delete a slice from an array. Replacement for np.delete.
    Only supports 2D arrays. Only supports float arrays.

    Args:
        array: Array to delete from.
        removal_slice: Slice to remove.
        axis: Axis to remove from.

    Returns:
        Array with the slice removed.
    """
    new_shape = list(array.shape)
    new_shape[axis] -= removal_slice.stop - removal_slice.start

    trimmed_array = np.zeros((new_shape[0], new_shape[1]), dtype=float)
    if axis == 0:
        trimmed_array[:removal_slice.start] = array[:removal_slice.start, :]
        trimmed_array[removal_slice.start:] = array[removal_slice.stop:, :]
    elif axis == 1:
        trimmed_array[:, :removal_slice.start] = array[:, :removal_slice.start]
        trimmed_array[:, removal_slice.start:] = array[:, removal_slice.stop:]

    return trimmed_array


@njit(cache=True, error_model="numpy", fastmath=True)
def shrink_fit(y: np.ndarray,
               nbases: int,
               smoothness: int,
               removal_slice: slice,
               fit_matrix: np.ndarray,
               covariance_matrix: np.ndarray,
               right_hand_side: np.ndarray,
               fixed_mean: np.ndarray,
               candidate_mean: float
               ) -> tuple[
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Shrink the least-squares fit. Expects the correct fit matrix, covariance matrix
    and right hand side of the larger model.

    Args:
        y: Target values.  [n]
        nbases: Number of basis functions.
        smoothness: Cost for each basis optimization.
        removal_slice: Slice of the basis functions to be removed.
        fit_matrix: Evaluation of the bases on the data. [n x nbases+nremovals]
        covariance_matrix: Covariance matrix of the model. [nbases+nremovals x nbases+nremovals]
        right_hand_side: Right hand side of the model. [nbases+nremovals]
        fixed_mean: Mean of the fixed basis. [nbases-1+nremovals]
        candidate_mean: Mean of the candidate basis.

    Returns:
        Generalised cross validation criterion.
        Coefficients of the basis. [nbases]
        Fit matrix. [n x nbases]
        Covariance matrix. [nbases x nbases]
        Cholesky decomposition of the covariance matrix. [nbases x nbases]
        Right hand side of the least-squares problem. [nbases]
        Mean of the fixed basis. [nbases-1]
        Mean of the candidate basis.
    """
    fit_matrix = delete(fit_matrix, removal_slice, axis=1)
    covariance_matrix = delete(covariance_matrix, removal_slice, axis=0)
    covariance_matrix = delete(covariance_matrix, removal_slice, axis=1)
    fixed_mean, candidate_mean = shrink_means(fixed_mean, candidate_mean, removal_slice)

    right_hand_side = np.delete(right_hand_side, removal_slice)

    chol = np.linalg.cholesky(covariance_matrix)

    coefficients = cho_solve_numpy(chol, right_hand_side)

    lof = generalised_cross_validation(y, fit_matrix, coefficients, nbases, smoothness)

    return (lof,
            coefficients,
            fit_matrix,
            covariance_matrix,
            np.tril(chol),
            right_hand_side,
            fixed_mean,
            candidate_mean)


@njit(cache=True, error_model="numpy", fastmath=True, parallel=False)
def setdiff1d(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Set difference of two arrays. Replacement for np.setdiff1d.

    Args:
        a: First array.
        b: Second array.

    Returns:
        Set difference of the two arrays.
    """
    result = []
    for i in a:
        if i not in b:
            result.append(i)
    return np.array(result)


@njit(cache=True, error_model="numpy", fastmath=True)
def tile(array: np.ndarray) -> np.ndarray:
    """
    Tile an array.

    Args:
        array: Array to tile.

    Returns:
        Tiled array.
    """
    array = list(array)
    return np.array([array, array]).T


@njit(cache=True, error_model="numpy", fastmath=True, parallel=False)
def expand_bases(x: np.ndarray,
                 y: np.ndarray,
                 max_nbases: int,
                 smoothness: int,
                 max_ncandidates: int,
                 aging_factor: float) -> tuple[
    int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray,
    float, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Expand the bases to the maximum allowed number of basis functions. By iteratively adding the basis
    that reduces the lack of fit criterion the most. Equivalent to the forward pass in the Mars paper, including
    the adaptions in the Fast Mars paper.

    Args:
        x: Data points. [n x d]
        y: Target values. [n]
        max_nbases: Maximum number of basis functions.
        smoothness: Cost for each basis optimization.
        max_ncandidates: Maximum number of candidates to consider.
        aging_factor: Factor to age the candidates.

    Returns:
        Number of basis functions.
        Covariates of the basis. [max_nbases x max_nbases]
        Nodes of the basis. [max_nbases x max_nbases]
        Hinges of the basis. [max_nbases x max_nbases]
        Signals product length of basis. [max_nbases x max_nbases]
        Coefficients of the basis. [nbases]
        Generalised cross validation criterion.
        Fit matrix. [n x nbases]
        Covariance matrix. [nbases x nbases]
        Cholesky decomposition of the covariance matrix. [nbases x nbases]
        Right hand side of the least-squares problem. [nbases]
        Mean of the fixed basis.
        Mean of the candidate basis.
    """
    nbases = 1
    covariates = np.zeros((max_nbases, max_nbases), dtype=np.int32)
    nodes = np.zeros((max_nbases, max_nbases), dtype=float)
    hinges = np.zeros((max_nbases, max_nbases), dtype=np.bool_)
    where = np.zeros((max_nbases, max_nbases), dtype=np.bool_)

    all_covariates = np.array(list(range(x.shape[1])))
    (lof, coefficients, fit_matrix, covariance_matrix, chol, right_hand_side,
     fixed_mean, candidate_mean, y_mean) = fit(x, y, nbases, covariates, nodes, hinges,
                                               where, smoothness)
    candidate_queue = np.zeros(nbases, dtype=float)
    for _ in range((max_nbases - nbases) // 2):
        best_lof = np.inf
        best_covariate = None
        best_node = None
        best_hinge = None
        best_where = None
        for parent_idx in np.argsort(candidate_queue)[:-max_ncandidates - 1:-1]:
            eligible_covariates = setdiff1d(all_covariates, covariates[where[:, parent_idx], parent_idx])
            basis_lof = np.inf
            parent_depth = np.sum(where[:, parent_idx])
            for cov in eligible_covariates:
                eligible_knots = x[
                    np.where(fit_matrix[:, parent_idx] > 0)[0], cov]
                eligible_knots[::-1].sort()
                additional_covariates = tile(covariates[:, parent_idx])
                additional_nodes = tile(nodes[:, parent_idx])
                additional_hinges = tile(hinges[:, parent_idx])
                additional_where = tile(where[:, parent_idx])

                additional_covariates[parent_depth + 1, :] = cov
                additional_nodes[parent_depth + 1, 0] = 0.0
                additional_nodes[parent_depth + 1, 1] = eligible_knots[0]
                additional_hinges[parent_depth + 1, 0] = False
                additional_hinges[parent_depth + 1, 1] = True
                additional_where[parent_depth + 1, :] = True

                nbases = add_bases(
                    nbases,
                    additional_covariates,
                    additional_nodes,
                    additional_hinges,
                    additional_where,
                    covariates, nodes, hinges, where
                )
                (
                    lof, coefficients, fit_matrix, covariance_matrix, chol,
                    right_hand_side,
                    fixed_mean, candidate_mean) = extend_fit(
                    x, y, nbases, covariates, nodes, hinges, where, smoothness, 2,
                    fit_matrix, covariance_matrix, right_hand_side, y_mean, fixed_mean,
                    candidate_mean)
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
                    (lof, coefficients, fit_matrix, covariance_matrix, chol,
                     right_hand_side, fixed_mean, candidate_mean) = update_fit(
                        x, y, nbases, covariates, nodes, where, smoothness, fit_matrix,
                        covariance_matrix, right_hand_side, y_mean, fixed_mean,
                        candidate_mean, old_node, parent_idx, chol)
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
                nbases = remove_bases(nbases, where, removal_slice)
                (
                    lof, coefficients, fit_matrix, covariance_matrix, chol,
                    right_hand_side,
                    fixed_mean, candidate_mean) = shrink_fit(
                    y, nbases, smoothness, removal_slice, fit_matrix, covariance_matrix,
                    right_hand_side, fixed_mean, candidate_mean)

            candidate_queue[parent_idx] = best_lof - basis_lof
        for unselected_idx in np.argsort(candidate_queue)[max_ncandidates:]:
            candidate_queue[unselected_idx] += aging_factor
        if best_covariate is not None:
            nbases = add_bases(nbases, best_covariate, best_node, best_hinge,
                               best_where, covariates, nodes, hinges,
                               where)
            (lof, coefficients, fit_matrix, covariance_matrix, chol, right_hand_side,
             fixed_mean, candidate_mean) = extend_fit(
                x, y, nbases, covariates, nodes, hinges, where, smoothness, 2,
                fit_matrix, covariance_matrix, right_hand_side, y_mean, fixed_mean,
                candidate_mean)
            candidate_queue = np.append(candidate_queue, [0, 0])

    return (nbases,
            covariates,
            nodes,
            hinges,
            where,
            coefficients,
            lof,
            fit_matrix,
            covariance_matrix,
            right_hand_side,
            fixed_mean,
            candidate_mean)


@njit(cache=True, error_model="numpy", fastmath=True, parallel=False)
def prune_bases(x: np.ndarray,
                y: np.ndarray,
                nbases: int,
                covariates: np.ndarray,
                nodes: np.ndarray,
                hinges: np.ndarray,
                where: np.ndarray,
                lof: float,
                fit_matrix: np.ndarray,
                covariance_matrix: np.ndarray,
                right_hand_side: np.ndarray,
                fixed_mean: np.ndarray,
                candidate_mean: float,
                smoothness: int) -> tuple[int, np.ndarray, np.ndarray]:
    """
    Prune the bases to the best fitting subset of the basis functions.
    By iteratively removing the basis that increases the lack of fit criterion the least.
    Equivalent to the backward pass in the Mars paper.

    Args:
        x: Data points. [n x d]
        y: Target values. [n]
        nbases: Number of basis functions.
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        hinges: Hinges of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]
        lof: Generalised cross validation criterion.
        fit_matrix: Evaluation of the bases on the data. [n x nbases]
        covariance_matrix: Covariance matrix of the model. [nbases x nbases]
        right_hand_side: Right hand side of the model. [nbases]
        fixed_mean: Mean of the fixed basis. [nbases-1]
        candidate_mean: Mean of the candidate basis.
        smoothness: Cost for each basis optimization.

    Returns:
        Number of basis functions.
        Signals product length of basis. [max_nbases x max_nbases]
        Coefficients of the basis. [nbases]
    """
    best_nbases = nbases
    best_where = where.copy()

    best_trimmed_nbases = nbases
    best_trimmed_where = where.copy()

    best_lof = lof

    while nbases > 1:
        best_trimmed_lof = np.inf

        previous_nbases = nbases
        previous_where = where.copy()
        previous_fit = fit_matrix.copy()
        previous_covariance = covariance_matrix.copy()
        previous_right_hand_side = right_hand_side.copy()
        previous_fixed_mean = fixed_mean.copy()
        previous_candidate_mean = candidate_mean

        # First basis function (constant 1) cannot be excluded
        for basis_idx in prange(1, nbases):
            removal_slice = slice(basis_idx, basis_idx + 1)
            nbases = remove_bases(nbases, where, removal_slice)
            (lof, coefficients, fit_matrix, covariance_matrix, chol, right_hand_side,
             fixed_mean, candidate_mean) = shrink_fit(
                y, nbases, smoothness, removal_slice, fit_matrix, covariance_matrix,
                right_hand_side, fixed_mean, candidate_mean)
            if lof < best_trimmed_lof:
                best_trimmed_lof = lof
                best_trimmed_nbases = nbases
                best_trimmed_where = where.copy()
            if lof < best_lof:
                best_lof = lof
                best_nbases = nbases
                best_where = where.copy()
            nbases = previous_nbases
            where = previous_where.copy()
            fit_matrix = previous_fit.copy()
            covariance_matrix = previous_covariance.copy()
            right_hand_side = previous_right_hand_side.copy()
            fixed_mean = previous_fixed_mean.copy()
            candidate_mean = previous_candidate_mean

        nbases = best_trimmed_nbases
        where = best_trimmed_where.copy()
        lof, coefficients, fit_matrix, covariance_matrix, chol, right_hand_side, fixed_mean, candidate_mean, y_mean = fit(
            x,
            y,
            nbases,
            covariates,
            nodes,
            hinges,
            where,
            smoothness)
    nbases = best_nbases
    where = best_where.copy()
    (lof, coefficients, fit_matrix, covariance_matrix, chol, right_hand_side,
     fixed_mean, candidate_mean, y_mean) = fit(x, y, nbases, covariates, nodes, hinges,
                                               where, smoothness)
    return nbases, where, coefficients


@njit(cache=True, error_model="numpy", fastmath=True)
def find_bases(x: np.ndarray,
               y: np.ndarray,
               max_nbases: int = 11,
               max_ncandidates: int = 5,
               aging_factor: float = 0.,
               smoothness: int = 3
               ) -> tuple[
    int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Find the best fitting basis for the given data.

    Args:
        x: Data points. [n x d]
        y: Target values. [n]
        max_nbases: Maximum Number of basis functions.
        max_ncandidates: Maximum number of recalculations per iteration. (See Fast Mars paper)
        aging_factor: Determines how fast unused parent basis funtions need recalculation. (See Fast Mars paper)
        smoothness: Cost for each basis optimization, used to determine the lack of fit criterion.

    Returns:
        Number of basis functions.
        Covariates of the basis. [max_nbases x max_nbases]
        Nodes of the basis. [max_nbases x max_nbases]
        Hinges of the basis. [max_nbases x max_nbases]
        Signals product length of basis. [max_nbases x max_nbases]
        Coefficients of the basis. [nbases]
    """
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    (nbases,
     covariates,
     nodes,
     hinges,
     where,
     coefficients,
     lof,
     fit_matrix,
     covariance_matrix,
     right_hand_side,
     fixed_mean,
     candidate_mean) = expand_bases(x, y, max_nbases, smoothness, max_ncandidates,
                                    aging_factor)
    nbases, where, coefficients = prune_bases(
        x, y, nbases, covariates, nodes, hinges, where, lof, fit_matrix,
        covariance_matrix, right_hand_side, fixed_mean, candidate_mean, smoothness)

    return nbases, covariates, nodes, hinges, where, coefficients


class OMARS:
    """
    Open Multivariate Adaptive Regression Splines (OMARS) model.

    Based on:
    - Friedman, J. (1991). Multivariate adaptive regression splines.
      The annals of statistics, 19(1), 1–67. http://www.jstor.org/stable/10.2307/2241837
    - Friedman, J. (1993). Fast MARS.
      Stanford University Department of Statistics,
      Technical Report No 110. https://statistics.stanford.edu/sites/default/files/LCS%20110.pdf

    To determine the bases use the find_bases method. The fitted model can than be used
    to predict new data by calling the model with the data as argument.
    """

    def __init__(self,
                 nbases: int,
                 covariates: np.ndarray,
                 nodes: np.ndarray,
                 hinges: np.ndarray,
                 where: np.ndarray,
                 coefficients: np.ndarray):
        """
        Initialize the OMARS model.

        Args:
            nbases: Number of basis functions.
            covariates: Covariates of the basis. [max_nbases x max_nbases]
            nodes: Nodes of the basis. [max_nbases x max_nbases]
            hinges: Hinges of the basis. [max_nbases x max_nbases]
            where: Where to apply the hinge. [max_nbases x max_nbases]
            coefficients: Coefficients of the basis. [max_nbases]
        """
        assert isinstance(nbases, int)
        assert covariates.ndim == 2
        assert nodes.ndim == 2
        assert hinges.ndim == 2
        assert where.ndim == 2
        assert coefficients.ndim == 1

        self.nbases = nbases
        max_prod_len = np.max(np.sum(where, axis=0)) + 1

        self.covariates = covariates[:max_prod_len, :nbases]
        self.nodes = nodes[:max_prod_len, :nbases]
        self.hinges = hinges[:max_prod_len, :nbases]
        self.where = where[:max_prod_len, :nbases]

        self.coefficients = coefficients[:nbases]

    def __str__(self) -> str:
        """
        Describes the basis of the model.

        Returns:
            Description of the basis.
        """
        desc = "basis: \n"
        for basis_idx in range(self.covariates.shape[1]):
            for func_idx in range(self.covariates.shape[0]):
                if self.where[func_idx, basis_idx]:
                    cov = self.covariates[func_idx, basis_idx]
                    node = self.nodes[func_idx, basis_idx]
                    hinge = self.hinges[func_idx, basis_idx]
                    desc += f"(x[{cov}] - {node}){u'\u208A' if hinge else ''}"
            desc += "\n"
        return desc

    def __call__(self, x: np.ndarray) -> np.ndarray:
        """
        Predict the target values for the given data points.

        Args:
            x: Data points. [n x d]

        Returns:
            Predicted target values.
        """
        assert x.ndim == 2

        return data_matrix(x,
                           0,
                           self.nbases,
                           self.covariates,
                           self.nodes,
                           self.hinges,
                           self.where) @ self.coefficients

    def __len__(self) -> int:
        """
        Number of basis functions.

        Returns:
            Number of basis functions.
        """
        return self.nbases

    def __getitem__(self, i: int) -> Self:
        """
        Return a submodel with only the i-th basis.

        Args:
            i: Index of the basis.

        Returns:
            Submodel with only the i-th basis.
        """
        assert isinstance(i, int)
        assert i < self.nbases

        idx = [slice(None), slice(i, i + 1)]
        return OMARS(1,
                     self.covariates[*idx],
                     self.nodes[*idx],
                     self.hinges[*idx],
                     self.where[*idx],
                     self.coefficients[idx[1]])

    def __eq__(self, other: Self) -> bool:
        """
        Check if two models are equal. Equality is defined by equal bases.

        Args:
            other: Other model.

        Returns:
            True if the models are equal, False otherwise.
        """
        self_idx = [slice(None), slice(self.nbases)]
        other_idx = [slice(None), slice(other.nbases)]

        return self.nbases == other.nbases and \
            np.array_equal(self.covariates[*self_idx], other.covariates[*other_idx]) and \
            np.array_equal(self.nodes[*self_idx], other.nodes[*other_idx]) and \
            np.array_equal(self.hinges[*self_idx], other.hinges[*other_idx]) and \
            np.array_equal(self.where[*self_idx], other.where[*other_idx])


if __name__ == "__main__":
    import tests.utils as utils

    x, y, y_true = utils.generate_data(100, 2)

    model = OMARS(*find_bases(x, y))

    print(model)
