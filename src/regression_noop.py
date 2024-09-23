from typing import Self

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def active_base_indices(where: np.ndarray) -> np.ndarray:
    """
    Get the indices of the active basis functions.

    Returns:
        Indices of the active basis functions.
    """
    return np.where(np.any(where, axis=0))[0]


def data_matrix(x: np.ndarray,
                basis_indices: np.ndarray,
                covariates: np.ndarray,
                nodes: np.ndarray,
                hinges: np.ndarray,
                where: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the data matrix for the given data and basis, which is the
    evaluation of the basis for the data points.

    Args:
        x: Data points. [n x d]
        basis_indices: Slice of the basis to be evaluated.
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        hinges: Hinges of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]

    Returns:
        Data matrix.
    """
    result = -nodes[:, basis_indices] + x[:, covariates[:, basis_indices]]
    np.maximum(0, result, where=hinges[:, basis_indices], out=result)

    result = result.prod(axis=1, where=where[:, basis_indices])

    result_mean = result.mean(axis=0)
    return result - result_mean, result_mean


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

        for i in range(chol.shape[0]):
            chol[i, i] = np.sqrt(diag[i] + multiplier / b[i] * u[i, i] ** 2)
            chol[i + 1:, i] *= chol[i, i]
            chol[i + 1:, i] += multiplier / b[i] * u[i, i] * u[i, i + 1:] / chol[i, i]

    return chol


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


def remove_bases(
        nbases,
        where: np.ndarray,
        removal_idx: int) -> int:
    """
    Remove basis from the model.

    Args:
        nbases: Number of basis functions.
        where: Signals product length of basis. [max_nbases x max_nbases]
        removal_idx: Slice of the basis to be removed.

    Returns:
        New number of basis functions.

    Notes:
        Only the where attributes is turned off, the other attributes (e.g. covariates) are deliberately not
        deleted to avoid recalculation in the expansion step.
    """
    where[:, removal_idx] = False
    nbases -= 1

    return nbases


def update_init(
        x: np.ndarray,
        old_node: float,
        parent_idx: int,
        nbases: int,
        covariates: np.ndarray,
        nodes: np.ndarray,
        where: np.ndarray,
        fit_matrix: np.ndarray,
        basis_mean: np.ndarray) -> tuple[np.ndarray, float]:
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
        basis_mean: Mean of the bases. [nbases]

    Returns:
        Update values of the data points.
        Mean of the updates.
    """
    prod_idx = np.sum(where[:, nbases - 1])
    new_node = nodes[prod_idx, nbases - 1]
    covariate = covariates[prod_idx, nbases - 1]

    update = x[:, covariate] - new_node
    update[x[:, covariate] >= old_node] = old_node - new_node
    update[x[:, covariate] < new_node] = 0

    if parent_idx != 0:  # Not Constant basis function, otherwise 1 anyway
        update *= fit_matrix[:, parent_idx - 1] + basis_mean[parent_idx - 1]

    update_mean = update.mean()
    update -= update_mean

    return update, update_mean


def calculate_fit_matrix(x: np.ndarray,
                         covariates: np.ndarray,
                         nodes: np.ndarray,
                         hinges: np.ndarray,
                         where) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate the data matrix for the given data.

    Args:
        x: Data points. [n x d]
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        hinges: Hinges of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]

    Returns:
        Fit matrix. [n x nbases]
        Mean of the bases. [nbases]
    """
    fit_matrix, basis_mean = data_matrix(x, active_base_indices(where), covariates,
                                         nodes, hinges, where)

    return fit_matrix, basis_mean


def extend_fit_matrix(x: np.ndarray,
                      nadditions: int,
                      fit_matrix: np.ndarray,
                      basis_mean: np.ndarray,
                      covariates: np.ndarray,
                      nodes: np.ndarray,
                      hinges: np.ndarray,
                      where: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Extend the data matrix by evaluating the newly added basis functions for the data points.

    Args:
        x: Data points. [n x d]
        nadditions: Number of newly added basis functions.
        fit_matrix: Evaluation of the bases on the data. [n x nbases-nadditions]
        basis_mean: Mean of the bases. [nbases-nadditions]
        covariates: Covariates of the basis. [max_nbases x max_nbases]
        nodes: Nodes of the basis. [max_nbases x max_nbases]
        hinges: Hinges of the basis. [max_nbases x max_nbases]
        where: Signals product length of basis. [max_nbases x max_nbases]

    Returns:
        Extended fit matrix. [n x nbases]
        Extended mean of the bases. [nbases]
    """
    ext_indices = active_base_indices(where)[-nadditions:]
    if fit_matrix.size:
        fit_matrix_ext, basis_mean_ext = data_matrix(x, ext_indices, covariates, nodes,
                                                     hinges, where)
        fit_matrix = np.hstack((
            fit_matrix,
            fit_matrix_ext
        ))
        basis_mean = np.append(basis_mean, basis_mean_ext)
    else:
        fit_matrix, basis_mean = data_matrix(x, ext_indices, covariates, nodes, hinges,
                                             where)

    return fit_matrix, basis_mean


def update_fit_matrix(fit_matrix: np.ndarray,
                      basis_mean: np.ndarray,
                      update: np.ndarray,
                      update_mean: float) -> None:
    """
    Update the data matrix by adding the update attribute to the last column.

    Args:
        fit_matrix: Evaluation of the bases on the data. [n x nbases]
        basis_mean: Mean of the bases. [nbases]
        update: Update values of the data points.
        update_mean: Mean of the updates.
    """
    fit_matrix[:, -1] += update
    basis_mean[-1] += update_mean


def calculate_covariance_matrix(fit_matrix: np.ndarray) -> np.ndarray:
    """
    Calculate the covariance matrix of the data matrix,
    which is the chosen formulation of the least-squares problem.

    Args:
        fit_matrix: Evaluation of the bases on the data. [n x nbases]

    Returns:
        Covariance matrix.
    """
    covariance_matrix = fit_matrix.T @ fit_matrix

    covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-8
    return covariance_matrix


def extend_covariance_matrix(covariance_matrix: np.ndarray,
                             nadditions: int,
                             fit_matrix: np.ndarray) -> np.ndarray:
    """
    Extend the covariance matrix by accounting for the newly added basis functions, expects and
    extended fit matrix.

    Args:
        covariance_matrix: Covariance matrix to extend. [nbases-nadditions x nbases-nadditions]
        nadditions: Number of newly added basis functions.
        fit_matrix: Evaluation of the bases on the data. [n x nbases]

    Returns:
        Extended covariance matrix. [nbases x nbases]
    """
    covariance_extension = fit_matrix.T @ fit_matrix[:, -nadditions:]

    if covariance_matrix.size:
        covariance_matrix = np.hstack((covariance_matrix,
                                       covariance_extension[:-nadditions]))
        covariance_matrix = np.vstack((covariance_matrix, covariance_extension.T))
    else:
        covariance_matrix = covariance_extension
    for j in range(1, nadditions + 1):
        covariance_matrix[-j, -j] += 1e-8

    return covariance_matrix


def update_covariance_matrix(covariance_matrix: np.ndarray,
                             update: np.ndarray,
                             fit_matrix: np.ndarray) -> np.ndarray:
    """
    Update the covariance matrix.
    Expects update initialisation and an updated fit matrix.

    Args:
        covariance_matrix: Covariance matrix to update. [nbases x nbases]
        update: Update values of the data points.
        fit_matrix: Evaluation of the bases on the data. [n x nbases]

    Returns:
        Covariance addition: Vector that was used to update the covariance matrix.
    """
    covariance_addition = np.zeros_like(covariance_matrix[-1, :])
    covariance_addition[:-1] = update @ fit_matrix[:, :-1]
    covariance_addition[-1] = 2 * fit_matrix[:, -1] @ update
    covariance_addition[-1] -= update @ update

    covariance_matrix[-1, :-1] += covariance_addition[:-1]
    covariance_matrix[:, -1] += covariance_addition

    return covariance_addition


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
    y_mean = y.mean()
    right_hand_side = fit_matrix.T @ (y - y_mean)
    return right_hand_side, y_mean


def extend_right_hand_side(right_hand_side: np.ndarray,
                           y: np.ndarray,
                           fit_matrix: np.ndarray,
                           y_mean: float,
                           nadditions: int) -> tuple[np.ndarray, float]:
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
    if not right_hand_side.size:
        y_mean = y.mean()
    right_hand_side = np.append(right_hand_side,
                                fit_matrix[:, -nadditions:].T @ (
                                        y - y_mean))
    return right_hand_side, y_mean


def update_right_hand_side(right_hand_side: np.ndarray,
                           y: np.ndarray,
                           y_mean: float,
                           update: np.ndarray) -> None:
    """
    Update the right hand side according to the basis update.

    Args:
        right_hand_side: Right hand side of the least-squares problem. [nbases]
        y: Target values. [n]
        y_mean: Mean of the target values.
        update: Update values of the data points.
    """
    right_hand_side[-1] += update.T @ (y - y_mean)


def generalised_cross_validation(y: np.ndarray,
                                 y_mean: float,
                                 fit_matrix: np.ndarray,
                                 coefficients: np.ndarray,
                                 nbases: int,
                                 smoothness: int) -> float:
    """
    Calculate the generalised cross validation criterion, the lack of fit criterion.

    Args:
        y: Target values. [n]
        y_mean: Mean of the target values.
        fit_matrix: Evaluation of the bases on the data. [n x nbases]
        coefficients: Coefficients of the basis. [nbases]
        nbases: Number of basis functions.
        smoothness: Cost for each basis optimization.

    Returns:
        Generalised cross validation criterion.
    """
    y_pred = fit_matrix @ coefficients + y_mean
    mse = np.sum((y - y_pred) ** 2)

    c_m = nbases + 1 + smoothness * (nbases - 1)
    lof = mse / len(y) / (1 - c_m / len(y) + 1e-6) ** 2
    return lof


def fit(x: np.ndarray,
        y: np.ndarray,
        nbases: int,
        covariates: np.ndarray,
        nodes: np.ndarray,
        hinges: np.ndarray,
        where: np.ndarray,
        smoothness: int) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float]:
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
        Basis mean. [nbases]
        Covariance matrix. [nbases x nbases]
        Cholesky decomposition of the covariance matrix. [nbases x nbases]
        Right hand side of the least-squares problem. [nbases]
        Mean of the target values.
    """
    fit_matrix, basis_mean = calculate_fit_matrix(x, covariates, nodes, hinges, where)
    covariance_matrix = calculate_covariance_matrix(fit_matrix)
    right_hand_side, y_mean = calculate_right_hand_side(y, fit_matrix)

    chol, lower = cho_factor(covariance_matrix, lower=True)

    coefficients = cho_solve((chol, lower), right_hand_side)

    lof = generalised_cross_validation(y, y_mean, fit_matrix, coefficients, nbases,
                                       smoothness)

    return (lof,
            coefficients,
            fit_matrix,
            basis_mean,
            covariance_matrix,
            np.tril(chol),
            right_hand_side,
            y_mean)


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
               basis_mean: np.ndarray,
               covariance_matrix: np.ndarray,
               right_hand_side: np.ndarray,
               y_mean: float) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    float]:
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
        basis_mean: Mean of the bases. [nbases-nadditions]
        covariance_matrix: Covariance matrix of the smaller model. [nbases-nadditions x nbases-nadditions]
        right_hand_side: Right hand side of the smaller model. [nbases-nadditions]
        y_mean: Mean of the target values.

    Returns:
        Generalised cross validation criterion.
        Coefficients of the basis. [nbases]
        Fit matrix. [n x nbases]
        Basis mean. [nbases]
        Covariance matrix. [nbases x nbases]
        Cholesky decomposition of the covariance matrix. [nbases x nbases]
        Right hand side of the least-squares problem. [nbases]
        Mean of the fixed basis. [nbases-1]
        Mean of the candidate basis.
    """
    fit_matrix, basis_mean = extend_fit_matrix(x, nadditions, fit_matrix, basis_mean,
                                               covariates, nodes, hinges, where)
    covariance_matrix = extend_covariance_matrix(covariance_matrix, nadditions,
                                                 fit_matrix)
    right_hand_side, y_mean = extend_right_hand_side(right_hand_side, y, fit_matrix,
                                                     y_mean,
                                                     nadditions)

    chol, lower = cho_factor(covariance_matrix, lower=True)

    coefficients = cho_solve((chol, lower), right_hand_side)

    lof = generalised_cross_validation(y, y_mean, fit_matrix, coefficients, nbases,
                                       smoothness)

    return (lof,
            coefficients,
            fit_matrix,
            basis_mean,
            covariance_matrix,
            np.tril(chol),
            right_hand_side,
            y_mean)


def update_fit(x: np.ndarray,
               y: np.ndarray,
               nbases: int,
               covariates: np.ndarray,
               nodes: np.ndarray,
               where: np.ndarray,
               smoothness: int,
               fit_matrix: np.ndarray,
               basis_mean: np.ndarray,
               covariance_matrix: np.ndarray,
               right_hand_side: np.ndarray,
               y_mean: float,
               old_node: float,
               parent_idx: int,
               chol: np.ndarray) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray]:
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
        basis_mean: Mean of the bases. [nbases]
        covariance_matrix: Covariance matrix of the model. [nbases x nbases]
        right_hand_side: Right hand side of the model. [nbases]
        y_mean: Mean of the target values.
        old_node: Previous node of the basis function.
        parent_idx: Index of the parent basis function.
        chol: Cholesky decomposition of the covariance matrix. [nbases x nbases]

    Returns:
        Generalised cross validation criterion.
        Coefficients of the basis. [nbases]
        Fit matrix. [n x nbases]
        Basis mean. [nbases]
        Covariance matrix. [nbases x nbases]
        Cholesky decomposition of the covariance matrix. [nbases x nbases]
        Right hand side of the least-squares problem. [nbases]
    """
    update, update_mean = update_init(x, old_node, parent_idx, nbases, covariates,
                                      nodes, where, fit_matrix, basis_mean)
    update_fit_matrix(fit_matrix, basis_mean, update, update_mean)
    covariance_addition = update_covariance_matrix(covariance_matrix, update,
                                                   fit_matrix)

    if covariance_addition.any():
        eigenvalues, eigenvectors = decompose_addition(covariance_addition)
        chol = update_cholesky(chol, eigenvectors, eigenvalues)

    update_right_hand_side(right_hand_side, y, y_mean, update)

    coefficients = cho_solve((chol, True), right_hand_side)

    lof = generalised_cross_validation(y, y_mean, fit_matrix, coefficients, nbases,
                                       smoothness)

    return (lof,
            coefficients,
            fit_matrix,
            basis_mean,
            covariance_matrix,
            np.tril(chol),
            right_hand_side)


def shrink_fit(y: np.ndarray,
               y_mean: float,
               nbases: int,
               smoothness: int,
               removal_idx: int,
               fit_matrix: np.ndarray,
               basis_mean: np.ndarray,
               covariance_matrix: np.ndarray,
               right_hand_side: np.ndarray,
               ) -> tuple[
    float,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray]:
    """
    Shrink the least-squares fit. Expects the correct fit matrix, covariance matrix
    and right hand side of the larger model.

    Args:
        y: Target values.  [n]
        y_mean: Mean of the target values.
        nbases: Number of basis functions.
        smoothness: Cost for each basis optimization.
        removal_idx: Index of the basis function to be removed.
        fit_matrix: Evaluation of the bases on the data. [n x nbases+1]
        basis_mean: Mean of the bases. [nbases+1]
        covariance_matrix: Covariance matrix of the model. [nbases+1 x nbases+1]
        right_hand_side: Right hand side of the model. [nbases+1]

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
    removal_idx -= 1

    fit_matrix = np.delete(fit_matrix, removal_idx, axis=1)
    basis_mean = np.delete(basis_mean, removal_idx)
    covariance_matrix = np.delete(covariance_matrix, removal_idx, axis=0)
    if covariance_matrix.size:
        covariance_matrix = np.delete(covariance_matrix, removal_idx, axis=1)

    right_hand_side = np.delete(right_hand_side, removal_idx)

    if fit_matrix.size:
        chol, lower = cho_factor(covariance_matrix, lower=True)

        coefficients = cho_solve((chol, True), right_hand_side)

        lof = generalised_cross_validation(y, y_mean, fit_matrix, coefficients, nbases,
                                           smoothness)
        chol = np.tril(chol)
    else:
        chol = None
        mse = np.sum((y - y_mean) ** 2)
        lof = mse / len(y) / (1 - 1 / len(y)) ** 2
        coefficients = None

    return (lof,
            coefficients,
            fit_matrix,
            basis_mean,
            covariance_matrix,
            chol,
            right_hand_side)


def expand_bases(x: np.ndarray,
                 y: np.ndarray,
                 max_nbases: int,
                 smoothness: int,
                 max_ncandidates: int,
                 aging_factor: float) -> tuple[
    int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
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
        Generalised cross validation criterion.
        Coefficients of the basis. [nbases]
        Fit matrix. [n x nbases]
        Basis mean. [nbases]
        Covariance matrix. [nbases x nbases]
        Cholesky decomposition of the covariance matrix. [nbases x nbases]
        Right hand side of the least-squares problem. [nbases]
        Mean of the target values.
    """
    # Bases (including the constant function)
    nbases = 1
    covariates = np.zeros((max_nbases, max_nbases), dtype=int)
    nodes = np.zeros((max_nbases, max_nbases), dtype=float)
    hinges = np.zeros((max_nbases, max_nbases), dtype=bool)
    where = np.zeros((max_nbases, max_nbases), dtype=bool)

    coefficients = np.empty(0, dtype=float)
    fit_matrix = np.empty((0, 0), dtype=float)
    covariance_matrix = np.empty((0, 0), dtype=float)
    right_hand_side = np.empty(0, dtype=float)

    lof = float()

    basis_mean = np.empty(0, dtype=float)
    y_mean = float()

    all_covariates = set(range(x.shape[1]))

    candidate_queue = [0.]
    for _ in range((max_nbases - nbases) // 2):
        best_lof = np.inf
        best_covariate = None
        best_node = None
        best_hinge = None
        best_where = None
        for parent_idx in np.argsort(candidate_queue)[:-max_ncandidates - 1:-1]:
            eligible_covariates = all_covariates - set(
                covariates[where[:, parent_idx], parent_idx])
            basis_lof = np.inf
            parent_depth = where[:, parent_idx].sum()
            for cov in eligible_covariates:
                if parent_idx == 0:  # constant function
                    eligible_knots = x[:, cov].copy()
                else:
                    eligible_knots = x[
                        np.where(fit_matrix[:, parent_idx - 1] > 0)[0], cov]
                eligible_knots[::-1].sort()
                additional_covariates = np.tile(covariates[:, parent_idx], (2, 1)).T
                additional_nodes = np.tile(nodes[:, parent_idx], (2, 1)).T
                additional_hinges = np.tile(hinges[:, parent_idx], (2, 1)).T
                additional_where = np.tile(where[:, parent_idx], (2, 1)).T

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
                (lof, coefficients, fit_matrix, basis_mean, covariance_matrix, chol,
                 right_hand_side, y_mean) = extend_fit(
                    x, y, nbases, covariates, nodes, hinges, where, smoothness, 2,
                    fit_matrix, basis_mean, covariance_matrix, right_hand_side, y_mean)
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
                    (lof, coefficients, fit_matrix, basis_mean, covariance_matrix, chol,
                     right_hand_side) = update_fit(
                        x, y, nbases, covariates, nodes, where, smoothness, fit_matrix,
                        basis_mean, covariance_matrix, right_hand_side, y_mean,
                        old_node, parent_idx, chol)
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
                for i in range(2):
                    nbases = remove_bases(nbases, where, nbases - 1)
                    # CARE remove basis decrements nbases
                    (lof, coefficients, fit_matrix, basis_mean, covariance_matrix, chol,
                     right_hand_side) = shrink_fit(
                        y, y_mean, nbases, smoothness, nbases, fit_matrix,
                        basis_mean, covariance_matrix, right_hand_side)

            candidate_queue[parent_idx] = best_lof - basis_lof
        for unselected_idx in np.argsort(candidate_queue)[max_ncandidates:]:
            candidate_queue[unselected_idx] += aging_factor
        if best_covariate is not None:
            nbases = add_bases(nbases, best_covariate, best_node, best_hinge,
                               best_where, covariates, nodes, hinges,
                               where)
            (lof, coefficients, fit_matrix, basis_mean, covariance_matrix, chol,
             right_hand_side, y_mean) = extend_fit(
                x, y, nbases, covariates, nodes, hinges, where, smoothness, 2,
                fit_matrix, basis_mean, covariance_matrix, right_hand_side, y_mean)
            candidate_queue.extend([0, 0])

    return (nbases,
            covariates,
            nodes,
            hinges,
            where,
            lof,
            coefficients,
            fit_matrix,
            basis_mean,
            covariance_matrix,
            right_hand_side,
            y_mean)


def prune_bases(x: np.ndarray,
                y: np.ndarray,
                nbases: int,
                covariates: np.ndarray,
                nodes: np.ndarray,
                hinges: np.ndarray,
                where: np.ndarray,
                lof: float,
                fit_matrix: np.ndarray,
                basis_mean: np.ndarray,
                covariance_matrix: np.ndarray,
                right_hand_side: np.ndarray,
                y_mean: float,
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
        basis_mean: Mean of the bases. [nbases]
        covariance_matrix: Covariance matrix of the model. [nbases x nbases]
        right_hand_side: Right hand side of the model. [nbases]
        y_mean: Mean of the target values.
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
        previous_basis_mean = basis_mean.copy()

        # Constant basis function cannot be excluded
        for basis_idx in range(1, nbases):
            nbases = remove_bases(nbases, where, basis_idx)
            (lof, coefficients, fit_matrix, basis_mean, covariance_matrix, chol,
             right_hand_side) = shrink_fit(
                y, y_mean, nbases, smoothness, basis_idx, fit_matrix,
                basis_mean, covariance_matrix, right_hand_side)
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
            basis_mean = previous_basis_mean.copy()

        nbases = best_trimmed_nbases
        where = best_trimmed_where.copy()
        (lof, coefficients, fit_matrix, basis_mean, covariance_matrix, chol,
         right_hand_side, y_mean) = fit(x, y, nbases, covariates, nodes, hinges, where,
                                        smoothness)
    nbases = best_nbases
    where = best_where.copy()
    (lof, coefficients, fit_matrix, basis_mean, covariance_matrix, chol,
     right_hand_side, y_mean) = fit(x, y, nbases, covariates, nodes, hinges,
                                    where, smoothness)
    return nbases, where, coefficients


def find_bases(x: np.ndarray,
               y: np.ndarray,
               max_nbases: int = 11,
               max_ncandidates: int = 5,
               aging_factor: float = 0.,
               smoothness: int = 3
               ) -> tuple[
    int, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
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
        Mean of the target values.
    """
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    (nbases,
     covariates,
     nodes,
     hinges,
     where,
     lof,
     coefficients,
     fit_matrix,
     basis_mean,
     covariance_matrix,
     right_hand_side,
     y_mean) = expand_bases(x, y, max_nbases, smoothness, max_ncandidates, aging_factor)
    nbases, where, coefficients = prune_bases(
        x, y, nbases, covariates, nodes, hinges, where, lof, fit_matrix, basis_mean,
        covariance_matrix, right_hand_side, y_mean, smoothness)

    return nbases, covariates, nodes, hinges, where, coefficients, y_mean


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
                 coefficients: np.ndarray,
                 y_mean: float) -> None:
        """
        Initialize the OMARS model.

        Args:
            nbases: Number of basis functions.
            covariates: Covariates of the basis. [max_nbases x max_nbases]
            nodes: Nodes of the basis. [max_nbases x max_nbases]
            hinges: Hinges of the basis. [max_nbases x max_nbases]
            where: Where to apply the hinge. [max_nbases x max_nbases]
            coefficients: Coefficients of the basis. [max_nbases]
            y_mean: Mean of the target values.
        """
        assert isinstance(nbases, int)
        assert covariates.ndim == 2
        assert nodes.ndim == 2
        assert hinges.ndim == 2
        assert where.ndim == 2
        assert coefficients.ndim == 1

        self.nbases = nbases
        self.max_prod_len = np.max(np.sum(where, axis=0)) + 1

        self.covariates = covariates[:self.max_prod_len, :nbases]
        self.nodes = nodes[:self.max_prod_len, :nbases]
        self.hinges = hinges[:self.max_prod_len, :nbases]
        self.where = where[:self.max_prod_len, :nbases]

        self.coefficients = coefficients[:nbases - 1]
        self.y_mean = y_mean

    def __str__(self) -> str:
        """
        Describes the basis of the model.

        Returns:
            Description of the basis.
        """
        desc = "Basis functions: \n"
        for basis_idx in active_base_indices(self.where):
            for func_idx in range(self.max_prod_len):
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

        fit_matrix, _ = data_matrix(x, active_base_indices(self.where), self.covariates,
                                    self.nodes, self.hinges, self.where)
        centered_y_pred = fit_matrix @ self.coefficients
        return centered_y_pred + self.y_mean

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

        nbases = 1
        covariates = self.covariates[:, i:i + 1]
        nodes = self.nodes[:, i:i + 1]
        hinges = self.hinges[:, i:i + 1]
        where = self.where[:, i:i + 1]
        if i != 0:
            coefficients = self.coefficients[i:i + 1]
        else:
            coefficients = [self.y_mean]
        return OMARS(nbases, covariates, nodes, hinges, where, coefficients,
                     self.y_mean)

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
