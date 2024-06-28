from typing import Self

import numpy as np
from scipy.linalg import cho_factor, cho_solve


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


class OMARS:
    """
    Open Multivariate Adaptive Regression Splines (OMARS) model.
    Can be used to determine multilinear relations in data by finding a best fit of a sum of basis functions.
    Based on:
    - Friedman, J. (1991). Multivariate adaptive regression splines.
      The annals of statistics, 19(1), 1–67. http://www.jstor.org/stable/10.2307/2241837
    - Friedman, J. (1993). Fast MARS.
      Stanford University Department of Statistics,
      Technical Report No 110. https://statistics.stanford.edu/sites/default/files/LCS%20110.pdf

    To determine the bases use the find_bases method. The fitted model can than be used to predict new data by calling
    the model with the data as argument.
    """

    def __init__(self,
                 max_nbases: int = 11,
                 max_ncandidates: int = 5,
                 aging_factor: float = 0.,
                 smoothness: float = 3):
        """
        Initialize the OMARS model.
        Args:
            max_nbases: Maximum number of basis functions.
            max_ncandidates: Maximum number of recalculations per iteration. (See Fast Mars paper)
            aging_factor: Determines how fast unused parent basis funtions need recalculation. (See Fast Mars paper)
            smoothness: Cost for each basis function optimization, used to determine the lack of fit criterion.

        Other attributes:
            nbases: Number of basis functions.
            covariates: Covariates of the basis functions.
            nodes: Nodes of the basis functions.
            hinges: Hinges of the basis functions.
            where: Which parts of the aforementioned arrays are used. (Basically determines the product length of the basis)
            coefficients: Coefficients for the superposition of bases.
            fit_matrix: Data matrix of the fitted data. (Evaluation of basis functions for the data points)
            covariance_matrix: Covariance matrix of the fitted data.
            right_hand_side: Right hand side of the fitted data.
            lof: Lack of fit criterion.
            indices: Indices of the data points that need to be recalculated.
            update: Update of the fit matrix.
            fixed_mean: Mean of the unchanging bases in this iteration.
            candidate_mean: Mean of the changing basis in this iteration.
            update_mean: Mean of update attribute.
            y_mean: Mean of the target values.
        """
        self.max_nbases = max_nbases
        self.max_ncandidates = max_ncandidates
        self.aging_factor = aging_factor
        self.smoothness = smoothness

        # Bases
        self.nbases = 1
        self.covariates = np.zeros((self.max_nbases, self.max_nbases), dtype=int)
        self.nodes = np.zeros((self.max_nbases, self.max_nbases), dtype=float)
        self.hinges = np.zeros((self.max_nbases, self.max_nbases), dtype=bool)
        self.where = np.zeros((self.max_nbases, self.max_nbases), dtype=bool)

        self.coefficients = np.array([], dtype=float)
        self.fit_matrix = np.array([[]], dtype=float)
        self.covariance_matrix = np.array([[]], dtype=float)
        self.right_hand_side = np.array([], dtype=float)

        self.lof = float()

        self.indices = np.array([], dtype=bool)
        self.update = np.array([], dtype=float)
        self.fixed_mean = np.array([], dtype=float)
        self.candidate_mean = float()
        self.update_mean = float()
        self.y_mean = float()

    def data_matrix(self, x: np.ndarray, basis_slice: slice) -> np.ndarray:
        """
        Calculate the data matrix for the given data and basis functions, which is the evaluation of the basis functions
        for the data points.

        Args:
            x: Data points. [n x d]
            basis_slice: Slice of the basis functions to be evaluated.

        Returns:
            Data matrix.
        """
        assert x.ndim == 2
        assert isinstance(basis_slice, slice)

        result = -self.nodes[:, basis_slice] + x[:, self.covariates[:, basis_slice]]
        np.maximum(0, result, where=self.hinges[:, basis_slice], out=result)

        return result.prod(axis=1, where=self.where[:, basis_slice])

    def __str__(self) -> str:
        """
        Describes the basis functions of the model.

        Returns:
            Description of the basis functions.
        """
        desc = "Basis functions: \n"
        for basis_idx in range(self.nbases):
            for func_idx in range(self.max_nbases):
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

        return self.data_matrix(x, slice(self.nbases)) @ self.coefficients

    def __len__(self) -> int:
        """
        Number of basis functions.

        Returns:
            Number of basis functions.
        """
        return self.nbases

    def __getitem__(self, i: int) -> Self:
        """
        Return a submodel with only the i-th basis function.

        Args:
            i: Index of the basis function.

        Returns:
            Submodel with only the i-th basis function.
        """
        assert isinstance(i, int)
        assert i < self.nbases

        sub_model = OMARS()
        sub_model.nbases = 1
        sub_model.covariates = self.covariates[:, i:i + 1]
        sub_model.nodes = self.nodes[:, i:i + 1]
        sub_model.hinges = self.hinges[:, i:i + 1]
        sub_model.where = self.where[:, i:i + 1]
        sub_model.coefficients = self.coefficients[i:i + 1]
        return sub_model

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

    def add_basis(self,
                  covariates: np.ndarray,
                  nodes: np.ndarray,
                  hinges: np.ndarray,
                  where: np.ndarray) -> None:
        """
        Add a basis functions to the model.

        Args:
            covariates: Covariates of the basis functions. (On which dimension the basis functions are applied). [maxnbases x nadditions]
            nodes: Nodes of the basis functions. [maxnbases x nadditions]
            hinges: Hinges of the basis functions. [maxnbases x nadditions]
            where: Which parts of the aforementioned arrays are used. (Basically determines the product length of the basis) [maxnbases x nadditions]
        """
        assert covariates.ndim == nodes.ndim == hinges.ndim == where.ndim == 2
        assert (covariates.shape[0] == nodes.shape[0]
                == hinges.shape[0] == where.shape[0] == self.max_nbases)
        assert covariates.shape[1] == nodes.shape[1] == hinges.shape[1] == where.shape[
            1]
        assert covariates.dtype == int
        assert nodes.dtype == float
        assert hinges.dtype == bool
        assert where.dtype == bool
        assert self.nbases + covariates.shape[1] <= self.max_nbases

        addition_slice = slice(self.nbases, self.nbases + covariates.shape[1])
        self.covariates[:, addition_slice] = covariates
        self.nodes[:, addition_slice] = nodes
        self.hinges[:, addition_slice] = hinges
        self.where[:, addition_slice] = where
        self.nbases += covariates.shape[1]

    def update_basis(self, covariates: np.ndarray,
                     nodes: np.ndarray,
                     hinges: np.ndarray,
                     where: np.ndarray) -> None:
        """
        Update the latest basis function.

        Args:
            covariates: Covariates of the basis functions. (On which dimension the basis functions are applied) [maxnbases x 1]
            nodes: Nodes of the basis functions. [maxnbases x 1]
            hinges: Hinges of the basis functions. [maxnbases x 1]
            where: Which parts of the aforementioned arrays are used. (Basically determines the product length of the basis) [maxnbases x 1]
        """
        assert covariates.ndim == nodes.ndim == hinges.ndim == where.ndim == 1
        assert covariates.shape == nodes.shape == hinges.shape == where.shape
        assert covariates.dtype == int
        assert nodes.dtype == float
        assert hinges.dtype == bool
        assert where.dtype == bool

        self.covariates[:, self.nbases - 1] = covariates
        self.nodes[:, self.nbases - 1] = nodes
        self.hinges[:, self.nbases - 1] = hinges
        self.where[:, self.nbases - 1] = where

    def remove_basis(self, removal_slice: slice) -> None:
        """
        Remove basis functions from the model.

        Args:
            removal_slice: Slice of the basis functions to be removed.

        Notes:
            Only the where attributes is turned off, the other attributes (e.g. covariates) are deliberately not
            deleted to avoid recalculation in the expansion step.
        """
        assert isinstance(removal_slice, slice)

        self.where[:, removal_slice] = False
        self.nbases -= removal_slice.stop - removal_slice.start

    def calculate_means(self, collapsed_fit: np.ndarray) -> None:
        """
        Calculate and store the means of the fixed and candidate basis functions.

        Args:
            collapsed_fit: Sum of the fit matrix. [nbases]
        """
        assert collapsed_fit.shape[0] == self.fit_matrix.shape[1]

        self.fixed_mean = collapsed_fit[:-1] / self.fit_matrix.shape[0]
        self.candidate_mean = collapsed_fit[-1] / self.fit_matrix.shape[0]

    def extend_means(self, collapsed_fit: np.ndarray, nadditions: int) -> None:
        """
        Extend the means of the fixed and candidate basis functions by accounting for newly added bases.

        Args:
            collapsed_fit: Sum of the fit matrix. [nadditions]
            nadditions: Number of newly added bases.
        """

        assert collapsed_fit.shape[0] == nadditions

        self.fixed_mean = np.append(self.fixed_mean, self.candidate_mean)
        self.fixed_mean = np.append(self.fixed_mean,
                                    collapsed_fit[:-1] / self.fit_matrix.shape[0])
        self.candidate_mean = collapsed_fit[-1] / self.fit_matrix.shape[0]

    def shrink_means(self, removal_slice: slice) -> None:
        """
        Shrink the means of the fixed and candidate basis functions accounting for removed bases.

        Args:
            removal_slice: Slice of the basis functions to be removed.
        """
        assert isinstance(removal_slice, slice)

        joint_mean = np.append(self.fixed_mean, self.candidate_mean)
        reduced_mean = np.delete(joint_mean, removal_slice)

        self.fixed_mean = reduced_mean[:-1]
        self.candidate_mean = reduced_mean[-1]

    def update_init(self, x: np.ndarray, old_node: float, parent_idx: int) -> None:
        """
        Initialize the update of the fit by precomputing the necessary update values, that allow for a fast update of
        the cholesky decomposition and therefore a faster least-squares fit.

        Args:
            x: Data points. [n x d]
            old_node: Previous node of the basis function.
            parent_idx: Index of the parent basis function.
        """
        assert x.ndim == 2
        assert x.shape[0] == self.fit_matrix.shape[0]
        assert isinstance(old_node, float)
        assert isinstance(parent_idx, int)

        prod_idx = np.sum(self.where[:, self.nbases - 1])
        new_node = self.nodes[prod_idx, self.nbases - 1]
        covariate = self.covariates[prod_idx, self.nbases - 1]

        self.indices = x[:, covariate] > new_node
        self.update = np.where(x[self.indices, covariate] >= old_node,
                               old_node - new_node,
                               x[self.indices, covariate] - new_node)
        self.update *= self.fit_matrix[self.indices, parent_idx]

        self.update_mean = np.sum(self.update) / len(x)
        self.candidate_mean += self.update_mean

    def calculate_fit_matrix(self, x: np.ndarray) -> None:
        """
        Calculate the data matrix for the given data.

        Args:
            x: Data points. [n x d]
        """
        assert x.ndim == 2

        self.fit_matrix = self.data_matrix(x, slice(self.nbases))

    def extend_fit_matrix(self, x: np.ndarray, nadditions: int) -> None:
        """
        Extend the data matrix by evaluating the newly added basis functions for the data points.

        Args:
            x: Data points. [n x d]
            nadditions: Number of newly added basis functions.
        """
        assert x.ndim == 2
        assert isinstance(nadditions, int)

        ext_slice = slice(self.nbases - nadditions, self.nbases)
        self.fit_matrix = np.column_stack((
            self.fit_matrix,
            self.data_matrix(x, ext_slice)
        ))

    def update_fit_matrix(self) -> None:
        """
        Update the data matrix by adding the update attribute to the last column.
        """
        self.fit_matrix[self.indices, -1] += self.update

    def calculate_covariance_matrix(self) -> None:
        """
        Calculate the covariance matrix of the data matrix, which is the chosen formulation of the least-squares problem.
        """
        self.covariance_matrix = self.fit_matrix.T @ self.fit_matrix
        collapsed_fit = np.sum(self.fit_matrix, axis=0)
        self.calculate_means(collapsed_fit)
        self.covariance_matrix -= np.outer(collapsed_fit, collapsed_fit) / \
                                  self.fit_matrix.shape[0]
        '''
        ATTENTION !!!
        The basis through the forward pass does not need to be linearly independent, 
        which may lead to a singular matrix. Here we can either add a small diagonal 
        value or set the dependent coefficients to 0 to obtain a unique solution again.
        '''
        self.covariance_matrix += np.eye(self.covariance_matrix.shape[0]) * 1e-8

    def extend_covariance_matrix(self, nadditions: int) -> None:
        """
        Extend the covariance matrix by accounting for the newly added basis functions, expects and
        extended fit matrix.

        Args:
            nadditions: Number of newly added basis functions.
        """
        assert isinstance(nadditions, int)

        covariance_extension = self.fit_matrix.T @ self.fit_matrix[:, -nadditions:]
        collapsed_fit = np.sum(self.fit_matrix[:, -nadditions:], axis=0)
        self.extend_means(collapsed_fit, nadditions)
        full_fit = np.append(self.fixed_mean, self.candidate_mean) * \
                   self.fit_matrix.shape[0]
        covariance_extension -= np.outer(full_fit, collapsed_fit) / \
                                self.fit_matrix.shape[0]

        self.covariance_matrix = np.column_stack((self.covariance_matrix,
                                                  covariance_extension[:-nadditions]))
        self.covariance_matrix = np.row_stack((self.covariance_matrix,
                                               covariance_extension.T))
        for j in range(1, nadditions + 1):
            self.covariance_matrix[-j, -j] += 1e-8

    def update_covariance_matrix(self) -> np.ndarray:
        """
        Update the covariance matrix. Expects update initialisation and an updated fit matrix.
        """
        covariance_addition = np.zeros_like(self.covariance_matrix[-1, :])
        covariance_addition[:-1] += np.tensordot(self.update,
                                                 self.fit_matrix[self.indices,
                                                 :-1] - self.fixed_mean,
                                                 axes=[[0], [0]])
        covariance_addition[-1] += (self.fit_matrix[self.indices, -1] - self.update) @ (self.update - self.update_mean)
        covariance_addition[-1] += self.update @ (self.fit_matrix[self.indices, -1] - self.candidate_mean)

        self.covariance_matrix[-1, :-1] += covariance_addition[:-1]
        self.covariance_matrix[:, -1] += covariance_addition

        return covariance_addition

    def decompose_addition(self, covariance_addition: np.ndarray) -> tuple[list[float],
    list[np.ndarray]]:
        """
        Decompose the addition to the covariance matrix, which is by adding to the last row and column of the matrix,
        into eigenvalues and eigenvectors to perform 2 rank-1 updates.

        Args:
            covariance_addition: Addition to the covariance matrix. (the same vector is applied to the row and column) [nbases]

        Returns:
            Eigenvalues and eigenvectors of the addition.
        """
        assert covariance_addition.shape == self.covariance_matrix[-1, :].shape

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

    def calculate_right_hand_side(self, y: np.ndarray) -> None:
        """
        Calculate the right hand side of the least-squares problem.

        Args:
            y: Target values. [n]
        """
        assert y.ndim == 1
        assert y.shape[0] == self.fit_matrix.shape[0]
        assert self.fit_matrix.shape[1] == self.covariance_matrix.shape[0]

        self.y_mean = np.mean(y)
        self.right_hand_side = self.fit_matrix.T @ (y - self.y_mean)

    def extend_right_hand_side(self, y: np.ndarray, nadditions: int) -> None:
        """
        Extend the right hand side of the least-squares problem by accounting for the newly added basis functions.

        Args:
            y: Target values. [n]
            nadditions: Number of newly added basis functions.
        """
        assert y.ndim == 1
        assert y.shape[0] == self.fit_matrix.shape[0]
        assert self.fit_matrix.shape[1] == self.covariance_matrix.shape[0]
        assert nadditions < self.nbases

        self.right_hand_side = np.append(self.right_hand_side,
                                         self.fit_matrix[:, -nadditions:].T @ (
                                                 y - self.y_mean))

    def update_right_hand_side(self, y: np.ndarray) -> None:
        """
        Update the right hand side according to the basis update.

        Args:
            y: Target values. [n]
        """
        assert y.ndim == 1
        assert y.shape[0] == self.fit_matrix.shape[0]

        self.right_hand_side[-1] += np.sum(
            self.update * (y[self.indices] - self.y_mean))

    def generalised_cross_validation(self, y: np.ndarray) -> None:
        """
        Calculate the generalised cross validation criterion, the lack of fit criterion.

        Args:
            y: Target values. [n]
        """
        assert y.ndim == 1

        y_pred = self.fit_matrix @ self.coefficients
        mse = np.sum((y - y_pred) ** 2)  # rhs instead of y?

        c_m = self.nbases + 1 + self.smoothness * (self.nbases - 1)
        self.lof = mse / len(y) / (1 - c_m / len(y)) ** 2

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the least-squares fit of the current model from scratch.

        Args:
            x: Data points. [n x d]
            y: Target values. [n]

        Returns:
            Cholesky decomposition of the covariance matrix. (for reusal) [nbases x nbases]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]

        self.calculate_fit_matrix(x)
        self.calculate_covariance_matrix()
        self.calculate_right_hand_side(y)

        chol, lower = cho_factor(self.covariance_matrix, lower=True)

        self.coefficients = cho_solve((chol, lower), self.right_hand_side)

        self.generalised_cross_validation(y)

        return np.tril(chol)

    def extend_fit(self, x: np.ndarray, y: np.ndarray, nadditions: int):
        """
        Extend the least-squares fit. Expects the correct fit matrix, covariance matrix and right hand side of the
        smaller model.

        Args:
            x: Data points. [n x d]
            y: Target values. [n]
            nadditions: Number of newly added basis functions.

        Returns:
            Cholesky decomposition of the covariance matrix. (for reusal) [nbases x nbases]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert isinstance(nadditions, int)

        self.extend_fit_matrix(x, nadditions)
        self.extend_covariance_matrix(nadditions)
        self.extend_right_hand_side(y, nadditions)

        chol, lower = cho_factor(self.covariance_matrix, lower=True)

        self.coefficients = cho_solve((chol, lower), self.right_hand_side)

        self.generalised_cross_validation(y)

        return np.tril(chol)

    def update_fit(self, x: np.ndarray, y: np.ndarray, chol: np.ndarray,
                   old_node: float,
                   parent_idx: int) -> np.ndarray:
        """
        Update the least-squares fit. Expects the correct fit matrix, covariance matrix and right hand side of the
        previous model. The only change in the bases occurred in the node of the last basis function, which is smaller
        than the old node.

        Args:
            x: Data points. [n x d]
            y: Target values. [n]
            chol: Cholesky decomposition of the covariance matrix. [nbases x nbases]
            old_node: Previous node of the basis function.
            parent_idx: Index of the parent basis function.

        Returns:
            Cholesky decomposition of the covariance matrix. (for reusal) [nbases x nbases]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert chol.shape[0] == chol.shape[1] == self.nbases
        assert self.nodes[
                   np.sum(self.where[:, self.nbases - 1]), self.nbases - 1] <= old_node
        assert parent_idx < self.nbases

        # Expects: Fit Mat/Cov Mat/RHS of same size model, with same v and u > t
        self.update_init(x, old_node, parent_idx)
        self.update_fit_matrix()
        covariance_addition = self.update_covariance_matrix()

        if covariance_addition.any():
            eigenvalues, eigenvectors = self.decompose_addition(covariance_addition)
            chol = update_cholesky(chol, eigenvectors, eigenvalues)

        self.update_right_hand_side(y)

        self.coefficients = cho_solve((chol, True), self.right_hand_side)

        self.generalised_cross_validation(y)

        return chol

    def shrink_fit(self, x: np.ndarray, y: np.ndarray,
                   removal_slice: slice) -> np.ndarray:
        """
        Shrink the least-squares fit. Expects the correct fit matrix, covariance matrix and right hand side of the
        larger model.

        Args:
            x: Data points. [n x d]
            y: Target values.  [n]
            removal_slice: Slice of the basis functions to be removed.

        Returns:
            Cholesky decomposition of the covariance matrix. (for reusal) [nbases x nbases]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert isinstance(removal_slice, slice)

        self.fit_matrix = np.delete(self.fit_matrix, removal_slice, axis=1)
        self.covariance_matrix = np.delete(self.covariance_matrix, removal_slice,
                                           axis=0)
        self.covariance_matrix = np.delete(self.covariance_matrix, removal_slice,
                                           axis=1)
        self.shrink_means(removal_slice)

        self.right_hand_side = np.delete(self.right_hand_side, removal_slice)

        chol, lower = cho_factor(self.covariance_matrix, lower=True)

        self.coefficients = cho_solve((chol, True), self.right_hand_side)

        self.generalised_cross_validation(y)

        return chol

    def expand_bases(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Expand the bases to the maximum allowed number of basis functions. By iteratively adding the basis
        that reduces the lack of fit criterion the most. Equivalent to the forward pass in the Mars paper, including
        the adaptions in the Fast Mars paper.

        Args:
            x: Data points. [n x d]
            y: Target values. [n]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]

        covariates = set(range(x.shape[1]))
        self.fit(x, y)
        candidate_queue = [0.] * self.nbases
        for _ in range((self.max_nbases - self.nbases) // 2):
            best_lof = np.inf
            best_covariate = None
            best_node = None
            best_hinge = None
            best_where = None
            for parent_idx in np.argsort(candidate_queue)[:-self.max_ncandidates - 1:-1]:
                eligible_covariates = covariates - set(
                    self.covariates[self.where[:, parent_idx], parent_idx])
                basis_lof = np.inf
                parent_depth = np.sum(self.where[:, parent_idx])
                for cov in eligible_covariates:
                    eligible_knots = x[
                        np.where(self.fit_matrix[:, parent_idx] > 0)[0], cov]
                    eligible_knots[::-1].sort()
                    additional_covariates = np.tile(self.covariates[:, parent_idx],
                                                    (2, 1)).T
                    additional_nodes = np.tile(self.nodes[:, parent_idx],
                                               (2, 1)).T  # not copied, issue?
                    additional_hinges = np.tile(self.hinges[:, parent_idx], (2, 1)).T
                    additional_where = np.tile(self.where[:, parent_idx], (2, 1)).T

                    additional_covariates[parent_depth + 1, :] = cov
                    additional_nodes[parent_depth + 1, 0] = 0.0
                    additional_nodes[parent_depth + 1, 1] = eligible_knots[0]
                    additional_hinges[parent_depth + 1, 0] = False
                    additional_hinges[parent_depth + 1, 1] = True
                    additional_where[parent_depth + 1, :] = True

                    self.add_basis(
                        additional_covariates,
                        additional_nodes,
                        additional_hinges,
                        additional_where
                    )
                    chol = self.extend_fit(x, y, 2)
                    old_node = eligible_knots[0]

                    for new_node in eligible_knots[1:]:
                        updated_nodes = additional_nodes[:, 1]
                        updated_nodes[parent_depth + 1] = new_node
                        self.update_basis(
                            additional_covariates[:, 1],
                            updated_nodes,
                            additional_hinges[:, 1],
                            additional_where[:, 1]
                        )
                        chol = self.update_fit(x, y, chol, old_node, parent_idx)
                        old_node = new_node
                        if self.lof < basis_lof:
                            basis_lof = self.lof
                        if self.lof < best_lof:
                            best_lof = self.lof
                            addition_slice = slice(self.nbases - 2, self.nbases)
                            best_covariate = self.covariates[:, addition_slice].copy()
                            best_node = self.nodes[:, addition_slice].copy()
                            best_hinge = self.hinges[:, addition_slice].copy()
                            best_where = self.where[:, addition_slice].copy()
                    removal_slice = slice(self.nbases - 2, self.nbases)
                    self.remove_basis(removal_slice)
                    self.shrink_fit(x, y, removal_slice)
                candidate_queue[parent_idx] = best_lof - basis_lof
            for unselected_idx in np.argsort(candidate_queue)[self.max_ncandidates:]:
                candidate_queue[unselected_idx] += self.aging_factor
            if best_covariate is not None:
                self.add_basis(best_covariate, best_node, best_hinge, best_where)
                self.extend_fit(x, y, 2)
                candidate_queue.extend([0, 0])

    def prune_bases(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Prune the bases to the best fitting subset of the basis functions. By iteratively removing the basis
        that increases the lack of fit criterion the least. Equivalent to the backward pass in the Mars paper.

        Args:
            x: Data points. [n x d]
            y: Target values. [n]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]

        best_model = self.__dict__.copy()
        best_model_where = self.where.copy()
        # where is not copied which is an issue, basically all of them are not copied
        best_trimmed_model = self.__dict__.copy()
        best_trimmed_model_where = self.where.copy()

        best_lof = self.lof

        while len(self) > 1:
            best_trimmed_lof = np.inf

            previous_model = self.__dict__.copy()
            previous_where = self.where.copy()
            previous_fit = self.fit_matrix.copy()
            previous_covariance = self.covariance_matrix.copy()
            previous_right_hand_side = self.right_hand_side.copy()
            previous_fixed_mean = self.fixed_mean.copy()
            previous_candidate_mean = self.candidate_mean

            # First basis function (constant 1) cannot be excluded
            for basis_idx in range(1, self.nbases):
                self.remove_basis(slice(basis_idx, basis_idx + 1))
                self.shrink_fit(x, y, slice(basis_idx, basis_idx + 1))
                if self.lof < best_trimmed_lof:
                    best_trimmed_lof = self.lof
                    best_trimmed_model = self.__dict__.copy()
                    best_trimmed_model_where = self.where.copy()
                if self.lof < best_lof:
                    best_lof = self.lof
                    best_model = self.__dict__.copy()
                    best_model_where = self.where.copy()
                self.__dict__.update(previous_model)
                self.where = previous_where.copy()
                self.fit_matrix = previous_fit.copy()
                self.covariance_matrix = previous_covariance.copy()
                self.right_hand_side = previous_right_hand_side.copy()
                self.fixed_mean = previous_fixed_mean.copy()
                self.candidate_mean = previous_candidate_mean

            self.__dict__.update(best_trimmed_model)
            self.where = best_trimmed_model_where
            self.fit(x, y)
        self.__dict__.update(best_model)
        self.where = best_model_where
        self.fit(x, y)

    def find_bases(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Find the best fitting basis functions for the given data.

        Args:
            x: Data points. [n x d]
            y: Target values. [n]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]

        self.expand_bases(x, y)
        self.prune_bases(x, y)
