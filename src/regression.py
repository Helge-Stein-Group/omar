from typing import Self

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from src.regression_numba import decompose_addition, update_cholesky


class OMARS:
    """
    Open Multivariate Adaptive Regression Splines (OMARS) model.
    Can be used to determine multi-linear relations in data by finding the best fit of
    a sum of basis functions.
    Based on:
    - Friedman, J. (1991). Multivariate adaptive regression splines.
      The annals of statistics, 19(1), 1–67.
      http://www.jstor.org/stable/10.2307/2241837
    - Friedman, J. (1993). Fast MARS.
      Stanford University Department of Statistics,
      Technical Report No 110.
      https://statistics.stanford.edu/sites/default/files/LCS%20110.pdf

    To determine the bases use the find_bases method.  The fitted model can than be used
    to predict new data by calling the model with the data as argument.
    """

    def __init__(self,
                 max_nbases: int = 11,
                 max_ncandidates: int = 11,
                 aging_factor: float = 0.,
                 smoothness: float = 3):
        """
        Initialize the OMARS model.
        Args:
            max_nbases: Maximum number of basis functions that should be odd.
                        (excluding the constant function)
            max_ncandidates: Maximum number of recalculations per iteration.
                             (See Fast Mars paper)
            aging_factor: Determines how fast unused parent basis functions
                          need recalculation. (See Fast Mars paper)
            smoothness: Cost for each basis function optimization,
                        used to determine the lack of fit criterion.

        Other attributes:
            nbases: Number of basis functions (excluding the constant function).
            covariates: Covariates of the basis functions.
            nodes: Nodes of the basis functions.
            hinges: Hinges of the basis functions.
            where: Which parts of the aforementioned arrays are used.
                   (Basically determines the product length of the basis)
            coefficients: Coefficients for the superposition of bases.
            fit_matrix: Data matrix of the fitted data.
                        (Evaluation of basis functions for the data points)
            covariance_matrix: Covariance matrix of the fitted data.
            right_hand_side: Right hand side of the fitted data.
            lof: Lack of fit criterion.
            indices: Indices of the data points that need to be recalculated.
            update: Update of the fit matrix.
            update_mean: Mean of update attribute.
            basis_mean: Mean of the basis functions.
            y_mean: Mean of the target values.
        """
        self.max_nbases = max_nbases
        self.max_ncandidates = max_ncandidates
        self.aging_factor = aging_factor
        self.smoothness = smoothness

        # Bases (including the constant function)
        self.nbases = 1
        self.covariates = np.zeros((self.max_nbases, self.max_nbases), dtype=int)
        self.nodes = np.zeros((self.max_nbases, self.max_nbases), dtype=float)
        self.hinges = np.zeros((self.max_nbases, self.max_nbases), dtype=bool)
        self.where = np.zeros((self.max_nbases, self.max_nbases), dtype=bool)

        self.coefficients = np.empty(0, dtype=float)
        self.fit_matrix = np.empty((0, 0), dtype=float)
        self.covariance_matrix = np.empty((0, 0), dtype=float)
        self.right_hand_side = np.empty(0, dtype=float)

        self.lof = float()

        self.update = np.empty(0, dtype=float)
        self.update_mean = float()

        self.basis_mean = np.empty(0, dtype=float)
        self.y_mean = float()

    def _active_base_indices(self) -> np.ndarray:
        """
        Get the indices of the active basis functions.

        Returns:
            Indices of the active basis functions.
        """
        return np.where(np.any(self.where, axis=0))[0]

    def _data_matrix(self, x: np.ndarray, basis_indices: np.ndarray) -> tuple[
        np.ndarray, np.ndarray]:
        """
        Calculate the data matrix for the given data and basis functions, which is the
        evaluation of the basis functions for the data points. Returns the CENTERED
        data matrix.

        Args:
            x: Data points. [n x d]
            basis_indices: Indices of the basis functions to be evaluated.

        Returns:
            Centered Data matrix.
        """
        assert x.ndim == 2
        assert isinstance(basis_indices, np.ndarray)

        result = -self.nodes[:, basis_indices] + x[:, self.covariates[:, basis_indices]]
        np.maximum(0, result, where=self.hinges[:, basis_indices], out=result)

        data_matrix = result.prod(axis=1, where=self.where[:, basis_indices])

        data_matrix_mean = data_matrix.mean(axis=0)
        return data_matrix - data_matrix_mean, data_matrix_mean

    def __str__(self) -> str:
        """
        Describes the basis functions of the model.

        Returns:
            Description of the basis functions.
        """
        desc = "Basis functions: \n"
        for basis_idx in self._active_base_indices():
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

        fit_matrix, _ = self._data_matrix(x, self._active_base_indices())
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
        if i != 0:
            sub_model.coefficients = self.coefficients[i:i + 1]
        else:
            sub_model.coefficients = [self.y_mean]

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

    def _add_basis(self,
                   covariates: np.ndarray,
                   nodes: np.ndarray,
                   hinges: np.ndarray,
                   where: np.ndarray) -> None:
        """
        Add a basis functions to the model.

        Args:
            covariates: Covariates of the basis functions. [maxnbases x nadditions]
                        (On which dimension the basis functions are applied).
            nodes: Nodes of the basis functions. [maxnbases x nadditions]
            hinges: Hinges of the basis functions. [maxnbases x nadditions]
            where: Which parts of the aforementioned arrays are used.
                   [maxnbases x nadditions]
                   (Basically determines the product length of the basis)
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

        addition_slice = slice(self.nbases, self.nbases + covariates.shape[1])
        self.covariates[:, addition_slice] = covariates
        self.nodes[:, addition_slice] = nodes
        self.hinges[:, addition_slice] = hinges
        self.where[:, addition_slice] = where
        self.nbases += covariates.shape[1]

    def _update_basis(self, covariates: np.ndarray,
                      nodes: np.ndarray,
                      hinges: np.ndarray,
                      where: np.ndarray) -> None:
        """
        Update the latest basis function.

        Args:
            covariates: Covariates of the basis functions. [maxnbases x 1]
                        (On which dimension the basis functions are applied)
            nodes: Nodes of the basis functions. [maxnbases x 1]
            hinges: Hinges of the basis functions. [maxnbases x 1]
            where: Which parts of the aforementioned arrays are used. [maxnbases x 1]
                   (Basically determines the product length of the basis)
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

    def _remove_basis(self, removal_idx: int) -> None:
        """
        Remove basis function from the model.

        Args:
            removal_idx: Index of the basis functions to be removed.

        Notes:
            Only the where attributes is turned off, the other attributes
            (e.g. covariates) are deliberately not deleted to avoid recalculation in
            the expansion step.
        """
        assert isinstance(removal_idx, int) or isinstance(removal_idx, np.int64)

        self.where[:, removal_idx] = False
        self.nbases -= 1

    def _update_init(self, x: np.ndarray, old_node: float, parent_idx: int) -> None:
        """
        Initialize the update of the fit by precomputing the necessary update values,
        that allow for a fast update of the cholesky decomposition and
        therefore a faster least-squares fit.

        Args:
            x: Standardized Data points. [n x d]
            old_node: Previous node of the basis function.
            parent_idx: Index of the parent basis function. (including constant basis)
        """
        assert x.ndim == 2
        assert x.shape[0] == self.fit_matrix.shape[0]
        assert isinstance(old_node, float)
        assert isinstance(parent_idx, int) or isinstance(parent_idx, np.int64)

        prod_idx = self.where[:, self.nbases - 1].sum()
        new_node = self.nodes[prod_idx, self.nbases - 1]
        covariate = self.covariates[prod_idx, self.nbases - 1]

        self.update = x[:, covariate] - new_node
        self.update[x[:, covariate] >= old_node] = old_node - new_node
        self.update[x[:, covariate] < new_node] = 0

        if parent_idx != 0:  # Not Constant basis function, otherwise 1 anyway
            self.update *= self.fit_matrix[:, parent_idx - 1] + self.basis_mean[
                parent_idx - 1]

        self.update_mean = self.update.mean()
        self.update -= self.update_mean

    def _calculate_fit_matrix(self, x: np.ndarray) -> None:
        """
        Calculate the data matrix for the given data.

        Args:
            x: Data points. [n x d]
        """
        assert x.ndim == 2

        self.fit_matrix, self.basis_mean = self._data_matrix(x,
                                                             self._active_base_indices())

    def _extend_fit_matrix(self, x: np.ndarray, nadditions: int) -> None:
        """
        Extend the data matrix by evaluating the newly added basis functions.

        Args:
            x: Standardized Data points. [n x d]
            nadditions: Number of newly added basis functions.
        """
        assert x.ndim == 2
        assert isinstance(nadditions, int)

        ext_indices = self._active_base_indices()[-nadditions:]
        if self.fit_matrix.size:
            fit_matrix_ext, basis_mean_ext = self._data_matrix(x, ext_indices)
            self.fit_matrix = np.hstack((
                self.fit_matrix,
                fit_matrix_ext
            ))
            self.basis_mean = np.append(self.basis_mean, basis_mean_ext)
        else:
            self.fit_matrix, self.basis_mean = self._data_matrix(x, ext_indices)

    def _update_fit_matrix(self) -> None:
        """
        Update the data matrix by adding the update attribute to the last column.
        """
        self.fit_matrix[:, -1] += self.update
        self.basis_mean[-1] += self.update_mean

    def _calculate_covariance_matrix(self) -> None:
        """
        Calculate the covariance matrix of the data matrix,
        which is the chosen formulation of the least-squares problem.
        """
        self.covariance_matrix = self.fit_matrix.T @ self.fit_matrix
        '''
        ATTENTION !!!
        The basis through the forward pass does not need to be linearly independent, 
        which may lead to a singular matrix. Here we can either add a small diagonal 
        value or set the dependent coefficients to 0 to obtain a unique solution again.
        '''
        self.covariance_matrix += np.eye(self.covariance_matrix.shape[0]) * 1e-8

    def _extend_covariance_matrix(self, nadditions: int) -> None:
        """
        Extend the covariance matrix by accounting for the newly added basis functions,
        expects an extended fit matrix.

        Args:
            nadditions: Number of newly added basis functions.
        """
        assert isinstance(nadditions, int)

        covariance_extension = self.fit_matrix.T @ self.fit_matrix[:, -nadditions:]

        self.covariance_matrix = np.hstack((self.covariance_matrix,
                                            covariance_extension[:-nadditions]))
        self.covariance_matrix = np.vstack((self.covariance_matrix,
                                            covariance_extension.T))
        for j in range(1, nadditions + 1):
            self.covariance_matrix[-j, -j] += 1e-8

    def _update_covariance_matrix(self) -> np.ndarray:
        """
        Update the covariance matrix. Expects update initialisation and
        an updated fit matrix.
        """
        covariance_addition = np.zeros_like(self.covariance_matrix[-1, :])
        covariance_addition[:-1] = self.update @ self.fit_matrix[:, :-1]
        covariance_addition[-1] = 2 * self.fit_matrix[:, -1] @ self.update
        covariance_addition[-1] -= self.update @ self.update

        self.covariance_matrix[-1, :-1] += covariance_addition[:-1]
        self.covariance_matrix[:, -1] += covariance_addition

        return covariance_addition

    def _calculate_right_hand_side(self, y: np.ndarray) -> None:
        """
        Calculate the right hand side of the least-squares problem.

        Args:
            y: Standardized Target values. [n]
        """
        assert y.ndim == 1
        assert y.shape[0] == self.fit_matrix.shape[0]
        assert self.fit_matrix.shape[1] == self.covariance_matrix.shape[0]

        self.y_mean = y.mean()
        self.right_hand_side = self.fit_matrix.T @ (y - self.y_mean)

    def _extend_right_hand_side(self, y: np.ndarray, nadditions: int) -> None:
        """
        Extend the right hand side of the least-squares problem by accounting for the
        newly added basis functions.

        Args:
            y: Standardized Target values. [n]
            nadditions: Number of newly added basis functions.
        """
        assert y.ndim == 1
        assert y.shape[0] == self.fit_matrix.shape[0]
        assert self.fit_matrix.shape[1] == self.covariance_matrix.shape[0]

        if not self.right_hand_side.size:
            self.y_mean = y.mean()
        self.right_hand_side = np.append(self.right_hand_side,
                                         self.fit_matrix[:, -nadditions:].T @ (
                                                 y - self.y_mean))

    def _update_right_hand_side(self, y: np.ndarray) -> None:
        """
        Update the right hand side according to the basis update.

        Args:
            y: Standardized Target values. [n]
        """
        assert y.ndim == 1
        assert y.shape[0] == self.fit_matrix.shape[0]

        self.right_hand_side[-1] += self.update.T @ (y - self.y_mean)

    def _generalised_cross_validation(self, y: np.ndarray) -> None:
        """
        Calculate the generalised cross validation criterion, the lack of fit criterion.

        Args:
            y: Standardized Target values. [n]
        """
        assert y.ndim == 1

        y_pred = self.fit_matrix @ self.coefficients + self.y_mean
        mse = np.sum((y - y_pred) ** 2)

        c_m = self.nbases + 1 + self.smoothness * (self.nbases - 1)
        self.lof = mse / len(y) / (1 - c_m / len(y) + 1e-6) ** 2

    def _fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Calculate the least-squares fit of the current model from scratch.

        Args:
            x: Standardized Data points. [n x d]
            y: Standardized Target values. [n]

        Returns:
            Cholesky decomposition of the covariance matrix. [nbases x nbases]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]

        self._calculate_fit_matrix(x)
        self._calculate_covariance_matrix()
        self._calculate_right_hand_side(y)

        chol, lower = cho_factor(self.covariance_matrix, lower=True)

        self.coefficients = cho_solve((chol, lower), self.right_hand_side)

        self._generalised_cross_validation(y)

        return np.tril(chol)

    def _extend_fit(self, x: np.ndarray, y: np.ndarray, nadditions: int):
        """
        Extend the least-squares fit. Expects the correct fit matrix, covariance matrix,
        and right hand side of the smaller model.

        Args:
            x: Standardized Data points. [n x d]
            y: Standardized Target values. [n]
            nadditions: Number of newly added basis functions.

        Returns:
            Cholesky decomposition of the covariance matrix. [nbases x nbases]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert isinstance(nadditions, int)

        self._extend_fit_matrix(x, nadditions)
        self._extend_covariance_matrix(nadditions)
        self._extend_right_hand_side(y, nadditions)

        chol, lower = cho_factor(self.covariance_matrix, lower=True)

        self.coefficients = cho_solve((chol, lower), self.right_hand_side)

        self._generalised_cross_validation(y)

        return np.tril(chol)

    def _update_fit(self, x: np.ndarray, y: np.ndarray, chol: np.ndarray,
                    old_node: float, parent_idx: int) -> np.ndarray:
        """
        Update the least-squares fit. Expects the correct fit matrix, covariance matrix,
        and right hand side of the previous model. The only change in the bases occurred
        in the node of the last basis function, which is smaller than the old node.

        Args:
            x: Standardized Data points. [n x d]
            y: Standardized Target values. [n]
            chol: Cholesky decomposition of the covariance matrix. [nbases x nbases]
            old_node: Previous node of the basis function.
            parent_idx: Index of the parent basis function. (including constant basis)

        Returns:
            Cholesky decomposition of the covariance matrix. [nbases x nbases]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert chol.shape[0] == chol.shape[1] == self.nbases - 1
        assert self.nodes[
                   np.sum(self.where[:, self.nbases - 1]), self.nbases - 1] <= old_node
        assert parent_idx < self.nbases

        self._update_init(x, old_node, parent_idx)
        self._update_fit_matrix()
        covariance_addition = self._update_covariance_matrix()

        if covariance_addition.any():
            eigenvalues, eigenvectors = decompose_addition(covariance_addition)
            chol = update_cholesky(chol, eigenvectors, eigenvalues)

        self._update_right_hand_side(y)

        self.coefficients = cho_solve((chol, True), self.right_hand_side)

        self._generalised_cross_validation(y)

        return chol

    def _shrink_fit(self, x: np.ndarray, y: np.ndarray,
                    removal_idx: int) -> np.ndarray:
        """
        Shrink the least-squares fit. Expects the correct fit matrix, covariance matrix,
        and right hand side of the larger model.

        Args:
            x: Standardized Data points. [n x d]
            y: Standardized Target values.  [n]
            removal_idx: Index of the basis function to be removed.

        Returns:
            Cholesky decomposition of the covariance matrix. [nbases x nbases]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert isinstance(removal_idx, int)

        removal_idx -= 1  # The data matrices etc. do not include the constant function.

        self.fit_matrix = np.delete(self.fit_matrix, removal_idx, axis=1)
        self.covariance_matrix = np.delete(self.covariance_matrix, removal_idx,
                                           axis=0)
        self.covariance_matrix = np.delete(self.covariance_matrix, removal_idx,
                                           axis=1)

        self.basis_mean = np.delete(self.basis_mean, removal_idx)

        self.right_hand_side = np.delete(self.right_hand_side, removal_idx)

        if self.fit_matrix.size:
            chol, lower = cho_factor(self.covariance_matrix, lower=True)

            self.coefficients = cho_solve((chol, True), self.right_hand_side)

            self._generalised_cross_validation(y)
        else:
            chol = None
            mse = np.sum((y - self.y_mean) ** 2)
            self.lof = mse / len(y) / (1 - 1 / len(y)) ** 2

        return chol

    def _expand_bases(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Expand the bases to the maximum allowed number of basis functions.
        By iteratively adding the basis that reduces the lack of fit criterion the most.
        Equivalent to the forward pass in the Mars paper, including the adaptions
        in the Fast Mars paper.

        Args:
            x: Standardized Data points. [n x d]
            y: Standardized Target values. [n]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]

        covariates = set(range(x.shape[1]))
        candidate_queue = [0.]  # One for the constant function
        for _ in range(self.max_nbases // 2):
            best_lof = np.inf
            best_covariate = None
            best_node = None
            best_hinge = None
            best_where = None
            basis_lofs = {}
            for parent_idx in np.argsort(candidate_queue)[:-self.max_ncandidates - 1:-1]:
                eligible_covariates = covariates - set(
                    self.covariates[self.where[:, parent_idx], parent_idx])
                basis_lof = np.inf
                parent_depth = self.where[:, parent_idx].sum()
                for cov in eligible_covariates:
                    if parent_idx == 0:  # constant function
                        eligible_knots = x[:, cov].copy()
                    else:
                        eligible_knots = x[
                            np.where(self.fit_matrix[:, parent_idx - 1] > 0)[0], cov]
                    eligible_knots[::-1].sort()
                    additional_covariates = np.tile(self.covariates[:, parent_idx], (2, 1)).T
                    additional_nodes = np.tile(self.nodes[:, parent_idx], (2, 1)).T
                    additional_hinges = np.tile(self.hinges[:, parent_idx], (2, 1)).T
                    additional_where = np.tile(self.where[:, parent_idx], (2, 1)).T

                    additional_covariates[parent_depth + 1, :] = cov
                    additional_nodes[parent_depth + 1, 0] = 0.0
                    additional_nodes[parent_depth + 1, 1] = eligible_knots[0]
                    additional_hinges[parent_depth + 1, 0] = False
                    additional_hinges[parent_depth + 1, 1] = True
                    additional_where[parent_depth + 1, :] = True

                    self._add_basis(
                        additional_covariates,
                        additional_nodes,
                        additional_hinges,
                        additional_where
                    )
                    chol = self._extend_fit(x, y, 2)
                    old_node = eligible_knots[0]
                    if self.lof < basis_lof:
                        basis_lof = self.lof
                    for new_node in eligible_knots[1:]:
                        updated_nodes = additional_nodes[:, 1]
                        updated_nodes[parent_depth + 1] = new_node
                        self._update_basis(
                            additional_covariates[:, 1],
                            updated_nodes,
                            additional_hinges[:, 1],
                            additional_where[:, 1]
                        )
                        chol = self._update_fit(x, y, chol, old_node, parent_idx)
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
                    for i in range(2):
                        self._remove_basis(self.nbases - 1)
                        # CARE remove basis decrements nbases
                        self._shrink_fit(x, y, self.nbases)
                basis_lofs[parent_idx] = basis_lof
            for idx, lof in basis_lofs.items():
                candidate_queue[idx] = best_lof - lof
            for unselected_idx in np.argsort(candidate_queue)[self.max_ncandidates:]:
                candidate_queue[unselected_idx] += self.aging_factor
            if best_covariate is not None:  # and spec not in model (once cleaned up)
                self._add_basis(best_covariate, best_node, best_hinge, best_where)
                self._extend_fit(x, y, 2)
                candidate_queue.extend([0, 0])
            else:  # We achieved the best we can do
                return

    def _prune_bases(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        Prune the bases to the best fitting subset of the basis functions.
        By iteratively removing the basis that increases the lack of fit criterion
        the least. Equivalent to the backward pass in the Mars paper.

        Args:
            x: Standardized Data points. [n x d]
            y: Standardized Target values. [n]
        """
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]

        best_model = self.__dict__.copy()
        best_model_where = self.where.copy()

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
            previous_basis_mean = self.basis_mean.copy()

            # Constant basis function cannot be excluded
            for basis_idx in range(1, self.nbases):
                self._remove_basis(basis_idx)
                self._shrink_fit(x, y, basis_idx)
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
                self.basis_mean = previous_basis_mean.copy()

            self.__dict__.update(best_trimmed_model)
            self.where = best_trimmed_model_where
            self._fit(x, y)
        self.__dict__.update(best_model)
        self.where = best_model_where
        self._fit(x, y)

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

        self._expand_bases(x, y)
        self._prune_bases(x, y)
