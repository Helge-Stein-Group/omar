from enum import Enum

import numpy as np
from beartype import beartype
from numba import njit
from jaxtyping import Float, Integer, jaxtyped
from beartype import beartype
from scipy.linalg import cho_factor, cho_solve

import fortran_backend as fortran


class Backend(Enum):
    """
    Backend for the omar model.
    """
    FORTRAN = 1
    PYTHON = 2


class OMAR:
    """
    Open Multivariate Adaptive Regression Splines (omar) model.
    Use it to find localised, linear relationships in your data.
    Based on:
    - Friedman, J. (1991). Multivariate adaptive regression splines.
      The annals of statistics, 19(1), 1–67.
      http://www.jstor.org/stable/10.2307/2241837
    - Friedman, J. (1993). Fast MARS.
      Stanford University Department of Statistics,
      Technical Report No 110.
      https://statistics.stanford.edu/sites/default/files/LCS%20110.
    - Oswin Krause and Christian Igel. “A More Efficient Rank-one Covariance Matrix Update for Evolution Strategies”.
      In Proceedings of the 2015 ACM Conference on Foundations of Genetic Algorithms XIII (FOGA ‘15). (2015).
      Association for Computing Machinery, New York, NY, USA, 129–136. https://doi.org/10.1145/2725494.2725496
    """

    @jaxtyped(typechecker=beartype)
    def __init__(self,
                 max_nbases: np.int64 | int = 11,
                 max_ncandidates: np.int64 | int = 11,
                 aging_factor: float = 0.,
                 penalty: float = 3,
                 backend: Backend = Backend.FORTRAN):
        """
        Initialize the omar model.
        Args:
            max_nbases: Maximum number of basis functions. Since they are added in pairs, this number should be odd.
            max_ncandidates: Maximum queue length for parent candidates. (See Fast Mars paper)
            aging_factor: Determines how fast unused parent basis functions need recalculation. (See Fast Mars paper)
            penalty: Cost for each basis function optimization, parameter of the generalized cross validation.
            backend: Backend for the model. "Fortran" should be chosen most of the time since it's way faster.

        Other attributes:
            nbases: Number of basis functions.
            mask: Flags active cells in the arrays describing the basis functions. Columns are multiplicative, while
            the rows are additive.
            truncated: Flags which basis function planes are truncated. Columns are multiplicative, while
            the rows are additive.
            cov: Determines the dimension this basis function is acting on. Columns are multiplicative, while
            the rows are additive.
            root: Determines the root of the basis function. Columns are multiplicative, while
            the rows are additive.
            coefficients: Coefficients for the superposition of bases.
            y_mean: Mean of the response variables, which makes it also the coefficient for the first (constant) basis.
        """
        assert max_nbases % 2 == 1, "Parameter \"max_nbases\" should be odd."
        assert max_ncandidates <= max_nbases, ("""Maximum queue length for parent candidates should be less than the
                                               maximum number of basis functions.""")

        self.max_nbases = max_nbases
        self.max_ncandidates = max_ncandidates
        self.aging_factor = aging_factor
        self.penalty = penalty
        self.backend = backend

        self.nbases = 1
        self.mask = np.zeros((self.max_nbases, self.max_nbases), dtype=bool)
        self.truncated = np.zeros((self.max_nbases, self.max_nbases), dtype=bool)
        self.cov = np.zeros((self.max_nbases, self.max_nbases), dtype=int)
        self.root = np.zeros((self.max_nbases, self.max_nbases), dtype=float)
        self.coefficients = np.empty(0, dtype=float)
        self.y_mean = float()

    @jaxtyped(typechecker=beartype)
    def __call__(self, x: Float[np.ndarray, "N d"]) -> Float[np.ndarray, "N"]:
        """
        Predict the response variables for the given predictor variables.

        Args:
            x: Predictor Variables.

        Returns:
            Predicted response variables.
        """
        pred = self.y_mean * np.ones(x.shape[0])
        if self._active_base_indices().any():
            data_matrix, _ = self._data_matrix(x, self._active_base_indices())
            centered_y_pred = data_matrix @ self.coefficients
            pred += centered_y_pred
        return pred

    @jaxtyped(typechecker=beartype)
    def __str__(self) -> str:
        """
        Describes the basis functions of the model.

        Returns:
            Description of the basis functions.
        """
        desc = "omar (Open Multivariate Adaptive Splines Regression) Model\n"
        desc += "Basis functions: \n"
        desc += f"{self.y_mean} * 1 + \n"
        for basis_idx in self._active_base_indices():
            for func_idx in range(self.max_nbases):
                if self.mask[func_idx, basis_idx]:
                    truncated = self.truncated[func_idx, basis_idx]
                    cov = self.cov[func_idx, basis_idx]
                    root = self.root[func_idx, basis_idx]

                    desc += f"{self.coefficients[basis_idx]:.2f} * (x[{cov}] - {root}){u'\u208A' if truncated else ''}"
            if np.any(self.mask[:, basis_idx]):
                desc += " + \n"
        return desc[:-4]

    @jaxtyped(typechecker=beartype)
    def __len__(self) -> np.int64 | int:
        """
        Number of basis functions.

        Returns:
            Number of basis functions.
        """
        return self.nbases

    @beartype
    @jaxtyped(typechecker=beartype)
    def __getitem__(self, i: np.int64 | int) -> "OMAR":
        """
        Return a submodel with only the i-th basis function.

        Args:
            i: Index of the basis function.

        Returns:
            Submodel with only the i-th basis function.
        """
        sub_model = OMAR()
        sub_model.nbases = 1
        sub_model.mask[:, i:i + 1] = self.mask[:, i:i + 1]
        sub_model.truncated[:, i:i + 1] = self.truncated[:, i:i + 1]
        sub_model.cov[:, i:i + 1] = self.cov[:, i:i + 1]
        sub_model.root[:, i:i + 1] = self.root[:, i:i + 1]

        if i != 0:
            sub_model.coefficients = self.coefficients[i:i + 1]
            sub_model.nbases = 2

        sub_model.y_mean = self.y_mean
        return sub_model

    @beartype
    @jaxtyped(typechecker=beartype)
    def __eq__(self, other: "OMAR") -> bool | np.bool:
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
            np.array_equal(self.mask[*self_idx], other.mask[*other_idx]) and \
            np.array_equal(self.truncated[*self_idx], other.truncated[*other_idx]) and \
            np.array_equal(self.cov[*self_idx], other.cov[*other_idx]) and \
            np.array_equal(self.root[*self_idx], other.root[*other_idx])

    @jaxtyped(typechecker=beartype)
    def _active_base_indices(self) -> Integer[np.ndarray, "{self.nbases}-1"]:
        """
        Get the indices of the active basis functions.

        Returns:
            Indices of the active basis functions.
        """
        if self.backend is Backend.PYTHON:
            return np.where(np.any(self.mask, axis=0))[0]
        elif self.backend is Backend.FORTRAN:
            # Fortran indexes from 1
            return fortran.backend.active_base_indices(self.mask, self.nbases) - 1

    @jaxtyped(typechecker=beartype)
    def _data_matrix(self, x: Float[np.ndarray, "N d"], basis_indices: Integer[np.ndarray, "{self.nbases}-1"]) \
            -> tuple[
                Float[np.ndarray, "N {self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1"]
            ]:
        """
        Evaluate the selected part of the model on the predictor variables x.

        Args:
            x: Predictor variables points.
            basis_indices: Indices of the basis functions to be evaluated.

        Returns:
            Centered data matrix, mean of the data matrix.
        """
        if self.backend is Backend.PYTHON:
            bases = x[:, self.cov[:, basis_indices]] - self.root[:, basis_indices]
            np.maximum(0, bases, where=self.truncated[:, basis_indices], out=bases)

            data_matrix = bases.prod(axis=1, where=self.mask[:, basis_indices])

            data_matrix_mean = data_matrix.mean(axis=0)
            data_matrix -= data_matrix_mean
        elif self.backend is Backend.FORTRAN:
            # Fortran indexes from 1
            data_matrix, data_matrix_mean = fortran.backend.data_matrix(x, basis_indices + 1, self.mask,
                                                                        self.truncated, self.cov + 1, self.root)
        else:
            raise NotImplementedError("Backend not implemented.")

        return data_matrix, data_matrix_mean

    @jaxtyped(typechecker=beartype)
    def _covariance_matrix(self, data_matrix: Float[np.ndarray, "N {self.nbases}-1"]) \
            -> Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"]:
        """
        Calculate the covariance matrix of the data matrix, which is the left hand side of the normal
        equations.

        Args:
            data_matrix: Centered data matrix.

        Returns:
            Covariance matrix.

        Notes:
            The bases in the expansion do not need to be linearly independent, which can lead to a singular matrix.
            While this is usually solved by pivoting, here a small diagonal value is added to the diagonal to obtain
            a unique solution again.
        """
        if self.backend is Backend.PYTHON:
            covariance_matrix = data_matrix.T @ data_matrix
            covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-8
        elif self.backend is Backend.FORTRAN:
            covariance_matrix = fortran.backend.covariance_matrix(data_matrix)
        else:
            raise NotImplementedError("Backend not implemented.")

        return covariance_matrix

    @jaxtyped(typechecker=beartype)
    def _rhs(self, y: Float[np.ndarray, "N"], data_matrix: Float[np.ndarray, "N {self.nbases}-1"]) \
            -> Float[np.ndarray, "{self.nbases}-1"]:
        """
        Calculate the right hand side of the normal equations.

        Args:
            y: Response variables.
            data_matrix: Centered data matrix.

        Returns:
            Right hand side of the normal equations.
        """
        if self.backend is Backend.PYTHON:
            rhs = data_matrix.T @ (y - self.y_mean)
        elif self.backend is Backend.FORTRAN:
            rhs = fortran.backend.rhs(y, self.y_mean, data_matrix)
        else:
            raise NotImplementedError("Backend not implemented.")

        return rhs

    @jaxtyped(typechecker=beartype)
    def _coefficients(self,
                      covariance_matrix: Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"],
                      rhs: Float[np.ndarray, "{self.nbases}-1"]) \
            -> tuple[
                Float[np.ndarray, "{self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"]
            ]:
        """
        Calculate the coefficients of the model by solving the normal equations via Cholesky decomposition.

        Args:
            covariance_matrix: Covariance matrix.
            rhs: Right hand side of the normal equations

        Returns:
            Coefficients of the model, Cholesky decomposition of the covariance matrix.
        """
        if self.backend is Backend.PYTHON:
            chol, lower = cho_factor(covariance_matrix, lower=True)
            self.coefficients = cho_solve((chol, lower), rhs)
        elif self.backend is Backend.FORTRAN:
            self.coefficients, chol = fortran.backend.coefficients(covariance_matrix, rhs)
        else:
            raise NotImplementedError("Backend not implemented.")

        return self.coefficients, np.tril(chol)

    @jaxtyped(typechecker=beartype)
    def _generalised_cross_validation(self,
                                      y: Float[np.ndarray, "N"],
                                      data_matrix: Float[np.ndarray, "N {self.nbases}-1"],
                                      chol: Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"]) -> float:
        """
        Calculate the generalised cross validation criterion, the lack of fit criterion. The rank can be computed
        efficiently from the diagonal of the Cholesky decomposition(which is triangular).

        Args:
            y: Response variables.
            data_matrix: Centered data matrix.
            chol: Cholesky decomposition of the covariance matrix.

        Returns:
            Lack of fit criterion.
        """
        if self.backend is Backend.PYTHON:
            if data_matrix.size != 0:
                y_pred = data_matrix @ self.coefficients + self.y_mean
                c_m = np.sum(np.abs(np.diag(chol)) != 0) * (1 + self.penalty) + 1 - self.penalty
            else:
                y_pred = self.y_mean
                c_m = 1 - self.penalty
            mse = np.sum((y - y_pred) ** 2) / len(y)

            if c_m != len(y):
                lof = mse / (1 - c_m / len(y)) ** 2
            else:
                print("Infinite lack of fit criterion, as the rank of the covariance matrix is equal to the number of \
                       response variables.")
                lof = np.inf

        elif self.backend is Backend.FORTRAN:
            lof = fortran.backend.generalised_cross_validation(y, self.y_mean, data_matrix, chol, self.coefficients,
                                                               self.penalty)
        else:
            raise NotImplementedError("Backend not implemented.")

        return lof

    @jaxtyped(typechecker=beartype)
    def _fit(self, x: Float[np.ndarray, "N d"], y: Float[np.ndarray, "N"]) \
            -> tuple[
                Float[np.ndarray, "N {self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1"],
                float
            ]:
        """
        Calculate the least-squares fit of the current model from scratch.

        Args:
            x: Predictor variables.
            y: Response variables.

        Returns:
            Centered data matrix, mean of the data matrix, covariance matrix, right hand side of the normal equations,
            Cholesky decomposition of the covariance matrix, coefficients of the model, lack of fit criterion
        """
        if self.backend is Backend.PYTHON:
            data_matrix, data_matrix_mean = self._data_matrix(x, self._active_base_indices())
            covariance_matrix = self._covariance_matrix(data_matrix)
            rhs = self._rhs(y, data_matrix)
            self.coefficients, chol = self._coefficients(covariance_matrix, rhs)
            lof = self._generalised_cross_validation(y, data_matrix, chol)
        elif self.backend is Backend.FORTRAN:
            # Fortran indexes from 1
            data_matrix, data_matrix_mean, covariance_matrix, rhs, chol, self.coefficients, lof = \
                fortran.backend.fit(x, y, self.y_mean, self.nbases, self.mask, self.truncated, self.cov + 1,
                                    self.root, self.penalty)
            chol = np.tril(chol)
        else:
            raise NotImplementedError("Backend not implemented.")

        return data_matrix, data_matrix_mean, covariance_matrix, rhs, chol, self.coefficients, lof

    @jaxtyped(typechecker=beartype)
    def _update_init(self,
                     x: Float[np.ndarray, "N d"],
                     data_matrix: Float[np.ndarray, "N {self.nbases}-1"],
                     data_matrix_mean: Float[np.ndarray, "{self.nbases}-1"],
                     prev_root: float, parent_idx: np.int64 | int) -> tuple[Float[np.ndarray, "N"], float]:
        """
        Initialize the update of the fit by precomputing the necessary update values, that allow for a fast update of
        the cholesky decomposition and therefore a faster least-squares fit.

        Args:
            x: Predictor Variables.
            data_matrix: Centered data matrix.
            data_matrix_mean: Mean of the data matrix.
            prev_root: Previous root of the basis function.
            parent_idx: Index of the parent basis function. (including constant basis)

        Returns:
            Update vector, mean of the update vector.
        """
        if self.backend is Backend.PYTHON:
            prod_idx = self.mask[:, self.nbases - 1].sum()
            root = self.root[prod_idx, self.nbases - 1]
            cov = self.cov[prod_idx, self.nbases - 1]

            update = x[:, cov] - root
            update[x[:, cov] >= prev_root] = prev_root - root
            update[x[:, cov] < root] = 0

            if parent_idx != 0:  # Not Constant basis function, otherwise 1 anyway
                update *= data_matrix[:, parent_idx - 1] + data_matrix_mean[parent_idx - 1]

            update_mean = update.mean()
            update -= update_mean
        elif self.backend is Backend.FORTRAN:
            # Fortran indexes from 1
            update, update_mean = fortran.backend.update_init(x, data_matrix, data_matrix_mean, prev_root,
                                                              parent_idx, self.nbases, self.mask, self.cov + 1,
                                                              self.root)
        else:
            raise NotImplementedError("Backend not implemented.")
        return update, update_mean

    @jaxtyped(typechecker=beartype)
    def _update_data_matrix(self,
                            data_matrix: Float[np.ndarray, "N {self.nbases}-1"],
                            data_matrix_mean: Float[np.ndarray, "{self.nbases}-1"],
                            update: Float[np.ndarray, "N"],
                            update_mean: float) \
            -> tuple[
                Float[np.ndarray, "N {self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1"]
            ]:
        """
        Update the data matrix to the latest root location.

        Args:
            data_matrix: Centered data matrix.
            data_matrix_mean: Mean of the data matrix.
            update: Update vector.
            update_mean: Mean of the update vector.

        Returns:
            Updated data matrix, mean of the updated data matrix.
        """
        if self.backend is Backend.PYTHON:
            data_matrix[:, -1] += update
            data_matrix_mean[-1] += update_mean
        elif self.backend is Backend.FORTRAN:
            # Fortran updates in place
            fortran.backend.update_data_matrix(data_matrix, data_matrix_mean, update, update_mean)
        else:
            raise NotImplementedError("Backend not implemented.")

        return data_matrix, data_matrix_mean

    @jaxtyped(typechecker=beartype)
    def _update_covariance_matrix(self,
                                  covariance_matrix: Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"],
                                  data_matrix: Float[np.ndarray, "N {self.nbases}-1"],
                                  update: Float[np.ndarray, "N"]) \
            -> tuple[
                Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1"]
            ]:
        """
        Update the covariance matrix to the latest root location.

        Args:
            covariance_matrix: Covariance matrix.
            data_matrix: Centered data matrix.
            update: Update vector.

        Returns:
            Updated covariance matrix, addition to the covariance matrix.
        """
        if self.backend is Backend.PYTHON:
            covariance_addition = np.zeros_like(covariance_matrix[-1, :])
            covariance_addition[:-1] = update @ data_matrix[:, :-1]
            covariance_addition[-1] = 2 * data_matrix[:, -1] @ update
            covariance_addition[-1] -= update @ update

            covariance_matrix[-1, :-1] += covariance_addition[:-1]
            covariance_matrix[:, -1] += covariance_addition
        elif self.backend is Backend.FORTRAN:
            # Fortran updates in place
            covariance_addition = fortran.backend.update_covariance_matrix(covariance_matrix, data_matrix,
                                                                           update)
        else:
            raise NotImplementedError("Backend not implemented.")

        return covariance_matrix, covariance_addition

    @jaxtyped(typechecker=beartype)
    def _update_rhs(self,
                    rhs: Float[np.ndarray, "{self.nbases}-1"],
                    update: Float[np.ndarray, "N"],
                    y: Float[np.ndarray, "N"]) \
            -> Float[np.ndarray, "{self.nbases}-1"]:
        """
        Update the right hand side to the latest root location.

        Args:
            rhs: Right hand side of the normal equations.
            update: Update vector.
            y: Response variables.

        Returns:
            Updated right hand side.
        """
        if self.backend is Backend.PYTHON:
            rhs[-1] += update.T @ (y - self.y_mean)
        elif self.backend is Backend.FORTRAN:
            # Fortran updates in place
            fortran.backend.update_rhs(rhs, update, y, self.y_mean)
        return rhs

    @jaxtyped(typechecker=beartype)
    def _update_coefficients(self,
                             chol: Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"],
                             covariance_addition: Float[np.ndarray, "{self.nbases}-1"],
                             rhs: Float[np.ndarray, "{self.nbases}-1"]) \
            -> tuple[
                Float[np.ndarray, "{self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"]
            ]:
        """
        Update the Choelsky decomposition and coefficients of the model to the least-squares fit given the new root.

        Args:
            chol: Previous Cholesky decomposition of the covariance matrix.
            covariance_addition: Addition to the covariance matrix.
            rhs: Updated Right hand side of the normal equations.

        Returns:
            Updated coefficients, updated Cholesky decomposition.
        """
        if self.backend is Backend.PYTHON:
            if np.any(covariance_addition):
                eigenvalues, eigenvectors = decompose_addition(covariance_addition)
                for val, vec in zip(eigenvalues, eigenvectors):
                    chol = update_cholesky(chol, vec, val)
            self.coefficients = cho_solve((chol, True), rhs)
        elif self.backend is Backend.FORTRAN:
            # Fortran updates in place
            fortran.backend.update_coefficients(self.coefficients, chol, covariance_addition, rhs)
        else:
            raise NotImplementedError("Backend not implemented.")

        return self.coefficients, np.tril(chol)

    @jaxtyped(typechecker=beartype)
    def _update_fit(self,
                    data_matrix: Float[np.ndarray, "N {self.nbases}-1"],
                    data_matrix_mean: Float[np.ndarray, "{self.nbases}-1"],
                    covariance_matrix: Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"],
                    rhs: Float[np.ndarray, "{self.nbases}-1"],
                    chol: Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"],
                    x: Float[np.ndarray, "N d"],
                    y: Float[np.ndarray, "N"],
                    prev_root: float,
                    parent_idx: np.int64 | int) \
            -> tuple[
                Float[np.ndarray, "N {self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1 {self.nbases}-1"],
                Float[np.ndarray, "{self.nbases}-1"],
                float
            ]:
        """
        Update the model to the latest root location and determine the least-squares fit.

        Args:
            data_matrix: Centered data matrix.
            data_matrix_mean: Mean of the data matrix.
            covariance_matrix: Covariance matrix.
            rhs: Right hand side of the normal equations.
            chol: Cholesky decomposition of the covariance matrix.
            x: Predictor Variables.
            y: Response Variables.
            prev_root: Previous root of the basis function.
            parent_idx: Index of the parent basis function. (including constant basis)

        Returns:
            Centered data matrix, mean of the data matrix, covariance matrix, right hand side of the normal equations,
            Cholesky decomposition of the covariance matrix, coefficients of the model, lack of fit criterion
        """
        if self.backend is Backend.PYTHON:
            update, update_mean = self._update_init(x, data_matrix, data_matrix_mean, prev_root, parent_idx)
            data_matrix, data_matrix_mean = self._update_data_matrix(data_matrix, data_matrix_mean, update, update_mean)
            covariance_matrix, covariance_addition = self._update_covariance_matrix(covariance_matrix, data_matrix,
                                                                                    update)
            rhs = self._update_rhs(rhs, update, y)
            self.coefficients, chol = self._update_coefficients(chol, covariance_addition, rhs)

            lof = self._generalised_cross_validation(y, data_matrix, chol)
        elif self.backend is Backend.FORTRAN:
            # Fortran updates in place, and indexes from 1
            data_matrix = np.asfortranarray(data_matrix)
            data_matrix_mean = np.asfortranarray(data_matrix_mean)
            covariance_matrix = np.asfortranarray(covariance_matrix)
            rhs = np.asfortranarray(rhs)
            chol = np.asfortranarray(chol)
            self.coefficients = np.asfortranarray(self.coefficients)
            lof = fortran.backend.update_fit(data_matrix, data_matrix_mean, covariance_matrix, rhs, chol,
                                             self.coefficients, x, y, prev_root, parent_idx + 1, self.y_mean,
                                             self.nbases, self.penalty, self.mask, self.cov + 1, self.root)
            chol = np.tril(chol)
        else:
            raise NotImplementedError("Backend not implemented.")

        return data_matrix, data_matrix_mean, covariance_matrix, rhs, chol, self.coefficients, lof

    @jaxtyped(typechecker=beartype)
    def _add_bases(self, parent: np.int64 | int, cov: np.int64 | int, root: float) -> None:
        """
        Add two bases functions to model, one truncated and one linear.

        Args:
            parent: Index of the parent basis function.
            cov: Determines the dimension this basis function is acting on.
            root: Determines the root of the basis function.
        """
        if self.backend is Backend.PYTHON:
            parent_depth = self.mask[:, parent].sum()

            self.mask[:, self.nbases - 2: self.nbases] = np.tile(self.mask[:, parent:parent + 1], (1, 2))
            self.mask[parent_depth + 1, self.nbases - 2:self.nbases] = True

            self.truncated[:, self.nbases - 2: self.nbases] = np.tile(self.truncated[:, parent:parent + 1], (1, 2))
            self.truncated[parent_depth + 1, self.nbases - 1] = True

            self.cov[:, self.nbases - 2: self.nbases] = np.tile(self.cov[:, parent:parent + 1], (1, 2))
            self.cov[parent_depth + 1, self.nbases - 2:self.nbases] = cov

            self.root[:, self.nbases - 2: self.nbases] = np.tile(self.root[:, parent:parent + 1], (1, 2))
            self.root[parent_depth + 1, self.nbases - 1] = root
        elif self.backend is Backend.FORTRAN:
            fortran.backend.add_bases(parent, cov, root, self.nbases, self.mask, self.truncated, self.cov,
                                      self.root)
        else:
            raise NotImplementedError("Backend not implemented.")

    @jaxtyped(typechecker=beartype)
    def _expand_bases(self, x: Float[np.ndarray, "N d"], y: Float[np.ndarray, "N"]) -> float:
        """
        Grow the model to the maximum number of basis functions by iteratively adding the basis that reduces
        the lack of fit criterion the most. Equivalent to the forward pass in the Mars paper, including the adaptions
        in the Fast Mars paper.

        Args:
            x: Predictor Variables.
            y: Response Variables.

        Returns:
            Lack of fit criterion
        """
        if self.backend is Backend.PYTHON:
            self.nbases = 1
            self.mask = np.zeros((self.max_nbases, self.max_nbases), dtype=bool)
            self.truncated = np.zeros((self.max_nbases, self.max_nbases), dtype=bool)
            self.cov = np.zeros((self.max_nbases, self.max_nbases), dtype=int)
            self.root = np.zeros((self.max_nbases, self.max_nbases), dtype=float)

            candidate_queue = [0.]  # One for the constant function
            for iteration in range(self.max_nbases // 2):
                best_lof = np.inf
                best_cov = None
                best_root = None
                best_parent = None

                parents = np.argsort(candidate_queue)
                pairs = []
                for i in range(min(self.max_ncandidates, self.nbases)):
                    parent = parents[i]
                    eligible_cov = set(range(x.shape[1])) - set(self.cov[self.mask[:, parent], parent])
                    for cov_idx in eligible_cov:
                        pairs.append((i, cov_idx))
                basis_lofs = [np.inf] * min(self.max_ncandidates, self.nbases)

                self.nbases += 2
                for i, cov_idx in pairs:
                    parent = parents[i]
                    self._add_bases(parent, cov_idx, 0.0)

                    if parent == 0:  # constant function
                        eligible_roots = x[:, cov_idx].copy()  # copy is important!
                    else:
                        eligible_roots = x[np.where(data_matrix[:, parent - 1] > 0)[0], cov_idx]
                    eligible_roots[::-1].sort()

                    for root_idx in range(len(eligible_roots)):
                        self.root[self.mask[:, parent].sum() + 1, self.nbases - 1] = eligible_roots[root_idx]

                        if root_idx == 0:
                            data_matrix, data_matrix_mean, covariance_matrix, rhs, chol, self.coefficients, lof = \
                                self._fit(x, y)
                        else:
                            data_matrix, data_matrix_mean, covariance_matrix, rhs, chol, self.coefficients, lof = \
                                self._update_fit(data_matrix, data_matrix_mean, covariance_matrix, rhs, chol, x, y,
                                                 eligible_roots[root_idx - 1], parent)

                        if lof < basis_lofs[i]:
                            basis_lofs[i] = lof
                        if lof < best_lof:
                            best_lof = lof
                            best_cov = cov_idx
                            best_root = eligible_roots[root_idx]
                            best_parent = parent
                for i in range(len(candidate_queue)):
                    if i < len(basis_lofs):
                        candidate_queue[parents[i]] = basis_lofs[i] - best_lof
                    else:
                        candidate_queue[parents[i]] -= self.aging_factor

                if best_cov is not None:
                    self._add_bases(best_parent, best_cov, best_root)
                    candidate_queue.extend([0, 0])
                else:
                    print(f"Cannot find additional bases in iteration {iteration}.")
                    self.mask[:, self.nbases - 2: self.nbases] = False
                    self.nbases -= 2
                    break

            data_matrix, data_matrix_mean, covariance_matrix, rhs, chol, self.coefficients, lof = self._fit(x, y)

        elif self.backend is Backend.FORTRAN:
            (lof, self.nbases, self.mask, self.truncated, self.cov, self.root,
             self.coefficients) = fortran.backend.expand_bases(x, y, self.y_mean,
                                                               self.max_nbases, self.max_ncandidates,
                                                               self.aging_factor,
                                                               self.penalty)
            # Fortran indexes from 1 and has a fixed output size, therefore requires trimming in case of early stopping
            self.cov -= 1
            self.coefficients = self.coefficients[:self.nbases - 1]
        else:
            raise NotImplementedError("Backend not implemented.")

        return lof

    @jaxtyped(typechecker=beartype)
    def _prune_bases(self,
                     x: Float[np.ndarray, "N d"],
                     y: Float[np.ndarray, "N"],
                     lof: float) -> float:
        """
        Prune the bases to the best fitting subset of the basis functions by iteratively removing the basis
        that increases the lack of fit criterion the least. Equivalent to the backward pass in the Mars paper.

        Args:
            x: Predictor Variables.
            y: Response Variables.
            lof: Lack of fit criterion.

        Returns:
            Lack of fit criterion.
        """
        if self.backend is Backend.PYTHON:
            best_nbases = self.nbases
            best_mask = self.mask.copy()
            best_lof = lof

            prev_mask = self.mask.copy()

            for iteration in range(self.nbases - 1):
                best_lof_trim = np.inf
                removal_idx = None

                for basis_idx in self._active_base_indices():
                    self.mask[:, basis_idx] = False
                    self.nbases -= 1

                    if self.nbases != 1:
                        data_matrix, data_matrix_mean, covariance_matrix, rhs, chol, self.coefficients, lof = self._fit(
                            x, y)
                    else:
                        lof = self._generalised_cross_validation(y, np.zeros((y.shape[0], 0)), np.empty((0, 0)))

                    if lof < best_lof_trim:
                        best_lof_trim = lof
                        removal_idx = basis_idx
                    if lof < best_lof:
                        best_lof = lof
                        best_nbases = self.nbases
                        best_mask = self.mask.copy()

                    self.mask[:, basis_idx] = prev_mask[:, basis_idx]
                    self.nbases += 1

                self.mask[:, removal_idx] = False
                self.nbases -= 1
                prev_mask[:, removal_idx] = False

            self.mask = best_mask
            self.nbases = best_nbases
            data_matrix, data_matrix_mean, covariance_matrix, rhs, chol, self.coefficients, lof = \
                self._fit(x, y)
        elif self.backend is Backend.FORTRAN:
            # Fortran indexes from 1
            self.mask = np.asfortranarray(self.mask)
            # Need mutable types for Fortran
            lof = np.array(lof, dtype=float)
            self.nbases = np.array(self.nbases, dtype=int)
            self.coefficients, self.mask = fortran.backend.prune_bases(x, y, self.y_mean,
                                                                       lof,
                                                                       self.nbases,
                                                                       self.mask,
                                                                       self.truncated,
                                                                       self.cov + 1,
                                                                       self.root,
                                                                       self.penalty)
            # Fortran has a fixed output size, therefore requires trimming in case of early stopping
            self.coefficients = self.coefficients[:self.nbases - 1]
            lof = float(lof)
        else:
            raise NotImplementedError("Backend not implemented.")

        return lof

    @jaxtyped(typechecker=beartype)
    def find_bases(self, x: Float[np.ndarray, "N d"], y: Float[np.ndarray, "N"]) -> float:
        """
        Find the best fitting basis functions for the given data.

        Args:
            x: Predictor Variables.
            y: Response Variables.

        Returns:
            Lack of fit criterion.
        """
        self.y_mean = y.mean()
        if self.backend is Backend.PYTHON:
            lof = self._expand_bases(x, y)
            lof = self._prune_bases(x, y, lof)
        elif self.backend is Backend.FORTRAN:
            (lof, self.nbases, self.mask, self.truncated, self.cov, self.root,
             self.coefficients) = fortran.backend.find_bases(x, y, self.y_mean, self.max_nbases,
                                                             self.max_ncandidates,
                                                             self.aging_factor, self.penalty)
            # Fortran indexes from 1 and has a fixed output size, therefore requires trimming in case of early stopping
            self.cov -= 1
            self.coefficients = self.coefficients[:self.nbases - 1]
        else:
            raise NotImplementedError("Backend not implemented.")

        return lof


@njit(cache=True, fastmath=True, error_model="numpy")
def decompose_addition(covariance_addition: Float[np.ndarray, "{self.nbases}-1"]) \
        -> tuple[Float[np.ndarray, "2"], Float[np.ndarray, "2 {self.nbases}-1"]]:
    """
    Decompose the addition to the covariance matrix,
    which was done by adding to the last row and column of the matrix,
    into eigenvalues and eigenvectors to perform 2 rank-1 updates.

    Args:
        covariance_addition: Addition to the covariance matrix. (the same vector is applied to the row and column)

    Returns:
        Eigenvalues and eigenvectors of the addition.

    Notes: Cant type annotate this, since numba & jaxtyping don't vibe
    """
    eigenvalue_intermediate = np.sqrt(
        covariance_addition[-1] ** 2 + 4 * np.sum(covariance_addition[:-1] ** 2))
    eigenvalues = np.array([
        (covariance_addition[-1] + eigenvalue_intermediate) / 2,
        (covariance_addition[-1] - eigenvalue_intermediate) / 2,
    ])

    eigenvectors = np.zeros((2, covariance_addition.shape[0]))
    eigenvectors[0, :] = np.array([*(covariance_addition[:-1] / eigenvalues[0]), 1])
    eigenvectors[1, :] = np.array([*(covariance_addition[:-1] / eigenvalues[1]), 1])
    eigenvectors[0] /= np.linalg.norm(eigenvectors[0])
    eigenvectors[1] /= np.linalg.norm(eigenvectors[1])

    return eigenvalues, eigenvectors


@njit(cache=False, error_model="numpy", fastmath=True, parallel=False)
def update_cholesky(chol: Float[np.ndarray, "{self.nbases} {self.nbases}"],
                    update_vector: Float[np.ndarray, "{self.nbases}"],
                    multiplier: float) -> Float[np.ndarray, "{self.nbases} {self.nbases}"]:
    """
    Compute the lower triangular Cholesky factor of thee rank-1 update
    chol @ chol.T + multiplier * update_vector @ update_vector.T.

    Args:
        chol: Cholesky decomposition of the original matrix, such that mat = chol @ chol.T
        update_vector: Vector defining rank-one update.
        multiplier: Scalar multiplier to rank-one update.

    Returns:
        Updated Cholesky decomposition.

    Notes:
        Algorithm according to [1] Oswin Krause. Christian Igel.
        A More Efficient Rank-one Covariance Matrix Update for Evolution Strategies.
        2015 ACM Conference. https://christian-igel.github.io/paper/AMERCMAUfES.pdf.
        Adapted for computation speed and parallelization.
        Cant typecheck this, since numba & jaxtyping don't vibe

    """
    diag = np.diag(chol).copy()
    chol = chol / diag
    diag **= 2

    u = np.zeros((chol.shape[0], update_vector.shape[0]))
    u[0, :] = update_vector
    u[0, 1:] -= update_vector[0] * chol[1:, 0]
    b = np.ones(chol.shape[0])
    for i in range(1, chol.shape[0]):
        u[i, :] = u[i - 1, :]
        u[i, i + 1:] -= u[i - 1, i] * chol[i + 1:, i]
        b[i] = b[i - 1] + multiplier * u[i - 1, i - 1] ** 2 / diag[i - 1]

    # Check for singular matrix
    updated_diag = diag + multiplier / b * np.diag(u) ** 2
    if np.any(updated_diag < 0):
        print("Singular Covariance Matrix! Increasing Regularisation drastically.")
        diag = diag + (1e-6 - updated_diag)

    for i in range(chol.shape[0]):
        chol[i, i] = np.sqrt(diag[i] + multiplier / b[i] * u[i, i] ** 2)
        chol[i + 1:, i] *= chol[i, i]
        chol[i + 1:, i] += multiplier / b[i] * u[i, i] * u[i, i + 1:] / chol[i, i]

    return chol
