from typing import Self

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def update_cholesky(chol: np.ndarray,
                    update_vectors: list[np.ndarray],
                    multipliers: list[float]) -> np.ndarray:
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
    def __init__(self,
                 max_nbases: int = 10,
                 max_ncandidates: int = 5,
                 aging_factor: float = 0.,
                 smoothness: float = 3):
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
        self.where[0, 0] = True  # Initial basis function

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
        assert x.ndim == 2
        assert isinstance(basis_slice, slice)

        result = -self.nodes[:, basis_slice] + x[:, self.covariates[:, basis_slice]]
        np.maximum(np.zeros_like(result), result, where=self.hinges[:, basis_slice], out=result)

        return result.prod(axis=1, where=self.where[:, basis_slice])[:, :self.nbases]

    def __str__(self) -> str:
        desc = "Basis functions: \n"
        for basis_idx in range(self.nbases):
            for func_idx in range(self.max_nbases):
                if self.where[func_idx, basis_idx]:
                    cov = self.covariates[func_idx, basis_idx]
                    node = self.nodes[func_idx, basis_idx]
                    hinge = self.hinges[func_idx, basis_idx]
                    desc += f"(x[{cov} - {node}]){u'\u208A' if hinge else ''}"
            desc += "\n"
        return desc

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2

        return self.data_matrix(x, slice(self.nbases)) @ self.coefficients

    def __len__(self) -> int:
        return self.nbases

    def __getitem__(self, i: int) -> Self:
        assert isinstance(i, int)
        assert i < self.nbases

        sub_model = OMARS()
        sub_model.nbases = 1
        sub_model.covariates = self.covariates[:, i]
        sub_model.nodes = self.nodes[:, i]
        sub_model.hinges = self.hinges[:, i]
        sub_model.where = self.where[:, i]
        sub_model.coefficients = self.coefficients[i]
        return sub_model

    def add_basis(self,
                  covariates: np.ndarray,
                  nodes: np.ndarray,
                  hinges: np.ndarray,
                  where: np.ndarray) -> None:
        assert covariates.ndim == nodes.ndim == hinges.ndim == where.ndim == 2
        assert (covariates.shape[0] == nodes.shape[0]
                == hinges.shape[0] == where.shape[0] == self.max_nbases)
        assert covariates.shape[1] == nodes.shape[1] == hinges.shape[1] == where.shape[1]
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
        assert isinstance(removal_slice, slice)

        self.where[:, removal_slice] = False
        self.nbases -= removal_slice.stop - removal_slice.start

    def calculate_means(self, collapsed_fit: np.ndarray):
        assert collapsed_fit.shape[0] == self.fit_matrix.shape[1]

        self.fixed_mean = collapsed_fit[:-1] / self.fit_matrix.shape[0]
        self.candidate_mean = collapsed_fit[-1] / self.fit_matrix.shape[0]

    def extend_means(self, collapsed_fit: np.ndarray, nadditions: int):
        assert nadditions + self.nbases < self.max_nbases
        assert collapsed_fit.shape[0] == nadditions

        self.fixed_mean = np.append(self.fixed_mean, self.candidate_mean)
        self.fixed_mean = np.append(self.fixed_mean,
                                    collapsed_fit[:-1] / self.fit_matrix.shape[0])
        self.candidate_mean = collapsed_fit[-1] / self.fit_matrix.shape[0]

    def update_init(self, x: np.ndarray, old_node: float):
        assert x.ndim == 2
        assert x.shape[0] == self.fit_matrix.shape[0]
        assert isinstance(old_node, float)

        prod_idx = np.sum(self.where[:, self.nbases - 1])
        new_node = self.nodes[prod_idx, self.nbases - 1]
        covariate = self.covariates[prod_idx, self.nbases - 1]

        self.indices = x[:, covariate] > new_node
        self.update = np.where(x[self.indices, covariate] >= old_node,
                               old_node - new_node,
                               x[self.indices, covariate] - new_node)
        self.update *= self.fit_matrix[self.indices, -1]

        self.update_mean = np.sum(self.update) / len(x)
        self.candidate_mean += self.update_mean

    def calculate_fit_matrix(self, x: np.ndarray):
        assert x.ndim == 2

        self.fit_matrix = self.data_matrix(x, slice(self.nbases))

    def extend_fit_matrix(self, x: np.ndarray, nadditions: int):
        assert x.ndim == 2
        assert isinstance(nadditions, int)

        self.fit_matrix = np.column_stack((self.fit_matrix, self.data_matrix(x,
                                                                             slice(self.nbases - nadditions,

                                                                                   self.nbases))))

    def update_fit_matrix(self):
        self.fit_matrix[self.indices, -1] += self.update

    def calculate_covariance_matrix(self):
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

    def extend_covariance_matrix(self, nadditions: int):
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
        covariance_addition = np.zeros_like(self.covariance_matrix[-1, :])
        covariance_addition[:-1] += np.tensordot(self.update,
                                                 self.fit_matrix[self.indices,
                                                 :-1] - self.fixed_mean,
                                                 axes=[[0], [0]])
        covariance_addition[-1] += np.tensordot(
            self.fit_matrix[self.indices, -1] - self.update,
            self.update - self.update_mean,
            axes=[[0], [0]]
        )
        covariance_addition[-1] += np.tensordot(
            self.update,
            self.fit_matrix[self.indices, -1] - self.candidate_mean,
            axes=[[0], [0]]
        )

        self.covariance_matrix[-1, :-1] += covariance_addition[:-1]
        self.covariance_matrix[:, -1] += covariance_addition

        return covariance_addition

    def decompose_addition(self, covariance_addition: np.ndarray) -> tuple[list[float],
    list[np.ndarray]]:
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

    def calculate_right_hand_side(self, y: np.ndarray):
        assert y.ndim == 1
        assert y.shape[0] == self.fit_matrix.shape[0]
        assert self.fit_matrix.shape[1] == self.covariance_matrix.shape[0]

        self.y_mean = np.mean(y)
        self.right_hand_side = self.fit_matrix.T @ (y - self.y_mean)

    def extend_right_hand_side(self, y: np.ndarray, nadditions: int):
        assert y.ndim == 1
        assert y.shape[0] == self.fit_matrix.shape[0]
        assert self.fit_matrix.shape[1] == self.covariance_matrix.shape[0]
        assert isinstance(nadditions, int)

        self.right_hand_side = np.append(self.right_hand_side,
                                         self.fit_matrix[:, -nadditions:].T @ (y - self.y_mean))

    def update_right_hand_side(self, y: np.ndarray) -> None:
        assert y.ndim == 1
        assert y.shape[0] == self.fit_matrix.shape[0]

        self.right_hand_side[-1] += np.sum(
            self.update * (y[self.indices] - self.y_mean))

    def generalised_cross_validation(self, y: np.ndarray) -> None:
        assert y.ndim == 1

        y_pred = self.fit_matrix @ self.coefficients
        mse = np.sum((y - y_pred) ** 2)  # rhs instead of y?

        c_m = self.nbases + 1 + self.smoothness * (self.nbases - 1)
        self.lof = mse / len(y) / (1 - c_m / len(y)) ** 2

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
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
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert isinstance(nadditions, int)

        # Expects: Fit Mat/Cov Mat/RHS of smaller model
        self.extend_fit_matrix(x, nadditions)
        self.extend_covariance_matrix(nadditions)
        self.extend_right_hand_side(y, nadditions)

        chol, lower = cho_factor(self.covariance_matrix, lower=True)

        self.coefficients = cho_solve((chol, lower), self.right_hand_side)

        self.generalised_cross_validation(y)

        return np.tril(chol)

    def update_fit(self, x: np.ndarray, y: np.ndarray, chol: np.ndarray, old_node: float) -> np.ndarray:
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert chol.shape[0] == chol.shape[1] == self.nbases
        assert self.nodes[np.sum(self.where[:, self.nbases - 1]), self.nbases - 1] <= old_node

        # Expects: Fit Mat/Cov Mat/RHS of same size model, with same v and u > t
        self.update_init(x, old_node)
        self.update_fit_matrix()
        covariance_addition = self.update_covariance_matrix()

        if covariance_addition.any():
            eigenvalues, eigenvectors = self.decompose_addition(covariance_addition)
            chol = update_cholesky(chol, eigenvectors, eigenvalues)

        self.update_right_hand_side(y)

        self.coefficients = cho_solve((chol, True), self.right_hand_side)

        self.generalised_cross_validation(y)

        return chol

    def shrink_fit(self, x: np.ndarray, y: np.ndarray, removal_slice: slice) -> np.ndarray:
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert isinstance(removal_slice, slice)

        self.fit_matrix = np.delete(self.fit_matrix, removal_slice, axis=1)
        self.covariance_matrix = np.delete(self.covariance_matrix, removal_slice, axis=0)
        self.covariance_matrix = np.delete(self.covariance_matrix, removal_slice, axis=1)
        self.right_hand_side = np.delete(self.right_hand_side, removal_slice)

        tri, lower = cho_factor(self.covariance_matrix, lower=True)

        self.coefficients = cho_solve((tri, True), self.right_hand_side)

        self.generalised_cross_validation(y)

        return tri

    def forward_pass(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]

        covariates = set(range(x.shape[1]))
        self.fit(x, y)
        candidate_queue = {0: 1.}
        while self.nbases < self.max_nbases:
            best_lof = np.inf
            best_covariate = None
            best_node = None
            best_hinge = None
            best_where = None
            for parent_idx in sorted(candidate_queue, key=candidate_queue.get)[:-self.max_ncandidates - 1:-1]:
                eligible_covariates = covariates - set(self.covariates[:, parent_idx])
                basis_lof = np.inf
                parent_depth = np.sum(self.where[:, parent_idx])
                for cov in eligible_covariates:
                    eligible_knots = x[np.where(self.fit_matrix[:, parent_idx] > 0)[0], cov]
                    eligible_knots[::-1].sort()
                    additional_covariates = np.tile(self.covariates[:, parent_idx], (2, 1)).T
                    additional_nodes = np.tile(self.nodes[:, parent_idx], (2, 1)).T
                    additional_hinges = np.tile(self.hinges[:, parent_idx], (2, 1)).T
                    additional_where = np.tile(self.where[:, parent_idx], (2, 1)).T

                    additional_covariates[parent_depth, 0] = cov
                    additional_nodes[parent_depth, 0] = 0.0
                    additional_hinges[parent_depth, 0] = False
                    additional_where[parent_depth, 0] = True

                    additional_covariates[parent_depth, 1] = cov
                    additional_nodes[parent_depth, 1] = eligible_knots[0]
                    additional_hinges[parent_depth, 1] = True
                    additional_where[parent_depth, 1] = True

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
                        updated_nodes[parent_depth] = new_node
                        self.update_basis(
                            additional_covariates[:, 1],
                            updated_nodes,
                            additional_hinges[:, 1],
                            additional_where[:, 1]
                        )
                        chol = self.update_fit(x, y, chol, old_node)
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
            for unselected_idx in sorted(candidate_queue, key=candidate_queue.get)[self.max_ncandidates:]:
                candidate_queue[unselected_idx] += self.aging_factor
            self.add_basis(best_covariate, best_node, best_hinge, best_where)
            self.extend_fit(x, y, 2)
            for i in range(2):
                candidate_queue[len(candidate_queue)] = 0

    def backward_pass(self, x, y):
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]

        best_covariates = self.covariates.copy()
        best_nodes = self.nodes.copy()
        best_hinges = self.hinges.copy()
        best_where = self.where.copy()
        best_nbases = self.nbases

        best_trimmed_where = None

        best_lof = self.lof

        while len(self) > 1:
            best_trimmed_lof = np.inf

            previous_where = self.where.copy()

            for basis_idx in range(self.nbases):
                if basis_idx == 0:
                    continue
                self.remove_basis(slice(basis_idx, basis_idx + 1))
                self.shrink_fit(x, y, slice(basis_idx, basis_idx + 1))
                if self.lof < best_trimmed_lof:
                    best_trimmed_lof = self.lof
                    best_trimmed_where = self.where.copy()
                if self.lof < best_lof:
                    best_lof = self.lof
                    best_covariates = self.covariates.copy()
                    best_nodes = self.nodes.copy()
                    best_hinges = self.hinges.copy()
                    best_where = self.where.copy()
                    best_nbases = self.nbases
                self.where = previous_where.copy()
                self.nbases += 1
            self.where = best_trimmed_where.copy()
            self.nbases -= 1

        self.covariates = best_covariates
        self.nodes = best_nodes
        self.hinges = best_hinges
        self.where = best_where
        self.nbases = best_nbases
        self.fit(x, y)

    def find_bases(self, x: np.ndarray, y: np.ndarray):
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]

        self.forward_pass(x, y)
        self.backward_pass(x, y)
