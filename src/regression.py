from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import cho_factor, cho_solve


def update_cholesky(tri: np.ndarray, update_vectors: list[np.ndarray],
                    multipliers: list[float]):
    assert tri.shape[0] == tri.shape[1]
    assert tri.shape[0] == len(update_vectors[0])
    assert len(update_vectors) == len(multipliers)

    for update_vec, multiplier in zip(update_vectors, multipliers):
        b = 1
        diag = tri.diagonal().copy()
        tri = tri / diag
        diag **= 2
        for i in range(tri.shape[0]):
            update_vec[i + 1:] -= update_vec[i] * tri[i + 1:, i]
            tri[i, i] = np.sqrt(diag[i] + multiplier / b * update_vec[i] ** 2)
            tri[i + 1:, i] *= tri[i, i]
            tri[i + 1:, i] += multiplier / b * update_vec[i] * update_vec[i + 1:] / tri[i, i]
            b += multiplier * update_vec[i] ** 2 / diag[i]
    return tri


@dataclass
class Basis:
    v: list[int] = field(default_factory=list)  # Index of variable
    t: np.ndarray = field(
        default_factory=lambda: np.array([], dtype=float))  # Knots
    hinge: list[bool] = field(default_factory=list)  # Whether to apply the hinge

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2

        if len(self.v) == 0:
            return np.ones(x.shape[0])
        result = x[:, self.v] - self.t
        np.maximum(np.zeros((x.shape[0], len(self.v)), dtype=float), result,
                   where=self.hinge, out=result)
        return result.prod(axis=1)

    def __eq__(self, other):
        return np.all(self.v == other.v) and np.all(self.t == other.t) and np.all(
            self.hinge == other.hinge)

    def add(self, v: int, t: float, hinge: bool):
        assert isinstance(v, int)
        assert isinstance(t, float)
        assert isinstance(hinge, bool)

        self.v.append(v)
        self.t = np.append(self.t, t)
        self.hinge.append(hinge)


@dataclass
class Model:
    basis: list[Basis] = field(default_factory=lambda: [Basis()])
    coefficients: np.ndarray = None
    fit_matrix: np.ndarray = None
    covariance_matrix: np.ndarray = None
    right_hand_side: np.ndarray = None
    gcv: float = None

    # Update parameters
    indices: np.ndarray = None
    update: np.ndarray = None
    fixed_mean: np.ndarray = None
    candidate_mean: float = None
    update_mean: float = None

    def data_matrix(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2

        return np.column_stack([basis(x) for basis in self.basis])

    def __str__(self):
        desc = "Basis functions: \n"
        for basis in self.basis:
            desc += f"{basis}\n"
        return desc

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2

        return self.data_matrix(x) @ self.coefficients

    def __len__(self):
        return len(self.basis)

    def __getitem__(self, i: int):
        assert isinstance(i, int)
        assert i < len(self.basis)

        return Model([self.basis[i]], self.coefficients[i].reshape(1, -1))

    def add(self, bases: list[Basis]) -> bool:
        assert all(isinstance(b, Basis) for b in bases)
        duplicates = False
        for basis in bases:
            if basis in self.basis:
                duplicates = True
            else:
                self.basis.append(basis)

        return duplicates

    def remove(self, i: int):
        assert isinstance(i, int)
        assert i < len(self.basis)

        del self.basis[i]
        return self

    def update_initialisation(self, x: np.ndarray, u: float, t: float, v: int,
                              selected_fit: np.ndarray):
        assert x.ndim == 2
        assert x.shape[0] == self.fit_matrix.shape[0]
        assert t <= u
        assert v < x.shape[1]
        assert selected_fit.shape[0] == x.shape[0]

        self.indices = x[:, v] > t
        self.update = np.where(x[self.indices, v] >= u, u - t, x[self.indices, v] - t)
        self.update *= selected_fit[self.indices]

        self.fixed_mean = np.mean(self.fit_matrix[:, :-1], axis=0)
        self.update_mean = np.sum(self.update) / len(x)
        self.candidate_mean = np.mean(self.fit_matrix[:, -1]) + self.update_mean

    def calculate_fit_matrix(self, x: np.ndarray):
        assert x.ndim == 2

        self.fit_matrix = self.data_matrix(x)

    def update_fit_matrix(self):
        self.fit_matrix[self.indices, -1] += self.update

    def calculate_covariance_matrix(self):
        self.covariance_matrix = self.fit_matrix.T @ self.fit_matrix
        collapsed_fit = np.sum(self.fit_matrix, axis=0)
        self.covariance_matrix -= np.outer(collapsed_fit, collapsed_fit) / \
                                  self.fit_matrix.shape[0]
        self.covariance_matrix += np.eye(self.covariance_matrix.shape[0]) * 1e-8

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

        self.right_hand_side = self.fit_matrix.T @ (y - np.mean(y))

    def update_right_hand_side(self, y: np.ndarray) -> None:
        assert y.ndim == 1
        assert y.shape[0] == self.fit_matrix.shape[0]

        self.right_hand_side[-1] += np.sum(self.update * (y[self.indices] - np.mean(y)))

    def calculate_gcv(self, y: np.ndarray, d: float = 3) -> None:
        assert y.ndim == 1
        assert isinstance(d, (int, float))

        y_pred = self.fit_matrix @ self.coefficients
        mse = np.sum((y - y_pred) ** 2)  # rhs instead of y?

        c_m = len(self.basis) + 1 + d * (len(self.basis) - 1)
        self.gcv = mse / len(y) / (1 - c_m / len(y)) ** 2

    def fit(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]

        '''
        ATTENTION !!!
        The basis through the forward pass does not need to be linearly independent, 
        which may lead to a singular matrix. Here we can either add a small diagonal 
        value or set the dependent coefficients to 0 to obtain a unique solution again.
        '''
        self.calculate_fit_matrix(x)
        self.calculate_covariance_matrix()
        self.calculate_right_hand_side(y)

        tri, lower = cho_factor(self.covariance_matrix, lower=True)

        self.coefficients = cho_solve((tri, lower), self.right_hand_side)

        self.calculate_gcv(y)

        return tri

    def update_fit(self, x: np.ndarray, y: np.ndarray, tri: np.ndarray, u: float,
                   t: float, v: int, selected_fit: np.ndarray) -> np.ndarray:
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert tri.shape[0] == tri.shape[1] == len(self.basis)
        assert t <= u

        self.update_initialisation(x, u, t, v, selected_fit)
        self.update_fit_matrix()
        covariance_addition = self.update_covariance_matrix()

        if covariance_addition.any():
            eigenvalues, eigenvectors = self.decompose_addition(covariance_addition)
            tri = update_cholesky(tri, eigenvectors, eigenvalues)

        self.update_right_hand_side(y)

        self.coefficients = cho_solve((tri, True), self.right_hand_side)

        self.calculate_gcv(y)

        return tri


def forward_pass(x: np.ndarray, y: np.ndarray, m_max: int, k: int = 5,
                 aging_factor: float = 0.) -> Model:
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    assert isinstance(m_max, int)
    assert isinstance(k, int)
    assert isinstance(aging_factor, float)

    covariates = set(range(x.shape[1]))
    model = Model()
    model.fit(x, y)
    candidate_queue = {basis_idx: 1. for basis_idx in range(len(model.basis))}
    while len(model) < m_max:
        best_gcv = np.inf
        best_candidate_model = None
        for m in sorted(candidate_queue, key=candidate_queue.get)[:-k - 1:-1]:
            selected_basis = model.basis[m]
            ineligible_covariates = set(selected_basis.v)
            eligible_covariates = covariates - ineligible_covariates
            basis_gcv = np.inf
            for v in eligible_covariates:
                eligible_knots = x[np.where(selected_basis(x) > 0)[0], v]
                eligible_knots[::-1].sort()
                candidate_model = deepcopy(model)
                unhinged_candidate = deepcopy(selected_basis)
                unhinged_candidate.add(v, 0.0, False)
                hinged_candidate = deepcopy(selected_basis)
                hinged_candidate.add(v, eligible_knots[0], True)
                duplicates = candidate_model.add([unhinged_candidate, hinged_candidate])
                tri = candidate_model.fit(x,
                                          y)  # TODO fit matrix only need extension not recalculation
                u = eligible_knots[0]

                for i, t in enumerate(eligible_knots[1:]):
                    hinged_candidate = deepcopy(selected_basis)
                    hinged_candidate.add(v, t, True)
                    if hinged_candidate in candidate_model.basis:
                        continue
                    candidate_model.basis[-1] = hinged_candidate
                    tri = candidate_model.update_fit(x, y, tri, u, t, v,
                                                     model.fit_matrix[:, m])
                    u = t
                    if candidate_model.gcv < basis_gcv:
                        basis_gcv = candidate_model.gcv
                    if candidate_model.gcv < best_gcv:
                        best_gcv = candidate_model.gcv
                        best_candidate_model = deepcopy(candidate_model)
            candidate_queue[m] = model.gcv - basis_gcv
        for unselected_basis in sorted(candidate_queue, key=candidate_queue.get)[k:]:
            candidate_queue[unselected_basis] += aging_factor
        model = best_candidate_model
        for i in range(len(model) - len(candidate_queue)):
            candidate_queue[len(model) - 1 - i] = np.inf

    return model


def backward_pass(x: np.ndarray, y: np.ndarray, model: Model) -> Model:
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    assert isinstance(model, Model)

    best_model = model
    best_trimmed_model = deepcopy(model)
    best_gcv = model.gcv

    while len(best_trimmed_model) > 1:
        best_trimmed_gcv = np.inf
        previous_model = deepcopy(best_trimmed_model)
        for i in range(len(previous_model)):
            if i == 0:  # First basis function (constant 1) cannot be excluded
                continue
            trimmed_model = deepcopy(previous_model).remove(i)
            trimmed_model.fit(x, y)
            if trimmed_model.gcv < best_trimmed_gcv:
                best_trimmed_gcv = trimmed_model.gcv
                best_trimmed_model = trimmed_model
            if trimmed_model.gcv < best_gcv:
                best_gcv = trimmed_model.gcv
                best_model = trimmed_model

    return best_model


def fit(x: np.ndarray, y: np.ndarray, m_max: int) -> Model:
    model = forward_pass(x, y, m_max)
    model = backward_pass(x, y, model)

    return model
