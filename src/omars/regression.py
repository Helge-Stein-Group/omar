from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np
from scipy.linalg import qr, solve_triangular


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
    gcv: float = None

    def data_matrix(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2

        return np.column_stack([basis(x) for basis in self.basis])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2

        return self.data_matrix(x) @ self.coefficients

    def __len__(self):
        return len(self.basis)

    def __getitem__(self, i: int):
        assert isinstance(i, int)
        assert i < len(self.basis)

        return Model([self.basis[i]], self.coefficients[i].reshape(1, -1))

    def add(self, bases: list[Basis]):
        assert all(isinstance(b, Basis) for b in bases)

        self.basis.extend(bases)
        return self

    def remove(self, i: int):
        assert isinstance(i, int)
        assert i < len(self.basis)

        del self.basis[i]
        return self

    def fit(self, x: np.ndarray, y: np.ndarray, d: float = 3):
        assert x.ndim == 2
        assert y.ndim == 1
        assert x.shape[0] == y.shape[0]
        assert isinstance(d, (int, float))

        '''
        ATTENTION !!!
        The basis through the forward pass does not need to be linearly independent, 
        which may lead to a singular matrix, this is can be solved by pivoting in the 
        Cholesky decomposition, in the paper a small value is added to the diagonal.
        '''
        self.fit_matrix = self.data_matrix(x)

        covariance_matrix = self.fit_matrix.T @ (
                self.fit_matrix - np.mean(self.fit_matrix, axis=0))
        covariance_matrix += np.eye(covariance_matrix.shape[0]) * 1e-6
        rhs = self.fit_matrix.T @ (y - np.mean(y))
        q,r = qr(covariance_matrix)
        self.coefficients = solve_triangular(r, q.T @ rhs)

        y_pred = self.fit_matrix @ self.coefficients
        mse = np.sum((y - y_pred) ** 2)

        c_m = len(self.basis) + d * (len(self.basis) - 1)
        self.gcv = mse / len(y) / (1 - c_m / len(y)) ** 2


def forward_pass(x: np.ndarray, y: np.ndarray, m_max: int) -> Model:
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    assert isinstance(m_max, int)

    covariates = set(range(x.shape[1]))
    model = Model()
    while len(model) < m_max:
        best_gcv = np.inf
        best_candidate_model = None
        for m, selected_basis in enumerate(model.basis):
            ineligible_covariates = set(selected_basis.v)
            eligible_covariates = covariates - ineligible_covariates
            for v in eligible_covariates:
                eligible_knots = x[:, v][np.where(selected_basis(x) > 0)]
                eligible_knots[::-1].sort()
                for t in eligible_knots:
                    candidate_model = deepcopy(model)
                    unhinged_candidate = deepcopy(selected_basis)
                    unhinged_candidate.add(v, 0.0, False)
                    hinged_candidate = deepcopy(selected_basis)
                    hinged_candidate.add(v, t, True)
                    candidate_model.add([unhinged_candidate, hinged_candidate])
                    candidate_model.fit(x, y)
                    if candidate_model.gcv < best_gcv:
                        best_gcv = candidate_model.gcv
                        best_candidate_model = candidate_model

        model = best_candidate_model

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
