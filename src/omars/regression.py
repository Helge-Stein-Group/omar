from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np


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
        return np.prod(
            np.maximum(
                np.zeros((x.shape[0], len(self.v)), dtype=float),
                x[:, self.v] - self.t,
                where=self.hinge),
            axis=1)

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
    singular_values: np.ndarray = None

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

        return Model([self.basis[i]], self.coefficients[i].reshape(1, -1),
                     self.singular_values[i].reshape(1, -1))

    def add(self, bases: list[Basis]):
        assert all(isinstance(b, Basis) for b in bases)

        self.basis.extend(bases)
        return self

    def remove(self, i: int):
        assert isinstance(i, int)
        assert i < len(self.basis)

        del self.basis[i]
        return self


def evaluate_model(x: np.ndarray, y: np.ndarray, model: Model,
                   d: float = 3) -> float:
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    assert isinstance(model, Model)
    assert isinstance(d, (int, float))

    '''
    ATTENTION !!!
    The basis through the forward pass does not need to be linearly independent, 
    which may lead to a singular matrix, this is can be solved by pivoting in the 
    Cholesky decomposition, in the paper a small value is added to the diagonal.
    Here numpy can deal with singular matrices, but the lack of fit needs to 
    be recalculated.
    '''
    data_matrix = model.data_matrix(x)
    coefficients, mse, rank, sv = np.linalg.lstsq(data_matrix, y, rcond=None)
    model.coefficients = coefficients
    model.singular_values = sv
    if mse.size == 0:
        y_pred = data_matrix @ coefficients
        mse = np.sum((y - y_pred) ** 2)

    c_m = len(model.singular_values) + d * (len(model)-1)
    return mse / len(y) / (1 - c_m / len(y)) ** 2


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
                eligible_knots = set(x[:, v][np.where(selected_basis(x) > 0)])
                for t in eligible_knots:
                    candidate_model = deepcopy(model)
                    unhinged_candidate = deepcopy(selected_basis)
                    unhinged_candidate.add(v, 0.0, False)
                    hinged_candidate = deepcopy(selected_basis)
                    hinged_candidate.add(v, t, True)
                    candidate_model.add([unhinged_candidate, hinged_candidate])
                    gcv = evaluate_model(x, y, candidate_model)
                    if gcv < best_gcv:
                        best_gcv = gcv
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
    best_gcv = evaluate_model(x, y, model)

    while len(best_trimmed_model) > 1:
        best_trimmed_gcv = np.inf
        previous_model = deepcopy(best_trimmed_model)
        for i in range(len(previous_model)):
            if i == 0:  # First basis function (constant 1) cannot be excluded
                continue
            trimmed_model = deepcopy(previous_model).remove(i)
            gcv = evaluate_model(x, y, trimmed_model)
            if gcv < best_trimmed_gcv:
                best_trimmed_gcv = gcv
                best_trimmed_model = trimmed_model
            if gcv < best_gcv:
                best_gcv = gcv
                best_model = trimmed_model

    return best_model


def fit(x: np.ndarray, y: np.ndarray, m_max: int) -> Model:
    model = forward_pass(x, y, m_max)
    model = backward_pass(x, y, model)

    return model
