from copy import deepcopy
from dataclasses import dataclass, field

import numpy as np

from omars.utils import Sign, hinge


@dataclass
class Basis:
    v: list[int] = field(default_factory=list)  # Index of variable
    t: list[float] = field(default_factory=list)  # Knot value
    sign: list[Sign] = field(default_factory=list)  # Sign of basis function

    def __call__(self, x: np.ndarray, i: int = 0) -> np.ndarray:
        assert x.ndim == 2
        assert isinstance(i, int)

        if not self.v or i == len(self.v):
            return np.ones(x.shape[0])
        else:
            sign = self.sign[i].value
            v = self.v[i]
            t = self.t[i]
            return hinge(sign * (x[:, v] - t)) * self.__call__(x, i + 1)

    def add(self, v: int, t: float, sign: Sign):
        assert isinstance(v, int)
        assert isinstance(t, float)
        assert isinstance(sign, Sign)

        self.v.append(v)
        self.t.append(t)
        self.sign.append(sign)


@dataclass
class Model:
    coefficients: np.ndarray = None
    basis: list[Basis] = field(default_factory=lambda: [Basis()])

    def data_matrix(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2

        return np.column_stack([basis(x) for basis in self.basis])

    def __call__(self, x: np.ndarray) -> np.ndarray:
        assert x.ndim == 2

        return self.data_matrix(x) @ self.coefficients

    def __len__(self):
        return len(self.basis)

    def add(self, bases: list[Basis]):
        assert all(isinstance(b, Basis) for b in bases)

        self.basis.extend(bases)
        return self

    def remove(self, i: int):
        assert isinstance(i, int)
        assert i < len(self.basis)

        del self.basis[i]
        return self


def evaluate_model_mse(x: np.ndarray, y: np.ndarray, model: Model) -> float:
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    assert isinstance(model, Model)

    '''
    ATTENTION !!!
    The basis through the forward pass does not need to be linearly independent, 
    which may lead to a singular matrix, this is can be solved by pivoting in the 
    Cholesky decomposition, in the paper a small value is added to the diagonal.
    Here numpy can deal with singular matrices, but the lack of fit needs to 
    be recalculated.
    '''
    data_matrix = model.data_matrix(x)
    coefficients, lof, rank, sv = np.linalg.lstsq(data_matrix, y, rcond=None)
    model.coefficients = coefficients
    if lof.size == 0:
        y_pred = data_matrix @ coefficients
        lof = np.sum((y - y_pred) ** 2)

    return lof / len(y)


def forward_pass(x: np.ndarray, y: np.ndarray, m_max: int) -> Model:
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    assert isinstance(m_max, int)

    covariates = set(range(x.shape[1]))
    model = Model()
    while len(model) < m_max:
        best_mse = np.inf
        best_candidate_model = None
        for m, selected_basis in enumerate(model.basis):
            ineligible_covariates = set(selected_basis.v)
            eligible_covariates = covariates - ineligible_covariates
            for v in eligible_covariates:
                eligible_knots = set([
                    sample[v] for sample in x if
                    selected_basis(sample.reshape(1, -1)) > 0
                ])
                for t in eligible_knots:
                    candidate_model = deepcopy(model)
                    pos_candidate = deepcopy(selected_basis)
                    pos_candidate.add(v, t, Sign.pos)
                    neg_candidate = deepcopy(selected_basis)
                    neg_candidate.add(v, t, Sign.neg)
                    candidate_model.add([pos_candidate, neg_candidate])
                    mse = evaluate_model_mse(x, y, candidate_model)
                    if mse < best_mse:
                        best_mse = mse
                        best_candidate_model = candidate_model

        model = best_candidate_model

    return model


def evaluate_model_gcv(x: np.ndarray, y: np.ndarray, model: Model,
                       c: float = 3) -> float:
    mse = evaluate_model_mse(x, y, model)
    c_m = len(model) + c * (len(model) - 1) / 2
    return mse / (1 - c_m / len(y)) ** 2
#Probem is here?

def backward_pass(x: np.ndarray, y: np.ndarray, model: Model) -> Model:
    assert x.ndim == 2
    assert y.ndim == 1
    assert x.shape[0] == y.shape[0]
    assert isinstance(model, Model)

    best_model = model
    best_trimmed_model = deepcopy(model)
    best_gcv = evaluate_model_gcv(x, y, model)

    while len(best_trimmed_model) > 1:
        best_trimmed_gcv = np.inf
        previous_model = deepcopy(best_trimmed_model)
        for i in range(len(previous_model)):
            if i == 0:  # First basis function (constant 1) cannot be excluded
                continue
            trimmed_model = deepcopy(previous_model).remove(i)
            gcv = evaluate_model_gcv(x, y, trimmed_model)
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
