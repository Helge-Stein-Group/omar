import numpy as np
from dataclasses import dataclass, field
from copy import deepcopy

from utils import Sign


@dataclass
class BasisFunctionDescription:
    v: list[int] = field(default_factory=list)  # Index of variable
    t: list[float] = field(default_factory=list)  # Knot value
    sign: list[Sign] = field(default_factory=list)  # Sign of basis function


@dataclass
class CandidateDescription:
    m: int = None  # Index of model function
    v: int = None  # Index of variable
    t: float = None  # Knot value
    lof: float = np.inf  # Lack of fit


def basis_function(
        X: np.ndarray, desc: BasisFunctionDescription, idx: int = 0
) -> np.ndarray:
    assert X.ndim == 2
    if not desc.v or idx == len(desc.v):
        return np.ones(X.shape[0])
    else:
        sign = desc.sign[idx].value
        v = desc.v[idx]
        t = desc.t[idx]
        return np.maximum(np.zeros(X.shape[0]), sign * (X[:, v] - t)) * basis_function(X, desc, idx + 1)


def evaluate_model(
        X: np.ndarray, y: np.ndarray, model_functions: list[BasisFunctionDescription]
) -> float:
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert all(isinstance(desc, BasisFunctionDescription) for desc in model_functions)

    # ATTENTION !!!
    # The basis through the forward pass does not need to be linearly independent, which may lead to a singular matrix,
    # this is can be solved by pivoting in the Cholesky decomposition, in the paper a small value is added to the diagonal.
    # Here numpy can deal with singular matrices, but the lack of fit needs to be recalculated.
    B = np.column_stack([basis_function(X, desc) for desc in model_functions])
    results = np.linalg.lstsq(B, y, rcond=None)
    coefficients = results[0]
    lof = results[1]
    if lof.size == 0:
        y_pred = B @ coefficients
        lof = np.sum((y - y_pred) ** 2)

    return lof / len(y)


def evaluate_candidate(
        X: np.ndarray,
        y: np.ndarray,
        selected_model_function: BasisFunctionDescription,
        v: int,
        t: float,
        model_functions: list[BasisFunctionDescription],
) -> float:
    addition1 = deepcopy(selected_model_function)
    addition1.v.append(v)
    addition1.t.append(t)
    addition1.sign.append(Sign.pos)
    addition2 = deepcopy(selected_model_function)
    addition2.v.append(v)
    addition2.t.append(t)
    addition2.sign.append(Sign.neg)
    return evaluate_model(X, y, model_functions + [addition1, addition2])


def forward_pass(
        X: np.ndarray, y: np.ndarray, M_max: int
) -> list[BasisFunctionDescription]:
    assert X.ndim == 2
    assert y.ndim == 1
    assert X.shape[0] == y.shape[0]
    assert isinstance(M_max, int)

    covariates = set(range(X.shape[1]))
    model_functions = [BasisFunctionDescription()]
    while len(model_functions) < M_max:
        best_candidate = CandidateDescription()
        for m, selected_model_function in enumerate(model_functions):
            ineligible_covariates = set(selected_model_function.v)
            eligible_covariates = covariates - ineligible_covariates
            for v in eligible_covariates:
                eligible_knots = set([
                    X[i, v] for i in range(X.shape[0]) if basis_function(X[i:i + 1, :], selected_model_function) > 0
                ])
                for t in eligible_knots:
                    lof = evaluate_candidate(
                        X, y, selected_model_function, v, t, model_functions
                    )
                    if lof < best_candidate.lof:
                        best_candidate.m = m
                        best_candidate.v = v
                        best_candidate.t = t
                        best_candidate.lof = lof

        addition1 = deepcopy(model_functions[best_candidate.m])
        addition1.v.append(best_candidate.v)
        addition1.t.append(best_candidate.t)
        addition1.sign.append(Sign.pos)

        addition2 = deepcopy(model_functions[best_candidate.m])
        addition2.v.append(best_candidate.v)
        addition2.t.append(best_candidate.t)
        addition2.sign.append(Sign.neg)

        model_functions.append(addition1)
        model_functions.append(addition2)

    return model_functions


def gcv_addition(r, N):
    c = 3
    k = (r - 1) / 2
    C_m = r + c * k
    return (1 - C_m / N) ** 2


def backward_pass(
        X: np.ndarray, y: np.ndarray, model_functions: list[BasisFunctionDescription]
) -> list[BasisFunctionDescription]:
    best_set = set(range(len(model_functions)))
    best_trimmed_set = best_set.copy()
    best_lof = evaluate_model(X, y, model_functions) / gcv_addition(len(best_trimmed_set), len(y))

    while len(best_trimmed_set) > 1:
        best_trimmed_lof = np.inf
        penalty_term = gcv_addition(len(best_trimmed_set)-1, len(y))
        for removal_idx in best_trimmed_set:
            if removal_idx == 0:  # First basis function (constant 1) cannot be excluded
                continue
            current_set = best_trimmed_set - {removal_idx}
            lof = (
                    evaluate_model(X, y, [model_functions[i] for i in current_set])
                    / penalty_term
            )
            if lof < best_trimmed_lof:
                best_trimmed_lof = lof
                best_trimmed_set = current_set
            if lof < best_lof:
                best_lof = lof
                best_set = current_set

    return [model_functions[i] for i in best_set]


def omars(
        X: np.ndarray, y: np.ndarray, M_max: int
) -> tuple[np.ndarray, list[BasisFunctionDescription]]:
    model_functions = forward_pass(X, y, M_max)
    model_functions = backward_pass(X, y, model_functions)

    B = np.column_stack([basis_function(X, desc) for desc in model_functions])
    coefficients, lof = np.linalg.lstsq(B, y, rcond=None)[:2]

    return coefficients, model_functions


def predict(
        X: np.ndarray,
        coefficients: np.ndarray,
        model_functions: list[BasisFunctionDescription],
) -> np.ndarray:
    B = np.column_stack([basis_function(X, desc) for desc in model_functions])
    return B @ coefficients
