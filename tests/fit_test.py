import numpy as np
from copy import deepcopy

import regression


def create_case():
    x = np.random.normal(size=(100, 3))
    x = x[x[:, 0].argsort()]
    y = np.random.normal(size=100)
    y = y[x[:, 0].argsort()]
    model = regression.Model()
    basis_addition = deepcopy(model.basis[0])
    basis_addition.add(0, 0.0, False)
    model.add([basis_addition])
    basis_addition = deepcopy(model.basis[0])
    basis_addition.add(0, x[-2, 0], True)
    model.add([basis_addition])
    return model, x, y


def update_case():
    model, x, y = create_case()

    model.calculate_fit_matrix(x)
    model.calculate_covariance_matrix()
    model.calculate_right_hand_side(y)

    add_basis = deepcopy(model.basis[0])
    add_basis.add(0, x[-5, 0], True)
    model.basis[-1] = add_basis

    return model, x, y


def fit_matrix_update_test():
    model, x, y = update_case()
    former_fit_matrix = model.fit_matrix.copy()

    model.update_fit_matrix([-1], x)
    updated_fit_matrix = model.fit_matrix.copy()

    model.calculate_fit_matrix(x)
    full_fit_matrix = model.fit_matrix.copy()

    assert np.allclose(updated_fit_matrix, full_fit_matrix)


def covariance_update_test():
    model, x, y = update_case()
    former_covariance = model.covariance_matrix.copy()

    model.update_covariance_matrix(x, y, x[-2, 0], x[-5, 0], 0)
    updated_covariance = model.covariance_matrix.copy()

    model.calculate_fit_matrix(x)
    model.calculate_covariance_matrix()
    full_covariance = model.covariance_matrix.copy()

    assert np.allclose(updated_covariance, full_covariance)


def right_hand_side_update_test():
    model, x, y = update_case()
    former_right_hand_side = model.right_hand_side.copy()

    model.update_right_hand_side(x, y, x[-2, 0], x[-5, 0], 0)
    updated_right_hand_side = model.right_hand_side.copy()

    model.calculate_fit_matrix(x)
    model.calculate_right_hand_side(y)
    full_right_hand_side = model.right_hand_side.copy()

    assert np.allclose(updated_right_hand_side, full_right_hand_side)

def decompose_test():
    model, x, y = update_case()
    former_covariance = model.covariance_matrix.copy()

    covariance_addition = model.update_covariance_matrix(x, y, x[-2, 0], x[-5, 0], 0)
    updated_covariance = model.covariance_matrix.copy()

    eigenvalues, eigenvectors = model.decompose_addition(covariance_addition)
    test_covariance = former_covariance + eigenvalues[0] * np.outer(eigenvectors[0], eigenvectors[0])
    test_covariance += eigenvalues[1] * np.outer(eigenvectors[1], eigenvectors[1])

    assert np.allclose(test_covariance, updated_covariance)

def cholesky_update_test():
    model, x, y = create_case()

    former_cholesky = model.fit(x, y)

    add_basis = deepcopy(model.basis[0])
    add_basis.add(0, x[-5, 0], True)
    model.basis[-1] = add_basis
    covariance_addition = model.update_covariance_matrix(x, y, x[-2, 0], x[-5, 0], 0)



def fit_test():
    model, x, y = create_case()

    model.fit(x, y)

    result = np.linalg.lstsq(model.fit_matrix, y, rcond=None)
    coefficients = result[0]

    residual1 = np.linalg.norm(y - model.fit_matrix @ coefficients)
    residual2 = np.linalg.norm(y - model.fit_matrix @ model.coefficients)
    assert np.allclose(residual1, residual2)


def fit_update_test():
    pass


fit_matrix_update_test()
right_hand_side_update_test()
#covariance_update_test()
#decompose_test()
#fit_test()

