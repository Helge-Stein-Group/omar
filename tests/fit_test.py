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


def fit_test():
    model, x, y = create_case()

    model.fit(x, y)

    result = np.linalg.lstsq(model.fit_matrix, y, rcond=None)
    coefficients = result[0]

    residual1 = np.linalg.norm(y - model.fit_matrix @ coefficients)
    residual2 = np.linalg.norm(y - model.fit_matrix @ model.coefficients)
    model.calculate_gcv(y)
    gcv2 = model.gcv
    model.coefficients = coefficients
    model.calculate_gcv(y)
    gcv1 = model.gcv
    assert np.abs(1 - residual1 / residual2) < 0.05
    assert np.abs(1 - gcv1 / gcv2) < 0.05


def update_case():
    model, x, y = create_case()

    model.calculate_fit_matrix(x)
    model.calculate_covariance_matrix()
    model.calculate_right_hand_side(y)

    add_basis = deepcopy(model.basis[0])
    add_basis.add(0, x[-5, 0], True)
    model.basis[-1] = add_basis

    return model, x, y


def update_fit_matrix_test():
    model, x, y = update_case()
    former_fit_matrix = model.fit_matrix.copy()

    model.update_initialisation(x, x[-2, 0], x[-5, 0], 0)
    model.update_fit_matrix()
    updated_fit_matrix = model.fit_matrix.copy()

    model.calculate_fit_matrix(x)
    full_fit_matrix = model.fit_matrix.copy()

    assert np.allclose(updated_fit_matrix, full_fit_matrix)


def update_covariance_matrix_test():
    model, x, y = update_case()
    former_covariance = model.covariance_matrix.copy()

    model.update_initialisation(x, x[-2, 0], x[-5, 0], 0)
    model.update_covariance_matrix()
    updated_covariance = model.covariance_matrix.copy()

    model.calculate_fit_matrix(x)
    model.calculate_covariance_matrix()
    full_covariance = model.covariance_matrix.copy()
    assert np.allclose(updated_covariance[:-1, :-1], full_covariance[:-1, :-1])
    assert np.allclose(updated_covariance[-1, :-1], full_covariance[-1, :-1])
    assert np.allclose(updated_covariance, full_covariance)

def update_covariance_matrix_twice_test():
    model, x, y = update_case()
    former_covariance = model.covariance_matrix.copy()

    model.update_initialisation(x, x[-2, 0], x[-5, 0], 0)
    model.update_covariance_matrix()
    model.update_fit_matrix() # this becomes necessary anyway for more than 1 update
    add_basis = deepcopy(model.basis[0])
    add_basis.add(0, x[-8, 0], True)
    model.basis[-1] = add_basis
    model.update_initialisation(x, x[-5, 0], x[-8, 0], 0)
    model.update_covariance_matrix()
    updated_covariance = model.covariance_matrix.copy()

    model.calculate_fit_matrix(x)
    model.calculate_covariance_matrix()
    full_covariance = model.covariance_matrix.copy()
    assert np.allclose(updated_covariance[:-1, :-1], full_covariance[:-1, :-1])
    assert np.allclose(updated_covariance[-1, :-1], full_covariance[-1, :-1])
    assert np.allclose(updated_covariance, full_covariance)


def update_right_hand_side_test():
    model, x, y = update_case()
    former_right_hand_side = model.right_hand_side.copy()

    model.update_initialisation(x, x[-2, 0], x[-5, 0], 0)
    model.update_right_hand_side(y)
    updated_right_hand_side = model.right_hand_side.copy()

    model.calculate_fit_matrix(x)
    model.calculate_right_hand_side(y)
    full_right_hand_side = model.right_hand_side.copy()

    assert np.allclose(updated_right_hand_side, full_right_hand_side)


def decompose_test():
    model, x, y = update_case()
    former_covariance = model.covariance_matrix.copy()

    model.update_initialisation(x, x[-2, 0], x[-5, 0], 0)
    covariance_addition = model.update_covariance_matrix()
    updated_covariance = model.covariance_matrix.copy()

    eigenvalues, eigenvectors = model.decompose_addition(covariance_addition)
    reconstructed_covariance = former_covariance + eigenvalues[0] * np.outer(eigenvectors[0], eigenvectors[0])
    reconstructed_covariance += eigenvalues[1] * np.outer(eigenvectors[1], eigenvectors[1])

    assert np.allclose(reconstructed_covariance, updated_covariance)


def update_cholesky_test():
    model, x, y = create_case()

    former_cholesky = model.fit(x, y)

    add_basis = deepcopy(model.basis[0])
    add_basis.add(0, x[-5, 0], True)
    model.basis[-1] = add_basis
    model.update_initialisation(x, x[-2, 0], x[-5, 0], 0)
    covariance_addition = model.update_covariance_matrix()
    eigenvalues, eigenvectors = model.decompose_addition(covariance_addition)
    updated_cholesky = regression.update_cholesky(former_cholesky, eigenvectors, eigenvalues)

    model.calculate_fit_matrix(x)
    model.calculate_covariance_matrix()
    model.calculate_right_hand_side(y)
    full_cholesky = model.fit(x, y)
    assert np.allclose(np.tril(updated_cholesky), np.tril(full_cholesky))


def update_fit_test():
    model, x, y = create_case()

    former_tri = model.fit(x, y)
    former_coefficients = model.coefficients.copy()

    add_basis = deepcopy(model.basis[0])
    add_basis.add(0, x[-5, 0], True)
    model.basis[-1] = add_basis

    updated_tri = model.update_fit(x, y, former_tri, x[-2, 0], x[-5, 0], 0)
    updated_coefficients = model.coefficients.copy()
    updated_gcv = model.gcv

    full_tri = model.fit(x, y)
    full_coefficients = model.coefficients.copy()
    full_gcv = model.gcv

    assert np.allclose(np.tril(updated_tri), np.tril(full_tri))
    assert np.allclose(updated_coefficients, full_coefficients)
    assert np.allclose(updated_gcv, full_gcv)


fit_test()

update_fit_matrix_test()

update_covariance_matrix_test()
update_right_hand_side_test()
decompose_test()
update_cholesky_test()
update_covariance_matrix_twice_test()

#update_fit_test()
