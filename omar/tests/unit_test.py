import numpy as np
from scipy.linalg import cho_factor
from copy import deepcopy

import omar
import omar.tests.utils as utils


def test_init():
    model = omar.OMAR()

    assert isinstance(model, omar.OMAR)


def test_len():
    model = omar.OMAR()

    assert len(model) == 1


def test_eq():
    model1 = omar.OMAR()
    model2 = omar.OMAR()
    model3 = omar.OMAR()
    model3.nbases += 1

    assert model1 == model2
    assert model1 != model3


def test_getitem():
    model = omar.OMAR()

    assert model[0] == model


def test_active_base_indices():
    model = utils.reference_model(utils.generate_data()[0])

    for backend in omar.Backend:
        model.backend = backend
        assert np.all(model._active_base_indices() == [1, 2, 3, 4]), f"{backend} Backend"


def test_str():
    model = utils.reference_model(utils.generate_data()[0])

    print(model.__str__())
    assert isinstance(model.__str__(), str)


def test_data_matrix() -> None:
    x, y, y_true = utils.generate_data()
    model = utils.reference_model(x)
    ref_data_matrix, ref_data_matrix_mean = utils.reference_data_matrix(x)

    for backend in omar.Backend:
        model.backend = backend
        assert np.allclose(ref_data_matrix,
                           model._data_matrix(x, model._active_base_indices())[0]), f"{backend} Backend: Data matrix"
        assert np.allclose(ref_data_matrix_mean,
                           model._data_matrix(x, model._active_base_indices())[
                               1]), f"{backend} Backend: Data matrix Mean"


def test_covariance_matrix():
    x, y, y_true = utils.generate_data()
    model = utils.reference_model(x)
    ref_data_matrix, ref_data_matrix_mean = utils.reference_data_matrix(x)
    ref_cov_matrix = ref_data_matrix.T @ ref_data_matrix

    for backend in omar.Backend:
        model.backend = backend
        assert np.allclose(ref_cov_matrix, model._covariance_matrix(ref_data_matrix)), f"{backend} Backend"


def test_rhs():
    x, y, y_true = utils.generate_data()
    model = utils.reference_model(x)
    ref_data_matrix, ref_data_matrix_mean = utils.reference_data_matrix(x)
    ref_rhs = utils.reference_rhs(y, ref_data_matrix)

    model.y_mean = y.mean()

    for backend in omar.Backend:
        model.backend = backend
        assert np.allclose(ref_rhs, model._rhs(y, ref_data_matrix)), f"{backend} Backend"


def test_coefficients():
    x, y, y_true = utils.generate_data()
    model = utils.reference_model(x)
    ref_data_matrix, ref_data_matrix_mean = utils.reference_data_matrix(x)
    ref_rhs = utils.reference_rhs(y, ref_data_matrix)
    ref_cov_matrix = ref_data_matrix.T @ ref_data_matrix
    ref_chol = np.tril(cho_factor(ref_cov_matrix, lower=True)[0])
    ref_coefficients = np.linalg.solve(ref_cov_matrix, ref_rhs)

    for backend in omar.Backend:
        model.backend = backend
        assert np.allclose(ref_coefficients,
                           model._coefficients(ref_cov_matrix, ref_rhs)[0]), f"{backend} Backend: Coefficients"
        assert np.allclose(ref_chol,
                           model._coefficients(ref_cov_matrix, ref_rhs)[1]), f"{backend} Backend: Chol"


def test_generalised_cross_validation():
    x, y, y_true = utils.generate_data()
    model = utils.reference_model(x)
    ref_data_matrix, ref_data_matrix_mean = utils.reference_data_matrix(x)
    ref_rhs = utils.reference_rhs(y, ref_data_matrix)
    ref_cov_matrix = ref_data_matrix.T @ ref_data_matrix
    ref_chol = np.tril(cho_factor(ref_cov_matrix, lower=True)[0])
    ref_coefficients = np.linalg.solve(ref_cov_matrix, ref_rhs)

    model.y_mean = y.mean()
    model.coefficients = ref_coefficients

    gcv = {}
    for backend in omar.Backend:
        model.backend = backend
        gcv[backend] = model._generalised_cross_validation(y, ref_data_matrix, ref_chol)
        assert gcv[backend] < 1.0, f"{backend} Backend"

    assert np.allclose(gcv[omar.Backend.FORTRAN], gcv[omar.Backend.PYTHON]), "Unaligned Backends"


def test_fit():
    x, y, y_true = utils.generate_data()
    model = utils.reference_model(x)
    ref_data_matrix, ref_data_matrix_mean = utils.reference_data_matrix(x)
    ref_cov_matrix = ref_data_matrix.T @ ref_data_matrix
    ref_rhs = utils.reference_rhs(y, ref_data_matrix)
    ref_chol = np.tril(cho_factor(ref_cov_matrix, lower=True)[0])
    ref_coefficients = np.linalg.solve(ref_cov_matrix, ref_rhs)

    model.y_mean = y.mean()

    for backend in omar.Backend:
        model.backend = backend
        data_matrix, data_matrix_mean, covariance_matrix, rhs, chol, coefficients, lof = model._fit(x, y)
        assert np.allclose(ref_data_matrix, data_matrix), f"{backend} Backend: Data matrix"
        assert np.allclose(ref_data_matrix_mean, data_matrix_mean), f"{backend} Backend: Data matrix Mean"
        assert np.allclose(ref_cov_matrix, covariance_matrix), f"{backend} Backend: Covariance matrix"
        assert np.allclose(ref_rhs, rhs), f"{backend} Backend: RHS"
        assert np.allclose(ref_chol, chol), f"{backend} Backend: Chol"
        assert np.allclose(ref_coefficients, coefficients), f"{backend} Backend: Coefficients"


def test_update():
    x, y, y_true = utils.generate_data()
    model = utils.reference_model(x)
    ref_data_matrix, ref_data_matrix_mean = utils.reference_data_matrix(x)
    ref_cov_matrix = ref_data_matrix.T @ ref_data_matrix
    ref_rhs = utils.reference_rhs(y, ref_data_matrix)
    ref_chol = np.tril(cho_factor(ref_cov_matrix, lower=True)[0])
    ref_coefficients = np.linalg.solve(ref_cov_matrix, ref_rhs)

    model.y_mean = y.mean()

    for backend in omar.Backend:
        model.backend = backend

        prev_root = x[np.argmin(np.abs(x[:, 1] - 0.8)), 1]
        model.root[2, 4] = prev_root
        next_roots = sorted([value for value in x[:, 1] if value < prev_root], reverse=True)[:3]

        data_matrix = ref_data_matrix.copy()
        data_matrix_mean = ref_data_matrix_mean.copy()
        covariance_matrix = ref_cov_matrix.copy()
        rhs = ref_rhs.copy()
        chol = ref_chol.copy()
        model.coefficients = ref_coefficients.copy()
        for i, next_root in enumerate(next_roots):
            model.root[2, 4] = next_root
            (data_matrix,
             data_matrix_mean,
             covariance_matrix,
             rhs, chol, coefficients, lof) = model._update_fit(data_matrix,
                                                               data_matrix_mean,
                                                               covariance_matrix,
                                                               rhs, chol,
                                                               x, y,
                                                               prev_root, 2)
            coefficients = coefficients.copy()

            comp_data_matrix, comp_data_matrix_mean = model._data_matrix(x, model._active_base_indices())
            comp_covariance_matrix = model._covariance_matrix(comp_data_matrix)
            comp_rhs = model._rhs(y, comp_data_matrix)
            comp_coefficients, comp_chol = model._coefficients(comp_covariance_matrix, comp_rhs)
            comp_lof = model._generalised_cross_validation(y, comp_data_matrix, comp_chol)

            assert np.allclose(data_matrix, comp_data_matrix), f"{backend} Backend {i}: Data matrix"
            assert np.allclose(data_matrix_mean, comp_data_matrix_mean), f"{backend} Backend {i}: Data matrix Mean"
            assert np.allclose(covariance_matrix, comp_covariance_matrix), f"{backend} Backend {i}: Covariance matrix"
            assert np.allclose(rhs, comp_rhs), f"{backend} Backend {i}: RHS"
            assert np.allclose(chol, comp_chol), f"{backend} Backend {i}: Chol"
            assert np.allclose(coefficients, comp_coefficients), f"{backend} Backend {i}: Coefficients"
            assert np.allclose(lof, comp_lof), f"{backend} Backend {i}: Generalised Cross Validation"

            prev_root = next_root


def test_expand_bases():
    x, y, y_true = utils.generate_data()
    model = utils.reference_model(x)
    model.y_mean = y.mean()

    model.mask = np.zeros_like(model.mask)
    model.nbases = 3
    model.mask[1, 1:3] = True
    ref_first_lof = model._fit(x, y)[-1]

    models = {}
    for backend in omar.Backend:
        models[backend] = omar.OMAR(backend=backend)
        models[backend].y_mean = y.mean()
        models[backend]._expand_bases(x, y)

        full_lof = models[backend]._fit(x, y)[-1]

        models[backend].mask = np.zeros_like(model.mask)
        models[backend].nbases = 3
        models[backend].mask[1, 1:3] = True
        first_lof = models[backend]._fit(x, y)[-1]

        assert full_lof < 1, f"{backend} Backend: Full LOF"
        assert first_lof <= ref_first_lof + 1e-6, f"{backend} Backend: First LOF"

    assert models[omar.Backend.FORTRAN].nbases == models[omar.Backend.PYTHON].nbases, "Unaligned Backends"

def test_prune_bases():
    x, y, y_true = utils.generate_data()
    model = utils.reference_model(x)
    model.y_mean = y.mean()

    ref_lof = model._fit(x, y)[-1]

    for backend in omar.Backend:
        test_model = deepcopy(model)

        test_model.mask[:, 5:] = np.random.choice(a=[False, True], size=test_model.mask[:, 5:].shape)
        test_model.mask[1, 5:] = True
        test_model.nbases = 11
        test_model.truncated[:, 5:] = np.random.choice(a=[False, True], size=test_model.truncated[:, 5:].shape)
        test_model.root[:, 5:] = np.random.choice(a=x[:, 0], size=test_model.root[:, 5:].shape)
        test_lof = test_model._fit(x,y)[-1]

        test_model.backend = backend
        test_lof = test_model._prune_bases(x, y, test_lof)

        assert test_model == model, f"{backend} Backend: \n {model} \n vs. \n {test_model}"

def test_find_bases():
    x, y, y_true = utils.generate_data()
    model = utils.reference_model(x)
    model.y_mean = y.mean()

    model.mask = np.zeros_like(model.mask)
    model.nbases = 3
    model.mask[1, 1:3] = True
    ref_first_lof = model._fit(x, y)[-1]

    models = {}
    for backend in omar.Backend:
        models[backend] = omar.OMAR(backend=backend)
        full_lof = models[backend].find_bases(x, y)

        models[backend].mask = np.zeros_like(model.mask)
        models[backend].nbases = 3
        models[backend].mask[1, 1:3] = True
        first_lof = models[backend]._fit(x, y)[-1]

        assert full_lof < 1, f"{backend} Backend: Full LOF"
        assert first_lof <= ref_first_lof + 1e-6, f"{backend} Backend: First LOF"

    assert models[omar.Backend.FORTRAN].nbases == models[omar.Backend.PYTHON].nbases, "Unaligned Backends"