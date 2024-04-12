import numpy as np
from copy import deepcopy

import regression


def create_case():
    x = np.random.normal(size=(100, 3))
    y = np.random.normal(size=100)
    model = regression.Model()
    basis_addition = deepcopy(model.basis[0])
    basis_addition.add(0, 0.0, False)
    model.add([basis_addition])
    basis_addition = deepcopy(model.basis[0])
    basis_addition.add(0, x[10, 0], False)
    model.add([basis_addition])
    return model, x, y


def fit_test():
    pass


def fit_insert_test():
    pass


def fit_update_test():
    pass
