import fortran.omars as fortran
import utils
from suites import *




def test_data_matrix() -> None:
    print(fortran.__doc__)
    x, y, y_true, nbases, covariates, nodes, hinges, where = utils.generate_data_and_splines(100, 2)

    basis_indices = fortran.omars.active_base_indices(where, nbases-1)
    fit_matrix, basis_mean = fortran.omars.data_matrix(x, basis_indices, covariates,
                                                          nodes, hinges, where)

    suite_data_matrix(x, basis_mean, fit_matrix)
