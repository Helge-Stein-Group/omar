# Fortran unit tests
test_utils_exe = executable('unit_test',
                           ['utils.f90', 'unit_test.f90', '../backend.f90'],
                           dependencies: [quadmath_dep, gomp, blas, lapack],
                           install: false)

# Register the test with Meson
test('fortran_utils_test', test_utils_exe) 
