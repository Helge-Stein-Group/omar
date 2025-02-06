py.extension_module('fortran_backend',
                     [
                     '''fortran_backend.f90''',
                     '''fortran_backendmodule.c''',
                     '''fortran_backend-f2pywrappers2.f90''',
                     fortranobject_c
                     ],
                     include_directories: [
                     inc_np,

                     ],
                     dependencies : [
                     py_dep,
                     quadmath_dep,
                     gomp,
                     blas,
                     lapack,
                     ],

                     install : true)

py.install_sources('omar.py')