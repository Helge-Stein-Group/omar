py.extension_module('backend',
                     [
                     '''backend.f90''',
                     '''backendmodule.c''',
                     '''backend-f2pywrappers2.f90''',
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