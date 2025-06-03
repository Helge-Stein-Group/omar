project('omar',
        ['c', 'fortran'],
        version : '0.1',
        meson_version: '>= 1.1.0',
        default_options : [
                            'warning_level=1',
                            'buildtype=release'
                          ])
fc = meson.get_compiler('fortran')

py = import('python').find_installation(pure: false)
py_dep = py.dependency()

incdir_numpy = run_command(py,
  ['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
  check : true
).stdout().strip()

incdir_f2py = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(numpy.f2py.get_include())'],
    check : true
).stdout().strip()

inc_np = include_directories(incdir_numpy)
inc_f2py = include_directories(incdir_f2py)
inc_np = include_directories(incdir_numpy, incdir_f2py)

fortranobject_c = run_command(py,
    ['-c', 'import os; os.chdir(".."); import numpy.f2py; print(os.path.join(numpy.f2py.get_include(), "fortranobject.c"))'],
    check : true
).stdout().strip()

# gh-25000
quadmath_dep = fc.find_library('quadmath', required: false)

gomp = declare_dependency(link_args : ['-lgomp'])

if host_machine.system() == 'windows'
  gcc_link_args = ['-lucrt', '-static']
  add_project_link_arguments(gcc_link_args, language: ['c', 'cpp', 'fortran'])
  
  blas = fc.find_library('openblas', dirs: ['C:/msys64/mingw64/lib'], required: true)
  lapack = blas  # OpenBLAS typically includes LAPACK functionality
else
  blas = declare_dependency(link_args : ['-lblas'])
  lapack = declare_dependency(link_args : ['-llapack'])
endif

subdir('omar')