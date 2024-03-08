@rem Make standalone earth using the Microsoft compiler (tested on vc9 and vc10).
@rem Apart from -DSTANDALONE and -DMAIN, the only unusual flag here is -TP.
@rem Needed because earth.c uses a subset of C90, which is not directly
@rem supported by the Microsoft compilers.  So fake it out by telling the
@rem compiler that earth.c is a C++ file.

cl -nologo -DSTANDALONE -DMAIN -TP -O2 -W3 -c ../src/earth.c
                                               @if %errorlevel% NEQ 0 goto error
cl -nologo -O2 -W3 ^
 -c ../src/blas/d_sign.c ../src/blas/daxpy.c ../src/blas/dcopy.c ^
 ../src/blas/ddot.c ../src/blas/dnrm2.c ../src/blas/dqrls.c ^
 ../src/blas/dqrsl.c ../src/blas/dscal.c ../src/blas/dtrsl.c ^
 ../src/R/dqrdc2.c
                                                @if %errorlevel% NEQ 0 goto error
link -nologo -out:earthmain-cl.exe ^
 earth.obj d_sign.obj daxpy.obj dcopy.obj ddot.obj dnrm2.obj dqrls.obj ^
 dqrsl.obj dscal.obj dtrsl.obj dqrdc2.obj
                                                @if %errorlevel% NEQ 0 goto error
earthmain-cl >temp.txt
                                                @if %errorlevel% NEQ 0 goto error
diff temp.txt earthmain-test.txt
                                                @if %errorlevel% NEQ 0 goto error
del temp.txt
                                                @if %errorlevel% NEQ 0 goto error
@goto done
:error
@echo ==== ERROR ====
:done
