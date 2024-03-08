@rem Make standalone earth using the gcc compiler (tested on gcc-4.6.3).
@rem I currently have access only to a Window's machine so this was tested with mingw

gcc -DSTANDALONE -DMAIN -Wall -pedantic -Wextra -O3 -std=gnu99 ^
 -o earthmain-gcc.exe ^
 ../src/earth.c ^
 ../src/blas/d_sign.c ../src/blas/daxpy.c ../src/blas/dcopy.c ^
 ../src/blas/ddot.c ../src/blas/dnrm2.c ../src/blas/dqrls.c ^
 ../src/blas/dqrsl.c ../src/blas/dscal.c ../src/blas/dtrsl.c ^
 ../src/R/dqrdc2.c
                                                @if %errorlevel% NEQ 0 goto error
earthmain-gcc >temp.txt
                                                @if %errorlevel% NEQ 0 goto error
diff temp.txt earthmain-test.txt
                                                @if %errorlevel% NEQ 0 goto error
del temp.txt
                                                @if %errorlevel% NEQ 0 goto error
@goto done
:error
@echo ==== ERROR ====
:done
