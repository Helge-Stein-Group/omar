CC= gcc
CFLAGS= -DSTANDALONE -DMAIN -Wall -pedantic -Wextra -O3 -std=gnu99 
BLASSRC= ../src/blas/d_sign.c ../src/blas/daxpy.c ../src/blas/dcopy.c ../src/blas/ddot.c ../src/blas/dnrm2.c ../src/blas/dqrls.c ../src/blas/dqrsl.c ../src/blas/dscal.c ../src/blas/dtrsl.c
RSRC= ../src/R/dqrdc2.c

earth:  
	$(CC) $(CFLAGS) -o earth ../src/earth.c $(BLASSRC) $(RSRC)

clean:
	rm earth
