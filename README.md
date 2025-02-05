# OMAR

This is a for now naive implementation of the multivariate adaptive
regression splines from Jerome H. Friedman.

## References

1. Friedman, J. (1991). Multivariate adaptive regression splines. The annals of statistics, 
   19(1), 1–67. http://www.jstor.org/stable/10.2307/2241837
2. Friedman, J. (1993). Fast MARS. Stanford University Department of Statistics, Technical Report No 110. 
   https://statistics.stanford.edu/sites/default/files/LCS%20110.pdf
3. Hastie, T., Tibshirani, R., & Friedman, J. The Elements of Statistical Learning (2nd Edition). (2009).  
   Springer Series in Statistic
4. Oswin Krause. Christian Igel. A More Efficient Rank-one Covariance Matrix Update for Evolution Strategies. 
   2015 ACM Conference. https://christian-igel.github.io/paper/AMERCMAUfES.pdf


For Windows make sure to install https://www.intel.com/content/www/us/en/developer/tools/oneapi/hpc-toolkit-download.html?packages=fortran-essentials&fortran-essentials-os=windows&fortran-essentials-win=offline
and follow https://numpy.org/doc/stable/f2py/windows/intel.html to compile fortran

on linux sudo apt-get install gfortran