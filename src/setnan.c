#include <stdint.h>

#include "Rcurand.h"


static inline void set_vec_nan(const R_xlen_t n, double *const x)
{
  for (R_xlen_t i=0; i<n; i++)
    x[i] = R_NaN;
}

static inline void set_vec_nanf(const R_xlen_t n, float *const x)
{
  for (R_xlen_t i=0; i<n; i++)
    x[i] = R_NaNf;
}



SEXP R_curand_setnan(SEXP n1_, SEXP n2_, SEXP type_)
{
  SEXP x;
  const int32_t n1 = INT(n1_);
  const int32_t n2 = INT(n2_);
  const R_xlen_t n = (R_xlen_t)n1*n1 + n2;
  
  const int type = INT(type_);
  
  if (type == TYPE_DOUBLE)
  {
    PROTECT(x = allocVector(REALSXP, n));
    set_vec_nan(n, REAL(x));
  }
  else if (type == TYPE_FLOAT)
  {
    PROTECT(x = allocVector(INTSXP, n));
    set_vec_nanf(n, FLOAT(x));
  }
  else
    error("impossible type\n");
  
  UNPROTECT(1);
  return x;
}
