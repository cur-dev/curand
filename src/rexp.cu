#include "include/cu_utils.hh"
#include "include/math_utils.hh"
#include "include/rand_utils.hh"
#include "include/Rcurand.h"


template <typename T>
__global__ void rexp(curandState *state, const T rate, const T ignored, const int gpulen, T *x)
{
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx >= gpulen)
    return;
  
  T tmp = curand_uniform(state + idx);
  x[idx] = -Log(tmp) / rate;
}



extern "C" SEXP R_curand_rexp(SEXP n1_, SEXP n2_, SEXP rate_, SEXP seed_, SEXP type_)
{
  SEXP x;
  const int32_t n1 = INT(n1_);
  const int32_t n2 = INT(n2_);
  const R_xlen_t n = (R_xlen_t)n1*n1 + n2;
  
  const unsigned int seed = INTEGER(seed_)[0];
  const int type = INT(type_);
  
  const double rate = REAL(rate_)[0];
  
  
  if (type == TYPE_DOUBLE)
  {
    PROTECT(x = allocVector(REALSXP, n));
    curand_rng_driver(seed, n, rate, 0.0, REAL(x), rexp);
  }
  else if (type == TYPE_FLOAT)
  {
    PROTECT(x = allocVector(INTSXP, n));
    curand_rng_driver(seed, n, (float)rate, 0.0f, FLOAT(x), rexp);
  }
  else
    error("impossible type\n");
  
  
  UNPROTECT(1);
  return x;
}
