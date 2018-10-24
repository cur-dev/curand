#include "include/cu_utils.hh"
#include "include/math_utils.hh"
#include "include/rand_utils.hh"
#include "include/Rcurand.h"


template <typename T>
__global__ void rweibull(curandState *state, const T shape, const T scale, const int gpulen, T *x)
{
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx >= gpulen)
    return;
  
  T tmp = curand_uniform(state + idx);
  x[idx] = Pow(-Pow(scale, shape) * Log(1-tmp), 1/shape);
}



extern "C" SEXP R_curand_rweibull(SEXP n1_, SEXP n2_, SEXP shape_, SEXP scale_, SEXP seed_, SEXP type_)
{
  SEXP x;
  const int32_t n1 = INT(n1_);
  const int32_t n2 = INT(n2_);
  const R_xlen_t n = (R_xlen_t)n1*n1 + n2;
  
  const unsigned int seed = INTEGER(seed_)[0];
  const int type = INT(type_);
  
  const double shape = REAL(shape_)[0];
  const double scale = REAL(scale_)[0];
  
  
  if (type == TYPE_DOUBLE)
  {
    PROTECT(x = allocVector(REALSXP, n));
    curand_rng_driver(seed, n, shape, scale, REAL(x), rweibull);
  }
  else if (type == TYPE_FLOAT)
  {
    PROTECT(x = allocVector(INTSXP, n));
    curand_rng_driver(seed, n, (float)shape, (float)scale, FLOAT(x), rweibull);
  }
  else
    error("impossible type\n");
  
  
  UNPROTECT(1);
  return x;
}
