#include "cu_utils.h"
#include "rand_utils.h"
#include "Rcurand.h"


template <typename T>
__global__ void rnorm(curandState *state, const T mean, const T sd, const int gpulen, T *x)
{
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx >= gpulen)
    return;
  
  T tmp = curand_normal(state + idx);
  x[idx] = sd*tmp + mean;
}



extern "C" SEXP R_curand_normal(SEXP n1_, SEXP n2_, SEXP mean_, SEXP sd_, SEXP seed_, SEXP type_)
{
  SEXP x;
  const int32_t n1 = INT(n1_);
  const int32_t n2 = INT(n2_);
  const R_xlen_t n = (R_xlen_t)n1*n1 + n2;
  
  const unsigned int seed = INTEGER(seed_)[0];
  const int type = INT(type_);
  
  const double mean = REAL(mean_)[0];
  const double sd = REAL(sd_)[0];
  
  
  if (type == TYPE_DOUBLE)
  {
    PROTECT(x = allocVector(REALSXP, n));
    curand_rng_driver(seed, n, mean, sd, REAL(x), rnorm);
  }
  else if (type == TYPE_FLOAT)
  {
    PROTECT(x = allocVector(INTSXP, n));
    curand_rng_driver(seed, n, (float)mean, (float)sd, FLOAT(x), rnorm);
  }
  else
    error("impossible type\n");
  
  
  UNPROTECT(1);
  return x;
}
