#include "cu_utils.h"
#include "rand_utils.h"
#include "Rcurand.h"


template <typename T>
__global__ void runif(curandState *state, const T min, const T max, const int gpulen, T *x)
{
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx >= gpulen)
    return;
  
  T tmp = curand_uniform(state + idx);
  x[idx] = min + (max - min)*tmp;
}



extern "C" SEXP R_curand_uniform(SEXP n1_, SEXP n2_, SEXP min_, SEXP max_, SEXP seed_, SEXP type_)
{
  SEXP x;
  const int32_t n1 = INT(n1_);
  const int32_t n2 = INT(n2_);
  const R_xlen_t n = (R_xlen_t)n1*n1 + n2;
  
  const unsigned int seed = INTEGER(seed_)[0];
  const int type = INT(type_);
  
  const double min = REAL(min_)[0];
  const double max = REAL(max_)[0];
  
  
  if (type == TYPE_DOUBLE)
  {
    PROTECT(x = allocVector(REALSXP, n));
    curand_rng_driver(seed, n, min, max, REAL(x), runif);
  }
  else if (type == TYPE_FLOAT)
  {
    PROTECT(x = allocVector(INTSXP, n));
    curand_rng_driver(seed, n, (float)min, (float)max, FLOAT(x), runif);
  }
  else
    error("impossible type\n");
  
  
  UNPROTECT(1);
  return x;
}
