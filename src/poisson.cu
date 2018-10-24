#include "include/cu_utils.hh"
#include "include/rand_utils.hh"
#include "include/Rcurand.h"


__global__ void rpois(curandState *state, const double lambda, const double ignore, const int gpulen, int *x)
{
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx >= gpulen)
    return;
  
  x[idx] = curand_poisson(state + idx, lambda);
}



extern "C" SEXP R_curand_poisson(SEXP n1_, SEXP n2_, SEXP lambda_, SEXP seed_)
{
  SEXP x;
  const int32_t n1 = INT(n1_);
  const int32_t n2 = INT(n2_);
  const R_xlen_t n = (R_xlen_t)n1*n1 + n2;
  
  const unsigned int seed = INTEGER(seed_)[0];
  const double lambda = REAL(lambda_)[0];
  
  
  PROTECT(x = allocVector(INTSXP, n));
  curand_rng_driver(seed, n, lambda, 0.0, INTEGER(x), rpois);
  
  
  UNPROTECT(1);
  return x;
}
