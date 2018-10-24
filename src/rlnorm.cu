#include "include/cu_utils.hh"
#include "include/rand_utils.hh"
#include "include/Rcurand.h"


template <typename T>
__global__ void rlnorm(curandState *state, const T meanlog, const T sdlog, const int gpulen, T *x)
{
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx >= gpulen)
    return;
  
  x[idx] = curand_log_normal(state + idx, meanlog, sdlog);
}



extern "C" SEXP R_curand_rlnorm(SEXP n1_, SEXP n2_, SEXP meanlog_, SEXP sdlog_, SEXP seed_, SEXP type_)
{
  SEXP x;
  const int32_t n1 = INT(n1_);
  const int32_t n2 = INT(n2_);
  const R_xlen_t n = (R_xlen_t)n1*n1 + n2;
  
  const unsigned int seed = INTEGER(seed_)[0];
  const int type = INT(type_);
  
  const double meanlog = REAL(meanlog_)[0];
  const double sdlog = REAL(sdlog_)[0];
  
  
  if (type == TYPE_DOUBLE)
  {
    PROTECT(x = allocVector(REALSXP, n));
    curand_rng_driver(seed, n, meanlog, sdlog, REAL(x), rlnorm);
  }
  else if (type == TYPE_FLOAT)
  {
    PROTECT(x = allocVector(INTSXP, n));
    curand_rng_driver(seed, n, (float)meanlog, (float)sdlog, FLOAT(x), rlnorm);
  }
  else
    error("impossible type\n");
  
  
  UNPROTECT(1);
  return x;
}
