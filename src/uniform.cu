#include "cuutils.h"
#include "Rcurand.h"


__global__ void setup_curand_rng(const int seed, curandState *state, const int gpulen)
{
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx >= gpulen)
    return;
  
  curand_init(seed, idx, 0, state + idx);
}

template <typename T>
__global__ void runif(curandState *state, const T min, const T max, const int gpulen, T *x)
{
  int idx = threadIdx.x + blockDim.x*blockIdx.x;
  if (idx >= gpulen)
    return;
  
  T tmp = curand_uniform(state + idx);
  x[idx] = min + (max - min)*tmp;
}



template <typename T>
static inline void curand_rng_driver(const unsigned int seed, const R_xlen_t n, const T min, const T max, T *x, void(*fp)(curandState *, const T, const T, const int, T *))
{
  int gpulen;
  curandState *state;
  T *x_gpu;
  
  get_gpulen(n, &gpulen);
  cudaMalloc(&state, gpulen*sizeof(*state));
  cudaMalloc(&x_gpu, gpulen*sizeof(*x_gpu));
  
  int runs = (int) MAX((R_xlen_t) n/gpulen, 1);
  int rem = (int) MAX((n - (R_xlen_t)(runs*gpulen)), 0);
  int runlen = MAX(gpulen/TPB, 1);
  
  setup_curand_rng<<<runlen, TPB>>>(seed, state, gpulen);
  for (int i=0; i<runs; i++)
  {
    fp<<<runlen, TPB>>>(state, min, max, gpulen, x_gpu);
    cudaMemcpy(x + (R_xlen_t)i*gpulen, x_gpu, gpulen*sizeof(*x_gpu), cudaMemcpyDeviceToHost);
  }
  
  if (rem)
  {
    runlen = MAX(rem/TPB, 1);
    fp<<<runlen, TPB>>>(state, min, max, gpulen, x_gpu);
    cudaMemcpy(x + (R_xlen_t)runs*gpulen, x_gpu, rem*sizeof(*x_gpu), cudaMemcpyDeviceToHost);
  }
  
  
  cudaFree(x_gpu);
  cudaFree(state);
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
