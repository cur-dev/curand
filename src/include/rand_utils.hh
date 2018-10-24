#ifndef _RAND_UTILS_H_
#define _RAND_UTILS_H_


#include <curand.h>
#include <curand_kernel.h>

#include "common.h"

__global__ void setup_curand_rng(const int seed, curandState *state, const int gpulen);


template <typename T, typename S>
static inline void curand_rng_driver(const unsigned int seed, const R_xlen_t n, const T a, const T b, S *x, void(*fp)(curandState *, const T, const T, const int, S *))
{
  int gpulen;
  curandState *state;
  S *x_gpu;
  
  get_gpulen(n, &gpulen);
  cudaMalloc(&state, gpulen*sizeof(*state));
  cudaMalloc(&x_gpu, gpulen*sizeof(*x_gpu));
  
  int runs = (int) MAX((R_xlen_t) n/gpulen, 1);
  int rem = (int) MAX((n - (R_xlen_t)(runs*gpulen)), 0);
  int runlen = MAX(gpulen/TPB, 1);
  
  setup_curand_rng<<<runlen, TPB>>>(seed, state, gpulen);
  for (int i=0; i<runs; i++)
  {
    fp<<<runlen, TPB>>>(state, a, b, gpulen, x_gpu);
    cudaMemcpy(x + (R_xlen_t)i*gpulen, x_gpu, gpulen*sizeof(*x_gpu), cudaMemcpyDeviceToHost);
  }
  
  if (rem)
  {
    runlen = MAX(rem/TPB, 1);
    fp<<<runlen, TPB>>>(state, a, b, gpulen, x_gpu);
    cudaMemcpy(x + (R_xlen_t)runs*gpulen, x_gpu, rem*sizeof(*x_gpu), cudaMemcpyDeviceToHost);
  }
  
  
  cudaFree(x_gpu);
  cudaFree(state);
}


#endif
