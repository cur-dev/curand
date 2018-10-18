#include "cu_utils.hh"


__global__ void setup_curand_rng(const int seed, curandState *state, const int gpulen)
{
  int idx = threadIdx.x+blockDim.x*blockIdx.x;
  if (idx >= gpulen)
    return;
  
  curand_init(seed, idx, 0, state + idx);
}
