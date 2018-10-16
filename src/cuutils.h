#ifndef _CUUTILS_H_
#define _CUUTILS_H_


#include <curand.h>
#include <curand_kernel.h>
#include <Rinternals.h>

#define TPB 512

#define GET_ID() (threadIdx.x+blockDim.x*blockIdx.x)


static inline void get_gpulen(const R_xlen_t n, int *const gpulen)
{
  if (n > (R_xlen_t)TPB)
    *gpulen = TPB*512;
  else
    *gpulen = (int) n;
}



#define CUCHECKRET(ret) { cu_check_ret((ret), __FILE__, __LINE__); }
static inline void cu_check_ret(cudaError_t code, char *file, int line)
{
  if (code != cudaSuccess) 
    error("CUDA error: %s %s %d\n", cudaGetErrorString(code), file, line); 
}


static inline int check_(size_t gpu_memneed)
{
  size_t gpu_memfree;
  size_t gpu_memtotl;
  CUCHECKRET(cudaMemGetInfo(&gpu_memfree, &gpu_memtotl));
  
  if (gpu_memfree < gpu_memneed)
    return 0;
  else
    return -1;
}


#endif
