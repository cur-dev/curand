#ifndef _RCURAND_MATH_UTILS_H_
#define _RCURAND_MATH_UTILS_H_


__device__ static inline float Pow(const float x, const float y)
{
  return powf(x, y);
}

__device__ static inline double Pow(const double x, const double y)
{
  return pow(x, y);
}

__device__ static inline float Log(const float x)
{
  return log(x);
}

__device__ static inline double Log(const double x)
{
  return log(x);
}


#endif
