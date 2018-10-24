#' curand_uniform
#' 
#' Generate from a uniform distribution using a gpu.
#' 
#' @param n 
#' The number of values to generate
#' @param min,max 
#' Parameters for uniform random variables.
#' @param seed 
#' Seed for the random number generation.
#' @param type
#' 'double' or 'float'
#' 
#' @references CUDA Toolkit Documentation for cuRAND
#' \url{https://docs.nvidia.com/cuda/curand/index.html}
#' 
#' @useDynLib curand R_curand_uniform
#' @export
curand_uniform = function(n, min=0, max=1, seed=getseed(), type="double")
{
  type = match.arg(tolower(type), c("double", "float"))
  type = ifelse(type == "double", TYPE_DOUBLE, TYPE_FLOAT)
  
  if (n < 0)
    stop("invalid arguments")
  else if (n == 0)
  {
    if (type == TYPE_DOUBLE)
      return(numeric(0))
    else
      return(float(0))
  }
  
  n1 = floor(sqrt(n))
  n2 = n - n1*n1
  
  if (min > max || is.badval(min) || is.badval(max))
  {
    warning("NAs produced")
    ret = setnan(n1, n2, type)
  }
  else
    ret = .Call(R_curand_uniform, as.integer(n1), as.integer(n2), as.double(min), as.double(max), as.integer(seed), type)
  
  if (type == TYPE_DOUBLE)
    ret
  else
    float32(ret)
}
