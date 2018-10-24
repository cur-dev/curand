#' curand_poisson
#' 
#' Generate from a poisson distribution using a gpu.
#' 
#' @param n 
#' The number of values to generate
#' @param lambda
#' Parameter for poisson random variables.
#' @param seed 
#' Seed for the random number generation.
#' @param type
#' Ignored; present only to match API with other generators. The return is
#' always a vector of ints.
#' 
#' @references CUDA Toolkit Documentation for cuRAND
#' \url{https://docs.nvidia.com/cuda/curand/index.html}
#' 
#' @useDynLib curand R_curand_poisson
#' @export
curand_poisson = function(n, lambda=1, seed=getseed(), type="double")
{
  n1 = floor(sqrt(n))
  n2 = n - n1*n1
  
  if (n < 0)
    stop("invalid arguments")
  else if (n == 0)
    return(integer(0))
  
  if (lambda < 0 || is.badval(lambda))
  {
    warning("NAs produced")
    ret = setnan(n1, n2, type)
  }
  else
    ret = .Call(R_curand_poisson, as.integer(n1), as.integer(n2), as.double(lambda), as.integer(seed))
  
  ret
}
