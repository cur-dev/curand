#' curand_normal
#' 
#' Generate from a normal distribution using a gpu.
#' 
#' @param n 
#' The number of values to generate
#' @param mean,sd 
#' Parameters for normal random variables.
#' @param seed 
#' Seed for the random number generation.
#' @param type
#' 'double' or 'float'
#' 
#' @useDynLib curand R_curand_normal
#' @export
curand_normal <- function(n, mean=0, sd=1, seed=getseed(), type="double")
{
  type = match.arg(tolower(type), c("double", "float"))
  type = ifelse(type == "double", TYPE_DOUBLE, TYPE_FLOAT)
  
  n1 = floor(sqrt(n))
  n2 = n - n1*n1
  
  if (sd < 0 || is.badval(mean) || is.badval(sd))
  {
    warning("NAs produced")
    ret = setnan(n1, n2, type)
  }
  else
    ret = .Call(R_curand_normal, as.integer(n1), as.integer(n2), as.double(mean), as.double(sd), as.integer(seed), type)
  
  if (type == TYPE_DOUBLE)
    ret
  else
    float32(ret)
}
