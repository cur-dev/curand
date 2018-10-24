#' curand_exponetial
#' 
#' Generate from a exponential distribution using a gpu.
#' 
#' @details
#' We use the cuRAND uniform generator together with the fact that if
#' \eqn{X \sim U(0, 1)} then \eqn{Y = -\lambda^{-1}\log(X) \sim Exp(\lambda)}.
#' 
#' @param n
#' The number of values to generate
#' @param rate
#' Parameter for exponential random variables.
#' @param seed 
#' Seed for the random number generation.
#' @param type
#' 'double' or 'float'
#' 
#' @references
#' Casella, G. and Berger, R.L., 2002. Statistical inference (Vol. 2). Pacific
#' Grove, CA: Duxbury.
#' 
#' Rizzo, M.L., 2007. Statistical computing with R. Chapman and Hall/CRC.
#' 
#' @useDynLib curand R_curand_exponential
#' @export
curand_exponetial = function(n, rate=1, seed=getseed(), type="double")
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
  
  if (rate < 0 || is.badval(rate))
  {
    warning("NAs produced")
    ret = setnan(n1, n2, type)
  }
  else
    ret = .Call(R_curand_exponential, as.integer(n1), as.integer(n2), as.double(rate), as.integer(seed), type)
  
  if (type == TYPE_DOUBLE)
    ret
  else
    float32(ret)
}
