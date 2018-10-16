# return a vector of NaN's of length n1^2 + n2
#' @useDynLib curand R_curand_setnan
setnan = function(n1, n2, type)
{
  .Call(R_curand_setnan, as.integer(n1), as.integer(n2), type)
}
