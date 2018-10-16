is.badval <- function(x)
{
  is.na(x) || is.nan(x) || is.infinite(x)
}
