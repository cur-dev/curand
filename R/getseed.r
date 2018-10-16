#' @useDynLib curand R_getseed
getseed = function()
{
  date = as.integer(Sys.Date())
  time = as.numeric(Sys.time())*100000
  pid = as.integer(Sys.getpid())
  
  .Call(R_getseed, date, time, pid)
}
