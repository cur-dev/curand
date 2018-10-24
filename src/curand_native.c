/* Automatically generated. Do not edit by hand. */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>

extern SEXP R_getseed(SEXP date, SEXP time_, SEXP pid);
extern SEXP R_curand_rexp(SEXP n1_, SEXP n2_, SEXP rate_, SEXP seed_, SEXP type_);
extern SEXP R_curand_rlnorm(SEXP n1_, SEXP n2_, SEXP meanlog_, SEXP sdlog_, SEXP seed_, SEXP type_);
extern SEXP R_curand_rnorm(SEXP n1_, SEXP n2_, SEXP mean_, SEXP sd_, SEXP seed_, SEXP type_);
extern SEXP R_curand_rpois(SEXP n1_, SEXP n2_, SEXP lambda_, SEXP seed_);
extern SEXP R_curand_runif(SEXP n1_, SEXP n2_, SEXP min_, SEXP max_, SEXP seed_, SEXP type_);
extern SEXP R_curand_rweibull(SEXP n1_, SEXP n2_, SEXP shape_, SEXP scale_, SEXP seed_, SEXP type_);
extern SEXP R_curand_setnan(SEXP n1_, SEXP n2_, SEXP type_);

static const R_CallMethodDef CallEntries[] = {
  {"R_getseed", (DL_FUNC) &R_getseed, 3},
  {"R_curand_rexp", (DL_FUNC) &R_curand_rexp, 5},
  {"R_curand_rlnorm", (DL_FUNC) &R_curand_rlnorm, 6},
  {"R_curand_rnorm", (DL_FUNC) &R_curand_rnorm, 6},
  {"R_curand_rpois", (DL_FUNC) &R_curand_rpois, 4},
  {"R_curand_runif", (DL_FUNC) &R_curand_runif, 6},
  {"R_curand_rweibull", (DL_FUNC) &R_curand_rweibull, 6},
  {"R_curand_setnan", (DL_FUNC) &R_curand_setnan, 3},
  {NULL, NULL, 0}
};

void R_init_coop(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
