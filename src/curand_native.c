/* Automatically generated. Do not edit by hand. */

#include <R.h>
#include <Rinternals.h>
#include <R_ext/Rdynload.h>
#include <stdlib.h>

extern SEXP R_getseed(SEXP date, SEXP time_, SEXP pid);
extern SEXP R_curand_normal(SEXP n1_, SEXP n2_, SEXP mean_, SEXP sd_, SEXP seed_, SEXP type_);
extern SEXP R_curand_poisson(SEXP n1_, SEXP n2_, SEXP lambda_, SEXP seed_);
extern SEXP R_curand_uniform(SEXP n1_, SEXP n2_, SEXP min_, SEXP max_, SEXP seed_, SEXP type_);
extern SEXP R_curand_setnan(SEXP n1_, SEXP n2_, SEXP type_);

static const R_CallMethodDef CallEntries[] = {
  {"R_getseed", (DL_FUNC) &R_getseed, 3},
  {"R_curand_normal", (DL_FUNC) &R_curand_uniform, 6},
  {"R_curand_poisson", (DL_FUNC) &R_curand_poisson, 4},
  {"R_curand_uniform", (DL_FUNC) &R_curand_uniform, 6},
  {"R_curand_setnan", (DL_FUNC) &R_curand_setnan, 3},
  {NULL, NULL, 0}
};

void R_init_coop(DllInfo *dll)
{
  R_registerRoutines(dll, NULL, CallEntries, NULL, NULL);
  R_useDynamicSymbols(dll, FALSE);
}
