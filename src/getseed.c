#include "include/Rcurand.h"


// Robert Jenkins' 96 bit Mix Function
static inline int mix_96(int a, int b, int c)
{
  a=a-b;  a=a-c;  a=a^(c >> 13);
  b=b-c;  b=b-a;  b=b^(a << 8);
  c=c-a;  c=c-b;  c=c^(b >> 13);
  a=a-b;  a=a-c;  a=a^(c >> 12);
  b=b-c;  b=b-a;  b=b^(a << 16);
  c=c-a;  c=c-b;  c=c^(b >> 5);
  a=a-b;  a=a-c;  a=a^(c >> 3);
  b=b-c;  b=b-a;  b=b^(a << 10);
  c=c-a;  c=c-b;  c=c^(b >> 15);
  
  return c;
}



SEXP R_getseed(SEXP date, SEXP time_, SEXP pid)
{
  SEXP ret;
  PROTECT(ret = allocVector(INTSXP, 1));
  unsigned int time = (unsigned int) DBL(time_);
  
  INT(ret) = mix_96(INT(date), time, INT(pid));
  
  UNPROTECT(1);
  return ret;
}
