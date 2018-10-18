# curand

* **Version:** 0.1-0
* **Status:** [![Build Status](https://travis-ci.org/wrathematics/curand.png)](https://travis-ci.org/wrathematics/curand)
* **License:** [BSD 2-Clause](http://opensource.org/licenses/BSD-2-Clause)
* **Author:** Drew Schmidt


Fast random number generators on gpu's via CUDA.


## Installation

<!-- To install the R package, run:

```r
install.package("curand")
``` -->

The development version is maintained on GitHub:

```r
remotes::install_github("wrathematics/curand")
```

You will need to have an installation of CUDA to build the package.



## Examples

All timings are from:

* A single Volta gpu on a DGX-1
* R 3.4.4
* CUDA 9.0.176

```r
n = 1e8
memuse::howbig(n)
## 762.939 MiB

system.time(stats::runif(n))
##  user  system elapsed 
## 2.440   0.272   2.709 
system.time(gpunif <- curand::curand_uniform(n))
##  user  system elapsed 
##   0.192   0.316   0.507 
head(gpunif)
## [1] 0.76890665 0.09101996 0.25215226 0.20540376 0.91397750 0.69127667
```

Long vectors are also supported:

```r
n = 2.5e9
memuse::howbig(n)
## 18.626 GiB

system.time(stats::runif(n))
##  user  system elapsed 
## 60.548   6.676  67.201 
system.time(gpunif <- curand::curand_uniform(n))
##  user  system elapsed 
## 4.792   7.952  12.739
```

The first execution of any generator will be slow due to CUDA initialization overhead (the above examples were not first executions)

```r
system.time(curand::curand_uniform(1))
##  user  system elapsed 
## 0.032   0.416   0.450 
system.time(curand::curand_uniform(1))
##  user  system elapsed 
## 0.000   0.000   0.001 
```

Generators are also generally slower than those in the stats package (e.g., `stats::runif()`) for small 

```r
system.time(curand_uniform(100))
##  user  system elapsed 
## 0.000   0.000   0.001 
system.time(runif(100))
## user  system elapsed 
##    0       0       0 
```
