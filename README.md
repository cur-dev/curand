# curand

* **Version:** 0.1-0
* **License:** [BSD 2-Clause](http://opensource.org/licenses/BSD-2-Clause)
* **Author:** Drew Schmidt


Fast random number generators on gpu's via NVIDIA® CUDA™. Not officially affiliated with or endorsed by NVIDIA in any way, but we like their work.


## Installation

<!-- To install the R package, run:

```r
install.package("curand")
``` -->

The development version is maintained on GitHub:

```r
remotes::install_github("wrathematics/curand")
```

You will need to have an installation of CUDA to build the package. You can download CUDA from the [nvidia website](https://developer.nvidia.com/cuda-downloads). You will also need the development version of the float package:

```r
remotes::install_github("wrathematics/float")
```

Also, R must have been compiled with `--enable-R-shlib=yes`. Otherwise, the package probably won't build. I hope to fix this eventually.



## Package Use

We offer several generators. Alphabetically, these are:

* `curand_log_normal()`
* `curand_normal()`
* `curand_poisson()`
* `curand_uniform()`
* `curand_weibull()`

These generators that behave similarly to those in the stats package. So `curand_uniform()` works like `runif()`, `curand_normal()` works like `rnorm()`, etc. Function calls map to the cuRAND, but the function arguments/behavior are similar to their base R counterparts.

Only one gpu will be used at a time. My opinion is that if you want to do multi-gpu, you should be distributing your work with something like MPI. See the [pbdMPI package](https://cran.r-project.org/web/packages/pbdMPI/index.html).

Generators do not respect `set.seed()` seeds. You have to pass a seed as an argument to the function. By default, a seed that mixes the bits of the date, time, and pid via a hash mixing function will be used.



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

32-bit floats are also natively supported:

```r
curand_uniform(100, type="float")
## # A float32 vector: 100
## [1] 0.810296 0.447812 0.064345 0.643384 0.495918 ... 

system.time(curand_uniform(1e8))
##  user  system elapsed 
## 0.172   0.332   0.504 
system.time(curand_uniform(1e8, type="float"))
##  user  system elapsed 
## 0.088   0.196   0.282 
```
