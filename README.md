# pymccorrelation

A tool to calculate correlation coefficients for data, using bootstrapping and/or perturbation to estimate the uncertainties on the correlation coefficient.
This was initially a python implementation of the [Curran (2014)](https://arxiv.org/abs/1411.3816) method for calculating uncertainties on Spearman's Rank Correlation Coefficient, but has since been expanded.
Curran's original C implementation is [`MCSpearman`](https://github.com/PACurran/MCSpearman/) ([ASCL entry](http://ascl.net/1504.008)).

Currently the following correlation coefficients can be calculated (with bootstrapping and/or perturbation):

* [Pearson's r](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
* [Spearman's rho](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
* [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)

Kendall's tau can also calculated when some of the data are left/right censored, following the method described by [Isobe+1986](https://ui.adsabs.harvard.edu/abs/1986ApJ...306..490I/abstract).

## Status

All three methods of computing (bootstrapping only, perturbing only, and composite) have been tested against Curran's code using the test data provided with `MCSpearman`.
Unit tests can be run using the `run_tests()` function or by running `python pymccorrelation.py`.
In the case of Speaman's rho, these are compared with the reesults of Curran's C implementation.

The python implementation is currently noticeably slower than the original C implementation.
For the test data (53 entries) and 1e5 iterations:

### Bootstrap

```
$ time ./mcspearman -infile test.data -i 10000 -method 1
...
real	0m0.389s
user	0m0.385s
sys	0m0.005s

$ time python3 pymccorrelation.py    # with only bootstrapping left uncommented
...
real	0m4.542s
user	0m4.475s
sys 	0m0.048s

```

### Perturbation

```
$ time ./mcspearman -infile test.data -i 10000 -method 2
...
real	0m0.330s
user	0m0.320s
sys	0m0.012s

$ time python3 pymccorrelation.py    # with only perturbation left uncommented
...
real	0m4.667s
user	0m4.622s
sys 	0m0.025s
```

### Bootstrap & Perturbation

```
$ time ./mcspearman -infile test.data -i 10000 -method 3
...
real	0m0.394s
user	0m0.380s
sys	0m0.011s

$ time python3 pymccorrelation.py     # with only composite method left uncommented
...
real	0m5.000s
user	0m4.703s
sys 	0m0.078s
```

## Requirements

- python3
- scipy
- numpy

