# pyMCSpearman

A tool to calculate Spearman's rank correlation coefficient for data, using bootstrapping and/or perturbation to estimate the uncertainties on the correlation coefficient.
This is a python implementation of [Curran (2014)](https://arxiv.org/abs/1411.3816).
Curran's original C implementation is [`MCSpearman`](https://github.com/PACurran/MCSpearman/) ([ASCL entry](http://ascl.net/1504.008)).

## Status

The python implementation is currently noticably slower than the original C implementation.
For the test data (53 entries) and 1e5 iterations:

```
$ time ./mcspearman -infile test.data -i 10000 -method 3
...
real	0m0.394s
user	0m0.380s
sys	0m0.011s

$ time python3 pyMCSpearman.py     # with only composite method left uncommented
...
real	0m5.000s
user	0m4.703s
sys 	0m0.078s
```

All three modes (bootstrapping only, perturbing only, and composite) have been tested against Curran's code using the test data provided with `MCSpearman`.

## Requirements

- python3
- scipy
- numpy

