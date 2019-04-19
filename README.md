# pyMCSpearman

A python implementation of [Curran (2014)](https://arxiv.org/abs/1411.3816).
The original C implementation can be found [here](https://github.com/PACurran/MCSpearman/) ([ASCL entry](http://ascl.net/1504.008)).

## Status

The python implementation is currently significantly slower than the original C implementation.

All three modes (bootstrapping only, perturbing only, and composite) have been tested against Curran's code using the test data provided with MCSpearman.

## Requirements

- python3
- scipy
- numpy

