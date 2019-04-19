# pyMCSpearman

A python implementation of [Curran (2014)](https://arxiv.org/abs/1411.3816).
The original C implementation can be found [here](https://github.com/PACurran/MCSpearman/) ([ASCL entry](http://ascl.net/1504.008)).

## Status

The bootstrapping only method has been validated against the original C code, using the provided test data.
Evaluation using perturbations currently gives different results, compared to the Curran implementation.

The python implementation is currently significantly slower than the original C implementation.

## Requirements

- python3
- scipy
- numpy

