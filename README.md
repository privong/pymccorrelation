# pymccorrelation

A tool to calculate correlation coefficients for data, using bootstrapping and/or perturbation to estimate the uncertainties on the correlation coefficient.
This was initially a python implementation of the [Curran (2014)](https://arxiv.org/abs/1411.3816) method for calculating uncertainties on Spearman's Rank Correlation Coefficient, but has since been expanded.
Curran's original C implementation is [`MCSpearman`](https://github.com/PACurran/MCSpearman/) ([ASCL entry](http://ascl.net/1504.008)).

Currently the following correlation coefficients can be calculated (with bootstrapping and/or perturbation):

* [Pearson's r](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)
* [Spearman's rho](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient)
* [Kendall's tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)

Kendall's tau can also calculated when some of the data are left/right censored, following the method described by [Isobe+1986](https://ui.adsabs.harvard.edu/abs/1986ApJ...306..490I/abstract).

## Requirements

- python3
- scipy
- numpy

## Installation

`pymccorrelation` is available via PyPi and can be installed with:

```
pip install pymccorrelation
```

## Usage

`pymccorrelation` exports a single function to the user (also called `pymccorrelation`).

```
from pymccorrelation import pymccorrelation

[... load your data ...]
```

The correlation coefficient can be one of `pearsonr`, `spearmanr`, or `kendallt`.

For example, to compute the Pearson's r for a sample, using 1000 bootstrapping iterations to estimate the uncertainties:

```
res = pymccorrelation(data['x'], data['y'],
                      coeff='pearsonr',
                      Nboot=1000)
```

The output, `res` is a tuple of length 2, and the two elements are:

* numpy array with the correlation coefficient (Pearson's r, in this case) percentiles (by default 16%, 50%, and 84%)
* numpy array with the p-value percentiles (by default 16%, 50%, and 84%)

The percentile ranges can be adjusted using the `percentiles` keyword argument.

Additionally, if the full posterior distribution is desired, that can be obtained by setting the `return_dist` keyword argument to `True`.
In that case, `res` becomes a tuple of length four:

* numpy array with the correlation coefficient (Pearson's r, in this case) percentiles (by default 16%, 50%, and 84%)
* numpy array with the p-value percentiles (by default 16%, 50%, and 84%)
* numpy array with full set of correlation coefficient values from the bootstrapping
* numpy array with the full set of p-values computed from the bootstrapping

Please see the docstring for the full set of arguments and information including measurement uncertainties (necessary for point perturbation) and for marking censored data.

## Citing

If you use this script as part of your research, I encourage you to cite the following papers:

* [Curran 2014](https://arxiv.org/abs/1411.3816): Describes the technique and application to Spearman's rank correlation coefficient
* [Privon+ 2020](https://ui.adsabs.harvard.edu/abs/2020ApJ...893..149P/abstract): First use of this software, as `pymcspearman`.

Please also cite [scipy](https://scipy.org/citing-scipy/) and [numpy](https://numpy.org/citing-numpy/).


If your work uses Kendall's tau with censored data please also cite:

* [Isobe+ 1986](https://ui.adsabs.harvard.edu/abs/1986ApJ...306..490I/abstract): Censoring of data when computing Kendall's rank correlation coefficient.
