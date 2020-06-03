"""
pymcspearman.py

Python implementation of Curran (2014) method for calculating Spearman's
rank correlation coefficient with uncertainties.

Copyright 2019-2020 George C. Privon

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

import numpy as _np
from scipy.stats import spearmanr as _spearmanr


def pymcspearman(x, y, dx=None, dy=None, Nboot=10000, Nperturb=10000,
                 bootstrap=True,
                 perturb=True,
                 percentiles=(16, 50, 84), return_dist=False):
    """
    Compute spearman rank coefficient with uncertainties using several methods.
    Arguments:
    x: independent variable array
    y: dependent variable array
    dx: uncertainties on independent variable (assumed to be normal)
    dy: uncertainties on dependent variable (assumed to be normal)
    Nboot: number of times to bootstrap (if bootstrap=True)
    Nperturb: number of times to perturb (if perturb=True)
    bootstrap: whether to include bootstrapping. True/False
    perturb: whether to include perturbation. True/False
    percentiles: list of percentiles to compute from final distribution
    return_dist: if True, return the full distribution of rho and p-value
    """

    if perturb and dx is None and dy is None:
        raise ValueError("dx or dy must be provided if perturbation is to be used.")
    if len(x) != len(y):
        raise ValueError("x and y must be the same length.")
    if dx is not None and len(dx) != len(x):
        raise ValueError("dx and x must be the same length.")
    if dy is not None and len(dy) != len(y):
        raise ValueError("dx and x must be the same length.")

    rho = []
    pval = []

    Nvalues = len(x)

    if bootstrap:
        members = _np.random.randint(0, high=Nvalues-1,
                                     size=(Nboot, Nvalues))
        for i in range(Nboot):
            xp = x[members[i, :]]
            yp = y[members[i, :]]
            if perturb:
                xp += _np.random.normal(size=Nvalues) * dx[members[i, :]]
                yp += _np.random.normal(size=Nvalues) * dy[members[i, :]]

            trho, tpval = _spearmanr(xp, yp)

            rho.append(trho)
            pval.append(tpval)
    elif perturb:
        for i in range(Nperturb):
            xp = x + _np.random.normal(size=Nvalues) * dx
            yp = y + _np.random.normal(size=Nvalues) * dy

            trho, tpval = _spearmanr(xp, yp)

            rho.append(trho)
            pval.append(tpval)
    else:
        import warnings as _warnings
        _warnings.warn("No bootstrapping or perturbation applied. Returning \
normal spearman rank values.")
        return _spearmanr(x, y)

    frho = _np.percentile(rho, percentiles)
    fpval = _np.percentile(pval, percentiles)

    if return_dist:
        return frho, fpval, rho, pval
    return frho, fpval


def run_tests():
    """
    Test output of pymcspearman against tabulated values from MCSpearman
    """

    from tempfile import NamedTemporaryFile as ntf
    from urllib.request import urlretrieve

    # get test data
    tfile = ntf()
    urlretrieve("https://raw.githubusercontent.com/PACurran/MCSpearman/master/test.data",
                tfile.name)
    # open temporary file
    data = _np.genfromtxt(tfile,
                         usecols=(0, 1, 2, 3),
                         dtype=[('x', float),
                                ('dx', float),
                                ('y', float),
                                ('dy', float)])

    # tabulated results from a MCSpearman run with 10000 iterations
    MCSres = [(0.8308, 0.001),  # spearman only
              (0.8213, 0.0470), # bootstrap only
              (0.7764, 0.0356), # perturbation only
              (0.7654, 0.0584)] # bootstrapping and perturbation

    # spearman only
    res = pymcspearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       bootstrap=False,
                       perturb=False,
                       return_dist=True)
    try:
        assert _np.isclose(MCSres[0][0], res[0],
                           atol=MCSres[0][1])
        _sys.stdout.write("Passed spearman check.\n")
    except AssertionError:
        _sys.stderr.write("Spearman comparison failed.\n")

    # bootstrap only
    res = pymcspearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       bootstrap=True,
                       perturb=False,
                       return_dist=True)
    try:
        assert _np.isclose(MCSres[1][0], _np.mean(res[2]),
                           atol=MCSres[1][1])
        _sys.stdout.write("Passed bootstrap only method check.\n")
    except AssertionError:
        _sys.stderr.write("Bootstrap only method comparison failed.\n")

    # perturbation only
    res = pymcspearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       bootstrap=False,
                       perturb=True,
                       return_dist=True)
    try:
        assert _np.isclose(MCSres[2][0], _np.mean(res[2]),
                           atol=MCSres[2][1])
        _sys.stdout.write("Passed perturbation only method check.\n")
    except AssertionError:
        _sys.stderr.write("Perturbation only method comparison failed.\n")

    # composite method
    res = pymcspearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       bootstrap=True,
                       perturb=True,
                       return_dist=True)
    try:
        assert _np.isclose(MCSres[3][0], _np.mean(res[2]),
                           atol=MCSres[3][1])
        _sys.stdout.write("Passed composite method check.\n")
    except AssertionError:
        _sys.stderr.write("Composite method comparison failed.\n")


def main():
    """
    run tests
    """

    run_tests()


if __name__ == "__main__":
    import sys as _sys
    _sys.stdout.write("\nModule run as a program. Running test suite.\n\n")
    main()
