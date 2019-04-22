"""
pymcspearman.py

Python implementation of Curran (2014) method for measuring correlation with
errors.
"""

import numpy as _np
from scipy.stats import spearmanr as _spearmanr


def pymcspearman(x, y, dx=None, dy=None, Nboot=10000, Nperturb=10000,
                 bootstrap=True,
                 perturb=True,
                 percentiles=(16, 50, 80), return_dist=False):
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
        for i in range(Nboot):
            members = _np.random.randint(0, high=Nvalues-1, size=Nvalues)
            xp = x[members]
            yp = y[members]
            if perturb:
                xp += _np.random.normal(size=Nvalues) * dx[members]
                yp += _np.random.normal(size=Nvalues) * dy[members]

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
        import warnings
        warnings.warn("No bootstrapping or perturbation applied. Returning \
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
        sys.stdout.write("Passed spearman check.\n")
    except AssertionError:
        sys.stderr.write("Spearman comparison failed.\n")

    # bootstrap only
    res = pymcspearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       bootstrap=True,
                       perturb=False,
                       return_dist=True)
    try:
        assert _np.isclose(MCSres[1][0], _np.mean(res[2]),
                           atol=MCSres[1][1])
        sys.stdout.write("Passed bootstrap only method check.\n")
    except AssertionError:
        sys.stderr.write("Bootstrap only method comparison failed.\n")

    # perturbation only
    res = pymcspearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       bootstrap=False,
                       perturb=True,
                       return_dist=True)
    try:
        assert _np.isclose(MCSres[2][0], _np.mean(res[2]),
                           atol=MCSres[2][1])
        sys.stdout.write("Passed perturbation only method check.\n")
    except AssertionError:
        sys.stderr.write("Perturbation only method comparison failed.\n")

    # composite method
    res = pymcspearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       bootstrap=True,
                       perturb=True,
                       return_dist=True)
    try:
        assert _np.isclose(MCSres[3][0], _np.mean(res[2]),
                           atol=MCSres[3][1])
        sys.stdout.write("Passed composite method check.\n")
    except AssertionError:
        sys.stderr.write("Composite method comparison failed.\n")


def main():
    """
    run tests
    """

    run_tests()


if __name__ == "__main__":
    import sys
    sys.stdout.write("\nModule run as a program. Running test suite.\n\n")
    main()
