"""
pyMCSpearman.py

Python implementation of Curran (2014) method for measuring correlation with
errors.
"""

import numpy as np
from scipy.stats import spearmanr


def pyMCSpearman(x, y, dx=None, dy=None, Nboot=10000, Nperturb=10000,
                 bootstrap=True,
                 perturb=True,
                 percentiles=[16,50,80], return_dist=False):
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
            members = np.random.randint(0, high=Nvalues-1, size=Nvalues)
            xp = x[members]
            yp = y[members]
            if perturb:
                xp += np.random.normal(size=Nvalues) * dx[members]
                yp += np.random.normal(size=Nvalues) * dy[members]

            trho, tpval = spearmanr(xp, yp)

            rho.append(trho)
            pval.append(tpval)
    elif perturb:
        for i in range(Nperturb):
            xp = x + np.random.normal(size=Nvalues) * dx
            yp = y + np.random.normal(size=Nvalues) * dy

            trho, tpval = spearmanr(xp, yp)

            rho.append(trho)
            pval.append(tpval)
    else:
        import warnings
        warnings.warn("No bootstrapping or perturbation applied. Returning normal spearman rank values.")
        return spearmanr(x,y)

    frho = np.percentile(rho, percentiles)
    fpval = np.percentile(pval, percentiles)

    if return_dist:
        return frho, fpval, rho, pval
    else:
        return frho, fpval


def main():
    """
    run tests
    """

    import os

    # load test data
    data = np.genfromtxt(os.environ['HOME'] + '/astro/software/MCSpearman/test.data',
                         usecols=(0, 1, 2, 3),
                         dtype=[('x', float),
                                ('dx', float),
                                ('y', float),
                                ('dy', float)])  

    # spearman only
    res = pyMCSpearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       bootstrap=False,
                       perturb=False,
                       return_dist=True)
    print("Spearman only: ")
    print("\trho=" , res[0])
    print("\tpval=" , res[1])

    # bootstrap only
    res = pyMCSpearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       bootstrap=True,
                       perturb=False,
                       return_dist=True)
    print("Bootstrap only: ")
    print("\trho=" , np.mean(res[2]), "+/-", np.std(res[2]))
    print("\tpval=" , np.mean(res[3]), "+/-", np.std(res[3]))

    # perturbation only
    res = pyMCSpearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       bootstrap=False,
                       perturb=True,
                       return_dist=True)
    print("Perturbation only: ")
    print("\trho=" , np.mean(res[2]), "+/-", np.std(res[2]))
    print("\tpval=" , np.mean(res[3]), "+/-", np.std(res[3]))

    # composite method
    res = pyMCSpearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       bootstrap=True,
                       perturb=True,
                       return_dist=True)
    print("Bootstrap & Perturbation: ")
    print("\trho=" , np.mean(res[2]), "+/-", np.std(res[2]))
    print("\tpval=" , np.mean(res[3]), "+/-", np.std(res[3]))


if __name__ == "__main__":
    import sys
    sys.stdout.write("\nModule run as a program. Running test suite.\n\n")
    main()
