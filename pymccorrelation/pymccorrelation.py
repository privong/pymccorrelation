"""
pymccorrelation.py

Python implementation of Curran (2014) method for calculating Spearman's
rank correlation coefficient with uncertainties. Extended to also calculate
Kendall's Tau.

Kendall tau implementation follow Isobe, Feigelson & Nelson (1986) method for
calculating the correlation coefficient with uncertainties on censored data
(upper/lowerlimit).

Copyright 2019-2020 George C. Privon, Yiqing Song

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
import scipy.stats as _st
from scipy.stats import spearmanr as _spearmanr


def perturb_values(x, y, dx, dy, Nperturb=10000):
    """
    For input points (x, y) with errors (dx, dy) return Nperturb sets of
    values draw from Gaussian distributions centered at x+-dx and y+-dy.
    """

    assert len(x) == len(y)
    assert len(dx) == len(dy)
    assert len(x) == len(dx)

    Nvalues = len(x)

    xp = _np.random.normal(loc=x,
                           scale=dx,
                           size=(Nperturb, Nvalues))
    yp = _np.random.normal(loc=y,
                           scale=dy,
                           size=(Nperturb, Nvalues))

    if Nperturb == 1:
        xp = xp.flatten()
        yp = yp.flatten()

    return xp, yp


def kendall(x, y,
            xlim=None, ylim=None):
    """
    Kendall tau wrapper function to determine if we need to handle censoring.
    If there is censoring, hand it off to the IFN 1986 generalized function.
    """

    if xlim is None and ylim is None:
        from scipy.stats import kendalltau
        return kendalltau(x, y)

    return kendall_IFN86(x, y, xlim, ylim)


def kendall_IFN86(x, y,
                  xlim, ylim):
    """
    Generalized kendall tau test described in Isobe, Feigelson & Nelson 1986
    ApJ 306, 490-507.

    Parameters:
        x: independent variable
        y: dependent variable
        xlim/ylim: censoring information for the variables. Values of
            (-1, 1, 0) correspond to (lower limit, upper limit, detection)
    Note that both x and y can be censored.
    """

    #TODO: vectorize this function, very slow

    # the argument variable should have the same length
    assert len(x) == len(y)
    assert len(xlim) == len(ylim)
    assert len(x) == len(xlim)

    num = len(x)
    #set up pair counters
    a = _np.zeros((num, num))
    b = _np.zeros((num, num))
    
    for i in range(num):
        for j in range(num):
            if x[i] == x[j]:
                a[i, j] = 0
            elif x[i] > x[j]: #if x[i] is definitely > x[j]
                if (xlim[i] == 0 or xlim[i] == -1) and (xlim[j] == 0 or xlim[j] == 1):
                    a[i, j] = -1
            else: #if x[i] is definitely < x[j], all other uncertain cases have aij=0
                if (xlim[i] == 0 or xlim[i] == 1) and (xlim[j] == 0 or xlim[j] == -1):
                    a[i, j] = 1
            
    for i in range(num):
        for j in range(num):
            if y[i] == y[j]:
                b[i, j] = 0
            elif y[i] > y[j]:
                if (ylim[i] == 0 or ylim[i] == -1) and (ylim[j] == 0 or ylim[j] == 1):
                    b[i, j] = -1
            else:
                 if (ylim[i] == 0 or ylim[i] == 0) and (ylim[j] == 0 or ylim[j] == -1):
                    b[i, j] = 1
                    
            
    S = _np.sum(a * b)
    var = (4 / (num * (num - 1) * (num - 2))) * \
          (_np.sum(a * _np.sum(a, axis=1, keepdims=True)) - _np.sum(a * a)) * \
          (_np.sum( b * _np.sum(b, axis=1, keepdims=True)) - _np.sum(b * b)) + \
          (2 / (num * (num - 1))) * \
          _np.sum(a * a) * _np.sum(b * b)
    z = S/ _np.sqrt(var)
    tau = z * _np.sqrt(2 * (2 * num + 5)) / (3 * _np.sqrt(num * (num - 1)))
    pval = _st.norm.sf(abs(z)) * 2
    return tau, pval


def pymccorrelation(x, y,
                    dx=None, dy=None,
                    xlim=None, ylim=None,
                    Nboot=None,
                    Nperturb=None,
                    coeff=None,
                    percentiles=(16, 50, 84),
                    return_dist=False):
    """
    Compute spearman rank coefficient with uncertainties using several methods.
    Arguments:
    x: independent variable array
    y: dependent variable array
    dx: uncertainties on independent variable (assumed to be normal)
    dy: uncertainties on dependent variable (assumed to be normal)
    xlim: censoring information for independent variable to compute
        generalized Kendall tau
        (-1, 1, 0) correspond to (lower limit, upper limit, detection)
    ylim: censoring information for dependent variable to compute generalized
        Kendall tau
        (-1, 1, 0) correspond to (lower limit, upper limit, detection)
    Nboot: number of times to bootstrap (does not boostrap if =None)
    Nperturb: number of times to perturb (does not perturb if =None)
    coeff: Correlation coefficient to compute. Must be one of:
        ['spearmanr', 'kendallt']
    percentiles: list of percentiles to compute from final distribution
    return_dist: if True, return the full distribution of rho and p-value
    """

    # do some checks on input array lengths and ensure the necessary data
    # is provided
    if Nperturb is not None and dx is None and dy is None:
        raise ValueError("dx or dy must be provided if perturbation is to be used.")
    if len(x) != len(y):
        raise ValueError("x and y must be the same length.")
    if dx is not None and len(dx) != len(x):
        raise ValueError("dx and x must be the same length.")
    if dy is not None and len(dy) != len(y):
        raise ValueError("dx and x must be the same length.")

    # TODO: add checks for censoring when not doing kendall tau
    coeffs_impl = ['spearmanr', 'kendallt']
    # make sure an implemented correlation coefficient type is requested
    if coeff not in coeffs_impl:
        raise ValueError("coeff must be one of " + ', '.join(coeffs_impl))

    # censoring is only implemented for kendall's tau, return an error
    # if censored data is provided
    if coeff != 'kendallt' and \
       ((xlim is not None or ylim is not None) or
        (_np.all(xlim == 0) and _np.all(ylim == 0))):
        raise ValueError('Censored data provided, but ' + coeff + ' does not \
support censored data.')

    Nvalues = len(x)

    # if no bootstrapping or correlation is requested, we can just
    # report the normal correlation coefficient values
    if Nboot is None and Nperturb is None:
        import warnings as _warnings
        _warnings.warn("No bootstrapping or perturbation applied. Returning \
normal " + coeff + " output.")
        if coeff == 'spearmanr':
            return _spearmanr(x, y)
        elif coeff == 'kendallt':
            # pass along the xlim/ylim arrays, and the wrapper will handle
            # the presence of censored data
            return kendall(x, y, xlim=xlim, ylim=ylim)
        #elif coeff == 'pearsonr':

    # if perturbing points, and we have censored data, set up an index

    if Nperturb is not None and (xlim is not None and ylim is not None):
        do_per = np.logical_and(xlim == 0,
                                ylim == 0)
    else:
        do_per = np.ones(len(x),
                         dtype=bool)

    if Nboot is not None:
        coeff = _np.zeros(Nboot)
        pval = _np.zeros(Nboot)
        # generate all the needed bootstrapping indices
        members = _np.random.randint(0, high=Nvalues-1,
                                     size=(Nboot, Nvalues))
        # loop over sets of bootstrapping indices and compute
        # correlation coefficient
        for i in range(Nboot):
            xp = x[members[i, :]]
            yp = y[members[i, :]]
            if Nperturb is not None:
                # perform 1 perturbation on top of the bootstrapping
                xp[do_per], yp[do_per] = perturb_values(x[members[i, :]][do_per],
                                                        y[members[i, :]][do_per],
                                                        dx[members[i, :]][do_per],
                                                        dy[members[i, :]][do_per],
                                                        Nperturb=1)

            coeff[i], pval[i] = _spearmanr(xp, yp)

    elif Nperturb is not None:
        coeff = _np.zeros(Nperturb)
        pval = _np.zeros(Nperturb)
        # generate Nperturb perturbed copies of the dataset
        xp[do_per], yp[do_per] = perturb_values(x[do_per],
                                                y[do_per],
                                                dx[do_per],
                                                dy[do_per],
                                                Nperturb=Nperturb)
        # loop over each perturbed copy and compute the correlation
        # coefficient
        for i in range(Nperturb):
            coeff[i], pval[i]= _spearmanr(xp[i, :], yp[i, :])
    else:
        import warnings as _warnings
        _warnings.warn("No bootstrapping or perturbation applied. Returning \
normal spearman rank values.")
        return _spearmanr(x, y)

    fcoeff = _np.percentile(coeff, percentiles)
    fpval = _np.percentile(pval, percentiles)

    if return_dist:
        return fcoeff, fpval, coeff, pval
    return fcoeff, fpval


def pymcspearman(x, y, dx=None, dy=None,
                 Nboot=None,
                 Nperturb=None,
                 percentiles=(16, 50, 84), return_dist=False):
    """
    Compute spearman rank coefficient with uncertainties using several methods.
    Arguments:
    x: independent variable array
    y: dependent variable array
    dx: uncertainties on independent variable (assumed to be normal)
    dy: uncertainties on dependent variable (assumed to be normal)
    Nboot: number of times to bootstrap (does not boostrap if =None)
    Nperturb: number of times to perturb (does not perturb if =None)
    percentiles: list of percentiles to compute from final distribution
    return_dist: if True, return the full distribution of rho and p-value
    """

    if Nperturb is not None and dx is None and dy is None:
        raise ValueError("dx or dy must be provided if perturbation is to be used.")
    if len(x) != len(y):
        raise ValueError("x and y must be the same length.")
    if dx is not None and len(dx) != len(x):
        raise ValueError("dx and x must be the same length.")
    if dy is not None and len(dy) != len(y):
        raise ValueError("dx and x must be the same length.")

    rho = []
    rho = []
    pval = []

    Nvalues = len(x)

    if Nboot is not None:
        # generate all the needed bootstrapping indices
        members = _np.random.randint(0, high=Nvalues-1,
                                     size=(Nboot, Nvalues))
        # loop over sets of bootstrapping indices and compute
        # correlation coefficient
        for i in range(Nboot):
            xp = x[members[i, :]]
            yp = y[members[i, :]]
            if Nperturb is not None:
                # return only 1 perturbation on top of the bootstrapping
                xp, yp = perturb_values(x[members[i, :]], y[members[i, :]],
                                        dx[members[i, :]], dy[members[i, :]],
                                        Nperturb=1)

            trho, tpval = _spearmanr(xp, yp)

            rho.append(trho)
            pval.append(tpval)
    elif Nperturb is not None:
        # generate Nperturb perturbed copies of the dataset
        xp, yp = perturb_values(x, y, dx, dy, Nperturb=Nperturb)
        # loop over each perturbed copy and compute the correlation
        # coefficient
        for i in range(Nperturb):
            trho, tpval = _spearmanr(xp[i, :], yp[i, :])

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


def pymckendall(x, y, xlim, ylim, dx=None, dy=None,
                Nboot=None,
                Nperturb=None,
                percentiles=(16,50,84), return_dist=False):
    """
    Compute Kendall tau coefficient with uncertainties using several methods.
    Arguments:
    x: independent variable array
    y: dependent variable array
    xlim: array indicating if x is upperlimit (1), lowerlimit(-1), or detection(0)
    ylim: array indicating if x is upperlimit (1), lowerlimit(-1), or detection(0)
    dx: uncertainties on independent variable (assumed to be normal)
    dy: uncertainties on dependent variable (assumed to be normal)
    Nboot: number of times to bootstrap (does not boostrap if =None)
    Nperturb: number of times to perturb (does not perturb if =None)
    percentiles: list of percentiles to compute from final distribution
    return_dist: if True, return the full distribution of rho and p-value
    """

    if Nperturb is not None and dx is None and dy is None:
        raise ValueError("dx or dy must be provided if perturbation is to be used.")
    if len(x) != len(y):
        raise ValueError("x and y must be the same length.")
    if dx is not None and len(dx) != len(x):
        raise ValueError("dx and x must be the same length.")
    if dy is not None and len(dy) != len(y):
        raise ValueError("dx and x must be the same length.")

    Nvalues = len(x)

    if Nboot is not None:
        tau = _np.zeros(Nboot)
        pval = _np.zeros(Nboot)
        members = _np.random.randint(0, high=Nvalues-1, size=(Nboot,Nvalues)) #randomly resample
        xp = x[members]
        yp = y[members]
        xplim = xlim[members] #get lim indicators for resampled x, y
        yplim = ylim[members]
        if Nperturb is not None:
            xp[xplim==0] += _np.random.normal(size=_np.shape(xp[xplim==0])) * dx[members][xplim==0] #only perturb the detections
            yp[yplim==0] += _np.random.normal(size=_np.shape(yp[yplim==0])) * dy[members][yplim==0] #only perturb the detections
        
        #calculate tau and pval for each iteration
        for i in range(Nboot):
            tau[i], pval[i] = kendall(xp[i, :], yp[i, :],
                                      xplim[i, :], yplim[i, :])
       
    elif Nperturb is not None:
        tau = _np.zeros(Nperturb)
        pval = _np.zeros(Nperturb)
        yp = [y] * Nperturb + _np.random.normal(size=(Nperturb, Nvalues)) * dy #perturb all data first
        xp = [x] * Nperturb + _np.random.normal(size=(Nperturb, Nvalues)) * dx
        yp[:, ylim!=0] = y[ylim!=0] #set upperlimits and lowerlimits to be unperturbed
        xp[:, xlim!=0] = x[xlim!=0] #so only real detections are perturbed
            
        for i in range(Nperturb):
            tau[i], pval[i] = kendall(xp[i, :], yp[i, :],
                                      xlim, ylim)

    else:
        import warnings as _warnings
        _warnings.warn("No bootstrapping or perturbation applied. Returning \
normal generalized kendall tau.")
        tau, pval = kendall(x, y, xlim, ylim)
        return tau, pval

    ftau = _np.nanpercentile(tau, percentiles)
    fpval = _np.nanpercentile(pval, percentiles)

    if return_dist:
        return ftau, fpval, tau, pval
    return ftau, fpval


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
                       Nboot=None,
                       Nperturb=None,
                       return_dist=True)
    try:
        assert _np.isclose(MCSres[0][0], res[0],
                           atol=MCSres[0][1])
        _sys.stdout.write("Passed spearman check.\n")
    except AssertionError:
        _sys.stderr.write("Spearman comparison failed.\n")

    # bootstrap only
    res = pymcspearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       Nboot=10000,
                       Nperturb=None,
                       return_dist=True)
    try:
        assert _np.isclose(MCSres[1][0], _np.mean(res[2]),
                           atol=MCSres[1][1])
        _sys.stdout.write("Passed bootstrap only method check.\n")
    except AssertionError:
        _sys.stderr.write("Bootstrap only method comparison failed.\n")

    # perturbation only
    res = pymcspearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       Nboot=None,
                       Nperturb=10000,
                       return_dist=True)
    try:
        assert _np.isclose(MCSres[2][0], _np.mean(res[2]),
                           atol=MCSres[2][1])
        _sys.stdout.write("Passed perturbation only method check.\n")
    except AssertionError:
        _sys.stderr.write("Perturbation only method comparison failed.\n")

    # composite method
    res = pymcspearman(data['x'], data['y'], dx=data['dx'], dy=data['dy'],
                       Nboot=10000,
                       Nperturb=10000,
                       return_dist=True)
    try:
        assert _np.isclose(MCSres[3][0], _np.mean(res[2]),
                           atol=MCSres[3][1])
        _sys.stdout.write("Passed composite method check.\n")
    except AssertionError:
        _sys.stderr.write("Composite method comparison failed.\n")

    # test Kendall tau IFN86 for consistency with scipy
    from scipy.stats import kendalltau
    sres = kendalltau(data['x'], data['y'])
    IFN86res = kendall_IFN86(data['x'], data['y'],
                             xlim=_np.zeros(len(data)),
                             ylim=_np.zeros(len(data)))
    try:
        assert _np.isclose(sres[0], IFN86res[0])
        assert _np.isclose(sres[1], IFN86res[1])
        _sys.stdout.write("Passed Kendall tau comparison.\n")
    except AssertionError:
        _sys.stderr.write("Kendall tau comparison with scipy failed.\n")


def main():
    """
    run tests
    """

    run_tests()


if __name__ == "__main__":
    import sys as _sys
    _sys.stdout.write("\nModule run as a program. Running test suite.\n\n")
    main()
