"""
pymckendall.py

Python implementation of Isobe, Feigelson & Nelson (1986) method for calculating Kendall
tau rank correlation coefficient with uncertainties on censored data (upper/lowerlimit).

"""

import numpy as np
import scipy.stats as st

#generalized kendall tau test described in Isobe,Feigelson & Nelson 1986
#need to come up some way to vectorize this function, very slow
def kendall(x,y,xlim,ylim):
    #x, y are two arrays, either may contain censored data
    #xlim, ylim are arrays indicating if x or y are lower or upperlimits,-1--lowerlimit,+1--upperlimit,0--detection
    num=len(x)#x,y should have same length
    #set up pair counters
    a=np.zeros((num,num))
    b=np.zeros((num,num))
    
    for i in range(num):
        for j in range(num):
            if x[i]==x[j]:
                a[i,j]=0
            elif x[i] > x[j]: #if x[i] is definitely > x[j]
                if (xlim[i]==0 or xlim[i]==-1) and (xlim[j]==0 or xlim[j]==1):
                    a[i,j]=-1
            else: #if x[i] is definitely < x[j], all other uncertain cases have aij=0
                if (xlim[i]==0 or xlim[i]==1) and (xlim[j]==0 or xlim[j]==-1):
                    a[i,j]=1
            
    for i in range(num):
        for j in range(num):
            if y[i]==y[j]:
                b[i,j]=0
            elif y[i] > y[j]:
                if (ylim[i]==0 or ylim[i]==-1) and (ylim[j]==0 or ylim[j]==1):
                    b[i,j]=-1
            else:
                 if (ylim[i]==0 or ylim[i]==0) and (ylim[j]==0 or ylim[j]==-1):
                    b[i,j]=1
                    
            
    S = np.sum(a*b)
    var = (4/(num*(num-1)*(num-2)))*(np.sum(a*np.sum(a,axis=1,keepdims=True))-np.sum(a*a))\
    *(np.sum(b*np.sum(b,axis=1,keepdims=True))-np.sum(b*b))+(2/(num*(num-1)))*np.sum(a*a)*np.sum(b*b)
    z=S/np.sqrt(var)
    tau=z*np.sqrt(2*(2*num+5))/(3*np.sqrt(num*(num-1)))
    pval=st.norm.sf(abs(z))*2
    return tau,pval


def pymckendall(x, y, xlim, ylim, dx=None, dy=None, Nboot=10000, Nperturb=10000,
    bootstrap=True,
    perturb=True,
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

    Nvalues = len(x)

    if bootstrap:
        tau = np.zeros(Nboot)
        pval = np.zeros(Nboot)
        members = np.random.randint(0, high=Nvalues-1, size=(Nboot,Nvalues)) #randomly resample
        xp = x[members]
        yp = y[members]
        xplim = xlim[members] #get lim indicators for resampled x, y
        yplim = ylim[members]
        if perturb:
            xp[xplim==0] += np.random.normal(size=np.shape(xp[xplim==0])) * dx[members][xplim==0] #only perturb the detections
            yp[yplim==0] += np.random.normal(size=np.shape(yp[yplim==0])) * dy[members][yplim==0] #only perturb the detections
        
        #calculate tau and pval for each iteration
        for i in range(Nboot):
            tau[i],pval[i] = kendall(xp[i,:], yp[i,:], xplim[i,:], yplim[i,:])
       
    elif perturb:
        tau = np.zeros(Nperturb)
        pval = np.zeros(Nperturb)
        yp=[y]*Nperturb+np.random.normal(size=(Nperturb,Nvalues))*dy #perturb all data first
        xp=[x]*Nperturb+np.random.normal(size=(Nperturb,Nvalues))*dx
        yp[:,ylim!=0]=y[ylim!=0] #set upperlimits and lowerlimits to be unperturbed
        xp[:,xlim!=0]=x[xlim!=0] #so only real detections are perturbed
            
        for i in range(Nperturb):
            tau[i], pval[i] = kendall(xp[i,:], yp[i,:], xlim, ylim) #need to vectorize!

    else:
        import warnings as _warnings
        _warnings.warn("No bootstrapping or perturbation applied. Returning \
    normal generalized kendall tau.")
        tau,pval=kendall(x, y, xlim, ylim)
        return tau,pval

    ftau = np.nanpercentile(tau, percentiles)
    fpval = np.nanpercentile(pval, percentiles)

    if return_dist:
        return ftau, fpval, tau, pval
    return ftau, fpval

