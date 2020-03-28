from codes.interpolation import *
from codes.constants import clight
import numpy as np

def d_L_tilda(z, Om0, OmL, atol=1.e-8, rtol=1.e-8):
    return d_L_vectorized(z, 1, Om0, OmL, atol = atol, rtol = rtol) / clight

def dL_poly_approx(z, ntr = 11, spacing='chebyshev1'):
    # train the polynomical
    # ntr=11 and py=px=11 minimizes node count to get desired accuracy
    Om0tr_min, Om0tr_max = 0., 1
    OmLtr_min, OmLtr_max = 0., 1
    
    Om0tr, Omltr = np.zeros(ntr+1),np.zeros(ntr+1)
    # Chebyshev type 1
    if spacing == 'chebyshev1':
        Om0tr = chebyshev_nodes1(Om0tr_min,Om0tr_max,ntr)[::-1]
        OmLtr = chebyshev_nodes1(OmLtr_min,OmLtr_max,ntr)[::-1]
    elif spacing == 'chebyshev2':
        Om0tr = chebyshev_nodes2(Om0tr_min,Om0tr_max,ntr)[::-1]
        OmLtr = chebyshev_nodes2(OmLtr_min,OmLtr_max,ntr)[::-1]
    elif spacing == 'even':
        Om0tr = np.linspace(Om0tr_min, Om0tr_max, ntr)
        OmLtr = np.linspace(OmLtr_min, OmLtr_max, ntr)
    else:
        raise ValueError('error dL_poly_approx_import: invalid spacing')
    
    dLgrid = np.zeros((ntr+1,ntr+1))
    for i, om0 in enumerate(Om0tr):
        for j, omL in enumerate(OmLtr):
            dLgrid[i][j] = d_L_tilda(z,om0,omL,1e-11,1e-11)
            
    poly2D = polyfit2d(Om0tr, OmLtr, dLgrid, kx=11, ky=11,order=None)
    
    return poly2D

def auto_corr_func(timeseries, lagmax):
    """
    compute auto correlation function
    """
    ts = np.asarray(timeseries)
    N = np.size(ts) - 1
    ts -= np.average(ts) # Set to mean 0
    corr_func = np.zeros(lagmax)
    for dt in range(lagmax):
        # sum of ts[t+dt]*ts[t]
        corr_func[dt] = (np.dot(timeseries[0:N-dt],timeseries[dt:N])) 
    if (corr_func[0]>0):
        corr_func /= corr_func[0] # normalize
    return corr_func

def compute_tcorr(timeseries,maxcorr):
    """
    compute auto-correlation time
    Parameters:
    -----------
    
    timeseries: 1d vector of values
    maxcorr: maximum autocorrelation lag to consider
    
    Returns:
    tau, mean, sigma: float scalars
        autocorrelation time, mean of the sequence and its rms 
    """
    timeseries = np.copy(timeseries)
    mean = np.average(timeseries)
    corrfxn = auto_corr_func(timeseries,maxcorr)
    tau = np.sum(corrfxn)-1
    var = np.var(timeseries)
    sigma = np.sqrt(var * tau / len(timeseries))
    return tau, mean, sigma

def minimize_by_differential_evolution(func, x0, atol=1.e-6, s=1.0, sigs=0.1, bounds=None, max_iter=250):
    """
    Parameters:
    ------------
    func - Python function object
           function to minimize, should expect x0 as a parameter vector
    x0   - vector of real numbers of shape (npop, nd), 
            where npop is population size and nd is the number of func parameters
    atol - real
            absolute tolerance threshold for change of population member positions
    s    - real 
            mean of the scaling parameter s
    sigs - real 
            rms dispersion of s for drawing Gaussian random numbers center on s
    bounds - array of tuples 
            bounds for the minimization exploration; define the region in which to search for the minimum
    """
    
    xnow = np.copy(x0)
    npop = np.shape(xnow)[0] #num of population member
    
    fnow = np.zeros((npop))
    for i in range(npop):
        fnow[i] = func(xnow[i, :])
        
    xnext = np.zeros_like(xnow)

    #Keeping track of our best result
    ndim = np.shape(xnow)[1]
    best_rslt = np.repeat(6, ndim)
    
    counter = 0
    while (abs_err(func, xnow, xnext) > atol):
        for i, xi in enumerate(xnow):
            r1, r2, r3 = rand_indicies(xnow, i)
            s_rand = np.random.normal(loc=s)
            xtry = r3 + s_rand *(r1 - r2)

            if (func(xtry) <= fnow[i]):
                xnext[i] = xtry
            else:
                xnext[i] = xnow[i]
            
            #updating our best result
            if func(xtry) < func(best_rslt):
                best_rslt = xtry
                
        xnow_temp = np.copy(xnow)
        xnow = np.copy(xnext)

        for i in range(npop):
            fnow[i]=func(xnow[i,:])
        xnext = xnow_temp #do this so we can properly calculate the absolute error
        
        counter += 1
        #make sure we're not over the max_iter
        if counter > max_iter:
            break
    return best_rslt