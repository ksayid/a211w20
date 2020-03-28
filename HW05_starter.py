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
