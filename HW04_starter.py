import numpy as np
from codes.cosmology import d_L_vectorized
from codes.interpolation import polyfit2d

def xtr(ntr, method, start = 0, stop = 1):
    '''
    Input number of training points: ntr
          method of spacing nodes: method
          on interval [start, stop]
    Return nodes spaced as specified.
    '''
    Om0tr_min, Om0tr_max = start, stop

    Om0tr, Omltr = [], []
    
    if method == 'chebyshev1':
        Om0tr = chebyshev_nodes1(Om0tr_min, Om0tr_max, ntr - 1)[::-1]
    elif method == 'chebyshev2':
        Om0tr = chebyshev_nodes2(Om0tr_min, Om0tr_max, ntr - 1)[::-1]
    elif method == 'even':
        Om0tr = np.linspace(Om0tr_min, Om0tr_max, ntr)
    else:
        raise ValueError('Error: Spacing method invalid')

    return Om0tr

def chebyshev_nodes1(a, b, N):
    assert(b>a)
    return a + 0.5*(b-a)*(1. + np.cos((2.*np.arange(N+1)+1)*np.pi/(2.*(N+1))))

def chebyshev_nodes2(a, b, N):
    assert(b>a)
    return a + 0.5*(b-a)*(1. + np.cos(np.arange(N+1)*np.pi/N))

def dL_poly_approx(z, ntr = 13, method = 'chebyshev1', poly_order = 13, H0 = 70):
    # Input z, number of training points, and method of
    # spacing nodes in order to train the polynomial.
    # We default to 13th order polynomial w/ chebyshev1
    # nodes since we found them to meet the 1e-4 threshold.
    Om0tr, OmLtr = xtr(ntr, method), xtr(ntr, method)

    dLgrid = np.zeros((ntr, ntr))
    for i, om0 in enumerate(Om0tr):
        for j, omL in enumerate(OmLtr):
            dLgrid[i][j] =  d_L_vectorized(z, H0, om0, omL, 1e-2, 1e-2) * H0 / 2.99792458e5
    
    poly2D = polyfit2d(Om0tr, OmLtr, dLgrid, kx = poly_order, \
                       ky = poly_order , order = None)
    
    return poly2D


