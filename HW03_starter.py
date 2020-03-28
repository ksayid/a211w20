from codes.cosmology import d_L_vectorized
from codes.interpolation import chebyshev_nodes1, chebyshev_nodes2, polyfit2d
import numpy as np


# The functions below are heavily based on functions made in
# homework 2. I restructured them a little for ease. 

def xtr(ntr, method):
    '''
    Input number of training points: ntr
          method of spacing nodes: method
    Output nodes for Om0tr, OmLtr. 
    '''
    Om0tr_min, Om0tr_max = 0, 1
    OmLtr_min, OmLtr_max = 0, 1

    Om0tr, Omltr = [], []
    
    if method == 'chebyshev1':
        Om0tr = chebyshev_nodes1(Om0tr_min, Om0tr_max, ntr - 1)[::-1]
    elif method == 'chebyshev2':
        Om0tr = chebyshev_nodes2(Om0tr_min, Om0tr_max, ntr - 1)[::-1]
    elif method == 'even':
        Om0tr = np.linspace(Om0tr_min, Om0tr_max, ntr)
    else:
        raise ValueError('Error: Spacing method invalid')

    return Om0tr, Om0tr

def dL_poly_approx(z, ntr = 13, method = 'chebyshev1', poly_order = 13, H0 = 70):
    # Input z, number of training points, and method of
    # spacing nodes in order to train the polynomial.
    # We default to 13th order polynomial w/ chebyshev1
    # nodes since we found them to meet the 1e-4 threshold.
    Om0tr, OmLtr = xtr(ntr, method)

    dLgrid = np.zeros((ntr, ntr))
    for i, om0 in enumerate(Om0tr):
        for j, omL in enumerate(OmLtr):
            dLgrid[i][j] =  d_L_vectorized(z, H0, om0, omL, 1e-4, 1e-4) * H0 / 2.99792458e5
    
    poly2D = polyfit2d(Om0tr, OmLtr, dLgrid, kx = poly_order, \
                       ky = poly_order , order = None)
    
    return poly2D



