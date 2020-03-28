import numpy as np

clight = 2.99792458e5 # c in km/s
# function that works only for models with OmL = 0
def d_L_simple(z, H0, Om0):
    q0 = 0.5 * Om0
    q0z = q0 * z
    return clight * z/H0 * (1. + (z-q0z) / (1. + q0z + np.sqrt(1. + 2.*q0z)))  

def dl_func(z, H0, Om0, OmL, Omk):
    z1 = 1.0 + z; z12 = z1 * z1
    return 1.0 / np.sqrt(z12*(Om0*z1 + Omk) + OmL)

def trapzd(func, a, b, hstep, *args):
    """
    integration using trapezoidal scheme
    
    Parameters:
    -----------
    func: python function object - function to integrate
    a, b: floats - integration interval
    hstep: float - step size to use
    *args: pointer to argument list to pass to dl_func
    
    Returns:
    --------
    float - value of the integral estimated using trapezoidal integration    
    """
    bma = np.abs(b-a)
            
    nstep = np.int(bma / hstep) 
    if nstep == 1:
        return 0.5*(func(a, *args) + func(b, *args)) * hstep
    else: 
        xd = a + np.arange(nstep) * hstep # np.arange creates a vector of values from 0 to nstep-1
        return (0.5*(func(a, *args) + func(b, *args)) + np.sum(func(xd[1:], *args))) * hstep
