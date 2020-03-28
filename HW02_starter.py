import numpy as np


class Cubic_Spline():
    def __init__(self, xi, fi):
        """
        (xi, fi) are training data
        """
        self.xi = xi
        self.fi = fi
        self.a, self.b, self.c, self.d = self.cubic_spline_coefficients()

    def cubic_spline_coefficients(self):
        """
        compute coefficients of the interpolating natural cubic spline
        see Appendix in the note for the derivation and details of the algorithm
        
        Parameters: 
            xi, fi: numpy float vectors
                    tabulated points and function values
                
        Returns:
            a, b, c, d: numpy float vectors
                    cubic spline coefficients 
                    dx = x - xi[i]
                    fx = a[i] + dx*(b[i] + c[i]*dx + d[i]*dx*dx)

        """

        delx = np.diff(self.xi); delf = np.diff(self.fi)
        # form matrices to solve for spline coefficients
        vx = np.zeros_like(self.xi)
        # form rhs vector using python's array slicing 
        vx[1:-1:] = 3.*(delf[1::]/delx[1::] - delf[:-1:]/delx[:-1:])
        # construct 3 diagonals
        nx = np.size(self.xi)
        diags = np.zeros((3, nx))
        diags[1,0] = 1.; diags[1,-1] = 1.
        diags[1,1:-1:] = 2. * (delx[1::] + delx[:-1:])
        diags[0,1:] = delx[:]
        diags[2,1:-1] = delx[1:]
        # solve for coefficients c using Thomas algorithm for tri-diagonal matrices
        # see https://www.cfd-online.com/Wiki/Tridiagonal_matrix_algorithm_-_TDMA_(Thomas_algorithm)
        ac, bc, cc, dc = map(np.array, (diags[0,:], diags[1,:], diags[2,:], vx)) # copy arrays

        '''Note: This loop can be improved with NumPy'''
        for k in range(1, nx):
            mk = ac[k] / bc[k-1]
            bc[k] = bc[k] - mk * cc[k-1] 
            dc[k] = dc[k] - mk * dc[k-1]
        
        c = np.zeros_like(bc)
        c[-1] = dc[-1] / bc[-1]

        '''Note: This loop can be improved with NumPy'''
        for k in range(nx-2, -1, -1):
            c[k] = (dc[k]-cc[k]*c[k+1])/bc[k]

        # now get the rest of the coefficients
        b = delf[::]/delx[::] - (c[1::] + 2.*c[:-1:])*delx[::]/3.
        d = (c[1::] - c[:-1:])/(3.*delx[::]) 
        a = self.fi
        return a, b, c, d

    def cubic_spline(self, xt):
        """
        piecewise linear approximation of f(x) given input of tabulated values of xi and fi
        note that xi are expected in ascending order
        
        Returns:
            vector of spline values at test points x 

        """

        '''Note: This loop can be improved with NumPy. 
                 See derivativve function for how to do so.'''
        n = np.size(self.xi) - 1
        fxt = np.empty_like(xt)
        for j, x in enumerate(xt):
            for i in range(n):
                if (x >= self.xi[i]) and (x <= self.xi[i+1]):
                    # reusing computations is always a good idea, but here we also can return dfdx
                    dx = x - self.xi[i]
                    fxt[j] = self.a[i] + dx*(self.b[i] + self.c[i]*dx + self.d[i]*dx*dx)
            
        return fxt

    def derivative(self, xt, order = 1):
        '''
        Description: Finds derivative of specified order at x-values xt.

        Area of Improvement: Convert For loops to Numpy
        Method: 
            (0): To handle the if condition....(help!)
                 I think we can handle this using Numpy boolean indexing.
                 So, something like xt[xt >= xi and xt <= xi2]
                 where xi2 = xi[1:]. 
            (1) Create separate functions for each order of derivative.
            (2) Remove last index of a and c arrays so a, b, c, d
                all have the same shape.
            (3) Compute array of dx values using np.diff(). 
                Now, dx should have the same shape as a, b, c, d.
            (4) Compute fxt.

        '''

        a, b, c, d = self.a, self.b, self.c, self.d

        n = np.size(self.xi) - 1
        fxt = np.empty_like(xt)
        for j, x in enumerate(xt):
            for i in range(n):
                if (x >= self.xi[i]) and (x <= self.xi[i+1]):
                    dx = x - self.xi[i]
                    if order == 0: # interpolation
                        fxt[j] = a[i] + dx * (b[i] + c[i] * dx + d[i] * dx * dx)
                    elif order == 1:
                        fxt[j] = b[i] + 2 * c[i] * dx + 3 * d[i] * dx * dx
                    elif order == 2:
                        fxt[j] = 2 * c[i] + 6 * d[i] * dx
                    elif order == 3:
                        fxt[j] = 6 * d[i]
                    else:
                        raise ValueError('Order not supported.')
                        # I know this derivative exists, but I think allowing the user to
                        # calculate this may do more harm than good, as user may not realize that
                        # the derivative is 6 everywhere (for order 4) or 0 everywhere (for order 5+).
        return fxt
        
    def integral(self, t0, t1, dx):
        '''
        Integrates given function func over interval [t0, t1]
        using step-size dx. 
        '''
        n = np.size(self.xi) - 1
        xt = np.arange(t0, t1, dx)
        int_val = np.empty_like(xt)
        for j, x in enumerate(xt):
            for i in range(n):
                if self.xi[i] <= x <= self.xi[i+1]:
                    int_val[j] = self.a[i] * dx + 0.5 * self.b[i] * dx ** 2 + (1 / 3) * self.c[i] * dx ** 3 + (1 / 4) * self.d[i] * dx ** 4
        
        return int_val.sum()