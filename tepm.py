"""
Classes and Functions involved in TNV recon
SEE NOTES ON TABLET FOR DETAILS
"""
from calendar import c
import math
from re import X
import numpy as np
from numba import jit,prange

### no numba test functions ###

def coefficients(x):
    " Find the coefficients of eigenvalues from the determinant of 3x3 Matrix X"
    if x.shape != (3,3):
        raise ValueError("Matrix must be 3x3")
    else:
        c2 = -x[0][0]-x[1][1]-x[2][2]
        c1 = x[0][0]*x[1][1] + x[0][0]*x[2][2] + x[1][1]*x[2][2]- \
                    x[0][1]*x[0][1] - x[0][2]*x[0][2] - x[1][2]*x[1][2] 
        c0 = x[0][0]*x[1][2]*x[1][2] + x[1][1]*x[0][2]*x[0][2] + \
                    x[2][2]*x[0][1]*x[0][1] - x[0][0]*x[1][1]*x[2][2] -\
                    2*x[0][2]*x[0][1]*x[1][2]  
        
        return c0,c1,c2 
    
def transform(c):
    " Find phi based on coefficietns c"
    p = c[2]*c[2] - 3*c[1]
    q = -(27/2)*c[0] - c[2]*c[2]*c[2] +(9/2)*c[2]*c[1]
    t = 2*p**(-3/2)*q # is this fastest? tmp = math.sqrt(p) / 1/(p*p*p)
    return p,q,t

def phi(transform,c):
    """Find phi based on coefficients and transform values

    Args:
        t (tuple):transform values
        c (tuple): coefficients
        
    ReturnsL:
        float: phi value)
    """    
    p,q = transform[0],transform[1]
    res = 27*((c[1]*c[1]*(p-c[1])/4)+(c[0]*(q+(27/4)*c[0])))
    return np.arctan(math.sqrt((res))/q)/3

def solve_val(phi, p, c):
    """ Find eigenvectors for 3x3 matrix using phi, p and coefficients

    Args:
        phi (float): phi value calculated with phi()
        p (float): p value calculated in transform transform()[0]
        c (tuple): coefficients calculated in coefficiants

    Returns:
        tuple: eigenvalues of matrix
    """    
    x1 = 2*np.cos(phi)
    tmp1 = np.cos(phi)
    tmp2 = math.sqrt(3)*np.sin(phi)
    x2 = -tmp1 - tmp2
    x3 = -tmp1 + tmp2
    sqrtp = math.sqrt(p)
    return sqrtp/3*x1 -c[2]/3, sqrtp*x2/3 - c[2]/3, sqrtp/3*x3-c[2]/3

def solve_J_val(x):
    """ Finc singular values of 3x3 hermitian matrix x

    Args:
        x (array): 3x3 hermitian matrix

    Returns:
        tuple: singular values of x
    """    
    coeffs = coefficients(x)
    transform = transform(coeffs)
    phi = phi(transform, coeffs)
    return solve_val(phi, transform[0], coeffs)

def solve_J_vec(x, eigenvalues):
    v0 = np.transpose(np.cross(x[0][:]-eigenvalues[0]*x[0][:]/np.sum(x[0][:]), x[1][:]-eigenvalues[0]*x[0][:]/np.sum(x[1][:])))
    v1 = np.transpose(np.cross(x[0][:]-eigenvalues[1]*x[0][:]/np.sum(x[0][:]), x[1][:]-eigenvalues[1]*x[0][:]/np.sum(x[1][:])))
    v2 = np.transpose(np.cross(x[0][:]-eigenvalues[2]*x[0][:]/np.sum(x[0][:]), x[1][:]-eigenvalues[2]*x[0][:]/np.sum(x[1][:])))
    return v0,v1,v2

### jit accelerated functions ###

@jit(nopython=True)
def solve_Z_val(Z, out):
    """ Find eigenvalues of N 3x3 hermitian matrices

    Args:
        Z (array): Nx3x3 array
        out (array): Nx3 array for output of eigenvalues

    Returns:
        array: Nx3 array of eigenvalues
    """    
    for i in prange(Z.shape[0]):
        x = Z[i]
        c2 = -x[0][0]-x[1][1]-x[2][2]
        c1 = x[0][0]*x[1][1] + x[0][0]*x[2][2] + x[1][1]*x[2][2]- \
                    x[0][1]*x[0][1] - x[0][2]*x[0][2] - x[1][2]*x[1][2] 
        c0 = x[0][0]*x[1][2]*x[1][2] + x[1][1]*x[0][2]*x[0][2] + \
                    x[2][2]*x[0][1]*x[0][1] - x[0][0]*x[1][1]*x[2][2] -\
                    2*x[0][2]*x[0][1]*x[1][2]
        p = c2*c2 - 3*c1
        q = -(27/2)*c0 - c2*c2*c2 +(9/2)*c2*c1
        t = 2*p**(-3/2)*q 
        phi = np.arctan(math.sqrt(27*((c1*c1*(p-c1)/4)+(c0*(q+(27/4)*c0))))/q)/3
        #phi = np.arctan(math.sqrt(np.abs((t*t)/4-1))/(t/2))/3
        x1 = 2*np.cos(phi)
        tmp1 = np.cos(phi)
        tmp2 = math.sqrt(3)*np.sin(phi)
        x2 = -tmp1 - tmp2
        x3 = -tmp1 + tmp2
        sqrtp = math.sqrt(p)
        lam0, lam1, lam2 = sqrtp/3*x1 -c2/3, sqrtp*x2/3 - c2/3, sqrtp/3*x3-c2/3
        out[i] = np.array([lam0, lam1, lam2])
    return out
    
@jit(nopython=True)
def solve_Z_vec(Z, eigenvals, out):
    """ Find eigenvectors of n 3x3 Hermitian martrices

    Args:
        x (array): Nx3x3x numpy array (containing N jacobians)
        eigenvalues (array): Nx3 numpy array of eigenvalues
        out (array): Nx3x3 matrix to fill with eigenvectors

    Returns:
        array: Nx3x3 matrix of eigenvectors
    """    
    for i in prange(Z.shape[0]):
        for j in prange(Z[i].shape[0]):
            tmp1 = Z[i][0][:]-eigenvals[i][j]*Z[i][0][:]/np.sum(Z[i][0][:])
            tmp2 = Z[i][1][:]-eigenvals[i][j]*Z[i][0][:]/np.sum(Z[i][1][:])
            tmp3 = np.transpose(np.cross(tmp1,tmp2))
            out[i][j]=tmp3
    return out

@jit(nopython=True)
def create_EPSILON(eigenvals, out, project = False):
    """ create Nx3x3 array of diagonalised eigenvalue matrices

    Args:
        eigenvals (array): Nx3 array of N x 3 eigenvalues
        out (array): Nx3x3 template to fill with N 3x3 diagnonal vectors
        project (bool, optional): Whether to project to unit ball

    Returns:
        _type_: _description_
    """    
    for i in prange(eigenvals.shape[0]):
        for j in prange(eigenvals.shape[1]):
            if project is False:
                out[i][j][j] = eigenvals[i][j]
            else:
                if eigenvals[i][j] < 1 and  eigenvals[i][j] > -1:
                    out[i][j][j] = eigenvals[i][j]
                else: 
                    out[i][j][j] = eigenvals[i][j]/np.abs(eigenvals[i][j])
    return out

@jit(nopython=True)
def project(x, vecs, epsilon, epsilon_p, out):
    for i in prange(x.shape[0]):
        out[i] = np.dot(x[i], np.dot(vecs[i], np.dot(np.transpose(epsilon[i]), np.dot(epsilon_p[i], np.transpose(vecs[i])))))
        
    return out
### classes ###

class Norm_Projector():
    def __init__(self, norm = None):
        if norm is not None:
            self.set_up(norm)
        
        def set_up(self, norm):
            self.norm = norm
            
        def direct(self, x):
            vals = solve_Z_val(x, np.zeros(x.shape[0],3))
            vecs = solve_Z_vec(x,vals, np.zeros(x.shape[0],3,3))
            epsilon = create_EPSILON(vals, np.zeros(x.shape[0],3,3))
            epsilon_p = create_EPSILON(vals, np.zeros(x.shape[0],3,3), project = True)
            
            
    


