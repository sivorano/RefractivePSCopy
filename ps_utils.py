# -*- coding: utf-8 -*-
"""
integration.py
---------------

Two methods of integration of a normal field to a depth function
by solving a Poisson equation. 
- The first, unbiased, implements a Poisson solver on an irregular domain.
  rather standard approach.
- The second implements the Simchony et al. method for integration of a normal
  field.
  
  
They are port of Yvain Quéau's Matlab implementation to Python.
See Yvain's code for more!

Author: Francois Lauze, University of Copenhagen
Date December 2015 / January 2016
"""

import numpy as np
from scipy import fftpack as fft
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve




def cdx(f):
    """
    central differences for f-
    """
    m = f.shape[0]
    west = [0] + list(range(m-1))
    east = list(range(1,m)) + [m-1]
    return 0.5*(f[east,:] - f[west,:])
    
def cdy(f):
    """
    central differences for f-
    """
    n = f.shape[1]
    south = [0] + list(range(n-1))
    north = list(range(1,n)) + [n-1]
    return 0.5*(f[:,north] - f[:,south])
    
def sub2ind(shape, X, Y):
    """
    An equivalent of Matlab sub2ind, but without 
    argument checking and for dim 2 only.
    """    
    Z = np.array(list(zip(X,Y))).T
    shape = np.array(shape)
    indices = np.dot(shape, Z)
    indices.shape = indices.size
    return indices
    
def tolist(A):
    """
    Linearize array to a 1D list
    """
    return list(np.reshape(A, A.size))
    

    
def simchony_integrate(n1, n2, n3, mask, p = None, q = None):
    """
    Integration of the normal field recovered from observations onto 
    a depth map via Simchony et al. hybrid DCT / finite difference
    methods.
    
    Done by solving via DCT a finite difference equation discretizing
    the equation:
        Laplacian(z) - Divergence((n1/n3, n2/n3)) = 0
    under proper boundary conditions ("natural" boundary conditions on 
    a rectangular domain)
    
    Arguments:
    ----------
    n1, n2, n3: nympy float arrays 
        the 3 components of the normal field. They must be 2D arrays
        of size (m,n). Array (function) n3 should never be 0.
       
    Returns:
    --------
        z : depth map obtained by integration of the field n1/n3, n2/n3
    """
    # first a bit paranoid, so check arguments
    if (type(n1) != np.ndarray) or (type(n2) != np.ndarray) or (type(n3) != np.ndarray):
        raise TypeError('One or more arguments are not numpy arrays.')
        
    if (len(n1.shape) != 2) or (len(n2.shape) != 2) or (len(n3.shape) != 2):
        raise TypeError('One or more arguments are not 2D arrays.')

    if (n1.shape != n2.shape) or (n1.shape != n3.shape):
        raise TypeError('Array dimensions mismatch.')

    try:
        n1 = n1.astype(float)
        n2 = n2.astype(float)
        n3 = n3.astype(float)
    except:
        raise TypeError('Arguments not all (convertible to) float.')
        
        
    # Hopefully on the safe side now
    m,n = n1.shape

    if p is None or q is None:
        p = -n1/n3
        q = -n2/n3

    # divergence of (p,q)
    px = cdx(p)
    qy = cdy(q)
    
    f = px + qy      

    # 4 edges
    f[0,1:-1]  = 0.5*(p[0,1:-1] + p[1,1:-1])    
    f[-1,1:-1] = 0.5*(-p[-1,1:-1] - p[-2,1:-1])
    f[1:-1,0]  = 0.5*(q[1:-1,0] + q[1:-1,1])
    f[1:-1,-1] = 0.5*(-q[1:-1,-1] - q[1:-1,-2])

    # 4 corners
    f[ 0, 0] = 0.5*(p[0,0] + p[1,0] + q[0,0] + q[0,1])
    f[-1, 0] = 0.5*(-p[-1,0] - p[-2,0] + q[-1,0] + q[-1,1])
    f[ 0,-1] = 0.5*(p[0,-1] + p[1,-1] - q[0,-1] - q[1,-1])
    f[-1,-1] = 0.5*(-p[-1,-1] - p[-2,-1] -q[-1,-1] -q[-1,-2])

    # cosine transform f (reflective conditions, a la matlab, 
    # might need some check)
    fs = fft.dct(f, axis=0, norm='ortho')
    fs = fft.dct(fs, axis=1, norm='ortho')

    # check that this one works in a safer way than Matlab!
    x, y = np.mgrid[0:m,0:n]
    denum = (2*np.cos(np.pi*x/m) - 2) + (2*np.cos(np.pi*y/n) -2)
    Z = fs/denum
    Z[0,0] = 0.0 
    # or what Yvain proposed, it does not really matters
    # Z[0,0] = Z[1,0] + Z[0,1]
    
    z = fft.dct(Z, type=3, norm='ortho', axis=0)
    z = fft.dct(z, type=3, norm='ortho', axis=1)
    out = np.where(mask == 0)
    z[out] = np.nan
    return z
# simchony()





def unbiased_integrate(n1, n2, n3, mask,p = None,q = None, order=2):
    """
    Constructs the finite difference matrix, domain and other information
    for solving the Poisson system and solve it. Port of Yvain's implementation, 
    even  respecting the comments :-)
    
    It creates a matrix A which is a finite difference approximation of 
    the neg-laplacian operator for the domain encoded by the mask, and a
    b matrix which encodes the neg divergence of (n1/n3, n2/n3).
    
    The depth is obtained by solving the discretized Poisson system
    Az = b, 
    z needs to be reformated/reshaped after that.
    """
    
    if p is None or q is None:    
        p = -n1/n3
        q = -n2/n3        
    
    # Calculate some usefuk masks
    m,n = mask.shape
    Omega = np.zeros((m,n,4))
    Omega_padded = np.pad(mask, (1,1), mode='constant', constant_values=0)
    Omega[:,:,0] = Omega_padded[2:,1:-1]*mask
    Omega[:,:,1] = Omega_padded[:-2,1:-1]*mask
    Omega[:,:,2] = Omega_padded[1:-1,2:]*mask
    Omega[:,:,3] = Omega_padded[1:-1,:-2]*mask
    del Omega_padded
    
    # Mapping    
    indices_mask = np.where(mask > 0)
    lidx = len(indices_mask[0])
    mapping_matrix = np.zeros(p.shape, dtype=int)
    mapping_matrix[indices_mask] = list(range(lidx))
    
    if order == 1:
        pbar = p.copy()
        qbar = q.copy()
    elif order == 2:
        pbar = 0.5*(p + p[list(range(1,m)) + [m-1], :])
        qbar = 0.5*(q + q[:, list(range(1,n)) + [n-1]])
        
    # System
    I = []
    J = []
    K = []
    b = np.zeros(lidx)


    # In mask, right neighbor in mask
    rset = Omega[:,:,2]
    X, Y = np.where(rset > 0)
    #indices_center = sub2ind(mask.shape, X, Y)
    I_center = mapping_matrix[(X,Y)].astype(int)
    #indices_neighbors = sub2ind(mask.shape, X, Y+1)
    I_neighbors = mapping_matrix[(X,Y+1)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] -= qbar[(X,Y)]
    
	
    #	In mask, left neighbor in mask
    lset = Omega[:,:,3]
    X, Y = np.where(lset > 0)
    #indices_center = sub2ind(mask.shape, X, Y)
    I_center = mapping_matrix[(X,Y)].astype(int)
    #indices_neighbors = sub2ind(mask.shape, X, Y-1)
    I_neighbors = mapping_matrix[(X,Y-1)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)  
    b[I_center] += qbar[(X,Y-1)]


    # In mask, top neighbor in mask
    tset = Omega[:,:,1]
    X, Y = np.where(tset > 0)
    #indices_center = sub2ind(mask.shape, X, Y)
    I_center = mapping_matrix[(X,Y)].astype(int)
    #indices_neighbors = sub2ind(mask.shape, X-1, Y)
    I_neighbors = mapping_matrix[(X-1,Y)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] += pbar[(X-1,Y)]


    #	In mask, bottom neighbor in mask
    bset = Omega[:,:,0]
    X, Y = np.where(bset > 0)
    #indices_center = sub2ind(mask.shape, X, Y)
    I_center = mapping_matrix[(X,Y)].astype(int)
    #indices_neighbors = sub2ind(mask.shape, X+1, Y)
    I_neighbors = mapping_matrix[(X+1,Y)]
    lic = len(X)
    A_center = np.ones(lic)
    A_neighbors = -A_center
    K += tolist(A_center) + tolist(A_neighbors)
    I += tolist(I_center) + tolist(I_center)
    J += tolist(I_center) + tolist(I_neighbors)
    b[I_center] -= pbar[(X,Y)]
    
    # Construction de A : 
    A = sp.csc_matrix((K, (I, J)))
    A = A + sp.eye(A.shape[0])*1e-9
    z = np.nan*np.ones(mask.shape)
    z[indices_mask] = spsolve(A, b)
    return z
    


def display_surface(z):
    """
    Display the computed depth function as a surface using 
    mayavi mlab.
    """
    from mayavi import mlab
    m, n = z.shape
    x, y = np.mgrid[0:m, 0:n]
    
    mlab.mesh(x, y, z, scalars=z, colormap="Greys")
    mlab.show()
    
    
def display_image(u):
    """
    Display a 2D imag
    """
    from matplotlib import pyplot as plt
    plt.imshow(u)
    plt.show()
    
    
    
def read_data_file(filename):
    """
    Read a matlab PS data file and returns
    - the images as a 3D array of size (m,n,nb_images)
    - the mask as a 2D array of size (m,n) with 
      mask > 0 meaning inside the mask
    - the light matrix S as a (nb_images, 3) matrix
    """
    from scipy.io import loadmat
    
    data = loadmat(filename)
    I = data['I']
    mask = data['mask']
    S = data['S']
    return I, mask, S
    
    
    
    
if __name__ == "__main__":
    from scipy.io import loadmat
    from numpy.linalg import inv, pinv    
    
    data = loadmat('Beethoven')
 
    I1 = data['I'][:,:,0]
    I2 = data['I'][:,:,1]
    I3 = data['I'][:,:,2]    
    S = data['S']    
    mask = data['mask']
    
    m, n = mask.shape
    
    iS = inv(S)
    nz = np.where(mask > 0)
    npix = len(nz[0])
    
    I = np.zeros((3,npix))
    I[0,:] = I1[nz]
    I[1,:] = I2[nz]
    I[2,:] = I3[nz]
    N = np.dot(iS, I)
    Rho = np.sqrt(N[0,:]**2 + N[1,:]**2 + N[2,:]**2)
    rho = np.zeros((m,n))
    rho[nz] = Rho
    
    #display_image(rho)
    
    n1 = np.zeros((m,n))
    n2 = np.zeros((m,n))
    n3 = np.ones((m,n))
    
    n1[nz] = N[0,:]/Rho
    n2[nz] = N[1,:]/Rho
    n3[nz] = N[2,:]/Rho
    
    z = simchony_integrate(n1, n2, n3, mask)
    #z = unbiased_integrate(n1, n2, n3, mask)

    
