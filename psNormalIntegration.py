import numpy as np
import numpy.linalg as lin
from scipy import ndimage
from scipy.ndimage import filters
import scipy
from scipy.sparse import coo_matrix,csr_matrix,bmat,lil_matrix
from scipy.optimize import nnls,lsq_linear,least_squares
from scipy.sparse.linalg import spsolve,lsqr,cg


#Code from francois
def make_bc_data(mask):
     """
     Create the data structure used to enforce some  null Neumann BC condition on
     some PDEs used in my Photometric Stereo Experiments.
     Argument:
     ---------
     mask: numpy array
         a binary mask of size (m,n).
     Returns:
     --------
         west, north, east, south, inside, n_pixels with
         west[i]  index of point at the "west"  of mask[inside[0][i],inside[1][i]]
         north[i] index of point at the "north" of mask[inside[0][i],inside[1][i]]
         east[i]  index of point at the "east"  of mask[inside[0][i],inside[1][i]]
         south[i] index of point at the "south" of mask[inside[0][i],inside[1][i]]
         inside: linear indices of points inside the mask
         n_pixels: number of inside / in domain pixels
     """
     m,n = mask.shape
     inside = np.where(mask)
     x, y = inside
     n_pixels = len(x)
     m2i = -np.ones(mask.shape)
     # m2i[i,j] = -1 if (i,j) not in domain, index of (i,j) else.
     m2i[(x,y)] = range(n_pixels)
     west  = np.zeros(n_pixels, dtype=int)
     north = np.zeros(n_pixels, dtype=int)
     east  = np.zeros(n_pixels, dtype=int)
     south = np.zeros(n_pixels, dtype=int)


     for i in range(n_pixels):
         xi = x[i]
         yi = y[i]
         wi = x[i] - 1
         ni = y[i] + 1
         ei = x[i] + 1
         si = y[i] - 1

         west[i]  = m2i[wi,yi] if (wi > 0) and (mask[wi, yi] > 0) else i
         north[i] = m2i[xi,ni] if (ni < n) and (mask[xi, ni] > 0) else i
         east[i]  = m2i[ei,yi] if (ei < m) and (mask[ei, yi] > 0) else i
         south[i] = m2i[xi,si] if (si > 0) and (mask[xi, si] > 0) else i

     return west, north, east, south, inside, n_pixels




def PoissonFDEequationsWithVonNeumann(mask,p,q,d=1,display = False,rot90 = False):
    """
    Creates the linear equation system for solving a 2D poisson problem.
    Assumes that the image is square.

    Uses the stencil 
    0  -1  0
    -1  4 -1
    0  -1  0

    For aproximating the laplacian Δz.
    """
    if rot90:
        pn = np.rot90(p).copy()
        qn = np.rot90(q).copy()
        nmask = np.rot90(mask).copy()
    else:
        pn = p
        qn = q
        nmask = mask
        
    west, north, east, south, inside, n_pixels = make_bc_data(nmask)
    
    pointinside = (lambda i : np.array([inside[0][i],inside[1][i]]))
    # We use central (and a bit of foward) aproximation for finding nabla u
    cdx =  1/2*(pn[inside][east] - pn[inside][west]) 
    cdy =  1/2*(qn[inside][north] - qn[inside][south])

 

    
    pinside = pn[inside]
    qinside = qn[inside] 
    A = lil_matrix((n_pixels, n_pixels))


    hasNorth =  (north != np.arange(0,n_pixels))
    hasSouth =  (south != np.arange(0,n_pixels))
    hasEast  =  (east != np.arange(0,n_pixels))
    hasWest =   (west != np.arange(0,n_pixels))

    directionCounts = (hasSouth)*1 + (hasNorth)*1 + (hasEast)*1 + (hasWest)*1
    hasNS = hasNorth * hasSouth
    hasEW = hasEast * hasWest


    if 1 == 1: 
        cdx[hasEast == 0] = cdx[hasEast == 0]*2 
        cdx[hasWest == 0] = cdx[hasWest == 0]*2 
        cdy[hasNorth == 0] = cdy[hasNorth == 0]*2 
        cdy[hasSouth == 0] = cdy[hasSouth == 0]*2 
            
   
    cd = cdx + cdy
    nb = cd

    cdxplot = np.zeros(nmask.shape)
    cdyplot = np.zeros(nmask.shape)
    cdplot  = np.zeros(nmask.shape)
    cdplot[inside] = cd
    cdxplot[inside] = cdx
    cdyplot[inside] = cdy
    # if display:
    #     plt.imshow(cdxplot) ; plt.show()
    #     plt.imshow(cdyplot) ; plt.show()
    #     plt.imshow(cdplot) ; plt.show()

    d2 = d*d
    
    def displayPoint(i):
        cmask = nmask.copy()
        cmask[inside[0][i],inside[1][i]] = 2
        plt.imshow(cmask)
        plt.show()
    
    for i in range(0,n_pixels):
        if directionCounts[i] == 4: # The point is inside, easy case
            A[i,i] = 4
            A[i,north[i]] = -1
            A[i,south[i]] = -1
            A[i,east[i]] = -1
            A[i,west[i]] = -1
            nb[i] *= d
        elif 2 == 1:
            A[i,i] =1
            nb[i] = 0
        elif directionCounts[i] == 3: # Just one missing coordinate
            A[i,i] = 4
            sign = -1

            if north[i] == i: #North is missing            
                A[i,south[i]] = -2
                A[i,east[i]] = -1
                A[i,west[i]] = -1
                nb[i] += 2*sign*qinside[i]
            elif south[i] == i:
                A[i,north[i]] = -2
                A[i,east[i]] = -1
                A[i,west[i]] = -1
                nb[i] += -2*sign*qinside[i]
            elif east[i] == i: 
                A[i,south[i]] = -1
                A[i,north[i]] = -1
                A[i,west[i]] = -2
                nb[i] += 2*sign*pinside[i]
            else: # west[i] == i: 
                A[i,south[i]] = -1
                A[i,north[i]] = -1
                A[i,east[i]] = -2
                nb[i] += -2*sign*pinside[i]
            nb[i] *= d
        elif directionCounts[i] == 2:
            #Find out if corner or what
            if hasNS[i]: #East and west is missing
                A[i,i] = 2
                A[i,south[i]] = -1
                A[i,north[i]] = -1
           
            elif hasEW[i]: #North and south
                A[i,i] = 2
                A[i,east[i]] = -1
                A[i,west[i]] = -1
            else:
                A[i,i] = 4
                nu = -1*np.array([(hasEast[i] == 0)*1 - (hasWest[i] == 0)*1,(hasNorth[i] == 0)*1 - (hasSouth[i] == 0)*1])
                mult = -1
                if hasNorth[i] + hasWest[i] == 0:
                    A[i,south[i]] = -1 + mult
                    A[i,east[i]]  = -1 + mult
                    nb[i] += 2*nu[0]*pinside[i] 
                    nb[i] += 2*nu[1]*qinside[i] 

                elif hasWest[i] + hasSouth[i] == 0:
                    A[i,north[i]] = -1 + mult
                    A[i,east[i]]  = -1 + mult
                    nb[i] +=  2*nu[0]*pinside[i] 
                    nb[i] +=  2*nu[1]*qinside[i]
                elif hasSouth[i] + hasEast[i] == 0:
                    A[i,north[i]] = -1 + mult
                    A[i,west[i]]  = -1 + mult
                    nb[i] +=  2*nu[0]*pinside[i]
                    nb[i] +=  2*nu[1]*qinside[i] 
                elif hasEast[i] + hasNorth[i] == 0:
                    A[i,south[i]] = -1 + mult
                    A[i,west[i]]  = -1 + mult
                    nb[i] += 2*nu[0]*pinside[i] 
                    nb[i] += 2*nu[1]*qinside[i] 
            nb[i] *= d
        elif directionCounts[i] == 1:
            A[i,i] = 1
            nb[i] = 0  
            sign = -1
            #Using linear intepolation
            if north[i] != i: #North not missing            
                A[i,north[i]] = -1
                nb[i] += 2*sign*qinside[i]
            if south[i] != i: #south not missing            
                A[i,south[i]] = -1
                nb[i] += -2*sign*qinside[i]
            if east[i] != i:  #east not missing            
                A[i,east[i]] = -1
                nb[i] += 2*sign*pinside[i]
            if west[i] != i: #west not missing            
                A[i,west[i]] = -1
                nb[i] += -2*sign*pinside[i]

            nb[i] *= d
            # Note, we dont do nb[i] *= d
        else:
            A[i,i] = 1
            nb[i] = 0
                        
    return (csr_matrix(A),nb)

#FIXME: Update discription to take stretch into acount
def PoissonFDEequationsWithVonNeumannAndStretch(mask,p,q,l,h,display = False,rot90 = False):
    """
    Creates the linear equation system for solving a 2D poisson problem.
    Assumes that the image is square, but not that the pixels are.
    """

    
    if rot90:
        pn = np.rot90(p).copy()
        qn = np.rot90(q).copy()
        nmask = np.rot90(mask).copy()
    else:
        pn = p
        qn = q
        nmask = mask
        
    west, north, east, south, inside, n_pixels = make_bc_data(nmask)
    
    pointinside = (lambda i : np.array([inside[0][i],inside[1][i]]))
    # We use central (and a bit of foward) aproximation for finding nabla u
    cdx =  1/(2*l)*(pn[inside][east] - pn[inside][west]) 
    cdy =  1/(2*h)*(qn[inside][north] - qn[inside][south])

    il = 1/l
    ih = 1/h
    il2 = l**(-2)
    ih2 = h**(-2)

    
    pinside = pn[inside]
    qinside = qn[inside] 
    A = lil_matrix((n_pixels, n_pixels))


    hasNorth =  (north != np.arange(0,n_pixels))
    hasSouth =  (south != np.arange(0,n_pixels))
    hasEast  =  (east != np.arange(0,n_pixels))
    hasWest =   (west != np.arange(0,n_pixels))

    directionCounts = (hasSouth)*1 + (hasNorth)*1 + (hasEast)*1 + (hasWest)*1
    hasNS = hasNorth * hasSouth
    hasEW = hasEast * hasWest


    #FIXME: Ussiker på sidste del
    if 1 == 1: 
        cdx[hasEast == 0] = cdx[hasEast == 0]*2
        cdx[hasWest == 0] = cdx[hasWest == 0]*2
        cdy[hasNorth == 0] = cdy[hasNorth == 0]*2 
        cdy[hasSouth == 0] = cdy[hasSouth == 0]*2 
            
   
    cd = cdx + cdy
    nb = cd

    cdxplot = np.zeros(mask.shape)
    cdyplot = np.zeros(mask.shape)
    cdplot  = np.zeros(mask.shape)
    cdplot[inside] = cd
    cdxplot[inside] = cdx
    cdyplot[inside] = cdy
    if display:
        plt.imshow(cdxplot) ; plt.show()
        plt.imshow(cdyplot) ; plt.show()
        plt.imshow(cdplot) ; plt.show()

    
    def displayPoint(i):
        cmask = mask.copy()
        cmask[inside[0][i],inside[1][i]] = 2
        plt.imshow(cmask)
        plt.show()
    
    for i in range(0,n_pixels):
        if directionCounts[i] == 4: # The point is inside, easy case
            A[i,i] = 2*(il2 + ih2)
            A[i,north[i]] = -1*ih2
            A[i,south[i]] = -1*ih2
            A[i,east[i]] = -1*il2
            A[i,west[i]] = -1*il2
        elif 2 == 1:
            A[i,i] =1
            nb[i] = 0
        elif directionCounts[i] == 3: # Just one missing coordinate
            A[i,i] = 2*(il2 + ih2)
            sign = -1

            if north[i] == i: #North is missing            
                A[i,south[i]] = -2*ih2
                A[i,east[i]] = -1*il2
                A[i,west[i]] = -1*il2
                nb[i] += ih*2*sign*qinside[i]
            elif south[i] == i:
                A[i,north[i]] = -2*ih2
                A[i,east[i]] = -1*il2
                A[i,west[i]] = -1*il2
                nb[i] += -ih*2*sign*qinside[i]
            elif east[i] == i: 
                A[i,south[i]] = -1*ih2
                A[i,north[i]] = -1*ih2
                A[i,west[i]] = -2*il2
                nb[i] += il*2*sign*pinside[i]
            else: # west[i] == i: 
                A[i,south[i]] = -1*ih2
                A[i,north[i]] = -1*ih2
                A[i,east[i]] = -2*il2
                nb[i] += -il*2*sign*pinside[i]
        elif directionCounts[i] == 2:
            #Find out if corner or what
            if hasNS[i]: #East and west is missing
                A[i,i] = 2*ih2
                A[i,south[i]] = -1*ih2
                A[i,north[i]] = -1*ih2
           
            elif hasEW[i]: #North and south
                A[i,i] = 2*il2
                A[i,east[i]] = -1*il2
                A[i,west[i]] = -1*il2
            else:
                A[i,i] = 2*(il2 + ih2)
                nu = -1*np.array([(hasEast[i] == 0)*1 - (hasWest[i] == 0)*1,(hasNorth[i] == 0)*1 - (hasSouth[i] == 0)*1])
                mult = -1
                if hasNorth[i] + hasWest[i] == 0:
                    A[i,south[i]] = -2 * ih2
                    A[i,east[i]]  = -2 * il2
                    # nb[i] += 2*nu[0]*pinside[i] 
                    # nb[i] += 2*nu[1]*qinside[i] 
                elif hasWest[i] + hasSouth[i] == 0:
                    A[i,north[i]] = -2 * ih2
                    A[i,east[i]]  = -2 * il2
                    # nb[i] +=  2*nu[0]*pinside[i] 
                    # nb[i] +=  2*nu[1]*qinside[i]
                elif hasSouth[i] + hasEast[i] == 0:
                    A[i,north[i]] = -2 * ih2
                    A[i,west[i]]  = -2 * il2
                    # nb[i] +=  2*nu[0]*pinside[i]
                    # nb[i] +=  2*nu[1]*qinside[i] 
                elif hasEast[i] + hasNorth[i] == 0:
                    A[i,south[i]] = -2 * ih2
                    A[i,west[i]]  = -2 * il2
                    # nb[i] += 2*nu[0]*pinside[i] 
                    # nb[i] += 2*nu[1]*qinside[i]
                
                nb[i] += 2*nu[0]*pinside[i]*il
                nb[i] += 2*nu[1]*qinside[i]*ih
                
        elif directionCounts[i] == 1:
            A[i,i] = 1
            nb[i] = 0  
            sign = -1
            #Using linear intepolation
            if north[i] != i: #North not missing            
                A[i,north[i]] = -1
                nb[i] += 2*sign*qinside[i]
            if south[i] != i: #south not missing            
                A[i,south[i]] = -1
                nb[i] += -2*sign*qinside[i]
            if east[i] != i:  #east not missing            
                A[i,east[i]] = -1
                nb[i] += 2*sign*pinside[i]
            if west[i] != i: #west not missing            
                A[i,west[i]] = -1
                nb[i] += -2*sign*pinside[i]
        else:
            A[i,i] = 1
            nb[i] = 0
                        
    return (csr_matrix(A),nb)



def PoissionPDequationWithNormPenalty(mask,p,q,lam,z0 = 0,d=1,display = False,rot90 = False):
    """
    Creates a linear equation system for solving the problem
    argmin_z ||∇z - (p,q)||² + ||z₀ - z||²

    """
    if rot90:
        pn = np.rot90(p).copy()
        qn = np.rot90(q).copy()
        nmask = np.rot90(mask).copy()
    else:
        pn = p
        qn = q
        nmask = mask


    # Equations for solving first part of the problem
    oA,ob  = PoissonFDEequationsWithVonNeumann(nmask,pn,qn,d = d)

    n_pixels = oA.shape[0]
    
    # A = lil_matrix((2*n_pixels, n_pixels))  

    b = np.zeros((2*n_pixels,))

    lamI = lil_matrix((n_pixels, n_pixels))

    
    for i in range(0,n_pixels):
        lamI[i,i] = lam

    

    liloA = lil_matrix(oA)

    A = scipy.sparse.vstack([liloA,lamI])
    
    b[0:n_pixels] = ob
    b[n_pixels:2*n_pixels] = z0*np.ones((n_pixels,))



    return (A.tocsr(),b)




def PoissonSolverPS(normals,mask,p = None, q = None,l=None,h= None,damp = 0,newNormal = False,rot90 = False):
    """
    Solves the 2D case of the possion problem given unit normals -normals-
    and a mask -mask-. Returns a touple (A,b,z) represenitng the linear system,
    with z being the solution to Ax = b.
    """

    if (p is None) or (q is None):
        p = -normals[0]/normals[2]
        q = -normals[1]/normals[2]
    
    # plt.imshow(p); plt.show()
    if newNormal:
        p = p*l
        q = q*h
        

    if l == None or h == None:
        A,b = PoissonFDEequationsWithVonNeumann(mask,p,q,rot90 = rot90)
    else:
        A,b = PoissonFDEequationsWithVonNeumannAndStretch(mask,p,q,l,h,rot90 = rot90)
    if rot90:
        nmask = np.rot90(mask).copy()
    else:
        nmask = mask
    z = np.zeros(nmask.shape)

    if damp == 0:
         vals = spsolve(A,b)
    else:
         vals = lsqr(A,b,damp=damp)[0]
         print("vals")
         print(vals)
    z[np.where(nmask > 0)] = vals
    z[nmask == 0] = np.nan
    if rot90:
        z = np.rot90(z, k=1, axes=(1,0))
    else:
        z = -z
    
    return (A,b,z)

def PoissonSolverNormPenalised(mask,p,q,lam,d=1,z0= 0,rot90 = False):

    #See https://docs.scipy.org/doc/scipy/reference/generated/scipy.sparse.linalg.lsqr.html
    A,b = PoissionPDequationWithNormPenalty(mask,p,q,lam,z0,d=d,rot90 = rot90)
    if rot90:
        nmask = np.rot90(mask).copy()
    else:
        nmask = mask
    z = np.zeros(nmask.shape)

    print("lsq_linear")
    A2,b2 = PoissonFDEequationsWithVonNeumann(mask,p,q,d=d,rot90 = rot90)
    # out = lsqr(A2,b2,damp = lam,show = True)

    # out = lsq_linear(A,b,verbose = 2)  #
    #Conjugate gradient on normal equations
    out = cg(A.T @ A,A.T @ b)
    # print(out)
    # vals =  out.x
    vals = out[0]
    # n_pixels = int(b.shape[0]/2)
    z[np.where(nmask > 0)] = vals
    z[nmask == 0] = np.nan
    if rot90:
        z = np.rot90(z, k=1, axes=(1,0))
    else:
        z = -z
    
    return (A,b,z)






"""
Working on single step integration
"""

# def sspsErrorNormal(m,I,S):
#      """
#      Returns ∑ (I - <m/|m|,S>)^2

#      Will assume that I is already masked. 
#      I must have shape (k,n)
#      m shape (k,3)
#      S shape (n,3)
#      """
#      mn = lin.norm(m,2,axis = 1)


#      mdmn = m/(np.array([mn]).T)
    

#      return lin.norm(I - mdmn @ S.T,2)**2

def sspsErrorNormal(m,I,S):
     """
     Returns ∑ (I - <m/|m|,S>)^2

     Will assume that I is already masked. 
     I must have shape (k,n)
     m shape (k,3)
     S shape (n,3)
     """
    

     return lin.norm(I - m @ S.T,2)**2

# def sspsErrorNormalGradient(m,I,S):
#      """
#      """

#      mn = np.array([lin.norm(m,2,axis = 1)]).T
     
#      g = np.zeros(m.shape)
#      for i in range(0,S.shape[0]):
#          firstterm = np.array([2*(I[:,i] - m/mn @ S[i])]).T
#          Sdmn = -S[i]/mn
#          mS = (((m @ S[i]))/(mn**3).T).T * m
#          g +=firstterm * (Sdmn + mS)
         
     
#      return g

def sspsErrorNormalGradient(m,I,S):
     """
     """

     mn = np.array([lin.norm(m,2,axis = 1)]).T
     
     g = np.zeros(m.shape)
     for i in range(0,S.shape[0]):
          firstterm = np.array([2*(I[:,i] - m @ S[i])]).T
          g += firstterm * -S[i]
     
     return g


def sspsErrorDerivatives(m,c,z,f,nabla):

     """
     Returns || m - ρf(∇z)||

     find a way to get ∇ z, find a way to get f
     """

     
     return  (c**2)/2 * np.norm(m - f(nabla @ z),2)**2

     
def sspsErrorDerivativesGradientM(m,c,z,f,nabla):

     """
     Returns || m - ρf(∇z)||

     find a way to get ∇ z, find a way to get f
     """
     mn = np.array([lin.norm(m,2,axis = 1)]).T

     
     (c**2) * (m - f(nabla @ z))
     return()
     
