
# -*- coding: utf-8 -*-
"""
Library for solving refractive PS problems

Created on Tue 1/9/2020

@author: Anders Samsø Birch
"""

import numpy as np
import numpy.linalg as lin
from scipy import ndimage
from scipy.ndimage import filters
import scipy
from scipy.sparse import coo_matrix,csr_matrix,bmat,lil_matrix
import matplotlib.pyplot as plt
from scipy.optimize import nnls,lsq_linear,least_squares,minimize
from scipy.sparse.linalg import spsolve
import matplotlib.image as mpimg
import math
from math import sqrt,sin,cos
import skimage
from skimage import color
from skimage import io
from os import listdir
import pslib
from pslib import hat,plotImage, plotImages
import mathutils
import ps_utils
import psNormalIntegration as psnormal
from scipy import ndimage


#From Fracois ps_utils 
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






# Same
def display_surface2(z, col = None,SaveName = None, SaveDir = None,mayavispecs = None):
    """
    Display the computed depth function as a surface using 
    mayavi mlab.
    """
    from mayavi import mlab
    m, n = z.shape
    x, y = np.mgrid[0:m, 0:n]
    kz = z.copy()
    kz = np.nan_to_num(kz,0)
    nMin = np.min(kz)
    nz = z.copy()
    nz = np.nan_to_num(nz,nan = nMin)
    # print(nz)
    mask = (np.isnan(z) == 0)
    if mayavispecs is not None:
        mlab.view(azimuth = mayavispecs["azimuth"], elevation = mayavispecs["elevation"],distance = mayavispecs["distance"])

    if col is None:
        fig = mlab.mesh(x, y, nz, scalars=nz,name = "test")
    else:
        ncol = col.copy()
        ncol[np.isnan(ncol) == True] = 0
        fig = mlab.mesh(x,y,nz, scalars=col,name = "test")

    if SaveName is not None and SaveDir is not None:
        mlab.savefig(SaveDir + SaveName + "Mayavi" + ".png")
        # mlab.clf()
        mlab.close(all=True)
    else:
        mlab.show()


def display_mesh(x,y,z,col = None,SaveName = None, SaveDir = None,mayavispecs = None):
    from mayavi import mlab
    zmin = np.amin(z[np.isnan(z) == False])
    nz = z.copy()
    nz[np.isnan(nz) == True] =  np.mean(nz[np.isnan(nz) == False]) #zmin
    ny = y.copy()
    ny[np.isnan(z) == True] = np.mean(ny[np.isnan(z) == False])
    nx = x.copy()
    nx[np.isnan(z) == True] = np.mean(nx[np.isnan(z) == False])


    
    if col is None:
        scalars = nz
    else:
        scalars = col
    print("Hej")

    # fig = mlab.figure(size = (500,500)) 
    mlab.mesh(nx, ny, nz, scalars=scalars,colormap = "gray")
    if SaveName is not None and SaveDir is not None:
        print("Hej")
        import time
        # mlab.options.offscreen = True
        # mlab.show()
        # imgmap = mlab.screenshot(mode='rgba',figure = fig, antialiased=True)

        # time.sleep(1)

        mlab.savefig(SaveDir + SaveName + "Mayavi" + ".png",size = (500,500),magnification = 1)
        print("Hej")
        mlab.clf()
        # mlab.close(all=True)
        # plt.imshow(imgmap)
        # mlab.close(fig)
        print("Hej")
        # plt.savefig(SaveDir + SaveName + "Mayavi" + ".png")
    else:
        mlab.show()

  
def display_surface(z, col = None,SaveName = None, SaveDir = None,mayavispecs = None):
    """
    Display the computed depth function as a surface using 
    mayavi mlab.
    """
    from mayavi import mlab
    
    INSIDΕ = True
    m, n = z.shape
    x, y = np.mgrid[0:m, 0:n]
    x = x.astype(float)
    y = y.astype(float)

    display_mesh(x,y,z,col,SaveName,SaveDir,mayavispecs)


    

def CentralDifference(p,q):
    """
    p is horizontal, q is vertiacal
    see page 7 for defintion
    """

    n,m = p.shape

    left1 = np.append([0],np.arange(0,n-1))
    right1 = np.append(np.arange(1,n),[n-1])

    left2 = np.append([0],np.arange(0,m-1))
    right2 = np.append(np.arange(1,m),[m-1])

    horizontalDiff = p[right1,:] - p[left1,:]
    verticalDiff = q[:,right2] - q[:,left2]

    return (1/2)*(horizontalDiff + verticalDiff)




def PhotometricStereoNormalsold(imgArr,mask,LightDirections, display = False):
    """
    Uses photometric stereo to find the normals of an surface from multiple picturs of
    a still camera with different light directions. Assumes the material to be lambertian
    """

    l, h = imgArr[:,:,0].shape
    count = imgArr.shape[2]

    #The indices we actually want to calculate on
    notMasked = np.where(mask > 0)#np.nonzero(mask)
    plt.imshow(mask); plt.show()
    # print(imgArr[notMasked].shape)

    if len(LightDirections) == 3:
        m  = (np.linalg.inv(LightDirections) @ imgArr[notMasked].T)
    else:
        m =  (np.linalg.pinv(LightDirections) @ imgArr[notMasked].T)
    print(m.shape)
    ρ = np.linalg.norm(m,axis=0)
    # if display or True:
    #     pslib.plotImage(pslib.maskedToImage(ρ,mask))

    print(ρ.shape)
    normals = m/np.tile(ρ, (3,1))
    # normals = (1/ρ * m.T).T

    n1 = np.zeros((l,h))
    n2 = np.zeros((l,h))
    n3 = np.ones((l,h))

    n1[notMasked] = normals[0]
    n2[notMasked] = normals[1]
    n3[notMasked] = normals[2]
    _,(ax1,ax2,ax3) = plt.subplots(1,3)
    ax1.imshow(n1)
    ax2.imshow(n2)
    ax3.imshow(n3)
    plt.show()

    return(n1,n2,n3)

def PhotometricStereoNormals(imgArr,mask,LightDirections, display = False):
    """
    Uses photometric stereo to find the normals of an surface from multiple picturs of
    a still camera with different light directions. Assumes the material to be lambertian
    """

    l, h = imgArr[:,:,0].shape
    count = imgArr.shape[2]

    #The indices we actually want to calculate on
    notMasked = np.where(mask > 0)
    # plt.imshow(mask); plt.show()

    J = np.zeros((count,len(notMasked[0])))
    for i in range(count):
        Ii = imgArr[:,:,i]
        J[i,:] = Ii[notMasked]

    piS = np.linalg.pinv(LightDirections)
    m = np.dot(piS,J)


    ρ = np.linalg.norm(m,axis=0)

    if display:
        pslib.plotImage(pslib.maskedToImage(ρ,mask))

    
    normals = m/np.tile(ρ, (3,1))

    n1 = np.zeros((l,h))
    n2 = np.zeros((l,h))
    n3 = np.ones((l,h))
    ρs = np.zeros((l,h))

    n1[notMasked] = normals[0,:]
    n2[notMasked] = normals[1,:]
    n3[notMasked] = normals[2,:]
    ρs[notMasked] = ρ
    if display:
        _,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.imshow(n1)
        ax2.imshow(n2)
        ax3.imshow(n3)
        plt.show()

    ns = np.zeros((3,l,h))
    ns[0] = n1
    ns[1] = n2
    ns[2] = n3

    return(ns,ρs)

def PhotometricStereoNormalsShadowThreshold(imgArr,mask,LightDirections, display = False,threshold = 0.01):
    """
    Another version of PhotometricStereoNormals that finds the normals while ignoring points
    with no signal.
    """

    l, h = imgArr[:,:,0].shape

    #The indices we actually want to calculate on
    notMasked = np.nonzero(mask)
    
    count = len(notMasked[0])
    imgCount = imgArr.shape[2]
    m = np.zeros((count,3))
    
    
    imgNotMasked = imgArr[notMasked]
    
    individualMasks = np.zeros((count,imgCount))
    for i in range(0,imgCount):
        individualMasks[:,i] = (imgArr[:,:,i] >= threshold)[notMasked]
    
    for i in range(0,count):
        tS = LightDirections[individualMasks[i] > 0]
        tImg = imgNotMasked[i][individualMasks[i] > 0]
        m[i] = np.linalg.pinv(tS) @ tImg
        
    ρ = np.linalg.norm(m,axis=1)
    print(display)
    

    
    normals = (1/ρ * m.T)

    n1 = np.zeros((l,h))
    n2 = np.zeros((l,h))
    n3 = np.ones((l,h))
    ρs = np.zeros((l,h))

    n1[notMasked] = normals[0]
    n2[notMasked] = normals[1]
    n3[notMasked] = normals[2]
    ρs[notMasked] = ρ

    if display :
        print(ρs)
        pslib.plotImage(ρs,"albedo")
    
    if display :
        _,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.imshow(n1)
        ax2.imshow(n2)
        ax3.imshow(n3)
        plt.show()

    ns = np.zeros((3,l,h))
    ns[0] = n1
    ns[1] = n2
    ns[2] = n3

    
    return(ns,ρs)



def rotnorms(onorms,mask,rotmat):
    """
    Returns a copy of onorms that is rotated by rotMat
    """
    
    norms = onorms.copy()
    notMasked = np.nonzero(mask)
    n1 = norms[0][notMasked]
    n2 = norms[1][notMasked]
    n3 = norms[2][notMasked]
    
    rotnorms = rotmat @ np.vstack((n1,n2,n3))
    norms[0][notMasked] = rotnorms[0]
    norms[1][notMasked] = rotnorms[1]
    norms[2][notMasked] = rotnorms[2]
    return norms




def PhotometricStereoStretchSolver(imgArr,mask,LightDirections,l = None,h = None,display = False
                                   ,threshold = 0.01,nanToZero = True,useUtil = False,newNormal = False,rot90 = False, damp = 0,Rotnormals = False,rotMat = None):
    """
    Solves the photometric stereo problem for the given images and light 
    directions, by finding the surface normals and solving the coresponing
    poisson equation problem.
    
    Returns an array z of the surface height.
    """

    (normals, ρ) = PhotometricStereoNormalsShadowThreshold(imgArr,mask,LightDirections,threshold = threshold,display = display)

    if Rotnormals:
        # rotMat = rotMat = np.array(mathutils.Matrix.Rotation(-2.5*np.pi/180, 3, 'X'))
        print(rotMat)
        # rotMat = rotMat = np.array(mathutils.Matrix.Rotation(-2.5*np.pi/180, 3, 'X'))
        old = normals.copy()
        normals = rotnorms(normals,mask,rotMat)

        if display:
            _,(axes) = plt.subplots(2,3)
            axes[0,0].imshow(normals[0])
            axes[0,1].imshow(normals[1])
            axes[0,2].imshow(normals[2]) # Upper right corner
            axes[1,0].imshow(old[0])
            axes[1,1].imshow(old[1])
            axes[1,2].imshow(old[2])
            plt.show()

        
    # normals = PhotometricStereoNormalsShadow(imgArr,mask,LightDirections)
    # normals = PhotometricStereoNormalsShadowThreshold(imgArr,mask,LightDirections,threshold = 0.01)
    p = -normals[0]/normals[2]
    q = -normals[1]/normals[2]

    print(useUtil)
    print((l,h))
    if useUtil:
        z = ps_utils.unbiased_integrate(normals[0],normals[1],normals[2],mask)
    elif damp == 0:
        z = psnormal.PoissonSolverPS(normals,mask,l = l, h = h,newNormal = newNormal,rot90 = rot90)[2]
        firstVal = z[True != np.isnan(z)][0]
        z = z - firstVal
    else:
        z = psnormal.PoissonSolverNormPenalised(mask,p,q,damp,d=1,rot90 = rot90)[2]
        
    if display:
    
        cd = CentralDifference(p,q)

        _,(ax1,ax2,ax3) = plt.subplots(1,3)
        ax1.imshow(p)
        ax2.imshow(q)
        ax3.imshow(cd)
        plt.show()

        plotImage(z)
        display_surface(z,col = ρ)

    return (z,ρ)



def refractivePlaneSolver(imgArr, mask, LightDirections, planeNormal,
                          etaIn, etaOut,display = False, debug = False, newNormal = False,
                          useUtil = False, Fresnel = True,LightStretch = True, autoRotate = 0,rot90 = False, damp = 0,
                          Rotnormals = True,threshold = 0.01, SaveName = None,SaveDir = None, mayavispecs = None):
    """
    Solves the ps-problem where a refractive plane is placed in front of the camera. 


    rotate90 tells the algorithm, if the lights do not match the orientation of the image
    """
    
    rotation = np.array(mathutils.Matrix.Rotation(autoRotate*np.pi/180, 3, 'Z')) 
    LightDirections = (rotation @ LightDirections.T).T

    
    out =  pslib.refractivePlaneLightAndCameraChange(imgArr, mask, LightDirections,
                                                     planeNormal, etaIn,etaOut,debug = debug,
                                                     Fresnel = Fresnel, LightStretch = LightStretch
                                                     ,Rotnormals = Rotnormals)
    (rotLight,imgArr,stretchX,stretchY) = out

    rotLight = (lin.inv(rotation) @ rotLight.T).T

    print("rotatedLight")
    print(rotLight)
    print(f"Stretch along X axis: {stretchX}")
    print(f"Stretch along Y axis: {stretchY}")
    #FIXME: It should be the first, but i am testing
    refractedCamDir = pslib.refractLight3d(np.array([0,0,1]), planeNormal, etaIn, etaOut)[0]
    # refractedCamDir = psl
    
    rotMat = pslib.matRotateToVector(refractedCamDir,np.array([0,0,1]))

    #FIXME: does l and h need to be inverted?
    (z,ρ) = PhotometricStereoStretchSolver(imgArr, mask, rotLight, l = stretchX,
                                          h = stretchY, display = display,
                                       newNormal = newNormal, useUtil = useUtil,rot90 = rot90,
                                       damp = damp,Rotnormals = Rotnormals,rotMat = rotMat,
                                       threshold = threshold)
    


    nz = pslib.bilinearStretch(z,stretchY,stretchX)
    nρ = pslib.bilinearStretch(ρ,stretchY,stretchX)
    if display:
        plotImages([nz,nρ])
    display_surface(nz,nρ,SaveName,SaveDir,mayavispecs)
    
    return nz


def refplanesolver2(imgArr, mask, LightDirections, planeNormal,
                          etaIn, etaOut,display = False, debug = False, newNormal = False,
                          useUtil = False, Fresnel = True,LightStretch = True, autoRotate = 0,rot90 = False, damp = 0,
                          Rotnormals = True,threshold = 0.01, SaveName = None,SaveDir = None, mayavispecs = None):
    """
    A new, more correct version of the refractive plane solver.
    """

    rotation = np.array(mathutils.Matrix.Rotation(autoRotate*np.pi/180, 3, 'Z')) 
    LightDirections = (rotation @ LightDirections.T).T

    
    out =  pslib.refractivePlaneLightAndCameraChange(imgArr, mask, LightDirections,
                                                     planeNormal, etaIn,etaOut,debug = debug,
                                                     Fresnel = Fresnel, LightStretch = LightStretch
                                                     ,Rotnormals = Rotnormals)
    (rotLight,imgArr,stretchX,stretchY) = out

    rotLight = (lin.inv(rotation) @ rotLight.T).T
    
    refractedCamDir = pslib.refractLight3d(np.array([0,0,1]), planeNormal, etaIn, etaOut)[0]
    rotMat = pslib.matRotateToVector(refractedCamDir,np.array([0,0,1]))


    (normals, ρ) = PhotometricStereoNormalsShadowThreshold(imgArr,mask,rotLight,threshold = threshold,display = display)
    # plotImages([normals[0],normals[1],normals[2]])

    if Rotnormals:
        print(rotMat)
        old = normals.copy()
        print("Using rotnormals")

        normals = rotnorms(normals,mask,rotMat)
        
        if display:
            _,(axes) = plt.subplots(2,3)
            axes[0,0].imshow(normals[0])
            axes[0,1].imshow(normals[1])
            axes[0,2].imshow(normals[2]) # Upper right corner
            axes[1,0].imshow(old[0])
            axes[1,1].imshow(old[1])
            axes[1,2].imshow(old[2])
            plt.show()

    θ = -np.angle(planeNormal[0] + 1j*planeNormal[1]) + np.pi/2
    rR = sqrt(refractedCamDir[0]**2 + refractedCamDir[1]**2 )
    rz = refractedCamDir[2]
    nR = sqrt(planeNormal[0]**2 + planeNormal[1]**2 )
    nz = planeNormal[2]
    σ1 =  1 - rR**2 + rR*rz*nR/nz
    σ2 = -rR*rz - (1 - rz**2)*nR/nz
    σ = lin.norm([σ1,σ2])

    nrotMat = pslib.matRotateAlongAxis(np.array([0,0,1]),θ)

    # Note that the following roations are equvalent!
    # print(nrotMat@rotMat)

    # trotMat = pslib.matRotateToVector(nrotMat@refractedCamDir,np.array([0,0,1]))

    # print(trotMat @ nrotMat)



    onormals2 = normals.copy()


    normals = rotnorms(normals,mask,nrotMat)

    if display:
    
        print("θ: %f" % θ)
        print("rR: %f" % rR)
        print("rz: %f" % rz)
        print("nR: %f" % nR)
        print("nz: %f" % nz)
        print("σ1: %f" % σ1)
        print("σ2: %f" % σ2)
        print("σ: %f"  % σ)

    on1 = -onormals2[0]/onormals2[2]
    on2 = -onormals2[1]/onormals2[2]
    n1 = -normals[0]/normals[2]
    n2 = -normals[1]/normals[2]

    # I think this is correct
    # p = -sin(-θ)*n2 + cos(-θ)*n1*σ
    # q = cos(-θ)*n2 + n1*sin(-θ)*σ

    # Should probably be this, see report and maple
    p = -(-sin(θ)*n2*σ - cos(θ)*n1)
    q = -(-cos(θ)*n2*σ + n1*sin(θ))

    if display:
        plotImages([-n1,-n2], "-n1 and -n2")

    rmask = pslib.rotateImg(mask*1.0,θ)
    nmask = (rmask > 0)
    nmask2= pslib.rotStretchImage(mask*1.,θ,σ,1) > 0
    z = psnormal.PoissonSolverPS(normals,mask,p = p, q=q,l = 1, h = 1,newNormal = newNormal,rot90 = True,damp = damp)[2]  #
    print(z)
    firstVal = z[True != np.isnan(z)][0]
    z = z - firstVal
    z = pslib.rotStretchImage(z,θ,σ,1)
    # plotImage(z)
    # print(z)
    nρ =  pslib.rotStretchImage(ρ,θ,σ,1) 
    display_surface(z,col = nρ, SaveName = SaveName, SaveDir = SaveDir, mayavispecs = mayavispecs)
    
    return z

def ApplyLightMatrix(S,vals = [1,1,1]):
    dz = np.array([[vals[0],0,0],[0,vals[1],0],[0,0,vals[2]]])
    dh = (dz @ S.T).T
    dhNormalised = 1/lin.norm(dh,2,axis = 1).reshape((-1,1)) * dh
    return dhNormalised


def refractionTest(dir,name,planedir,etaIn,etaOut,threshold = 0.02,
                   lightlim = 7, debug = False,display = True,
                   useUtil = False, newNormal = False,verbose = True,
                   Fresnel = True, LightStretch = True, autoRotate = 0,
                   matToLighs = [1,1,1],
                   dontRefract = False, datapoint = 6, rot90 = False,
                   damp = 0,SaveName = None, SaveDir = "./ExperimentData/",
                   mayavispecs = None,rotImg = False, V2 = True):
    """
    Simple test function for refractive plane, to tidy up code
    """
    rotAng = 0
    if rotImg:
        rotAng = -np.angle(planedir[0] + 1j*planedir[1]) - np.pi/2
    imgs,imgMask,S = pslib.imgImporter(dir,name,"Δ",threshold = threshold,
                                       lightlim = lightlim,debug =False,
                                       datapoint = datapoint,SaveName = SaveName,
                                       SaveDir = SaveDir,rotAng = rotAng)
    planedir = pslib.matRotateAlongAxis(np.array([0,0,1]),rotAng) @ planedir

    
    if verbose:
        print("First image")
        plt.imshow(imgs[:,:,0]); plt.show()

        print("Image mask")
        plt.imshow(imgMask); plt.show()

        print("Light directions")
        print(S)

        print("Plane normal")
        print(planedir)
    S = ApplyLightMatrix(S,matToLighs)
    
    if V2:
        return refplanesolver2(imgs,imgMask,S,planedir,etaIn,
                                 etaOut,debug=True,display = display, newNormal = newNormal,
                                 useUtil = useUtil,Fresnel = Fresnel, autoRotate = autoRotate,
                                 rot90 = rot90, damp = damp,threshold = threshold,SaveName = SaveName, SaveDir = SaveDir,mayavispecs = mayavispecs, LightStretch = LightStretch)
    else:
        return refractivePlaneSolver(imgs,imgMask,S,planedir,etaIn,
                                 etaOut,debug=True,display = display, newNormal = newNormal,
                                 useUtil = useUtil,Fresnel = Fresnel, autoRotate = autoRotate,
                                 rot90 = rot90, damp = damp,threshold = threshold,SaveName = SaveName, SaveDir = SaveDir,mayavispecs = mayavispecs, LightStretch = LightStretch)


def degToNormal(x,y,z):
    """
    Finds the surface normal of a plane with original normal (0,0,1), rotated (x,y,z) degrees euler
    in blender.
    """
    factor = 2*np.pi/360
    n = mathutils.Vector([0,0,1])
    rot = mathutils.Euler([x*factor,y*factor,z*factor])

    n.rotate(rot)
    
    return np.array(n)

def vecToArray(x):
    return  np.array([x.x,x.y,x.z])





def pinholeMapPlane(camspecs,us,vs):

    camplace = camspecs["camplace"]
    displacement = camspecs["displacement"]

    k = camspecs["k"]
    m = camspecs["m"]
    q = camspecs["q"]


    xs = k*us + camplace[0] + displacement[0]
    ys = m*vs + camplace[1] + displacement[1]
    zs = q*vs + camplace[2] + displacement[2]

    display_mesh(xs,ys,zs)



def refplanePinholeGradientSimple(normals,us,vs,η,camspecs,useScaled = False):
    """
    Calculates the gradients required for solving the pinhole PS problem, where there is a
    refractive plane at z = 0. 
    This uses the simplifying assumption.
    
    It calculates the derivatives via the formulas
    zᵤ = η• n₁/<n,D>
    zᵥ = η• (n_1 • u • q + n₂ • (q•d₂ - m•d₃))/((d₃ + v•q) • <n,D>)

    """
    d2 = camspecs["d2"]
    camspecs["d3"] = camspecs["d3"]
    d3 = camspecs["d3"]
    k = camspecs["k"]
    m = camspecs["m"]
    q = camspecs["q"]
    n1 = normals[0]
    n2 = normals[1]
    n3 = normals[2]

    print(camspecs)

    # This is <n,D>
    nD = η* n1 * us*k + η* n2 *(d2 + m*vs) + n3 * (d3 + vs*q)

    # The common factor for both zᵤ and zᵥ
    cf = η/(nD)

    zu = cf * ( -n1*k)

    n1p = n1*us*q*k

    n2p = n2 * (q*d2 - m * d3)
    
    zv =( cf * ( n1p  + n2p ))/(d3 + vs*q)
    return (zu,zv)





def refplanePinholePS(imgArr, mask, LightDirections, etaIn,
                      etaOut,camspecs,Fresnel = True,LightStretch = True, display = False,
                      useUtil = False,useScaled = False,rot90 = True,damp = 0.,
                      SaveName = None,SaveDir = None,mayavispecs = None,threshold = 0.01):
    """
    Solves the simplified pinhole PS with a refractive plane problem.
    """

    # We start by finding the directions of the light bellow the plane
    
    # We assume that the plane i placed at z=0
    planeNormal = np.array([0,0,1])

    η = etaIn/etaOut

    out =  pslib.refractivePlaneLightAndCameraChange(imgArr, mask, LightDirections,
                                                     planeNormal, etaIn,etaOut,debug = False,
                                                     Fresnel = Fresnel,LightStretch = LightStretch,
                                                     camDirAbove = hat(camspecs["camdir"]))
    (rotLight,imgArr,stretchX,stretchY) = out


    # Our coordinates
    us,vs = pslib.pinholeCoordinates(mask.shape,camspecs,useScaled = False)

    # plotImages([us,vs])
    direcsX  =  camspecs["k"]*us
    direcsY  =  camspecs["m"]*vs + camspecs["d2"]
    direcsZ  =  camspecs["q"]*vs + camspecs["d3"]
    # plotImages([direcsX,direcsY,direcsZ])
    
    

    

    # Then, we find the normals and then the gradients
    #Find normals

    (normals,ρ) = PhotometricStereoNormalsShadowThreshold(imgArr,mask,rotLight,display = display,threshold = threshold)

    #Find gradients

    ps,qs = refplanePinholeGradientSimple(normals,us,vs,η,camspecs,useScaled = useScaled)

    # We now integrate the gradients to find the heights
    stepSize = us[0,1] - us[0,0]
    if useUtil:
        t = np.ones(ps.shape)
        

        # z1 = ps_utils.simchony_integrate(t,t,t,np.rot90(mask),np.rot90(ps)*stepSize,np.rot90(qs)*stepSize)
        z1 = ps_utils.unbiased_integrate(t,t,t,np.rot90(mask),np.rot90(ps)*stepSize,np.rot90(qs)*stepSize)
        
        # z1 = np.rot90(z1,axes = (1,0))
        z1 = z1 - z1[True != np.isnan(z1)][0]
        # plotImage(z1)
        nz2 = -np.exp(-z1)
        # plotImage(nz2)
        nz = nz2
        # nz = ps_utils.unbiased_integrate(t,t,t,mask,ps,qs)
        # nz = nz - nz[True != np.isnan(nz)][0]

    elif damp == 0:
        print("Not using ps_utils")
        A, b  = psnormal.PoissonFDEequationsWithVonNeumann(mask,ps,qs,d = stepSize,rot90 = rot90)
        nmask = np.rot90(mask)
        z = np.zeros(nmask.shape)
        # print("Scaling b by stepsize")
        # nb = b*stepSize
        vals = spsolve(A,b)
        # vals = lsq_linear(A,b,verbose= 2)
        # print(vals)
        # vals = vals.x
        print("--------------Hej--------------------")
        z[np.where(nmask > 0)] = vals
        z[nmask == 0] = np.nan
        z = z - z[True != np.isnan(z)][0]
        # plotImage(-z)
        nz = -np.exp(z)
        # plotImage(nz)
        
        # z  = psnormal.PoissonSolverNormPenalised(mask,ps,qs,damp,d = stepSize,rot90 = rot90)[2]
    else:
        # nz = PoissonSolverPS(normals,mask,ps,qs,1,1)
        #FIXME: THere should be some scaling here? Look at old report
        stepSize = us[0,1] - us[0,0]
        print("StepSize: %f" % stepSize)

        z  = psnormal.PoissonSolverNormPenalised(mask,ps,qs,damp,d = stepSize,rot90 = rot90)[2]
    
        # plotImage(-z)
        nz = -np.exp(z)
        # plotImage(nz)


    d2 = camspecs["d2"]
    d3 = camspecs["d3"]
    k = camspecs["k"]
    m = camspecs["m"]
    q = camspecs["q"]
    cz = camspecs["cz"]
 
    #FIXME: I think this is needed. Why?
    # us = us/2
    # vs = vs/2
    # plotImage(us)

    nz = np.rot90(nz,axes = (1,0))
    xs =k*us*η*nz/(q*vs + d3)
    ys = (m*vs + d2)*η*nz/(q*vs + d3)

    # xs =k*vs*η*nz/(q*us + d3)
    # ys = (m*us + d2)*η*nz/(q*us + d3)


    
    # display_mesh(xs,ys,nz)
    # display_mesh(xs2,ys2,nz2)
    
    
    if display:
        # pslib.plotImage(p)
        # pslib.plotImage(q)
        # pinholeMapPlane(camspecs,us,vs)
        plotImage(nz)
        
        _,(axes) = plt.subplots(2,2)
        axes[0,0].imshow(ps)
        axes[1,0].imshow(qs)
        axes[0,1].imshow((-normals[0]/normals[2]))
        axes[1,1].imshow(-normals[1]/normals[2])
        axes[0,0].set_title("zᵤ")
        axes[1,0].set_title("zᵥ")
        axes[0,1].set_title("-n₁/n₃")
        axes[1,1].set_title("-n₂/n₃")
        
        plt.show()

        
        _,(axes) = plt.subplots(1,3)
        axes[0].imshow(xs)
        axes[1].imshow(ys)
        axes[2].imshow(nz)
        axes[0].set_title("X values")
        axes[1].set_title("Y values")
        axes[2].set_title("Z values")
        plt.show()

    display_mesh(xs,ys,nz,ρ,SaveName = SaveName,SaveDir = SaveDir,mayavispecs = mayavispecs)
        
        


        

    return (xs,ys,nz)

def pinholeCalculateCamspecs2(camplace,focallen,width,imgshape,sign = 1):
    """
    Calculates any remaing usefull camera specs, and returns a dictionary with them
    """

    camspecs = {}

    height = imgshape[1]/imgshape[0]*width

    camdir = hat(camplace)

    displacement =  sign*camdir * focallen
    d1, d2, d3 = displacement


    if abs(camdir[0]) > 0.0001:
        raise Exception("Can't handle this yet")

    if lin.norm(camdir[0:2]) < 10**-5:
        xyvec = np.array([1,0,0])
    else:
        xyvec = hat(np.array([-camdir[1],camdir[0],0]))

    print(xyvec)
    
    #FIXME: Is this correct?
    zvec = np.cross(camdir,xyvec)

    print(zvec)
    k = -sign*xyvec[0]
    m = -sign*zvec[1]
    q = -sign*zvec[2]


    camspecs["displacement"] = displacement
    camspecs["camdir"] = camdir
    camspecs["camplace"] = camplace

    camspecs["cz"] = camplace[2]
    camspecs["d1"] = d1
    camspecs["d2"] = d2
    camspecs["d3"] = d3

    camspecs["k"] = k
    camspecs["m"] = m
    camspecs["q"] = q

    camspecs["camWidth"] = width
    camspecs["camHeight"] = height
    camspecs["focallen"] = focallen

    return camspecs


def rescalefw(focallen,width,shape):
    """
    A function for rescalling the focallength and width to match the shape of an image
    """
    nfocal = shape[0]*focallen/width
    nwidth = shape[0]
    return nfocal, nwidth


def pinholeRefractionTest(dir,name,etaIn,etaOut,focallen,width,camplace,rescale = False,useUtil = False,Fresnel = True,LightStretch = True, display = True,autoRotate = 0,rot90 = True,datapoint = 4,threshold = 0.02,lightlim = 4,damp = 0.,SaveName = None, SaveDir = None,mayavispecs = None):
    """
    A simple fuction to tidy up testing code.
    """
    R = np.array(mathutils.Matrix.Rotation(autoRotate*np.pi/180, 3, 'Z')) 

    # We start by importing the data
    (imgArr,mask,S) = pslib.imgImporter(dir,name,"Δ",threshold = threshold,lightlim = lightlim,datapoint = datapoint,SaveName = SaveName,SaveDir = SaveDir)

    if rescale:
        nfocallen, nwidth = rescalefw(focallen,width,imgArr[:,:,0].shape)
    else:
        nfocallen, nwidth = focallen, width

    # We calculate the camera specs used in the further functionsc
    camspecs = pinholeCalculateCamspecs2(R @ camplace,nfocallen,nwidth,imgArr[:,:,0].shape)

    # We find the height

    (x,y,z) = refplanePinholePS(imgArr,mask,S,etaIn,etaOut,camspecs, display = display, useUtil = useUtil,rot90 = rot90,Fresnel = Fresnel,LightStretch = LightStretch,damp = damp,SaveName = SaveName, SaveDir = SaveDir,mayavispecs = mayavispecs,threshold = threshold)

    if display:
        # display_mesh(x,y,z)
        ()
    return (x,y,z)


def orthoFitSphere(z,us = None,vs = None, x0 = None,lim = 10**-4 ,useG = True, Own = False,display = True,verbose = False):
    """
    Fits a sphere to the observed data z. 
    """
    if x0 is None:
        x0 = np.array([z.shape[0]/2,z.shape[1]/2,-30,100])
    if us is None or vs is None:
        us, vs = np.mgrid[0:z.shape[0], 0:z.shape[1]]
    mask =  np.isnan(z) == False
    args = {"us":us, "vs": vs, "zs": z,"mask": mask}
    if Own:
        def fun(x):
            return  pslib.sphereErrorFunction2(x,args = args)
        def grad(x):
            return  pslib.sphereErrorGradient(x,args = args)
        x = pslib.steepestDescent(fun,grad,x0,lim = lim,verbose = verbose)
    else:
        if useG:
            par = minimize(pslib.sphereErrorFunction2,x0 = x0,args = args,jac = pslib.sphereErrorGradient,method = "CG")
            x = par.x
            
        else:
            par = minimize(pslib.sphereErrorFunction2,x0 = x0,args = args,method = "Nelder-Mead")
            x = par.x
        print(par)
    cu = x[0]
    cv = x[1]
    cz = x[2]
    r = x[3]

    sus = us[np.where(mask)]
    svs = vs[np.where(mask)]


    vals = r**2 - (sus - cu)**2 - (svs - cv)**2
    tvals = r**2 - (us - cu)**2 - (vs - cv)**2
    nmask =  vals>= 0
    nmask2 =  tvals>= 0

    # print(sus)
    # print(tvals)
    # print(nmask2)
    # plotImages((nmask2,mask,nmask2*mask))
    # plotImage(mask*nmask2)
    
    mmask = (mask*nmask2) > 0
    rszs = np.sqrt(vals[nmask]) + cz
    szs = np.zeros(sus.shape)
    szs[nmask] = rszs

    nz = np.zeros(mask.shape)

    nz[np.where(mask)] = szs

    fval = pslib.sphereErrorFunction2(x,args)
    fgrad = pslib.sphereErrorGradient(x,args)
    fgradNorm = lin.norm(fgrad,2)

    # print(np.where(mmask))
    errors = nz[np.where(mmask)] - z[np.where(mmask)]
    print(f"Final value: %f\nFinal gradient: %s\nFinal gradient norm: %f" % (fval,str(fgrad),fgradNorm))

    print(f"MSE: %f" % (np.mean(errors*errors)))
    
    if display:
        display_surface2(nz)

    return nz
    
    
    






if (__name__ == "__main__" ) and True:
    print("Initializing tests")


    
    camplace = np.array([0,0,0.01])
    focallen = 50 * 0.001
    width = 30 * 0.001

    camspecsOrthoSphere = {"azimuth": 90,"elevation": 70,"distance": "auto"}

    threshold = 0.03

    planeNormal = np.array([0,0,1])
    # refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneX37Bounce2/","orthoSphereWithPlaneX37Bounce2",degToNormal(37,0,0),1,1.45,datapoint = 4,rot90 = True,lightlim = 5,threshold = 0.07,display = True,verbose = False,damp = 1)

    # (imgs,mask,S) = pslib.imgImporter("./data/Perspective/WithPlane/Sphere/perspSphereZ-0.02WithPlaneNoTiltL3Bounce2/","perspSphereZ-0.02WithPlaneNoTiltL3Bounce2","Δ",threshold = threshold,lightlim = 4,datapoint = 6)
    # camspecs = pinholeCalculateCamspecs2(np.array([0,0,0.03]),0.05,0.035,(500,500))

    # (imgs,mask,S) = pslib.imgImporter("./data/Perspective/WithPlane/Sphere/perspInternalSphere0.01VeryWeakTilt2Bounce/","perspInternalSphere0.01VeryWeakTilt2Bounce","Δ",threshold = threshold,lightlim = 6,datapoint = 4)
    #Not sure if placemnt is correct
    # camspecs = pinholeCalculateCamspecs2(np.array([0,-0.003,0.03]),0.05,0.035,(500,500))
    # pinholeRefractionTest(r"./data/Perspective/WithPlane/Sphere/perspSphereZ-0.02WithPlaneWithMediumTiltL3Bounce2/","perspSphereZ-0.02WithPlaneWithMediumTiltL3Bounce2",1,1.45,0.05,0.035,np.array([0,-0.01,0.03]),threshold = 0.1, lightlim = 5,display = True,datapoint = 6)


    # (imgs,mask,S) = pslib.imgImporter("./data/Perspective/WithPlane/Sphere/perspSphereZ-0.02WithPlaneWithMediumTiltL3Bounce2/","perspSphereZ-0.02WithPlaneWithMediumTiltL3Bounce2","Δ",threshold = threshold,lightlim = 4,datapoint = 6)
    # camspecs = pinholeCalculateCamspecs2(np.array([0,-0.01,0.03]),0.05,0.035,(500,500))
    
    (imgs,mask,S) = pslib.imgImporter("./data/Perspective/WithPlane/Sphere/perspSphereZ-0.02WithPlaneNoTiltL3Bounce2/","perspSphereZ-0.02WithPlaneNoTiltL3Bounce2","Δ",threshold = threshold,lightlim = 4,datapoint = 6)
    camspecs = pinholeCalculateCamspecs2(np.array([0,0,0.03]),0.05,0.035,(500,500))
    # (imgs,mask,S) = pslib.imgImporter("./data/Perspective/WithPlane/Cube/perspCubeZ-0.02WithPlaneWithMediumTiltL3Bounce2/","perspCubeZ-0.02WithPlaneWithMediumTiltL3Bounce2","Δ",threshold = threshold,lightlim = 5,datapoint = 6)
    


    nmasked = np.where(mask)

    nS,fresT = pslib.refractLight3d(S,planeNormal,1,1.45)

    (onorms,oρ) = PhotometricStereoNormalsShadowThreshold(imgs,mask,nS,display = False,threshold = threshold)
    # pslib.plotImage(oρ,vmin = np.min(oρ[nmasked]))
    pslib.plotImage(oρ,vmin = 0.29)
    plt.imshow(oρ,vmin = 0.3)
    plt.colorbar()
    plt.show()
    # pslib.plotImage(oρ)
    
    nimgs = imgs.copy()

    for i in range(0,len(S)):
        lense = pslib.lightLensingVec(nS[i],planeNormal,1/1.45)
        nimgs[:,:,i] *= lense
        print(lense)

    (norms,ρ) = PhotometricStereoNormalsShadowThreshold(nimgs,mask,nS,display = False,threshold = threshold)
    # pslib.plotImage(ρ,vmin = np.min(ρ[nmasked]))
    pslib.plotImage(ρ,vmin = 0.29)
    # pslib.plotImage(ρ)
    

    us,vs = pslib.pinholeCoordinates(mask.shape,camspecs)


    # camspecs2 = pinholeCalculateCamspecs2(np.array([0.003,0,0.03]),0.05,0.035,(500,500))

    # lensingold = pslib.pinholeLightLensing(us,vs,mask,camspecs,1/1.45)
    lensing = pslib.pinholeLightLensing2(us,vs,mask,camspecs,1/1.45)
    bimgs = nimgs.copy()
    
    for i in range(0,len(S)):
        bimgs[:,:,i] *= lensing


    
    (norms,bρ) = PhotometricStereoNormalsShadowThreshold(bimgs,mask,nS,display = False,threshold = threshold)

    # plotImages([lensingold,lensing])
    pslib.plotImage(lensing,"lensing")
    pslib.plotImage(bρ,"bρ",vmin = 0.29)
    plotImage(ndimage.uniform_filter(bρ ,5),vmin = 0.29)
    nρ = ρ*lensing
    # pslib.plotImage(nρ,vmin = np.min(nρ[nmasked]))
    pslib.plotImage(nρ,"nρ",vmin = 0.29)
    # pslib.plotImage(nρ)
    print("standard deviations")
    print(np.std(oρ[np.where(mask)])/np.mean(oρ[np.where(mask)]))
    print(np.std(ρ[np.where(mask)])/np.mean(ρ[np.where(mask)]))
    print(np.std(nρ[np.where(mask)])/np.mean(nρ[np.where(mask)]))
    print(np.std(bρ[np.where(mask)])/np.mean(bρ[np.where(mask)]))
    
    
    
    plotImage( ρ/0.3,vmin = 1.)
    # plotImage(ndimage.gaussian_filter(ρ/0.3 ,2),vmin = 1.)
    plotImage(ndimage.uniform_filter(ρ/0.31 ,3),vmin = 1.)
    
    testρ = ρ.copy()
    testρ[np.where(mask)] = 0.33/testρ[np.where(mask)]
    plotImage(testρ)

    pinholeRefractionTest(r"./data/Perspective/NoPlane/Sphere/perspSphereZ-0.02CamAbove/","perspSphereZ-0.02CamAbove",1,1,0.05,0.035,np.array([0,0,1]),threshold = 0.12, lightlim = 7)

    (imgs,mask,S) = pslib.imgImporter(r"./data/Perspective/NoPlane/Sphere/perspSphereZ-0.02CamAbove/","perspSphereZ-0.02CamAbove","Δ",threshold = 0.12,lightlim = 4,datapoint = 4)
        
    (norms,ρ) = PhotometricStereoNormalsShadowThreshold(imgs,mask,S,display = False,threshold = threshold)

    plotImage(ndimage.uniform_filter(ρ ,6),vmin = 0.72)
