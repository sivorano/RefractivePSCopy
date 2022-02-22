# -*- coding: utf-8 -*-
"""
Created on Wen 5/6/2020

@author: Anders Samsø Birch

A library for genereal functions used for PS and refractivePS
"""

import numpy as np
import numpy.linalg as lin
import mathutils
import matplotlib.pyplot as plt
import math

# from mayavi import mlab
from math import sin, cos,sqrt,floor, tan
from scipy.spatial.transform import Rotation as Rot
import PIL
import PIL.Image
from os import listdir
import shutil
import skimage.io


def hat(vec):
    """
    Returns vec normalized, that is, with length 1
    """
    return vec/lin.norm(vec)

def hatvecs(mat):
    return mat/np.array([lin.norm(mat,2,axis = 1)]).T
    
def matRotateAlongAxis(ax,θ):
    """
    Returns a matrix M which corresponds to rotating along axis *ax* with angle
    *θ*.
    """

    rotvec = hat(ax) * θ

    M = Rot.from_rotvec(rotvec).as_matrix()

    return M


def matRotateToVector(fromVec,toVec):
    """
    Returns a matrix M rotates *fromVec* to *toVec* along axis fromVec×toVec
    """
    
    limit = 10**-5
    # Takes care of the edge case where rotax would be undefined (the angle
    # will be so small that this doesn't really matter)
    if lin.norm(np.cross(fromVec,toVec)) < limit:
        rotax = fromVec
    else:
        rotax = hat(np.cross(fromVec,toVec))

    θ = np.arccos( fromVec@toVec) # angle between vectors
    rotMat = matRotateAlongAxis(rotax,θ)
    return rotMat

def rotateImg(img,rotAng):
    """
    Uses pillow to return a rotated verison of img using bilinear interpolation
    """
    pimg = PIL.Image.fromarray(img) 
    rotimg = pimg.rotate(rotAng*180/np.pi,resample = PIL.Image.BILINEAR)
    return np.array(rotimg)


def bilinearStretch(arr,xstretch,ystretch):
    """
    Uses pillow to return a stretched verison of arr using bilinear interpolation
    """
    pimg = PIL.Image.fromarray(arr)
    pimg = pimg.resize((math.floor(arr.shape[0]*ystretch),
                        math.floor(arr.shape[1]*xstretch)),
                       resample = PIL.Image.BILINEAR)

    return np.array(pimg)

def rotStretchImage(img,rotAng,xs,ys):
    """
    Returns an img that has been rotated, stretched and then rerotated
    """

    m,n = img.shape
    pimg = PIL.Image.fromarray(img)
    # pimg.show()
    padded = PIL.Image.new(pimg.mode, (m*3,n*3),color = np.nan)
    padded.paste(pimg,(m,n))
    # padded.show()
    paddedarr = np.array(padded)
    rotarr = rotateImg(paddedarr,rotAng)
    stretched = bilinearStretch(rotarr,xs,ys)
    print(stretched.shape)
    backrot = rotateImg(stretched,-rotAng)
    return backrot[floor(m*xs*ys):floor(m*xs*2*ys),floor(n*ys*xs):floor(2*n*ys*xs)]


def imgImporter(path,filename,seperator,threshold = 0.02,lightlim = 3,debug = False,datapoint = 6,
                SaveName = None,SaveDir = "./ExperimentData/",rotAng = 0):
    """
    Function for importing sythetic image data created via blender.
    """

    imgs = []
    S = []
    names = listdir(path)
    names.sort()
    first = True
    for name in names:
        if first and SaveName is not None:
            shutil.copy2(path + name, SaveDir + SaveName + ".png")
        # print(name)
        if debug:
            print("imgImporter debug: name ->")
            print(name)
            
        if str.startswith(name,filename):
            img = skimage.color.rgb2gray(skimage.io.imread(path + name))
            
            print(name)
            print(filename)
            vec = np.array(eval(str.split(name,seperator)[datapoint]))
            vec = vec/np.linalg.norm(vec)
            if rotAng == 0:
                imgs.append(img)
                S.append(vec)
            else:
                # pimg = PIL.Image.fromarray(img) 
                rotimg = rotateImg(img,rotAng)
                imgs.append(np.array(rotimg))
                S.append(matRotateAlongAxis(np.array([0,0,1]),rotAng) @ vec)
        first = False

    imgs = np.array(imgs)
    print(f"imgs shape: %s" % str(imgs.shape))
    
    imgs = np.moveaxis(imgs,0,2)
    
        
    S = np.array(S)

    imgMasks = (imgs > threshold).astype(int)
    imgMask = (np.sum(imgMasks,axis = 2) >= lightlim).astype(int)

    return (imgs,imgMask,S)



def plotImage(img,title = "",vmin = None,vmax = None):
    """
    Plots the input image using matplotlib
    """
    plt.imshow(img,vmin = vmin,vmax = vmax)
    plt.title(title)
    plt.show()
    return

def plotImages(imgList, title = "",vmin = None,vmax = None):
    """
    Plots multiple images using matplotlib
    """
    n = len(imgList)
    _,(axes) = plt.subplots(1,n)
    for i in range(0,n):
        axes[i].imshow(imgList[i],vmin = vmin,vmax = vmax)
    plt.title(title)
    plt.show()
    return


def maskedToImage(arr,mask):
    """
    Takes an masked array arr and returns arr restored
    """
    notMasked = np.nonzero(mask)
    newImage = np.zeros(mask.shape)
    newImage[notMasked] = arr
    return newImage



def FresnelReflectanceCalculator(etaIn,etaOut,InAngle):
    """
    Calculates the amount of light that is reflected acording to fresnels law.
    Assumes that the light is unpolarised.
    """
    OutAngle = np.arcsin(etaIn/etaOut*np.sin(InAngle))
    etaInv = etaOut/etaIn
    Rp = ((etaInv* cos(InAngle) - cos(OutAngle))/ (etaInv* cos(InAngle) + cos(OutAngle)))**2
    Rs = ((cos(InAngle) - etaInv*cos(OutAngle))/ (cos(InAngle) + etaInv*cos(OutAngle)))**2
    
    # Rp =  (tan(InAngle - OutAngle)/tan(InAngle + OutAngle))**2
    # Rs =  (sin(InAngle - OutAngle)/sin(InAngle + OutAngle))**2

    return (Rp + Rs)/2


def normalToXYplaneDecomp(n,orthogonal = False):
    """
    Finds vectors x= [a,0,b] and y=[0,c,d] which spans the associated plane of n.
    Assumes that neither n[0] nor n[1] is 0, and will have a > 0, c>0.
    If orthogonal is set, will instead find y = [c,d,e] st. x•y=n*y=0 (y = ±x×n)
    """
    n1, n2, n3  = n

    # n₁ + c₁n₃ = 0 ⇒ c₁=-n₁/n₃
    c1 = -n1/n3

    c2 = -n2/n3

    x = np.array([1,0,c1])
    x = hat(x)
    if 0 > x[0]:
        x = -x
    # y = hat(y)
    if orthogonal:
        y = np.cross(n,x)
    else:
        y = np.array([0,1,c2])
    y = hat(y)
    
    if 0 > y[1]:
        y = -y
    return (x,y)


def rotRefract3d(inVec,normal,etaIn,etaOut):
    """
    Uses rotations to reract light
    """
    InAngle = np.arccos(normal @ inVec)
    OutAngle = np.arcsin (np.sin(InAngle)* etaIn/etaOut)
    R = matRotateAlongAxis(np.cross(normal,inVec),OutAngle)
    return R @ normal

def rot3dvec(inVec,normal,etaIn,etaOut):
    """
    Uses the following theory to refract light
    https://graphics.stanford.edu/courses/cs148-10-summer/docs/2006--degreve--reflection_refraction.pdf
    """
    eta = etaIn/etaOut
    ni = -inVec
    θi = np.arccos(np.dot(ni,normal))
    θt = np.arcsin(np.sin(θi)* etaIn/etaOut)
    t = eta*ni + (eta*cos(θi)- sqrt(1 -(sin(θt))**2))*normal
    return -t
    
def rot3dvec2(inVec,normal,etaIn,etaOut):
    """
    Uses the following theory to refract light
    https://www.starkeffects.com/snells-law-vector.shtml
    """
    eta = etaIn/etaOut
    n = normal
    i = -inVec

    v1 = eta*np.cross(n,np.cross(-n,i))
    cp = np.cross(n,i)
    v2 = n*sqrt(1 - eta**2 * np.dot(cp,cp))

    t = v1 -v2
    return -t
    

def refractLight3d(inVec,normal,etaIn,etaOut, tol = 10**(-5), debug = False,Fresnel = False):
    """ 
    Calculates the outgoing direction of light with direction *inVec*
    going from a material with refraction index etaIn, to a material with
    surface normal *normal* and refractive index etaOut
    
    *normal* is assumed  to be an unit vector (length 1), as is *inVec*.
    If *inVec* is sufficiently close to ±*normal*, up to a norm difference of
    tol, inVec is returned  
    """

    #In case of multiple vectors, the function is recursivly applied
    if type(inVec) == list:
        print(inVec)
    if len(inVec.shape) > 1:
        outVecs = []
        trans = []
        for i in range(0,inVec.shape[0]):
            ref, T = refractLight3d(inVec[i],normal,etaIn,etaOut,Fresnel = Fresnel)
            outVecs.append(ref)
            trans.append(T)
        return (np.array(outVecs),np.array(trans))

    # In the case of normal = ± inVec, the vector will just output invec
    if tol > lin.norm(normal - inVec) or tol > lin.norm(inVec - normal):
        if Fresnel:
            T = FresnelReflectanceCalculator(etaIn, etaOut, 0.)
        else:
            T = None
        return [inVec,T]

    if  lin.norm(normal @ inVec) < tol:
        print(normal)
        print(inVec)
        raise Exception("normal and inVec are orthogonal")

    
    inVec = hat(inVec)
    normal = hat(normal)
    
    # We start by creating a new basis, such that applying Sneel's law is
    # easy    

    vec1 = normal
    vec2 = inVec - (inVec @ normal) * normal 
    vec2 = vec2/lin.norm(vec2)
    vec3 = hat(np.cross(vec1,vec2)) # hat should not be nesecary due to vec1 and vec2 is ⊥
    
    coordMat = np.array([vec1,vec2,vec3]) # Matrix from E to new basis B
    reCoordMat = coordMat.T #Inverse is transpose by ortonormality of B
    
    nNormal = hat(coordMat @ normal)
    nInVec = hat(coordMat @ inVec)
    
    InAngle = np.arccos(nNormal @ nInVec)
    OutAngle = np.arcsin (np.sin(InAngle)* etaIn/etaOut)

    if debug:
        print("InAngle {}".format(InAngle))
        print("OutAngle {}".format(OutAngle))
    
    rot = np.array(mathutils.Matrix.Rotation(OutAngle, 3, 'Z')) # Correct angle and axis?
    #FIXME: Should this be nInVec?
    newLightDir = hat(reCoordMat @ (rot @ nNormal)) 

    if Fresnel:
        T = 1 - FresnelReflectanceCalculator(etaIn,etaOut,InAngle)
    else:
        T = None
    return [newLightDir,T]
    




def nanToMin(z):
    """
    Returns a copy of *z* where the nan values are replaced with the min
    value of z.
    """
    kz = z.copy()
    kz = np.nan_to_num(kz,0)
    nMin = np.min(kz)
    nz = z.copy()
    nz = np.nan_to_num(nz,nan = nMin)
    return nz


#FIXME: is wrong, look at francois theory
def planeStrech(planeNormal, upperDir, refractedDir):
    """
    Finds how the stretch factors in the x and y directions resulting from
    the camera with direction *upperDir* above plane and *refractedDir* bellow plane
    having light bended by the plane with planenormal *planeNormal*
    
    Finds how much a plane defined by normal *upperDir* gets stretched by being
    refracted by passing through a plane with normal *planeNormal*, and bending with
    direction *refractedDir*
    """
    # We find a vector decomp of the new camera direction plane
    # we assume that n3 is not 0


    #Vectors spanning the plane deffined by upperDir
    upperPlaneXvec, upperPlaneYvec = normalToXYplaneDecomp(upperDir)
    #Vectors spanning the refractive plane
    oPx, oPy = normalToXYplaneDecomp(planeNormal)

    
    #Vectors spanning the plane deffined by refractedDir
    nPx, nPy = normalToXYplaneDecomp(refractedDir)


    colFactorX = 1/(upperPlaneXvec@oPx)
    colFactorY = 1/(upperPlaneYvec@oPy)
    
    #Where the light (1,0) along the upperDir plane collides with the refractive plane
    oColX = oPx*colFactorX
    oColY = oPy*colFactorY

    
    #We project these points down on the refractedDir plane
    stretchX = oColX @ nPx
    stretchY = oColY @ nPy

    return (stretchX, stretchY)

def lightLensing(θ,η,V2 = False):
    """
    Returns the amount light gets stretched when entering a new material with the etaIn/etaOut = η
    at angle θ
    """
    
    φ = np.arcsin(η*sin(θ))

    si = np.array([sin(-θ),cos(-θ)])
    sih = np.array([si[1],-si[0]])
    so = np.array([sin(-φ),cos(-φ)])
    # 
    # print(si)
    # print(sih)
    # print(so)
    #    si + sih - r * si = [x,0]
    #  r


    # r = (si[0] + sih[0])/si[0]
    r = (si[1] + sih[1])/si[1]
    # print(r)

    x = si[0] + sih[0] - r*si[0]
    # print(x)
    # print(f"x: %f" % x)

    vec = so * x*so[0] - np.array([x,0])
    # print(x*so[0])

    if V2:
        vec2 = -so*(1/η)*(r - 1) - np.array([x,0])
        print(-so*(1/η)*(r - 1))
        print(vec2)
        return lin.norm(vec2)
    # print(vec)

    return lin.norm(vec)

def lightLensingVec(invec,planeVec,η,V2 = False):
    """
    Returns the amount light gets stretched when entering a new material with the etaIn/etaOut = η
    with *invec* being the ingoing vector and *planeVec* being the surfac enormal
    """


    
    θ = np.arccos(np.dot(hat(invec),hat(planeVec)))
    # print(f"θ: %f" % θ)
    return  lightLensing(θ,η,V2)



#FIXME: No longer changes camera, change name
def refractivePlaneLightAndCameraChange(imgArr, mask, LightDirections, planeNormal,
                                        etaIn,etaOut,debug = False, Fresnel = True,
                                        LightStretch = True,camDirAbove = np.array([0,0,1]),
                                        Rotnormals = False):
    """
    """
    dotprod = LightDirections @ planeNormal
    for i in range(0,len(dotprod)):
        if dotprod[i] < 0:
            raise Exception("Light {} ({}) is not above the plane".format(i,LightDirections[i]))
    if planeNormal[2] <= 0:
        raise Exception("The plane normal {} must point opwards".format(planeNormal))
    

    # First, we need to find the new light directions, as they are changed by the
    # refractive plane

    
    (refractedLD,lightToObjTransmisionRate) = refractLight3d(LightDirections, planeNormal, etaIn, etaOut,Fresnel = Fresnel)


    # We also need to find the direction light would take from the camera

    #FIXME: Swap etain and etaOut? proably not
    refractedCamDir = refractLight3d(camDirAbove, planeNormal, etaIn, etaOut)[0]
    # refractedCamDir = refractLight3d(camDirAbove, planeNormal, etaOut, etaIn)[0]
    objToCamTransmisionRate = refractLight3d(refractedCamDir, planeNormal, etaOut, etaIn,Fresnel = Fresnel)[1]

    if debug:
        print("refractedCamDir {} {}".format(refractedCamDir, lin.norm(refractedCamDir)))


    #Is this stretch correct, or deos it need to be inverted?
    stretchX, stretchY = planeStrech(planeNormal,camDirAbove,refractedCamDir)
    


    nimgArr = imgArr.copy()

    
    if LightStretch:
        mns = []
        mnss = []
        for i in range(0,imgArr.shape[2]):
            # print("Streching Light")
            stretchXlight, stretchYlight = planeStrech(planeNormal,LightDirections[i],refractedLD[i])
            s2 = lightLensingVec(LightDirections[i],planeNormal,etaIn/etaOut)
            # print("Stretch1: %f   Stretch2: %f" % (stretchXlight * stretchYlight, s2))

            angle = np.dot(hat(LightDirections[i]),hat(planeNormal))
            # print("LightDirection: %s  - angle %f" % (LightDirections[i],angle))
            mns.append(np.mean( np.sort( nimgArr[:,:,i].flatten() )[-5:]))
            # nimgArr[:,:,i] *= (stretchXlight * stretchYlight)
            # print(etaIn/etaOut)
            nimgArr[:,:,i] *= s2
            mnss.append(np.mean( np.sort( nimgArr[:,:,i].flatten() )[-5:]))
        # print(mns)
        # print(mnss)
        # print(np.std(mns))
        # print(np.std(mnss))
    if Fresnel:
        for i in range(0,imgArr.shape[2]):
            nimgArr[:,:,i] *= 1/lightToObjTransmisionRate[i]
            ()

    # As the light need to be given in directions relative to the refracted cam dir, we rotate
    # the light sources
    #FIXME: Should we rotate in the oposite direction
    # Yes i think, write in repport
    # rotMat = matRotateToVector(camDirAbove,refractedCamDir)
    rotMat = matRotateToVector(refractedCamDir,camDirAbove)
    print(rotMat)
    # rotMat = matRotateToVector(refractedCamDir,np.array([0,0,1]))
    #FIXME: Testing if should be applied to normals
    if Rotnormals:
        rotLight = refractedLD
    else:
        rotLight = (rotMat @ refractedLD.T).T

    
    return (rotLight,nimgArr,stretchX,stretchY)


def pinholeCoordinates(shape,camspecs, useScaled = False):
    """
    Finds arrays us and vs representing the position on the camera plane.
    """

    width = camspecs["camWidth"]
    height = camspecs["camHeight"]


    n,m = shape # n is the height in pixels, m is the width

    cN = (n - 1)/2
    cm = m/2

    ArrN = (np.arange(-(n-1)/2,(n-1)/2 + 1,1))*height/n
    ArrM = (np.arange(-(m-1)/2,(m-1)/2 + 1,1))*width/m

    vs = np.tile(ArrN,(m,1)).T
    us = np.tile(ArrM,(n,1))

    vs = np.flip(vs)
    return (us,vs)

def pinholeXYfromZ(camspecs,z):
    """
    Finds the X and Y coordiates of a surface, when z is the depths observed from a pinhole
    camera with specs *camspecs*    
    """
    us,vs = pinholeCoordinates(z.shape,camspecs)
    d2 = camspecs["d2"]
    d3 = camspecs["d3"]
    m = camspecs["m"]
    q = camspecs["q"]

    
    
    xs = us*η*nz/(q*vs + d3)
    ys = (m*vs + d2)*η*nz/(q*vs + d3)

    return xs,ys


def pinholeHeightSimplified(us,vs,eta,camspecs):
    """
    The height of the simplified D vector in the refractive pinhole case: (d3 + vs * q)
    Only good at low angles, as cos(x) ≈ 1 for x close to 0
    """

    d3 = camspecs["d3"]
    q = camspecs["q"]

    return (d3 + vs*q)


def pinholeHeightTrue(us,vs,eta,camspecs):
    """
    
    """

    d1 = camspecs["d1"] # Should be 0
    d2 = camspecs["d2"]
    d3 = camspecs["d3"]
    k = camspecs["k"]
    m = camspecs["m"]
    q = camspecs["q"]


    dv1 = (k*us + d1)
    dv2 = (m*vs + d2)
    dv3 = (q*vs + d3)

    
    dvN = dv1**2 + dv2**2 + dv3**2

    norm = np.sqrt(dvN - (eta**2) * (dv1**2 + dv2**2 ))

    return norm




def pinholeDirections(us,vs,camspecs):
    """
    Returns an array with the x, y and z directions of the pinhole camera.

    """
    direcsX  =  camspecs["k"]*us
    direcsY  =  camspecs["m"]*vs + camspecs["d2"]
    direcsZ  =  camspecs["q"]*vs + camspecs["d3"]

    direcs = np.array([direcsX,direcsY,direcsZ])
    plotImages(direcs,title ="directions")

    return direcs

def pinholeLightLensing(us,vs,mask,camspecs,η):
    """
    Returns the outgoinglight lensing factorts for each point on the pinhole camera picture
    """
    # We can calculate these the normal way, and then take the inverse!
    # vs = np.flip(vs)

    direcs = pinholeDirections(us,vs,camspecs)
    print(direcs.shape)

    #we must normalise the directions
    normDirecs = direcs/lin.norm(direcs,axis = 0)

    angles = np.arccos(normDirecs[2])
    print(angles.shape)

    # mangles = angles[np.where(mask)]
    
    # mapfun = (lambda θ: lightLensing(θ,η))
    # lightlensefun = np.vectorize(mapfun)
    # lensing = lightlensefun(mangles)
    # lensarr = np.zeros(mask.shape)
    # lensarr[np.where(mask)] = lensing

    mangles = angles.flatten()
    
    mapfun = (lambda θ: lightLensing(θ,η))
    lightlensefun = np.vectorize(mapfun)
    lensing = lightlensefun(mangles)
    lensarr = lensing.reshape(mask.shape)

    return lensarr
    
def pinholeLightLensing2(us,vs,mask,camspecs,η):
    """
    Returns the outgoinglight lensing factorts for each point on the pinhole camera picture
    """
    # We can calculate these the normal way, and then take the inverse!
    # vs = np.flip(vs)

    direcs = pinholeDirections(us,vs,camspecs)
    print(direcs.shape)

    #we must normalise the directions
    normDirecs = direcs/lin.norm(direcs,axis = 0)

    
    angles = np.arccos(normDirecs[2])
    anglesBellow = np.arcsin(η*np.sin(angles))

    plotImages([angles*180/np.pi,anglesBellow*180/np.pi],"angles")
    
    print(angles.shape)

    mangles = anglesBellow.flatten()
    
    mapfun = (lambda θ: lightLensing(θ,1/η))
    lightlensefun = np.vectorize(mapfun)
    ffun = (lambda θ: 1 - FresnelReflectanceCalculator(η,1,θ))
    fresnelfun = np.vectorize(ffun)
    
    lensing = lightlensefun(mangles)
    lensarr = lensing.reshape(mask.shape)

    frenel = fresnelfun(mangles)
    fresnelarr =  frenel.reshape(mask.shape)

    return lensarr


def pinholeLightAdjust():
    """
    """
    

def backtrace(abar,r,c,fun,grad,x,p):
    """
    port of old matlab code
    function [a] = backtrace(abar,r,c,f,df,x,p)
    k = 0;
    a = abar;
    while (f(x + a*p) > f(x) + c*a* transpose(df(x))*p)
        a = r*a;
        k = k + 1;
    end
    %f(x + a*p);
    %f(x) + c*a* transpose(df(x))*p;
    end
    """
    a = abar
    k = 0
    while fun(x + a*p) > fun(x) + c*a* np.dot(grad(x),p):
        a = r*a
        k = k +1
    return [a,k]

def steepestDescent(fun,grad,x0,lim,verbose = True,iterLim = 500,convLim = 10**(-6)):
    """
    Port of old matlab code
    
    """
    min = x0
    x = x0
    i = 0
    als = []
    while lim < lin.norm(grad(x)) and (i < iterLim and  not (i > 20 and sum(als[-10:]) < convLim)):
        p = - grad(x) # our descent direction
        (a,k) = backtrace(1,0.5,0.001,fun,grad,x,-grad(x))
        als.append(a)
        x = x + a*p
        # if verbose and (i % 10) == 0:
        #     print(i)
        #     print(x)
        #     print(a)
        #     print(p)
        #     print(lin.norm(grad(x)))
        #     print( sum(als[-10:]))
        i = i + 1

    if verbose:
        if i > iterLim:
            print("Stopped due to reaching iteration limit")
        if (i > 20 and sum(als[-10:]) < convLim):
            print("Stopped due to low convergence rate")
        print(f"Iterations: %f" % i)
        print(f"Final point: %s" % str((x)))
        print(f"Final value: %s" % str(fun(x)))
        print(f"Final gradient: %s" % str(grad(x)))
    return x


def sphereErrorFunction(mask,us,vs,zs,r,cu,cv,cz):
    """
    Returns the mean squared error of a sphere wrt. *zs*, where the error is estimated
    as the dirence between what is observed and the circle formula.
    """
    count = np.sum(mask)


    isnotnat = np.where(mask)

    sus = us[np.where(mask)]
    svs = vs[np.where(mask)]
    szs = zs[np.where(mask)]
    
    errors = np.sqrt((sus - cu)**2 + (svs - cv)**2 + (szs - cz)**2) - r


    return np.mean(errors*errors)
    
def sphereErrorFunction2(x,args):
    """
    A modified version of sphereErrorFunction for easier usage in NO algorithms
    """
    cu = x[0]
    cv = x[1]
    cz = x[2]
    r = x[3]
    us = args["us"]
    vs = args["vs"]
    zs = args["zs"]
    mask = args["mask"]
    
    return sphereErrorFunction(mask,us,vs,zs,r,cu,cv,cz)


def sphereErrorGradient(x,args):
    """
    Returns the gradient to the error function descriped above
    """
    cu = x[0]
    cv = x[1]
    cz = x[2]
    r = x[3]
    us = args["us"]
    vs = args["vs"]
    zs = args["zs"]
    mask = args["mask"]
    count = np.sum(mask)

    sus = us[np.where(mask)]
    svs = vs[np.where(mask)]
    szs = zs[np.where(mask)]


    square = np.sqrt((sus - cu)**2 + (svs - cv)**2 + (szs - cz)**2)

    firstterm =  2*(square - r)

    dx = np.mean(firstterm* -2*(sus-cu)/square)
    dy = np.mean(firstterm* -2*(svs-cv)/square)
    dz = np.mean(firstterm* -2*(szs-cz)/square)
    dr = np.mean(firstterm* -2)

    return np.array([dx,dy,dz,dr])
    
    
