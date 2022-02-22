# -*- coding: utf-8 -*-
"""
File for refraction unittests
"""
import pslib
import numpy as np
import numpy.linalg as lin
import math
import refplanePS
from pslib import plotImage
import matplotlib.pyplot as plt
import psNormalIntegration as psnormal



def almosteq(a,b,limit = 0.001):
    if np.isscalar(a) and np.isscalar(b):
        return abs(a-b) < limit
    else:
        return lin.norm(a - b,2) < limit

def HeightAndLength(arr):
    """
    """
    ind = np.argwhere(arr)
    hmin = np.min(ind[:,0])
    hmax = np.max(ind[:,0])
    lmin = np.min(ind[:,1])
    lmax = np.max(ind[:,1])
    return (hmax - hmin, lmax - lmin)

def test_hat():
    assert almosteq(np.array([0,0,1]),pslib.hat(np.array([0,0,1])))
    assert almosteq(np.array([0,-1,1])* 1/math.sqrt(2),pslib.hat(np.array([0,-1,1])))
    assert almosteq(np.array([0,1,0]),pslib.hat(np.array([0,2,0])))
    assert not almosteq(np.array([0,2,0]),pslib.hat(np.array([0,2,0])))
    assert not almosteq(np.array([0,1,0]),pslib.hat(np.array([0,0,1])))


def test_rotmat():

    a = np.array([0,0,1]) 
    b = np.array([1,0,0])
    R = pslib.matRotateToVector(a,b)
    c = np.array([1,1,0])
    assert almosteq(R@a,b)
    assert almosteq(R@c,np.array([0,1,-1]))

    a2 = pslib.hat(np.array([1,1,0]))
    b2 = np.array([0,0,1])
    print(np.cross(a2,b2))
    R2 = pslib.matRotateToVector(a2,b2)
    c2 = np.array([1,0,0])
    print(R2@a2)
    print(R2 @ c2)
    assert almosteq(R2@a2,b2)

    a3 = np.array([0,0,1])
    b3 = pslib.hat(np.array([1,1,0]))
    R3 = pslib.matRotateToVector(a3,b3)
    c3 = np.array([1,1,0])
    assert(almosteq(R3@ a3,b3))
    

def test_refractLight3d():
    inVec1 = np.array([0,0,1])
    normal1 = np.array([0,0,1])
    nLd1 = pslib.refractLight3d(inVec1,normal1,1,1,tol = 10**(-10))[0]

    assert(almosteq(inVec1,nLd1))

    inVec2 = pslib.hat(np.array([0,1,1]))
    normal2 = np.array([0,0,1])
    nLd2 = pslib.refractLight3d(inVec2,normal2,1,1,tol = 10**(-10))[0]

    assert(almosteq(inVec2,nLd2))

    inVec3 = pslib.hat(np.array([0,1,1]))
    normal3 = np.array([0,0,1])
    nLd3 = pslib.refractLight3d(inVec3,normal3,1,1.5,tol = 10**(-10))[0]

    #The out angle should be 28.1255
    outAng = 28.1255 * np.pi/180
    # R = pslib.matRotateToVector(normal3,inVec3)
    R = pslib.matRotateAlongAxis((np.cross(normal3,inVec3)),outAng)

    nLd3Answer = refplanePS.degToNormal(-28.1255,0,0)
    assert(almosteq(nLd3Answer,nLd3))
    assert(almosteq(R@normal3,nLd3))

    nLd32 = pslib.refractLight3d(nLd3,normal3,1.5,1,tol = 10**(-10))[0]

    #We should get the same out if we invert it
    assert(almosteq(inVec3,nLd32))


    inVec4 = pslib.hat(np.array([-2,-3,4]))
    normal4 = pslib.hat(np.array([-4,4,2]))
    assert((inVec4 @ normal4) > 0)
    nLd4 = pslib.refractLight3d(inVec4,normal4,1,1.5,tol = 10**(-10))[0]
    nLd42 = pslib.rotRefract3d(inVec4,normal4,1,1.5)
    assert(almosteq(nLd4,nLd42))


    inVec5 = np.array([inVec4,inVec3,inVec1])
    print(inVec5)
    normal5 = normal4
    nLd5 = pslib.refractLight3d(inVec5,normal5,1,1.5)[0]
    print(nLd5)
    assert(almosteq(nLd5[0],nLd4))


    inVec6 = np.array([[ 0.35260614 ,0.25620446,  0.90001566],
              [ 0.09639991, -0.29699971,  0.94999907],
              [ 0.13459931, -0.41459786,  0.89999536],
              [-0.43581708, -0.     ,     0.90003526],
              [ 0.43581708,  0.  ,        0.90003526],
              [-0.35260614,  0.25620446,  0.90001566]])
    normal6 = [0.21217766, 0. , 0.97723109]
    nLd6 = pslib.refractLight3d(inVec6,normal6,1,1.5)[0]
    maps = (lambda x: pslib.rotRefract3d(x,normal6,1,1.5))
    for i in range(0,6):
        assert(almosteq(nLd6[i],maps(inVec6[i])))
    
    ()
    
def test_refractivePlaneLightAndCameraChange():
    img0 = np.array([0])
    inVec1 = np.array([[ 0.35260614 ,0.25620446,  0.90001566],
              [ 0.09639991, -0.29699971,  0.94999907],
              [ 0.13459931, -0.41459786,  0.89999536],
              [-0.43581708, -0.     ,     0.90003526],
              [ 0.43581708,  0.  ,        0.90003526],
              [-0.35260614,  0.25620446,  0.90001566]])
    # normal1 = [0.21217766, 0. , 0.97723109]
    normal1 = pslib.hat(np.array([0,0,1]))
    refracted1 = pslib.refractLight3d(inVec1,normal1,1,1.5)[0]
    out1 = pslib.refractivePlaneLightAndCameraChange(img0,img0,inVec1,normal1,1,1.5)
    assert(lin.norm(refracted1 - out1[0]) < 0.01)

    normal2 = pslib.hat(np.array([0,1,1]))
    refracted2 = pslib.refractLight3d(inVec1,normal2,1,1)[0]
    out2 = pslib.refractivePlaneLightAndCameraChange(img0,img0,inVec1,normal2,1,1)
    print(refracted2)
    print(out2[0])
    assert((lin.norm(refracted2 - out2[0]) < 0.01))


    
def test_restretch():
    z1 = refplanePS.refractionTest("./data/refractivePlane22.5degY12.25degX/","refractivePlane22.5degY12.25degX", refplanePS.degToNormal(22.5,12.25,0),1,1.5,display = False,verbose = False)
    z2 = refplanePS.refractionTest("./data/refractivePlane22.5degY/","refractivePlane22.5degY",(refplanePS.degToNormal(22.5,0,0)),1,1.5,display = False,verbose = False)
    z3 = refplanePS.refractionTest("./data/refractivePlaneInternalSphere0.01Y45deg/","refractivePlaneInternalSphere0.01Y45deg",(refplanePS.degToNormal(45,0,0)),1,1.5,lightlim = 4,display = False,verbose = False)
    hl1 = HeightAndLength(np.isnan(z1) == 0)
    hl2 = HeightAndLength(np.isnan(z2) == 0)
    hl3 = HeightAndLength(np.isnan(z3) == 0)
    assert((max(hl1) - min(hl1))/max(hl1) < 0.02)
    assert((max(hl2) - min(hl2))/max(hl2) < 0.02)
    assert((max(hl3) - min(hl3))/max(hl3) < 0.02)




def test_pinholeCoordinates():

    camspecs = {"camWidth" : 0.1, "camHeight" : 0.3}
    arr1 = np.ones((2,5,1))
    out1 = pslib.pinholeCoordinates(arr1,camspecs)
    assert almosteq(out1[0][0],np.array([-0.04,-0.02,0,0.02,0.04]))
    assert almosteq(out1[1][:,0],np.array([-0.075,0.075]))

def test_pinholeHeight():
    camspecs = {"k" : 1,"m" : 3,"q" : 2, "d1" : 0, "d2" : -1, "d3" : 1}

    vs = np.array([[1,2],[-1,5]])
    us = np.array([[-4,3],[2,1]])
    eta = 1/1.5

    h1 = pslib.pinholeHeightSimplified(us,vs,eta,camspecs)
    print(h1)
    ans1 = np.array([[3,5],[-1,11]])
    
    assert almosteq(h1,ans1)


def test_camspecs():
    #First, a simple test
    camplace1 = np.array([0,0,1])
    focallen1 = 0.05
    width1 = 0.01
    imgshape1 = (500,500)
    cs1 = refplanePS.pinholeCalculateCamspecs2(camplace1,focallen1,width1,imgshape1)
    assert almosteq(focallen1,cs1["d3"])
    assert almosteq(0,cs1["d2"])
    assert almosteq(0,cs1["d1"])
    assert almosteq(-1.,cs1["k"]) 
    assert almosteq(-1,cs1["m"]) 
    assert almosteq(0,cs1["q"]) 
    assert almosteq(width1,cs1["camWidth"]) 
    assert almosteq(width1,cs1["camHeight"]) 
    assert almosteq(focallen1,cs1["focallen"]) 
    

    camplace2 = np.array([0,1,1])
    focallen2 = 0.5
    width2 = 0.1
    imgshape2 = (500,600)
    cs2 = refplanePS.pinholeCalculateCamspecs2(camplace2,focallen2,width2,imgshape2)
    assert almosteq(-focallen2/np.sqrt(2),cs2["d3"])
    assert almosteq(-focallen2/np.sqrt(2),cs2["d2"])
    assert almosteq(0,cs2["d1"])
    assert almosteq(-1.,cs2["k"]) 
    assert almosteq(-1/np.sqrt(2),cs2["m"])
    assert almosteq(1/np.sqrt(2),cs2["q"]) 
    assert almosteq(width2,cs2["camWidth"]) 
    assert almosteq(width2*600/500,cs2["camHeight"]) 
    assert almosteq(focallen2,cs2["focallen"]) 

    
    camplace3 = np.array([0,-1,1])
    focallen3 = 0.2
    width3 = 0.1
    imgshape3 = (600,500)
    cs3 = refplanePS.pinholeCalculateCamspecs2(camplace3,focallen3,width3,imgshape3)
    assert almosteq(-focallen3/np.sqrt(2),cs3["d3"])
    assert almosteq(focallen3/np.sqrt(2),cs3["d2"])
    assert almosteq(0,cs3["d1"])
    assert almosteq(1.,cs3["k"]) 
    assert almosteq(1/np.sqrt(2),cs3["m"]) 
    assert almosteq(1/np.sqrt(2),cs3["q"]) 
    assert almosteq(width3,cs3["camWidth"]) 
    assert almosteq(width3*500/600,cs3["camHeight"]) 
    assert almosteq(focallen3,cs3["focallen"]) 


    #The problem lies in this function
    assert False
    ()

    
    
def test_normintegration():

    mask = np.ones((5,5))
    mask[0,:] = 0
    mask[-1:,:] = 0
    mask[:,0] = 0
    mask[:,-1:] = 0
    print(mask)

    A,b = psnormal.PoissionPDequationWithNormPenalty(mask,mask,mask,0.5,rot90 = True)

    print(A.toarray())
    print(b)
    (nA,nB,nz) = psnormal.PoissonSolverNormPenalised(mask,mask,mask,0.001,rot90=True)
    print(nA.toarray())
    print(nB)
    print(nz)
    plotImage(nz)
    assert False

    ()
