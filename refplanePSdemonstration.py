# -*- coding: utf-8 -*-
"""
Created on Sun 6/9/2020

@author: Anders Samsø Birch
"""

import numpy as np
import pslib
import refplanePS
from pslib import hat
from refplanePS import refractionTest,degToNormal, pinholeRefractionTest


SimplePlane = True
Orthocam = True
PinholeCam = False
Insect = False
Problems = True
vecToArray = (lambda x : np.array([x.x,x.y,x.z]))

"""
Note: Blender has swapped x and y axis
"""
# Runs

R = pslib.matRotateToVector(np.array([1,0,0]),np.array([0,1,0]))


if Problems:
    """
    Looking at some of the more problematic cases
    """



    
    print("Sphere with large tilt")
    zsb2x225 = refractionTest(r"./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneBounce2CloseLightsTiltX225/","orthoSphereWithPlaneBounce2CloseLightsTiltX225",degToNormal(22.5,0,0),1,1.5,datapoint = 4,rot90 = True,LightStretch = False,Fresnel = False,lightlim = 10,threshold = 0.07,newNormal = True,damp = 0)
    print("Sphere with large tilt, light corrected")

    zsb2x225 = refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneBounce2CloseLightsTiltX225/","orthoSphereWithPlaneBounce2CloseLightsTiltX225",degToNormal(22.5,0,0),1,1.5,datapoint = 4,rot90 = True,LightStretch = True,Fresnel = True,lightlim = 4,threshold = 0.02,newNormal = True,damp = 0)
    
    # "./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneBounce2CloseLightsTiltX225/","orthoSphereWithPlaneBounce2CloseLightsTiltX225"
    # zsb2x225 = refractionTest(r"./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneBounce2CloseLightsTiltX225/","orthoSphereWithPlaneBounce2CloseLightsTiltX225",degToNormal(22.5,0,0),1,1.5,datapoint = 4,rot90 = True,LightStretch = True,Fresnel = True,lightlim = 4,threshold = 0.02,newNormal = True,damp = 0)
    
    print("Now with Weak dampening")
    # zsb2x225Damped = refractionTest(r"./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneBounce2CloseLightsTiltX225/","orthoSphereWithPlaneBounce2CloseLightsTiltX225",degToNormal(22.5,0,0),1,1.5,datapoint = 4,rot90 = True,lightlim = 10,threshold = 0.07,newNormal = True,damp = 0.0001)
    
    ()


if Insect:

    # Needs brighter lights
    # z = refractionTest("./data/Orthograhic/WithPlane/orthoGraphosomaSmallWithPlane/","orthoGraphosomaSmallWithPlane",np.array([0,0,1]),1,1,datapoint = 6,threshold = 0.007,rot90 = True,damp = 0)

    # Fails because Ω is not connected
    # z = refractionTest("./data/Orthograhic/NoPlane/Insect/orthoGraphosomaSmall/","orthoGraphosomaSmall",np.array([0,0,1]),1,1,datapoint = 6,threshold = 0.007,rot90 = True,damp = 0.0001)
 
    # print(np.nanmin(z))
    # nz = z.copy()
    # nz[np.isnan(nz)] = 0

    # ind = np.unravel_index(np.argmin(nz, axis=None), nz.shape)
    # print(ind) # (187, 212)
    # print(nz[ind])

    # nz[nz < 0 ] = 0
    # refplanePS.display_surface(nz)
    
    pinholeRefractionTest("./data/Perspective/NoPlane/Insect/perspGraphosomaSmallF50W35/","perspGraphosomaSmallF50W35",1,1,0.05,0.035,np.array([0.,0,0.03]),threshold = 0.01,lightlim = 3,datapoint = 6)


if SimplePlane and PinholeCam:
    print("Showing examples of the simple pinhole refraction solver")

    print("Examples without a plane")

    print("Sphere with no tilt")
    xsphere, ysphere, zsphere = pinholeRefractionTest("./data/Perspective/NoPlane/Sphere/perspSphereZ-0.02CamAbove/","perspSphereZ-0.02CamAbove",1,1,0.05,0.035,np.array([0,0,1]),threshold = 0.02, lightlim = 7)
    
    print("Cube with no tilt")

    xcube, ycube, zcube = pinholeRefractionTest("./data/Perspective/NoPlane/Cube/perspCubeZ-0.01CamAbove/","perspCubeZ-0.01CamAbove",1,1,0.05,0.035,np.array([0,0,1]))

    print("Sphere with weak tilt")

    pinholeRefractionTest("./data/Perspective/NoPlane/Sphere/perspSphereWeakTilt/","perspSphereWeakTilt",1,1,0.05,0.035,np.array([0.,-0.009,0.028]),threshold = 0.01,lightlim = 6)



    print("Sphere with no tilt with plane")

    pinholeRefractionTest("./data/Perspective/WithPlane/Sphere/perspSphereWithPlaneNoTilt/","perspSphereWithPlaneNoTilt",1,1.5,0.05,0.035,np.array([0.,0,0.03]),threshold = 0.01,lightlim = 6)


    print("Sphere with some tilt with plane")
    pinholeRefractionTest("./data/Perspective/WithPlane/Sphere/perspInternalSphere0.01VeryWeakTilt/","perspInternalSphere0.01VeryWeakTilt",1,1.5,0.05,0.035,np.array([0.,0.0003,0.03]),threshold = 0.01,lightlim = 6)

    print("Sphere with larger tilt with plane - does not work all that well")
    pinholeRefractionTest("./data/Perspective/WithPlane/Sphere/perspSphereWithPlaneWeakTilt/","perspSphereWithPlaneWeakTilt",1,1.5,0.05,0.035,np.array([0.,0.0007,0.03]),threshold = 0.01,lightlim = 6)
    
if SimplePlane and Orthocam:



    print("Examples without any plane")
    
    # refractionTest("./data/Orthograhic/NoPlane/Sphere/orthoSphereNoPlane/","orthoSphereNoPlane",np.array([0,0,1]),1,1,datapoint = 4,rot90 = True)
    # refractionTest("./data/Orthograhic/NoPlane/Cube/orthoCubeNoPlane/","orthoCubeNoPlane",np.array([0,0,1]),1,1,datapoint = 4,rot90 = True)
    # refractionTest("./data/Orthograhic/NoPlane/Skull/orthoSkullNoPlane/","orthoSkullNoPlane",np.array([0,0,1]),1,1,datapoint = 6,rot90 = True,threshold = 0.01)
    
    # print("Showing examples of simple orthographic refraction solver:")

    # print("Example No plane:")
    # print("Here there is no plane. The function is given an etaIn of 1 and etaOut of 1, resulting in no refraction.")


    print("Example with parallel plane")

    
    
    # print("Sphere with plane, with corrections")
    # refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneNoTilt/","orthoSphereWithPlaneNoTilt",np.array([0,0,1]),1,1.5,datapoint = 4,rot90 = True)

    # print("Sphere with plane, without corrections")
    # refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneNoTilt/","orthoSphereWithPlaneNoTilt",np.array([0,0,1]),1,1,datapoint = 4,rot90 = True)

    # print("Cube with plane, with corrections")
    # refractionTest("./data/Orthograhic/WithPlane/Cube/orthoCubeWithPlaneNoTilt/","orthoCubeWithPlaneNoTilt",np.array([0,0,1]),1,1.5,datapoint = 4,rot90 = True)    
    # print("Cube with plane, without corrections")    
    # refractionTest("./data/Orthograhic/WithPlane/Cube/orthoCubeWithPlaneNoTilt/","orthoCubeWithPlaneNoTilt",np.array([0,0,1]),1,1.,datapoint = 4,rot90 = True)



    print("sphere with tilted plane, corrected")
    # refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneX1125/","orthoSphereWithPlaneX1125",degToNormal(11.25,0,0),1,1.5,datapoint = 4,rot90 = True)
    # print("sphere with tilted plane, corrected, but not for tilt")
    # refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneX1125/","orthoSphereWithPlaneX1125",degToNormal(0,0,0),1,1.5,datapoint = 4,rot90 = True)
    # print("sphere with tilted plane, not corrected")
    # refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneX1125/","orthoSphereWithPlaneX1125",degToNormal(0,0,0),1,1.,datapoint = 4,rot90 = True)

    print("sphere with rather tilted plane, corrected")    
    refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneX225/","orthoSphereWithPlaneX225",degToNormal(22.5,0,0),1,1.5,datapoint = 4,rot90 = True)
    # print("sphere with rather tilted plane, corrected, but not for tilt")    
    # refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneX225/","orthoSphereWithPlaneX225",degToNormal(0,0,0),1,1.5,datapoint = 4,rot90 = True)
    # print("sphere with rather tilted plane, not corrected")    
    # refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneX225/","orthoSphereWithPlaneX225",degToNormal(0,0,0),1,1.,datapoint = 4,rot90 = True)

    
