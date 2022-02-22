import refplanePS
import numpy as np
from refplanePS import degToNormal
import pslib
from pslib import plotImage,plotImages


#FIXME: These actually have etaOut of 1.45 - change this
etaIn = 1
etaOut = 1.45


"""
Ortho Sphere
"""

# camspecsOrthoSphere = {"azimuth": 90,"elevation": 70,"distance": "auto"}
camspecsOrthoSphere = {"azimuth": 90,"elevation": 70,"distance": 10}

orthoSphereNP = refplanePS.refractionTest(r"./data/Orthograhic/NoPlane/Sphere/orthoSphereNoPlane/","orthoSphereNoPlane",degToNormal(0,0,0),1,1,datapoint = 4,rot90 = True,lightlim = 6,threshold = 0.15,SaveName = "orthoSphereNoPlane",SaveDir = r"./ExperimentData/Sphere/",mayavispecs = camspecsOrthoSphere,display = False,verbose = False)
# yosntzFitS = refplanePS.orthoFitSphere(orthoSphereNP,x0 = np.array([250,250,-65,144]),Own = True,display = False)


orthoSphereWPNT = refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneNoTiltLight7Z0.3/","orthoSphereWithPlaneNoTiltLight7Z0.3",degToNormal(0,0,0),1,etaOut,datapoint = 4,rot90 = True,lightlim = 10,threshold = 0.07,SaveName = "orthoSpherePlaneParallel",SaveDir = r"./ExperimentData/Sphere/",mayavispecs = camspecsOrthoSphere,display = False,verbose = False)


#FIXME: THis and bellow need same tilit
orthoSphereWPNTB2 = refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneNoTiltLight7Z0.3Bounce2/","orthoSphereWithPlaneNoTiltLight7Z0.3Bounce2",degToNormal(0,0,0),1,etaOut,datapoint = 4,rot90 = True,lightlim = 5,threshold = 0.07,SaveName = "orthoSpherePlaneParallelBounce2",SaveDir = r"./ExperimentData/Sphere/",mayavispecs = camspecsOrthoSphere,display = False,verbose = False)

orthoSphereWPWTB2 =  refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneBounce2CloseLightsTiltX5/","orthoSphereWithPlaneBounce2CloseLightsTiltX5",degToNormal(5,0,0),1,etaOut,datapoint = 4,rot90 = True,lightlim = 10,threshold = 0.07,SaveName = "orthoSpherePlaneTilt2Bounce",SaveDir = r"./ExperimentData/Sphere/",mayavispecs = camspecsOrthoSphere,display = False,verbose = False)


"""
Fit spheres
# """
yosntzFitS = refplanePS.orthoFitSphere(orthoSphereNP,x0 = np.array([250,250,-65,144]),Own = True,display = False)

yosntzFitS = refplanePS.orthoFitSphere(orthoSphereWPNT,x0 = np.array([250,250,-26,150]),Own = True,display = False)

# # plotImage(oswtz)
oswtzFitS = refplanePS.orthoFitSphere(orthoSphereWPNTB2,x0 = np.array([250,250,-10,140]),Own = True,display = False)

# # plotImage(oswtb2z)
oswtb2zFitS = refplanePS.orthoFitSphere(orthoSphereWPWTB2,x0 = np.array([255,250,-27,145]),Own = True,display = False)




"""
Perspective Sphere
"""

camspecsPerspSphere = {"azimuth": 90,"elevation": 70,"distance": 10}

perspSphereNP = refplanePS.pinholeRefractionTest(r"./data/Perspective/NoPlane/Sphere/perspSphereZ-0.02CamAbove/","perspSphereZ-0.02CamAbove",1,1,0.05,0.035,np.array([0,0,1]),threshold = 0.08, lightlim = 5,SaveName = "perspSphereCamAbove",SaveDir = r"./ExperimentData/Sphere/",mayavispecs = camspecsPerspSphere,display = False)

# perspSphereNP = refplanePS.pinholeRefractionTest(r"./data/Perspective/NoPlane/Sphere/per/","perspSphereZ-0.02CamAbove",1,1,0.05,0.035,np.array([0,0,1]),threshold = 0.5, lightlim = 7,SaveName = "perspSphereCamAbove",SaveDir = r"./ExperimentData/Sphere/",mayavispecs = camspecsPerspSphere,display = False)

perspSphereWP = refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Sphere/perspSphereWithPlaneNoTilt/","perspSphereWithPlaneNoTilt",1,etaOut,0.05,0.035,np.array([0,0,1]),threshold = 0.1, lightlim = 7,SaveName = "perspSphereWithPlaneNoTilt",SaveDir = r"./ExperimentData/Sphere/",mayavispecs = camspecsPerspSphere,display = False)

perspSphereWPB2 = refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Sphere/perspSphereZ-0.02WithPlaneNoTiltL3Bounce2/","perspSphereZ-0.02WithPlaneNoTiltL3Bounce2",1,etaOut,0.05,0.035,np.array([0,0,1]),threshold = 0.07, lightlim = 5,SaveName = "perspSphereWithPlaneNoTiltBounce2",SaveDir = r"./ExperimentData/Sphere/",mayavispecs = camspecsPerspSphere,display = False,datapoint = 6)


perspSphereWPWTB2 = refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Sphere/perspInternalSphere0.01VeryWeakTilt2Bounce/","perspInternalSphere0.01VeryWeakTilt2Bounce",1,etaOut,0.05,0.035,np.array([0,-0.003,0.03]),threshold = 0.10, lightlim = 6,SaveName = "perspSphereWithPlaneWithTilt2Bounce",SaveDir = r"./ExperimentData/Sphere/",mayavispecs = camspecsPerspSphere,display = False)

"""
Fit spheres Perspective
"""
fac1 = 140/(np.nanmax(perspSphereNP[2]) -np.nanmin(perspSphereNP[2]))
yosntzFitS = refplanePS.orthoFitSphere(perspSphereNP[2]*fac1,us =perspSphereNP[0]*fac1, vs = perspSphereNP[1]*fac1 ,x0 = np.array([0,0,0,1]),Own = True,display = False)

fac2 = 140/(np.nanmax(perspSphereWP[2]) -np.nanmin(perspSphereWP[2]))

# # # plotImage(oswtz)
oswtzFitS = refplanePS.orthoFitSphere(perspSphereWP[2]*fac2,us =perspSphereWP[0]*fac2, vs = perspSphereWP[1]*fac2,x0 = np.array([250,250,-10,140]),Own = True,display = False)

fac3 = 140/(np.nanmax(perspSphereWPB2[2]) -np.nanmin(perspSphereWPB2[2]))

# # # plotImage(oswtb2z)
oswtb2zFitS = refplanePS.orthoFitSphere(perspSphereWPB2[2]*fac3,us =perspSphereWPB2[0]*fac3, vs = perspSphereWPB2[1]*fac3,x0 = np.array([255,250,-27,145]),Own = True,display = False)

fac4 = 140/(np.nanmax(perspSphereWPWTB2[2]) -np.nanmin(perspSphereWPWTB2[2]))

oswtb2zFitS = refplanePS.orthoFitSphere(perspSphereWPWTB2[2]*fac4,us =perspSphereWPWTB2[0] *fac4, vs = perspSphereWPWTB2[1] * fac4,x0 = np.array([255,250,-27,145]),Own = True,display = False)






"""
Ortho Cube
"""
camspecsOrthoCube = {"azimuth": 45,"elevation": 70,"distance": 10}


refplanePS.refractionTest(r"./data/Orthograhic/NoPlane/Cube/orthoCubeNoPlane/","orthoCubeNoPlane",degToNormal(0,0,0),1,1,datapoint = 4,rot90 = True,lightlim = 10,threshold = 0.07,SaveName = "orthoCubeNoPlane",SaveDir = r"./ExperimentData/Cube/",mayavispecs = camspecsOrthoCube,display = False,verbose = False)


refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Cube/orthoCubeWithPlaneNoTilt/","orthoCubeWithPlaneNoTilt",degToNormal(0,0,0),1,etaOut,datapoint = 4,rot90 = True,lightlim = 10,threshold = 0.01,SaveName = "orthoCubeWithPlaneParallel",SaveDir = r"./ExperimentData/Cube/",mayavispecs = camspecsOrthoCube,display = False,verbose = False)

refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Cube/orthoCubeWithPlaneBounce2CloseLightsNoTilt/","orthoCubeWithPlaneBounce2CloseLightsNoTilt",degToNormal(0,0,0),1,etaOut,datapoint = 4,rot90 = True,lightlim = 10,threshold = 0.01,SaveName = "orthoCubeWithPlaneParallel2Bounce",SaveDir = r"./ExperimentData/Cube/",mayavispecs = camspecsOrthoCube,display = False,verbose = False)


refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Cube/orthoCubeWithPlaneBounce2CloseLightsTiltX5/","orthoCubeWithPlaneBounce2CloseLightsTiltX5",degToNormal(5,0,0),1,etaOut,datapoint = 4,rot90 = True,lightlim = 10,threshold = 0.01,SaveName = "orthoCubeWithPlaneTilt2Bounce",SaveDir = r"./ExperimentData/Cube/",mayavispecs = camspecsOrthoCube,display = False,verbose = False)





"""
Perspective Cube
"""

camspecsPerspCube = {"azimuth": 45,"elevation": 70,"distance": 10}



refplanePS.pinholeRefractionTest(r"./data/Perspective/NoPlane/Cube/perspCubeZ-0.02NoPlaneL3Bounce8/","perspCubeZ-0.02NoPlaneL3Bounce8",1,1,0.05,0.035,np.array([0,0,1]),threshold = 0.02, lightlim = 7,SaveName = "perspCubeNoPlaneNoTilt",SaveDir = r"./ExperimentData/Cube/",mayavispecs = camspecsPerspCube,display = False,datapoint = 6)


# refplanePS.pinholeRefractionTest(r"./data/Perspective/NoPlane/Cube/perspCubeZ-0.02NoPlaneL3/","perspCubeZ-0.02NoPlaneL3",1,1,0.05,0.035,np.array([0,0,1]),threshold = 0.02, lightlim = 7,SaveName = "perspCubeNoPlaneNoTilt",SaveDir = r"./ExperimentData/Cube/",mayavispecs = camspecsPerspCube,display = False,datapoint = 6)




refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Cube/perspCubeZ-0.02WithPlaneNoTiltL3Bounce8/","perspCubeZ-0.02WithPlaneNoTiltL3Bounce8",1,etaOut,0.05,0.035,np.array([0,0,0.03]),threshold = 0.1, lightlim = 7,SaveName = "perspCubeWithPlaneNoTilt",SaveDir = r"./ExperimentData/Cube/",mayavispecs = camspecsPerspCube,display = False,datapoint = 6)


refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Cube/perspCubeZ-0.2WithPlaneNoTiltBounce2/","perspCubeZ-0.2WithPlaneNoTiltBounce2",1,etaOut,0.05,0.035,np.array([0,0,0.03]),threshold = 0.02, lightlim = 7,SaveName = "perspCubeWithPlaneNoTiltBounce2",SaveDir = r"./ExperimentData/Cube/",mayavispecs = camspecsPerspCube,display = False)



refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Cube/perspCubeZ-0.2WithPlaneWeakTiltBounce2/","perspCubeZ-0.2WithPlaneWeakTiltBounce2",1,etaOut,0.05,0.035,np.array([0,-0.003,0.03]),threshold = 0.02, lightlim = 7,SaveName = "perspCubeWithPlaneWeakTiltBounce2",SaveDir = r"./ExperimentData/Cube/",mayavispecs = camspecsPerspCube,display = False)

"""
Ortho Skull
"""
camspecsOrthoSkull = {"azimuth": 45,"elevation": 70,"distance": 10}


refplanePS.refractionTest(r"./data/Orthograhic/NoPlane/Skull/orthoSkullNoPlane/","orthoSkullNoPlane",degToNormal(0,0,0),1,1,datapoint = 6,rot90 = True,lightlim = 5,threshold = 0.04,SaveName = "orthoSkullNoPlane",SaveDir = r"./ExperimentData/Skull/",mayavispecs = camspecsOrthoSkull,display = False,verbose = False)


refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Skull/orthoSkullWithPlaneParallel/","orthoSkullWithPlaneParallel",degToNormal(0,0,0),1,etaOut,datapoint = 6,rot90 = True,lightlim = 5,threshold = 0.01,SaveName = "orthoSkullWithPlaneParallel",SaveDir = r"./ExperimentData/Skull/",mayavispecs = camspecsOrthoSkull,display = False,verbose = False)



refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Skull/orthoSkullWithPlaneParallelBounce2/","orthoSkullWithPlaneParallelBounce2",degToNormal(0,0,0),1,etaOut,datapoint = 6,rot90 = True,lightlim = 5,threshold = 0.01,SaveName = "orthoSkullWithPlaneParallelBounce2",SaveDir = r"./ExperimentData/Skull/",mayavispecs = camspecsOrthoSkull,display = False,verbose = False)


refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Skull/orthoSkullWithPlaneTiltX5Bounce2/","orthoSkullWithPlaneTiltX5Bounce2",degToNormal(5,0,0),1,etaOut,datapoint = 6,rot90 = True,lightlim = 5,threshold = 0.01,SaveName = "orthoSkullWithPlaneTilt5ParallelBounce2",SaveDir = r"./ExperimentData/Skull/",mayavispecs = camspecsOrthoSkull,display = False,verbose = False)

# refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/","orthoSkullNoPlane",degToNormal(0,0,0),1,1,datapoint = 4,rot90 = True,lightlim = 10,threshold = 0.07,SaveName = "orthoSkullNoPlane",SaveDir = r"./ExperimentData/Skull",mayavispecs = camspecsOrthoSkull,display = False,verbose = False)

"""
Perspective Skull
"""
camspecsPerspSkull = {"azimuth": 45,"elevation": 70,"distance": 10}





refplanePS.pinholeRefractionTest(r"./data/Perspective/NoPlane/Skull/perspSkullZ-0.02NoPlaneL3V2/","perspSkullZ-0.02NoPlaneL3",1,1,0.05,0.035,np.array([0,0,0.03]),threshold = 0.07, lightlim = 7,SaveName = "perspSkullNoPlane",SaveDir = r"./ExperimentData/Skull/",mayavispecs = camspecsPerspSkull,display = False,datapoint = 6)

refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Skull/perspSkullZ-0.02WithPlaneNoTilt/","perspSkullZ-0.02WithPlaneNoTilt",1,etaOut,0.05,0.035,np.array([0,0,0.03]),threshold = 0.02, lightlim = 7,SaveName = "perspSkullWithPlaneNoTilt",SaveDir = r"./ExperimentData/Skull/",mayavispecs = camspecsPerspSkull,display = False,datapoint = 6)


refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Skull/perspSkullWithPlaneBounce2/","perspSkullWithPlaneBounce2",1,etaOut,0.05,0.035,np.array([0,0,0.03]),threshold = 0.02, lightlim = 7,SaveName = "perspSkullWithPlaneNoTiltBounce2",SaveDir = r"./ExperimentData/Skull/",mayavispecs = camspecsPerspSkull,display = False,datapoint = 6)



refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Skull/perspSkullZ-0.2WithPlaneWeakTiltBounce2/","perspSkullZ-0.2WithPlaneWeakTiltBounce2",1,etaOut,0.05,0.035,np.array([0,-0.003,0.03]),threshold = 0.02, lightlim = 7,SaveName = "perspSkullWithPlaneWeakTiltBounce2",SaveDir = r"./ExperimentData/Skull/",mayavispecs = camspecsPerspSkull,display = False,datapoint = 6)




"""
Ortho Insect
"""

camspecsOrthoInsect = {"azimuth": 45,"elevation": 70,"distance": 10}


# orthoGraphosomaSmallNoPlaneCloseLights 

refplanePS.refractionTest(r"./data/Orthograhic/NoPlane/Insect/orthoGraphosomaSmallNoPlaneCloseLights/","orthoGraphosomaSmallNoPlaneCloseLights",degToNormal(0,0,0),1,1,datapoint = 6,rot90 = True,lightlim = 5,threshold = 0.01,SaveName = "orthoInsectNoPlane",SaveDir = r"./ExperimentData/Insect/",mayavispecs = camspecsOrthoInsect,display = False,verbose = False)

refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Insect/orthoGraphosomaSmallWithPlaneNoTiltBounce2CloseLights/","orthoGraphosomaSmallWithPlaneNoTiltBounce2CloseLights",degToNormal(0,0,0),1,etaOut,datapoint = 6,rot90 = True,lightlim = 5,threshold = 0.005,SaveName = "orthoInsectWithPlane2Bounce",SaveDir = r"./ExperimentData/Insect/",mayavispecs = camspecsOrthoInsect,display = False,verbose = False)

#FIXME: Use new data
# refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Insect/orthoGraphosomaSmallWithPlane/","orthoGraphosomaSmallWithPlane",degToNormal(0,0,0),1,etaOut,datapoint = 6,rot90 = True,lightlim = 4,threshold = 0.0040,SaveName = "orthoInsectWithPlane",SaveDir = r"./ExperimentData/Insect/",mayavispecs = camspecsOrthoInsect,display = False,verbose = False)
refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Insect/orthoInsectWithPlaneNoTiltLight3Bounce8/","orthoInsectWithPlaneNoTiltLight3Bounce8",degToNormal(0,0,0),1,etaOut,datapoint = 6,rot90 = True,lightlim = 4,threshold = 0.0040,SaveName = "orthoInsectWithPlane",SaveDir = r"./ExperimentData/Insect/",mayavispecs = camspecsOrthoInsect,display = False,verbose = False)


refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Insect/orthoGraphosomaSmallWithPlaneTiltX5Bounce2CloseLights/","orthoGraphosomaSmallWithPlaneTiltX5Bounce2CloseLights",degToNormal(5,0,0),1,etaOut,datapoint = 6,rot90 = True,lightlim = 5,threshold = 0.005,SaveName = "orthoInsectWithPlaneTilt2Bounce",SaveDir = r"./ExperimentData/Insect/",mayavispecs = camspecsOrthoInsect,display = False,verbose = False)



# refplanePS.refractionTest(r"./data/Orthograhic/NoPlane/Insect/orthoGraphosomaSmall/","orthoGraphosomaSmall",degToNormal(0,0,0),1,1,datapoint = 6,rot90 = True,lightlim = 5,threshold = 0.005,SaveName = "orthoInsectNoPlane",SaveDir = r"./ExperimentData/Insect/",mayavispecs = camspecsOrthoInsect,display = False,verbose = False)

# refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Insect/orthoGraphosomaSmallWithPlane/","orthoGraphosomaSmallWithPlane",degToNormal(0,0,0),1,1,datapoint = 6,rot90 = True,lightlim = 10,threshold = 0.005,SaveName = "orthoInsectWithPlaneParallel",SaveDir = "./ExperimentData/Insect/",mayavispecs = camspecsOrthoInsect,display = False,verbose = False)


"""
Perspective Insect
"""

camspecsPerspInsect = {"azimuth": 45,"elevation": 70,"distance": 10}

   
refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Insect/perspInsectZ-0.03NoPlaneL3/","perspInsectZ-0.03NoPlaneL3",1,1,0.05,0.035,np.array([0,0,0.03]),threshold = 0.005, lightlim = 5,SaveName = "perspInsectNoPlane",SaveDir = r"./ExperimentData/Insect/",mayavispecs = camspecsPerspInsect,display = False,datapoint = 6)


   
refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Insect/perspInsectZ-0.3WithPlaneNoTiltMoreLightBounce2/","perspInsectZ-0.3WithPlaneNoTiltMoreLightBounce2",1,etaOut,0.05,0.035,np.array([0,0,0.03]),threshold = 0.005, lightlim = 5,SaveName = "perspInsectWithPlaneNoTiltBounce2",SaveDir = r"./ExperimentData/Insect/",mayavispecs = camspecsPerspInsect,display = False,datapoint = 6)

refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Insect/perspInsectZ-0.3WithPlaneNoTiltMoreLight/","perspInsectZ-0.3WithPlaneNoTiltMoreLight",1,etaOut,0.05,0.035,np.array([0,0,0.03]),threshold = 0.005, lightlim = 5,SaveName = "perspInsectWithPlaneNoTilt",SaveDir = r"./ExperimentData/Insect/",mayavispecs = camspecsPerspInsect,display = False,datapoint = 6)


refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Insect/perspInsectZ-0.3WithPlaneWeakTiltMoreLightBounce2/","perspInsectZ-0.3WithPlaneWeakTiltMoreLightBounce2",1,etaOut,0.05,0.035,np.array([0,-0.003,0.03]),threshold = 0.005, lightlim = 5,SaveName = "perspInsectWithPlaneWeakTiltBounce2",SaveDir = r"./ExperimentData/Insect/",mayavispecs = camspecsPerspInsect,display = False,datapoint = 6)




"""
Rotated stretch
"""

#Example with sphere at low angles and high angles



"""
Lightstretch
"""

# camspecsLSortho = {"azimuth": 90,"elevation": 70,"distance": "auto"}

# zsb2x225 = refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneBounce2CloseLightsTiltX225/","orthoSphereWithPlaneBounce2CloseLightsTiltX225",degToNormal(22.5,0,0),1,etaOut,datapoint = 4,rot90 = True,LightStretch = False,Fresnel = False,lightlim = 10,threshold = 0.07,newNormal = True,mayavispecs = camspecsLSortho ,damp = 0,SaveName = "orthoSphereNoLightstretch",SaveDir = r"./ExperimentData/Lightstretch/",display = False)

# zsb2x225 = refplanePS.refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneBounce2CloseLightsTiltX225/","orthoSphereWithPlaneBounce2CloseLightsTiltX225",degToNormal(22.5,0,0),1,etaOut,datapoint = 4,rot90 = True,LightStretch = True,Fresnel = True,lightlim = 4,threshold = 0.02,newNormal = True, mayavispecs = camspecsLSortho , damp = 0,SaveName = "orthoSphereWithLightstretch",SaveDir = r"./ExperimentData/Lightstretch/",display = False)



"""
Uncorrected models, to illustratrate the effect of not using correct stuff
"""
camspecsUncorrectedOrtho = {"azimuth": 90,"elevation": 70,"distance": 10}


refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneBounce2CloseLightsTiltX225/","orthoSphereWithPlaneBounce2CloseLightsTiltX225",degToNormal(22.5,0,0),1,1,datapoint = 4,rot90 = True,LightStretch = False,Fresnel = False,lightlim = 10,threshold = 0.07,mayavispecs = camspecsUncorrectedOrtho ,damp = 0,SaveName = "orthoSphereTiltedUncorrectedEta1",SaveDir = r"./ExperimentData/Uncorrected/",display = False,verbose = False)


refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneBounce2CloseLightsTiltX225/","orthoSphereWithPlaneBounce2CloseLightsTiltX225",degToNormal(22.5,0,0),1,etaOut,datapoint = 4,rot90 = True,LightStretch = False,Fresnel = False,lightlim = 10,threshold = 0.07,mayavispecs = camspecsUncorrectedOrtho ,damp = 0,SaveName = "orthoSphereTiltedUncorrectedNoFresnelLL",SaveDir = r"./ExperimentData/Uncorrected/",display = False,verbose = False)


refplanePS.refractionTest(r"./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneBounce2CloseLightsTiltX225/","orthoSphereWithPlaneBounce2CloseLightsTiltX225",degToNormal(0,0,0),1,etaOut,datapoint = 4,rot90 = True,LightStretch = True,Fresnel = True,lightlim = 10,threshold = 0.07,mayavispecs = camspecsUncorrectedOrtho ,damp = 0,SaveName = "orthoSphereTiltedUncorrectedWrongPlanenormal",SaveDir = r"./ExperimentData/Uncorrected/",display = False,verbose = False)


refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Sphere/perspInternalSphere0.01VeryWeakTilt2Bounce/","perspInternalSphere0.01VeryWeakTilt2Bounce",1,1,0.05,0.035,np.array([0,-0.003,0.03]),threshold = 0.07, lightlim = 6,SaveName = "perspSphereWithPlaneWithTilt2BounceNoRefraction",SaveDir = r"./ExperimentData/Uncorrected/",mayavispecs = camspecsUncorrectedOrtho,display = False)


refplanePS.pinholeRefractionTest(r"./data/Perspective/WithPlane/Sphere/perspInternalSphere0.01VeryWeakTilt2Bounce/","perspInternalSphere0.01VeryWeakTilt2Bounce",1,etaOut,0.05,0.035,np.array([0,0,0.03]),threshold = 0.07, lightlim = 6,SaveName = "perspSphereWithPlaneWithTilt2BounceWrongDirection",SaveDir = r"./ExperimentData/Uncorrected/",mayavispecs = camspecsUncorrectedOrtho,display = False)




"""
Problems at high angles
"""

orthoSphereWPX225 =  refplanePS.refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneBounce2CloseLightsTiltX225/","orthoSphereWithPlaneBounce2CloseLightsTiltX225",degToNormal(22.5,0,0),1,etaOut,datapoint = 4,rot90 = True,lightlim = 7,threshold = 0.1,SaveName = "orthoSpherePlaneTilt225Bounce2",SaveDir = r"./ExperimentData/HighTilt/",mayavispecs = camspecsOrthoSphere,display = False,verbose = False)


orthoSphereWPX32 =  refplanePS.refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneX32Bounce2/","orthoSphereWithPlaneX32Bounce2",degToNormal(32,0,0),1,etaOut,datapoint = 4,rot90 = True,lightlim = 10,threshold = 0.07,SaveName = "orthoSpherePlaneTilt32Bounce2",SaveDir = r"./ExperimentData/HighTilt/",mayavispecs = camspecsOrthoSphere,display = False,verbose = False)

orthoSphereWPX37 =  refplanePS.refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneX37Bounce2/","orthoSphereWithPlaneX37Bounce2",degToNormal(37,0,0),1,etaOut,datapoint = 4,rot90 = True,lightlim = 5,threshold = 0.07,SaveName = "orthoSpherePlaneTilt37Bounce2",SaveDir = r"./ExperimentData/HighTilt/",mayavispecs = camspecsOrthoSphere,display = False,verbose = False)

orthoSphereWPX15Y15 =  refplanePS.refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneTiltX15Y15LightsNearPlaneDirBounce2/","orthoSphereWithPlaneTiltX15Y15LightsNearPlaneDirBounce2",degToNormal(15,15,0),1,etaOut,datapoint = 4,rot90 = True,lightlim = 5,threshold = 0.1,SaveName = "orthoSpherePlaneTiltX15Y15Bounce2CloseLights",SaveDir = r"./ExperimentData/HighTilt/",mayavispecs = camspecsOrthoSphere,display = False,verbose = False)

# orthoSphereWPX37Damped =  refplanePS.refractionTest("./data/Orthograhic/WithPlane/Sphere/orthoSphereWithPlaneX37Bounce2/","orthoSphereWithPlaneX37Bounce2",degToNormal(37,0,0),1,etaOut,datapoint = 4,rot90 = True,lightlim = 5,threshold = 0.07,SaveName = "orthoSpherePlaneTilt37Bounce2Damped",SaveDir = r"./ExperimentData/HighTilt/",mayavispecs = camspecsOrthoSphere,display = False,verbose = False,damp = 0.0002)
# 0.00001 - 2.912391
# 0.0001  - 2.121822
# 0.001   - 66.964999
# 0.0002  - 1.582304
# 0.0003  - 3.132738


"""
Fit spheres High angles
"""
yosntzFitS = refplanePS.orthoFitSphere(orthoSphereWPX225,x0 = np.array([250,250,-65,144]),Own = True,display = False)


# # plotImage(oswtz)
oswtzFitS = refplanePS.orthoFitSphere(orthoSphereWPX32,x0 = np.array([250,250,-10,140]),Own = True,display = False)

# # plotImage(oswtb2z)
oswtb2zFitS = refplanePS.orthoFitSphere(orthoSphereWPX37,x0 = np.array([255,250,-27,145]),Own = True,display = False)

oswtb2zFitS = refplanePS.orthoFitSphere(orthoSphereWPX15Y15,x0 = np.array([255,250,-27,145]),Own = True,display = False)



oswtb2zFitS = refplanePS.orthoFitSphere(orthoSphereWPX37Damped,x0 = np.array([255,250,-27,145]),Own = True,display = False)
