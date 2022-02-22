# Run as: blender -b <filename> -P <this_script> -- <filename> <image_path>

import sys, os
import mathutils # math functions for blender
import numpy as np
from math import tan, atan,sin,asin,sqrt,atan2, pi,cos,acos
import bpy # the blender functions themself
import numpy.linalg as lin
# Note that blender needs tto have scipy installed!
from scipy.spatial.transform import Rotation as Rot
#import pslib

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

def lightCamRotator(obj,focusPoint = mathutils.Vector((0,0,0)),
                  distance = 10.0, direction = mathutils.Vector((0,0,1))):
    """
    Rotates a light or a camera so that it is placed at focusPoint + distance * direc.
    
    We assume that the object obj is locked onto some other object located at the
    focusPoint, in order for it to get the right orrientation.
    """
    direc = direction.copy()
    
    mathutils.Vector.normalize(direc)
    newLocation = focusPoint + distance * direc

    obj.location = newLocation


def renderer(path,filename):
    """
    Makes blender run a render, and saves the resulting png at path as filename.
    """
    bpy.context.scene.render.filepath = path + filename
    bpy.ops.render.render(write_still=True)



def RotationRenderer(objectsToRotate,objectRotations,camera,light,cameraDirections,
                     camToLightDir,filename = "coolimg",path = "",cameraDistance = 20,
                     lightDistance = 10, rotate = np.array([0,0,1])):
    """
    Goes through each cameraDirection, light direction and object orrientation 
    and generates a rendered image for each, saving them as pngs. 
    
    The name contains filename, as well as information about which render it 
    is, seperated by 'Δ'.
    """
    i = 0
    j = 0
    k = 0

    
    #Contains the original orientations of the object in a dictionary.
    file =  open(filename + ".csv", 'w+')
    originalRotations = {}
    for obj in objectsToRotate:
        originalRotations[obj.name] = obj.rotation_euler.copy()
        
    for objrot in objectRotations:
        print("Working on orientation %s" % str(objrot))

        rotstring = ""
        csvrotstring = ""
        for obj in objectsToRotate:
            print("Rotating object %s" % obj.name)
            print("Current orientation of the object: %s" % str(obj.rotation_euler))
            rot = originalRotations[obj.name].copy()
            print("Applying rotation %s to original orientation %s" % (rot,objrot)) 
            rot.rotate(objrot.copy())
            obj.rotation_euler = rot.copy()
            print("New orientation of the object: %s" % str(obj.rotation_euler))
            csvrotstring = csvrotstring + ("\t%s\t%.3f\t%.3f\t%.3f" % (obj.name,obj.rotation_euler.x,obj.rotation_euler.y,obj.rotation_euler.z))
            rotstring =rotstring + ("Δ%s orientationΔ%.3f,%.3f,%.3f" % (obj.name,obj.rotation_euler.x,obj.rotation_euler.y,obj.rotation_euler.z))
            #+  "Δ" + obj.name + " orientationΔ" + str(rot.x) + "," + str(rot.y) + "," + str(rot.z) + ""
            
        
        for direc in cameraDirections:
            lightCamRotator(camera,distance = cameraDistance, direction = direc)
            for lightdirec in camToLightDir(direc):
                

                R = matRotateToVector(np.array([0,0,1]),hat(rotate))
                print(R)

                print(lightdirec)
                lightdirec = mathutils.Vector(R @ lightdirec)
                print(lightdirec)
                lightCamRotator(light,distance = lightDistance, direction = lightdirec)
                camDir = direc*cameraDistance
                lightDir = lightdirec * lightDistance                
                nfilename = filename + "ΔnumberΔ" + str(k) + "," + str(j) +  rotstring   + ("ΔlightΔ[%.3f,%.3f,%.3f]Δ" % (lightDir[0],lightDir[1],lightDir[2]))
                nfilename = nfilename + ("camplaceΔ[%.3f,%.3f,%.3f]" % (camDir[0],camDir[1],camDir[2]))
                printline =  str(k) + "\t" + str(j) + csvrotstring +  ("\t%.3f\t%.3f\t%.3f\n" % (lightDir[0],lightDir[1],lightDir[2]))
                printline = printline + ("camplace : [%.3f,%.3f,%.3f]" % (camDir[0],camDir[1],camDir[2]))
                print("Renering image %d %d" %(k,j))
                
                filecheck = path + nfilename + ".png"
                if os.path.isfile(filecheck):
                    print("\n\n\n-------File already exists, will not render it --------")
                else:
                    print("File %s did not exist" % (filecheck))
                    renderer(path,nfilename)
                    file.write(printline)
                    file.flush()
                j += 1
            i += 1
        j = 0
        k = k + 1
    file.close()
    print("\n\n\nFinished :)")


def circularVectors(heights, n):
    """
    Genrates n vectors for each h in heights, with z value h
    and angles 2π•i/n aorund the sphere.

    Assumed that ∀h ∈ heights, h ∈ [0,1]
    """
    
    
    if type(heights) is float or type(heights) is 0:
        heights = [heights]

    vectors = []
    for h in heights:
        vec =  mathutils.Vector((sqrt(1 - h**2),0,h))
        for i in range(0,n):
            vec.rotate(mathutils.Matrix.Rotation(-2*pi/n,3,"Z"))
            vectors.append(vec.copy())
            
    return vectors


#
#def objectRorations(xList,n):
#    rotations = []
#
#    for i in range(0,len(xList)):
#        z = xList[i]
#        
#        
#        for j in range(0,n):
#            rotz = mathutils.Matrix.Rotation(2*np.pi/n, 3, 'X')
#            if j == 0:
#                if i > 0:
#                    r1 = mathutils.Matrix.Rotation(-xList[i-1], 3, 'Y')
#                    r2 = mathutils.Matrix.Rotation(z, 3, 'Y')
#                    r1.rotate(r2)
#                    r1.rotate(rotz)
#                    rotations.append(r1.copy().to_euler())
#                else:
#                    r1 = mathutils.Matrix.Rotation(z, 3, 'Y')
#                    r1.rotate(rotz)
#                    rotations.append(r1.copy().to_euler())
#            else:
#                rotations.append(rotz.copy().to_euler())
#    
#    return rotations

def objectRotations(xList,n):
    """
    Creates n new rotations pr element in xList, each coresponing to aplying
    an xList[i] rotation along the x axis, and then an j*2*pi/n rotation
    alog the y axis.
    """
    rotations = []
    
    rotx = mathutils.Matrix.Rotation(2*np.pi/n, 3, 'X')
    for i in range(0,len(xList)):
        z = xList[i]
        
        rot = mathutils.Matrix.Rotation(z, 3, 'Y')
        rotations.append(rot.copy().to_euler())
        for j in range(0,n-1):
            rot.rotate(rotx)
            rotations.append(rot.copy().to_euler())
    
    return (rotations)

def CircularLightHeights(heights,n):
    """
    Creates n unit circle vectors with z value heights[i] for each element in
    heights, as well as (0,0,1).
    """
    vecs = circularVectors(heights,n)
    #vecs.append(mathutils.Vector((0,0,1)))
    return (lambda x: vecs)

def vechat(vec):
    norm = vec.x**2 + vec.y**2 + vec.z**2
    return vec/(norm**0.5)

# Assume the last argument is image path
extraArgs = sys.argv[5:]
imagePath = extraArgs[-1]
filename = extraArgs[-2]
rotations = eval(extraArgs[0])
rotationsCount = eval(extraArgs[1])
heights = eval(extraArgs[2])
heightsCount = eval(extraArgs[3])
rotLight = np.array(eval(extraArgs[4]))
samples = eval(extraArgs[5])
lightType = extraArgs[6]
lightEnergy = eval(extraArgs[7])
lightDistance = eval(extraArgs[8])
cameraType = extraArgs[9]
cameraDistance = eval(extraArgs[10])
cameraFocalLen = eval(extraArgs[11])
cameraSensorWidth = eval(extraArgs[12])


names = extraArgs[12:-2]

bpy.context.scene.cycles.samples = samples

for arg in extraArgs:
    print(arg)

print(names)

if os.path.exists(imagePath):
    objectsToRotate = []
    for obj in bpy.data.objects:
        if obj.type == "CAMERA":
                camera = obj
        if obj.type == "LIGHT":
                light = obj
        if obj.name in names: 
            print(obj.name)
            objectsToRotate.append(obj)
            
    bpy.data.lights[0].type = lightType
    bpy.data.lights[0].energy = lightEnergy
    bpy.data.cameras[0].type = cameraType

    if cameraType == "PERSP":
        print("Changing focal len to %f" % cameraFocalLen )
        bpy.data.cameras[0].lens = cameraFocalLen #NOTE: in milimeters
        bpy.data.cameras[0].sensor_width  = cameraSensorWidth
    RotationRenderer(objectsToRotate,objectRotations(rotations,rotationsCount),camera,light,[vechat(camera.location)],
                     CircularLightHeights(heights,heightsCount),filename = filename,
                     path = imagePath, cameraDistance = cameraDistance, lightDistance = lightDistance,rotate = rotLight)
    
else:
    print("Missing Imagepath:", imagePath) 


