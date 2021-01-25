
import os, sys, yaml
import numpy as np
from pathlib import Path
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..")))

from meshcnn.models.layers.mesh_prepare import fill_from_file,remove_non_manifolds, build_gemm


surfaceTypes = ['Plane','Revolution', 'Cylinder','Extrusion','Cone','Other','Sphere','Torus','BSpline']
favoredSurfaceTypeIndices = [1,3,4,5,6,7,8]

def parseYamlFile(featPath):
    data = None
    with open(featPath, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data

def getSurfaceTypeFaceCount(featPath):
    featData = parseYamlFile(featPath)
    surfaceTypeFaceCount = np.zeros(len(surfaceTypes), dtype=int)
    for surface in featData['surfaces']:
        surfaceId = surfaceTypes.index(surface['type'])
        surfaceTypeFaceCount[surfaceId] += len(surface['face_indices'])
    return surfaceTypeFaceCount

def getTargetDatasetRoot(links):
    assert(len(links) > 0)
    targetObj = os.readlink(links[0])
    pathToObjDir = os.path.split(targetObj)[0]
    objPrefixPath, _ = os.path.split(pathToObjDir)
    return os.path.split(objPrefixPath)[0]

def objPathToFeatPath(objPath):
    pathToObjDir = os.path.split(objPath)[0]
    globalObjPath, sampleId = os.path.split(pathToObjDir)
    datasetPath = os.path.split(globalObjPath)[0]
    featDirPath = os.path.join(datasetPath,"feat",sampleId)
    return os.path.join(featDirPath, next(os.walk(featDirPath))[2][0])

def getObjLinks(datasetRoot):
    objLinkPaths = []
    for root, _, fnames in os.walk(datasetRoot):
        for fname in fnames:
            if (os.path.splitext(fname)[1] == ".obj"):
                objLinkPaths.append(os.path.join(root, fname))
    return objLinkPaths

def selectSample(surfaceTypeFaceCount):
    return surfaceTypeFaceCount[favoredSurfaceTypeIndices].sum() > 0

def addMesh(srcPath, dstPath):
    os.symlink(os.path.abspath(srcPath),os.path.join(dstPath,os.path.basename(srcPath)))

if len(sys.argv) < 4:
    print("Wrong parameters")
    exit(1)

oldDatasetRoot = sys.argv[1]
newDatasetRoot = sys.argv[2]
testTrainRatio = float(sys.argv[3])

objLinks = getObjLinks(oldDatasetRoot)

if (len(objLinks) == 0):
    print("No obj files found in", oldDatasetRoot)
    exit(1)

dstTrainPath = os.path.join(newDatasetRoot,"train")
dstTestPath = os.path.join(newDatasetRoot,"test")

try:
    Path(dstTrainPath).mkdir(parents=True, exist_ok=False)
    Path(dstTestPath).mkdir(parents=True, exist_ok=False)
except FileExistsError as f_error:
    print(f_error)
    exit(1)

targetDatasetRoot = getTargetDatasetRoot(objLinks)

print("Resampling dataset {} (targets in {}) with {} obj files into {}".format(oldDatasetRoot, targetDatasetRoot, len(objLinks), newDatasetRoot))

objLinks.sort()
objFileTargets = list(map(lambda link: os.readlink(link),objLinks))
featFilePaths = list(map(objPathToFeatPath,objFileTargets))

totalSurfaceTypeFaceCount = np.zeros(len(surfaceTypes), dtype=int)
selectedObjPaths = []

for objLink in objLinks:
    objPath = os.readlink(objLink)
    surfaceTypeFaceCount = getSurfaceTypeFaceCount(objPathToFeatPath(objPath))
    if (selectSample(surfaceTypeFaceCount)):
        selectedObjPaths.append(objPath)
        print("{}: {}".format(os.path.relpath(objPath,targetDatasetRoot), surfaceTypeFaceCount))
        totalSurfaceTypeFaceCount += surfaceTypeFaceCount
        if (len(selectedObjPaths) % 20 == 0):
            print("Totals after {} samples: {}".format(len(selectedObjPaths), totalSurfaceTypeFaceCount * 100 / totalSurfaceTypeFaceCount.sum()))

random.shuffle(selectedObjPaths)
numTotalSamples = len(selectedObjPaths)
numTestSamples = int(numTotalSamples * testTrainRatio)
numTrainSamples = numTotalSamples - numTestSamples

for samplePath in selectedObjPaths[:numTrainSamples]:
    addMesh(samplePath, dstTrainPath)

for samplePath in selectedObjPaths[numTrainSamples:]:
    addMesh(samplePath, dstTestPath)

print("Resampled dataset {} samples, ({} train, {} test), surface frequencies {}".format(numTotalSamples, numTrainSamples, numTestSamples, totalSurfaceTypeFaceCount * 100 / totalSurfaceTypeFaceCount.sum()))




