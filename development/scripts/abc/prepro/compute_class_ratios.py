
import os, sys, yaml
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..")))

from meshcnn.models.layers.mesh_prepare import fill_from_file,remove_non_manifolds, build_gemm


surfaceTypes = ['Plane','Revolution', 'Cylinder','Extrusion','Cone','Other','Sphere','Torus','BSpline']

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

if len(sys.argv) < 2:
    print("Wrong parameters")
    exit(1)

datasetRoot = sys.argv[1]

objLinks = getObjLinks(datasetRoot)

if (len(objLinks) == 0):
    print("No obj files found in", datasetRoot)
    exit(1)

targetDatasetRoot = getTargetDatasetRoot(objLinks)

print("Computing surface type ratios in {} (targets in {}) for {} obj files".format(datasetRoot, targetDatasetRoot, len(objLinks)))

header = " ".join(x.center(10) for x in surfaceTypes)
print("\t\t\t\t\t\t    " + header)

objLinks.sort()
objFileTargets = list(map(lambda link: os.readlink(link),objLinks))
featFilePaths = list(map(objPathToFeatPath,objFileTargets))

totalSurfaceTypeFaceCount = np.zeros(len(surfaceTypes), dtype=int)

i = 0
for objLink in objLinks:
    objPath = os.readlink(objLink)
    featFilePath = objPathToFeatPath(objPath)
    surfaceTypeFaceCount = getSurfaceTypeFaceCount(featFilePath)
    totalSurfaceTypeFaceCount += surfaceTypeFaceCount
    print("{} {}".format(os.path.basename(objPath), " ".join(str(x).center(10) for x in surfaceTypeFaceCount)))
    i += 1
    if (i % 20 == 0):
        avgSurfaceTypeFaceCount = totalSurfaceTypeFaceCount * 100 / totalSurfaceTypeFaceCount.sum()
        print("Totals after {} samples:\t\t\t\t{}".format(i, " ".join("{:10.2f}".format(x) for x in avgSurfaceTypeFaceCount)),file=sys.stderr)

avgSurfaceTypeFaceCount = totalSurfaceTypeFaceCount * 100 / totalSurfaceTypeFaceCount.sum()
print("\t\t" + header)
print("Totals: {}".format(" ".join("{:10.4f}".format(x) for x in avgSurfaceTypeFaceCount)))




