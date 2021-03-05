
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

def getSurfaceTypeFaceCount(sourcePath):
    surfaceTypeCount = np.zeros(len(surfaceTypes), dtype=int)

    if (os.path.splitext(sourcePath)[1] == ".feat"):
        featData = parseYamlFile(sourcePath)
        for surface in featData['surfaces']:
            surfaceId = surfaceTypes.index(surface['type'])
            surfaceTypeCount[surfaceId] += len(surface['face_indices'])
    else:
        assert os.path.splitext(sourcePath)[1] == ".eseg"
        segLabels = np.loadtxt(open(sourcePath, 'r'), dtype='int')
        (uniqueSurfTypes, counts) = np.unique(segLabels, return_counts=True)
        for i in range(len(uniqueSurfTypes)):
            surfaceTypeCount[uniqueSurfTypes[i]] += counts[i]

    return surfaceTypeCount

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
    if (not os.path.exists(featDirPath)):
        return None
    return os.path.join(featDirPath, next(os.walk(featDirPath))[2][0])

def objPathToSegPath(objPath):
    segFName = os.path.splitext(os.path.basename(objPath))[0] + ".eseg"
    datasetPath = os.path.split(os.path.split(objPath)[0])[0]
    segPath = os.path.join(datasetPath,"seg",segFName)
    assert os.path.exists(segPath), "Seg file {} does not exist".format(segPath)
    return segPath

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

#targetDatasetRoot = getTargetDatasetRoot(objLinks)

print("Computing surface type ratios in {} for {} obj files".format(datasetRoot, len(objLinks)))

header = " ".join(x.center(10) for x in surfaceTypes)
print("\t\t\t\t\t\t    " + header)

#objLinks.sort()

totalSurfaceTypeCount = np.zeros(len(surfaceTypes), dtype=int)

i = 0
for objLink in objLinks:
    #objPath = os.readlink(objLink) if os.path.islink(objLink) else objLink
    #featFilePath = objPathToFeatPath(objPath)
    surfaceTypeCount = getSurfaceTypeFaceCount(objPathToSegPath(objLink))
    totalSurfaceTypeCount += surfaceTypeCount
    print("{:50} {}".format(os.path.basename(objLink), " ".join(str(x).center(10) for x in surfaceTypeCount)))
    i += 1
    if (i % 20 == 0):
        avgSurfaceTypeFaceCount = totalSurfaceTypeCount * 100 / totalSurfaceTypeCount.sum()
        print("Totals after {} samples:\t\t\t\t{}".format(i, " ".join("{:10.2f}".format(x) for x in avgSurfaceTypeFaceCount)),file=sys.stderr)

avgSurfaceTypeFaceCount = totalSurfaceTypeCount * 100 / totalSurfaceTypeCount.sum()
print("\t\t" + header)
print("Totals: {}".format(" ".join("{:10.4f}".format(x) for x in avgSurfaceTypeFaceCount)))




