
import os, sys, yaml
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..")))

from meshcnn.models.layers.mesh_prepare import from_scratch


surfaceTypes = ['Plane','Revolution', 'Cylinder','Extrusion','Cone','Other','Sphere','Torus','BSpline']

def loadMesh(path):
    class Options:
        def __getitem__(self, item):
            return eval('self.' + item)
    options = Options()
    options.num_aug = 0
    return from_scratch(path,options)

def parseYamlFile(featPath):
    data = None
    with open(featPath, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data

def getSurfaceByFace(featPath):
    featData = parseYamlFile(featPath)
    faceToSurface = {}
    surfaces = featData['surfaces']
    for surface in surfaces:
        surfaceId = surfaceTypes.index(surface['type'])
        for faceIndex in surface['face_indices']:
            assert (not (faceIndex in faceToSurface)),"Face with two surfaces?!"
            faceToSurface[faceIndex] = surfaceId
    return faceToSurface



def objPathToFeatPath(objPath):
    pathToObjDir = os.path.split(objPath)[0]
    globalObjPath, sampleId = os.path.split(pathToObjDir)
    datasetPath = os.path.split(globalObjPath)[0]
    featDirPath = os.path.join(datasetPath,"feat",sampleId)
    return os.path.join(featDirPath, next(os.walk(featDirPath))[2][0])

if len(sys.argv) < 2:
    print("Wrong parameters")
    exit(1)

datasetRoot = sys.argv[1]

segPath = os.path.join(datasetRoot,"seg")

if (os.path.exists(segPath)):
    print("Segmentation files already created")
    exit(1)

os.mkdir(segPath)

print("Creating segmentation files in {} for obj files in {}".format(segPath, datasetRoot))

objFilePaths = []
for root, _, fnames in os.walk(datasetRoot):
    for fname in fnames:
        if (os.path.splitext(fname)[1] == ".obj"):
            objFilePaths.append(os.path.join(root, fname))

print(len(objFilePaths), " obj files found")
objFilePaths.sort()
objFileTargets = list(map(lambda link: os.readlink(link),objFilePaths))
featFilePaths = list(map(objPathToFeatPath,objFileTargets))

for link,objFile,featFile in zip(objFilePaths,objFileTargets,featFilePaths):
    segFileName = os.path.splitext(os.path.basename(link))[0] + ".seg"
    segFilePath = os.path.join(segPath,segFileName)
    print("{} -> {} -> {}".format(segFilePath,objFile,featFile))
    meshData = loadMesh(objFile)
    faceToSurface = getSurfaceByFace(featFile)
    print("Mesh with {} edges".format(len(meshData.edges)))




