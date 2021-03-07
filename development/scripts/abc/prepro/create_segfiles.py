
import os, sys, yaml
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..")))

from meshcnn.models.layers.mesh_prepare import fill_from_file,remove_non_manifolds, build_gemm


surfaceTypes = ['Plane','Revolution', 'Cylinder','Extrusion','Cone','Other','Sphere','Torus','BSpline']
segExt = ".eseg"
ssegExt = ".seseg"

def loadMesh(path):
    class MeshData:
        def __getitem__(self, item):
            return eval('self.' + item)
    meshData = MeshData()
    meshData.edge_areas = []
    meshData.vs, meshData.faces = fill_from_file(meshData, path)
    meshData.v_mask = np.ones(len(meshData.vs), dtype=bool)
    faces, face_areas = remove_non_manifolds(meshData, meshData.faces)
    build_gemm(meshData, faces, face_areas)
    return meshData

def parseYamlFile(featPath):
    data = None
    with open(featPath, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data

def getFacesSurfaceTypes(featPath, numFaces):
    featData = parseYamlFile(featPath)
    facesSurfaceTypes = -1 * np.ones(numFaces, dtype=int)
    for surface in featData['surfaces']:
        surfaceId = surfaceTypes.index(surface['type'])
        faceIndices = np.array(surface['face_indices'], dtype=int)
        facesSurfaceTypes[faceIndices] = surfaceId
        # for faceIndex in faceIndices:
        #     assert (facesSurfaceTypes[faceIndex] == -1),"Face with two surfaces?!"
        #     facesSurfaceTypes[faceIndex] = surfaceId
    return facesSurfaceTypes

def getEdgeHardLabels(meshData,featPath):
    facesSurfacesTypes = getFacesSurfaceTypes(featPath,len(meshData.faces))
    edgesSurfaceTypes =-1 * np.ones(meshData.edges_count, dtype=int)

    for faceId, face in enumerate(meshData.faces):
        faceSurfaceType = facesSurfacesTypes[faceId]
        for i in range(3):
            edge = sorted(list((face[i], face[(i + 1) % 3])))
            matchingEdgeIds = list(set(meshData.ve[edge[0]]).intersection(meshData.ve[edge[1]]))
            if (len(matchingEdgeIds) > 1):
                assert False,"more than one edge"
            elif (len(matchingEdgeIds) == 0):
                print(" {}: Edge {} in face {} not found in mesh".format(os.path.basename(featPath), edge, faceId))
                continue
            edgesSurfaceTypes[matchingEdgeIds[0]] = faceSurfaceType
    return edgesSurfaceTypes

def getEdgeSoftLabels(hardLabels, mesh):
    numClasses = len(surfaceTypes)
    gemmEdges = np.array(mesh.gemm_edges)
    softLabels = -1 * np.ones((mesh.edges_count, numClasses), dtype='float64')
    for ei in range(mesh.edges_count):
        prob = np.zeros(numClasses)
        segIds, counts = np.unique(hardLabels[gemmEdges[ei]], return_counts=True)
        prob[segIds] = counts / float(len(gemmEdges[ei]))
        softLabels[ei, :] = prob
    return softLabels

def createSegFiles(segFilePath, ssegFilePath, featFilePath, mesh):
    hardLabels = getEdgeHardLabels(mesh,featFilePath)
    softLabels = getEdgeSoftLabels(hardLabels,mesh)
    try:
        file = open(segFilePath,'w')
    except OSError as e:
        print('open() failed', e)
    else:
        with file:
            for hardLabel in hardLabels:
                file.write(str(hardLabel) + '\n')

    np.savetxt(ssegFilePath,softLabels, fmt='%f')

def getTargetDatasetRoot(links):
    assert(len(links) > 0)
    targetObj = os.readlink(links[0]) if os.path.islink(links[0]) else links[0]
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

if __name__=='__main__':

    if len(sys.argv) < 2:
        print("Wrong parameters")
        exit(1)

    datasetRoot = sys.argv[1]

    objLinks = getObjLinks(datasetRoot)

    if (len(objLinks) == 0):
        print("No obj files found in", datasetRoot)
        exit(1)

    segLinkPath = os.path.join(datasetRoot,"seg")
    ssegLinkPath = os.path.join(datasetRoot,"sseg")
    targetDatasetRoot = getTargetDatasetRoot(objLinks)
    targetSegPath = os.path.join(targetDatasetRoot,"seg")
    targetSSegPath = os.path.join(targetDatasetRoot,"sseg")


    print("Creating seg & sseg links in {} (targets in {}) for {} obj files in {}".format(datasetRoot, targetDatasetRoot,
                                                                                        len(objLinks),datasetRoot))

    if (not os.path.exists(targetSegPath)):
        os.mkdir(targetSegPath)

    if (not os.path.exists(targetSSegPath)):
        os.mkdir(targetSSegPath)

    if (not os.path.exists(segLinkPath)):
        os.symlink(targetSegPath, segLinkPath)

    if (not os.path.exists(ssegLinkPath)):
        os.symlink(targetSSegPath, ssegLinkPath)


    objLinks.sort()
    objFileTargets = list(map(lambda link: os.readlink(link),objLinks)) if os.path.islink(objLinks[0]) else objLinks

    for objPath in objFileTargets:
        fileNamePrefix = os.path.splitext(os.path.basename(objPath))[0]
        segFilePath = os.path.join(targetSegPath,fileNamePrefix+ segExt)
        ssegFilePath = os.path.join(targetSSegPath,fileNamePrefix+ ssegExt)

        if (os.path.exists(segFilePath) and os.path.exists(ssegFilePath)):
            continue

        featFilePath = objPathToFeatPath(objPath)
        meshData = loadMesh(objPath)
        createSegFiles(segFilePath, ssegFilePath, featFilePath, meshData)
        print("{} + {} -> {} -> {} ({} edges)".format(os.path.relpath(objPath,targetDatasetRoot),
                                                os.path.relpath(featFilePath,targetDatasetRoot),
                                                os.path.relpath(segFilePath,targetDatasetRoot),
                                                      os.path.relpath(ssegFilePath, targetDatasetRoot),
                                                      meshData.edges_count))






