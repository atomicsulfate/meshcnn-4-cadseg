
import os, sys
import argparse
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..")))
from scripts.abc.prepro.create_segfiles import getEdgeHardLabels, objPathToFeatPath
from meshcnn.models.layers.mesh_prepare import fill_from_file,remove_non_manifolds, build_gemm


def colorMesh(mesh, segLabels, srcPath, dstPath):
    dstFilePath = os.path.join(dstPath, os.path.basename(srcPath))
    faces = mesh.faces
    new_indices = np.zeros(mesh.v_mask.shape[0], dtype=np.int32)
    new_indices[mesh.v_mask] = np.arange(0, np.ma.where(mesh.v_mask)[0].shape[0])

    with open(dstFilePath, 'w') as f:
        for vi, v in enumerate(mesh.vs):
            f.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for face_id in range(len(faces) - 1):
            f.write("f %d %d %d\n" % (faces[face_id][0] + 1, faces[face_id][1] + 1, faces[face_id][2] + 1))
        f.write("f %d %d %d" % (faces[-1][0] + 1, faces[-1][1] + 1, faces[-1][2] + 1))
        i = 0
        for edge in mesh.edges:
            f.write("\ne %d %d %d" % (new_indices[edge[0]] + 1, new_indices[edge[1]] + 1,segLabels[i]))
            i += 1

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
    meshData.faces_count = len(faces)

    return meshData

def tryLoadMesh(objPath):
    try:
        mesh = loadMesh(objPath)
    except AssertionError:
        print("Non-manifold mesh:", objPath)
        return None
    return mesh

def objPathToSegPath(objPath):
    pathToObjDir,objFName = os.path.split(objPath)
    datasetPath = os.path.split(os.path.split(pathToObjDir)[0])[0] if os.path.basename(pathToObjDir) != "obj" \
        else os.path.split(pathToObjDir)[0]
    segDirPath = os.path.join(datasetPath,"seg")
    segFName = os.path.splitext(objFName)[0] + ".eseg"
    return os.path.join(segDirPath, segFName)

def loadSegLabels(segPath):
    return np.loadtxt(open(segPath, 'r'), dtype='float64')

parser = argparse.ArgumentParser("Color mesh surfaces by type")
parser.add_argument('--src', nargs='+', type=str, help="Path to source obj files")
parser.add_argument('--dst', required=True, type=str, help="Path where colored obj files will be saved")
# parser.add_argument('--minEdges', type=int, default=0, help="Min number of edges to add a mesh to the dataset")
# parser.add_argument('--maxEdges', type=int, default=math.inf, help="Max number of edges to add a mesh to the dataset")
# parser.add_argument('--maxSamples', type=int, default=math.inf, help="Max dataset size")
# parser.add_argument('--testTrainRatio', type=float, default=0.2, help="#test samples/#train samples")
# parser.add_argument('--excludeNonManifolds', action='store_true', help="Exclude meshes that are not manifolds")

args = parser.parse_args()

srcPaths = args.src
dstPath = args.dst


for srcPath in srcPaths:
    segPath = objPathToSegPath(srcPath)
    mesh = tryLoadMesh(srcPath)

    if (mesh == None):
        continue

    segLabels = None
    if (os.path.exists(segPath)):
        segLabels = loadSegLabels(segPath)
    else:
        segLabels = getEdgeHardLabels(mesh, objPathToFeatPath(srcPath))

    if (not os.path.exists(dstPath)):
        os.mkdir(dstPath)

    print("Color mesh {} with labels {}, result in {}".format(srcPath,segPath, dstPath))
    colorMesh(mesh,segLabels,srcPath, dstPath)









