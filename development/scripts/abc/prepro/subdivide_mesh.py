
import os, sys, yaml
from pathlib import Path
import argparse
import math
import numpy as np
import random
import pymesh
import numpy as np

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..")))

from meshcnn.models.layers.mesh_prepare import fill_from_file,remove_non_manifolds, build_gemm

# def parseYamlFile(path):
#     data = None
#     with open(path, 'r') as stream:
#         try:
#             data = yaml.safe_load(stream)
#         except yaml.YAMLError as exc:
#             print(exc)
#     return data

# def takeMesh(statPath,minEdges,maxEdges):
#     for root, _, fnames in sorted(os.walk(statPath)):
#         for fname in fnames:
#             if (os.path.splitext(fname)[1] == ".yml"):
#                 path = os.path.join(root,fname)
#                 data = parseYamlFile(path)
#                 faces = data['#faces']
#                 edges = faces*1.5
#                 if (edges >=minEdges and edges<=maxEdges):
#                     print("Adding {} with {} faces and {} edges".format(path,faces,edges))
#                     return True
#                 else:
#                     return False

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
    datasetPath = os.path.split(os.path.split(pathToObjDir)[0])[0]
    segDirPath = os.path.join(datasetPath,"seg")
    segFName = os.path.splitext(objFName)[0] + ".eseg"
    return os.path.join(segDirPath, segFName)

def loadSegLabels(segPath):
    return np.loadtxt(open(segPath, 'r'), dtype='float64')

def subdivideMesh(mesh, srcPath, dstPath, segLabels):
    pmesh = pymesh.load_mesh(srcPath)

    for i in range(mesh.edge_count):
        edgeLabel = segLabels[i]
        mesh.edges[i]
    pmesh = pymesh.subdivide(pmesh, order=1, method="simple")
    pymesh.save_mesh(dstPath,pmesh)
    pmesh.vertices
    return

parser = argparse.ArgumentParser("Subdivide mesh")
parser.add_argument('--src', required=True, type=str, help="Path to source obj file")
parser.add_argument('--dst', required=True, type=str, help="Path to new obj file")
# parser.add_argument('--minEdges', type=int, default=0, help="Min number of edges to add a mesh to the dataset")
# parser.add_argument('--maxEdges', type=int, default=math.inf, help="Max number of edges to add a mesh to the dataset")
# parser.add_argument('--maxSamples', type=int, default=math.inf, help="Max dataset size")
# parser.add_argument('--testTrainRatio', type=float, default=0.2, help="#test samples/#train samples")
# parser.add_argument('--excludeNonManifolds', action='store_true', help="Exclude meshes that are not manifolds")

args = parser.parse_args()

srcPath = args.src
dstPath = args.dst


if (not os.path.exists(srcPath)):
    print(srcPath,"does not exist")
    exit(1)

if (os.path.exists(dstPath)):
    print(dstPath, "already exists")
    exit(1)

segPath = objPathToSegPath(srcPath)

if (not os.path.exists(segPath)):
    print(segPath,"does not exist")
    exit(1)

mesh = tryLoadMesh(srcPath)

if (mesh == None):
    exit(1)

segLabels = loadSegLabels(segPath)

print("Subdivide mesh {} with labels {}  ({} faces, {} edges), result in {}".format(srcPath,segPath, mesh.faces_count,
                                                                               mesh.edges_count, dstPath))
subdivideMesh(mesh,srcPath, dstPath, segLabels)








