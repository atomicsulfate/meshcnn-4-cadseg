
import os, sys, yaml
from pathlib import Path
import argparse
import math
import numpy as np
import random
import pymesh

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
    # To speed up processing, a edgeFaceRatio of 1.5 can be assumed and then build_gemm can be skipped.
    # meshData.edges_count = meshData.faces_count * 1.5
    build_gemm(meshData, faces, face_areas)
    meshData.faces_count = len(faces)

    return meshData

def takeMesh(objPath,minEdges,maxEdges):
    try:
        mesh = loadMesh(objPath)
    except AssertionError:
        print("Skip non-manifold mesh:", objPath)
        return False

    if (mesh.edges_count >=minEdges and mesh.edges_count <=maxEdges and pymesh.load_mesh(objPath).is_closed()):
        print("Adding {} with {} faces and {} edges, edgeFaceRatio {}".format(objPath,mesh.faces_count,mesh.edges_count,
                                                                              mesh.edges_count/ mesh.faces_count))
        return True
    else:
        print("Discarding {} with {} edges".format(objPath,mesh.edges_count))
        return False

def addMesh(srcPath, dstPath):
    os.symlink(os.path.abspath(srcPath),os.path.join(dstPath,os.path.basename(srcPath)))
    return

def findSamples(objPath,minEdges,maxEdges,maxSamples):
    numSamples = 0
    sampleSrcPaths = []
    for root, _, fnames in sorted(os.walk(objPath)):
        for fname in fnames:
            path = os.path.join(root, fname)
            if (os.path.splitext(fname)[1] == ".obj"):
                # basedir = os.path.basename(root)
                # statPath = os.path.join(srcPath, "stat", basedir)
                if (takeMesh(path, minEdges, maxEdges)):
                    sampleSrcPaths.append(path)
                    numSamples += 1
                    if (numSamples >= maxSamples):
                        return sampleSrcPaths
    return sampleSrcPaths


parser = argparse.ArgumentParser("Compile ABC dataset with specified restrictions")
parser.add_argument('--src', required=True, type=str, help="Path where source ABC dataset is located.")
parser.add_argument('--dst', required=True, type=str, help="Path where the resulting dataset will be placed")
parser.add_argument('--minEdges', type=int, default=0, help="Min number of edges to add a mesh to the dataset")
parser.add_argument('--maxEdges', type=int, default=math.inf, help="Max number of edges to add a mesh to the dataset")
parser.add_argument('--maxSamples', type=int, default=math.inf, help="Max dataset size")
parser.add_argument('--testTrainRatio', type=float, default=0.2, help="#test samples/#train samples")
parser.add_argument('--excludeNonManifolds', action='store_true', help="Exclude meshes that are not manifolds")
parser.add_argument('--excludeOpenMeshes', action='store_true', help="Exclude open meshes")

args = parser.parse_args()

srcPath = args.src
dstPath = args.dst
minEdges = args.minEdges
maxEdges = args.maxEdges
maxSamples = args.maxSamples

print("Compile dataset from {} to {} with #edges in [{},{}] and {} samples".format(srcPath,dstPath,minEdges,maxEdges,maxSamples))

objPath = os.path.join(srcPath,"obj")

if (not os.path.exists(objPath)):
    print(srcPath,"does not exist")
    exit(1)

dstTrainPath = os.path.join(dstPath,"train")
dstTestPath = os.path.join(dstPath,"test")

try:
    Path(dstTrainPath).mkdir(parents=True, exist_ok=True)
    Path(dstTestPath).mkdir(parents=True, exist_ok=True)
except FileExistsError as f_error:
    print(f_error)
    exit(1)

sampleSrcPaths = findSamples(objPath,minEdges,maxEdges,maxSamples)
random.shuffle(sampleSrcPaths)

numTotalSamples = len(sampleSrcPaths)
numTestSamples = int(numTotalSamples * args.testTrainRatio)
numTrainSamples = numTotalSamples - numTestSamples

sampleIdx = 0
for samplePath in sampleSrcPaths[:numTrainSamples]:
    addMesh(samplePath, dstTrainPath)

for samplePath in sampleSrcPaths[numTrainSamples:]:
    addMesh(samplePath, dstTestPath)

print("Dataset compiled with {} samples ({} train, {} test)".format(numTotalSamples,numTrainSamples,numTestSamples))


