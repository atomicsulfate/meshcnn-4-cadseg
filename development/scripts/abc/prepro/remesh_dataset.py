import os
import argparse
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial
import sys
import yaml
import bisect
import gmsh
import threading
from multiprocessing import Process

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..")))

from meshcnn.models.layers.mesh_prepare import build_gemm, remove_non_manifolds
import cadmesh

surfaceTypes = ['Plane', 'Revolution', 'Cylinder', 'Extrusion', 'Cone', 'Other', 'Sphere', 'Torus', 'BSpline']


def getSampleName(surfaceType, id):
    return "{}_{}".format(surfaceType, id)

def getMeshFaceCount(mesh):
    return round(len(mesh["faces"]))

def getEstMeshEdgeCount(mesh):
    return getMeshFaceCount(mesh)* 1.5

def getExactMeshEdgeCount(mesh):
    class MeshData:
        def __getitem__(self, item):
            return eval('self.' + item)

    meshData = MeshData()
    meshData.edge_areas = []
    meshData.vs = mesh["vertices"]
    meshData.faces = mesh["face_indices"]
    meshData.v_mask = np.ones(len(meshData.vs), dtype=bool)
    faces, face_areas = remove_non_manifolds(meshData, meshData.faces)
    build_gemm(meshData, faces, face_areas)
    return meshData.edges_count


def estimateFactor(polynomial, target_edges):
    eq = (polynomial - target_edges)
    for factor_inv in eq.roots():
        if (np.isreal(factor_inv) and factor_inv > 0):
            factor = 1.0 / factor_inv
            #print("root {}, new factor {}".format(factor_inv, factor))
            return factor
    return None

def interpolateFactor(inv_factors, num_edges, target_edges):

    # Increase slowlier the factor until an upper limit is found.
    #poly_degree = 2 if max(num_edges) < target_edges else 1
    # min_factor = min(inv_factors)
    # max_factor = max(inv_factors)
    # factor = estimateFactor(Polynomial.fit(inv_factors, num_edges, 2), target_edges)
    # constrain_factor_range = max(num_edges) > target_edges
    # factor_out_of_range = constrain_factor_range and (factor < min_factor or factor > max_factor)
    # factor_already_used = 1.0/factor in inv_factors
    # print("2deg interpolation: {}, oor {}, already used {}".format(1.0/factor,factor_out_of_range, factor_already_used))
    # if (factor == None or factor_out_of_range or factor_already_used):
    factor = estimateFactor(Polynomial.fit(inv_factors, num_edges, 1), target_edges)
    #print("Fallback to linear regression, factor", 1.0/factor)
    return factor

init_mesh_size_error = 0.02

def createMeshWithEdgeCount(stepPath, max_edges, meshSizeInitFactor):
    # Fit the function 1/mesh_size_factor -> mesh edges to a quadratic polynomial.
    inv_factors = []
    num_edges = []
    current_factor = meshSizeInitFactor
    min_edges = max_edges*(1-init_mesh_size_error)
    target_edges = round(min_edges + (max_edges - min_edges) * 0.5)
    num_iters = 0
    max_iters = 10
    max_error = 0.08
    init_iters_incr_error = 3
    min_factor = 0.005

    while (num_iters < max_iters):
        print("Trying with factor", current_factor)
        try:
            mesh = cadmesh.mesh_model(stepPath, max_size=current_factor, terminal=0)
        except Exception as error:
            gmsh.finalize()
            raise error
        # face_count = getMeshFaceCount(mesh)
        # if (face_count == 0):
        #     # restart gmsh, sometimes it deadlocks
        #     print("Internal gmsh error! Restarting gmsh")
        #     geom.__exit__()
        #     geom.__enter__()
        #     continue
        curr_num_edges = getEstMeshEdgeCount(mesh)
        edge_count = getExactMeshEdgeCount(mesh)
        if min_edges <= edge_count <= max_edges:
            print("Target edges MET with factor {}: {} [{},{}]".format(current_factor, curr_num_edges, min_edges, max_edges))
            #geom.save_geometry("test.msh")
            return mesh

        if (current_factor >= 1 and curr_num_edges > max_edges):
            raise Exception("Unexpectedly large mesh size for mesh factor {}: {}".format(current_factor, curr_num_edges))

        print("Target edge count missed with factor {}: {} [{},{}]".format(current_factor, curr_num_edges,min_edges,max_edges))

        if (num_iters > init_iters_incr_error and curr_num_edges < max_edges):
            new_error_limit = (num_iters - init_iters_incr_error)/(max_iters-init_iters_incr_error)* (max_error - init_mesh_size_error)+ init_mesh_size_error
            print("New error limit {}%".format(new_error_limit*100))
            error = (max_edges - curr_num_edges) / max_edges
            print("Current error {}%".format(error * 100))
            if (error <= new_error_limit):
                print("Good enough approximation found with factor {}: {}, error {}%".format(current_factor, curr_num_edges, error * 100))
                return mesh

        if (num_iters+1 >= max_iters):
            break

        current_inv_factor = 1.0/current_factor

        ins_idx = bisect.bisect_left(inv_factors, current_inv_factor)
        inv_factors.insert(ins_idx, current_inv_factor)
        num_edges.insert(ins_idx, curr_num_edges)

        if (len(inv_factors) == 1):
            current_factor = current_factor / 2
            print("inv factors:", inv_factors)
            print("num edges:", num_edges)
        else:
            if (len(inv_factors) > 2):
                if (num_edges[1] < target_edges):
                    inv_factors = inv_factors[1:]
                    num_edges = num_edges[1:]
                else:
                    inv_factors = inv_factors[:2]
                    num_edges = num_edges[:2]

                # target_idx = bisect.bisect_left(num_edges, target_edges)
                # max_idx = min(target_idx + 1, len(num_edges))
                # right_size_length = max_idx - target_idx
                # left_size_length = 2 - right_size_length
                # min_idx = target_idx - left_size_length
                # inv_factors = inv_factors[min_idx: max_idx]
                # num_edges = num_edges[min_idx: max_idx]

            print("inv factors:", inv_factors)
            print("num edges:", num_edges)
            interpolatedFactor = interpolateFactor(inv_factors,num_edges,target_edges)
            current_factor = max(interpolatedFactor,min_factor) if interpolatedFactor != None else min_factor
        num_iters += 1

    raise Exception("Didn't converge to target edges in 20 iterations")

def getObjLinks(datasetRoot):
    objLinkPaths = []
    for root, _, fnames in os.walk(datasetRoot):
        for fname in fnames:
            if (os.path.splitext(fname)[1] == ".obj"):
                objLinkPaths.append(os.path.join(root, fname))
    return objLinkPaths

def objPathToStepPath(objPath):
    pathToObjDir = os.path.split(objPath)[0]
    globalObjPath, sampleId = os.path.split(pathToObjDir)
    datasetPath = os.path.split(globalObjPath)[0]
    stepDirPath = os.path.join(datasetPath,"step",sampleId)
    return os.path.join(stepDirPath, next(os.walk(stepDirPath))[2][0])

def createStepLinks(stepPaths,dstStepPath):
    linkPaths = []
    for stepPath in stepPaths:
        fName = os.path.basename(stepPath)
        fNumber = fName.split("_")[0]
        dstPath = os.path.join(dstStepPath, fNumber)
        Path(dstPath).mkdir(exist_ok=True)
        linkPath = os.path.join(dstPath,fName)
        if (not os.path.exists(linkPath)):
            os.symlink(stepPath,linkPath)
        linkPaths.append(linkPath)
    return linkPaths

def writeFeat(path, mesh):
    print(path)
    with open(path, "w") as fili:
        yaml.dump(mesh["features"], fili, indent=2)

def writeObj(path, mesh):
    verts = mesh["vertices"]
    faces = mesh["faces"]
    print("Generated model with %i vertices and %i faces." % (len(verts), len(faces)))
    with open(path, "w") as fili:
        for v in verts:
            fili.write("v %f %f %f\n" % (v[0], v[1], v[2]))
        for f in faces:
            fili.write("f %i//%i %i//%i %i//%i\n" % (f[0], f[3], f[1], f[4], f[2], f[5]))

def thread_function(stepLinks, meshSizeInitFactor):
    for stepLink in stepLinks:
        print("Remesh {}".format(stepLink))
        numberDir = os.path.basename(os.path.dirname(stepLink))
        stepFileName = os.path.basename(stepLink)
        featFileName = os.path.splitext(stepFileName.replace("_step_", "_features_"))[0] + ".yml"
        objFileName = os.path.splitext(stepFileName.replace("_step_", "_trimesh_"))[0] + ".obj"
        featPath = os.path.join(os.path.join(dstFeatPath, numberDir), featFileName)
        objPath = os.path.join(os.path.join(dstObjPath, numberDir), objFileName)
        if (os.path.exists(featPath) and os.path.exists(objPath)):
            continue
        try:
            mesh = createMeshWithEdgeCount(stepLink, targetEdges, meshSizeInitFactor)
        except Exception as error:
            print("Cannot mesh model {}: {}".format(stepLink, error))
            continue
        Path(os.path.dirname(featPath)).mkdir()
        Path(os.path.dirname(objPath)).mkdir()
        writeFeat(featPath, mesh)
        writeObj(objPath, mesh)

if __name__ == '__main__':

    parser = argparse.ArgumentParser("Remesh dataset of target edge count")
    parser.add_argument('--src', required=True, type=str, help="Path where source dataset is located")
    parser.add_argument('--dst', required=True, type=str, help="Path where dataset will be saved")
    parser.add_argument('--targetEdges', required=True, type=int, help="Number of edges in a mesh")
    parser.add_argument('--meshSizeInitFactor', type=float, default=0.2, help="Initial mesh size factor")
    parser.add_argument('--numThreads', type=int, default=7, help="Number of threads")

    args = parser.parse_args()

    srcPath = args.src
    dstPath = args.dst
    targetEdges = args.targetEdges
    meshSizeInitFactor = args.meshSizeInitFactor
    numThreads = args.numThreads

    print("Remesh dataset from {} to {} with #edges {}".format(srcPath,dstPath,targetEdges))

    objPaths = list(map(lambda link: os.readlink(link),getObjLinks(srcPath)))
    stepPaths = list(map(objPathToStepPath,objPaths))

    dstObjPath = os.path.join(dstPath,"obj")
    dstStepPath = os.path.join(dstPath,"step")
    dstFeatPath = os.path.join(dstPath,"feat")
    try:
        Path(dstObjPath).mkdir(parents=True, exist_ok=True)
        Path(dstStepPath).mkdir(parents=True, exist_ok=True)
        Path(dstFeatPath).mkdir(parents=True, exist_ok=True)
    except FileExistsError as f_error:
        print(f_error)
        exit(1)

    stepLinks =  createStepLinks(stepPaths, dstStepPath)


    thLinkCount = int(len(stepLinks) / numThreads)

    threads = []
    for index in range(numThreads):
        thListBegin = index*thLinkCount
        thListEnd = len(stepLinks) if index == (numThreads-1) else thListBegin+thLinkCount
        print("Process {} takes models [{},{}]".format(index,thListBegin,thListEnd))
        thList = stepLinks[thListBegin:thListEnd]
        th = Process(target=thread_function, args=(thList,meshSizeInitFactor))
        threads.append(th)
        th.start()

    for th in threads:
        th.join()


