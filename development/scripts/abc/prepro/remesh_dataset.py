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
from multiprocessing import current_process, active_children
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
import math
import random
import signal
import resource
import traceback

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

def hasOnlyPlaneAndCylinderSurfs(mesh):
    surfaces = mesh["features"]["surfaces"]
    for surface in surfaces:
        type = surface["type"]
        if (type != "Plane" and type != "Cylinder"):
            return False
    return True

init_mesh_size_error = 0.005

def createMeshWithEdgeCount(stepPath, max_edges, meshSizeInitFactor):
    # Fit the function 1/mesh_size_factor -> mesh edges to a quadratic polynomial.
    inv_factors = []
    num_edges = []
    current_factor = meshSizeInitFactor
    min_edges = max_edges*(1-init_mesh_size_error)
    target_edges = round(min_edges + (max_edges - min_edges) * 0.5)
    num_iters = 0
    max_iters = 40
    max_error = 0.02
    init_iters_incr_error = 10
    min_factor = 0.005

    while (num_iters < max_iters):
        #print("Trying with factor", current_factor)
        try:
            mesh = cadmesh.mesh_model(stepPath, max_size=current_factor, terminal=0)
        except Exception as error:
            try:
                gmsh.finalize()
            except Exception as error:
                print(error)
            os.remove(stepPath)
            raise Exception("Deleting {} due to error: {}".format(stepPath,error))
        # face_count = getMeshFaceCount(mesh)
        # if (face_count == 0):
        #     # restart gmsh, sometimes it deadlocks
        #     print("Internal gmsh error! Restarting gmsh")
        #     geom.__exit__()
        #     geom.__enter__()
        #     continue
        if (mesh == None):
            os.remove(stepPath)
            raise Exception("Deleting invalid model {}".format(os.path.basename(stepPath)))

        if (hasOnlyPlaneAndCylinderSurfs(mesh)):
            os.remove(stepPath)
            raise Exception("Deleting {}, it only has plane and cylinder surfaces".format(os.path.basename(stepPath)))

        curr_num_edges = getEstMeshEdgeCount(mesh)

        if (current_factor >= 1 and curr_num_edges > max_edges):
            os.remove(stepPath)
            raise Exception("Deleting unexpectedly large mesh size for mesh factor {}: {}".format(current_factor, curr_num_edges))

        try:
            edge_count = getExactMeshEdgeCount(mesh)
        except Exception as error:
            #os.remove(stepPath)
            print("{} non-manifold? ({})".format(os.path.basename(stepPath),error))
        else:
            if min_edges <= edge_count <= max_edges:
                print("Target edges MET with factor {}: {} [{},{}]".format(current_factor, curr_num_edges, min_edges, max_edges))
                #geom.save_geometry("test.msh")
                return mesh
        if (num_iters > init_iters_incr_error and curr_num_edges < max_edges):
            new_error_limit = (num_iters - init_iters_incr_error)/(max_iters-init_iters_incr_error)* (max_error - init_mesh_size_error)+ init_mesh_size_error
            #print("New error limit {}%".format(new_error_limit*100))
            error = (max_edges - curr_num_edges) / max_edges
            #print("Current error {}%".format(error * 100))
            if (error <= new_error_limit):
                print("Good enough approximation found with factor {}: {}, error {}%".format(current_factor, curr_num_edges, error * 100))
                return mesh

        #print("Target edge count missed with factor {}: {} [{},{}]".format(current_factor, curr_num_edges,min_edges,max_edges))

        if (num_iters+1 >= max_iters):
            break

        current_inv_factor = 1.0/current_factor

        ins_idx = bisect.bisect_left(inv_factors, current_inv_factor)
        inv_factors.insert(ins_idx, current_inv_factor)
        num_edges.insert(ins_idx, curr_num_edges)

        if (len(inv_factors) == 1):
            current_factor = current_factor / 2 if current_factor <= 1 else 1
            #print("inv factors:", inv_factors)
            #print("num edges:", num_edges)
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

            #print("inv factors:", inv_factors)
            #print("num edges:", num_edges)
            interpolatedFactor = interpolateFactor(inv_factors,num_edges,target_edges)
            current_factor = max(interpolatedFactor,min_factor) if interpolatedFactor != None else min_factor
        num_iters += 1

    raise Exception("Didn't converge to target edges in {} iterations".format(max_iters))

def getStepPaths(datasetRoot):
    stepDir = os.path.join(datasetRoot, "step")
    if (not os.path.exists(stepDir)):
        return None
    stepPaths = []
    for root, _, fnames in os.walk(stepDir):
        for fname in fnames:
            if (os.path.splitext(fname)[1] == ".step"):
                stepPaths.append(os.path.join(root, fname))
    return stepPaths

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

def parseYamlFile(featPath):
    data = None
    with open(featPath, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data

workerTerminated = False

def sigHandler(signum, frame):
    global workerTerminated
    print("Worker termination delayed while writing results")
    workerTerminated = True

def workerFunc(stepLinks, meshSizeInitFactor, dstFeatPath, dstObjPath, targetEdges):
    global workerTerminated

    memLimit = 4*2**30 # set max memory per worker to 4 GB
    resource.setrlimit(resource.RLIMIT_DATA, (memLimit, memLimit))

    origHandler = signal.getsignal(signal.SIGTERM)

    for stepLink in stepLinks:
        print("{}: Remesh {}".format(current_process().name, stepLink))
        numberDir = os.path.basename(os.path.dirname(stepLink))
        stepFileName = os.path.basename(stepLink)
        featFileName = os.path.splitext(stepFileName.replace("_step_", "_features_"))[0] + ".yml"
        objFileName = os.path.splitext(stepFileName.replace("_step_", "_trimesh_"))[0] + ".obj"
        featPath = os.path.join(os.path.join(dstFeatPath, numberDir), featFileName)
        objPath = os.path.join(os.path.join(dstObjPath, numberDir), objFileName)

        if (os.path.exists(featPath) and os.path.exists(objPath)):
            print("Skipping already existing mesh for", stepLink)
            continue

        try:
            mesh = createMeshWithEdgeCount(stepLink, targetEdges, meshSizeInitFactor)
        except Exception as error:
            print("Cannot mesh model {}: {}".format(os.path.basename(stepLink), error))
            #print(traceback.print_exc())
            continue

        # Keep worker from getting killed while writing results
        signal.signal(signal.SIGTERM, sigHandler)

        Path(os.path.dirname(featPath)).mkdir()
        Path(os.path.dirname(objPath)).mkdir()
        writeFeat(featPath, mesh)
        writeObj(objPath, mesh)

        signal.signal(signal.SIGTERM,origHandler)
        if (workerTerminated):
            return

def getPendingModels(stepPaths, completedTaskIds, taskSize):
    pendingStepPaths = []
    completedTaskIds.sort()
    firstTaskIdx = 0
    for taskId in completedTaskIds:
        lastTaskIdx = taskId*taskSize
        for i in range(firstTaskIdx,lastTaskIdx):
            stepPath = stepPaths[i]
            if (os.path.exists(stepPath)):
                pendingStepPaths.append(stepPath)
        firstTaskIdx = lastTaskIdx+taskSize
    for i in range(firstTaskIdx, len(stepPaths)):
        stepPath = stepPaths[i]
        if (os.path.exists(stepPath)):
            pendingStepPaths.append(stepPath)
    return pendingStepPaths


watchdogRefreshed = True

def killPoolWorkers(workerIds):
    for workerId in workerIds:
        print('Killing worker {}'.format(workerId))
        try:
            os.kill(workerId, signal.SIGTERM)
        except Exception:
            continue

def tickWatchdog(timeoutSecs, workerIds):
    global watchdogRefreshed

    if watchdogRefreshed:
        watchdogRefreshed = False
        threading.Timer(timeoutSecs, tickWatchdog, args=(timeoutSecs, workerIds)).start()
    else:
        print("Timeout after {} secs, killing workers".format(timeoutSecs))
        killPoolWorkers(workerIds)

def meshStepFilesInPool(stepPaths, numThreads, meshSizeInitFactor, dstFeatPath, dstObjPath, targetEdges, totalFilesDone):
    global watchdogRefreshed

    finishedTasks = []
    futureToTaskId = {}
    numFiles = len(stepPaths)
    taskSize = 10 #min(10, max(5, int(numFiles / 8000)))
    timeoutSecs = taskSize * 40
    taskCount = math.ceil(numFiles / taskSize)
    print("Scheduling {} tasks of {} models each".format(taskCount, taskSize))

    executor = ProcessPoolExecutor(max_workers=numThreads)

    for i in range(0, numFiles, taskSize):
        taskId = int(i / taskSize)
        lastPath = min(numFiles, i + taskSize)
        taskInput = stepPaths[i: lastPath]
        # print("Task {} models [{},{}]".format(taskId, i, lastPath))
        futureToTaskId[executor.submit(workerFunc, taskInput, meshSizeInitFactor, dstFeatPath, dstObjPath, targetEdges)] = taskId

    workerIds = [child.pid for child in active_children()]
    for workerId in workerIds:
        print('Worker pid is {}'.format(workerId))

    print("Setting timeout to {} seconds".format(timeoutSecs))
    watchdogRefreshed = True
    watchdog = threading.Timer(timeoutSecs, tickWatchdog, args=(timeoutSecs, workerIds))
    watchdog.start()

    try:
        for future in as_completed(futureToTaskId):
            taskId = futureToTaskId[future]
            future.result()
            watchdogRefreshed = True
            print("Task {} finished".format(taskId))
            finishedTasks.append(taskId)
            totalFilesDone += taskSize
            print("TOTAL FILES DONE: {}".format(totalFilesDone))
    except Exception as error:
        if (isinstance(error, BrokenProcessPool)):
            print("Worker killed:", error)
        else:
            print("Unknown error:", error)
        print("Shutting down pool")
        for future in futureToTaskId:
            future.cancel()
        watchdog.cancel()
        executor.shutdown(wait=False)
        killPoolWorkers(workerIds)
        print("Shutdown finished, reinit new Pool")
        return getPendingModels(stepPaths, finishedTasks, taskSize), totalFilesDone

    watchdog.cancel()
    executor.shutdown(wait=True)
    return [], totalFilesDone

def meshStepFiles(stepPaths, numThreads, meshSizeInitFactor, dstFeatPath, dstObjPath, targetEdges):

   stepPathsLeft = stepPaths
   totalFilesDone = 0
   while(len(stepPathsLeft) > 0):
       random.shuffle(stepPathsLeft)
       print("Models left: {}, done: {}".format(len(stepPathsLeft), totalFilesDone))
       stepPathsLeft, totalFilesDone = meshStepFilesInPool(stepPathsLeft, numThreads, meshSizeInitFactor,
                                                           dstFeatPath, dstObjPath, targetEdges, totalFilesDone)

def skipModel(statPath, maxFaceCount, maxSurfCount):
    stats = parseYamlFile(statPath)
    if (stats == None):
        return False

    faceCount = stats['#faces']
    surfCount = stats['#surfs']

    if (faceCount > maxFaceCount or surfCount > maxSurfCount):
        print("Model {} is too large: #faces {} (max {}), #surfs {} (max {})".format(os.path.basename(statPath),
                                                                                     faceCount, maxFaceCount, surfCount,
                                                                                     maxSurfCount))
        return True

    surfs = stats['surfs']
    surfTypeCounts = {surfType : 0 for surfType in surfaceTypes}
    for surf in surfs:
        surfTypeCounts[surf] = surfTypeCounts[surf] + 1
    if (surfTypeCounts['Plane'] + surfTypeCounts['Cylinder'] == len(surfs)):
        print("Model {} has only planes and cylinders".format(os.path.basename(statPath)))
        return True
    return False

def filterModels(stepPaths, statDir, maxFaceCount, maxSurfCount, dstFeatPath, dstObjPath,):
    filteredStepPaths = []
    for stepPath in stepPaths:
        stepFileName = os.path.basename(stepPath)
        stepFileNumber = stepFileName.split("_")[0]
        statPath = os.path.join(statDir, os.path.join(stepFileNumber,
                                                      os.path.splitext(stepFileName.replace("_step_", "_stats_"))[
                                                          0] + ".yml"))
        if (not os.path.exists(statPath)):
            continue

        numberDir = os.path.basename(os.path.dirname(stepPath))
        stepFileName = os.path.basename(stepPath)
        featFileName = os.path.splitext(stepFileName.replace("_step_", "_features_"))[0] + ".yml"
        objFileName = os.path.splitext(stepFileName.replace("_step_", "_trimesh_"))[0] + ".obj"
        featPath = os.path.join(os.path.join(dstFeatPath, numberDir), featFileName)
        objPath = os.path.join(os.path.join(dstObjPath, numberDir), objFileName)

        if (os.path.exists(featPath) and os.path.exists(objPath)):
            #print("Step {} already meshed".format(stepPath))
            continue

        filteredStepPaths.append(stepPath)
        # if (skipModel(statPath,maxFaceCount,maxSurfCount)):
        #     os.remove(stepPath)
        # else:
        #     filteredStepPaths.append(stepPath)
    return filteredStepPaths

def main():
    parser = argparse.ArgumentParser("Remesh dataset of target edge count")
    parser.add_argument('--src', required=True, type=str, help="Path where source dataset is located")
    parser.add_argument('--dst', required=True, type=str, help="Path where dataset will be saved")
    parser.add_argument('--targetEdges', required=True, type=int, help="Number of edges in a mesh")
    parser.add_argument('--meshSizeInitFactor', type=float, default=0.2, help="Initial mesh size factor")
    parser.add_argument('--numThreads', type=int, default=7, help="Number of threads")
    parser.add_argument('--maxFaceCount', type=int, default=math.inf, help="Max face count in original ABC mesh, larger models will be skipped")
    parser.add_argument('--maxSurfCount', type=int, default=150, help="Max surface count, larger models will be skipped")

    args = parser.parse_args()

    srcPath = args.src
    dstPath = args.dst
    targetEdges = args.targetEdges
    meshSizeInitFactor = args.meshSizeInitFactor
    numThreads = args.numThreads
    maxFaceCount = args.maxFaceCount
    maxSurfCount = args.maxSurfCount

    print("Remesh dataset from {} to {} with #edges {}".format(srcPath,dstPath,targetEdges))

    absSrcPath = os.path.abspath(srcPath)
    stepPaths = getStepPaths(absSrcPath)
    if (stepPaths == None):
        objPaths = list(map(lambda link: os.readlink(link),getObjLinks(srcPath)))
        stepPaths = list(map(objPathToStepPath,objPaths))

    #stepPaths.sort()
    statDir = os.path.join(absSrcPath, "stat")
    dstObjPath = os.path.join(dstPath,"obj")
    dstFeatPath = os.path.join(dstPath,"feat")
    stepPaths = filterModels(stepPaths, statDir, maxFaceCount, maxSurfCount,  dstFeatPath, dstObjPath)

    try:
        Path(dstObjPath).mkdir(parents=True, exist_ok=True)
        Path(dstFeatPath).mkdir(parents=True, exist_ok=True)
    except FileExistsError as f_error:
        print(f_error)
        exit(1)

    meshStepFiles(stepPaths, numThreads, meshSizeInitFactor, dstFeatPath, dstObjPath, targetEdges)

if __name__ == '__main__':
    main()
