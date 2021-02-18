# Script to delete dataset samples with meshes larger than a given # of edges or without segmentation data (feat files).

import os, sys, yaml, shutil
from pathlib import Path

def parseYamlFile(path):
    data = None
    with open(path, 'r') as stream:
        try:
            data = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    return data

def takeMesh(statPath,maxEdges):
    for root, _, fnames in sorted(os.walk(statPath)):
        for fname in fnames:
            if (os.path.splitext(fname)[1] == ".yml"):
                path = os.path.join(root,fname)
                data = parseYamlFile(path)
                faces = data['#faces']
                edges = faces*1.5
                if (edges>maxEdges):
                    print("Removing {} with {} faces, {} edges".format(os.path.relpath(path,statPath),faces,edges))
                    return True
                else:
                    return False

def addMesh(srcPath, dstPath):
    os.symlink(os.path.abspath(srcPath),os.path.join(dstPath,os.path.basename(srcPath)))
    return

def findSamples(srcPath,objPath,maxEdges):
    sampleSrcPaths = []
    totalSize = 0
    for root, _, fnames in sorted(os.walk(objPath)):
        for fname in fnames:
            path = os.path.join(root, fname)
            if (os.path.splitext(fname)[1] == ".obj"):
                basedir = os.path.basename(root)
                statPath = os.path.join(srcPath, "stat", basedir)
                if (takeMesh(statPath, maxEdges)):
                    sampleSrcPaths.append(path)
                    fileSize = os.path.getsize(path) / (1024*1024)
                    print("Size: {} MB".format(fileSize))
                    totalSize += fileSize
    return sampleSrcPaths, totalSize

def removeLargeSamples():
    srcPath = sys.argv[1]
    maxEdges = int(sys.argv[2])

    print("Removing samples from {} with more than {} edges".format(srcPath, maxEdges))

    objPath = os.path.join(srcPath, "obj")

    if (not os.path.exists(objPath)):
        print(srcPath, "does not exist")
        exit(1)

    sampleSrcPaths, totalSize = findSamples(srcPath, objPath, maxEdges)

    if (len(sampleSrcPaths) > 0):
        logPath = os.path.join(srcPath, "delete_log.txt")
        deleteLog = open(logPath, 'a+')
        for samplePath in sampleSrcPaths:
            os.remove(samplePath)
            deleteLog.write(os.path.relpath(samplePath, srcPath) + "\n")

    print("Removing {} samples with total size {} MB".format(len(sampleSrcPaths), totalSize))

def removeEmptyDirs(rootPath):
    for root, dirs, fnames in sorted(os.walk(rootPath)):
        if (len(fnames) == 0 and len(dirs) == 0):
            print("Remove empty directory", root)
            os.rmdir(root)


def syncFiles(type, deleteObjWoTypeFile):
    srcPath = sys.argv[1]

    print("Removing samples from {} with no feat files and orphan {} files".format(srcPath, type))

    objPath = os.path.join(srcPath, "obj")
    typePath = os.path.join(srcPath, type)
    if (not os.path.exists(objPath)):
        print(objPath, "does not exist")
        exit(1)

    if (not os.path.exists(typePath)):
        print(typePath, "does not exist")
        exit(1)

    removeEmptyDirs(objPath)
    removeEmptyDirs(typePath)

    objIds = next(os.walk(objPath))[1]
    typeIds = next(os.walk(typePath))[1]

    incompleteIds = sorted(set(objIds).symmetric_difference(typeIds)) if deleteObjWoTypeFile else sorted(set(typeIds).difference(objIds))

    deleteLog = open(os.path.join(srcPath, "delete_log.txt"), 'a+')

    for incompleteId in incompleteIds:
        sampleObjPath = os.path.join(objPath,str(incompleteId))
        sampleTypePath =os.path.join(typePath,str(incompleteId))

        hasObj = os.path.exists(sampleObjPath) and len(os.listdir(sampleObjPath)) > 0
        hasType = os.path.exists(sampleTypePath) and len(os.listdir(sampleTypePath)) > 0
        if(not hasObj):
            typeFilePath = os.path.join(sampleTypePath, next(os.walk(sampleTypePath))[2][0])
            assert hasType,"Sample should have {} file".format(type)
            print("Remove orphan features", typeFilePath)
            shutil.rmtree(sampleTypePath)
        elif(not hasType):
            objFilePath = os.path.join( sampleObjPath, next(os.walk(sampleObjPath))[2][0])
            assert hasObj, "Sample should have obj file"
            print("Remove object with no {} file: {}".format(type,objFilePath))
            shutil.rmtree(sampleObjPath)
            deleteLog.write(os.path.relpath(sampleObjPath, srcPath) + "\n")
        else:
            assert False,"Sample should be incomplete"



if len(sys.argv) == 3:
    removeLargeSamples()
elif len(sys.argv) == 2:
    syncFiles("feat", True)
    syncFiles("step", False)
else:
    print("Wrong parameters")
    exit(1)



