
import os, sys, yaml
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
                if (edges<=maxEdges):
                    print("Adding {} with {} faces and {} edges".format(path,faces,edges))
                    return True
                else:
                    return False

def addMesh(srcPath, dstPath):
    os.symlink(os.path.abspath(srcPath),os.path.join(dstPath,os.path.basename(srcPath)))
    return

def findSamples(objPath,maxEdges,maxSamples):
    numSamples = 0
    sampleSrcPaths = []
    for root, _, fnames in sorted(os.walk(objPath)):
        for fname in fnames:
            path = os.path.join(root, fname)
            if (os.path.splitext(fname)[1] == ".obj"):
                basedir = os.path.basename(root)
                statPath = os.path.join(srcPath, "stat", basedir)
                if (takeMesh(statPath, maxEdges)):
                    sampleSrcPaths.append(path)
                    numSamples += 1
                    if (numSamples >= maxSamples):
                        return sampleSrcPaths
    return sampleSrcPaths

if len(sys.argv) < 4:
    print("Wrong parameters")
    exit(1)

srcPath,dstPath = sys.argv[1:3]
maxEdges = int(sys.argv[3])
maxSamples = int(sys.argv[4])

print("Compile dataset from {} to {} with max {} edges and {} samples".format(srcPath,dstPath,maxEdges,maxSamples))

objPath = os.path.join(srcPath,"obj")

if (not os.path.exists(objPath)):
    print(srcPath,"does not exist")
    exit(1)

dstTrainPath = os.path.join(dstPath,"train")
dstTestPath = os.path.join(dstPath,"test")

try:
    Path(dstTrainPath).mkdir(parents=True, exist_ok=False)
    Path(dstTestPath).mkdir(parents=True, exist_ok=False)
except FileExistsError as f_error:
    print(f_error)
    exit(1)

sampleSrcPaths = findSamples(objPath,maxEdges,maxSamples)

testSamplesRatio = 0.1
numTotalSamples = len(sampleSrcPaths)
numTestSamples = int(numTotalSamples * 0.1)
numTrainSamples = numTotalSamples - numTestSamples

sampleIdx = 0
for samplePath in sampleSrcPaths[:numTrainSamples]:
    addMesh(samplePath, dstTrainPath)

for samplePath in sampleSrcPaths[numTrainSamples:]:
    addMesh(samplePath, dstTestPath)

print("Dataset compiled with {} samples ({} train, {} test)".format(numTotalSamples,numTrainSamples,numTestSamples))


