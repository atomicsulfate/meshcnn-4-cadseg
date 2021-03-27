
import os, sys, yaml
from pathlib import Path
import argparse
import math
import numpy as np
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..")))

surfaceTypes = ['Plane','Revolution', 'Cylinder','Extrusion','Cone','Other','Sphere','Torus','BSpline']

surfaceTypesToAugment = ['Revolution', 'Extrusion','Cone','Sphere','Torus','BSpline']

def getObjs(datasetRoot, filterFunc = None):
    objPaths = []
    for root, _, fnames in os.walk(datasetRoot):
        for fname in fnames:
            if (os.path.splitext(fname)[1] == ".obj") and (filterFunc == None or filterFunc(fname)):
                objPaths.append(os.path.join(root, fname))
    return objPaths

def addMesh(srcPath, dstPath):
    os.symlink(os.path.abspath(srcPath),os.path.join(dstPath,os.path.basename(srcPath)))
    return

def linkSegFiles(objPath, srcSegPath, srcSsegPath, dstSegPath, dstSsegPath):
    fileNamePrefix = os.path.splitext(os.path.basename(objPath))[0]
    srcSegFilePath = os.path.join(srcSegPath, fileNamePrefix +  ".eseg")
    srcSsegFilePath = os.path.join(srcSsegPath, fileNamePrefix + ".seseg")
    dstSegFilePath = os.path.join(dstSegPath, os.path.basename(srcSegFilePath))
    dstSsegFilePath = os.path.join(dstSsegPath, os.path.basename(srcSsegFilePath))
    if (os.path.exists(dstSegFilePath)):
        os.remove(dstSegFilePath)
    os.symlink(os.path.abspath(srcSegFilePath), dstSegFilePath)
    if (os.path.exists(dstSsegFilePath)):
        os.remove(dstSsegFilePath)
    os.symlink(os.path.abspath(srcSsegFilePath), dstSsegFilePath)


parser = argparse.ArgumentParser("Augment ABC dataset with synthetic samples")
parser.add_argument('--src', required=True, type=str, help="Path where source ABC dataset is located.")
parser.add_argument('--synth', required=True, type=str, help="Path where synthetic samples are located.")
parser.add_argument('--dst', required=True, type=str, help="Path where augmented ABC dataset will be placed.")
parser.add_argument('--numSamples', required=True, type=int, help="Num synthetic samples per surface type.")
parser.add_argument('--ratio', type=float, default=0.2, help="#test samples/#train samples")

args = parser.parse_args()
srcPath = args.src
synthPath = args.synth
dstPath = args.dst
ratio = args.ratio
numSamplesPerType = args.numSamples

print("Augment dataset in {}, into {} with {} samples per surface type from {}, test/train ratio {}".format(srcPath,dstPath, numSamplesPerType, synthPath, ratio))

dstTrainPath = os.path.join(dstPath, "train")
dstTestPath = os.path.join(dstPath, "test")
targetSegPath = os.path.join(dstPath, "seg")
targetSsegPath = os.path.join(dstPath, "sseg")

if (not os.path.exists(srcPath)):
    print(srcPath, "does not exist")
    exit(1)

if (not os.path.exists(synthPath)):
    print(synthPath, "does not exist")
    exit(1)

objLinks = getObjs(srcPath)

if (len(objLinks) == 0):
    print("No obj files found in", srcPath)
    exit(1)

objFiles = list(map(lambda link: os.readlink(link),objLinks)) if os.path.islink(objLinks[0]) else objLinks

def filterAugObj(fname):
    name = os.path.splitext(fname)[0]
    surfaceType, number = name.split("_")
    return (surfaceType in surfaceTypesToAugment) and (int(number) < numSamplesPerType)

augObjFiles = getObjs(synthPath,filterAugObj)

if (len(augObjFiles) == 0):
    print("No obj files found in", synthPath)
    exit(1)

srcDatasetRoot = os.path.split(os.path.split(os.path.split(objFiles[0])[0])[0])[0]
srcsegPath = os.path.join(srcDatasetRoot, "seg")
if (not os.path.exists(srcsegPath)):
    print(srcsegPath, "does not exist")
    exit(1)
srcSsegPath = os.path.join(srcDatasetRoot, "sseg")
if (not os.path.exists(srcSsegPath)):
    print(srcSsegPath, "does not exist")
    exit(1)

augDatasetRoot = os.path.split(os.path.split(augObjFiles[0])[0])[0]
augSegPath = os.path.join(augDatasetRoot, "seg")
if (not os.path.exists(augSegPath)):
    print(augSegPath, "does not exist")
    exit(1)
augSsegPath = os.path.join(augDatasetRoot, "sseg")
if (not os.path.exists(augSsegPath)):
    print(augSsegPath, "does not exist")
    exit(1)

try:
    Path(dstTrainPath).mkdir(parents=True, exist_ok=True)
    Path(dstTestPath).mkdir(parents=True, exist_ok=True)
    Path(targetSegPath).mkdir(parents=True, exist_ok=True)
    Path(targetSsegPath).mkdir(parents=True, exist_ok=True)
except FileExistsError as f_error:
    print(f_error)
    exit(1)

random.shuffle(objFiles)
numABCSamples = len(objFiles)
numTestSamples = int(numABCSamples * ratio)
numABCTrainSamples = numABCSamples - numTestSamples
numAugSamples = len(augObjFiles)
numTotalSamples = numAugSamples + numABCSamples

for objPath in objFiles:
    linkSegFiles(objPath, srcsegPath, srcSsegPath, targetSegPath, targetSsegPath)

for objPath in augObjFiles:
    linkSegFiles(objPath, augSegPath, augSsegPath, targetSegPath, targetSsegPath)

for samplePath in objFiles[:numABCTrainSamples]:
    addMesh(samplePath, dstTrainPath)

for samplePath in objFiles[numABCTrainSamples:]:
    addMesh(samplePath, dstTestPath)

for samplePath in augObjFiles:
    addMesh(samplePath, dstTrainPath)

print("Dataset split with {} samples ({}+{} train, {} test)".format(numTotalSamples, numABCTrainSamples,numAugSamples,
                                                                    numTestSamples))


