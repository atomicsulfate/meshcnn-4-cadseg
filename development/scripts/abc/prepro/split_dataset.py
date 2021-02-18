
import os, sys, yaml
from pathlib import Path
import argparse
import math
import numpy as np
import random
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..")))

def getObjLinks(datasetRoot):
    objLinkPaths = []
    for root, _, fnames in os.walk(datasetRoot):
        for fname in fnames:
            if (os.path.splitext(fname)[1] == ".obj"):
                objLinkPaths.append(os.path.join(root, fname))
    return objLinkPaths

def addMesh(srcPath, dstPath):
    os.symlink(os.path.abspath(srcPath),os.path.join(dstPath,os.path.basename(srcPath)))
    return

parser = argparse.ArgumentParser("Split dataset into random test and train groups with the given ratio")
parser.add_argument('--src', required=True, type=str, help="Path where source ABC dataset is located.")
parser.add_argument('--dst', required=True, type=str, help="Path where resplit ABC dataset will be placed.")
parser.add_argument('--ratio', type=float, default=0.2, help="#test samples/#train samples")

args = parser.parse_args()
srcPath = args.src
dstPath = args.dst
ratio = args.ratio

print("Split dataset in {}, into {} with test/train ratio {}".format(srcPath,dstPath, ratio))

dstTrainPath = os.path.join(dstPath, "train")
dstTestPath = os.path.join(dstPath, "test")

if (not os.path.exists(srcPath)):
    print(srcPath, "does not exist")
    exit(1)

objLinks = getObjLinks(srcPath)

if (len(objLinks) == 0):
    print("No obj files found in", srcPath)
    exit(1)

objFiles = list(map(lambda link: os.readlink(link),objLinks))

try:
    Path(dstTrainPath).mkdir(parents=True, exist_ok=False)
    Path(dstTestPath).mkdir(parents=True, exist_ok=False)
except FileExistsError as f_error:
    print(f_error)
    exit(1)

random.shuffle(objFiles)
numTotalSamples = len(objFiles)
numTestSamples = int(numTotalSamples * ratio)
numTrainSamples = numTotalSamples - numTestSamples

sampleIdx = 0
for samplePath in objFiles[:numTrainSamples]:
    addMesh(samplePath, dstTrainPath)

for samplePath in objFiles[numTrainSamples:]:
    addMesh(samplePath, dstTestPath)

print("Dataset split with {} samples ({} train, {} test)".format(numTotalSamples,numTrainSamples,numTestSamples))


