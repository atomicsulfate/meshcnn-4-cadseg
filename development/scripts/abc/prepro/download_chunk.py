
import os, sys
from pathlib import Path

def downloadChunk(chunk, chunkType, datasetRoot):
    typeRoot = os.path.join(datasetRoot, chunkType)
    url, filename = chunk.split()
    dstPath = os.path.join(typeRoot,filename)

    if (not os.path.exists(dstPath)):
        print('Downloading chunk from {} to {}'.format(url,dstPath))
        os.system("curl --insecure -o {} {}".format(dstPath, url))

    os.system("7z x {} -o{}".format(dstPath, typeRoot))
    os.system("rm {}".format(dstPath))

def downloadChunkType(chunkType, chunkIndex, datasetRoot):
    indexFileName = chunkType + "_v00.txt"
    path = os.path.join(datasetRoot,indexFileName)
    file = open(path, "r")
    chunks = file.readlines()

    if( chunkIndex == None):
        for chunk in chunks:
            downloadChunk(chunk, chunkType, datasetRoot)
    else:
        downloadChunk(chunks[chunkIndex], chunkType, datasetRoot)

chunkIndex = None

if len(sys.argv) > 3:
    print("Wrong parameters")
    exit(1)
elif len(sys.argv) == 3:
    chunkIndex = int(sys.argv[1])
    datasetRoot = sys.argv[2]

Path(datasetRoot).mkdir(parents=True, exist_ok=True)

#downloadChunkType("stat", chunkIndex)
#downloadChunkType("feat", chunkIndex)
#downloadChunkType("obj", chunkIndex)
downloadChunkType("step",chunkIndex, datasetRoot)


