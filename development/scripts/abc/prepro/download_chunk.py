
import os, sys

datasetRoot = "datasets/abc";

def downloadChunk(chunk, chunkType):
    typeRoot = os.path.join(datasetRoot, chunkType)
    url, filename = chunk.split()
    dstPath = os.path.join(typeRoot,filename)

    if (not os.path.exists(dstPath)):
        print('Downloading chunk from {} to {}'.format(url,dstPath))
        os.system("curl --insecure -o {} {}".format(dstPath, url))

    os.system("7z x {} -o{}".format(dstPath, typeRoot))
    os.system("rm {}".format(dstPath))

def downloadChunkType(chunkType, chunkIndex):
    indexFileName = chunkType + "_v00.txt"
    path = os.path.join(datasetRoot,indexFileName)
    file = open(path, "r")
    chunks = file.readlines()

    if( chunkIndex == None):
        for chunk in chunks:
            downloadChunk(chunk, chunkType)
    else:
        downloadChunk(chunks[chunkIndex], chunkType)

chunkIndex = None

if len(sys.argv) > 2:
    print("Wrong parameters")
    exit(1)
elif len(sys.argv) == 2:
    chunkIndex = int(sys.argv[-1])

#downloadChunkType("stat", chunkIndex)
downloadChunkType("feat", chunkIndex)
downloadChunkType("obj", chunkIndex)


