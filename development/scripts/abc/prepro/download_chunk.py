
import os, sys

datasetRoot = "datasets/abc";

def downloadChunk(chunk):
    typeRoot = os.path.join(datasetRoot, chunkType)
    url, filename = chunk.split()
    dstPath = os.path.join(typeRoot,filename)

    if (not os.path.exists(dstPath)):
        print('Downloading chunk from {} to {}'.format(url,dstPath))
        os.system("curl --insecure -o {} {}".format(dstPath, url))

    os.system("7z x {} -o{}".format(dstPath, typeRoot))
    os.system("rm {}".format(dstPath))


chunkIndex = None

if len(sys.argv) < 2 or len(sys.argv) > 3:
    print("Wrong parameters")
    exit(1)
elif len(sys.argv) == 3:
    chunkIndex = int(sys.argv[-1])

chunkType = sys.argv[-len(sys.argv)+1]

indexFileName = chunkType + "_v00.txt"
path = os.path.join(datasetRoot,indexFileName)
file = open(path, "r")
chunks = file.readlines()

if( chunkIndex == None):
    for chunk in chunks:
        downloadChunk(chunk)
else:
    downloadChunk(chunks[chunkIndex])
