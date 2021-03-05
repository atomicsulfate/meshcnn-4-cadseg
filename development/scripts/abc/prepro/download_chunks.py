
import os, os.path
from pathlib import Path
import argparse

def downloadChunk(chunk, chunkType, datasetRoot):
    typeRoot = os.path.join(datasetRoot, chunkType)
    Path(typeRoot).mkdir(parents=True, exist_ok=True)
    url, filename = chunk.split()
    dstPath = os.path.join(typeRoot,filename)

    if (not os.path.exists(dstPath)):
        print('Downloading chunk from {} to {}'.format(url,dstPath))
        os.system("curl --insecure -o {} {}".format(dstPath, url))

    os.system("7z x {} -aoa -o{}".format(dstPath, typeRoot))
    os.remove(dstPath)

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

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Remesh dataset of target edge count")
    parser.add_argument('--types', nargs='+', required=True, type=str, help="Chunk types: obj, feat, step, stat, meta...")
    parser.add_argument('--indices', nargs='+', type=int, help="Path to source obj files")
    parser.add_argument('--dst', required=True, type=str, help="Path where dataset will be saved")
    args = parser.parse_args()

    datasetRoot = args.dst
    types = args.types
    indices = args.indices

    print("Downloading chunks {} of types {} into {}".format(indices, types, datasetRoot))

    for i in indices:
        for type in types:
            downloadChunkType(type, i, datasetRoot)


