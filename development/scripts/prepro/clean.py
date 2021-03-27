
import os, sys,shutil


basePath = sys.argv[-1]
print('Cleaning ', basePath)

for root, _, fnames in sorted(os.walk(basePath)):
    if os.path.basename(root) == "cache":
        shutil.rmtree(root)
        print(root, "deleted");
    origFile = None
    for fname in fnames:
        if (fname == "mean_std_cache.p"):
            cachePath = os.path.join(root, fname)
            os.remove(cachePath)
            print(cachePath, "deleted")
        elif (fname.endswith(".orig")):
            origFile = fname

    if (origFile != None):
        for fname in fnames:
            if (fname.endswith(".obj")):
                derivedObjPath = os.path.join(root,fname)
                os.remove(derivedObjPath)
                print(derivedObjPath, "deleted")
        fNameNoExt = os.path.splitext(origFile)[0];
        srcPath = os.path.join(root,origFile)
        dstPath = os.path.join(root, fNameNoExt + ".obj")
        os.rename(srcPath, dstPath)
        print(srcPath, "renamed to", dstPath)
