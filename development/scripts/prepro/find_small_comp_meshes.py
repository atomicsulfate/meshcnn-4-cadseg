
import os, sys,shutil
import pymesh

basePath = sys.argv[-2]
minFaceRatio = float(sys.argv[-1])

print('Find small comp meshes in {} (minFaceRatio {})', basePath, minFaceRatio)


for root, _, fnames in sorted(os.walk(basePath)):
    for fname in fnames:
        if (fname.endswith(".obj")):
            objPath = os.path.join(root,fname)
            mesh = pymesh.load_mesh(objPath)
            minFaces = len(mesh.faces) * minFaceRatio
            components = pymesh.separate_mesh(mesh)
            if len(components) == 1:
                continue
            smallComponentSizes = []
            for comp in components:
                if (len(comp.faces) < minFaces):
                    smallComponentSizes.append(len(comp.faces))
            if (len(smallComponentSizes) > 0):
                print("Mesh {}: {} faces, {} cmps, smallComponents({})".format(objPath, len(mesh.faces), len(components), smallComponentSizes))


