
import os, sys
import pymesh

basePath = sys.argv[1]

if __name__ == '__main__':

    print('Find open meshes in {}'.format(basePath))

    for root, _, fnames in sorted(os.walk(basePath)):
        for fname in fnames:
            if (fname.endswith(".obj")):
                objPath = os.path.join(root,fname)
                mesh = pymesh.load_mesh(objPath)
                if (not mesh.is_closed()):
                    print(objPath)


