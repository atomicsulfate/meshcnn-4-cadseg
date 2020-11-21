
import os, sys

'''
Simplifies mesh to target number of faces
Requires Blender 2.8
Author: Rana Hanocka

@input: 
    <obj_file>
    <target_faces> number of target faces
    <outfile> name of simplified .obj file

@output:
    simplified mesh .obj
    to run it from cmd line:
    /opt/blender/blender --background --python blender_process.py /home/rana/koala.obj 1000 /home/rana/koala_1000.obj
'''

basePath = sys.argv[-2]
targetFaces = int(sys.argv[-1])

print('Simplifying meshes in {} to {} faces (~ {} edges)'.format(basePath,targetFaces,targetFaces*1.5))

for root, _, fnames in sorted(os.walk(basePath)):
    for fname in fnames:
        if fname.startswith(os.path.basename(root)):
            origPath = os.path.join(root,fname)
            if fname.endswith(".obj"):
                renamedPath = os.path.splitext(origPath)[0]+".orig"
                print("Renaming {} to {}".format(origPath, renamedPath))
                os.rename(origPath,renamedPath)
                origPath= renamedPath
            dstPath = os.path.join(root,str(targetFaces) + ".obj")
            if (not os.path.exists(dstPath)):
                os.system("/usr/bin/blender --background --python blender_process.py {} {} {}".format(origPath,str(targetFaces),dstPath))
                print("Mesh {} created".format(dstPath))

