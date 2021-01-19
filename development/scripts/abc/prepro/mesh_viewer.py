import matplotlib.colors as colors
import numpy as np
import meshplot as mp
import webbrowser
import tempfile
import os
import pythreejs as p3s

# Visualize obj meshes (including segmentation colors if available) using meshplot.

V = np.array
surface_color = np.array(colors.to_rgb(colors.CSS4_COLORS['lightgrey']))
edge_colors = ("red","green", "blue","cyan","magenta","yellow","orange","purple","black")
# ['Plane','Revolution', 'Cylinder','Extrusion','Cone','Other','Sphere','Torus','BSpline']

def parse_obje(obj_file, scale_by):
    vs = []
    faces = []
    edges = []

    def add_to_edges():
        if edge_c >= len(edges):
            for _ in range(len(edges), edge_c + 1):
                edges.append([])
        edges[edge_c].append(edge_v)

    def fix_vertices():
        nonlocal vs, scale_by
        vs = V(vs)
        z = vs[:, 2].copy()
        vs[:, 2] = vs[:, 1]
        vs[:, 1] = z
        max_range = 0
        for i in range(3):
            min_value = np.min(vs[:, i])
            max_value = np.max(vs[:, i])
            max_range = max(max_range, max_value - min_value)
            vs[:, i] -= min_value
        if not scale_by:
            scale_by = max_range
        vs /= scale_by

    with open(obj_file) as f:
        for line in f:
            line = line.strip()
            splitted_line = line.split()
            if not splitted_line:
                continue
            elif splitted_line[0] == 'v':
                vs.append([float(v) for v in splitted_line[1:]])
            elif splitted_line[0] == 'f':
                faces.append([int(c.split("//")[0]) - 1 for c in splitted_line[1:]])
            elif splitted_line[0] == 'e':
                if len(splitted_line) >= 4:
                    edge_v = [int(c) - 1 for c in splitted_line[1:-1]]
                    edge_c = int(splitted_line[-1])
                    add_to_edges()

    vs = V(vs)
    fix_vertices()
    faces = V(faces, dtype=int)
    edges = [V(c, dtype=int) for c in edges]
    return (vs, faces, edges), scale_by


def view_meshes(files, outPath):

    tmpDir = None
    autoOpen = False
    if (len(outPath) == 0):
        outPath = tempfile.mkdtemp()
        autoOpen = True

    mp.website()
    shading = {"width": 1024,
               "height": 768,
               "background": "#fffdd1"}


    p = None
    meshCount = len(files)
    meshIndex = 0

    for file in files:
        mesh, scale = parse_obje(file, 0)
        #if (p == None):
            #p = mp.subplot(mesh[0], mesh[1], shading=shading,s=[1,meshCount,meshIndex])
            #mp.plot(mesh[0], mesh[1], shading=shading)
        p = mp.plot(mesh[0], mesh[1],surface_color, shading=shading,return_plot=True)
        colorIndex = 0
        for edgeClass in mesh[2]:
            if (len(edgeClass) > 0):
                p.add_edges(mesh[0],edgeClass,shading={"line_color": edge_colors[colorIndex]})
            colorIndex += 1
        # else:
        #     mp.subplot(mesh[0],mesh[1], shading=shading,data=p,s=[1,meshCount,meshIndex])
        meshIndex += 1
        outFile = os.path.join(outPath,os.path.basename(os.path.splitext(file)[0]) + ".html")
        # max_x_current = mesh[0][:, 0].max()
        # mesh[0][:, 0] += max_x + offset
        # plot = plot_mesh(mesh, surfaces, segments, plot=plot, show=file == files[-1])
        # max_x += max_x_current + offset

        # tt = p3s.TextTexture(string="Plane", color="red", size="60")
        # sm = p3s.SpriteMaterial(map=tt)
        # text = p3s.Sprite(material=sm)
        # text.scale = [0.5,0.5,1]
        # text.center = [1.0,1.0]
        # p._scene.add(text)

        p.save(outFile)
        if (autoOpen):
            webbrowser.open(outFile)

if __name__=='__main__':
    import argparse

    parser = argparse.ArgumentParser("view meshes")
    parser.add_argument('--files', nargs='+', type=str, help="list of 1 or more .obj files")
    parser.add_argument('--outPath', type=str,  default="", help="output path for generated .html file")
    args = parser.parse_args()

    # view meshes
    view_meshes(args.files, args.outPath)

