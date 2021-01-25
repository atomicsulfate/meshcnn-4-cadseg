
import os
import argparse
from pathlib import Path
import pygmsh
import gmsh
import meshio
from math import pi
import numpy as np

def createSphere(radius):
    with pygmsh.geo.Geometry() as geom:
        geom.add_ball(x0=[0,0,0],radius=radius,with_volume=False, mesh_size=0.03)
        mesh = geom.generate_mesh(dim=2)
        print(len(mesh.get_cells_type("triangle")))
        return mesh

def createCone(center=[0,0,0], axis=[0,0,1], radius0=1, radius1=0, angle=2 * pi):
    with pygmsh.occ.Geometry() as geom:
        # geom.characteristic_length_min = 0.1
        # geom.characteristic_length_max = 0.1
        geom.add_cone(center, axis, radius0, radius1, angle, mesh_size=0.03)
        mesh = geom.generate_mesh(dim=2)
        print(len(mesh.get_cells_type("triangle")))
        return mesh

def createTorus(irad, orad, R = np.eye(3), x0 = np.array([0.0, 0.0, 0.0]), variant = "extrude_lines"):
    #irad = radius of extruded circle
    #orad = radius from x0 to circle center.
    #variant = "extrude_lines" or "extrude_circle"

    with pygmsh.geo.Geometry() as geom:
        geom.add_torus(irad, orad, mesh_size= 0.03, R=R, x0=x0, variant=variant)
        mesh = geom.generate_mesh(dim=2)
        print(len(mesh.get_cells_type("triangle")))
        return mesh


def saveMesh(dstPath, mesh):
    mesh.remove_lower_dimensional_cells()
    print(mesh)
    mesh.write(dstPath)

parser = argparse.ArgumentParser("Create dataset of synthetic mesh samples")
parser.add_argument('--dst', required=True, type=str, help="Path where dataset will be saved")
# parser.add_argument('--minEdges', type=int, default=0, help="Min number of edges to add a mesh to the dataset")
# parser.add_argument('--maxEdges', type=int, default=math.inf, help="Max number of edges to add a mesh to the dataset")
# parser.add_argument('--maxSamples', type=int, default=math.inf, help="Max dataset size")
# parser.add_argument('--testTrainRatio', type=float, default=0.2, help="#test samples/#train samples")
# parser.add_argument('--excludeNonManifolds', action='store_true', help="Exclude meshes that are not manifolds")

args = parser.parse_args()

dstPath = args.dst

try:
    Path(dstPath).mkdir(parents=True, exist_ok=True)
except FileExistsError as f_error:
    print(f_error)
    exit(1)


# TODO: Circle normals are sometimes wrong
#saveMesh(os.path.join(dstPath, "sphere1.obj"),createSphere(1))

# TODO: Remove cone bottom (plane)
#saveMesh(os.path.join(dstPath, "cone1.obj"),createCone())

saveMesh(os.path.join(dstPath, "torus1.obj"),createTorus(0.2,0.9,variant = "extrude_circle"))









