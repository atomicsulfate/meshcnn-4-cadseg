
import os
import argparse
from pathlib import Path
import pygmsh
import gmsh
from math import pi, ceil
import numpy as np

def createMeshWithEdgeCount(minEdges, maxEdges, surfaceFunc, *args, **kwargs):
    # Fit the function mesh_size_factor -> mesh edges to a quadratic polynomial.
    domain = [0.001, 1.0]
    mesh_size_factors_inv = []
    num_edges = []
    current_mesh_size_factor = 1.0

    while(True):
        kwargs['mesh_size_factor'] = current_mesh_size_factor;
        mesh = surfaceFunc(*args, **kwargs);
        numEdges = len(mesh.get_cells_type("triangle")) * 1.5
        if minEdges <= numEdges <= maxEdges:
            print("Target edge count MET with factor {}: {} [{},{}]".format(current_mesh_size_factor, numEdges, minEdges, maxEdges))
            return mesh
        print("Target edge count missed with factor {}: {} [{},{}]".format(current_mesh_size_factor, numEdges, minEdges,
              maxEdges))
        mesh_size_factors_inv.append(1.0/current_mesh_size_factor)
        num_edges.append(numEdges)
        if (len(mesh_size_factors_inv) < 3):
            current_mesh_size_factor = current_mesh_size_factor / 2
        else:
            polynomial = np.polynomial.Polynomial.fit(mesh_size_factors_inv, num_edges, 2,domain)
            eq = (polynomial - maxEdges)
            for new_mesh_size_factor_inv in eq.roots():
                new_mesh_size_factor = 1.0 / new_mesh_size_factor_inv
                if (new_mesh_size_factor > 0):
                    current_mesh_size_factor = new_mesh_size_factor
                    break

def createCylinder(mesh_size_factor, radius, length, angle=2*pi):
    with pygmsh.geo.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        side_idcs_to_remove = []
        num_sections = 3
        if (angle == pi):
            num_sections = 4
            side_idcs_to_remove = list(range(2, num_sections))
        elif (angle < pi):
            num_sections = ceil(2*pi / angle)
            side_idcs_to_remove = list(range(1,num_sections))
        elif (angle < 2*pi):
            num_sections= ceil( 2*pi / (2*pi - angle))
            side_idcs_to_remove = list(range(0, num_sections-1))

        circle = geom.add_circle(x0=[0,0,0],radius=radius,num_sections=num_sections)
        surfaces = geom.extrude(circle.plane_surface, [0.0, 0.0, length])

        geom.remove(surfaces[1], False)  # remove volume (must be done first!)
        geom.remove(circle.plane_surface) # remove bottom
        geom.remove(surfaces[0])  # remove top
        sides = surfaces[2]
        for side_index in side_idcs_to_remove:
            geom.remove(sides[side_index])

        return geom.generate_mesh(dim=2)


# TODO: Circle normals are sometimes wrong
def createSphere(mesh_size_factor, radius):
    with pygmsh.geo.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        geom.add_ball(x0=[0,0,0],radius=radius,with_volume=False)
        return geom.generate_mesh(dim=2)

#def sampleSphereParams(n):


def createCone(mesh_size_factor, axis=[0,0,1], radius0=1, radius1=0, angle=2 * pi):
    # radius1 radius at the tip
    # angle revolution angle
    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        cone = geom.add_cone([0,0,0], axis, radius0, radius1, angle)
        geom.synchronize()
        #gmsh.model.mesh.setSize(gmsh.model.getBoundary(cone.dim_tags, False, False, True), mesh_size)
        geom.env.remove(cone.dim_tags)
        geom.env.remove([geom.env.getEntities(2)[1]]) # remove cone base.
        return geom.generate_mesh(dim=2)

def createTorus(mesh_size_factor, irad, orad, R = np.eye(3), variant = "extrude_lines"):
    #irad = radius of extruded circle
    #orad = radius from x0 to circle center.
    #variant = "extrude_lines" or "extrude_circle"

    with pygmsh.geo.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        geom.add_torus(irad, orad, R=R, x0=[0.0, 0.0, 0.0], variant=variant)
        return geom.generate_mesh(dim=2)

def createPlaneSurfaceFromBSpline(geom, control_points):
    pointInstances = [geom.add_point(x) for x in control_points]
    pointInstances.append(pointInstances[0])  # close the curve.
    bspline = geom.add_bspline(pointInstances)
    curve_loop = geom.add_curve_loop([bspline])
    return geom.add_plane_surface(curve_loop)

def extrude(geom, input_surface, length):
    surfaces = geom.extrude(input_surface, [0.0, 0.0, length * 1.0])
    geom.remove(surfaces[1], False)  # remove volume (must be done first!)
    geom.remove(input_surface)  # remove bottom
    geom.remove(surfaces[0])  # remove top
    return geom.generate_mesh(dim=2)

def createExtrusionSurface(mesh_size_factor, points, length):
    # So far extrude only bspline surfaces.
    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        surface = createPlaneSurfaceFromBSpline(geom,points)
        return extrude(geom, surface, length)

def revolve(geom, input_surface, rot_axis=[0.0, 0.0, 1.0], rot_point=[0.0, 0.0, 0.0], angle=2 * pi):
    # num_layers: Optional[Union[int, List[int]]] = None,
    # heights: Optional[List[float]] = None,
    # recombine: bool = False,

    surfaces = geom._revolve(input_surface, rot_axis, rot_point, angle)
    geom.remove(surfaces[1], False)  # remove volume (must be done first!)
    geom.remove(input_surface)  # remove one end
    #geom.remove(surfaces[0])  # remove the other end
    return geom.generate_mesh(dim=2)

def createRevolutionSurface(mesh_size_factor, points, surface_type, angle=2*pi, rot_axis=[0.0, 0.0, 1.0], rot_point=[0.0, 0.0, 0.0]):
    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        if (surface_type == "polygon"):
            surface = geom.add_polygon(points)
        else:
            surface = createPlaneSurfaceFromBSpline(geom,points)
        return revolve(geom, surface, rot_axis, rot_point, angle)

#TODO
def createBSpline():
    return None

def saveMesh(dstPath, mesh):
    #print(len(mesh.get_cells_type("triangle")))
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


#saveMesh(os.path.join(dstPath, "sphere1.obj"),createSphere(1))

#saveMesh(os.path.join(dstPath, "cone1.obj"),createCone())

#saveMesh(os.path.join(dstPath, "torus1.obj"),createTorus(0.2,0.9,variant = "extrude_circle"))


saveMesh(os.path.join(dstPath, "cyl.obj"), createMeshWithEdgeCount(4500, 5000, createCylinder,
                                                                   radius=1, length=5, angle = 2*pi/3))
#saveMesh(os.path.join(dstPath, "cylinder2div3pi.obj"),createCylinder( radius=1, length=5, angle=2*pi/3))
#saveMesh(os.path.join(dstPath, "cylinderpi.obj"),createCylinder( radius=1, length=5, angle=pi))
#saveMesh(os.path.join(dstPath, "cylinderpihalf.obj"),createCylinder( radius=1, length=5, angle=pi/2))
#saveMesh(os.path.join(dstPath, "cylinderpififth.obj"),createCylinder( radius=1, length=5, angle=pi/5))

# saveMesh(os.path.join(dstPath, "polygon_rev1.obj"), createRevolutionSurface([
#                 [0.0, 0.2, 0.0],
#                 [0.0, 1.2, 0.0],
#                 [0.0, 1.2, 1.0],
#             ], surface_type="polygon"))
# saveMesh(os.path.join(dstPath, "bspline_rev1.obj"), createRevolutionSurface([
#                 [0.0, 0.2, 0.0],
#                 [0.0, 1.2, 0.0],
#                 [0.0, 1.2, 1.0],
#             ], surface_type="bspline"))

# saveMesh(os.path.join(dstPath, "bspline_extr1.obj"), createMeshWithEdgeCount(4500, 5000, createExtrusionSurface, [
#                 [0.0, 0.2, 0.0],
#                 [0.0, 1.2, 0.0],
#                 [1.0, 1.2, 0.0]
#             ], 1.0))











