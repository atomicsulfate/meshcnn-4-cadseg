import os
import argparse
from pathlib import Path
import pygmsh
import gmsh
from math import pi, ceil, inf
import random
import numpy as np

surfaceTypes = ['Plane', 'Revolution', 'Cylinder', 'Extrusion', 'Cone', 'Other', 'Sphere', 'Torus', 'BSpline']


def getSampleName(surfaceType, id):
    return "{}_{}".format(surfaceType, id)


def getMeshEdgeCount(mesh):
    return round(len(mesh.get_cells_type("triangle")) * 1.5)


def createLabelFiles(mesh, surfaceType, filename, segPath, ssegPath):
    edge_count = getMeshEdgeCount(mesh)
    label = surfaceTypes.index(surfaceType)
    hard_labels = int(label) * np.ones(edge_count, dtype='int')
    np.savetxt(os.path.join(segPath, filename + ".eseg"), hard_labels, fmt='%d')

    soft_labels = np.zeros((edge_count, len(surfaceTypes)), dtype='float64')
    soft_labels[:, label] = 1.0
    np.savetxt(os.path.join(ssegPath, filename + ".seseg"), soft_labels, fmt='%f')


def createMeshWithEdgeCount(minEdges, maxEdges, surfaceFunc, *args, **kwargs):
    # Fit the function mesh_size_factor -> mesh edges to a quadratic polynomial.
    mesh_size_factors_inv = []
    num_edges = []
    current_mesh_size_factor = 1

    while (True):
        kwargs['mesh_size_factor'] = current_mesh_size_factor
        mesh = surfaceFunc(*args, **kwargs)
        numEdges = getMeshEdgeCount(mesh)
        if minEdges <= numEdges <= maxEdges:
            print(
                "Target edge count MET with factor {}: {} [{},{}]".format(current_mesh_size_factor, numEdges, minEdges,
                                                                          maxEdges))
            return mesh
        mesh_size_factors_inv.append(1.0 / current_mesh_size_factor)
        num_edges.append(numEdges)
        if (len(mesh_size_factors_inv) < 3):
            current_mesh_size_factor = current_mesh_size_factor / 4
        else:
            if (len(mesh_size_factors_inv) > 3):
                print("Target edge count missed with factor {}: {} [{},{}]".format(current_mesh_size_factor, numEdges,
                                                                                   minEdges,
                                                                                   maxEdges))
            degree = 2  # if len(mesh_size_factors_inv) < 6 else 1
            polynomial = np.polynomial.Polynomial.fit(mesh_size_factors_inv, num_edges, degree)
            target_edges = round(minEdges + (maxEdges-minEdges)* 0.7)
            eq = (polynomial - target_edges)
            for new_mesh_size_factor_inv in eq.roots():
                new_mesh_size_factor = 1.0 / new_mesh_size_factor_inv
                if (new_mesh_size_factor > 0):
                    current_mesh_size_factor = new_mesh_size_factor
                    break


def sample2DOutline():
    min_angle_offset = pi / 10
    point_count = random.uniform(3,10)
    total_angle = random.uniform(pi / 2, 2 * pi)
    points = []
    current_angle = 0

    while (len(points) < point_count):
        r = random.random()
        point = [r * np.cos(current_angle), r * np.sin(current_angle)]
        max_angle = total_angle - min_angle_offset * max(0, (point_count - len(points) - 1));
        current_angle = random.uniform(current_angle+min_angle_offset, max_angle)
        points.append(point)
    return points

def createPolygonMesh(points, mesh_size_factor):
    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        geom.add_polygon(points)
        return geom.generate_mesh(dim=2)

def createPlaneSurfaceFromBSpline(geom, control_points):
    pointInstances = [geom.add_point(x) for x in control_points]
    pointInstances.append(pointInstances[0])  # close the curve.
    bspline = geom.add_bspline(pointInstances)
    curve_loop = geom.add_curve_loop([bspline])
    return geom.add_plane_surface(curve_loop)


def createBSplinePlaneMesh(control_points, mesh_size_factor):
    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        createPlaneSurfaceFromBSpline(geom, control_points)
        return geom.generate_mesh(dim=2)

def createPlaneSurfaces(objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count):
    surfaceType = "Plane"
    surface_generators = [createBSplinePlaneMesh, createPolygonMesh]
    generator_names = ["BSpline", "Polygon"]
    for i in range(count):
        name = getSampleName(surfaceType, i)
        objPath = os.path.join(objDirPath, name + ".obj")

        while(True):
            # meshing fails sometimes (intersections?, too small areas?), retry until it succeeds.
            points = sample2DOutline()
            print("Create {} with {} of {} points".format(name, generator_names[i % 2], len(points)))
            try:
                mesh = createMeshWithEdgeCount(minEdges, maxEdges, surface_generators[i % 2], points)
            except Exception as error:
                print("Meshing error:", error)
            else:
                break

        saveMesh(objPath, mesh)
        createLabelFiles(mesh, surfaceType, name, segDirPath, ssegDirPath)


def createCylinder(radius, length, angle, mesh_size_factor):
    with pygmsh.geo.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        side_idcs_to_remove = []
        num_sections = 3
        if (angle == pi):
            num_sections = 4
            side_idcs_to_remove = list(range(2, num_sections))
        elif (angle < pi):
            num_sections = ceil(2 * pi / angle)
            side_idcs_to_remove = list(range(1, num_sections))
        elif (angle < 2 * pi):
            num_sections = ceil(2 * pi / (2 * pi - angle))
            side_idcs_to_remove = list(range(0, num_sections - 1))

        circle = geom.add_circle(x0=[0, 0, 0], radius=radius, num_sections=num_sections)
        surfaces = geom.extrude(circle.plane_surface, [0.0, 0.0, length])

        geom.remove(surfaces[1], False)  # remove volume (must be done first!)
        geom.remove(circle.plane_surface)  # remove bottom
        geom.remove(surfaces[0])  # remove top
        sides = surfaces[2]
        for side_index in side_idcs_to_remove:
            geom.remove(sides[side_index])

        return geom.generate_mesh(dim=2)


def createCylinders(objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count):
    surfaceType = "Cylinder"
    min_angle = pi / 20
    params = [1, 1, 2 * pi]
    for i in range(count):
        name = getSampleName(surfaceType, i)
        objPath = os.path.join(objDirPath, name + ".obj")
        print("Create {} with radius {}, length {}, angle {}".format(name, params[0], params[1], params[2]))
        mesh = createMeshWithEdgeCount(minEdges, maxEdges, createCylinder, *params)
        saveMesh(objPath, mesh)
        createLabelFiles(mesh, surfaceType, name, segDirPath, ssegDirPath)
        params = np.random.rand(3) * [1, 1, 2 * pi - min_angle] + [0, 0, min_angle]


def createSphere(azimuth, inclination, mesh_size_factor):
    # azimuth [0,pi]
    # inclination [0,2*pi]

    # Can't use OCC's addSphere, plane surfaces cannot be removed later.

    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)

        center = geom.add_point([0, 0])
        a_start = geom.add_point([1, 0])
        a_end = geom.add_point([np.cos(azimuth), np.sin(azimuth)])
        azimuth_arc = geom.add_circle_arc(start=a_start, end=a_end, center=center)
        azimuth_surface = geom.add_plane_surface(
            geom.add_curve_loop([geom.add_line(center, a_start), azimuth_arc, geom.add_line(a_end, center)]))

        surfaces = geom._revolve(azimuth_surface, rotation_axis=[1.0, 0.0, 0.0], point_on_axis=[0.0, 0.0, 0.0],
                                 angle=inclination)
        geom.remove(surfaces[1], False)  # remove volume (must be done first!)
        geom.remove(azimuth_surface)  # remove one end
        if (surfaces[0].dim_tag[1] != azimuth_surface.dim_tag[1]):
            geom.remove(surfaces[0])  # remove the other end
        if (len(surfaces[2]) > 1):
            geom.remove(surfaces[2][1])
        return geom.generate_mesh(dim=2)


def createSpheres(objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count):
    surfaceType = "Sphere"
    min_angle = pi / 20
    angles = [pi, 2 * pi]
    for i in range(count):
        name = getSampleName(surfaceType, i)
        objPath = os.path.join(objDirPath, name + ".obj")
        print("Create {} with angles {}, {}".format(name, angles[0], angles[1]))
        mesh = createMeshWithEdgeCount(minEdges, maxEdges, createSphere, *angles)
        saveMesh(objPath, mesh)
        createLabelFiles(mesh, surfaceType, name, segDirPath, ssegDirPath)
        angles = np.random.rand(2) * [pi - min_angle, 2 * pi - min_angle] + min_angle


def createCone(radius0, height, radius1, angle, mesh_size_factor):
    # radius1 radius at the tip
    # angle revolution angle
    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        cone = geom.add_cone([0, 0, 0], [0, 0, height], radius0, radius1, angle)
        geom.synchronize()
        geom.env.remove(cone.dim_tags)  # remove volume
        surfaces = geom.env.getEntities(2)

        for i in range(1, len(surfaces)):
            geom.env.remove([surfaces[i]])  # remove all plane surfaces.
        return geom.generate_mesh(dim=2)


def createCones(objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count):
    surfaceType = "Cone"
    min_angle = pi / 20
    max_r1_factor = 0.9  # set a maximum radius1 ratio wrt radius0 to keep the cone from looking like a cylinder.
    params = [1, 1, 0, 2 * pi]
    for i in range(count):
        name = getSampleName(surfaceType, i)
        objPath = os.path.join(objDirPath, name + ".obj")
        print("Create {} with r0 {}, r1 {}, height {}, angle {}".format(name, *params))
        mesh = createMeshWithEdgeCount(minEdges, maxEdges, createCone, *params)
        saveMesh(objPath, mesh)
        createLabelFiles(mesh, surfaceType, name, segDirPath, ssegDirPath)
        params = np.random.rand(4)
        params = params * [1, 1, max_r1_factor * params[0], 2 * pi - min_angle] + [0, 0, 0, min_angle]


def createTorus(orad, irad, angle, mesh_size_factor):
    # irad = radius of extruded circle
    # orad = radius from x0 to circle center.
    # angle [0,2*pi], extrusion angle

    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        p = [geom.add_point([orad, 0.0]),
             geom.add_point([orad + irad, 0.0]),
             geom.add_point([orad, irad]),
             geom.add_point([orad - irad, 0.0]),
             geom.add_point([orad, -irad])]
        arcs = []
        for k in range(1, len(p) - 1):
            arcs.append(geom.add_circle_arc(p[k], p[0], p[k + 1]))
        arcs.append(geom.add_circle_arc(p[-1], p[0], p[1]))

        plane_surface = geom.add_plane_surface(geom.add_curve_loop(arcs))
        revolve(geom, plane_surface, rot_axis=[0.0, 1.0, 0.0], rot_point=[0.0, 0.0, 0.0], angle=angle)
        return geom.generate_mesh(dim=2)


def createTori(objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count):
    surfaceType = "Torus"
    min_angle = pi / 20
    max_irad_factor = 0.8
    params = [1, 0.25, 2 * pi]
    for i in range(count):
        name = getSampleName(surfaceType, i)
        objPath = os.path.join(objDirPath, name + ".obj")
        print("Create {} with orad {}, irad {}, angle {}".format(name, params[0], params[1], params[2]))
        mesh = createMeshWithEdgeCount(minEdges, maxEdges, createTorus, *params)
        saveMesh(objPath, mesh)
        createLabelFiles(mesh, surfaceType, name, segDirPath, ssegDirPath)
        params = np.random.rand(3)
        params = params * [1, params[0] * max_irad_factor, 2 * pi - min_angle] + [0, 0, min_angle]


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
        surface = createPlaneSurfaceFromBSpline(geom, points)
        return extrude(geom, surface, length)


def revolve(geom, input_surface, rot_axis=[0.0, 0.0, 1.0], rot_point=[0.0, 0.0, 0.0], angle=2 * pi):
    # num_layers: Optional[Union[int, List[int]]] = None,
    # heights: Optional[List[float]] = None,
    # recombine: bool = False,

    surfaces = geom._revolve(input_surface, rot_axis, rot_point, angle)
    geom.remove(surfaces[1], False)  # remove volume (must be done first!)
    geom.remove(input_surface)  # remove one end
    if (surfaces[0].dim_tag[1] != input_surface.dim_tag[1]):
        geom.remove(surfaces[0])  # remove the other end


def createRevolutionSurface(mesh_size_factor, points, surface_type, angle=2 * pi, rot_axis=[0.0, 0.0, 1.0],
                            rot_point=[0.0, 0.0, 0.0]):
    with pygmsh.occ.Geometry() as geom:
        gmsh.option.setNumber("Mesh.MeshSizeFactor", mesh_size_factor)
        if (surface_type == "polygon"):
            surface = geom.add_polygon(points)
        else:
            surface = createPlaneSurfaceFromBSpline(geom, points)
        revolve(geom, surface, rot_axis, rot_point, angle)
        return geom.generate_mesh(dim=2)


# TODO
def createBSpline():
    return None


def saveMesh(dstPath, mesh):
    mesh.remove_lower_dimensional_cells()
    mesh.write(dstPath)


parser = argparse.ArgumentParser("Create dataset of synthetic mesh samples")
parser.add_argument('--dst', required=True, type=str, help="Path where dataset will be saved")
parser.add_argument('--minEdges', required=True, type=int, help="Min number of edges in a mesh")
parser.add_argument('--maxEdges', required=True, type=int, help="Max number of edges in a mesh")
parser.add_argument('--numSamples', required=True, type=int, help="Num samples per surface type")

args = parser.parse_args()

dstPath = args.dst
minEdges = args.minEdges
maxEdges = args.maxEdges
numSurfaceSamples = args.numSamples


def createPath(path):
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
    except FileExistsError as f_error:
        print(f_error)
        exit(1)


objPath = os.path.join(dstPath, "obj")
segPath = os.path.join(dstPath, "seg")
ssegPath = os.path.join(dstPath, "sseg")
createPath(objPath)
createPath(segPath)
createPath(ssegPath)

createPlaneSurfaces(objPath, segPath, ssegPath, minEdges, maxEdges, numSurfaceSamples)

# createCylinders(objPath,segPath,ssegPath,minEdges,maxEdges,numSurfaceSamples)

# createSpheres(objPath,segPath,ssegPath,minEdges,maxEdges,numSurfaceSamples)

# createCones(objPath,segPath,ssegPath,minEdges,maxEdges,numSurfaceSamples)

# createTori(objPath,segPath,ssegPath,minEdges,maxEdges,numSurfaceSamples)


# saveMesh(os.path.join(dstPath, "torus1.obj"),createTorus(0.2,0.9,variant = "extrude_circle"))


# saveMesh(os.path.join(dstPath, "cyl.obj"), createMeshWithEdgeCount(4500, 5000, createCylinder,
#                                                                    radius=1, length=5, angle = 2*pi/3))
# saveMesh(os.path.join(dstPath, "cylinder2div3pi.obj"),createCylinder( radius=1, length=5, angle=2*pi/3))
# saveMesh(os.path.join(dstPath, "cylinderpi.obj"),createCylinder( radius=1, length=5, angle=pi))
# saveMesh(os.path.join(dstPath, "cylinderpihalf.obj"),createCylinder( radius=1, length=5, angle=pi/2))
# saveMesh(os.path.join(dstPath, "cylinderpififth.obj"),createCylinder( radius=1, length=5, angle=pi/5))

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
