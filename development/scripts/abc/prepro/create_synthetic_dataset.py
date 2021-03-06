import os
import argparse
from pathlib import Path
import pygmsh
import gmsh
from math import pi, ceil, inf
import random
import numpy as np
from numpy.polynomial import Polynomial
import sympy as sp
import sys
import glob
import traceback

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__),"../../..")))

from meshcnn.models.layers.mesh_prepare import build_gemm, remove_non_manifolds

surfaceTypes = ['Plane', 'Revolution', 'Cylinder', 'Extrusion', 'Cone', 'Other', 'Sphere', 'Torus', 'BSpline']


def getFacesSurfaceTypes(meshData, surfsLabels):
    planeLabel = surfaceTypes.index('Plane')
    facesSurfaceTypes = -1 * np.ones(len(meshData.faces), dtype=int)

    faceTags, _ = gmsh.model.mesh.getElementsByType(2)
    allSurfs = gmsh.model.getEntities(2)

    for surf in allSurfs:
        surfTag = surf[1]
        surfFaceTags = gmsh.model.mesh.getElementsByType(2, surfTag)[0]
        label = surfsLabels[surfTag] if surfTag in surfsLabels else planeLabel
        for surfFaceTag in surfFaceTags:
            faceIndex = np.where(faceTags==surfFaceTag) # Slow!
            facesSurfaceTypes[faceIndex] = label

    return facesSurfaceTypes

def getEdgeHardLabels(meshData,surfs, sampleName):
    facesSurfacesTypes = getFacesSurfaceTypes(meshData, surfs)
    edgesSurfaceTypes =-1 * np.ones(meshData.edges_count, dtype=int)

    for faceId, face in enumerate(meshData.faces):
        faceSurfaceType = facesSurfacesTypes[faceId]
        for i in range(3):
            edge = sorted(list((face[i], face[(i + 1) % 3])))
            matchingEdgeIds = list(set(meshData.ve[edge[0]]).intersection(meshData.ve[edge[1]]))
            if (len(matchingEdgeIds) > 1):
                assert False,"more than one edge"
            elif (len(matchingEdgeIds) == 0):
                print(" {}: Edge {} in face {} not found in mesh".format(sampleName, edge, faceId))
                continue
            edgesSurfaceTypes[matchingEdgeIds[0]] = faceSurfaceType
    return edgesSurfaceTypes

def getEdgeSoftLabels(hardLabels, mesh):
    numClasses = len(surfaceTypes)
    gemmEdges = np.array(mesh.gemm_edges)
    softLabels = -1 * np.ones((mesh.edges_count, numClasses), dtype='float64')
    for ei in range(mesh.edges_count):
        prob = np.zeros(numClasses)
        segIds, counts = np.unique(hardLabels[gemmEdges[ei]], return_counts=True)
        prob[segIds] = counts / float(len(gemmEdges[ei]))
        softLabels[ei, :] = prob
    return softLabels

def createSegFiles(mesh, surfs, sampleName, segPath, ssegPath):
    hardLabels = getEdgeHardLabels(mesh, surfs, sampleName)
    softLabels = getEdgeSoftLabels(hardLabels,mesh)

    try:
        file = open(os.path.join(segPath, sampleName + ".eseg"),'w')
    except OSError as e:
        print('open() failed', e)
    else:
        with file:
            for hardLabel in hardLabels:
                file.write(str(hardLabel) + '\n')

    np.savetxt(os.path.join(ssegPath, sampleName + ".seseg"),softLabels, fmt='%f')

def getSampleName(surfaceType, id):
    return "{}_{}".format(surfaceType, id)

def getMeshFaceCount(mesh):
    return round(len(mesh.get_cells_type("triangle")))

def getEstMeshEdgeCount(mesh):
    return getMeshFaceCount(mesh)* 1.525

def computeMeshData(mesh):
    class MeshData:
        def __getitem__(self, item):
            return eval('self.' + item)

    try:
        meshData = MeshData()
        meshData.edge_areas = []
        meshData.vs = mesh.points
        meshData.faces = mesh.get_cells_type('triangle')
        meshData.v_mask = np.ones(len(meshData.vs), dtype=bool)
        faces, face_areas = remove_non_manifolds(meshData, meshData.faces)
        build_gemm(meshData, faces, face_areas)
        mesh.meshData = meshData
    except Exception as error:
        print("mesh error", error)

def estimateFactor(polynomial, target_edges):
    eq = (polynomial - target_edges)
    for factor_inv in eq.roots():
        if (np.isreal(factor_inv) and factor_inv > 0):
            factor = 1.0 / factor_inv
            #print("root {}, new factor {}".format(factor_inv, factor))
            return factor
    return None

def createMeshWithEdgeCount(min_edges, max_edges, objPath, name, segDirPath, ssegDirPath, surface_func,
                            geom, *args, **kwargs):
    # Fit the function 1/mesh_size_factor -> mesh edges to a quadratic polynomial.
    inv_factors = []
    num_edges = []
    current_factor = 1
    target_edges = round(min_edges + (max_edges - min_edges) * 0.7)
    last_num_edges = -inf
    last_factor = current_factor
    num_iters = 0
    max_iters = 20

    while (num_iters < max_iters):
        gmsh.clear()
        gmsh.model.mesh.clear()
        #print("Trying with factor", current_factor)
        gmsh.option.setNumber("Mesh.MeshSizeFactor", current_factor)
        surfaces = surface_func(geom, *args,**kwargs)
        mesh = geom.generate_mesh(dim=2) #, verbose=True)
        face_count = getMeshFaceCount(mesh)
        if (face_count == 0):
            # restart gmsh, sometimes it deadlocks
            #geom.__exit__()
            #geom.__enter__()
            raise(Exception("Internal gmsh error! Restarting gmsh"))
        curr_num_edges = getEstMeshEdgeCount(mesh)
        computeMeshData(mesh)
        if min_edges <= mesh.meshData.edges_count <= max_edges:
            print("Target edges MET with factor {}: {} [{},{}]".format(current_factor, mesh.meshData.edges_count, min_edges, max_edges))
            #geom.save_geometry("test.msh")
            saveMesh(objPath, mesh)
            createSegFiles(mesh.meshData, surfaces, name, segDirPath, ssegDirPath)
            return

        if (len(num_edges) == 0 and curr_num_edges > max_edges):
            raise Exception("Unexpectedly large mesh size for mesh factor 1")
        elif (current_factor < last_factor and curr_num_edges < last_num_edges):
            #raise Exception("Unexpected decrease of mesh size with increase in mesh factor")
            inv_factors = []
            num_edges = []

        # Leave only the 3 closest values to target_edges
        inv_factors.append(1.0/current_factor)
        num_edges.append(curr_num_edges)
        #print("Target edge count missed with factor {}: {} [{},{}]".format(current_factor, curr_num_edges,min_edges,max_edges))
        last_num_edges = curr_num_edges
        last_factor = current_factor
        if (len(inv_factors) == 1):
            current_factor = current_factor / 2
        else:
            current_factor = estimateFactor(Polynomial.fit(inv_factors, num_edges, 2), target_edges)
            if (current_factor == None):
                # Fallback to linear regression
                current_factor = estimateFactor(Polynomial.fit(inv_factors, num_edges, 1), target_edges)
                if (current_factor == None):
                    raise Exception("Factor could not be estimated")
        num_iters += 1
    raise Exception("Didn't converge to target edges in {} iterations".format(max_iters))

def polygonSelfIntersects(vertices):
    lastSide = sp.Segment(vertices[-1],vertices[0])
    for i in range(0,len(vertices)-1):
        side = sp.Segment(vertices[i],vertices[i+1])
        intersections = lastSide.intersection(side)
        if (len(intersections) > 0):
            intersection = intersections[0]
            if (not intersection.equals(lastSide.p1) and not intersection.equals(lastSide.p2)):
                #print("{} intersects with {} at {}".format(lastSide,side, intersection))
                return True
    return False

def sample2DOutline(total_angle_range=(pi / 2, 2 * pi)):
    total_angle = random.uniform(total_angle_range[0],total_angle_range[1])
    assert total_angle < 2 * pi, "Invalid angle"
    max_point_count = total_angle * 8 / pi
    point_count = round(random.uniform(3,max_point_count))
    min_angle_offset = pi / 8
    max_angle_offset = 3*pi/4
    max_radius = random.uniform(0.01, 1)
    min_radius = 0.5 * max_radius
    points = []
    current_angle = 0

    while(True):
        while (len(points) < point_count):
            remain_angles = max(0, (point_count - len(points) - 1))
            if (remain_angles > 0):
                max_angle = total_angle - min_angle_offset * max(0, (point_count - len(points) - 1))
                assert max_angle > 0, "Invalid angle"
                current_angle = random.uniform(current_angle+min_angle_offset, min(current_angle+max_angle_offset,max_angle))
            else:
                current_angle = total_angle
            r = random.uniform(min_radius, max_radius)
            point = [r * np.cos(current_angle), r * np.sin(current_angle)]
            assert current_angle > 0, "Invalid angle"
            assert current_angle <= total_angle, "Invalid angle"
            points.append(point)
        if not polygonSelfIntersects(points):
            return points
        else:
            print("Discarding self-intersection 2D outline")
            points = []
            current_angle = 0

def createPolygonMesh(geom, points, closeMesh):
    surface = geom.add_polygon(points)
    return [surface.dim_tag[1]]

def createPlaneSurfaceFromBSpline(geom, control_points):
    pointInstances = [geom.add_point(x) for x in control_points]
    pointInstances.append(pointInstances[0])  # close the curve.
    bspline = geom.add_bspline(pointInstances)
    curve_loop = geom.add_curve_loop([bspline])
    return geom.add_plane_surface(curve_loop)

def createBSplinePlaneMesh(geom, control_points, closeMesh):
    return [createPlaneSurfaceFromBSpline(geom, control_points).dim_tag[1]]

def createPlaneSurfaces(geom, objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count, closeMeshes):
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
                createMeshWithEdgeCount(minEdges, maxEdges, objPath, name, segDirPath, ssegDirPath,
                                        surface_generators[i % 2], geom, points, closeMeshes)
            except Exception as error:
                print("Meshing error:", error)
            else:
                break

def createCylinder(geom, radius, length, angle, closeMesh):
    center = geom.add_point([0, 0])
    a_start = geom.add_point([radius, 0])

    if (angle < 2*pi):
        a_end = geom.add_point([radius * np.cos(angle), radius * np.sin(angle)])
        base_arc = geom.add_circle_arc(start=a_start, end=a_end, center=center)
        curves = [geom.add_line(center, a_start), base_arc, geom.add_line(a_end, center)]
    else:
        a_middle =  geom.add_point([-radius, 0])
        base_arc_a = geom.add_circle_arc(start=a_start, end=a_middle, center=center)
        base_arc_b = geom.add_circle_arc(start=a_middle, end=a_start, center=center)
        curves = [base_arc_b, base_arc_a]
    base_surface = geom.add_plane_surface(geom.add_curve_loop(curves))
    cyl_surf_type = surfaceTypes.index("Cylinder")
    extr_surfs = extrude(geom, base_surface, length,closeMesh)

    assert len(extr_surfs) == 2 or len(extr_surfs) == 3, "Unexpected number of surfaces"

    if (len(extr_surfs) == 2):
        return {surf.dim_tag[1]: cyl_surf_type for surf in extr_surfs}
    else:
        return {extr_surfs[1].dim_tag[1]: cyl_surf_type}

def createCylinders(geom, objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count, closeMeshes):
    surfaceType = "Cylinder"
    firstNumber = getLastFileNumber(surfaceType, objDirPath) + 1

    min_angle = pi / 6 # Keep cylinder from looking like a plane
    params = [1, 1, 2 * pi]
    for i in range(firstNumber, count):
        name = getSampleName(surfaceType, i)
        objPath = os.path.join(objDirPath, name + ".obj")
        while True:
            print("Create {} with radius {}, length {}, angle {}".format(name, params[0], params[1], params[2]))
            try:
                createMeshWithEdgeCount(minEdges, maxEdges, objPath, name, segDirPath, ssegDirPath,
                                        createCylinder, geom, *params, closeMeshes)
                #geom.save_geometry(name + ".msh")
            except Exception as error:
                print("Meshing error:", error)
                #traceback.print_exc()
                params = np.random.rand(3) * [1, 1, 2 * pi - min_angle] + [0, 0, min_angle]
            else:
                break
        params = np.random.rand(3) * [1, 1, 2 * pi - min_angle] + [0, 0, min_angle]

defMeshSize = 1

def createSphere(geom, azimuth, inclination, closeMesh, r=1.0, meshSizeCenter = 0, meshSizeSphere = 0):
    # azimuth [0,2*pi]
    # inclination [0,pi]

    # Can't use OCC's addSphere, plane surfaces cannot be removed later.
    #geom.env.addSphere(0,0,0,radius=r, angle1=-pi/2, angle2=inclination-pi/2, angle3=azimuth)


    center = geom.add_point([0, 0], mesh_size=meshSizeCenter)
    a_start = geom.add_point([0, 0, r], mesh_size=meshSizeSphere)
    a_end = geom.add_point([r*np.sin(inclination), 0, r*np.cos(inclination)],mesh_size=meshSizeSphere)
    inclination_arc = geom.add_circle_arc(start=a_start, end=a_end, center=center)
    curves = [geom.add_line(center, a_start), inclination_arc, geom.add_line(a_end, center)]
    inclination_surface = geom.add_plane_surface(geom.add_curve_loop(curves))

    surfaces = geom._revolve(inclination_surface, rotation_axis=[0.0, 0.0, 1.0], point_on_axis=[0.0, 0.0, 0.0],
                             angle=azimuth)

    geom.remove(surfaces[1], False)  # remove volume (must be done first!)
    nonPlanarSurfs = {surfaces[2][0].dim_tag[1]: surfaceTypes.index("Sphere")}

    if (not closeMesh or azimuth >= 2*pi):
        geom.remove(inclination_surface)  # remove one end
        if (surfaces[0].dim_tag[1] != inclination_surface.dim_tag[1]):
            geom.remove(surfaces[0])  # remove the other end
    if (len(surfaces[2]) > 1):
        coneSurf = surfaces[2][1]
        if (closeMesh):
            nonPlanarSurfs[coneSurf.dim_tag[1]] = surfaceTypes.index("Plane") if (abs(azimuth - pi/2) < pi/90) else surfaceTypes.index("Cone")
        else:
            geom.remove(coneSurf)
    return nonPlanarSurfs

def createSpheres(geom, objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count, closeMeshes):
    surfaceType = "Sphere"
    min_angle = pi / 20
    angles = [2*pi, pi]
    for i in range(count):
        name = getSampleName(surfaceType, i)
        objPath = os.path.join(objDirPath, name + ".obj")

        while True:
            print("Create {} with angles {}, {}".format(name, angles[0], angles[1]))
            try:
                createMeshWithEdgeCount(minEdges, maxEdges, objPath, name, segDirPath, ssegDirPath,
                                        createSphere, geom, *angles, closeMeshes)
                #geom.save_geometry(name + ".msh")
            except Exception as error:
                print("Meshing error:", error)
                angles = np.random.rand(2) * [2 *pi - min_angle,  pi - min_angle] + min_angle
            else:
                break
        angles = np.random.rand(2) * [2 *pi - min_angle,  pi - min_angle] + min_angle


def createCone(geom, radius0, height, radius1, angle, closeMesh):
    # radius1 radius at the tip
    # angle revolution angle
    cone = geom.add_cone([0, 0, 0], [0, 0, height], radius0, radius1, angle)
    geom.synchronize()
    geom.env.remove(cone.dim_tags)  # remove volume
    surfaces = geom.env.getEntities(2)

    if (not closeMesh):
        for i in range(1, len(surfaces)):
            geom.env.remove([surfaces[i]])  # remove all plane surfaces.

    return {surfaces[0][1]: surfaceTypes.index("Cone")}

def createCones(geom, objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count, closeMeshes):
    surfaceType = "Cone"
    min_angle = pi / 6 # Keep cone from looking like a plane
    max_r1_factor = 0.7  # set a maximum radius1 ratio wrt radius0 to keep the cone from looking like a cylinder.
    params = [1, 1, 0, 2 * pi]
    min_height_factor = 0.2 # set min height factor wrt radius0 to keep cone from looking like a plane
    max_height_factor = 5 # set max height factor wrt radius0 to keep cone from looking like a cylinder

    for i in range(count):
        name = getSampleName(surfaceType, i)
        objPath = os.path.join(objDirPath, name + ".obj")

        while True:
            print("Create {} with r0 {}, r1 {}, height {}, angle {}".format(name, *params))
            try:
                createMeshWithEdgeCount(minEdges, maxEdges, objPath, name, segDirPath, ssegDirPath,
                                        createCone, geom, *params, closeMeshes)
            except Exception as error:
                print("Meshing error:", error)
                params = np.random.rand(4)
                params = params * [1, 1, max_r1_factor * params[0], 2 * pi - min_angle] + [0, 0, 0, min_angle]
            else:
                break

        params = np.random.rand(4)
        params = params * [1, (max_height_factor-min_height_factor)*params[0], max_r1_factor * params[0], 2 * pi - min_angle] \
                 + [0, min_height_factor*params[0], 0, min_angle]


def createTorus(geom, orad, irad, angle, circleAngle, closeMesh):
    # irad = radius of extruded circle
    # orad = radius from x0 to circle center.
    # angle [0,2*pi], extrusion angle
    # circleAngle, circle arc angle
    if (circleAngle < 2*pi):
        p = [geom.add_point([orad, 0.0]),
             geom.add_point([orad + irad, 0.0]),
             geom.add_point([orad + irad*np.cos(circleAngle), irad*np.sin(circleAngle)])]
    else:
        p = [geom.add_point([orad, 0.0]),
             geom.add_point([orad + irad, 0.0]),
             geom.add_point([orad, irad]),
             geom.add_point([orad - irad, 0.0]),
             geom.add_point([orad, -irad])]
    arcs = []
    for k in range(1, len(p) - 1):
        arcs.append(geom.add_circle_arc(p[k], p[0], p[k + 1]))

    if (circleAngle < 2*pi):
        arcs.append(geom.add_line(p[-1], p[0]))
        arcs.append(geom.add_line(p[0], p[1]))
    else:
        arcs.append(geom.add_circle_arc(p[-1], p[0], p[1]))

    plane_surface = geom.add_plane_surface(geom.add_curve_loop(arcs))

    revolve(geom, plane_surface, rot_axis=[0.0, 1.0, 0.0], rot_point=[0.0, 0.0, 0.0], angle=angle, closeMesh=closeMesh)

    torusLabel = surfaceTypes.index("Torus")
    planeLabel = surfaceTypes.index("Plane")
    coneLabel = surfaceTypes.index("Cone")
    surfsTags = [surf[1] for surf in geom.env.getEntities(2)]
    if (angle < 2 * pi):
        if (not closeMesh):
            geom.env.remove([(2, surfsTags[0])])
            geom.env.remove([(2, surfsTags[-1])])
        torusTags = surfsTags[2:-2]
        surfDict = {surf: torusLabel for surf in torusTags}
        surfDict[surfsTags[1]] = torusLabel
        surfDict[surfsTags[2]] = coneLabel
        surfDict[surfsTags[-2]] = planeLabel
        return surfDict
    else:
        geom.env.remove([(2, surfsTags[0])])
        return {surf: torusLabel for surf in surfsTags}

def createTori(geom, objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count, closeMeshes):
    surfaceType = "Torus"
    min_angle = pi / 20
    min_circle_angle = pi / 4
    max_irad_factor = 0.8
    params = [1, 0.25, 2 * pi, 2*pi]
    for i in range(count):
        name = getSampleName(surfaceType, i)
        objPath = os.path.join(objDirPath, name + ".obj")

        while True:
            print("Create {} with orad {}, irad {}, angle {}, circle angle {}".format(name, params[0], params[1], params[2], params[3]))
            try:
                createMeshWithEdgeCount(minEdges, maxEdges, objPath, name, segDirPath, ssegDirPath,
                                        createTorus, geom, *params, closeMeshes)
            except Exception as error:
                print("Meshing error:", error)
                params = np.random.rand(4)
                params = params * [1, params[0] * max_irad_factor, 2 * pi - min_angle,2 * pi - min_circle_angle] + [0, 0, min_angle, min_circle_angle]
            else:
                break
        params = np.random.rand(4)
        params = params * [1, params[0] * max_irad_factor, 2 * pi - min_angle,2 * pi - min_circle_angle] + [0, 0, min_angle, min_circle_angle]


def extrude(geom, input_surface, length, closeMesh):
    surfaces = geom.extrude(input_surface, [0.0, 0.0, length])
    if (not closeMesh):
        geom.remove(surfaces[1], False)  # remove volume (must be done first!)
        geom.remove(input_surface)  # remove bottom
        geom.remove(surfaces[0])  # remove top
    return surfaces[2]

def createExtrusionSurface(geom, points, length, closeMesh):
    # So far extrude only bspline surfaces.
    surface = createPlaneSurfaceFromBSpline(geom, points)
    extrSurfs = extrude(geom, surface, length, closeMesh, )
    extrType = surfaceTypes.index("Extrusion")
    return {surf.dim_tag[1]: extrType  for surf in extrSurfs}

def getLastFileNumber(surfaceName, path):
    pattern = os.path.join(path, surfaceName + "_*.obj")
    fileList = glob.glob(pattern)
    maxNumber = -1
    for file in fileList:
        fileName = os.path.basename(os.path.splitext(file)[0])
        fileNumber = int(fileName[len(surfaceName)+1:])
        maxNumber = max(maxNumber,fileNumber)

    if (maxNumber > -1):
        print("{} files exists up to number {}".format(surfaceName, maxNumber))
    return maxNumber

def createExtrusionSurfaces(geom, objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count, closeMeshes):
    surfaceType = "Extrusion"
    firstNumber = getLastFileNumber(surfaceType, objDirPath)+1
    for i in range(firstNumber, count):
        name = getSampleName(surfaceType, i)
        objPath = os.path.join(objDirPath, name + ".obj")

        while(True):
            # meshing fails sometimes (intersections?, too small areas?), retry until it succeeds.
            points = sample2DOutline()
            length = random.random()
            print("Create {} with BSpline of {} points, length {}".format(name, len(points), length))
            try:
                createMeshWithEdgeCount(minEdges, maxEdges, objPath, name, segDirPath, ssegDirPath,
                                        createExtrusionSurface, geom, points, length, closeMeshes)
            except Exception as error:
                print("Meshing error:", error)
            else:
                break

def revolve(geom, input_surface, rot_axis=[1.0, 0.0, 0.0], rot_point=[0.0, 0.0, 0.0], angle=2 * pi, closeMesh=True):
    geom._revolve(input_surface, rot_axis, rot_point, angle)
    geom.env.remove(geom.env.getEntities(3))

def createRevolutionSurface(geom, points, surface_type, angle=2 * pi, rot_axis=[1.0, 0.0, 0.0], rot_point=[0.0, 0.0, 0.0], closeMesh=True):
    if (surface_type == "polygon"):
        surface = geom.add_polygon(points)
    else:
        surface = createPlaneSurfaceFromBSpline(geom, points)
    revolve(geom, surface, rot_axis, rot_point, angle, closeMesh)

    revLabel = surfaceTypes.index("Revolution")

    revSurfs = [surf[1] for surf in geom.env.getEntities(2)]

    if (angle < 2*pi):
        if (not closeMesh):
            geom.env.remove([(2, revSurfs[0])])
            geom.env.remove([(2, revSurfs[-1])])
        revSurfs = revSurfs[1:-1]
    else:
        geom.env.remove([(2, revSurfs[0])])

    return { revSurf : revLabel for revSurf in revSurfs}

def createRevolutionSurfaces(geom, objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count, closeMeshes):
    surfaceType = "Revolution"
    generator_types = ["polygon", "bspline"]
    angle = 2*pi
    firstNumber = getLastFileNumber(surfaceType, objDirPath)+1
    for i in range(firstNumber,count):
        name = getSampleName(surfaceType, i)
        objPath = os.path.join(objDirPath, name + ".obj")

        while(True):
            # meshing fails sometimes (intersections?, too small areas?), retry until it succeeds.
            points = sample2DOutline((pi/2,pi))
            print("Create {} with {} of {} points, angle {}".format(name, generator_types[i%2],len(points), angle))
            try:
                createMeshWithEdgeCount(minEdges, maxEdges, objPath, name, segDirPath, ssegDirPath,
                                        createRevolutionSurface, geom, points, generator_types[i%2], angle, closeMesh=closeMeshes)

            except Exception as error:
                print("Meshing error:", error)
            else:
                break
        angle = random.uniform(pi / 8, 2 * pi)

def createBSplineSurface(geom, r, degree_inc, degree_az, num_divs_az, num_divs_inc, ctrl_point_heights, closeMesh):
    azimuth = 2*pi
    inclination = pi

    tmpSurfaces = []
    # if (closeMesh):
    #     # Add a sphere wedge with same azimuth and inclination, then replace spherical side by bspline.
    #     tmpSurfDict = createSphere(geom,azimuth,inclination, closeMesh, r) #, defMeshSize, defMeshSize/1)
    #     surfaces = geom.env.getEntities(2)
    #     sphereType = surfaceTypes.index("Sphere")
    #     for surf in surfaces:
    #         if (surf[1] in tmpSurfDict and tmpSurfDict[surf[1]] == sphereType):
    #             geom.env.remove([surf])
    #         else:
    #             tmpSurfaces.append(surf)

    inc = 0
    inc_step = inclination / num_divs_inc
    az_step = azimuth / num_divs_az
    point_tags = []
    for i in range(num_divs_inc+1):
        az = 0
        for j in range(num_divs_az+1):
            radius = r
            # Do not displace control points at the boundary so that later bspline surface can be sewed to the whole volume.
            if (i != 0 and i!= num_divs_inc and j != 0 and j != num_divs_az):# and (i % 3) == 0 and (j % 3) == 0):
                radius += ctrl_point_heights[i*(num_divs_az+1)+j]
            p =geom.add_point([radius*np.sin(inc)*np.cos(az), radius*np.sin(inc)*np.sin(az), radius*np.cos(inc)]) #, mesh_size=defMeshSize/1)
            point_tags.append(p._id)
            az += az_step
        inc += inc_step

    bsplineType = surfaceTypes.index("BSpline")
    bsplineTag = geom.env.addBSplineSurface(point_tags, num_divs_az+1, degreeU=degree_az, degreeV=degree_inc)
    return {bsplineTag: bsplineType}

    # if (not closeMesh):
    #     return {bsplineTag: bsplineType}

    # geom.env.synchronize()
    # xmin, ymin, zmin, xmax, ymax, zmax = gmsh.model.getBoundingBox(-1, -1)
    # bboxDiag = np.sqrt(((np.array([xmin,ymin,zmin])-np.array([xmax,ymax,zmax]))**2).sum())
    # #print("bbox diag:", bboxDiag)
    # tmpSurfaces += [(2,bsplineTag)]
    # finalEntities = geom.env.healShapes(tmpSurfaces, tolerance=bboxDiag/1e3) #, fixDegenerated=True, fixSmallEdges=True, fixSmallFaces=True, sewFaces=True, makeSolids=True)
    # volumeCreated = False
    # finalSurfaces = {}
    # for ent in finalEntities:
    #     if (ent[0] == 3):
    #         volumeCreated = True
    #     elif (ent[0] == 2):
    #         finalSurfaces[ent[1]] = bsplineType
    #     else:
    #         break
    #
    # if (not volumeCreated):
    #     raise Exception("BSpline surface could not be closed")
    #
    # surfDict = {bsplineTag: bsplineType}
    # for i in range(len(finalSurfaces)):
    #     tmpSurfTag = tmpSurfaces[i][1]
    #     if tmpSurfTag in tmpSurfDict:
    #         surfDict[finalSurfaces[i]] = tmpSurfDict[tmpSurfTag]
    # geom.env.remove(tmpSurfaces)
    # return surfDict
    #return finalSurfaces

def createBSplineSurfaces(geom, objDirPath, segDirPath, ssegDirPath, minEdges, maxEdges, count, closeMeshes):
    surfaceType = "BSpline"

    firstNumber = getLastFileNumber(surfaceType, objDirPath) + 1

    for i in range(firstNumber, count):
        name = getSampleName(surfaceType, i)
        objPath = os.path.join(objDirPath, name + ".obj")

        while True:
            # inclination = np.random.uniform(min_angle, pi)
            # azimuth = np.random.uniform(min_angle, 2*pi)
            radius = 1
            degree_inc = round(np.random.uniform(3, 6))
            degree_az = round(np.random.uniform(3, 6))
            num_divs_inc = round(np.random.uniform(2*degree_inc, 4*degree_inc))
            num_divs_az = round(np.random.uniform(2*degree_az, 4*degree_az))
            num_ctrl_points = (num_divs_inc+1)*(num_divs_az+1)
            ctrl_point_heights = np.random.default_rng().random(num_ctrl_points) * radius/2 - radius/4

            print("Create {} of degree {}x{} with {}x{} ctrl points around sphere"
                  .format(name, degree_az, degree_inc, num_divs_az+1, num_divs_inc+1))
            try:
                createMeshWithEdgeCount(minEdges, maxEdges, objPath, name, segDirPath, ssegDirPath,
                                        createBSplineSurface, geom, radius, degree_inc, degree_az,
                                        num_divs_az, num_divs_inc, ctrl_point_heights, closeMeshes)
               #geom.save_geometry(name + ".msh")
            except Exception as error:
                print("Meshing error:", error)
            else:
                break

def saveMesh(dstPath, mesh):
    mesh.remove_lower_dimensional_cells()
    mesh.write(dstPath)


parser = argparse.ArgumentParser("Create dataset of synthetic mesh samples")
parser.add_argument('--dst', required=True, type=str, help="Path where dataset will be saved")
parser.add_argument('--minEdges', required=True, type=int, help="Min number of edges in a mesh")
parser.add_argument('--maxEdges', required=True, type=int, help="Max number of edges in a mesh")
parser.add_argument('--numSamples', required=True, type=int, help="Num samples per surface type")
parser.add_argument('--closeMeshes', action='store_true', help="Whether to close mesh using plane surfaces")

args = parser.parse_args()

dstPath = args.dst
minEdges = args.minEdges
maxEdges = args.maxEdges
numSurfaceSamples = args.numSamples
closeMeshes = args.closeMeshes


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

with pygmsh.occ.Geometry() as geom:
    #gmsh.option.setNumber("Mesh.RandomFactor", 1e-4)
    #gmsh.option.setNumber("Mesh.ScalingFactor", 10000)
    #gmsh.option.setNumber("Mesh.MeshSizeMin", 0.001)
    #gmsh.option.setNumber("Mesh.ReparamMaxTriangles", 10)
    #gmsh.option.setNumber("General.NumThreads", 6)
    #gmsh.option.setNumber("Mesh.RefineSteps", 2)
    # gmsh.option.setNumber("General.Terminal", 1)
    gmsh.option.setNumber("Mesh.AlgorithmSwitchOnFailure", 0) # Fallback to Mesh-Adapt ends hanging up sometimes.
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi", 8)
    createCylinders(geom, objPath, segPath, ssegPath, minEdges, maxEdges, numSurfaceSamples, closeMeshes)
    #createPlaneSurfaces(geom, objPath, segPath, ssegPath, minEdges, maxEdges, numSurfaceSamples, closeMeshes)
    #createSpheres(geom, objPath,segPath,ssegPath,minEdges,maxEdges,numSurfaceSamples, closeMeshes)
    # createCones(geom, objPath,segPath,ssegPath,minEdges,maxEdges,numSurfaceSamples, closeMeshes)
    #createTori(geom, objPath,segPath,ssegPath,minEdges,maxEdges,numSurfaceSamples, closeMeshes)
    #createRevolutionSurfaces(geom, objPath, segPath, ssegPath, minEdges, maxEdges, numSurfaceSamples, closeMeshes)
    #createExtrusionSurfaces(geom, objPath, segPath, ssegPath, minEdges, maxEdges, numSurfaceSamples, closeMeshes)
    createBSplineSurfaces(geom, objPath, segPath, ssegPath, minEdges, maxEdges, numSurfaceSamples, closeMeshes)





