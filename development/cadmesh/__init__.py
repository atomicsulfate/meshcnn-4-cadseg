from OCC.Extend.TopologyUtils import TopologyExplorer
from OCC.Core.GProp import GProp_GProps
from OCC.Core.BRepAdaptor import BRepAdaptor_Curve, BRepAdaptor_Surface, BRepAdaptor_CompCurve, BRepAdaptor_Curve2d
from OCC.Core.gp import *
from OCC.Core.BRepTools import *
from OCC.Core.BRep import *
from OCC.Core.TopoDS import *
import gmsh
import sys
import os
import numpy as np
import json
import glob
import fileinput
import random
import yaml
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.BRepBndLib import brepbndlib_Add
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.BRepMesh import BRepMesh_IncrementalMesh
from OCC.Core.STEPControl import STEPControl_Reader, STEPControl_Writer, STEPControl_AsIs
from OCC.Core.IFSelect import IFSelect_RetDone, IFSelect_ItemsByEntity
from OCC.Core.TColStd import TColStd_Array1OfReal, TColStd_Array2OfReal
from OCC.Core.TColgp import TColgp_Array1OfPnt, TColgp_Array2OfPnt, TColgp_Array1OfPnt2d
from OCC.Core.BRepGProp import (brepgprop_SurfaceProperties,
                                brepgprop_VolumeProperties)
from OCC.Core.Geom2dAdaptor import Geom2dAdaptor_Curve
from OCCUtils.edge import Edge
from OCCUtils.Topology import Topo

np.set_printoptions(precision=17)

def read_step_file(filename, return_as_shapes=False, verbosity=False):
    assert os.path.isfile(filename)
    step_reader = STEPControl_Reader()
    status = step_reader.ReadFile(filename)
    if status == IFSelect_RetDone:  # check status
        if verbosity:
            failsonly = False
            step_reader.PrintCheckLoad(failsonly, IFSelect_ItemsByEntity)
            step_reader.PrintCheckTransfer(failsonly, IFSelect_ItemsByEntity)
        shapes = []
        try:
            total_roots = step_reader.NbRootsForTransfer();
            for nr in range(1,total_roots+1):
                ok = step_reader.TransferRoot(nr)
                if not ok:
                    break
                _nbs = step_reader.NbShapes()
                shapes.append(step_reader.Shape(nr))  # a compound
                #assert not shape_to_return.IsNull()
        except:
            print("No Shape", nr)
    else:
        raise AssertionError("Error: can't read file.")
    #if return_as_shapes:
    #    shape_to_return = TopologyExplorer(shape_to_return).solids()

    return shapes


def get_boundingbox(shape, tol=1e-6, use_mesh=True):
    """ return the bounding box of the TopoDS_Shape `shape`
    Parameters
    ----------
    shape : TopoDS_Shape or a subclass such as TopoDS_Face
        the shape to compute the bounding box from
    tol: float
        tolerance of the computed boundingbox
    use_mesh : bool
        a flag that tells whether or not the shape has first to be meshed before the bbox
        computation. This produces more accurate results
    """
    bbox = Bnd_Box()
    bbox.SetGap(tol)
    if use_mesh:
        mesh = BRepMesh_IncrementalMesh()
        mesh.SetParallelDefault(True)
        mesh.SetShape(shape)
        mesh.Perform()
        assert mesh.IsDone()
    brepbndlib_Add(shape, bbox, use_mesh)

    xmin, ymin, zmin, xmax, ymax, zmax = bbox.Get()
    return [xmin, ymin, zmin, xmax, ymax, zmax, xmax-xmin, ymax-ymin, zmax-zmin]

edge_map = {0: "Line", 1: "Circle", 2: "Ellipse", 3: "Hyperbola", 4: "Parabola", 5: "Bezier", 6: "BSpline", 7: "Other"}
surf_map = {0: "Plane", 1: "Cylinder", 2: "Cone", 3: "Sphere", 4: "Torus", 5: "Bezier", 6: "BSpline", 7: "Revolution", 8: "Extrusion", 9: "Other"}
gmsh_map = {"Surface of Revolution": "Revolution", "Surface of Extrusion": "Extrusion", "Plane": "Plane", "Cylinder": "Cylinder",\
           "Cone": "Cone", "Torus": "Torus", "Sphere": "Sphere", "Bezier surface": "Bezier", "BSpline surface": "BSpline", "Unknown": "Other"}


# def convert_curve(curve):
#     d1_feat = {"type": edge_map[curve.GetType()]}
#     c_type = d1_feat["type"]
#     if c_type == "Line":
#         c = curve.Line()
#         d1_feat["location"] = list(c.Location().Coord())
#         d1_feat["direction"] = list(c.Direction().Coord())
#         scale_factor = 1000.0
#         #occ_node_s = occ_brt.Pnt(topods_Vertex(list(occ_topo.vertices())[elemNodeTags[0][0]-1 - occ_offset]))
#         #occ_node_e = occ_brt.Pnt(topods_Vertex(list(occ_topo.vertices())[elemNodeTags[0][-1]-1 - occ_offset]))
#         #print(occ_node_s.Coord(), curve.Value(curve.FirstParameter()).Coord(), v_nodes[elemNodeTags[0][0]-1], occ_node_e.Coord(), curve.Value(curve.LastParameter()).Coord(), v_nodes[elemNodeTags[0][-1]-1])
#         #print(c.Location().Coord(), c.Direction().Coord())
#         #print("E", np.allclose(np.array(curve.Value(curve.LastParameter()).Coord()), np.array(c.Location().Coord())+curve.LastParameter()*np.array(c.Direction().Coord())))
#         #print("S", np.allclose(np.array(curve.Value(curve.FirstParameter()).Coord()), np.array(c.Location().Coord())+curve.FirstParameter()*np.array(c.Direction().Coord())))
#         #print(e, nodeTags, nodeCoords, nodeParams, gmsh.model.getType(e[0], e[1]), elemTypes, elemTags, elemNodeTags)
#     elif c_type == "Circle":
#         c = curve.Circle()
#         d1_feat["location"] = list(c.Location().Coord())
#         d1_feat["z_axis"] = list(c.Axis().Direction().Coord())
#         d1_feat["radius"] = c.Radius()
#         d1_feat["x_axis"] = list(c.XAxis().Direction().Coord())
#         d1_feat["y_axis"] = list(c.YAxis().Direction().Coord())
#         scale_factor = 1.0
#         #print(c.Location().Coord(), c.Axis().Direction().Coord(), c.Radius())
#     elif c_type == "Ellipse":
#         c = curve.Ellipse()
#         d1_feat["focus1"] = list(c.Focus1().Coord())
#         d1_feat["focus2"] = list(c.Focus2().Coord())
#         d1_feat["x_axis"] = list(c.XAxis().Direction().Coord())
#         d1_feat["y_axis"] = list(c.YAxis().Direction().Coord())
#         d1_feat["z_axis"] = list(c.Axis().Direction().Coord())
#         d1_feat["maj_radius"] = c.MajorRadius()
#         d1_feat["min_radius"] = c.MinorRadius()
#         scale_factor = 1.0
#         #print(c.Focus1().Coord(), c.Focus2().Coord(), c.XAxis().Direction().Coord(), c.YAxis().Direction().Coord(), c.Axis().Direction().Coord(), c.MajorRadius(), c.MinorRadius())
#     elif c_type == "BSpline":
#         c = curve.BSpline()
#         #print(dir(c))
#         c.SetNotPeriodic()
#         d1_feat["rational"] = c.IsRational()
#         d1_feat["closed"] = c.IsClosed()
#         #d1_feat["periodic"] = c.IsPeriodic()
#         d1_feat["continuity"] = c.Continuity()
#         d1_feat["degree"] = c.Degree()
#         p = TColgp_Array1OfPnt(1, c.NbPoles())
#         c.Poles(p)
#         points = []
#         for pi in range(p.Length()):
#             points.append(list(p.Value(pi+1).Coord()))
#         d1_feat["poles"] = points
#
#         k = TColStd_Array1OfReal(1, c.NbPoles() + c.Degree() + 1)
#         c.KnotSequence(k)
#         knots = []
#         for ki in range(k.Length()):
#             knots.append(k.Value(ki+1))
#         d1_feat["knots"] = knots
#
#         w = TColStd_Array1OfReal(1, c.NbPoles())
#         c.Weights(w)
#         weights = []
#         for wi in range(w.Length()):
#             weights.append(w.Value(wi+1))
#         d1_feat["weights"] = weights
#
#         scale_factor = 1.0
#         #print(c.Knots())
#         #d1_feat[""] =
#         #d1_feat[""] =
#         #d1_feat[""] =
#         #d1_feat[""] =
#         #print(c.IsRational(), c.IsClosed(), c.IsPeriodic(), c.Continuity(), c.Degree())
#     else:
#         print("Unsupported type", c_type)
#     return d1_feat

# def convert_2dcurve(curve):
#     d1_feat = {"type": edge_map[curve.GetType()], "interval": [curve.FirstParameter(), curve.LastParameter()]}
#     c_type = d1_feat["type"]
#     if c_type == "Line":
#         c = curve.Line()
#         d1_feat["location"] = list(c.Location().Coord())
#         d1_feat["direction"] = list(c.Direction().Coord())
#         #scale_factor = 1000.0
#     elif c_type == "Circle":
#         c = curve.Circle()
#         d1_feat["location"] = list(c.Location().Coord())
#         d1_feat["radius"] = c.Radius()
#         d1_feat["x_axis"] = list(c.XAxis().Direction().Coord())
#         d1_feat["y_axis"] = list(c.YAxis().Direction().Coord())
#     elif c_type == "Ellipse":
#         c = curve.Ellipse()
#         d1_feat["focus1"] = list(c.Focus1().Coord())
#         d1_feat["focus2"] = list(c.Focus2().Coord())
#         d1_feat["x_axis"] = list(c.XAxis().Direction().Coord())
#         d1_feat["y_axis"] = list(c.YAxis().Direction().Coord())
#         #d1_feat["z_axis"] = list(c.Axis().Direction().Coord())
#         d1_feat["maj_radius"] = c.MajorRadius()
#         d1_feat["min_radius"] = c.MinorRadius()
#         #scale_factor = 1.0
#     elif c_type == "BSpline":
#         c = curve.BSpline()
#         c.SetNotPeriodic()
#         d1_feat["rational"] = c.IsRational()
#         d1_feat["closed"] = c.IsClosed()
#         #d1_feat["periodic"] = c.IsPeriodic()
#         d1_feat["continuity"] = c.Continuity()
#         d1_feat["degree"] = c.Degree()
#         p = TColgp_Array1OfPnt2d(1, c.NbPoles())
#         c.Poles(p)
#         points = []
#         for pi in range(p.Length()):
#             points.append(list(p.Value(pi+1).Coord()))
#         d1_feat["poles"] = points
#
#         k = TColStd_Array1OfReal(1, c.NbPoles() + c.Degree() + 1)
#         c.KnotSequence(k)
#         knots = []
#         for ki in range(k.Length()):
#             knots.append(k.Value(ki+1))
#         d1_feat["knots"] = knots
#
#         w = TColStd_Array1OfReal(1, c.NbPoles())
#         c.Weights(w)
#         weights = []
#         for wi in range(w.Length()):
#             weights.append(w.Value(wi+1))
#         d1_feat["weights"] = weights
#     else:
#         print("Unsupported type", c_type)
#     return d1_feat

def convert_surface(surf):
    
    d2_feat = {"type": surf_map[surf.GetType()]}
#    s_type = d2_feat["type"]
#     if s_type == "Plane":
#         s = surf.Plane()
#         d2_feat["location"] = list(s.Location().Coord())
#         d2_feat["z_axis"] = list(s.Axis().Direction().Coord())
#         d2_feat["x_axis"] = list(s.XAxis().Direction().Coord())
#         d2_feat["y_axis"] = list(s.YAxis().Direction().Coord())
#         d2_feat["coefficients"] = list(s.Coefficients())
#
#     elif s_type == "Cylinder":
#         s = surf.Cylinder()
#         d2_feat["location"] = list(s.Location().Coord())
#         d2_feat["z_axis"] = list(s.Axis().Direction().Coord())
#         d2_feat["x_axis"] = list(s.XAxis().Direction().Coord())
#         d2_feat["y_axis"] = list(s.YAxis().Direction().Coord())
#         d2_feat["coefficients"] = list(s.Coefficients())
#         d2_feat["radius"] = s.Radius()
#
#     elif s_type == "Cone":
#         s = surf.Cone()
#         d2_feat["location"] = list(s.Location().Coord())
#         d2_feat["z_axis"] = list(s.Axis().Direction().Coord())
#         d2_feat["x_axis"] = list(s.XAxis().Direction().Coord())
#         d2_feat["y_axis"] = list(s.YAxis().Direction().Coord())
#         d2_feat["coefficients"] = list(s.Coefficients())
#         d2_feat["radius"] = s.RefRadius()
#         d2_feat["angle"] = s.SemiAngle()
#         d2_feat["apex"] = list(s.Apex().Coord())
#
#     elif s_type == "Sphere":
#         s = surf.Sphere()
#         d2_feat["location"] = list(s.Location().Coord())
#         d2_feat["x_axis"] = list(s.XAxis().Direction().Coord())
#         d2_feat["y_axis"] = list(s.YAxis().Direction().Coord())
#         d2_feat["coefficients"] = list(s.Coefficients())
#         d2_feat["radius"] = s.Radius()
#
#     elif s_type == "Torus":
#         s = surf.Torus()
#         d2_feat["location"] = list(s.Location().Coord())
#         d2_feat["z_axis"] = list(s.Axis().Direction().Coord())
#         d2_feat["x_axis"] = list(s.XAxis().Direction().Coord())
#         d2_feat["y_axis"] = list(s.YAxis().Direction().Coord())
#         #d2_feat["coefficients"] = list(s.Coefficients())
#         d2_feat["max_radius"] = s.MajorRadius()
#         d2_feat["min_radius"] = s.MinorRadius()
#
#
#     elif s_type == "Bezier":
#         print("BEZIER SURF")
#
#     elif s_type == "BSpline":
#         c = surf.BSpline()
#         c.SetUNotPeriodic()
#         c.SetVNotPeriodic()
#         d2_feat["u_rational"] = c.IsURational()
#         d2_feat["v_rational"] = c.IsVRational()
#         d2_feat["u_closed"] = c.IsUClosed()
#         d2_feat["v_closed"] = c.IsVClosed()
#         #d2_feat["u_periodic"] = c.IsUPeriodic()
#         #d2_feat["v_periodic"] = c.IsVPeriodic()
#         d2_feat["continuity"] = c.Continuity()
#         d2_feat["u_degree"] = c.UDegree()
#         d2_feat["v_degree"] = c.VDegree()
#
#         p = TColgp_Array2OfPnt(1, c.NbUPoles(), 1, c.NbVPoles())
#         c.Poles(p)
#         points = []
#         for pi in range(p.ColLength()):
#             elems = []
#             for pj in range(p.RowLength()):
#                 elems.append(list(p.Value(pi+1, pj+1).Coord()))
#             points.append(elems)
#         d2_feat["poles"] = points
#
#         k = TColStd_Array1OfReal(1, c.NbUPoles() + c.UDegree() + 1)
#         c.UKnotSequence(k)
#         knots = []
#         for ki in range(k.Length()):
#             knots.append(k.Value(ki+1))
#         d2_feat["u_knots"] = knots
#
#         k = TColStd_Array1OfReal(1, c.NbVPoles() + c.VDegree() + 1)
#         c.VKnotSequence(k)
#         knots = []
#         for ki in range(k.Length()):
#             knots.append(k.Value(ki+1))
#         d2_feat["v_knots"] = knots
#
#         w = TColStd_Array2OfReal(1, c.NbUPoles(), 1, c.NbVPoles())
#         c.Weights(w)
#         weights = []
#         for wi in range(w.ColLength()):
#             elems = []
#             for wj in range(w.RowLength()):
#                 elems.append(w.Value(wi+1, wj+1))
#             weights.append(elems)
#         d2_feat["weights"] = weights
#
#         scale_factor = 1.0
#
#     elif s_type == "Revolution":
#         s = surf.AxeOfRevolution()
#         c = surf.BasisCurve()
#         #print(surf, dir(surf), dir(c))
#         d1_feat = convert_curve(c)
#         d2_feat["location"] = list(s.Location().Coord())
#         d2_feat["z_axis"] = list(s.Direction().Coord())
#         d2_feat["curve"] = d1_feat
#
#     elif s_type == "Extrusion":
# #                print(dir(surf.Direction()))
#         c = surf.BasisCurve()
#         d1_feat = convert_curve(c)
#         d2_feat["direction"] = list(surf.Direction().Coord())
#         d2_feat["curve"] = d1_feat
#
#     else:
#         print("Unsupported type", s_type)
    
    return d2_feat


def mesh_model(model, max_size=1e-5, tolerance=1e-7, repair=False, terminal=1):
    # In/Output definitions
    #fil = model.split("/")[-1][:-5]
    #folder = "/".join(model.split("/")[:-1])
    scale_factor = 1000.0
    verts = []
    #norms = []
    faces = []
    #curvs = []
    vert_map = {}
    #d1_feats = []
    d2_feats = []
    #t_curves = []
    #norm_map = {}
    with fileinput.FileInput(model, inplace=True) as fi:
        for line in fi:
            print(line.replace("UNCERTAINTY_MEASURE_WITH_UNIT( LENGTH_MEASURE( 1.00000000000000E-06 )",
                           "UNCERTAINTY_MEASURE_WITH_UNIT( LENGTH_MEASURE( 1.00000000000000E-17 )"), end='')

    #stats = {}

    # OCC definitions
    occ_steps = read_step_file(model)

    total_edges = 0
    total_surfs = 0

    for l in range(len(occ_steps)):
        topo = TopologyExplorer(occ_steps[l])
        total_edges += len(list(topo.edges()))
        total_surfs += len(list(topo.faces()))
        # vol = brepgprop_VolumeProperties(occ_steps[l], occ_props, tolerance)
        # print(dir(occ_props), dir(occ_props.PrincipalProperties()), dir(occ_props.volume()), occ_props.Mass())
        # sur = brepgprop_SurfaceProperties(occ_steps[l], occ_props, tolerance)
        # print(vol, "Test", sur)

    if (total_surfs > 300):
        print("Skipping model {}, too many surfaces: {}".format(os.path.basename(model), total_surfs))
        return

    #print(total_surfs, "surfaces")

    #stats["#parts"] = len(occ_steps)
    #stats["model"] = model
    #print("Reading step %s with %i parts."%(model,len(occ_steps)))
    #tot = 0
    #for s in occ_steps:
    #    occ_topo = TopologyExplorer(s)
    #    print(s)
    #    print(len(list(occ_topo.edges())))
    #    tot += len(list(occ_topo.edges()))
    occ_cnt = 0
    bbox =  get_boundingbox(occ_steps[occ_cnt], use_mesh=True)
    diag = np.sqrt(bbox[6]**2+bbox[7]**2+bbox[8]**2)
    max_length = diag * max_size#, 9e-06
    tolerance = diag * tolerance
    #print(fil, diag, max_length, tolerance)
    # stats["bbox"] = bbox
    # stats["max_length"] = float(max_length)
    # stats["tolerance"] = float(tolerance)
    # stats["diag"] = float(diag)

    occ_topo = TopologyExplorer(occ_steps[occ_cnt])
    #occ_top = Topo(occ_steps[occ_cnt])
    #occ_props = GProp_GProps()
    #occ_brt = BRep_Tool()

    # Gmsh definitions
    gmsh.initialize()
    gmsh.option.setNumber("General.Terminal", terminal)
    gmsh.clear()
    if (tolerance > 1e6):
        print("Ignoring large tolerance:", tolerance)
    else:
        gmsh.option.setNumber("Geometry.Tolerance", tolerance)
    gmsh.option.setNumber("Geometry.OCCFixDegenerated", 0)
    gmsh.option.setNumber("Geometry.OCCFixSmallEdges", 0)
    gmsh.option.setNumber("Geometry.OCCFixSmallFaces", 0)
    gmsh.option.setNumber("Geometry.OCCSewFaces", 0)
    #gmsh.option.setNumber("Mesh.MeshSizeMax", max_length)
    gmsh.option.setNumber("Mesh.MeshSizeFactor", max_size)
    gmsh.option.setNumber("Mesh.AlgorithmSwitchOnFailure", 0)  # Fallback to Mesh-Adapt ends hanging up sometimes.
    # gmsh.option.setNumber("General.NumThreads", 6)
    # gmsh.option.setNumber("Mesh.MaxNumThreads1D",6)
    # gmsh.option.setNumber("Mesh.MaxNumThreads2D",6)
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 1)
    gmsh.option.setNumber("Mesh.MinimumElementsPerTwoPi",8)
    gmsh.option.setNumber("General.ExpertMode",1)

    gmsh.open(model)
    gmsh_edges = gmsh.model.getEntities(1)
    gmsh_surfs = gmsh.model.getEntities(2)
    gmsh_entities = gmsh.model.getEntities()

    # stats["#edges"] = total_edges
    # stats["#surfs"] = total_surfs
    # stats["volume"] = vol
    # stats["surface"] = sur
    # stats["curves"] = []
    # stats["surfs"] = []
    # stats["#points"] = 0
    #print("Number of surfaces: %i, Number of curves: %i"%(total_surfs, total_edges))
    #print(total_edges, total_surfs, len(gmsh_edges), len(gmsh_surfs))
    if not total_edges == len(gmsh_edges):
        print("Skipping due to wrong EDGES", model)
        return
    if not total_surfs == len(gmsh_surfs):
        print("Skipping due to wrong SURFS", model)
        return

    gmsh.model.mesh.generate(2)

    #print("Reading curvature")
    v_cnt = 1
    #v_nodes = []
    occ_offset = 0
    invalid_model = False
    #c_cnt = 0

    #v_cont_cnt = 0
    #print(len(list(occ_topo.edges())), len(list(occ_topo.solids())), len(list(occ_topo.faces())), len(list(occ_topo.vertices())))
    for e in gmsh_entities[:]:
        #print(e)
        nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(e[0], e[1], True)
        #elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(e[0], e[1])
        n_id = e[1] - occ_offset
        #print(e, occ_offset, n_id)
        #print(e, nodeTags, nodeCoords, nodeParams, gmsh.model.getType(e[0], e[1]), elemTypes, elemTags, elemNodeTags)
        if e[0] == 0: # Process points
            #print(e[1], nodeCoords)
            vert_map[e[1]] = v_cnt
            verts.append([nodeCoords[0] * 1000.0, nodeCoords[1] * 1000.0, nodeCoords[2] * 1000.0])
            v_cnt += 1
            #stats["#points"] += 1
            
            #pass
        if e[0] == 1: # Process contours
            if n_id - 1 == len(list(occ_topo.edges())):
                #print("CNT", occ_cnt)
                occ_cnt += 1
                occ_offset = e[1] - 1
                #n_id = 1
                occ_topo = TopologyExplorer(occ_steps[occ_cnt])
                #occ_top = Topo(occ_steps[occ_cnt])
                #print("Defunct curve", n_id, len(list(occ_topo.edges())))
                #continue
            #print(n_id)
            #curve = BRepAdaptor_Curve(list(occ_topo.edges())[n_id-1])
            # Add type and parametric nodes/indices
            #print("type", edge_map[curve.GetType()])
            if gmsh.model.getType(e[0], e[1]) == "Unknown":
                #print("Skipping OtherCurve", nodeTags)
                continue

            for i, n in enumerate(nodeTags):
                if n >= v_cnt:
                    vert_map[n] = v_cnt
                    verts.append([nodeCoords[i*3] * 1000.0, nodeCoords[i*3+1] * 1000.0, nodeCoords[i*3+2] * 1000.0])
                    v_cnt += 1
                #else:
                    #print(n, v_cnt)

            #print(v_ind, type(v_ind), v_par, type(v_par))
            #stats["curves"].append(edge_map[curve.GetType()])
            #print(n_id, edge_map[curve.GetType()], gmsh.model.getType(e[0], e[1]))
            #print(list(occ_topo.edges()), n_id-1)
            #c_type = edge_map[curve.GetType()]#gmsh.model.getType(e[0], e[1])
            # if not gmsh.model.getType(e[0], e[1]) == edge_map[curve.GetType()]:
            #     print("Skipped due to non matching edges ", model, gmsh.model.getType(e[0], e[1]), edge_map[curve.GetType()])
            #     #invalid_model = True
            #     #break

            #d1_feat = convert_curve(curve)
            
            #edg = list(occ_topo.edges())[n_id-1]
            
            #for f in occ_top.faces_from_edge(edg):
                #ee = (e)
                #print(dir(ee))
                #d1_feat = {}
                #su = BRepAdaptor_Surface(f)
                #c = BRepAdaptor_Curve2d(edg, f)
                #t_curve = {"surface": f, "3dcurve": c_cnt, "2dcurve": convert_2dcurve(c)}
                #print(edge_map[c.GetType()], surf_map[su.GetType()], edge_map[curve.GetType()])
                #d1f = convert_2dcurve(c)
                #print(d1f)
                #ccnt += 1
                #print(d1_feat)
                #t_curves.append(t_curve)

            # if len(elemNodeTags) > 0:
            #     #v_ind = [int(elemNodeTags[0][0]) - 1] # first vertex
            #     v_ind = [int(nodeTags[-2])-1]
            #     for no in nodeTags[:-2]:
            #         v_ind.append(int(no) - 1) # interior vertices
            #     v_ind.append(int(nodeTags[-1])-1)
            #     #v_ind.append(int(elemNodeTags[0][-1]) - 1) # last vertex
            #     #d1_feat["vert_indices"] = v_ind
            #     #v_par = [float(curve.FirstParameter())] # first param
            #     v_par = [float(nodeParams[-2]*scale_factor)]
            #     for no in nodeParams[:-2]:
            #         v_par.append(float(no*scale_factor)) # interior params
            #     v_par.append(float(nodeParams[-1]*scale_factor))
            #     #v_par.append(float(curve.LastParameter())) # last param
            #     #d1_feat["vert_parameters"] = v_par
            # else:
            #     print("No nodetags", edge_map[curve.GetType()], elemNodeTags)

            #print("VERTS", len(d1_feat["vert_indices"]), len(d1_feat["vert_parameters"]))
            #d1_feats.append(d1_feat)
            #c_cnt += 1
            #t_curve = curve.Trim(curve.FirstParameter(), curve.LastParameter(), 0.0001).GetObject()
            #print(curve.FirstParameter(), curve.LastParameter())


    gmsh_entities = gmsh.model.getEntities(2)
    #print("Processing {} surfaces".format(len(gmsh_entities)))
    n_cnt = 1
    occ_offset = 0
    occ_cnt = 0
    occ_topo = TopologyExplorer(occ_steps[occ_cnt])
    #occ_top = Topo(occ_steps[occ_cnt])
    f_cnt = 0
    f_sum = 0
    #first_face = True
    #mean_curv = 0.0
    #curv_cnt = 0
    #gaus_curv = 0.0
    s_cnt = 0
    
    for e in gmsh_entities[:]:
        #print(e)
        nodeTags, nodeCoords, nodeParams = gmsh.model.mesh.getNodes(e[0], e[1], True)
        elemTypes, elemTags, elemNodeTags = gmsh.model.mesh.getElements(e[0], e[1])
        n_id = e[1] - occ_offset
        #print(e, occ_offset, n_id)
        #print(e, nodeTags, nodeCoords, nodeParams, gmsh.model.getType(e[0], e[1]), elemTypes, elemTags, elemNodeTags)
        if e[0] == 2:
            #print(e, gmsh.model.getType(e[0], e[1]), elemTypes)
            if n_id - 1 == len(list(occ_topo.faces())):
                #print("CNT", occ_cnt)
                occ_cnt += 1
                occ_offset = e[1] - 1
                n_id = 1
                occ_topo = TopologyExplorer(occ_steps[occ_cnt])
                #occ_top = Topo(occ_steps[occ_cnt])
            # if "getNormals" in dir(gmsh.model):
            #     nls = gmsh.model.getNormals(e[1], nodeParams)
            # else:
            #     nls = gmsh.model.getNormal(e[1], nodeParams)
            #curvMax, curvMin, dirMax, dirMin = gmsh.model.getPrincipalCurvatures(e[1], nodeParams)
            #surf = BRepAdaptor_Surface(list(occ_topo.faces())[n_id-1])
            norm_map = {}
            for i, n in enumerate(nodeTags):
                #norms.append([nls[i*3], nls[i*3+1], nls[i*3+2]])
                #curvs.append([curvMin[i], curvMax[i], dirMin[i*3], dirMin[i*3+1], dirMin[i*3+2], dirMax[i*3], dirMax[i*3+1], dirMax[i*3+2]])
                #curv_cnt += 1
                #mean_curv += (curvMin[i] + curvMax[i])/2.0
                #gaus_curv += (curvMin[i] * curvMax[i])
                norm_map[n] = n_cnt
                n_cnt += 1
                if n in vert_map.keys():
                    #v = verts[vert_map[n]-1]
                    #print("Vert contained", n)
                    #v_cont_cnt += 1
#                    assert(v[0] == nodeCoords[i*3] * 1000.0 and v[1] == nodeCoords[i*3+1] * 1000.0 and v[2] == nodeCoords[i*3+2] * 1000.0)
                    continue
                else:
                    vert_map[n] = v_cnt
                #occ_node = surf.Value(nodeParams[i], nodeParams[i+1])
                #vertices.append([occ_node.X(), occ_node.Y(), occ_node.Z()])
                verts.append([nodeCoords[i*3] * 1000.0, nodeCoords[i*3+1] * 1000.0, nodeCoords[i*3+2] * 1000.0])
                #print("S", occ_node.Coord(), [nodeCoords[i*3]*1000, nodeCoords[i*3+1]*1000, nodeCoords[i*3+2]*1000])
                #print(occ_node.Coord(), nodeCoords[i*3:(i+1)*3])
                v_cnt += 1

            d2_faces = []
            for i, t in enumerate(elemTypes):
                for j in range(len(elemTags[i])):
                    faces.append([vert_map[elemNodeTags[i][j*3]], vert_map[elemNodeTags[i][j*3+1]], vert_map[elemNodeTags[i][j*3+2]], norm_map[elemNodeTags[i][j*3]], norm_map[elemNodeTags[i][j*3+1]], norm_map[elemNodeTags[i][j*3+2]]]) 
                    d2_faces.append(f_cnt)
                    f_cnt += 1
            #print(len(list(occ_topo.faces())), n_id-1)
            surf = BRepAdaptor_Surface(list(occ_topo.faces())[n_id-1])
            #print("type", edge_map[curve.GetType()])
            #if gmsh.model.getType(e[0], e[1]) == "Unknown":
            #    print("Skipping OtherCurve", nodeTags)
            #    continue  
            #print(surf)
            g_type = gmsh_map[gmsh.model.getType(e[0], e[1])]
            if g_type != "Other" and not g_type == surf_map[surf.GetType()]:
                print("Skipped due to non matching surfaces ", model, g_type, surf_map[surf.GetType()])
                return
                #invalid_model = True
                #break

            #stats["surfs"].append(surf_map[surf.GetType()])

            d2_feat = convert_surface(surf)
            d2_feat["face_indices"] = d2_faces
            
            # for tc in t_curves:
            #     if tc["surface"] == list(occ_topo.faces())[n_id-1]:
            #         tc["surface"] = s_cnt

            # if len(elemNodeTags) > 0:
            #     #print(len(elemNodeTags[0]), len(nodeTags), len(nodeParams))
            #     v_ind = []#int(elemNodeTags[0][0])] # first vertex
            #     for no in nodeTags:
            #         v_ind.append(int(no) - 1) # interior vertices
            #     #v_ind.append(int(elemNodeTags[0][-1])) # last vertex
            #     d2_feat["vert_indices"] = v_ind
            #     v_par = []#float(surf.FirstParameter())] # first param
            #     for io in range(int(len(nodeParams)/2)):
            #         v_par.append([float(nodeParams[io*2]*scale_factor), float(nodeParams[io*2+1]*scale_factor)]) # interior params
            #     #v_par.append(float(surf.LastParameter())) # last param
            #     d2_feat["vert_parameters"] = v_par
            # else:
            #     print("No nodetags", edge_map[surf.GetType()], elemNodeTags)

            f_sum += len(d2_feat["face_indices"])
            d2_feats.append(d2_feat)
            s_cnt += 1

    if invalid_model:
        return

    #stats["#sharp"] = 0
    #stats["gaus_curv"] = float(gaus_curv / curv_cnt)
    #stats["mean_curv"] = float(mean_curv / curv_cnt)

    if not f_sum == len(faces):
        print("Skipping due to wrong FACES", model)
        return
    # sharp flags not needed
    # if True:
    #     vert2norm = {}
    #     for f in faces:
    #         #print(f)
    #         for fii in range(3):
    #             if f[fii] in vert2norm:
    #                 vert2norm[f[fii]].append(f[fii+3])
    #             else:
    #                 vert2norm[f[fii]] = [f[fii+3]]
    #     for d1f in d1_feats:
    #         sharp = True
    #         for vi in d1f["vert_indices"][1:-1]:
    #             #print(vi, vert2norm.keys())
    #             nos = list(set(vert2norm[vi + 1]))
    #             if len(nos) == 2:
    #                 n0 = np.array(norms[nos[0]])
    #                 n1 = np.array(norms[nos[1]])
    #                 #print(np.linalg.norm(n0), np.linalg.norm(n1))
    #                 if np.abs(n0.dot(n1)) > 0.95:
    #                     sharp = False
    #                     #break
    #             else:
    #                 sharp = False
    #         if sharp:
    #             stats["#sharp"] += 1
    #         d1f["sharp"] = sharp

    #stats["#verts"] = len(verts)
    #stats["#faces"] = len(faces)
    #stats["#norms"] = len(norms)

    #with open("results/" + file + ".json", "w") as fil:
    #    json.dump(d1_feats, fil, sort_keys=True, indent=2)
    #with open("results/" + file + "_faces.json", "w") as fil:
     #   json.dump(d2_feats, fil, sort_keys=True, indent=2)
    
    features = {"surfaces": d2_feats}
    if True:
        # res_path = folder.replace("/step/", "/feat/")
        # fip = fil.replace("_step_", "_features_")
        # print("%s/%s.yml"%(res_path, fip))
        # with open("%s/%s.yml"%(res_path, fip), "w") as fili:
        #     yaml.dump(features, fili, indent=2)

        # res_path = folder.replace("/step/", "/stat/")
        # fip = fil.replace("_step_", "_stats_")
        # with open("%s/%s.yml"%(res_path, fip), "w") as fili:
        #     yaml.dump(stats, fili, indent=2)

        # print("Generated model with %i vertices and %i faces." %(len(verts), len(faces)))
        # res_path = folder.replace("/step/", "/obj/")
        # fip = fil.replace("_step_", "_trimesh_")
        # with open("%s/%s.obj"%(res_path, fip), "w") as fili:
        #     for v in verts:
        #         fili.write("v %f %f %f\n"%(v[0], v[1], v[2]))
        #     for vn in norms:
        #         #print(np.linalg.norm(vn))
        #         fili.write("vn %f %f %f\n"%(vn[0], vn[1], vn[2]))
        #     for vn in curvs:
        #         fili.write("vc %f %f %f %f %f %f %f %f\n"%(vn[0], vn[1], vn[2], vn[3], vn[4], vn[5], vn[6], vn[7]))
        #     for f in faces:
        #         fili.write("f %i//%i %i//%i %i//%i\n"%(f[0], f[3], f[1], f[4], f[2], f[5]))

        faces = np.array(faces)
        face_indices = faces[:, :3] - 1
        #norm_indices = faces[:, 3:] - 1
    gmsh.clear()
    gmsh.finalize()
    #print(curvs)
    return {"features": features, "vertices": np.array(verts), "faces": faces, "face_indices": face_indices}
