import sys
import numpy as np
import math
import pandas as pd

# this is done automatically when running the script from inside FreeCAD
# import FreeCAD as App
# import Part
# import Sketcher

# there are two main modes of using this script ('local', 'internet')
mode = 'local'

# when using 'local' mode, 4-digit NACA profiles for the hub and tip:
profile_hub = '0012'
profile_tip = '6414'

# when using 'internet' mode, airfoils to be used from http://airfoiltools.com
# none airfoil can end at (1, 0), i.e. no sharp trailing edge
airfoils = ('a18sm-il', 'avistar-il', 'ls413-il', 'pmc19sm-il', 'wb140-il')

chord_hub = 340 # mm
chord_tip = 0.2*chord_hub # mm
pitch_tip = 60 # degree
x_tip = 0.1*chord_hub # mm
y_tip = 0.2*chord_hub # mm
span = 1200 # mm

# should be equal to len(airfoils) if you are using 'internet' mode
if mode == 'local':
    n_span = 11
elif mode == 'internet':
    n_span = len(airfoils)
else:
    print('Unknown mode')
    sys.exit()

# trailing edge type ('blunt', 'round')
TE_type = 'blunt'

# discretization of the trailing edge
if TE_type == 'round':
    n_intervals_TE = 6 # should be an even number, as to ensure a mid point for the trailing edge line

### END USER-DEFINED PARAMETERS ###


# for the definition of the equations below, see https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil
def get_yt(x, t):
    return 5*t*(0.2969*x**0.5 - 0.1260*x - 0.3516*x**2 + 0.2843*x**3 - 0.1015*x**4)

def get_yc(x, m, p):
    return np.where((x >= 0) & (x < p), m/p**2 * (2*p*x - x**2), m/(1-p)**2 * ((1 - 2*p) + 2*p*x - x**2))

def get_dycdx(x, m, p):
    return np.where((x >= 0) & (x < p), 2*m/p**2 * (p - x), 2*m/(1 - p)**2 * (p - x))

def get_theta(x, m, p):
    dycdx = get_dycdx(x, m, p)
    return np.arctan(dycdx)

def get_coords(x, m, p, t):
    # symmetric airfoils
    if m == 0 and p == 0:
        yt = get_yt(x, t)

        # x_upper, x_lower, y_upper, y_lower
        return [x, x, yt, -yt]

    # cambered airfoils
    else:
        yt = get_yt(x, t)
        yc = get_yc(x, m, p)
        theta = get_theta(x, m, p)

        # x_upper, x_lower, y_upper, y_lower
        return [x - yt*np.sin(theta), x + yt*np.sin(theta), yc + yt*np.cos(theta), yc - yt*np.cos(theta)]


if mode == 'local':
    # we need to properly discretize the chord direction of the airfoil, especially at the leading edge, where the curvature is stronger
    x = []
    x.extend(np.linspace(0.00, 0.05, 4, endpoint=False))
    x.extend(np.linspace(0.05, 0.15, 4, endpoint=False))
    x.extend(np.linspace(0.15, 0.35, 2, endpoint=False))
    x.extend(np.linspace(0.35, 0.75, 2, endpoint=False))
    x.extend(np.linspace(0.75, 1.00, 3))
    x = np.array(x)

    # for the definition of the variables below, see https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil
    m_hub = int(profile_hub[0]) / 100
    p_hub = int(profile_hub[1]) / 10
    t_hub = int(profile_hub[2] + profile_hub[3]) / 100

    m_tip = int(profile_tip[0]) / 100
    p_tip = int(profile_tip[1]) / 10
    t_tip = int(profile_tip[2] + profile_tip[3]) / 100


# some constrains are not mandatory for the successful creation of the solid;
# I have not managed so far to find a way to create a fully constrained sketch for the airfoild profile in all cases
constrained = True

document = 'Unnamed'

App.newDocument()
App.activeDocument().addObject('PartDesign::Body','Body')
App.ActiveDocument.getObject('Body').Label = 'Body'

spans = np.linspace(0.0, span, n_span) # mm
chords = np.linspace(chord_hub, chord_tip, n_span) # mm
pitchs = np.linspace(0, pitch_tip, n_span) # degree
delta_x = np.linspace(0, x_tip, n_span) # mm
delta_y = np.linspace(0, y_tip, n_span) # mm

if mode == 'local':
    m = np.linspace(m_hub, m_tip, n_span)
    p = np.linspace(p_hub, p_tip, n_span)
    t = np.linspace(t_hub, t_tip, n_span)


sketch_names = []
for span_id, span_height in enumerate(spans):
    # retrieve current airfoil coordinates
    if mode == 'local':
        x_upper, x_lower, y_upper, y_lower = get_coords(x, m[span_id], p[span_id], t[span_id])

    else:
        data = pd.read_csv('http://airfoiltools.com/airfoil/lednicerdatfile?airfoil=' + airfoils[span_id])
        data[data.columns[0]] = data[data.columns[0]].str.strip()
        data[['x', 'y']] = data[data.columns[0]].str.split(' ', n=1, expand=True)
        data = data.drop(data.columns[0], axis=1)
        data = data.astype(float)

        # the lednicer dat files in http://airfoiltools.com are structured in such a way that the rows start at the leading edge,
        # cover the suction (upper) side until the trailing edge,
        # then come back to the leading edge and cover the pressure (lower) side until the trailing edge;
        # data['x'][0] contains the number of points discretizing the suction (upper) side,
        # whereas data['y'][0] contains the number of points discretizing the pressure (lower) side
        x_upper = np.array(data['x'][1:int(data['x'][0])+1])
        y_upper = np.array(data['y'][1:int(data['x'][0])+1])
        x_lower = np.array(data['x'][int(data['x'][0])+1:])
        y_lower = np.array(data['y'][int(data['x'][0])+1:])

    sketch = 'Sketch' + str(span_id)
    sketch_names.append(sketch)

    App.getDocument('Unnamed').getObject('Body').newObject('Sketcher::SketchObject',sketch)
    App.getDocument('Unnamed').getObject(sketch).AttachmentSupport = (App.getDocument('Unnamed').getObject('XY_Plane'),[''])
    App.getDocument('Unnamed').getObject(sketch).MapMode = 'FlatFace'
    App.getDocument('Unnamed').getObject(sketch).AttachmentOffset = App.Placement(App.Vector(delta_x[span_id],delta_y[span_id],span_height),App.Rotation(App.Vector(0,0,1),pitchs[span_id]))

    x_upper_chord = x_upper*chords[span_id]
    x_lower_chord = x_lower*chords[span_id]
    y_upper_chord = y_upper*chords[span_id]
    y_lower_chord = y_lower*chords[span_id]

    vectors_upper = []
    vectors_lower = []

    points_upper = []
    points_lower = []

    point_ids_upper = []
    point_ids_lower = []

    # each geometrical entity in FreeCAD has an ID, assigned by order of creation
    object_id = -1
    for i in range(len(x_upper)):
        # suction side (upper points)
        object_id += 1
        vectors_upper.append(App.Vector(x_upper_chord[-i-1], y_upper_chord[-i-1], 0))
        points_upper.append(Part.Point(vectors_upper[i]))
        point_ids_upper.append(object_id)
        App.getDocument('Unnamed').getObject(sketch).addGeometry(points_upper[i], True) # the boolean flag is construction mode or not

        # constraining the points does not ensure the spline by these points will be constrained; I don't know why
        if constrained:
            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceX',point_ids_upper[i],1,x_upper_chord[-i-1]))
            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceY',point_ids_upper[i],1,y_upper_chord[-i-1]))

    for i in range(len(x_lower)-1):
        # pressure side (lower points)
        object_id += 1
        vectors_lower.append(App.Vector(x_lower_chord[i+1], y_lower_chord[i+1], 0))
        points_lower.append(Part.Point(vectors_lower[i]))
        point_ids_lower.append(object_id)
        App.getDocument('Unnamed').getObject(sketch).addGeometry(points_lower[i], True) # the boolean flag is construction mode or not

        # constraining the points does not ensure the spline by these points will be constrained; I don't know why
        if constrained:
            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceX',point_ids_lower[i],1,x_lower_chord[i+1]))
            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceY',point_ids_lower[i],1,y_lower_chord[i+1]))


    # We don't need to create the begin of chord (0,0) point, as it is automatically defined in FreeCAD and has id = -1

    # end of chord (auxiliary point) # not being used for the moment
    #    object_id += 1
    #    chord_end_id = object_id
    #
    #    chord_end_vector = App.Vector(chords[span_id], 0, 0)
    #    chord_end_point = Part.Point(chord_end_vector)
    #
    #    App.getDocument('Unnamed').getObject(sketch).addGeometry(chord_end_point, True)
    #    if constrained:
    #        App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceX',chord_end_id,1,chords[span_id]))
    #        #App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceY',chord_end_id,1,0))
    #        App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('PointOnObject',chord_end_id,1,-1)) # the end of chord always lies in the x axis


    if TE_type == 'round':
        # Straight line between first and last point
        App.getDocument('Unnamed').getObject(sketch).addGeometry(Part.LineSegment(vectors_upper[0], vectors_lower[-1]), True)

        # get the id of the newly created line
        #TE_line_id = len(App.getDocument('Unnamed').getObject(sketch).Geometry) - 1
        object_counter = 0
        for id, element in enumerate(App.getDocument('Unnamed').getObject(sketch).Geometry):
            #print(id, element, element.__class__.__name__)
            if 'LineSegment' in element.__class__.__name__:
                object_counter += 1
            if object_counter == 1:
                TE_line_id = id
                break
        print("TE_line_id =", TE_line_id)

        # now add constraints
        constraintList = []
        constraintList.append(Sketcher.Constraint('Coincident', TE_line_id, 1, point_ids_upper[ 0], 1))
        constraintList.append(Sketcher.Constraint('Coincident', TE_line_id, 2, point_ids_lower[-1], 1))
        App.getDocument('Unnamed').getObject(sketch).addConstraint(constraintList)
        del constraintList


        # Split newly created line
        TE_mid_x = (x_upper_chord[-1] + x_lower_chord[-1]) / 2
        TE_mid_y = (y_upper_chord[-1] + y_lower_chord[-1]) / 2
        TE_mid_vector = App.Vector(TE_mid_x, TE_mid_y, 0)
        App.getDocument('Unnamed').getObject(sketch).split(TE_line_id, TE_mid_vector)

        # get id
        #TE_lower_id = len(App.getDocument('Unnamed').getObject(sketch).Geometry) - 1
        object_counter = 0
        for id, element in enumerate(App.getDocument('Unnamed').getObject(sketch).Geometry):
            #print(id, element, element.__class__.__name__)
            if 'LineSegment' in element.__class__.__name__:
                object_counter += 1
            if object_counter == 2:
                TE_lower_id = id
                break
        print("TE_lower_id =", TE_lower_id)

        #sys.exit()
        # now add constraints
        if constrained:
            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceX',-1,1,TE_line_id,2,TE_mid_x))
            #App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceY',-1,1,TE_line_id,2,TE_mid_y))


        # Draw circle
        circle = Part.Circle(TE_mid_vector, App.Vector(0.000000, 0.000000, 1.000000), 0.428400)
        arcOfCircle = Part.ArcOfCircle(circle , 4.712389, 7.853982)
        App.getDocument('Unnamed').getObject(sketch).addGeometry(arcOfCircle,True)

        # get id
        #TE_arc_id = len(App.getDocument('Unnamed').getObject(sketch).Geometry) - 1
        object_counter = 0
        for id, element in enumerate(App.getDocument('Unnamed').getObject(sketch).Geometry):
            #print(id, element, element.__class__.__name__)
            if 'ArcOfCircle' in element.__class__.__name__:
                object_counter += 1
            if object_counter == 1:
                TE_arc_id = id
                break
        print("TE_arc_id =", TE_arc_id)

        # now add constraints
        constraintList = []
        constraintList.append(Sketcher.Constraint('Coincident', TE_arc_id, 3, TE_line_id         , 2))
        constraintList.append(Sketcher.Constraint('Coincident', TE_arc_id, 1, point_ids_lower[-1], 1))
        constraintList.append(Sketcher.Constraint('Coincident', TE_arc_id, 2, point_ids_upper[ 0], 1))
        App.getDocument('Unnamed').getObject(sketch).addConstraint(constraintList)
        del constraintList


        #sys.exit()
        # create points for the circular trailing edge
        TE_radius_ids = []
        TE_radius_vectors = []
        for temp_line_id in range(1, n_intervals_TE):
            App.getDocument('Unnamed').getObject(sketch).addGeometry(Part.LineSegment(TE_mid_vector, App.Vector(TE_mid_x + 0.1, TE_mid_y - 0.1, 0.000000)), True)

            # get id
            object_counter = 0
            for id, element in enumerate(App.getDocument('Unnamed').getObject(sketch).Geometry):
                #print(id, element, element.__class__.__name__)
                if 'LineSegment' in element.__class__.__name__:
                    object_counter += 1
                if object_counter == 2 + temp_line_id:
                    TE_radius_ids.append(id)
                    break
            print("temp_line_id =", temp_line_id, "TE_radius_ids =", TE_radius_ids[temp_line_id - 1])

            # now add constraints
            constraintList = []
            constraintList.append(Sketcher.Constraint('Coincident'   , TE_radius_ids[temp_line_id - 1], 1, TE_line_id, 2))
            constraintList.append(Sketcher.Constraint('PointOnObject', TE_radius_ids[temp_line_id - 1], 2, TE_arc_id    ))
            App.getDocument('Unnamed').getObject(sketch).addConstraint(constraintList)
            del constraintList

            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Angle',TE_lower_id,1,TE_radius_ids[temp_line_id - 1],1,math.radians(180/n_intervals_TE*temp_line_id)))

            # get vector of end point (which lies on the arc)
            TE_radius_vectors.append(App.getDocument('Unnamed').getObject(sketch).Geometry[TE_radius_ids[temp_line_id - 1]].EndPoint)


        #sys.exit()
        # we need to create unified lists for the variables below, comprising the upper, lower and TE parts of the airfoil
        # they start by the trailing edge point and go counter clockwise across the airfoil
        vectors = []
        for i in range(int(len(TE_radius_vectors)/2), len(TE_radius_vectors)):
            vectors.append(TE_radius_vectors[i])
        vectors.extend(vectors_upper)
        vectors.extend(vectors_lower)
        for i in range(int(len(TE_radius_vectors)/2)):
            vectors.append(TE_radius_vectors[i])
        print("len(vectors) =", len(vectors))

        points = []
        for i in range(int(len(TE_radius_vectors)/2), len(TE_radius_vectors)):
            points.append(Part.Point(TE_radius_vectors[i]))
        points.extend(points_upper)
        points.extend(points_lower)
        for i in range(int(len(TE_radius_vectors)/2)):
            points.append(Part.Point(TE_radius_vectors[i]))
        print("len(points) =", len(points))

        point_ids = []
        point_ids.extend(point_ids_upper)
        point_ids.extend(point_ids_lower)
        print("len(point_ids) =", len(point_ids))

        #sys.exit()
        for i in range(len(vectors)):
            App.getDocument('Unnamed').getObject(sketch).addGeometry(points[i],True)

        # add spline
        _finalbsp_poles = []
        _finalbsp_knots = []
        _finalbsp_mults = []

        spline_upper = Part.BSplineCurve()
        spline_upper.interpolate(vectors, PeriodicFlag=True)
        spline_upper.increaseDegree(3)

        _finalbsp_poles.extend(spline_upper.getPoles())
        _finalbsp_knots.extend(spline_upper.getKnots())
        _finalbsp_mults.extend(spline_upper.getMultiplicities())

        App.getDocument('Unnamed').getObject(sketch).addGeometry(Part.BSplineCurve(_finalbsp_poles,_finalbsp_mults,_finalbsp_knots,True,3,None,False),False)

        del(_finalbsp_poles)
        del(_finalbsp_knots)
        del(_finalbsp_mults)

        # get id
        object_counter = 0
        for id, element in enumerate(App.getDocument('Unnamed').getObject(sketch).Geometry):
            #print(id, element, element.__class__.__name__)
            if 'BSplineCurve' in element.__class__.__name__:
                object_counter += 1
            if object_counter == 1:
                spline_upper_id = id
                break
        print("spline_upper_id =", spline_upper_id)

        # now add constraints
        conList = []
        for i in range(len(vectors)):
            conList.append(Sketcher.Constraint('InternalAlignment:Sketcher::BSplineKnotPoint',TE_radius_ids[-1] + 1 + i,1,spline_upper_id,i))
        App.getDocument('Unnamed').getObject(sketch).addConstraint(conList)
        del conList

        constraintList = []
        current_id = TE_radius_ids[-1]
        for i in range(int(len(TE_radius_vectors)/2), len(TE_radius_vectors)):
            current_id += 1
            constraintList.append(Sketcher.Constraint('Coincident', current_id, 1, TE_radius_ids[i], 2))
        for i in range(len(point_ids)):
            current_id += 1
            constraintList.append(Sketcher.Constraint('Coincident', current_id, 1, point_ids[i], 1))
        for i in range(int(len(TE_radius_vectors)/2)):
            current_id += 1
            constraintList.append(Sketcher.Constraint('Coincident', current_id, 1, TE_radius_ids[i], 2))
        App.getDocument('Unnamed').getObject(sketch).addConstraint(constraintList)
        del constraintList


        # split spline at LE and TE
        App.getDocument('Unnamed').getObject(sketch).split(spline_upper_id,App.Vector(0,0,0))
        App.getDocument('Unnamed').getObject(sketch).split(spline_upper_id,TE_radius_vectors[int(len(TE_radius_vectors)/2)])


    # blunt trailing edge
    else:
        # Since some meshing software for CFD require the definition of the leading edge and the trailing edge, we will need to use two splines,
        # one for the suction side, one for the pressure side;

        #sys.exit()
        # we need to create unified lists for the variables below, comprising the upper and lower parts of the airfoil
        vectors = []
        vectors.extend(vectors_upper)
        vectors.extend(vectors_lower)
        print("len(vectors) =", len(vectors))

        points = []
        points.extend(points_upper)
        points.extend(points_lower)
        print("len(points) =", len(points))

        point_ids = []
        point_ids.extend(point_ids_upper)
        point_ids.extend(point_ids_lower)
        print("len(point_ids) =", len(point_ids))

        #sys.exit()
        for i in range(len(vectors)):
            App.getDocument('Unnamed').getObject(sketch).addGeometry(points[i],True)

        #sys.exit()
        # suction side (upper spline)
        _finalbsp_poles = []
        _finalbsp_knots = []
        _finalbsp_mults = []

        spline_upper = Part.BSplineCurve()
        spline_upper.interpolate(vectors, PeriodicFlag=False)
        spline_upper.increaseDegree(3)

        _finalbsp_poles.extend(spline_upper.getPoles())
        _finalbsp_knots.extend(spline_upper.getKnots())
        _finalbsp_mults.extend(spline_upper.getMultiplicities())

        App.getDocument('Unnamed').getObject(sketch).addGeometry(Part.BSplineCurve(_finalbsp_poles,_finalbsp_mults,_finalbsp_knots,False,3,None,False),False)

        del(_finalbsp_poles)
        del(_finalbsp_knots)
        del(_finalbsp_mults)

        # get id
        object_counter = 0
        for id, element in enumerate(App.getDocument('Unnamed').getObject(sketch).Geometry):
            #print(id, element, element.__class__.__name__)
            if 'BSplineCurve' in element.__class__.__name__:
                object_counter += 1
            if object_counter == 1:
                spline_upper_id = id
                break
        print("spline_upper_id =", spline_upper_id)

        conList = []
        for i in range(len(point_ids)):
            conList.append(Sketcher.Constraint('InternalAlignment:Sketcher::BSplineKnotPoint',len(point_ids) + i,1,spline_upper_id,i))

            # we could avoid having to pass by conList, but then the code takes longer to execute; I don't know why
            # App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('InternalAlignment:Sketcher::BSplineKnotPoint',point_ids_upper[i],1,spline_upper_id,i))

        App.getDocument('Unnamed').getObject(sketch).addConstraint(conList)
        del conList

        constraintList = []
        for i in range(len(point_ids)):
            constraintList.append(Sketcher.Constraint('Coincident', len(point_ids) + i, 1, point_ids[i], 1))
        App.getDocument('Unnamed').getObject(sketch).addConstraint(constraintList)
        del constraintList

#        if constrained:
#            # blocking the spline doesn't impede its trailing edge point from moving when creating the trailing edge; I don't know why
#            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Block',spline_upper_id))


        #sys.exit()
        # pressure side (lower spline)
        # split at point (0,0), which must be part of the airfoil profile
        App.getDocument('Unnamed').getObject(sketch).split(spline_upper_id,App.Vector(0,0,0))

        # let's retrieve the id of the newly created lower spline
        spline_counter = 0
        for id, element in enumerate(App.getDocument('Unnamed').getObject(sketch).Geometry):
            if 'BSplineCurve' in element.__class__.__name__:
                spline_counter += 1
            if spline_counter == 2:
                spline_lower_id = id
                break


        # The trailing edge will be blunt, but some meshing software are able to automatically change it for round
        #sys.exit()
        # trailing edge upper
        TE_upper_id = len(App.getDocument('Unnamed').getObject(sketch).Geometry)
        TE_upper_line = Part.LineSegment(vectors_upper[0],vectors_lower[-1])
        App.getDocument('Unnamed').getObject(sketch).addGeometry(TE_upper_line,False)

        constraintList = []
        constraintList.append(Sketcher.Constraint('Coincident', TE_upper_id, 1, spline_upper_id, 1))
        constraintList.append(Sketcher.Constraint('Coincident', TE_upper_id, 2, spline_lower_id, 2))
        #constraintList.append(Sketcher.Constraint('Vertical', TE_upper_id)) # may not be always the case
        App.getDocument('Unnamed').getObject(sketch).addConstraint(constraintList)
        del constraintList

        #sys.exit()
        # trailing edge lower
        # split at mid point of line
        TE_lower_id = len(App.getDocument('Unnamed').getObject(sketch).Geometry)
        TE_mid_x = (x_upper_chord[-1] + x_lower_chord[-1]) / 2
        TE_mid_y = (y_upper_chord[-1] + y_lower_chord[-1]) / 2
        TE_mid_vector = App.Vector(TE_mid_x, TE_mid_y, 0)
        App.getDocument('Unnamed').getObject(sketch).split(TE_upper_id, TE_mid_vector)

        if constrained:
            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceX',-1,1,TE_upper_id,2,TE_mid_x))
            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceY',-1,1,TE_upper_id,2,TE_mid_y))


    #sys.exit()
    # end of span loop

#sys.exit()
App.getDocument('Unnamed').getObject('Body').newObject('PartDesign::AdditiveLoft','AdditiveLoft')
App.getDocument('Unnamed').getObject('AdditiveLoft').Profile = App.getDocument('Unnamed').getObject('Sketch0')
App.getDocument('Unnamed').getObject('Sketch0').Visibility = False

for i in range(1, n_span):
    App.getDocument('Unnamed').getObject('AdditiveLoft').Sections += [(App.getDocument('Unnamed').getObject(sketch_names[i]), [''])] # sketch_names[1:]
    App.getDocument('Unnamed').getObject(sketch_names[i]).Visibility = False

# Show result
App.ActiveDocument.recompute()
