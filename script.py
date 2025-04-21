import sys
import numpy as np
import pandas as pd

# this is done automatically when running the script from inside FreeCAD
# import FreeCAD as App
# import Part
# import Sketcher

# there are two main modes of using this script ('local', 'internet')
mode = 'local'

# when using 'local' mode, 4-digit NACA profiles for the hub and tip:
profile_hub = '0012'
profile_tip = '2414'

# when using 'internet' mode, airfoils to be used from http://airfoiltools.com
# should be an odd number and none airfoil can end at (1, 0), i.e. no sharp trailing edge
airfoils = ('a18sm-il', 'avistar-il', 'ls413-il')

chord_hub = 340 # mm
chord_tip = 0.2*chord_hub # mm
pitch_tip = 60 # degree
x_tip = 0.1*chord_hub # mm
y_tip = 0.2*chord_hub # mm
span = 1200 # mm

# should be an odd number and equal to len(airfoils) if you are using 'internet' mode
if mode == 'local':
    n_span = 11
elif mode == 'internet':
    n_span = len(airfoils)
else:
    print('Unknown mode')
    sys.exit()

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


# we need to properly discretize the chord direction of the airfoil, especially at the leading edge, where the curvature is stronger
x = []
x.extend(np.linspace(0.00, 0.05, 10, endpoint=False))
x.extend(np.linspace(0.05, 0.15, 10, endpoint=False))
x.extend(np.linspace(0.15, 0.35, 10, endpoint=False))
x.extend(np.linspace(0.35, 0.75, 10, endpoint=False))
x.extend(np.linspace(0.75, 1.00, 11))
x = np.array(x)

# for the definition of the variables below, see https://en.wikipedia.org/wiki/NACA_airfoil#Equation_for_a_cambered_4-digit_NACA_airfoil
m_hub = int(profile_hub[0]) / 100
p_hub = int(profile_hub[1]) / 10
t_hub = int(profile_hub[2] + profile_hub[3]) / 100

m_tip = int(profile_tip[0]) / 100
p_tip = int(profile_tip[1]) / 10
t_tip = int(profile_tip[2] + profile_tip[3]) / 100

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

        x_upper = np.array(data['x'][1:int(data['x'][0])+1])
        x_lower = np.array(data['x'][int(data['x'][0])+1:])
        y_upper = np.array(data['y'][1:int(data['y'][0])+1])
        y_lower = np.array(data['y'][int(data['y'][0])+1:])

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
        vectors_upper.append(App.Vector(x_upper_chord[i], y_upper_chord[i], 0))
        points_upper.append(Part.Point(vectors_upper[i]))
        point_ids_upper.append(object_id)
        App.getDocument('Unnamed').getObject(sketch).addGeometry(points_upper[i], True) # the boolean flag is construction mode or not
        # constraining the points does not ensure the spline by these points will be constrained; I don't know why
#        if constrained:
#            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceX',point_ids_upper[i],1,x_upper_chord[i]))
#            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceY',point_ids_upper[i],1,y_upper_chord[i]))

    for i in range(len(x_lower)):
        # pressure side (lower points)
        object_id += 1
        vectors_lower.append(App.Vector(x_lower_chord[i], y_lower_chord[i], 0))
        points_lower.append(Part.Point(vectors_lower[i]))
        point_ids_lower.append(object_id)
        App.getDocument('Unnamed').getObject(sketch).addGeometry(points_lower[i], True) # the boolean flag is construction mode or not
        # constraining the points does not ensure the spline by these points will be constrained; I don't know why
#        if constrained:
#            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceX',point_ids_lower[i],1,x_lower_chord[i]))
#            App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('DistanceY',point_ids_lower[i],1,y_lower_chord[i]))


    # We don't need to create the begin of chord (0,0) point, as it is automatically defined in FreeCAD

    #sys.exit()
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


    # Since some meshing software for CFD require the definition of the leading edge and the trailing edge, we will need to use two splines,
    # one for the suction side, one for the pressure side;
    # there will be a small corner in the root (begin of chord) point,
    # but I have not managed so far to successfully apply a tangent constraint on the region

    #sys.exit()
    # suction side (upper spline)
    object_id += 1
    spline_upper_id = object_id

    _finalbsp_poles = []
    _finalbsp_knots = []
    _finalbsp_mults = []

    spline_upper = Part.BSplineCurve()
    spline_upper.interpolate(vectors_upper, PeriodicFlag=False)
    spline_upper.increaseDegree(3)

    _finalbsp_poles.extend(spline_upper.getPoles())
    _finalbsp_knots.extend(spline_upper.getKnots())
    _finalbsp_mults.extend(spline_upper.getMultiplicities())

    App.getDocument('Unnamed').getObject(sketch).addGeometry(Part.BSplineCurve(_finalbsp_poles,_finalbsp_mults,_finalbsp_knots,False,3,None,False),False)

    del(_finalbsp_poles)
    del(_finalbsp_knots)
    del(_finalbsp_mults)

    if constrained:
        conList = []
        for i in range(len(x_upper)):
            conList.append(Sketcher.Constraint('InternalAlignment:Sketcher::BSplineKnotPoint',point_ids_upper[i],1,spline_upper_id,i))

            # we could avoid having to pass by conList, but then the code takes longer to execute; I don't know why
            #App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('InternalAlignment:Sketcher::BSplineKnotPoint',point_ids_upper[i],1,spline_upper_id,i))

        App.getDocument('Unnamed').getObject(sketch).addConstraint(conList)
        del conList

        # the first point of the upper spline is equivalent to the origin of the x,y system (0,0), which has id = -1 in FreeCAD
        App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Coincident', point_ids_upper[0], 1, -1, 1))

        App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Block',spline_upper_id))


    #sys.exit()
    # pressure side (lower spline)
    object_id += 1
    spline_lower_id = object_id

    _finalbsp_poles = []
    _finalbsp_knots = []
    _finalbsp_mults = []

    spline_lower = Part.BSplineCurve()
    spline_lower.interpolate(vectors_lower, PeriodicFlag=False)
    spline_lower.increaseDegree(3)

    _finalbsp_poles.extend(spline_lower.getPoles())
    _finalbsp_knots.extend(spline_lower.getKnots())
    _finalbsp_mults.extend(spline_lower.getMultiplicities())

    App.getDocument('Unnamed').getObject(sketch).addGeometry(Part.BSplineCurve(_finalbsp_poles,_finalbsp_mults,_finalbsp_knots,False,3,None,False),False)

    del(_finalbsp_poles)
    del(_finalbsp_knots)
    del(_finalbsp_mults)

    if constrained:
        conList = []
        for i in range(len(x_lower)):
            conList.append(Sketcher.Constraint('InternalAlignment:Sketcher::BSplineKnotPoint',point_ids_lower[i],1,spline_lower_id,i))

            # we could avoid having to pass by conList, but then the code takes longer to execute; I don't know why
            #App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('InternalAlignment:Sketcher::BSplineKnotPoint',point_ids_lower[i],1,spline_lower_id,i))

        App.getDocument('Unnamed').getObject(sketch).addConstraint(conList)
        del conList

        # the first point of the lower spline is equivalent to the first point of the upper spline
        App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Coincident', point_ids_lower[0], 1, point_ids_upper[0], 1))

        App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Block',spline_lower_id))


    # The trailing edge will be blunt, but some meshing software are able to automatically change it for round

    #sys.exit()
    # trailing edge upper
    object_id += 1
    TE_upper_id = object_id
    #TE_upper_line = Part.LineSegment(vectors_upper[-1],chord_end_vector)
    TE_upper_line = Part.LineSegment(vectors_upper[-1],App.Vector(x_upper_chord[-1], y_upper_chord[-1] - 0.1, 0))
    App.getDocument('Unnamed').getObject(sketch).addGeometry(TE_upper_line,False)

    App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Coincident', TE_upper_id, 1, point_ids_upper[-1], 1))
    #App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Coincident', TE_upper_id, 2, chord_end_id, 1))
    App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Vertical', TE_upper_id))


    #sys.exit()
    # trailing edge lower
    object_id += 1
    TE_lower_id = object_id
    #TE_lower_line = Part.LineSegment(vectors_lower[-1],chord_end_vector)
    TE_lower_line = Part.LineSegment(vectors_lower[-1],App.Vector(x_lower_chord[-1], y_lower_chord[-1] + 0.1, 0))
    App.getDocument('Unnamed').getObject(sketch).addGeometry(TE_lower_line,False)

    App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Coincident', TE_lower_id, 1, point_ids_lower[-1], 1))
    #App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Coincident', TE_lower_id, 2, chord_end_id, 1))
    App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Vertical', TE_lower_id))

    # join upper and lower parts of trailing edge
    App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Coincident', TE_upper_id, 2, TE_lower_id, 2))
    App.getDocument('Unnamed').getObject(sketch).addConstraint(Sketcher.Constraint('Equal',TE_upper_id,TE_lower_id))


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
