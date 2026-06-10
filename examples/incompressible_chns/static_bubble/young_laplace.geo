SetFactory("Built-in");

// ============================================================
// Parameters
// ============================================================
L  = 0.5;
xc = 0;
yc = 0;

R   = 0.4;    // bubble radius
eps = 3e-3;    // interface thickness

h_int = 0.0080; // near-interface target size 
h_in  = 0.20;   // inside bubble
h_far = 0.80;   // far field

dmin = 0.2;     // band around interface
dmax = 3;

r_in = R - 4.0*eps; // refined interior radius
If (r_in < 0)
  r_in = 0;
EndIf

// ============================================================
// Outer square, with x_min and y_min split at quarter-circle endpoints
// ============================================================

// Corner points
Point(1) = {0, 0, 0, h_far};
Point(2) = {L, 0, 0, h_far};
Point(3) = {L, L, 0, h_far};
Point(4) = {0, L, 0, h_far};

// Quarter-circle center and endpoints
Point(10) = {xc, yc, 0, h_int};      // center
Point(11) = {xc+R, yc,   0, h_int};  // point on y_min
Point(12) = {xc,   yc+R, 0, h_int};  // point on x_min

// Boundary curves
Line(1) = {1,11};   // y=0, from x=0 to x=R
Line(5) = {11,2};   // y=0, from x=R to x=L

Line(2) = {2,3};    // x=L  -> x_max
Line(3) = {3,4};    // y=L  -> y_max

Line(6) = {4,12};   // x=0, from y=L to y=R
Line(4) = {12,1};   // x=0, from y=R to y=0

Curve Loop(1) = {1,5,2,3,6,4};
Plane Surface(1) = {1};

// ============================================================
// Embedded quarter circle (NOT a hole)
// ============================================================

// Quarter circle in the first quadrant: from (R,0) to (0,R)
Circle(11) = {11,10,12};

// Embed into surface so mesh conforms
Curve{11} In Surface{1};

// ============================================================
// Background fields
// ============================================================

// Distance to quarter-circle => refine band near interface
Field[1] = Distance;
Field[1].CurvesList = {11};
Field[1].Sampling = 300;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_int;
Field[2].SizeMax = h_far;
Field[2].DistMin = dmin;
Field[2].DistMax = dmax;

// Distance to center => refine inside quarter bubble
Field[10] = Distance;
Field[10].PointsList = {10};

Field[11] = Threshold;
Field[11].InField = 10;
Field[11].SizeMin = h_in;
Field[11].SizeMax = h_far;
Field[11].DistMin = 0.0;
Field[11].DistMax = r_in;

// Combine
Field[12] = Min;
Field[12].FieldsList = {2, 11};
Background Field = 12;

// ============================================================
// Mesh options
// ============================================================
Mesh.Algorithm = 6; // Frontal-Delaunay
Mesh.ElementOrder = 1;
Mesh.CharacteristicLengthExtendFromBoundary = 0;

// Force msh format that preserves names reliably
Mesh.MshFileVersion = 4.1;

// ============================================================
// Physical groups: NAMED + FIXED TAGS to match .prm
// ============================================================

// Domain
Physical Surface("domain", 1) = {1};

// Boundaries
Physical Curve("y_min", 2) = {1,5};
Physical Curve("x_max", 3) = {2};
Physical Curve("y_max", 4) = {3};
Physical Curve("x_min", 5) = {6,4};