SetFactory("Built-in");

// ============================================================
// Parameters
// ============================================================
L  = 5.0;
xc = 2.5;
yc = 2.5;

R   = 0.15;      // bubble radius
eps = 3e-3;   // interface thickness

h_int = 0.0030;   // near-interface target size 
h_in  = 0.10;   // inside bubble
h_far = 0.10;    // far field

dmin = 5.0*eps;     // band around interface
dmax = 100.0*eps;

r_in = R - 4.0*eps; // refined interior radius
If (r_in < 0)
  r_in = 0;
EndIf

// ============================================================
// Outer square
// ============================================================
Point(1) = {0, 0, 0, h_far};
Point(2) = {L, 0, 0, h_far};
Point(3) = {L, L, 0, h_far};
Point(4) = {0, L, 0, h_far};

Line(1) = {1,2}; // y=0  -> y_min
Line(2) = {2,3}; // x=L  -> x_max
Line(3) = {3,4}; // y=L  -> y_max
Line(4) = {4,1}; // x=0  -> x_min

Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

// ============================================================
// Embedded circle (NOT a hole)
// ============================================================
Point(10) = {xc, yc, 0, h_int};  // center
Point(11) = {xc+R, yc,   0, h_int};
Point(12) = {xc,   yc+R, 0, h_int};
Point(13) = {xc-R, yc,   0, h_int};
Point(14) = {xc,   yc-R, 0, h_int};

Circle(11) = {11,10,12};
Circle(12) = {12,10,13};
Circle(13) = {13,10,14};
Circle(14) = {14,10,11};

// Embed into surface so mesh conforms
Curve{11,12,13,14} In Surface{1};

// ============================================================
// Background fields
// ============================================================

// distance to circle => refine band near interface
Field[1] = Distance;
Field[1].CurvesList = {11,12,13,14};
Field[1].Sampling = 300;

Field[2] = Threshold;
Field[2].InField = 1;
Field[2].SizeMin = h_int;
Field[2].SizeMax = h_far;
Field[2].DistMin = dmin;
Field[2].DistMax = dmax;

// distance to center => refine inside bubble
Field[10] = Distance;
Field[10].PointsList = {10};

Field[11] = Threshold;
Field[11].InField = 10;
Field[11].SizeMin = h_in;
Field[11].SizeMax = h_far;
Field[11].DistMin = 0.0;
Field[11].DistMax = r_in;

// combine
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

// domain
Physical Surface("domain", 1) = {1};

// boundaries
Physical Curve("y_min", 2) = {1};
Physical Curve("x_max", 3) = {2};
Physical Curve("y_max", 4) = {3};
Physical Curve("x_min", 5) = {4};
