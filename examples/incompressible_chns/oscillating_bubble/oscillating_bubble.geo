SetFactory("Built-in");

// ============================================================
// 2D bubble oscillations mesh
// Domain: [-5e-4, 5e-4]^2
// Bubble interface: embedded circle (NOT a hole)
// Refinement: symmetric band around interface (inside + outside)
// ============================================================

// --------------------
// Geometry
// --------------------
xmin = -5e-4;
xmax =  5e-4;
ymin = -5e-4;
ymax =  5e-4;

xc = 0.0;
yc = 0.0;
R  = 1.25e-4;

// --------------------
// Mesh sizes (tune)
// --------------------
h_far = 6.25e-5;  // coarse away from interface
h_int = 3.91e-6;  // fine within band

band  = 3.0e-5;   // fully fine if dist <= band
band2 = 2.0*band; // fully coarse if dist >= band2

// --------------------
// Outer square
// --------------------
Point(1) = {xmin, ymin, 0, h_far};
Point(2) = {xmax, ymin, 0, h_far};
Point(3) = {xmax, ymax, 0, h_far};
Point(4) = {xmin, ymax, 0, h_far};

Line(1) = {1,2}; // y_min
Line(2) = {2,3}; // x_max
Line(3) = {3,4}; // y_max
Line(4) = {4,1}; // x_min

Curve Loop(1) = {1,2,3,4};
Plane Surface(1) = {1};

// --------------------
// Embedded circle interface
// IMPORTANT: give these points a COARSE lc, not h_int,
// otherwise point-based lc can over-refine.
// --------------------
Point(10) = {xc, yc, 0, h_far};    // center (unused for sizing)
Point(11) = {xc+R, yc,   0, h_far};
Point(12) = {xc,   yc+R, 0, h_far};
Point(13) = {xc-R, yc,   0, h_far};
Point(14) = {xc,   yc-R, 0, h_far};

Circle(11) = {11,10,12};
Circle(12) = {12,10,13};
Circle(13) = {13,10,14};
Circle(14) = {14,10,11};

Curve{11,12,13,14} In Surface{1};

// --------------------
// Background size field: distance to interface
// dist = 0 on the circle, grows both inside and outside -> symmetric band
// --------------------
Field[1] = Distance;
Field[1].CurvesList = {11,12,13,14};
Field[1].Sampling = 800;

Field[2] = Threshold;
Field[2].IField  = 1;
Field[2].LcMin   = h_int;  // fine near interface
Field[2].LcMax   = h_far;  // coarse away
Field[2].DistMin = band;   // dist <= band  -> LcMin
Field[2].DistMax = band2;  // dist >= band2 -> LcMax

Background Field = 2;

// --------------------
// Mesh options
// Key: ignore point-based characteristic lengths to prevent
// unintended refinement inside the bubble.
// --------------------
Mesh.Algorithm = 6;
Mesh.ElementOrder = 1;
Mesh.MshFileVersion = 4.1;

Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;

Mesh.MeshSizeMin = h_int;
Mesh.MeshSizeMax = h_far;

// --------------------
// Physical groups (stable tags)
// --------------------
Physical Surface("domain", 1) = {1};

Physical Curve("y_min", 2) = {1};
Physical Curve("x_max", 3) = {2};
Physical Curve("y_max", 4) = {3};
Physical Curve("x_min", 5) = {4};

//Physical Curve("bubble_interface", 6) = {11,12,13,14};
