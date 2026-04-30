SetFactory("Built-in");

// ============================================================
// 0 GLOBAL SCALE
// ============================================================
scale = 1.0;

// ============================================================
// 1 DOMAIN GEOMETRY
// ============================================================

// Main tank dimensions
domain_length = scale * 11.985;
domain_height = scale * 1.850;

// Right ramp geometry
ramp_x_start = scale * 8.270;
ramp_x_mid   = scale * 8.669;
ramp_y_mid   = scale * 0.021;
ramp_y_top   = scale * 0.350;

// ============================================================
// 2 WAVEMAKER GEOMETRY
// ============================================================

// Capsule-shaped wavemaker attached to bottom wall
wavemaker_x_center = scale * 1.66;
wavemaker_width    = scale * 0.400;
wavemaker_height   = scale * 1.200;
wavemaker_radius   = wavemaker_width / 2.0;

// ============================================================
// 3 MESH SIZE PARAMETERS
// ============================================================

// Global mesh sizes
mesh_size_far        = scale * 0.200;
mesh_size_interface  = scale * 0.008;
mesh_size_wavemaker  = scale * 0.02;
mesh_size_right_wall = scale * 0.02;

// Absolute lower bound
mesh_size_min = 1e-4;

// ============================================================
// 4 INTERFACE REFINEMENT PARAMETERS
// ============================================================

interface_y = scale * 0.400;

// Asymmetric half-widths of the refined band
interface_band_below = scale * 0.2;
interface_band_above = scale * 0.4;

// Smoothness of the vertical transition of the band thickness
interface_asymmetry_width = scale * 0.05;

// Smoothness of the mesh-size transition outside the band
interface_transition_width = scale * 0.3;

// Smooth transition in x to reduce refinement behind wavemaker
interface_x_transition_width = scale * 0.01;

// ============================================================
// 5 WAVEMAKER REFINEMENT PARAMETERS
// ============================================================

wavemaker_refine_dist_min = scale * 0.01;
wavemaker_refine_dist_max = scale * 0.65;

// ============================================================
// 6 RIGHT WALL REFINEMENT PARAMETERS
// ============================================================

right_wall_refine_dist_min = 0.03;
right_wall_refine_dist_max = scale * 0.10;

// ============================================================
// 7 GEOMETRY POINTS
// ============================================================

// ---- Outer boundary points
Point(1)  = {0,             0,               0, mesh_size_far}; // bottom-left
Point(2)  = {0,             domain_height,   0, mesh_size_far}; // top-left
Point(3)  = {domain_length, domain_height,   0, mesh_size_far}; // top-right
Point(4)  = {domain_length, ramp_y_top,      0, mesh_size_far}; // right wall lower point
Point(12) = {ramp_x_start,  0,               0, mesh_size_far}; // ramp start
Point(13) = {ramp_x_mid,    ramp_y_mid,      0, mesh_size_far}; // ramp middle point

// ---- Wavemaker points
Point(5)  = {wavemaker_x_center - wavemaker_width/2.0, 0.0,                                  0, mesh_size_wavemaker};
Point(6)  = {wavemaker_x_center + wavemaker_width/2.0, 0.0,                                  0, mesh_size_wavemaker};
Point(7)  = {wavemaker_x_center - wavemaker_width/2.0, wavemaker_height - wavemaker_radius, 0, mesh_size_wavemaker};
Point(8)  = {wavemaker_x_center + wavemaker_width/2.0, wavemaker_height - wavemaker_radius, 0, mesh_size_wavemaker};
Point(9)  = {wavemaker_x_center,                     wavemaker_height - wavemaker_radius,     0, mesh_size_wavemaker};
Point(10) = {wavemaker_x_center,                     wavemaker_height,                        0, mesh_size_wavemaker};
Point(11) = {wavemaker_x_center + wavemaker_width/2.0, interface_y - 0.200, 0, mesh_size_wavemaker};

// ============================================================
// 8 BOUNDARY CURVES
// ============================================================

// Bottom wall left of wavemaker
Line(121) = {1, 5};

// Wavemaker boundary
Line(102)   = {5, 7};
Circle(103) = {7, 9, 10};
Circle(104) = {10, 9, 8};
Line(105)   = {8, 11}; 
Line(1051)  = {11, 6}; 

// Bottom wall between wavemaker and ramp
Line(106) = {6, 12};

// Ramp
Line(110) = {12, 13};
Line(111) = {13, 4};

// Right wall
Line(107) = {4, 3};

// Top wall
Line(108) = {3, 2};

// Left wall
Line(122) = {2, 1};

// ============================================================
// 9 FLUID DOMAIN SURFACE
// ============================================================

Line Loop(200) = {121, 102, 103, 104, 105, 1051, 106, 110, 111, 107, 108, 122};
Plane Surface(1) = {200};

// ============================================================
// 10 MESH SIZE FIELDS
// ============================================================

// ------------------------------------------------------------
// 10.1 Interface refinement field
// Smooth refinement around y = interface_y, only mainly to the
// right of the wavemaker
// ------------------------------------------------------------
Field[2] = MathEval;
Field[2].F = Sprintf(
  "%g - (%g-%g)*0.25*(1-tanh((abs(y-%g) - ((%g+%g)/2 + (%g-%g)/2 * tanh((y-%g)/%g)))/%g))*(1+tanh((x-%g)/%g))",
  mesh_size_far,
  mesh_size_far,
  mesh_size_interface,
  interface_y,
  interface_band_above,
  interface_band_below,
  interface_band_above,
  interface_band_below,
  interface_y,
  interface_asymmetry_width,
  interface_transition_width,
  wavemaker_x_center,
  interface_x_transition_width
);

// ------------------------------------------------------------
// 10.2 Distance to wavemaker
// ------------------------------------------------------------
Field[3] = Distance;
Field[3].CurvesList = {103, 104, 105};
Field[3].Sampling   = 200;

// ------------------------------------------------------------
// 10.3 Threshold refinement near wavemaker
// ------------------------------------------------------------
Field[4] = Threshold;
Field[4].InField      = 3;
Field[4].SizeMin      = mesh_size_wavemaker;
Field[4].SizeMax      = mesh_size_far;
Field[4].DistMin      = wavemaker_refine_dist_min;
Field[4].DistMax      = wavemaker_refine_dist_max;
Field[4].StopAtDistMax = 1;

// ------------------------------------------------------------
// 10.4 Distance to right wall
// ------------------------------------------------------------
Field[5] = Distance;
Field[5].CurvesList = {107};
Field[5].Sampling   = 200;

// ------------------------------------------------------------
// 10.5 Threshold refinement near right wall
// ------------------------------------------------------------
Field[6] = Threshold;
Field[6].InField      = 5;
Field[6].SizeMin      = mesh_size_right_wall;
Field[6].SizeMax      = mesh_size_far;
Field[6].DistMin      = right_wall_refine_dist_min;
Field[6].DistMax      = right_wall_refine_dist_max;
Field[6].StopAtDistMax = 1;

// ------------------------------------------------------------
// 10.6 Combine all refinement fields
// Take the finest size requested by any local criterion
// ------------------------------------------------------------
Field[10] = Min;
Field[10].FieldsList = {2, 4, 6};

// ------------------------------------------------------------
// 10.7 Safety floor to enforce strictly positive mesh size
// ------------------------------------------------------------
Field[90] = MathEval;
Field[90].F = Sprintf("%g", mesh_size_min);

Field[91] = Max;
Field[91].FieldsList = {10, 90};

// Final background field
Background Field = 91;

// ============================================================
// 11 GLOBAL MESH CONTROL
// ============================================================

// Enforce that only the background field controls the mesh size
Mesh.MeshSizeFromPoints = 0;
Mesh.MeshSizeFromCurvature = 0;
Mesh.MeshSizeExtendFromBoundary = 0;

// Global safety bounds
Mesh.MeshSizeMin = mesh_size_min;
Mesh.MeshSizeMax = mesh_size_far;

// 2D meshing / optimization
Mesh.Algorithm = 6;       // Frontal-Delaunay
Mesh.Optimize = 1;
Mesh.OptimizeNetgen = 1;

// ============================================================
// 12 PHYSICAL GROUPS
// ============================================================

Physical Line("x_min", 1)            = {122};
Physical Line("x_max", 2)            = {107};
Physical Line("y_min", 3)            = {121, 106};
Physical Line("y_max", 4)            = {108};
Physical Line("batteur_v_0", 5)      = {102, 103, 104};
Physical Line("batteur_v_libre", 6)  = {105, 1051};
Physical Line("ramp", 7)             = {110, 111};

Physical Surface("fluid_domain", 101) = {1};
