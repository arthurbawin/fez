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
mesh_size_interface  = scale * 0.04;
mesh_size_wavemaker  = scale * 0.01;
mesh_size_right_wall = scale * 0.02;
mesh_interface_width = scale * 0.1;
mesh_size_batteur    = scale * 0.01;

// Absolute lower bound
mesh_size_min = 1e-4;

interface_y = scale * 0.400;


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
// 10 REFINEMENT
// ============================================================



// Créer un champ Distance par rapport à la ligne 105 (batteur)
Field[1] = Distance;
Field[1].CurvesList = {105};  // Lien vers la ligne 105
Field[1].Sampling = 200;      // Nombre d'échantillons

// Créer un champ de raffinement basé sur la distance
Field[2] = Threshold;
Field[2].InField = 1;                // Utilisation du champ Distance
Field[2].SizeMin = mesh_size_wavemaker;  // Taille de maillage pour la zone proche du batteur
Field[2].SizeMax = mesh_size_interface;       // Taille de maillage pour la zone loin du batteur
Field[2].DistMin = 0;                  // Distance minimale où le raffinement commence
Field[2].DistMax = scale * 0.08;      // Zone de raffinement autour de la ligne 105 (ajuster selon tes besoins)
Field[2].StopAtDistMax = 1;           // Arrêter le raffinement après une certaine distance

Field[3] = MathEval;
Field[3].F = Sprintf(
    "%g + 0.5 * (%g - %g) * (1 + tanh((x - 1.66)/%g))",
    mesh_size_far,        // taille du maillage dans la zone non raffinée
    mesh_size_interface,  // taille du maillage dans la zone raffinée
    mesh_size_far,        // taille du maillage dans la zone non raffinée
    mesh_interface_width  // largeur de la transition douce
);

Field[4] = Min;
Field[4].FieldsList = {2, 3};

Background Field = 4;

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
