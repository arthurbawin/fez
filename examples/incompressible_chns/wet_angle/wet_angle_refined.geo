// =============================================================
// Domaine 1 x 0.3
// Raffinement autour d'un demi-cercle
// Centre : (0.5, 0.0)
// Rayon  : 0.25
// =============================================================

lc = 2.5e-2;

// -------------------------------------------------------------
// Domaine
// -------------------------------------------------------------

Point(1) = {0.0, 0.0, 0.0, lc};
Point(2) = {1.0, 0.0, 0.0, lc};
Point(3) = {1.0, 0.3, 0.0, lc};
Point(4) = {0.0, 0.3, 0.0, lc};

Line(1) = {1, 2}; // y_min
Line(2) = {2, 3}; // x_max
Line(3) = {3, 4}; // y_max
Line(4) = {4, 1}; // x_min

Curve Loop(1) = {1, 2, 3, 4};
Plane Surface(1) = {1};

// -------------------------------------------------------------
// Demi-cercle de référence pour le raffinement
// Centre (0.5,0), rayon 0.25
// -------------------------------------------------------------

Point(100) = {0.50, -0.15, 0.0, lc};
Point(101) = {0.65, -0.15, 0.0, lc};
Point(102) = {0.35, -0.15, 0.0, lc};

Circle(100) = {101,100,102};
Circle(101) = {102,100,101};

// -------------------------------------------------------------
// Champ de raffinement
// -------------------------------------------------------------

Field[1] = Distance;
Field[1].CurvesList = {100,101};
Field[1].Sampling = 1000;

Field[2] = Threshold;
Field[2].InField = 1;

// taille de maille sur l'interface
Field[2].SizeMin = lc/5;

// taille loin de l'interface
Field[2].SizeMax = lc;

// distance <= DistMin -> SizeMin
Field[2].DistMin = 0.20;

// transition vers SizeMax
Field[2].DistMax = 0.30;

Background Field = 2;

// -------------------------------------------------------------
// Paramètres de maillage
// -------------------------------------------------------------

Mesh.Algorithm = 6;

Mesh.CharacteristicLengthFromPoints = 0;
Mesh.CharacteristicLengthFromCurvature = 0;
Mesh.CharacteristicLengthExtendFromBoundary = 0;

// bornes globales cohérentes avec le champ
Mesh.CharacteristicLengthMin = lc/8;
Mesh.CharacteristicLengthMax = lc;

// -------------------------------------------------------------
// Physical groups
// -------------------------------------------------------------

Physical Curve("x_min", 1) = {4};
Physical Curve("x_max", 2) = {2};
Physical Curve("y_min", 3) = {1};
Physical Curve("y_max", 4) = {3};

Physical Surface("domain", 10) = {1};
