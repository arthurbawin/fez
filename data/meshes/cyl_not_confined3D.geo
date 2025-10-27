SetFactory("OpenCASCADE");

// ================= Paramètres =================
H = 7;         // hauteur (y)
L = 15;         // longueur (x)
l = 3;         // épaisseur (z)
r = 0.5;        // rayon du trou
cx = L/4; cy = H/2; // centre du trou
lc_domain = H/4;
eps = 1e-6;     // marge pour BoundingBox

// ============== Base 2D : rectangle percé ==============
Point(1) = {0, 0, 0, lc_domain};
Point(2) = {L, 0, 0, lc_domain};
Point(3) = {L, H, 0, lc_domain};
Point(4) = {0, H, 0, lc_domain};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// Trou circulaire
Circle(5) = {cx, cy, 0, r, 2*Pi, 0};

// Surface 2D avec trou
Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5};
Plane Surface(1) = {1, 2};

// ============== Extrusion 3D du domaine ==============
out[] = Extrude {0, 0, l} { Surface{1}; };
// out[0] = face z=l ; out[1] = volume ; puis faces latérales
vol_dom = out[1];

// ============== Sélections utiles (tags) ==============
cylSurf[]   = Surface In BoundingBox{cx - r - eps, cy - r - eps, -eps, cx + r + eps, cy + r + eps, l + eps};
inletSurf[] = Surface In BoundingBox{-eps, -eps, -eps, eps, H+eps, l+eps};
outletSurf[] = Surface In BoundingBox{L-eps, -eps, -eps, L+eps, H+eps, l+eps};
topSurf[]    = Surface In BoundingBox{-eps, H-eps, -eps, L+eps, H+eps, l+eps};
bottomSurf[] = Surface In BoundingBox{-eps, -eps, -eps, L+eps, +eps, l+eps};
frontSurf[]  = Surface In BoundingBox{-eps, -eps, l-eps, L+eps, H+eps, l+eps};
backSurf[]   = Surface In BoundingBox{-eps, -eps, -eps, L+eps, H+eps, eps};

// =====================================================================
// ============== Zones de raffinement 3D par champs Box ===============
// =====================================================================
// Box “large” (équivalent du grand trapèze), sur toute l’épaisseur
Field[10] = Box;
Field[10].VIn  = lc_domain/3;   // taille DANS la boîte
Field[10].VOut = lc_domain;     // taille HORS de la boîte
Field[10].XMin = cx - 3*r;
Field[10].XMax = L;
Field[10].YMin = cy - 4*r;
Field[10].YMax = cy + 4*r;
Field[10].ZMin = 0;
Field[10].ZMax = l;

// Box “proche” (raffinement plus fort autour du cylindre, aval)
Field[11] = Box;
Field[11].VIn  = lc_domain/6;
Field[11].VOut = lc_domain;
Field[11].XMin = cx - 1*r;
Field[11].XMax = L;
Field[11].YMin = cy - 1*r;
Field[11].YMax = cy + 1*r;
Field[11].ZMin = 0;
Field[11].ZMax = l;

// (Optionnel) petite boîte serrée autour du trou
// Décommente si tu veux encore affiner au contact
// Field[12] = Box;
// Field[12].VIn  = lc_domain/6;
// Field[12].VOut = lc_domain;
// Field[12].XMin = cx - 1.2*r;  Field[12].XMax = cx + 1.2*r;
// Field[12].YMin = cy - 1.2*r;  Field[12].YMax = cy + 1.2*r;
// Field[12].ZMin = 0;           Field[12].ZMax = l;

// Fusion : on prend la maille la plus fine
Field[20] = Min;
Field[20].FieldsList = {10, 11/*, 12*/};
Background Field = 20;

// ============== Options maillage ==============
Mesh.CharacteristicLengthExtendFromBoundary = 0;
Mesh.CharacteristicLengthFromPoints = 1;
Mesh.CharacteristicLengthMin = lc_domain/8;
Mesh.CharacteristicLengthMax = lc_domain;

// ============== Groupes physiques (3D) ==============
Physical Volume("Domain") = {vol_dom};
Physical Surface("Inlet")   = {inletSurf[]};
Physical Surface("Outlet")  = {outletSurf[]};
Physical Surface("Top")     = {topSurf[]};
Physical Surface("Bottom")  = {bottomSurf[]};
Physical Surface("Front")   = {frontSurf[]};
Physical Surface("Back")    = {backSurf[]};
Physical Surface("InnerBoundary") = {cylSurf[]};
