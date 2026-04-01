SetFactory("OpenCASCADE");

// =========================
// Paramètres
// =========================
Lx = 1.0;   // largeur
Ly = 10.0;   // hauteur
Ny = 400.0;    // nombre de subdivisions selon y

// =========================
// Géométrie : 4 frontières complètes
// =========================
Point(1) = {0,  0,  0};
Point(2) = {Lx, 0,  0};
Point(3) = {Lx, Ly, 0};
Point(4) = {0,  Ly, 0};

Line(1) = {1, 2}; // bas
Line(2) = {2, 3}; // droite
Line(3) = {3, 4}; // haut
Line(4) = {4, 1}; // gauche

Curve Loop(1) = {1, 2, 3, 4};

Plane Surface(1) = {1};

// =========================
// Maillage structuré
// =========================
// 1 seule subdivision selon x  => 2 points
Transfinite Line {1, 3} = 2;

// Ny subdivisions selon y => Ny+1 points
Transfinite Line {2, 4} = Ny + 1;

// Surface transfinie avec diagonales orientées
Transfinite Surface {1} Right;

// On garde des triangles
Mesh.RecombineAll = 0;



// =========================
// Physical groups
// =========================
Physical Curve("bottom") = {1};
Physical Curve("right")  = {2};
Physical Curve("top")    = {3};
Physical Curve("left")   = {4};
Physical Surface("domain") = {1};