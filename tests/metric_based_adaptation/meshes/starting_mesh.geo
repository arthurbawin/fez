
// Size of the square
Lx = 2;
Ly = 1;
// Characteristic length of the mesh elements
lc = 0.15;

// Geometric entities (building blocks) : points and lines
Point(1) = {-Lx, -Ly, 0, lc};
Point(2) = { Lx, -Ly, 0, lc};
Point(3) = { Lx,  Ly, 0, lc};
Point(4) = {-Lx,  Ly, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// A close loop to define a surface
Curve Loop(1) = {1, 2, 3, 4};

// The interior
Plane Surface(1) = {1};

// Physical entities : named domains
Physical Curve("x_min")    = {4};
Physical Curve("x_max")    = {2};
Physical Curve("y_min")    = {1};
Physical Curve("y_max")    = {3};
Physical Surface("Domain") = {1};
