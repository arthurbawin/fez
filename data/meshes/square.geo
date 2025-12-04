
// Size of the square
L = 1;
// Characteristic length of the mesh elements
lc = 0.1;

// Geometric entities (building blocks) : points and lines
Point(1) = {-L, -L, 0, lc};
Point(2) = { L, -L, 0, lc};
Point(3) = { L, L, 0, lc};
Point(4) = {-L, L, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

// A close loop to define a surface
Curve Loop(1) = {1, 2, 3, 4};

// The interior
Plane Surface(1) = {1};

// Physical entities : named domains
Physical Curve("Boundary")  = {1,2,3,4};
Physical Surface("Domain") = {1};
