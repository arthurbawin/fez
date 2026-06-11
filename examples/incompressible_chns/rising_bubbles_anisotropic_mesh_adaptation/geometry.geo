
// Size of the square
Lx = 1;
Ly = 2;
// Characteristic length of the mesh elements
lc = 0.15;

// Geometric entities (building blocks) : points and lines
Point(1) = {0, 0, 0, lc};
Point(2) = { Lx, 0, 0, lc};
Point(3) = { Lx,  Ly, 0, lc};
Point(4) = {0,  Ly, 0, lc};

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


Transfinite Curve{1} = 33; // h = 1/32
Transfinite Curve{3} = 33;

Transfinite Curve{2} = 65; 
Transfinite Curve{4} = 65;

//Transfinite Curve{1} = 65; // h = 1/64
//Transfinite Curve{3} = 65;

//Transfinite Curve{2} = 129; 
//Transfinite Curve{4} = 129;

//Transfinite Curve{1} = 129; // h = 1/128
//Transfinite Curve{3} = 129;

//Transfinite Curve{2} = 257; 
//Transfinite Curve{4} = 257;

//Transfinite Curve{1} = 257; // h = 1/256
//Transfinite Curve{3} = 257;

//Transfinite Curve{2} = 513; 
//Transfinite Curve{4} = 513;

Transfinite Surface{1} = {1, 2, 3, 4} Alternate;