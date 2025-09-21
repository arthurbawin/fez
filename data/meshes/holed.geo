// Parameters
X0 = 10.;
Y0 = 10.;
Lx = 5.0;      
Ly = 1.0;
R = 0.15;       // circle radius
lc = 0.25;     // mesh size

// -----------------------------------------------------------------------------
// Points
// Square corners
Point(1) = {X0 + 0, Y0 + 0, 0, lc};
Point(2) = {X0 + Lx, Y0 + 0, 0, lc};
Point(3) = {X0 + Lx, Y0 + Ly, 0, lc};
Point(4) = {X0 + 0, Y0 + Ly, 0, lc};

// Circle center
Point(9) = {X0 + Lx/2, Y0 + Ly/2, 0, lc};

// Circle points (4 points)
Point(5) = {X0 + Lx/2+R, Y0 + Ly/2, 0, lc};
Point(6) = {X0 + Lx/2,   Y0 + Ly/2+R, 0, lc};
Point(7) = {X0 + Lx/2-R, Y0 + Ly/2, 0, lc};
Point(8) = {X0 + Lx/2,   Y0 + Ly/2-R, 0, lc};

// -----------------------------------------------------------------------------
// Lines
// Square edges
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

// Circle arcs (4 quarter arcs)
Circle(5) = {5, 9, 6};
Circle(6) = {6, 9, 7};
Circle(7) = {7, 9, 8};
Circle(8) = {8, 9, 5};

// -----------------------------------------------------------------------------
// Curve loops and surfaces
Curve Loop(1) = {1,2,3,4};  // outer square
Curve Loop(2) = {5,6,7,8};  // inner circle

Plane Surface(1) = {1,2};   // annular domain

// -----------------------------------------------------------------------------
// Physical groups
Physical Surface("Domain") = {1};
Physical Curve("OuterBoundary") = {1,2,3,4};
Physical Curve("InnerBoundary") = {5,6,7,8};


// -----------------------------------------------------------------------------
// Transfinite / structured mesh (optional)
//Transfinite Line {1,2,3,4} = 10;
//Transfinite Surface {1} = {1,2,3,4};
//Mesh 2;  // optional: recombine for quads
