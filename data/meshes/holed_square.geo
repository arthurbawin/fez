// Parameters
L = 1.0;       // outer square side
R = 0.15;       // half side of inner square
lc = 0.25;     // mesh size

// -----------------------------------------------------------------------------
// Outer square points
Point(1) = {0, 0, 0, lc};
Point(2) = {L, 0, 0, lc};
Point(3) = {L, L, 0, lc};
Point(4) = {0, L, 0, lc};

// -----------------------------------------------------------------------------
// Inner square points (centered at L/2,L/2)
xc = L/2;
yc = L/2;
Point(5) = {xc-R, yc-R, 0, lc};
Point(6) = {xc+R, yc-R, 0, lc};
Point(7) = {xc+R, yc+R, 0, lc};
Point(8) = {xc-R, yc+R, 0, lc};

// -----------------------------------------------------------------------------
// Outer square lines
Line(1) = {1,2};
Line(2) = {2,3};
Line(3) = {3,4};
Line(4) = {4,1};

// Inner square lines
Line(5) = {5,6};
Line(6) = {6,7};
Line(7) = {7,8};
Line(8) = {8,5};

// -----------------------------------------------------------------------------
// Curve loops and surfaces
Curve Loop(1) = {1,2,3,4};   // outer loop
Curve Loop(2) = {5,6,7,8};   // inner loop (hole)

Plane Surface(1) = {1,2};    // annular domain (square minus inner square)

// -----------------------------------------------------------------------------
// Physical groups
Physical Surface("Domain") = {1};
Physical Curve("SquareBoundary") = {1,2,3,4};
Physical Curve("InnerBoundary") = {5,6,7,8};
