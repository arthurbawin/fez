SetFactory("OpenCASCADE");

X0 = 0.;
Y0 = 0.;
H = 16;
L = 32;
r = 0.5;
cx = 8;
cy = 8;
lc_domain = H/5;

Point(1) = {X0 + 0, Y0 + 0, 0, lc_domain};
Point(2) = {X0 + L, Y0 + 0, 0, lc_domain};
Point(3) = {X0 + L, Y0 + H, 0, lc_domain};
Point(4) = {X0 + 0, Y0 + H, 0, lc_domain};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Circle(5) = {X0 + cx, Y0 + cy, 0, r, 2*Pi, 0};

Characteristic Length{ PointsOf{ Curve{5}; } } = lc_domain/20;

Curve Loop(1) = {1, 2, 3, 4};
Curve Loop(2) = {5};

Plane Surface(1) = {1,2};

Physical Curve("Outlet") = {2};
Physical Curve("Inlet") = {4};
Physical Curve("NoFlux") = {1,3};
Physical Curve("InnerBoundary") = {5};
Physical Surface("Domain") = {1};