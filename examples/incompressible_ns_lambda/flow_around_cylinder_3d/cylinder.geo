SetFactory("OpenCASCADE");

X0 = 0.;
Y0 = 0.;
Z0 = 0.;

Lx = 32;
Ly = 16;
Lz = 8;

r = 0.5;
cx = 8;
cy = 8;
cz = 8;
lc_domain = Ly/5;
lc_cyl    = lc_domain/20;

// Box corners
Point(1) = {X0 +  0, Y0 +  0, Z0 +  0, lc_domain};
Point(2) = {X0 + Lx, Y0 +  0, Z0 +  0, lc_domain};
Point(3) = {X0 + Lx, Y0 + Ly, Z0 +  0, lc_domain};
Point(4) = {X0 +  0, Y0 + Ly, Z0 +  0, lc_domain};
Point(5) = {X0 +  0, Y0 +  0, Z0 + Lz, lc_domain};
Point(6) = {X0 + Lx, Y0 +  0, Z0 + Lz, lc_domain};
Point(7) = {X0 + Lx, Y0 + Ly, Z0 + Lz, lc_domain};
Point(8) = {X0 +  0, Y0 + Ly, Z0 + Lz, lc_domain};

// Box volume
Box(1) = {0, 0, 0, Lx, Ly, Lz};

// Cylinder (hole)
Cylinder(2) = {cx, cy, 0, 0, 0, Lz, r, 2*Pi};
MeshSize { PointsOf{ Volume{2}; }} = lc_cyl;

// Boolean subtraction: box minus cylinder
BooleanDifference(3) = { Volume{1}; Delete; } 
                       { Volume{2}; Delete; };

tol = 1e-3;
Physical Surface("Back") = Surface In BoundingBox{ 0 - tol,  0 - tol,  0 - tol, Lx + tol, Ly + tol, 0.0 + tol};
Physical Surface("Front")    = Surface In BoundingBox{ 0 - tol,  0 - tol, Lz - tol, Lx + tol, Ly + tol, Lz + tol};

Physical Surface("Bottom")  = Surface In BoundingBox{ 0 - tol,  0 - tol,  0 - tol, Lx + tol,  0 + tol, Lz + tol};
Physical Surface("Top") = Surface In BoundingBox{ 0 - tol, Ly - tol,  0 - tol, Lx + tol, Ly + tol, Lz + tol};

Physical Surface("Inlet")   = Surface In BoundingBox{ 0 - tol,  0 - tol,  0 - tol,  0 + tol, Ly + tol, Lz + tol};
Physical Surface("Outlet")  = Surface In BoundingBox{Lx - tol,  0 - tol,  0 - tol, Lx + tol, Ly + tol, Lz + tol};

// Cylinder skin (side surface)
Physical Surface("Cylinder") = Surface In BoundingBox{cx - r - tol, cy - r - tol, 0 - tol, cx + r + tol, cy + r + tol, Lz + tol};

// Physical volume: the meshed region (box minus cylinder)
Physical Volume("Domain") = Volume{:};