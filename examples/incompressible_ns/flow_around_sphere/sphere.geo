SetFactory("OpenCASCADE");

lc = 2; 
box_x = 22.0;
box_y = 10.0;
box_z = 10.0;
sphere_r = 0.5;
sphere_center_x = 6.0;
sphere_center_y = 5.0;
sphere_center_z = 5.0;

Box(1) = {0, 0, 0, box_x, box_y, box_z};

Sphere(2) = {sphere_center_x, sphere_center_y, sphere_center_z, sphere_r};

BooleanDifference{ Volume{1}; Delete; }{ Volume{2}; Delete; }

Physical Surface("Outlet") = {13};
Physical Surface("Back") = {12};
Physical Surface("Top") = {11};
Physical Surface("Front") = {10};
Physical Surface("Bottom") = {9};
Physical Surface("Inlet") = {8};

Physical Surface("Sphere") = {7};

MeshSize{ PointsOf{ Surface{:}; } } = lc;
MeshSize{ PointsOf{ Surface{7}; } } = lc/10;

Physical Volume("FluidVolume") = {1};
