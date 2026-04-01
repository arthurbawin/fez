//------------------------------------------------------------
// Domaine 2D rectangulaire avec cylindre
// Gmsh .geo — kernel OpenCASCADE
//------------------------------------------------------------
SetFactory("OpenCASCADE");

//==============================
// Paramètres généraux
//==============================
H  = 40;     
L  = 60;        
r  = 0.5;      

cx = 20;        
cy = H/2; 

lc_cyl = r/40;
lc_far = H/20;
lc_end = 2*r;
lc_near = r/5;

//==============================
// Géométrie principale 
//==============================
Point(1) = {0,  0, 0, lc_far};
Point(2) = {L,  0, 0, lc_far};
Point(3) = {L, cy-20*r,0,lc_end};
Point(4) = {L, cy+20*r,0,lc_end};
Point(5) = {L,  H, 0, lc_far};
Point(6) = {0,  H, 0, lc_far};

Line(1) = {1, 2};  
Line(2) = {2, 3};   
Line(3) = {3, 4};    
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};    

Curve Loop(1) = {1, 2, 3, 4, 5, 6}; 

Circle(7) = {cx, cy, 0, r, 0, 2*Pi};  
Curve Loop(2) = {7};                  

Plane Surface(1) = {1, 2};

//==============================
// Contrôle des tailles de mailles
//==============================


// Points d'accroche
Point(1210) = {cx, cy-7*r, 0, lc_near};
Point(1310) = {cx, cy+7*r, 0, lc_near};

Line(2110) = {1210, 3};
Line(2210) = {1310, 4};
Line{2110, 2210} In Surface{1};

// Cercle (arc gauche)
pc1_c = newp; Point(pc1_c) = {cx,       cy,       0, lc_near};

c1510_1 = newc; Circle(c1510_1) = {1210, pc1_c, 1310};

Curve{c1510_1} In Surface{1};

MeshSize{Point{pc1_c}} = lc_near;

MeshSize { PointsOf{ Curve{7}; } } = lc_cyl;

MeshSize { PointsOf{Curve{c1510_1}; } } = lc_near;

//==============================
// Raffinement supplémentaire du wake
//==============================

// Points internes dans le sillage
//pW0 = newp; Point(pW0) = {cx + 1.2*r, cy, 0, r/10};
//pW1 = newp; Point(pW1) = {cx + 20.0*r, cy, 0, r/8};
//pW2 = newp; Point(pW2) = {cx + 40.0*r, cy, 0, r/5};
//
//// Ligne centrale du wake
//lW1 = newl; Line(lW1) = {pW0, pW1};
//lW2 = newl; Line(lW2) = {pW1, pW2};
//
//Line{lW1, lW2} In Surface{1};
//
//// Tailles locales sur les points du wake
//MeshSize{Point{pW0}} = r/10;
//MeshSize{Point{pW1}} = r/8;
//MeshSize{Point{pW2}} = r/5;
//
//// Raffinement autour de l'arc aval + ligne centrale du wake
//Field[1] = Distance;
//Field[1].CurvesList = {c1510_1, lW1, lW2};
//Field[1].Sampling = 150;
//
//Field[2] = Threshold;
//Field[2].InField = 1;
//Field[2].SizeMin = r/10;
//Field[2].SizeMax = lc_far;
//Field[2].DistMin = 0.5*r;
//Field[2].DistMax = 2.5*r;
//
//Background Field = 2;


//==============================
// Options maillage
//==============================
Mesh.CharacteristicLengthFromPoints     = 1;
Mesh.CharacteristicLengthFromCurvature  = 1;
Mesh.CharacteristicLengthExtendFromBoundary = 1;
Mesh.CharacteristicLengthMin = lc_cyl;
Mesh.CharacteristicLengthMax = lc_far;

//==============================
// Groupes physiques
//==============================
Physical Curve("Outlet")        = {2,3,4};
Physical Curve("Inlet")         = {6};
Physical Curve("NoFlux")        = {1, 5};
Physical Curve("InnerBoundary") = {7};
Physical Surface("Domain")      = {1};