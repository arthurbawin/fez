//------------------------------------------------------------
// Canal 3D avec cylindre LISSE + n_slices (n_slices+1 cercles)
// Built-in kernel
// Raffinement autour du cylindre (avant + après) sur back/front
//------------------------------------------------------------
SetFactory("Built-in");

//==============================
// Paramètres
//==============================
H  = 160;
L  = 240;
l  = 2;
r  = 0.5;

cx = L/4;
cy = H/2;

lc_cyl = r/25;
lc_far = 30*r;
lc_end = 8*r;
lc_near = r/5;

n_cyl    = 2; // nb de points sur le cercle (plus = plus lisse)
n_slices = 4;  // nb de tranches en z -> (n_slices+1) cercles

// ===========================
// Geometrie général (z=0)
//============================
Point(1) = {0,  0, 0, lc_far};
Point(2) = {L,  0, 0, lc_far};
Point(3) = {L, cy-(H/6), 0, lc_end};
Point(4) = {L, cy+(H/6), 0, lc_end};
Point(5) = {L,  H, 0, lc_far};
Point(6) = {0,  H, 0, lc_far};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 5};
Line(5) = {5, 6};
Line(6) = {6, 1};

Curve Loop(1) = {1, 2, 3, 4, 5, 6};

// ===========================
// Geometrie général (z=l)
//============================
Point(101) = {0,  0, l, lc_far};
Point(102) = {L,  0, l, lc_far};
Point(103) = {L, cy-(H/6), l, lc_end};
Point(104) = {L, cy+(H/6), l, lc_end};
Point(105) = {L,  H, l, lc_far};
Point(106) = {0,  H, l, lc_far};

Line(21) = {101, 102};
Line(22) = {102, 103};
Line(23) = {103, 104};
Line(24) = {104, 105};
Line(25) = {105, 106};
Line(26) = {106, 101};

Curve Loop(3) = {21, 22, 23, 24, 25, 26};

//==============================
// Arêtes verticales canal
//==============================
Line(31) = {1, 101};
Line(32) = {2, 102};
Line(33) = {3, 103};
Line(34) = {4, 104};
Line(35) = {5, 105};
Line(36) = {6, 106};

//==============================
// Faces latérales canal
//==============================
Line Loop(4) = {1, 32, -21, -31};
Plane Surface(3) = {4};

Line Loop(5) = {2, 33, -22, -32};
Plane Surface(4) = {5};

Line Loop(6) = {3, 34, -23, -33};
Plane Surface(5) = {6};

Line Loop(7) = {4, 35, -24, -34};
Plane Surface(6) = {7};

Line Loop(8) = {5, 36, -25, -35};
Plane Surface(7) = {8};

Line Loop(9) = {6, 31, -26, -36};
Plane Surface(8) = {9};

//============================================================
// CYLINDRE avec n_slices : (n_slices+1) cercles en z
//============================================================

// --- Listes
cyl_pts[]      = {}; // taille (n_slices+1)*n_cyl
ctr_pts[]      = {}; // taille (n_slices+1)
cyl_arcs[]     = {}; // taille (n_slices+1)*n_cyl
cyl_vlines[]   = {}; // taille (n_slices)*n_cyl
cyl_surfaces[] = {}; // taille (n_slices)*n_cyl

// --- 1) Centres + points des cercles z_k
For k In {0:n_slices}
  z_k = l * k / n_slices;

  ctr = newp;
  Point(ctr) = {cx, cy, z_k, lc_cyl};
  ctr_pts[] += ctr;

  For i In {0:n_cyl-1}
    theta = 2*Pi*i/n_cyl;

    p = newp;
    Point(p) = {cx + r*Cos(theta), cy + r*Sin(theta), z_k, lc_cyl};
    // index global : k*n_cyl + i
    cyl_pts[] += p;
  EndFor
EndFor

// --- 2) Lignes verticales entre k et k+1
For k In {0:n_slices-1}
  For i In {0:n_cyl-1}
    p_bot = cyl_pts[k*n_cyl + i];
    p_top = cyl_pts[(k+1)*n_cyl + i];

    lv = newc;
    Line(lv) = {p_bot, p_top};
    cyl_vlines[] += lv;
  EndFor
EndFor

// --- 3) Arcs (cercles) à chaque z_k
For k In {0:n_slices}
  ctr = ctr_pts[k];

  For i In {0:n_cyl-1}
    next_i = (i+1) % n_cyl;

    p_i    = cyl_pts[k*n_cyl + i];
    p_next = cyl_pts[k*n_cyl + next_i];

    arc = newc;
    Circle(arc) = {p_i, ctr, p_next};
    cyl_arcs[] += arc;
  EndFor
EndFor

// --- 4) Surfaces latérales entre cercles successifs
For k In {0:n_slices-1}
  For i In {0:n_cyl-1}
    next_i = (i+1) % n_cyl;

    arc_bot = cyl_arcs[k*n_cyl + i];
    arc_top = cyl_arcs[(k+1)*n_cyl + i];

    v_i    = cyl_vlines[k*n_cyl + i];
    v_next = cyl_vlines[k*n_cyl + next_i];

    ll = newcl;
    Line Loop(ll) = {arc_bot, v_next, -arc_top, -v_i};

    s = news;
    Surface(s) = {ll};
    cyl_surfaces[] += s;
  EndFor
EndFor

// --- Line loops des cercles extrêmes (z=0 et z=l) pour percer back/front
ll_cyl0 = newcl;
For i In {0:n_cyl-1}
  tmp_arc = cyl_arcs[0*n_cyl + i];
  ll_cyl0_arcs[] += tmp_arc;
EndFor
Line Loop(ll_cyl0) = {ll_cyl0_arcs[]};

ll_cyl1 = newcl;
For i In {0:n_cyl-1}
  tmp_arc = cyl_arcs[n_slices*n_cyl + i];
  ll_cyl1_arcs[] += tmp_arc;
EndFor
Line Loop(ll_cyl1) = {ll_cyl1_arcs[]};

//============================================================
// Surfaces Back et Front (avec trou cylindre)
//============================================================
back  = news;
Plane Surface(back) = {1, ll_cyl0};

front = news;
Plane Surface(front) = {3, ll_cyl1};

//==============================
// RAFFINEMENT (Back z=0)
//==============================

// Points d'accroche
Point(1210) = {cx, cy-5*r, 0, lc_near};
Point(1310) = {cx, cy+5*r, 0, lc_near};

Line(2110) = {1210, 3};
Line(2210) = {1310, 4};
Line{2110, 2210} In Surface{back};

// Cercle 5*r (arc gauche)
pc1_c = newp; Point(pc1_c) = {cx,       cy,       0, lc_near};

c1510_1 = newc; Circle(c1510_1) = {1310, pc1_c, 1210};

Curve{c1510_1} In Surface{back};

MeshSize{Point{pc1_c}} = lc_near;

//==============================
// RAFFINEMENT (Front z=l)
//==============================

Point(1200) = {cx, cy-5*r, l, lc_near};
Point(1300) = {cx, cy+5*r, l, lc_near};

Line(2100) = {1300, 104};
Line(2200) = {1200, 103};
Line{2100, 2200} In Surface{front};

// Cercle 5*r (arc gauche)
pf1_c = newp; Point(pf1_c) = {cx,       cy,       l, lc_near};

c1500_1 = newc; Circle(c1500_1) = {1300, pf1_c, 1200};

Curve{c1500_1} In Surface{front};

MeshSize{Point{pf1_c}} = lc_near;

//==============================
// Volume (surface loop canal + surface loop cylindre)
//==============================
Surface Loop(100) = {3, 4, 5, 6, 7, 8, back, front};
Surface Loop(101) = {cyl_surfaces[]};

Volume(1) = {100, 101};

//==============================
// Physiques
//==============================
Physical Volume("Domain") = {1};

Physical Surface("Inlet")         = {8};
Physical Surface("Outlet")        = {4, 5, 6};
Physical Surface("Top")           = {7};
Physical Surface("Bottom")        = {3};
Physical Surface("Front")         = {front};
Physical Surface("Back")          = {back};
Physical Surface("InnerBoundary") = {cyl_surfaces[]};

// option : taille au point 4 (comme ton ancien code)
MeshSize { Point{4} } = lc_end;
MeshSize { Point{1}} = lc_far;
MeshSize { Point{2}} = lc_far;
MeshSize { Point{5}} = lc_far;
MeshSize { Point{6}} = lc_far;
MeshSize { Point{101}} = lc_far;
MeshSize { Point{102}} = lc_far;
MeshSize { Point{105}} = lc_far;
MeshSize { Point{106}} = lc_far;
