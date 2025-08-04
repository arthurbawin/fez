
p1 = -0.5;
p2 = 1;
h1 = -0.5;
h2 = 1.5;
lc = 0.1;

Point(1) = {p1, h1, 0, lc};
Point(2) = {p2, h1, 0, lc};
Point(3) = {p2, h2, 0, lc};
Point(4) = {p1, h2, 0, lc};

Line(1) = {1, 2};
Line(2) = {2, 3};
Line(3) = {3, 4};
Line(4) = {4, 1};

Curve Loop(1) = {1, 2, 3, 4};

Plane Surface(1) = {1};

Physical Point("PointPression") = {1};
Physical Curve("Bord") = {1,2,3,4};
Physical Surface("Domaine") = {1};

//Transfinite Curve{1,3} = 7;
//Transfinite Curve{2,4} = 9;
//Transfinite Surface{1} Left;

Mesh 2;

// We now define several constants to fine-tune how the mesh will be partitioned
DefineConstant[
  partitioner = {0, Choices{0="Metis", 1="SimplePartition"},
    Name "Parameters/0Mesh partitioner"}
  N = {3, Min 1, Max 256, Step 1,
    Name "Parameters/1Number of partitions"}
  topology = {0, Choices{0, 1},
    Name "Parameters/2Create partition topology (BRep)?"}
  ghosts = {0, Choices{0, 1},
    Name "Parameters/3Create ghost cells?"}
  physicals = {0, Choices{0, 1},
    Name "Parameters/3Create new physical groups?"}
  write = {1, Choices {0, 1},
    Name "Parameters/3Write file to disk?"}
  split = {0, Choices {0, 1},
    Name "Parameters/4Write one file per partition?"}
];

// Should we create the boundary representation of the partition entities?
Mesh.PartitionCreateTopology = topology;

// Should we create ghost cells?
Mesh.PartitionCreateGhostCells = ghosts;

// Should we automatically create new physical groups on the partition entities?
Mesh.PartitionCreatePhysicals = physicals;

// Should we keep backward compatibility with pre-Gmsh 4, e.g. to save the mesh
// in MSH2 format?
Mesh.PartitionOldStyleMsh2 = 0;

// Should we save one mesh file per partition?
Mesh.PartitionSplitMeshFiles = split;

If (partitioner == 0)
  // Use Metis to create N partitions
  PartitionMesh N;
  // Several options can be set to control Metis: `Mesh.MetisAlgorithm' (1:
  // Recursive, 2: K-way), `Mesh.MetisObjective' (1: min. edge-cut, 2:
  // min. communication volume), `Mesh.PartitionTriWeight' (weight of
  // triangles), `Mesh.PartitionQuadWeight' (weight of quads), ...
Else
  // Use the `SimplePartition' plugin to create chessboard-like partitions
  Plugin(SimplePartition).NumSlicesX = N;
  Plugin(SimplePartition).NumSlicesY = 1;
  Plugin(SimplePartition).NumSlicesZ = 1;
  Plugin(SimplePartition).Run;
EndIf

// Save mesh file (or files, if `Mesh.PartitionSplitMeshFiles' is set):
If(write)
  Save "foo.msh";
EndIf