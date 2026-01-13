#ifndef MESH_H
#define MESH_H

#include <deal.II/distributed/tria_base.h>
#include <parameter_reader.h>

using namespace dealii;

/**
 * FIXME: This actually creates a mesh if the deal.II routines are used,
 * so it would make more sense to call it create_mesh.
 * 
 * Create the parallel triangulation, either from a Gmsh .msh4 mesh file,
 * or from the deal.II meshing routine set in the parameter file with the
 * given arguments.
 * 
 * A serial triangulaation is created on each rank, and is then partitioned.
 */
template <int dim, int spacedim = dim>
void read_mesh(
  parallel::DistributedTriangulationBase<dim, spacedim> &triangulation,
  ParameterReader<spacedim>                                  &param);

#endif