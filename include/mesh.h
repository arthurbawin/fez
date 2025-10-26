#ifndef MESH_H
#define MESH_H

#include <parameter_reader.h>

// #include <deal.II/grid/tria.h>
#include <deal.II/distributed/tria_base.h>

using namespace dealii;

// template <int dim>
// Triangulation<dim> read_mesh(const std::string &meshFile, const MPI_Comm
// &comm);

template <int dim, int spacedim = dim>
void read_mesh(
  parallel::DistributedTriangulationBase<dim, spacedim> &triangulation,
  ParameterReader<dim>                            &param);

void read_gmsh_physical_names(const std::string                   &meshFile,
                              std::map<unsigned int, std::string> &tag2name,
                              std::map<std::string, unsigned int> &name2tag);

#endif