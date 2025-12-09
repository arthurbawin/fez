#ifndef MESH_H
#define MESH_H

#include <deal.II/distributed/tria_base.h>
#include <parameter_reader.h>

using namespace dealii;

template <int dim, int spacedim = dim>
void read_mesh(
  parallel::DistributedTriangulationBase<dim, spacedim> &triangulation,
  ParameterReader<dim>                                  &param);

// To remove: used only by make_grid() in standalone examples
void read_gmsh_physical_names(const std::string                   &meshFile,
                              std::map<unsigned int, std::string> &tag2name,
                              std::map<std::string, unsigned int> &name2tag);

/**
 *
 */
template <int dim, int spacedim = dim>
void print_partition_gmsh(
  parallel::DistributedTriangulationBase<dim, spacedim> &triangulation,
  const ParameterReader<dim>                            &param);

#endif