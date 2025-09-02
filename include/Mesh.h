#ifndef MESH_H
#define MESH_H

#include <deal.II/grid/tria.h>

using namespace dealii;

template <int dim>
Triangulation<dim> read_mesh(const std::string &meshFile, const MPI_Comm &comm);

void
read_gmsh_physical_names(const std::string &meshFile,
                         std::map<unsigned int, std::string> &tag2name,
                         std::map<std::string, unsigned int> &name2tag);

#endif