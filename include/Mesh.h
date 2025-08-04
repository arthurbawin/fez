#ifndef MESH_H
#define MESH_H

#include <deal.II/grid/tria.h>

using namespace dealii;

template <int dim>
Triangulation<dim> read_mesh(const std::string &meshFile, const MPI_Comm &comm);

#endif