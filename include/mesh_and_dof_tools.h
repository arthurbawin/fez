#ifndef MESH_AND_DOF_TOOLS_H
#define MESH_AND_DOF_TOOLS_H

#include <deal.II/dofs/dof_handler.h>
#include <utilities.h>

using namespace dealii;

/**
 * Return the set of mesh vertices which belong to cell faces on the given
 * boundary "boundary_id". Mesh vertex indices are not unique across MPI ranks,
 * so this returns a set of Points instead of a more manageable global vertex
 * index.
 */
template <int dim>
std::set<Point<dim>, PointComparator<dim>>
get_mesh_vertices_on_boundary(const DoFHandler<dim>   &dof_handler,
                              const types::boundary_id boundary_id);

#endif