#ifndef MESH_AND_DOF_TOOLS_H
#define MESH_AND_DOF_TOOLS_H

#include <deal.II/dofs/dof_handler.h>
#include <utilities.h>

using namespace dealii;

/**
 * Fill @p owned_vertices with a flag stating if the i-th local mesh vertex
 * is owned or not on the partition @p subdomain_id.
 *
 * This function exists because
 * GridTools::get_locally_owned_vertices(triangulation) provides mesh vertices
 * with multiple owners at the boundary of a partition, where we would want a
 * single owner. This function leaves out vertices touching ONLY ghost-cells,
 * which would be marked as owned with GridTools::get_locally_owned_vertices.
 */
template <int dim>
void get_owned_mesh_vertices(const Triangulation<dim> &triangulation,
                             const types::subdomain_id subdomain_id,
                             std::vector<bool>        &owned_vertices);

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
