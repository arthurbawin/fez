#ifndef MESH_AND_DOF_TOOLS_H
#define MESH_AND_DOF_TOOLS_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/vector.h>
#include <utilities.h>

#include <map>

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

/**
 * Copy selected components of a finite-element solution from a source
 * DoFHandler to a destination DoFHandler defined on the *same* triangulation
 * (hence the same parallel partition), possibly with different component
 * layouts. @p source_comp_to_dest_comp maps source components to destination
 * components; mapped components must share their unit support points.
 *
 * Only locally owned destination cells are written. Because both DoFHandlers
 * share the triangulation, a locally owned destination cell corresponds to a
 * locally owned source cell, so the source values are always available (no
 * ghost exchange is required). The caller must compress @p destination after
 * the call.
 */
template <int dim, typename VectorType>
void extract_subsolution(
  const DoFHandler<dim>                      &dof_handler_source,
  const DoFHandler<dim>                      &dof_handler_destination,
  const VectorType                           &source,
  VectorType                                 &destination,
  const std::map<unsigned int, unsigned int> &source_comp_to_dest_comp)
{
  const auto &fe_source      = dof_handler_source.get_fe();
  const auto &fe_destination = dof_handler_destination.get_fe();

  // Precompute, once, the source-dof -> destination-local-dof correspondence
  // (identical for every cell of a non-hp discretization).
  std::vector<unsigned int> source_to_destination(
    fe_source.dofs_per_cell, numbers::invalid_unsigned_int);

  const auto &source_support_points = fe_source.get_unit_support_points();
  const auto &destination_support_points =
    fe_destination.get_unit_support_points();

  for (unsigned int i = 0; i < fe_source.dofs_per_cell; ++i)
  {
    const unsigned int source_component =
      fe_source.system_to_component_index(i).first;
    const auto mapped_component =
      source_comp_to_dest_comp.find(source_component);
    if (mapped_component == source_comp_to_dest_comp.end())
      continue;

    for (unsigned int j = 0; j < fe_destination.dofs_per_cell; ++j)
      if (fe_destination.system_to_component_index(j).first ==
            mapped_component->second &&
          source_support_points[i].distance(destination_support_points[j]) <
            1e-12)
      {
        source_to_destination[i] = j;
        break;
      }

    AssertThrow(source_to_destination[i] != numbers::invalid_unsigned_int,
                ExcMessage("Could not match a source support point in the "
                           "destination finite-element system."));
  }

  auto source_cell      = dof_handler_source.begin_active();
  auto destination_cell = dof_handler_destination.begin_active();

  Vector<double> source_local_values(fe_source.dofs_per_cell);
  std::vector<types::global_dof_index> destination_local_indices(
    fe_destination.dofs_per_cell);

  for (; source_cell != dof_handler_source.end();
       ++source_cell, ++destination_cell)
  {
    AssertThrow(destination_cell != dof_handler_destination.end(),
                ExcMessage("Source and destination triangulations have "
                           "different active-cell layouts."));
    if (!destination_cell->is_locally_owned())
      continue;

    source_cell->get_dof_values(source, source_local_values);
    destination_cell->get_dof_indices(destination_local_indices);

    for (unsigned int i = 0; i < fe_source.dofs_per_cell; ++i)
      if (source_to_destination[i] != numbers::invalid_unsigned_int)
        destination[destination_local_indices[source_to_destination[i]]] =
          source_local_values[i];
  }

  AssertThrow(destination_cell == dof_handler_destination.end(),
              ExcMessage("Source and destination triangulations have "
                         "different numbers of active cells."));
}

#endif
