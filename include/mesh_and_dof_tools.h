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

/**
 * Extrait une sous-solution d'un vecteur source et l'injecte dans un vecteur de
 * destination, en gérant des espaces d'éléments finis (FESystem)
 * potentiellement différents, tant qu'ils partagent la même Triangulation (le
 * même maillage).
 */
template <int dim, typename VectorType>
void extract_subsolution(
  const DoFHandler<dim>                      &dof_handler_source,
  const DoFHandler<dim>                      &dof_handler_destination,
  const VectorType                           &source,
  VectorType                                 &destination,
  const std::map<unsigned int, unsigned int> &source_comp_to_dest_comp);

/* ---------------- template and inline functions ----------------- */

template <int dim, typename VectorType>
void extract_subsolution(
  const DoFHandler<dim>                      &dof_handler_source,
  const DoFHandler<dim>                      &dof_handler_destination,
  const VectorType                           &source,
  VectorType                                 &destination,
  const std::map<unsigned int, unsigned int> &source_comp_to_dest_comp)
{
  const auto &fe_source = dof_handler_source.get_fe();
  const auto &fe_dest   = dof_handler_destination.get_fe();

  // 1. pre-compute source-to-destination local dof mapping on the reference
  // cell
  std::vector<unsigned int> source_to_dest_local_dof(
    fe_source.dofs_per_cell, numbers::invalid_unsigned_int);

  const std::vector<Point<dim>> &source_support_points =
    fe_source.get_unit_support_points();
  const std::vector<Point<dim>> &dest_support_points =
    fe_dest.get_unit_support_points();

  for (unsigned int i = 0; i < fe_source.dofs_per_cell; ++i)
  {
    const unsigned int comp_s = fe_source.system_to_component_index(i).first;

    // only process components that are part of the extraction map
    if (source_comp_to_dest_comp.find(comp_s) != source_comp_to_dest_comp.end())
    {
      const unsigned int target_comp_d = source_comp_to_dest_comp.at(comp_s);

      // find the matching dof in the destination cell
      for (unsigned int j = 0; j < fe_dest.dofs_per_cell; ++j)
      {
        const unsigned int comp_d = fe_dest.system_to_component_index(j).first;

        // match: same (mapped) component AND same support point location
        if (comp_d == target_comp_d &&
            source_support_points[i].distance(dest_support_points[j]) < 1e-12)
        {
          source_to_dest_local_dof[i] = j;
          break;
        }
      }
    }
  }

  // 2. iterate over active cells of both dof handlers simultaneously
  auto cell_source = dof_handler_source.begin_active();
  auto cell_dest   = dof_handler_destination.begin_active();

  Vector<double> local_values_source(fe_source.dofs_per_cell);
  std::vector<types::global_dof_index> local_indices_dest(
    fe_dest.dofs_per_cell);

  for (; cell_source != dof_handler_source.end(); ++cell_source, ++cell_dest)
  {
    // only write on locally owned cells
    if (cell_dest->is_locally_owned())
    {
      // read source values for this cell
      cell_source->get_dof_values(source, local_values_source);

      // get destination global dof indices for this cell
      cell_dest->get_dof_indices(local_indices_dest);

      // On injecte les valeurs
      for (unsigned int i = 0; i < fe_source.dofs_per_cell; ++i)
      {
        if (source_to_dest_local_dof[i] != numbers::invalid_unsigned_int)
        {
          const unsigned int dest_local_idx = source_to_dest_local_dof[i];
          const types::global_dof_index dest_global_idx =
            local_indices_dest[dest_local_idx];

          destination[dest_global_idx] = local_values_source[i];
        }
      }
    }
  }
}

#endif
