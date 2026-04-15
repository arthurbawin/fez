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

/*
 * Create a map to go from tensor-valued data stored at (owned) mesh vertices
 * to the global dof indices of their components represented as an FE field.
 *
 * Important: To identify the dofs to the vertex indices, this function assumes
 * an isoparametric representation of the FE field.
 *
 * This map can be used to transfer tensor-valued data stored as an std::vector
 * to its representation as an LA::ParVectorType, which handles the ghost
 * entries.
 */
template <int dim, int n_components>
void create_mesh_vertex_to_tensor_dofs_maps(
  const unsigned int     component_offset,
  const DoFHandler<dim> &dof_handler,
  const IndexSet        &locally_relevant_dofs,
  std::vector<std::array<types::global_dof_index, n_components>>
    &vertex_to_dofs,
  std::vector<std::pair<types::global_vertex_index, unsigned int>>
    &dofs_to_vertex);

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


template <int dim, int n_components>
void create_mesh_vertex_to_tensor_dofs_maps(
  const unsigned int     component_offset,
  const DoFHandler<dim> &dof_handler,
  const IndexSet        &locally_relevant_dofs,
  std::vector<std::array<types::global_dof_index, n_components>>
    &vertex_to_dofs,
  std::vector<std::pair<types::global_vertex_index, unsigned int>>
    &dofs_to_vertex)
{
  vertex_to_dofs.clear();
  dofs_to_vertex.clear();

  const unsigned int n_vertices = dof_handler.get_triangulation().n_vertices();

  vertex_to_dofs.resize(n_vertices);
  dofs_to_vertex.resize(locally_relevant_dofs.n_elements(),
                        {numbers::invalid_unsigned_int,
                         numbers::invalid_unsigned_int});

  const unsigned int n_dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

  // Loop over owned and ghost cells
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    const auto &fe = cell->get_fe();
    cell->get_dof_indices(local_dof_indices);

    /**
     * In debug, check that we are indeed in an isoparametric setting.
     * This is not a sufficient condition, but at least the number of shape
     * functions should match the number of mesh vertices per cell.
     */
    if constexpr (running_in_debug_mode())
    {
      for (unsigned int i = 0; i < fe.n_base_elements(); ++i)
      {
        Assert(
          fe.base_element(i).n_dofs_per_cell() == cell->n_vertices(),
          ExcMessage(
            "Expected an isoparametric DoFHandler, but there are base elements "
            "in the FESystem with more dofs per cell than mesh vertices."));
      }
    }

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
    {
      const auto         component_shape = fe.system_to_component_index(i);
      const unsigned int comp            = component_shape.first;
      const unsigned int shape           = component_shape.second;

      // Use "shape" as the vertex index (which matches the shape function index
      // if isoparametric)
      const auto vertex_index = cell->vertex_index(shape);
      const int  c            = comp - component_offset;

      if (0 <= c && c < n_components)
      {
        // To go from vertex-based data to isoparametric dof data
        vertex_to_dofs[vertex_index][c] = local_dof_indices[i];

        // To go from isoparametric dof data to vertex-based data
        dofs_to_vertex[locally_relevant_dofs.index_within_set(
          local_dof_indices[i])] = {vertex_index, c};
      }
    }
  }
}

#endif
