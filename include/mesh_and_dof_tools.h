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

  // 1. Pré-calcul de la correspondance (mapping) sur la cellule de référence
  // Pour chaque DoF local de la source, on trouve l'indice du DoF local de
  // destination correspondant
  std::vector<unsigned int> source_to_dest_local_dof(
    fe_source.dofs_per_cell, numbers::invalid_unsigned_int);

  const std::vector<Point<dim>> &source_support_points =
    fe_source.get_unit_support_points();
  const std::vector<Point<dim>> &dest_support_points =
    fe_dest.get_unit_support_points();

  for (unsigned int i = 0; i < fe_source.dofs_per_cell; ++i)
  {
    const unsigned int comp_s = fe_source.system_to_component_index(i).first;

    // Si cette composante source fait partie de celles qu'on veut extraire
    if (source_comp_to_dest_comp.find(comp_s) != source_comp_to_dest_comp.end())
    {
      const unsigned int target_comp_d = source_comp_to_dest_comp.at(comp_s);

      // On cherche le DoF équivalent dans la cellule de destination
      for (unsigned int j = 0; j < fe_dest.dofs_per_cell; ++j)
      {
        const unsigned int comp_d = fe_dest.system_to_component_index(j).first;

        // Correspondance parfaite : même composante (mappée) ET même position
        // spatiale
        if (comp_d == target_comp_d &&
            source_support_points[i].distance(dest_support_points[j]) < 1e-12)
        {
          source_to_dest_local_dof[i] = j;
          break; // Trouvé, on passe au DoF source suivant
        }
      }
    }
  }

  // 2. Itération sur les cellules actives des deux DoFHandlers simultanément
  auto cell_source = dof_handler_source.begin_active();
  auto cell_dest   = dof_handler_destination.begin_active();

  Vector<double> local_values_source(fe_source.dofs_per_cell);
  std::vector<types::global_dof_index> local_indices_dest(
    fe_dest.dofs_per_cell);

  for (; cell_source != dof_handler_source.end(); ++cell_source, ++cell_dest)
  {
    // Important pour le MPI : on n'écrit que sur les cellules possédées
    // localement
    if (cell_dest->is_locally_owned())
    {
      // Récupère les valeurs du vecteur source pour cette cellule
      cell_source->get_dof_values(source, local_values_source);

      // Récupère les indices globaux de destination pour cette cellule
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
