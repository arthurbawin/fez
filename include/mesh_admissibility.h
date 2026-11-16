#ifndef MESH_ADMISSIBILITY_H
#define MESH_ADMISSIBILITY_H

#include <deal.II/base/mpi.h>
#include <deal.II/hp/fe_values.h>

#include <cmath>

/**
 * Check the trial position x(X) on the reference mesh. A positive determinant
 * at the assembly quadrature points is necessary, but does not certify
 * validity elsewhere in a cell. Supports ordinary and hp DoFHandlers.
 * All ranks participate, including those without locally owned cells.
 */
template <int dim, typename VectorType>
bool mesh_is_admissible(const dealii::Mapping<dim>    &reference_mapping,
                        const dealii::DoFHandler<dim> &dof_handler,
                        const dealii::Quadrature<dim> &quadrature,
                        const dealii::FEValuesExtractors::Vector &position,
                        const VectorType &evaluation_point)
{
  // hp::FEValues retains a reference to the mapping collection.
  const dealii::hp::MappingCollection<dim> mappings(reference_mapping);
  dealii::hp::FEValues<dim>                hp_values(mappings,
                                      dof_handler.get_fe_collection(),
                                      dealii::hp::QCollection<dim>(quadrature),
                                      dealii::update_gradients);
  std::vector<dealii::Tensor<2, dim>>      gradients(quadrature.size());
  unsigned int                             locally_valid = 1;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      hp_values.reinit(cell);
      hp_values.get_present_fe_values()[position].get_function_gradients(
        evaluation_point, gradients);
      for (const auto &F : gradients)
      {
        const double J = dealii::determinant(F);
        if (!std::isfinite(J) || J <= 0.)
        {
          locally_valid = 0;
          break;
        }
      }
      if (!locally_valid)
        break;
    }
  return dealii::Utilities::MPI::min(locally_valid,
                                     evaluation_point.get_mpi_communicator()) ==
         1;
}

#endif
