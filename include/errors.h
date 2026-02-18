#ifndef ERRORS_H
#define ERRORS_H

#include <deal.II/base/function.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

/**
 * Generic function to compute Sobolev norms of the error.
 */
template <int dim, typename VectorType>
double
compute_error_norm(const Triangulation<dim>           &tria,
                   const Mapping<dim>                 &mapping,
                   const DoFHandler<dim>              &dof_handler,
                   const VectorType                   &current_solution,
                   const Function<dim>                &exact_solution,
                   Vector<double>                     &cellwise_errors,
                   const Quadrature<dim>              &q,
                   const VectorTools::NormType         norm_type,
                   const ComponentSelectFunction<dim> *component_function)
{
  VectorTools::integrate_difference(mapping,
                                    dof_handler,
                                    current_solution,
                                    exact_solution,
                                    cellwise_errors,
                                    q,
                                    norm_type,
                                    component_function);
  double norm =
    VectorTools::compute_global_error(tria, cellwise_errors, norm_type);
  return norm;
}

#endif
