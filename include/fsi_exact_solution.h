#ifndef FSI_MMS_SOLUTION_H
#define FSI_MMS_SOLUTION_H

#include <components_ordering.h>
#include <deal.II/base/function.h>
#include <manufactured_solution.h>

using namespace dealii;

/**
 * Exact solution for the FSI solvers when performing a convergence study with a
 * manufactured solution.
 *
 * This class is defined in its own file because the ScratchData base class
 * needs the full definition of FSIExactSolution, which requires including e.g.
 * monolithic_fsi_solver.h in scratch_data.h. If we want to use an alias for the
 * ScratchDataFSI template in FSI solvers, then it requires forward declaring
 * the full class, whereas for other solvers we can simply declare a templated
 * alias in scratch_data.h (e.g., template <int dim> using
 * ScratchDataIncompressibleNS = ScratchData<dim>;)
 *
 * The reason why the scratch needs the present definition is because
 * convergence studies for the FSI solvers require the time derivatives of the
 * mesh position, which is not accessible through deal.II's Function<dim>
 * interface (it provides the gradients, but not the time derivative).
 */
template <int dim>
class FSIExactSolution : public Function<dim>
{
public:
  FSIExactSolution(const double             time,
                   const ComponentOrdering &ordering,
                   const ManufacturedSolutions::ManufacturedSolution<dim> &mms)
    : Function<dim>(ordering.n_components, time)
    , ordering(ordering)
    , mms(mms)
  {}

  virtual void set_time(const double new_time) override
  {
    FunctionTime<double>::set_time(new_time);
    mms.set_time(new_time);
  }

  virtual double value(const Point<dim>  &p,
                       const unsigned int component = 0) const override
  {
    if (ordering.is_velocity(component))
      return mms.exact_velocity->value(p, component - ordering.u_lower);
    else if (ordering.is_pressure(component))
      return mms.exact_pressure->value(p);
    else if (ordering.is_position(component))
      return mms.exact_mesh_position->value(p, component - ordering.x_lower);
    else if (ordering.is_lambda(component))
      // This exact solution should only be called when source terms are
      // applied to the Lagrange multiplier equation, that is, if
      // LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM is defined.
      // Otherwise, the function lagrange_multiplier() below should be called.
      // instead. It can only be called at quadrature nodes on faces, where
      // the normal is well-defined.
      return mms.exact_lagrange_multiplier->value(p,
                                                  component - ordering.l_lower);
    else
      DEAL_II_ASSERT_UNREACHABLE();
  }

  double time_derivative(const Point<dim>  &p,
                         const unsigned int component = 0) const
  {
    if (ordering.is_velocity(component))
      return mms.exact_velocity->time_derivative(p,
                                                 component - ordering.u_lower);
    else if (ordering.is_pressure(component))
      return mms.exact_pressure->time_derivative(p);
    else if (ordering.is_position(component))
      return mms.exact_mesh_position->time_derivative(p,
                                                      component -
                                                        ordering.x_lower);
    else if (ordering.is_lambda(component))
      return mms.exact_lagrange_multiplier->time_derivative(p,
                                                            component -
                                                              ordering.l_lower);
    else
      DEAL_II_ASSERT_UNREACHABLE();
  }

  /**
   * Exact Lagrange multiplier requires local unit normal vector
   */
  void lagrange_multiplier(const Point<dim>     &p,
                           const double          mu_viscosity,
                           const Tensor<1, dim> &normal_to_solid,
                           Tensor<1, dim>       &lambda) const
  {
    Tensor<2, dim> sigma;
    sigma                 = 0;
    const double pressure = mms.exact_pressure->value(p);
    for (unsigned int d = 0; d < dim; ++d)
      sigma[d][d] = -pressure;
    Tensor<2, dim> grad_u = mms.exact_velocity->gradient_vj_xi(p);
    sigma += mu_viscosity * (grad_u + transpose(grad_u));
    lambda = -sigma * normal_to_solid;
  }

  virtual Tensor<1, dim>
  gradient(const Point<dim> &p, const unsigned int component = 0) const override
  {
    if (ordering.is_velocity(component))
      return mms.exact_velocity->gradient(p, component - ordering.u_lower);
    else if (ordering.is_pressure(component))
      return mms.exact_pressure->gradient(p);
    else if (ordering.is_position(component))
      return mms.exact_mesh_position->gradient(p, component - ordering.x_lower);
    else if (ordering.is_lambda(component))
      return Tensor<1, dim>();
    else
      DEAL_II_ASSERT_UNREACHABLE();
  }

public:
  const ComponentOrdering                         &ordering;
  ManufacturedSolutions::ManufacturedSolution<dim> mms;
};

#endif
