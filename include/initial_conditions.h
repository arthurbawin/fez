#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>

using namespace dealii;

/**
 * Initial conditions, which are required for problems with a
 * time evolution equation. In particular, an initial pressure
 * is not required for incompressible flow problems.
 *
 * The initial condition for the mesh position variable is
 * always the initial mesh, so this condition does not need to
 * be specified.
 */
namespace Parameters
{
  // All initial conditions are nodal for now (suitable for continuous Galerkin,
  // but not for DG)

  /**
   * Initial velocity condition.
   *
   * Because most applications (FSI, Cahn-Hilliard, etc.) will be tackled in a
   * monolithic fashion, the number of components for the solution vector varies
   * from one problem to another, and we need to specify the u_lower bound and
   * the number of components at creation.
   *
   * As a result, the actual initial_velocity (and other fields) can only be
   * created once the number of vector components is known, that is, when the
   * type of problem is selected (flow only, FSI, Cahn-Hilliard, etc.)
   *
   * FIXME: Think of a better way of creating these functions...
   */
  template <int dim>
  class InitialVelocity : public Function<dim>
  {
  public:
    const unsigned int                              u_lower;
    std::shared_ptr<Functions::ParsedFunction<dim>> initial_velocity_callback;

  public:
    InitialVelocity(
      const unsigned int                              u_lower,
      const unsigned int                              n_components,
      std::shared_ptr<Functions::ParsedFunction<dim>> initial_velocity_callback)
      : Function<dim>(n_components)
      , u_lower(u_lower)
      , initial_velocity_callback(initial_velocity_callback)
    {}

    virtual double value(const Point<dim> &p,
                         unsigned int      component) const override
    {
      for (unsigned int d = 0; d < dim; ++d)
        if (component == u_lower + d)
          return initial_velocity_callback->value(p, d);
      return 0.;
    }
  };

  /**
   *
   */
  template <int dim>
  class InitialConditions
  {
  public:
    InitialConditions()
      : initial_velocity_callback(
          std::make_shared<Functions::ParsedFunction<dim>>(dim))
    {}

    /**
     * Create the actual initial velocity.
     * This must be called by the various derived solver, for which
     * the number of variables (components) and layout is known.
     */
    void create_initial_velocity(const unsigned int u_lower,
                                 const unsigned int n_components)
    {
      initial_velocity = std::make_shared<Parameters::InitialVelocity<dim>>(
        u_lower, n_components, initial_velocity_callback);
    }

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);

  public:
    std::shared_ptr<Functions::ParsedFunction<dim>> initial_velocity_callback;
    std::shared_ptr<InitialVelocity<dim>>           initial_velocity;
  };

  template <int dim>
  void InitialConditions<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Initial conditions");
    {
      prm.enter_subsection("velocity");
      initial_velocity_callback->declare_parameters(prm);
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <int dim>
  void InitialConditions<dim>::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Initial conditions");
    {
      prm.enter_subsection("velocity");
      initial_velocity_callback->parse_parameters(prm);
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }
} // namespace Parameters

#endif