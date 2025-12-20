#ifndef INITIAL_CONDITIONS_H
#define INITIAL_CONDITIONS_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>

using namespace dealii;

namespace Parameters
{
  template <int dim>
  class InitialVelocity;
  template <int dim>
  class InitialCHNSTracer;

  /**
   * Structure storing the initial conditions. This is essentially a collection
   * of pointers to functions defined hereafter.
   *
   * Initial conditions are required for fields with a time evolution equation.
   * In particular, an initial pressure is not required for incompressible flow
   * problems. Although this is not a time evolution per se, the "initial
   * condition" for the mesh position in pseudosolid problems is the initial
   * mesh, so this condition does not need to be specified either.
   *
   * All initial conditions are nodal for now (suitable for continuous Galerkin,
   * but not for DG).
   *
   * Note: Most applications (FSI, Cahn-Hilliard, etc.) are tackled in a
   * monolithic fashion, so the number of components for the solution vector
   * varies from one solver to another, and we need to specify for each field
   * its lower bound and the number of components at creation. This is done by
   * initializing the initial conditions for all relevant fields, by calling,
   * e.g., create_initial_velocity() with the appropriate lower bound and number
   * of component.
   *
   * FIXME: Think of a better way of creating these functions...
   */
  template <int dim>
  class InitialConditions
  {
  public:
    InitialConditions()
      : initial_velocity_callback(
          std::make_shared<Functions::ParsedFunction<dim>>(dim))
      , initial_chns_tracer_callback(
          std::make_shared<Functions::ParsedFunction<dim>>(1))
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

    /**
     * Create the actual CHNS tracer, once the components and layout are known.
     */
    void create_initial_chns_tracer(const unsigned int phi_lower,
                                    const unsigned int n_components)
    {
      initial_chns_tracer =
        std::make_shared<Parameters::InitialCHNSTracer<dim>>(
          phi_lower, n_components, initial_chns_tracer_callback);
    }

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);

  public:
    // Flow velocity data
    std::shared_ptr<Functions::ParsedFunction<dim>> initial_velocity_callback;
    std::shared_ptr<InitialVelocity<dim>>           initial_velocity;

    // CHNS tracer data
    std::shared_ptr<Functions::ParsedFunction<dim>>
                                            initial_chns_tracer_callback;
    std::shared_ptr<InitialCHNSTracer<dim>> initial_chns_tracer;

    // If true, the initial condition is specified by a manufactured solution
    bool set_to_mms;
  };

  /**
   * Initial condition for the flow velocity.
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
   * Initial condition for the scalar tracer phi in CHNS models.
   * This is a function with @p n_components components, which only fills
   * the phi_lower-th component.
   */
  template <int dim>
  class InitialCHNSTracer : public Function<dim>
  {
  public:
    const unsigned int phi_lower;
    std::shared_ptr<Functions::ParsedFunction<dim>>
      initial_chns_tracer_callback;

  public:
    InitialCHNSTracer(const unsigned int phi_lower,
                      const unsigned int n_components,
                      std::shared_ptr<Functions::ParsedFunction<dim>>
                        initial_chns_tracer_callback)
      : Function<dim>(n_components)
      , phi_lower(phi_lower)
      , initial_chns_tracer_callback(initial_chns_tracer_callback)
    {}

    virtual double value(const Point<dim> &p,
                         unsigned int      component) const override
    {
      if (component == phi_lower)
        return initial_chns_tracer_callback->value(p);
      return 0.;
    }
  };

  template <int dim>
  void InitialConditions<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Initial conditions");
    {
      prm.declare_entry("to mms",
                        "false",
                        Patterns::Bool(),
                        "If true, initial conditions are specified by the "
                        "prescribed manufactured solution.");
      prm.enter_subsection("velocity");
      initial_velocity_callback->declare_parameters(prm, dim);
      prm.leave_subsection();
      prm.enter_subsection("cahn hilliard tracer");
      initial_chns_tracer_callback->declare_parameters(prm, 1);
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <int dim>
  void InitialConditions<dim>::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Initial conditions");
    {
      set_to_mms = prm.get_bool("to mms");
      prm.enter_subsection("velocity");
      initial_velocity_callback->parse_parameters(prm);
      prm.leave_subsection();
      prm.enter_subsection("cahn hilliard tracer");
      initial_chns_tracer_callback->parse_parameters(prm);
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }
} // namespace Parameters

#endif