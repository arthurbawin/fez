#ifndef INCOMPRESSIBLE_CHNS_SOLVER_H
#define INCOMPRESSIBLE_CHNS_SOLVER_H

#include <copy_data.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <navier_stokes_solver.h>
#include <scratch_data.h>

using namespace dealii;

/**
 * Quasi-incompressible Cahn-Hilliard Navier-Stokes solver.
 * TODO: Add equations.
 */
template <int dim, bool with_moving_mesh = false>
class CHNSSolver : public NavierStokesSolver<dim, with_moving_mesh>
{
  using ScratchData = ScratchDataCHNS<dim, with_moving_mesh>;

public:
  /**
   * Constructor
   */
  CHNSSolver(const ParameterReader<dim> &param);

  /**
   * Destructor
   */
  virtual ~CHNSSolver() {}

  /**
   * Apply initial condition on the tracer (phase marker)
   */
  virtual void set_solver_specific_initial_conditions() override;

  /**
   * Apply exact tracer and potential
   */
  virtual void set_solver_specific_exact_solution() override;

  virtual void create_solver_specific_zero_constraints() override;
  virtual void create_solver_specific_nonzero_constraints() override;

  /**
   *
   */
  virtual void create_sparsity_pattern() override;

  virtual void compute_solver_specific_errors() override;

  /**
   * Assemble the linearized Jacobian matrix at the current evaluation point
   */
  virtual void assemble_matrix() override;

  /**
   * Compute the element-wise matrix. This function is passed to
   * WorkStream::run to perform multithreaded assembly if supported
   * (i.e., if using thread-safe matrix and vector wrappers).
   */
  void assemble_local_matrix(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData                                          &scratchData,
    CopyData                                             &copy_data);

  /**
   * Compute the element-wise matrix using finite differences.
   * FIXME: Rethink this to handle the finite difference computations
   * in the base class and not in derived solvers.
   */
  void assemble_local_matrix_finite_differences(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData                                          &scratchData,
    CopyData                                             &copy_data);

  /**
   * Assemble the element-wise matrix computed with assemble_local_matrix
   * into the global matrix. Passed to WorkStream::run (see above).
   */
  void copy_local_to_global_matrix(const CopyData &copy_data);

  virtual void compare_analytical_matrix_with_fd() override;

  /**
   * Assemble the Newton residual at the current evaluation point
   */
  virtual void assemble_rhs() override;

  /**
   * See assemble_local_matrix.
   */
  void
  assemble_local_rhs(const typename DoFHandler<dim>::active_cell_iterator &cell,
                     ScratchData &scratchData,
                     CopyData    &copy_data);

  /**
   * See copy_local_to_global_matrix.
   */
  void copy_local_to_global_rhs(const CopyData &copy_data);

protected:
  virtual std::vector<std::pair<std::string, unsigned int>>
  get_additional_variables_description() const override
  {
    std::vector<std::pair<std::string, unsigned int>> description;
    description.push_back({"tracer", 1});
    description.push_back({"potential", 1});
    return description;
  }

  virtual const FESystem<dim> &get_fe_system() const override { return *fe; }

  virtual bool uses_hp_capabilities() const override { return false; };

protected:
  std::shared_ptr<FESystem<dim>> fe;

  static constexpr ConstexprComponentOrderingCHNS<dim, with_moving_mesh>
    const_ordering = {};

  FEValuesExtractors::Scalar tracer_extractor;
  FEValuesExtractors::Scalar potential_extractor;
  ComponentMask              tracer_mask;
  ComponentMask              potential_mask;

protected:
  /**
   * Source term.
   */
  class SourceTerm : public Function<dim>
  {
  public:
    SourceTerm(const double                        time,
               const ComponentOrdering            &ordering,
               const Parameters::SourceTerms<dim> &source_terms)
      : Function<dim>(ordering.n_components, time)
      , ordering(ordering)
      , source_terms(source_terms)
    {}

    virtual void set_time(const double new_time) override
    {
      source_terms.set_time(new_time);
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      // source_terms.fluid_source is a function with dim+1 components
      for (unsigned int d = 0; d < dim; ++d)
      {
        values[ordering.u_lower + d] = source_terms.fluid_source->value(p, d);
        if constexpr (with_moving_mesh)
          values[ordering.x_lower + d] =
            source_terms.pseudosolid_source->value(p, d);
      }
      values[ordering.p_lower] = source_terms.fluid_source->value(p, dim);
      values[ordering.phi_lower] =
        source_terms.cahnhilliard_source->value(p, 0);
      values[ordering.mu_lower] = source_terms.cahnhilliard_source->value(p, 1);
    }

  protected:
    const ComponentOrdering     &ordering;
    Parameters::SourceTerms<dim> source_terms;
  };

  /**
   * Exact solution when performing a convergence study with a manufactured
   * solution.
   */
  class MMSSolution : public Function<dim>
  {
  public:
    MMSSolution(const double             time,
                const ComponentOrdering &ordering,
                const ManufacturedSolutions::ManufacturedSolution<dim> &mms)
      : Function<dim>(ordering.n_components, time)
      , ordering(ordering)
      , n_components(ordering.n_components)
      , u_lower(ordering.u_lower)
      , p_lower(ordering.p_lower)
      , x_lower(ordering.x_lower)
      , phi_lower(ordering.phi_lower)
      , mu_lower(ordering.mu_lower)
      , mms(mms)
    {}

    // Update time in the mms functions
    virtual void set_time(const double new_time) override
    {
      mms.set_time(new_time);
    }

    /**
     * Setting the individual value function is required for e.g.
     * constraining a pressure point, which calls value() for pressure.
     */
    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      Assert(component < n_components, ExcMessage("Component mismatch"));
      if constexpr (with_moving_mesh)
        if (ordering.is_position(component))
          return mms.exact_mesh_position->value(p,
                                                component - ordering.x_lower);
      if (ordering.is_velocity(component))
        return mms.exact_velocity->value(p, component - ordering.u_lower);
      else if (ordering.is_pressure(component))
        return mms.exact_pressure->value(p);
      else if (ordering.is_tracer(component))
        return mms.exact_tracer->value(p);
      else if (ordering.is_potential(component))
        return mms.exact_potential->value(p);
      else
        DEAL_II_ASSERT_UNREACHABLE();
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      Assert(values.size() == n_components, ExcMessage("Component mismatch"));
      for (unsigned int d = 0; d < dim; ++d)
      {
        values[u_lower + d] = mms.exact_velocity->value(p, d);
        if constexpr (with_moving_mesh)
          values[x_lower + d] = mms.exact_mesh_position->value(p, d);
      }
      values[p_lower]   = mms.exact_pressure->value(p);
      values[phi_lower] = mms.exact_tracer->value(p);
      values[mu_lower]  = mms.exact_potential->value(p);
    }

    /**
     * Required for H1 norm computations on invididual components
     */
    virtual Tensor<1, dim>
    gradient(const Point<dim>  &p,
             const unsigned int component = 0) const override
    {
      Assert(component < n_components, ExcMessage("Component mismatch"));
      if constexpr (with_moving_mesh)
        if (ordering.is_position(component))
          return mms.exact_mesh_position->gradient(p,
                                                   component -
                                                     ordering.x_lower);
      if (ordering.is_velocity(component))
        return mms.exact_velocity->gradient(p, component - ordering.u_lower);
      else if (ordering.is_pressure(component))
        return mms.exact_pressure->gradient(p);
      else if (ordering.is_tracer(component))
        return mms.exact_tracer->gradient(p);
      else if (ordering.is_potential(component))
        return mms.exact_potential->gradient(p);
      else
        DEAL_II_ASSERT_UNREACHABLE();
    }

    virtual void
    vector_gradient(const Point<dim>            &p,
                    std::vector<Tensor<1, dim>> &gradients) const override
    {
      Assert(gradients.size() == n_components,
             ExcMessage("Component mismatch"));
      for (unsigned int d = 0; d < dim; ++d)
      {
        gradients[u_lower + d] = mms.exact_velocity->gradient(p, d);
        if constexpr (with_moving_mesh)
          gradients[x_lower + d] = mms.exact_mesh_position->gradient(p, d);
      }
      gradients[p_lower]   = mms.exact_pressure->gradient(p);
      gradients[phi_lower] = mms.exact_tracer->gradient(p);
      gradients[mu_lower]  = mms.exact_potential->gradient(p);
    }

  protected:
    const ComponentOrdering                         &ordering;
    const unsigned int                               n_components;
    const unsigned int                               u_lower;
    const unsigned int                               p_lower;
    const unsigned int                               x_lower;
    const unsigned int                               phi_lower;
    const unsigned int                               mu_lower;
    ManufacturedSolutions::ManufacturedSolution<dim> mms;
  };

  /**
   * Source term called when performing a convergence study.
   * This function calls the derivatives functions from the given
   * pre-set manufactured solution.
   */
  class MMSSourceTerm : public Function<dim>
  {
  public:
    MMSSourceTerm(const double                time,
                  const ComponentOrdering    &ordering,
                  const ParameterReader<dim> &param)
      : Function<dim>(ordering.n_components, time)
      , n_components(ordering.n_components)
      , u_lower(ordering.u_lower)
      , p_lower(ordering.p_lower)
      , x_lower(ordering.x_lower)
      , phi_lower(ordering.phi_lower)
      , mu_lower(ordering.mu_lower)
      , physical_properties(param.physical_properties)
      , cahn_hilliard_param(param.cahn_hilliard)
      , mms(param.mms)
    {}

    // Update time in the mms functions
    virtual void set_time(const double new_time) override
    {
      mms.set_time(new_time);
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override;

  protected:
    const unsigned int                               n_components;
    const unsigned int                               u_lower;
    const unsigned int                               p_lower;
    const unsigned int                               x_lower;
    const unsigned int                               phi_lower;
    const unsigned int                               mu_lower;
    const Parameters::PhysicalProperties<dim>       &physical_properties;
    const Parameters::CahnHilliard<dim>             &cahn_hilliard_param;
    ManufacturedSolutions::ManufacturedSolution<dim> mms;
  };
};

#endif
