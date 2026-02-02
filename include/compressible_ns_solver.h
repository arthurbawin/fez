#ifndef COMPRESSIBLE_NS_SOLVER_H
#define COMPRESSIBLE_NS_SOLVER_H

#include <copy_data.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <navier_stokes_solver.h>
#include <scratch_data.h>

using namespace dealii;

/**
 * Compressible Navier-Stokes solver.
 * TODO : Add equations and model.
 */
template <int dim>
class CompressibleNSSolver : public NavierStokesSolver<dim>
{
  using ScratchData = ScratchDataCompressibleNS<dim>;

public:
  /**
   * Constructor
   */
  CompressibleNSSolver(const ParameterReader<dim> &param);

  /**
   * Destructor
   */
  virtual ~CompressibleNSSolver() {}

  /**
   * Apply initial condition on the temperature and pressure
   */
  virtual void set_solver_specific_initial_conditions() override;

  /**
   * Apply exact temperature
   */
  virtual void set_solver_specific_exact_solution() override;

  /**
   *
   */
  virtual void create_sparsity_pattern() override;

  /**
   *
   */
  virtual void output_results() override;

  /**
   * Get the FESystem of the derived solver
   */
  virtual const FESystem<dim> &get_fe_system() const override { return *fe; }

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
  std::shared_ptr<FESystem<dim>> fe;

  // Non-owning pointer to base class fixed_mapping, used for clarity.
  const Mapping<dim> *mapping;

  FEValuesExtractors::Scalar temperature_extractor;
  ComponentMask temperature_mask;

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
      , n_components(ordering.n_components)
      , u_lower(ordering.u_lower)
      , p_lower(ordering.p_lower)
      , mms(mms)
    {}

    // Update time in the mms functions
    virtual void set_time(const double new_time) override
    {
      mms.set_time(new_time);
    }

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      Assert(component < n_components, ExcMessage("Component mismatch"));
      if (component < dim)
        return mms.exact_velocity->value(p, component);
      else if (component == dim)
        return mms.exact_pressure->value(p);
      else if (component == dim + 1)
        return mms.exact_temperature->value(p);
      else
        DEAL_II_ASSERT_UNREACHABLE();
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      Assert(values.size() == n_components, ExcMessage("Component mismatch"));
      for (unsigned int d = 0; d < dim; ++d)
        values[u_lower + d] = mms.exact_velocity->value(p, d);
      values[p_lower] = mms.exact_pressure->value(p);
      // FIXME: A completer pour T
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim>  &p,
             const unsigned int component = 0) const override
    {
      Assert(component < n_components, ExcMessage("Component mismatch"));
      if (component < dim)
        return mms.exact_velocity->gradient(p, component);
      else
        return mms.exact_pressure->gradient(p);
      // FIXME: A completer pour T
    }

  protected:
    const unsigned int n_components;
    const unsigned int u_lower;
    const unsigned int p_lower;
    // MMS cannot be const since its internal time must be updated
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
    MMSSourceTerm(
      const double                               time,
      const ComponentOrdering                   &ordering,
      const Parameters::PhysicalProperties<dim> &physical_properties,
      const ManufacturedSolutions::ManufacturedSolution<dim> &mms)
      : Function<dim>(ordering.n_components, time)
      , n_components(ordering.n_components)
      , u_lower(ordering.u_lower)
      , p_lower(ordering.p_lower)
      , physical_properties(physical_properties)
      , mms(mms)
    {}

    // Update time in the mms functions
    virtual void set_time(const double new_time) override
    {
      mms.set_time(new_time);
    }

    /**
     * Evaluate the combined velocity-pressure source term for the
     * incompressible Navier-Stokes momentum-mass equations.
     */
    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override;

  protected:
    const unsigned int                         n_components;
    const unsigned int                         u_lower;
    const unsigned int                         p_lower;
    const Parameters::PhysicalProperties<dim> &physical_properties;
    // MMS cannot be const since its internal time must be updated
    ManufacturedSolutions::ManufacturedSolution<dim> mms;
  };
};

#endif