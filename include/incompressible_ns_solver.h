#ifndef INCOMPRESSIBLE_NS_SOLVER_H
#define INCOMPRESSIBLE_NS_SOLVER_H

#include <copy_data.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <navier_stokes_solver.h>
#include <scratch_data.h>

using namespace dealii;

/**
 * Incompressible Navier-Stokes solver.
 * Solves the nonstabilized incompressible Navier-Stokes equations :
 *
 *                                                   - div(u) = 0,
 *
 *          dudt + (u dot grad) u + grad(p) - nu * lap(u) + f = 0,
 *
 * where the fluid density rho is absorbed in the pressure, that is, we
 * actually solve for p/rho instead of p.
 *
 * Note that since div(u) = 0, the grad(div(u)) term obtained by expanding
 * div(sigma(u,p)) has been removed. This should be kept in mind when:
 *
 *    - Writing source terms for test verification with manufactured solutions.
 *      In particular, the implemented source term is
 *
 *  f = -(du_mms/dt + (u_mms dot grad) u_mms + grad_p_mms - nu * lap_u_mms)
 *
 *    - Enforcing natural boundary conditions. The natural boundary condition
 *      arising from the solved equations is the open boundary condition:
 *
 *                   (-pI + nu*grad(u)) \cdot n = g,
 *
 *      rather than the traction boundary condition:
 *
 *              (-pI + nu*(grad(u) + grad(u)^T)) \cdot n = g.
 *
 * Because the system is not stabilized with e.g. PSPG terms, LBB stable mixed
 * finite elements should be used, the most straightforward being the P2-P1
 * Taylor-Hood element.
 */
template <int dim>
class NSSolver : public NavierStokesSolver<dim>
{
public:
  /**
   * Constructor
   */
  NSSolver(const ParameterReader<dim> &param);

  /**
   * Destructor
   */
  virtual ~NSSolver() {}

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
  virtual const FESystem<dim> &get_fe_system() const override { return fe; }

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
    ScratchDataIncompressibleNS<dim>                     &scratchData,
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
                     ScratchDataIncompressibleNS<dim> &scratchData,
                     CopyData                         &copy_data);

  /**
   * See copy_local_to_global_matrix.
   */
  void copy_local_to_global_rhs(const CopyData &copy_data);

protected:
  FESystem<dim> fe;

  // Non-owning pointer to base class fixed_mapping, used for clarity.
  const Mapping<dim> *mapping;

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
      else
        return mms.exact_pressure->value(p);
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      Assert(values.size() == n_components, ExcMessage("Component mismatch"));
      for (unsigned int d = 0; d < dim; ++d)
        values[u_lower + d] = mms.exact_velocity->value(p, d);
      values[p_lower] = mms.exact_pressure->value(p);
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