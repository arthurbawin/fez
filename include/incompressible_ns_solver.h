#ifndef INCOMPRESSIBLE_NS_SOLVER_H
#define INCOMPRESSIBLE_NS_SOLVER_H

#include <copy_data.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/lac/affine_constraints.h>
#include <error_handler.h>
#include <generic_solver.h>
#include <parameter_reader.h>
#include <scratch_data.h>
#include <time_handler.h>
#include <types.h>

using namespace dealii;

/**
 * Incompressible Navier-Stokes solver.
 * Solves the nonstabilized incompressible Navier-Stokes equations :
 *
 *                                                     div(u) = 0,
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
class IncompressibleNavierStokesSolver : public GenericSolver<LA::ParVectorType>
{
public:
  /**
   * Constructor
   */
  IncompressibleNavierStokesSolver(const ParameterReader<dim> &param);

  /**
   * Destructor
   */
  virtual ~IncompressibleNavierStokesSolver() {}

  /**
   * Solve the flow problem
   */
  virtual void run() override;

  /**
   * Initialize the dof handler and allocate parallel vectors
   */
  void setup_dofs();

  /**
   * Create the homogeneous constraints
   */
  void create_zero_constraints();

  /**
   * (Re-)create the nonhomogeneous constraints
   */
  void create_nonzero_constraints();

  virtual AffineConstraints<double> &get_nonzero_constraints() override
  {
    return nonzero_constraints;
  }

  /**
   *
   */
  void constrain_pressure_point(AffineConstraints<double> &constraints,
                                const bool                 set_to_zero);

  /**
   * Create the sparsity pattern and allocate matrix
   */
  void create_sparsity_pattern();

  /**
   * Apply initial conditions
   */
  void set_initial_conditions();

  /**
   * Set solution to exact solution, if provided
   */
  void set_exact_solution();

  /**
   * Recreate and apply nonhomogeneous constraints
   */
  void update_boundary_conditions();

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
    ScratchDataNS<dim>                                   &scratchData,
    CopyData                                             &copy_data);

  /**
   * Assemble the element-wise matrix computed with assemble_local_matrix
   * into the global matrix. Passed to WorkStream::run (see above).
   */
  void copy_local_to_global_matrix(const CopyData &copy_data);

  void compare_analytical_matrix_with_fd();

  /**
   * Assemble the Newton residual at the current evluation point
   */
  virtual void assemble_rhs() override;

  /**
   * See assemble_local_matrix.
   */
  void
  assemble_local_rhs(const typename DoFHandler<dim>::active_cell_iterator &cell,
                     ScratchDataNS<dim> &scratchData,
                     CopyData           &copy_data);

  /**
   * See copy_local_to_global_matrix.
   */
  void copy_local_to_global_rhs(const CopyData &copy_data);

  /**
   * Solve the linear system J(u) * du = -NL(u).
   */
  virtual void
  solve_linear_system(const bool apply_inhomogeneous_constraints) override;

  /**
   *
   */
  void compute_errors();

  /**
   * Write the velocity and pressure to vtu file.
   */
  void output_results() const;

  /**
   * Reset the resolution related structures (mesh, dof_handler, etc.) in
   * between two runs, e.g. when performing convergence tests.
   */
  void reset();

  /**
   * Update time in all relevant structures (boundary conditions, source terms,
   * exact solution).
   */
  void set_time();

  /**
   * Exact solution when performing a convergence study with a manufactured
   * solution.
   */
  class MMSSolution : public Function<dim>
  {
  public:
    MMSSolution(const double                                           time,
                const ManufacturedSolution::ManufacturedSolution<dim> &mms)
      : Function<dim>(n_components, time)
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
    static constexpr unsigned int n_components = dim + 1;
    static constexpr unsigned int u_lower      = 0;
    static constexpr unsigned int p_lower      = dim;

    // MMS cannot be const since its internal time must be updated
    ManufacturedSolution::ManufacturedSolution<dim> mms;
  };

  /**
   * Source term called when performing a convergence study.
   * This function calls the derivatives functions from the given
   * pre-set manufactured solution.
   */
  class MMSSourceTerm : public Function<dim>
  {
  public:
    MMSSourceTerm(const double                          time,
                  const Parameters::PhysicalProperties &physical_properties,
                  const ManufacturedSolution::ManufacturedSolution<dim> &mms)
      : Function<dim>(n_components, time)
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
    static constexpr unsigned int         n_components = dim + 1;
    static constexpr unsigned int         u_lower      = 0;
    static constexpr unsigned int         p_lower      = dim;
    const Parameters::PhysicalProperties &physical_properties;

    // MMS cannot be const since its internal time must be updated
    ManufacturedSolution::ManufacturedSolution<dim> mms;
  };

protected:
  // Ordering of the FE system for the incompressible NS solver.
  // Each field is in the half-open [lower, upper)
  // Check for matching component by doing e.g.:
  // if(u_lower <= comp && comp < u_upper)
  static constexpr unsigned int n_components = dim + 1;
  static constexpr unsigned int u_lower      = 0;
  static constexpr unsigned int u_upper      = dim;
  static constexpr unsigned int p_lower      = dim;
  static constexpr unsigned int p_upper      = dim + 1;

  const FEValuesExtractors::Vector velocity_extractor;
  const FEValuesExtractors::Scalar pressure_extractor;

  /**
   * Quality-of-life functions to check which field a given component is
   */
  inline bool is_velocity(const unsigned int component) const
  {
    return u_lower <= component && component < u_upper;
  }
  inline bool is_pressure(const unsigned int component) const
  {
    return p_lower <= component && component < p_upper;
  }

protected:
  ParameterReader<dim> param;

  QSimplex<dim>     quadrature;
  QSimplex<dim - 1> face_quadrature;

  parallel::fullydistributed::Triangulation<dim> triangulation;
  std::shared_ptr<Mapping<dim>>                  mapping;
  FESystem<dim>                                  fe;
  DoFHandler<dim>                                dof_handler;
  TimeHandler                                    time_handler;

  const ComponentMask velocity_mask;
  const ComponentMask pressure_mask;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  AffineConstraints<double> zero_constraints;
  AffineConstraints<double> nonzero_constraints;

  types::global_dof_index constrained_pressure_dof = numbers::invalid_dof_index;
  Point<dim>              constrained_pressure_support_point;

  LA::ParMatrixType              system_matrix;
  std::vector<LA::ParVectorType> previous_solutions;

  std::shared_ptr<Function<dim>> source_terms;
  std::shared_ptr<Function<dim>> exact_solution;

  TableHandler forces_table;
  TableHandler cylinder_position_table;
};

#endif