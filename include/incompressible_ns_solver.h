#ifndef INCOMPRESSIBLE_NS_SOLVER_H
#define INCOMPRESSIBLE_NS_SOLVER_H

#include <assembly/assembler.h>
#include <copy_data.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <linear_solver.h>
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
  using ScratchData = NavierStokesScratch::ScratchDataIncompressibleNS<dim>;
  using CopyData    = CopyDataBase<1>;
  using Assembler   = Assembly::AssemblerBase<ScratchData, CopyData>;

public:
  /**
   * Constructor
   */
  NSSolver(const ParameterReader<dim> &param);

  /**
   * Destructor
   */
  virtual ~NSSolver() = default;

  /**
   * Reset data specific to this solver (e.g., the preconditioner).
   */
  virtual void reset_solver_specific_data() override;

  /**
   * Create the scratch data structure for this solver.
   */
  virtual void create_scratch_data() override;

  /**
   * Create the volume and boundary assemblers for this solver.
   */
  virtual void setup_assemblers() override;

  /**
   * Create the homogeneous boundary conditions. If 'pressure null space' is
   * enabled, also create the constant pressure mode vector.
   */
  virtual void create_zero_constraints() override;

  /**
   * Create the matrix sparsity pattern, given the finite element spaces and
   * constraints.
   */
  virtual void create_sparsity_pattern() override;

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

  /**
   * Create and initialize the chosen preconditioner.
   */
  void create_preconditioner();

  /**
   * Assemble the scaled pressure mass matrix -Mp/nu of block_schur_data.
   * Used for the block Schur preconditioner.
   */
  void assemble_pressure_mass_matrix();

  /**
   * Assemble the diagonal of the velocity mass matrix, diag(Mu), of
   * block_schur_data, used in the Laplacian part A10 (c0 diag(Mu))^{-1} A01
   * of the Cahouet-Chabard block Schur preconditioner, and as the scaling Q
   * of the LSC preconditioner. Possible hanging nodes constraints are not
   * applied to this diagonal, whose entries are all positive.
   */
  void assemble_velocity_mass_diagonal();

  /**
   * Preconditioner update heuristic, returns true if the preconditioner should
   * be (re)built before the next linear solve. The preconditioner is rebuilt
   * - if it does not exist yet, or if the mesh or dof numbering has changed,
   * - for steady problems, at each linear solve, as the Jacobian matrix may
   *   change significantly between Newton iterations,
   * - for unsteady problems, the preconditioner is reused across Newton
   * iterations and time steps, unless:
   *   - the leading BDF coefficient changed (time step or BDF order),
   *   - the last GMRES solve did not converge,
   *   - at the first Newton iteration of a time step, the convergence of the
   *     first solve of the previous time step, in iterations per decade of
   *     residual reduction, has slowed down too much compared to the first
   *     solve of a time step with the current preconditioner.
   * @p first_solve_of_time_step is true for the first linear solve of a time
   * step.
   */
  bool should_rebuild_preconditioner(const bool first_solve_of_time_step) const;

  /**
   * Solve the linear system for a single nonlinear solver iteration.
   * GMRES is currently the only iterative solver supported for this solver.
   * The preconditioner is rebuilt according to should_rebuild_preconditioner().
   * If GMRES does not converge with a reused preconditioner, the preconditioner
   * is rebuilt and the system is solved again once.
   */
  virtual void solve_linear_system_iterative() override;

  /**
   * Create pressure_null_vector, the normalized constant pressure mode: 1 on
   * the pressure dofs which are not constrained, 0 elsewhere, normalized.
   * Created with the zero constraints.
   *
   * See attach_pressure_null_space() and finalize_rhs().
   */
  void create_pressure_null_vector();

  /**
   * Attach the constant pressure mode to the system matrix, as its null space
   * and the null space of its transpose, so that the iterative solver handles
   * the singular system (the right-hand side is projected onto the range of the
   * system matrix). See finalize_rhs() below for a more thorough explanation.
   *
   * @note This function is only called when 'pressure null space' is enabled,
   * which requires an iterative solver.
   */
  void attach_pressure_null_space();

  /**
   * Make the right-hand side b := -R(u) (i.e., the negative of the nonlinear
   * Newton residual) consistent with the singular Jacobian matrix.
   *
   * @note This function is called by the nonlinear solver after each RHS
   * assembly, but does nothing unless 'pressure null space' is enabled, which
   * requires an iterative solver.
   *
   * @note The argument below relies on the incompressible Navier-Stokes weak form on
   * a fixed mesh, with the normal velocity strongly constrained on the whole
   * boundary. It does not hold as is for the other solvers, e.g., with a moving
   * boundary, or with a Lagrange multiplier imposing the velocity weakly, where
   * the null spaces of J and J^T have other components.
   *
   * When solving the linear problem J * delta = b, a solution exists if b is in
   * range(J) (or equivalently if b is orthogonal to null(J^T)), and it is
   * unique if null(J) is only the zero vector. When the pressure is pinned at
   * one dof, the system matrix is nonsingular: null(J^T) is trivial and thus
   * orthogonal to every b, a solution always exists, and this solution is
   * unique.
   *
   * When 'pressure null space' is enabled, the normal velocity is prescribed on
   * the whole boundary, and no additional treatment is applied for the
   * pressure. As a result, the system matrix is singular, and the issues of
   * existence and uniqueness must be addressed. Indeed, the vector w, defined
   * to be equal to 1 on the pressure dofs which are not constrained and 0
   * elsewhere (e.g., at constrained hanging nodes), spans the null space of
   * both J and J^T, for different reasons:
   *
   * 1. J * w = 0: (J * w)_i is the derivative of the residual row b_i w.r.t. a
   * constant pressure p. The continuity rows don't depend on p (apart from the
   * PSPG terms, see below), and in the momentum rows, the pressure gradient
   * term reads
   *
   *       −∫Ω p (∇·φ_i) dx = −p ∫Ω ∇·φ_i dx = −p ∫∂Ω φ_i·n ds.
   *
   * Because the normal velocity is constrained on the whole boundary, the
   * unconstrained velocity shape functions φ_i have no normal trace on the
   * boundary, so ∫∂Ω φ_i·n ds = 0, and (J * w)_i = 0. In the stabilization
   * terms (SUPG-PSPG), pressure only appears through ∇p, which vanishes for a
   * constant p. As constants satisfy the hanging node constraints, w represents
   * a constant pressure in the condensed system, and the rows of the
   * constrained dofs only have a diagonal entry, in a column where w is zero.
   *
   * As a result, the solution (delta) is not unique:
   * remove_mean_pressure_from_solution() removes the mean pressure from the
   * converged vector to yield a unique solution.
   *
   * 2. J^T * w = 0: this is because (J^T w)_j is the derivative w.r.t. dof j of
   * the residual rows b_i weighted by w, here the continuity rows:
   *
   *     b_i = ∫Ω q_i (∇·u_h − f_p) dx + PSPG terms (tested with ∇q_i).
   *
   * Since the pressure test functions sum to one (Σ_i q_i = 1, also after
   * condensation of the hanging node constraints), and the PSPG terms vanish in
   * the sum (Σ_i ∇q_i = ∇1 = 0):
   *
   *     Σ_i b_i = ∫Ω (Σ_i q_i)(∇·u_h − f_p) dx
   *             = ∫Ω (∇·u_h − f_p) dx
   *             = ∫∂Ω u_h·n ds − ∫Ω f_p dx      (*)
   *
   * The derivative of (*) w.r.t. the pressure dofs is zero, and w.r.t. an
   * unconstrained velocity dof j it is ∫∂Ω φ_j·n ds = 0: the normal velocity
   * is constrained on the whole boundary, so the unconstrained velocity shape
   * functions have no normal trace there. The constrained dofs do not
   * contribute, as their column of J only has a diagonal entry, in a row where
   * w is zero.
   *
   * The issue is that b may not be orthogonal to w. Indeed, w^T b = Σ_i b_i is
   * (*): it vanishes for the exact solution, but the discrete boundary velocity
   * is an interpolation of the prescribed one, so that it is only zero up to
   * the interpolation error.
   *
   * As a result, when b is not orthogonal to w, there is no exact solution to
   * the linear system. Splitting b as
   *
   *     b = b_R + c * n,  with n := w / |w|, c := n^T * b, b_R in range(J),
   *
   * we have
   *
   *     ||J * delta - b||^2 = ||J * delta - b_R||^2 + c^2,
   *
   * and the linear residual can't be reduced below |c|, the norm of the
   * component of b in null(J^T). The Newton solver, which measures
   * ||b||, would then stagnate, possibly above its tolerance, although each
   * linear solve is exact, and run out of iterations.
   *
   * To solve this, two modifications are applied:
   *
   * a. Instead of solving the ill-posed system J * delta = b, the null space w
   * is passed to the iterative solver (as the null space of J^T), which then
   * solves J * delta = b_R. This is done by calling
   * attach_pressure_null_space() before solving the linear system.
   *
   * b. The present function, called by the Newton solver after each assembly of
   * the right-hand side, applies the projection of the linear solver:
   *
   *     b <- b - c * n = b_R.
   *
   * The Newton solver then measures convergence w.r.t. the residual
   * of the compatible problem, which its updates *do* reduce. As the removed
   * component is of the order of the interpolation error, this does not affect
   * the accuracy of the solution.
   *
   * TL;DR:
   *  1. J * w = 0: delta is not unique, so the mean pressure is removed after
   * the solve.
   *  2. J^T * w = 0: range(J) = w^⊥, so a solution exists only if w^T b = 0.
   * When it doesn't, the residual has a floor at |c|, which leads to (a) the
   * projection by PETSc and (b) the same projection in finalize_rhs().
   */
  virtual void finalize_rhs() override;

protected:
  /**
   * Get the descriptions (name and number of components) of the variables
   * solved for in the derived solvers, aside from velocity, pressure and mesh
   * position.
   */
  virtual std::vector<std::pair<std::string, unsigned int>>
  get_additional_variables_description() const override
  {
    return std::vector<std::pair<std::string, unsigned int>>();
  }

  /**
   * Return the FESystem used by this solver.
   */
  virtual const FESystem<dim> &get_fe_system() const override { return *fe; }

  /**
   * Return whether this solver uses hp capabilities.
   */
  virtual bool uses_hp_capabilities() const override { return false; };

protected:
  std::unique_ptr<FESystem<dim>> fe;

  // Non-owning pointer to base class fixed_mapping, used for convenience.
  const Mapping<dim> *mapping;

  std::unique_ptr<ScratchData> scratch_data;

  std::vector<std::unique_ptr<Assembler>> assemblers;

  /**
   * PETSc-based preconditioner (e.g. ILU or AMG through BoomerAMG)
   */
  std::unique_ptr<PETScWrappers::PreconditionBase> preconditioner;

  /**
   * Data for the block Schur preconditioners (fieldsplit_schur_cc and
   * fieldsplit_schur_lsc), which only depend on the mesh, the constraints and
   * the viscosity.
   */
  struct BlockSchurPreconditionerData
  {
    // Locally owned velocity and pressure dofs, to define the field split.
    IndexSet velocity_dofs;
    IndexSet pressure_dofs;

    // Scaled pressure mass matrix -Mp/nu, which approximates the Schur
    // complement of the full system matrix. It has the size of the system
    // matrix, but only its pressure rows and columns are filled.
    LA::ParMatrixType pressure_mass_matrix;

    // Diagonal of the velocity mass matrix, diag(Mu). With Cahouet-Chabard
    // (unsteady problems only), scaled by the leading BDF coefficient c0 at
    // each preconditioner setup, it gives the diagonal of the time derivative
    // term of the momentum block. With LSC, it is the scaling Q.
    LA::ParVectorType velocity_mass_diagonal;

    // Whether the matrices above were assembled since the last setup of the
    // sparsity pattern.
    bool is_assembled = false;
  } block_schur_data;

  /**
   * Data used by should_rebuild_preconditioner().
   */
  struct PreconditionerRebuildData
  {
    // Leading BDF coefficient when the preconditioner was last created
    double bdf_c0_at_setup = 0.;

    // GMRES iterations per decade of residual reduction of the first solve of
    // a time step with the current preconditioner (0 if there was none yet)
    double iterations_per_decade_after_setup = 0.;

    // GMRES iterations per decade of residual reduction of the latest first
    // solve of a time step
    double iterations_per_decade_at_last_first_solve = 0.;

    // Time iteration of the last linear solve
    unsigned int time_iteration_at_last_solve = numbers::invalid_unsigned_int;

    // Outcome of the last linear solve
    LinearSolvers::IterativeSolveInfo last_solve_info = {0, true, 0.};
  } rebuild_data;

  // Normalized constant pressure mode
  LA::ParVectorType pressure_null_vector;

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
      FunctionTime<double>::set_time(new_time);
      mms.set_time(new_time);
    }

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      Assert(component < n_components, ExcMessage("Component mismatch"));
      if (component >= u_lower && component < u_lower + dim)
        return mms.exact_velocity->value(p, component - u_lower);
      Assert(component == p_lower, ExcMessage("Component mismatch"));
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
      if (component >= u_lower && component < u_lower + dim)
        return mms.exact_velocity->gradient(p, component - u_lower);
      Assert(component == p_lower, ExcMessage("Component mismatch"));
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
      FunctionTime<double>::set_time(new_time);
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
