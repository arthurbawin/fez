#ifndef LINEAR_SOLVER_H
#define LINEAR_SOLVER_H

#include <deal.II/base/index_set.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/solver_gmres.h>
#include <generic_solver.h>
#include <mumps_solver.h>
#include <parameters.h>
#include <types.h>

namespace LinearSolvers
{
  using namespace dealii;

  /**
   * Solve the linear system with MUMPS.
   *
   * If @p mumps_solver is provided, this solver is used, which avoids recomputing
   * the symbolic factorization step if the sparsity pattern of the matrix has
   * not changed since the last solve. The PETScWrappers::SparseDirectMUMPSReuse
   * class (implemented in mumps_solver.h) also checks the return code of MUMPS,
   * and throws an error on a non-success instead of returning a solution vector
   * containing nans.
   *
   * If @p mumps_solver is null, a new PETScWrappers::SparseDirectMUMPS is created
   * and used only for this solve.
   */
  void
  solve_mumps(GenericSolver<LA::ParVectorType>      *solver,
              const Parameters::LinearSolver        &linear_solver_param,
              LA::ParMatrixType                     &system_matrix,
              const IndexSet                        &locally_owned_dofs,
              const AffineConstraints<double>       &zero_constraints,
              PETScWrappers::SparseDirectMUMPSReuse *mumps_solver = nullptr);

  /**
   * Summary of an iterative linear solve, used to determine if the preconditioner should be rebuilt.
   */
  struct IterativeSolveInfo
  {
    unsigned int n_iterations;
    bool         converged;
  };

  /**
   * Solve the linear system using GMRES.
   * If VectorType is a PETSc parallel vector, then PETSc's implementation of
   * GMRES is used. If VectorType is a parallel block vector, deal.II's
   * implementation is used instead (TODO, not yet implemented).
   *
   * This function is templatized on PreconditionerType, to use preconditioners
   * that are either derived from PETScWrappers::PreconditionBase, or
   * provided by custom class templates or linear operators, such as block Schur
   * complement preconditioners (see deal.II's step-31.cc).
   *
   * Return the number of iterations and whether GMRES converged. If it did not,
   * the last iterate is still used as the Newton update.
   */
  template <typename VectorType,
            typename MatrixType,
            typename PreconditionerType>
  IterativeSolveInfo
  solve_gmres(GenericSolver<VectorType>           *solver,
              const Parameters::LinearSolver      &linear_solver_param,
              MatrixType                          &system_matrix,
              const IndexSet                      &locally_owned_dofs,
              const AffineConstraints<double>     &zero_constraints,
              std::unique_ptr<PreconditionerType> &preconditioner);

  /**
   * Solve the linear system using the conjugate gradient method.
   */
  void
  solve_cg(GenericSolver<LA::ParVectorType> *solver,
           const Parameters::LinearSolver   &linear_solver_param,
           LA::ParMatrixType                &system_matrix,
           const IndexSet                   &locally_owned_dofs,
           const AffineConstraints<double>  &zero_constraints,
           std::unique_ptr<PETScWrappers::PreconditionBase> &preconditioner);

  /**
   * Create and initialize an off-the-shelf preconditioner, available in the
   * PETScWrappers namespace. This function does not initialize an AMG
   * preconditioner, it should be created in the dedicated solver with the
   * proper parameters instead.
   */
  void create_preconditioner(
    const Parameters::LinearSolver                   &linear_solver_param,
    const PETScWrappers::MatrixBase                  &system_matrix,
    std::unique_ptr<PETScWrappers::PreconditionBase> &preconditioner);
} // namespace LinearSolvers

/* ---------------- Template functions ----------------- */

namespace LinearSolvers
{
  template <typename VectorType,
            typename MatrixType,
            typename PreconditionerType>
  IterativeSolveInfo
  solve_gmres(GenericSolver<VectorType>           *solver,
              const Parameters::LinearSolver      &linear_solver_param,
              MatrixType                          &system_matrix,
              const IndexSet                      &locally_owned_dofs,
              const AffineConstraints<double>     &zero_constraints,
              std::unique_ptr<PreconditionerType> &preconditioner)
  {
    if (!preconditioner)
    {
      if constexpr (std::is_base_of_v<PETScWrappers::PreconditionBase,
                                      std::decay_t<PreconditionerType>>)
      {
        solver->computing_timer.enter_subsection("Set up preconditioner");
        create_preconditioner(linear_solver_param,
                              system_matrix,
                              preconditioner);
        solver->computing_timer.leave_subsection();
      }
      else
        AssertThrow(false,
                    ExcMessage(
                      "Off-the-shelf preconditioner initialization is only "
                      "possible if the preconditioner is derived from a "
                      "PETScWrappers::PreconditionBase."));
    }

    VectorType &newton_update = solver->get_newton_update();
    VectorType &system_rhs    = solver->get_system_rhs();

    VectorType completely_distributed_solution(locally_owned_dofs,
                                               solver->mpi_communicator);

    // Stop when residual is below the absolute tolerance, or below the
    // relative tolerance times the initial residual. Since the initial guess is
    // zero, the initial residual is the right-hand side (preconditioned, if
    // left preconditioning is used).
    ReductionControl solver_control(linear_solver_param.max_iterations,
                                    linear_solver_param.absolute_tolerance,
                                    linear_solver_param.relative_tolerance);

    bool converged = true;

    solver->computing_timer.enter_subsection("Solve GMRES");
    try
    {
      if constexpr (std::is_same_v<VectorType, LA::ParVectorType>)
      {
        // PETSc's GMRES:
        PETScWrappers::SolverGMRES gmres_solver(
          solver_control,
          PETScWrappers::SolverGMRES::AdditionalData(
            linear_solver_param.max_krylov_vectors,
            linear_solver_param.right_preconditioning));
        gmres_solver.solve(system_matrix,
                           completely_distributed_solution,
                           system_rhs,
                           *preconditioner);
      }
      else
      {
        // deal.II's GMRES, to handle e.g. block vectors/matrix with a block
        // Schur complement (TODO)
        SolverGMRES<VectorType> gmres_solver(
          solver_control,
          typename SolverGMRES<VectorType>::AdditionalData(
            linear_solver_param.max_krylov_vectors,
            linear_solver_param.right_preconditioning));
        gmres_solver.solve(system_matrix,
                           completely_distributed_solution,
                           system_rhs,
                           *preconditioner);
      }
    }
    catch (const SolverControl::NoConvergence &e)
    {
      converged = false;

      // Use the last iterate as an inexact Newton update, and let the
      // nonlinear solver decide whether to stop.
      solver->pcout << "Warning: GMRES did not converge after " << e.last_step
                    << " iterations (residual = " << e.last_residual << ")"
                    << std::endl;
    }
    solver->computing_timer.leave_subsection("Solve GMRES");

    if (linear_solver_param.verbosity == Parameters::Verbosity::verbose)
      solver->pcout << solver_control.last_step()
                    << " GMRES iterations needed to obtain convergence"
                    << std::endl;

    newton_update = completely_distributed_solution;
    zero_constraints.distribute(newton_update);

    return {solver_control.last_step(), converged};
  }
} // namespace LinearSolvers

#endif
