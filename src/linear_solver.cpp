
#include <deal.II/base/timer.h>
#include <deal.II/lac/petsc_compatibility.h>
#include <generic_solver.h>
#include <linear_solver.h>

#if defined(DEAL_II_WITH_PETSC)
#  include <petscmat.h>
#endif

namespace LinearSolvers
{
  void solve_mumps(GenericSolver<LA::ParVectorType>      *solver,
                   const Parameters::LinearSolver        &linear_solver_param,
                   LA::ParMatrixType                     &system_matrix,
                   const IndexSet                        &locally_owned_dofs,
                   const AffineConstraints<double>       &zero_constraints,
                   PETScWrappers::SparseDirectMUMPSReuse *mumps_solver)
  {
    TimerOutput::Scope t(solver->computing_timer, "Solve direct");

    const bool verbose =
      linear_solver_param.verbosity == Parameters::Verbosity::verbose;

    if (verbose)
      solver->pcout << "Solving with MUMPS..." << std::endl;

    LA::ParVectorType &newton_update = solver->get_newton_update();
    LA::ParVectorType &system_rhs    = solver->get_system_rhs();

    LA::ParVectorType completely_distributed_solution(locally_owned_dofs,
                                                      solver->mpi_communicator);

    if (mumps_solver)
    {
      // Use existing solver
      mumps_solver->solve(system_matrix,
                          completely_distributed_solution,
                          system_rhs);
    }
    else
    {
      // Create a SparseDirectMUMPS solver and solve
      SolverControl                    solver_control;
      PETScWrappers::SparseDirectMUMPS linear_solver(solver_control);
      linear_solver.solve(system_matrix,
                          completely_distributed_solution,
                          system_rhs);
    }

    newton_update = completely_distributed_solution;
    zero_constraints.distribute(newton_update);

    if (verbose)
      solver->pcout << "Done" << std::endl;
  }

  void
  solve_cg(GenericSolver<LA::ParVectorType> *solver,
           const Parameters::LinearSolver   &linear_solver_param,
           LA::ParMatrixType                &system_matrix,
           const IndexSet                   &locally_owned_dofs,
           const AffineConstraints<double>  &zero_constraints,
           std::unique_ptr<PETScWrappers::PreconditionBase> &preconditioner)
  {
    TimerOutput::Scope t(solver->computing_timer, "Solve CG");

    const bool verbose =
      linear_solver_param.verbosity == Parameters::Verbosity::verbose;

    if (!preconditioner)
    {
      solver->computing_timer.enter_subsection("Set up preconditioner");
      create_preconditioner(linear_solver_param, system_matrix, preconditioner);
      solver->computing_timer.leave_subsection();
    }

    LA::ParVectorType &newton_update = solver->get_newton_update();
    LA::ParVectorType &system_rhs    = solver->get_system_rhs();

    LA::ParVectorType completely_distributed_solution(locally_owned_dofs,
                                                      solver->mpi_communicator);

    SolverControl           solver_control(linear_solver_param.max_iterations,
                                 linear_solver_param.tolerance);
    PETScWrappers::SolverCG cg_solver(solver_control);

    cg_solver.solve(system_matrix,
                    completely_distributed_solution,
                    system_rhs,
                    *preconditioner);

    if (verbose)
      solver->pcout << solver_control.last_step()
                    << " CG iterations needed to obtain convergence"
                    << std::endl;

    newton_update = completely_distributed_solution;
    zero_constraints.distribute(newton_update);
  }

  void create_preconditioner(
    const Parameters::LinearSolver                   &linear_solver_param,
    const PETScWrappers::MatrixBase                  &system_matrix,
    std::unique_ptr<PETScWrappers::PreconditionBase> &preconditioner)
  {
    switch (linear_solver_param.preconditioner)
    {
      case Parameters::LinearSolver::PreconditionerType::none:
        preconditioner =
          std::make_unique<PETScWrappers::PreconditionNone>(system_matrix);
        break;
      case Parameters::LinearSolver::PreconditionerType::ilu:
      {
        AssertThrow(
          Utilities::MPI::n_mpi_processes(
            system_matrix.get_mpi_communicator()) == 1,
          ExcMessage(
            "PETSc's ILU preconditioner only has a serial implementation."));
        PETScWrappers::PreconditionILU::AdditionalData data(
          linear_solver_param.ilu_fill_level);
        preconditioner =
          std::make_unique<PETScWrappers::PreconditionILU>(system_matrix, data);
      }
      break;
      case Parameters::LinearSolver::PreconditionerType::block_jacobi:
        preconditioner =
          std::make_unique<PETScWrappers::PreconditionBlockJacobi>(
            system_matrix);
        break;
      case Parameters::LinearSolver::PreconditionerType::amg:
        // Let each dedicated solver initialize the AMG preconditioner with its
        // proper parameters
        AssertThrow(false,
                    ExcMessage("AMG preconditioner should be initialized in "
                               "the relevant dedicated solver."));
      default:
        DEAL_II_NOT_IMPLEMENTED();
    }
  }
} // namespace LinearSolvers
