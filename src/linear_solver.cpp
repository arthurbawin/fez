
#include <deal.II/base/timer.h>
#include <generic_solver.h>
#include <linear_solver.h>

void solve_linear_system_direct(
  GenericSolver<LA::ParVectorType> *solver,
  const Parameters::LinearSolver   &linear_solver_param,
  LA::ParMatrixType                &system_matrix,
  const IndexSet                   &locally_owned_dofs,
  const AffineConstraints<double>  &zero_constraints)
{
  TimerOutput::Scope t(solver->computing_timer, "Solve direct");

  const bool verbose =
      linear_solver_param.verbosity == Parameters::Verbosity::verbose;

  if(verbose)
    solver->pcout << "Entering direct solver" << std::endl;

  LA::ParVectorType &newton_update = solver->get_newton_update();
  LA::ParVectorType &system_rhs    = solver->get_system_rhs();

  LA::ParVectorType completely_distributed_solution(locally_owned_dofs,
                                                    solver->mpi_communicator);

#if defined(FEZ_WITH_PETSC)
  // Solve with MUMPS
  SolverControl                    solver_control;
  PETScWrappers::SparseDirectMUMPS linear_solver(solver_control);
#elif defined(FEZ_WITH_TRILINOS)
  // Solve with MUMPS through Amesos
  TrilinosWrappers::SolverDirect::AdditionalData data(true, "Amesos_Mumps");
  TrilinosWrappers::SolverDirect                 linear_solver(data);
#endif
  
  linear_solver.solve(system_matrix,
                      completely_distributed_solution,
                      system_rhs);

  newton_update = completely_distributed_solution;
  zero_constraints.distribute(newton_update);

  if(verbose)
    solver->pcout << "Leaving  direct solver" << std::endl;
}

void solve_linear_system_direct(
  GenericSolver<LA::ParVectorType> *solver,
  const Parameters::LinearSolver   &linear_solver_param,
  LA::ParMatrixType                &system_matrix,
  const IndexSet                   &locally_owned_dofs,
  const AffineConstraints<double>  &zero_constraints,
  PETScWrappers::SparseDirectMUMPSReuse &direct_solver)
{
  TimerOutput::Scope t(solver->computing_timer, "Solve direct");
  solver->pcout << "Entering direct solver" << std::endl;

  LA::ParVectorType &newton_update = solver->get_newton_update();
  LA::ParVectorType &system_rhs    = solver->get_system_rhs();

  LA::ParVectorType completely_distributed_solution(locally_owned_dofs,
                                                    solver->mpi_communicator);

#if defined(FEZ_WITH_PETSC)
  direct_solver.solve(system_matrix,
                      completely_distributed_solution,
                      system_rhs);
#elif defined(FEZ_WITH_TRILINOS)
  // Solve with MUMPS through Amesos
  TrilinosWrappers::SolverDirect::AdditionalData data(true, "Amesos_Mumps");
  TrilinosWrappers::SolverDirect                 linear_solver(data);

  linear_solver.solve(system_matrix,
                      completely_distributed_solution,
                      system_rhs);
#endif

  newton_update = completely_distributed_solution;
  zero_constraints.distribute(newton_update);
  solver->pcout << "Leaving  direct solver" << std::endl;
}

void solve_linear_system_iterative(
  GenericSolver<LA::ParVectorType> *solver,
  const Parameters::LinearSolver   &linear_solver_param,
  LA::ParMatrixType                &system_matrix,
  const IndexSet                   &locally_owned_dofs,
  const AffineConstraints<double>  &zero_constraints)
{
  TimerOutput::Scope t(solver->computing_timer, "Solve iterative");

  LA::ParVectorType &newton_update = solver->get_newton_update();
  LA::ParVectorType &system_rhs    = solver->get_system_rhs();

  LA::ParVectorType completely_distributed_solution(locally_owned_dofs,
                                                    solver->mpi_communicator);


  SolverControl   solver_control(linear_solver_param.max_iterations,
                                 linear_solver_param.tolerance);
  LA::SolverGMRES linear_solver(solver_control);

  #if defined(FEZ_WITH_PETSC)
    LA::MPI::PreconditionAMG::AdditionalData data;
    AssertThrow(false, ExcMessage("Configure PETSc with Hypre to use BoomerAMG"));
  #else
      const bool elliptic = false;
      const bool higher_order_elements = true;
      const unsigned int n_cycles = 1;
      const bool w_cycle  = false;
      const double aggregation_threshold = 1e-10;

      LA::MPI::PreconditionAMG::AdditionalData data(
        elliptic,
        higher_order_elements,
        n_cycles,
        w_cycle,
        aggregation_threshold);
  #endif
  LA::MPI::PreconditionAMG preconditioner;
  preconditioner.initialize(system_matrix, data);

  linear_solver.solve(system_matrix,
                      completely_distributed_solution,
                      system_rhs,
                      preconditioner);

  solver->pcout << "   Solved in " << solver_control.last_step() << " iterations."
        << std::endl;

  newton_update = completely_distributed_solution;
  zero_constraints.distribute(newton_update);
}