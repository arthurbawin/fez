
#include <deal.II/base/timer.h>
#include <generic_solver.h>
#include <linear_direct_solver.h>

void solve_linear_system_direct(
  GenericSolver<LA::ParVectorType> *solver,
  LA::ParMatrixType                &system_matrix,
  const IndexSet                   &locally_owned_dofs,
  const AffineConstraints<double>  &zero_constraints)
{
  TimerOutput::Scope t(solver->computing_timer, "Solve direct");

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
}