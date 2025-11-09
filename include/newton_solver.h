#ifndef NEWTON_SOLVER_H
#define NEWTON_SOLVER_H

/**
 * This structure and solve function are borrowed from the Lethe project.
 */

#include <nonlinear_solver.h>

/**
 * Generic Newton nonlinear solver.
 */
template <typename VectorType>
class NewtonSolver : public NonLinearSolver<VectorType>
{
public:
  /**
   * Constructor
   */
  NewtonSolver(const Parameters::NonLinearSolver &param,
               GenericSolver<VectorType>         *solver)
    : NonLinearSolver<VectorType>(param, solver)
  {}

public:
  /**
   * Solve the nonlinear problem with Newton-Raphson's method.
   */
  void solve(const bool first_step) override
  {
    bool         stop = false;
    unsigned int iter = 0;
    // double       norm_increment = 0;
    double norm_residual = 0;
    double last_residual;
    bool   recompute_rhs = true;

    auto       solver = this->solver;
    const bool verbose =
      this->param.verbosity == Parameters::Verbosity::verbose;

    while (!stop)
    {
      solver->evaluation_point = solver->present_solution;

      // Assemble residual and check if tolerance is reached
      // Linfty norm does not scale with the number of RHS entries
      if (recompute_rhs)
      {
        solver->assemble_rhs();
        norm_residual = solver->system_rhs.l2_norm();
      }

      if (verbose)
      {
        solver->pcout << "Newton iter. " << std::setw(2) << iter << ": "
                      << std::scientific
                      << std::setprecision(8)
                      // << "incr. = "  << norm_increment << " "
                      << "         nonlinear residual = " << norm_residual
                      << std::endl;
      }

      // Abort if residual is too high
      if (iter > 0 && norm_residual > this->param.divergence_tolerance)
      {
        // throw std::runtime_error("Nonlinear solver diverged: " +
        // std::to_string(norm_residual) " > divergence tolerance = " +
        // std::to_string(this->param.divergence_tolerance));
        throw std::runtime_error("Nonlinear solver diverged");
      }

      if (iter == 0)
        last_residual = norm_residual;

      if (norm_residual <= this->param.tolerance)
      {
        solver->pcout
          << "Stopping because residual is below prescribed tolerance ("
          << std::setprecision(2) << this->param.tolerance << ")" << std::endl;
        break;
      }

      // Assemble matrix and solve
      solver->assemble_matrix();
      solver->solve_linear_system(first_step);
      // norm_increment = solver->newton_update.linfty_norm();

      if (this->param.enable_line_search)
      {
        double       norm_ls_residual;
        unsigned int ls_iter = 0;

        for (double alpha = 1.; alpha > 0.1; alpha /= 2., ++ls_iter)
        {
          // Compute NL(u + alpha * du) and check if residual decreases
          solver->local_evaluation_point = solver->present_solution;
          solver->local_evaluation_point.add(alpha, solver->newton_update);
          solver->distribute_nonzero_constraints();
          solver->evaluation_point = solver->local_evaluation_point;
          solver->assemble_rhs(); // NL(u + alpha * du)
          norm_ls_residual = solver->system_rhs.l2_norm();

          solver->pcout << "\tLine search with alpha = " << std::fixed
                        << std::setprecision(3) << alpha << std::scientific
                        << std::setprecision(8)
                        << " : res = " << norm_ls_residual << std::endl;

          // Exit if next residual is below tolerance
          if (norm_ls_residual <= this->param.tolerance)
          {
            solver->pcout << "Stopping because residual is below "
                             "prescribed tolerance ("
                          << std::setprecision(2) << this->param.tolerance
                          << ")" << std::endl;
            last_residual = norm_ls_residual;
            stop          = true;
            break;
          }

          // Accept step if an Armijo-like sufficient condition is satisfied,
          // exit line search and continue Newton iterations.
          if (norm_ls_residual <= 0.1 * last_residual)
          {
            // RHS was just computed, no need to recompute
            recompute_rhs = false;
            last_residual = norm_ls_residual;
            norm_residual = norm_ls_residual;
            break;
          }

          // If residual increased, backtrack and exit
          // Do not reject first iteration
          if (norm_ls_residual > last_residual && ls_iter > 0)
          {
            solver->pcout << "\tRejecting last step and backtracking"
                          << std::endl;
            // RHS will need to be recomputed for backtracked solution
            recompute_rhs = true;
            alpha *= 2.;
            solver->local_evaluation_point = solver->present_solution;
            solver->local_evaluation_point.add(alpha, solver->newton_update);
            solver->distribute_nonzero_constraints();
            solver->evaluation_point = solver->local_evaluation_point;
            break;
          }

          last_residual = norm_ls_residual;
        }
      }
      else
      {
        // Increment solution and go to next iteration
        solver->local_evaluation_point = solver->present_solution;
        solver->local_evaluation_point.add(1., solver->newton_update);
        solver->distribute_nonzero_constraints();
        solver->evaluation_point = solver->local_evaluation_point;
      }

      solver->present_solution = solver->evaluation_point;
      ++iter;

      if(iter > this->param.max_iterations)
        stop = true;
    }

    if(iter > this->param.max_iterations && norm_residual > this->param.tolerance)
    {
      throw std::runtime_error("Nonlinear solver did not converge");
    }
  }
};

#endif