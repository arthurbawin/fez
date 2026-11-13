#ifndef NEWTON_SOLVER_H
#define NEWTON_SOLVER_H

/**
 * This structure and solve function are borrowed from the Lethe project.
 */

#include <nonlinear_solver.h>
#include <time_handler.h>

#include <cmath>

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
  void solve(const TimeHandler &time_handler) override
  {
    bool         solution_found = false;
    unsigned int iter           = 0;
    // double       norm_increment = 0;
    double norm_residual = 0;
    double last_residual;
    bool   recompute_rhs          = true;
    bool   assemble               = false;
    bool   preconditioner_rebuilt = false;

    auto       solver = this->solver;
    const bool verbose =
      this->param.verbosity == Parameters::Verbosity::verbose;

    auto &present_solution = solver->get_present_solution();

    // Throw on failure for these cases:
    // - simulation is steady-state
    // - time adaptation is disabled
    // - this step or the previous was a BDF starting step
    const bool throw_on_failure =
      time_handler.is_steady() ||
      !(solver->get_time_parameters().adaptation.enable) ||
      time_handler.is_starting_step() ||
      time_handler.last_step_was_starting_step();

    solver->evaluation_point = present_solution;

    while (true)
    {
      // Assemble residual and check if tolerance is reached
      // Linfty norm does not scale with the number of RHS entries
      if (recompute_rhs)
      {
        solver->assemble_rhs();
        norm_residual = solver->system_rhs.l2_norm();
      }

      if (verbose)
      {
        solver->pcout
          << "Newton iter. " << std::setw(2) << iter << ": " << std::scientific
          << std::setprecision(8)
          // << "incr. = "  << norm_increment << " "
          << "         nonlinear residual = "
          << norm_residual
          // Print (M) if the matrix was assembled, and (P) if the
          // preconditioner was rebuilt, to obtain this residual norm
          << (assemble || preconditioner_rebuilt ?
                std::string("\t(") + (assemble ? "M" : "") +
                  (preconditioner_rebuilt ? "P" : "") + ")" :
                "")
          << std::endl;
      }

      // Abort if residual too high or not finite (e.g. NaN from inverted mesh)
      if (!std::isfinite(norm_residual) ||
          (iter > 0 && norm_residual > this->param.divergence_tolerance))
      {
        if (throw_on_failure)
          throw std::runtime_error("Nonlinear solver diverged");
        else
          break;
      }

      if (iter == 0)
        last_residual = norm_residual;

      if (norm_residual <= this->param.tolerance)
      {
        if (verbose)
          solver->pcout
            << "Stopping because residual is below prescribed tolerance ("
            << std::setprecision(2) << this->param.tolerance << ")"
            << std::endl;
        solution_found = true;
        break;
      }

      // Stop if the maximum number of Newton iterations is reached
      if (iter >= this->param.max_iterations)
      {
        if (verbose)
          solver->pcout << "Stopping because the maximum number of iterations ("
                        << this->param.max_iterations << ") is reached"
                        << std::endl;
        break;
      }

      // Assemble matrix and solve
      assemble = this->reassembly_heuristic(norm_residual, last_residual);
      if (assemble)
        solver->assemble_matrix();
      solver->solve_linear_system();
      preconditioner_rebuilt = solver->preconditioner_rebuilt_at_last_solve;
      // norm_increment = solver->newton_update.linfty_norm();

      if (this->param.enable_line_search)
      {
        // Fraction of the residual decrease predicted by the linearization
        // that a step must achieve to be accepted (Armijo condition)
        constexpr double armijo_coeff = 1e-4;
        // Smallest step length tried before the line search fails
        // This is 1 / 2^10, so 11 attempts total.
        constexpr double min_alpha = 1. / 1024.;

        double norm_ls_residual;
        bool   step_accepted = false;
        last_residual        = norm_residual;

        for (double alpha = 1.; alpha >= min_alpha; alpha /= 2.)
        {
          // Compute NL(u + alpha * du) and check if residual decreases
          solver->local_evaluation_point = present_solution;
          solver->local_evaluation_point.add(alpha, solver->newton_update);
          solver->distribute_nonzero_constraints();
          solver->evaluation_point = solver->local_evaluation_point;
          solver->assemble_rhs(); // NL(u + alpha * du)
          norm_ls_residual = solver->system_rhs.l2_norm();

          // Accept step if the residual is below tolerance, or if the Armijo
          // sufficient decrease condition is satisfied: the residual decreased
          // by at least a fraction armijo_coeff of the decrease
          // alpha * norm_residual predicted by the linearization.
          // Reject a non-finite (e.g., with NaNs) residual, and try again with
          // a smaller alpha.
          const bool accepted =
            std::isfinite(norm_ls_residual) &&
            (norm_ls_residual <= this->param.tolerance ||
             norm_ls_residual <= (1. - armijo_coeff * alpha) * norm_residual);

          // An accepted full step is reported by the next Newton iteration
          // summary. Otherwise the same residual is printed twice, which is
          // just noise.
          if (verbose && !(accepted && alpha == 1.))
          {
            solver->pcout << "\tLine search with alpha = " << std::fixed
                          << std::setprecision(3) << alpha << std::scientific
                          << std::setprecision(8)
                          << " : res = " << norm_ls_residual << std::endl;
          }

          // Exit line search and continue Newton iterations
          if (accepted)
          {
            step_accepted = true;
            break;
          }
        }

        if (!step_accepted)
        {
          // No step length down to min_alpha decreases the residual enough,
          // e.g., because the initial guess is too far from the solution.
          // Restore the last iterate and exit with nonconverged solution.
          if (verbose)
            solver->pcout << "\tLine search failed: no step length satisfies "
                             "the Armijo condition"
                          << std::endl;
          solver->evaluation_point = present_solution;
          if (throw_on_failure)
            throw std::runtime_error("Nonlinear solver line search failed");
          else
            break;
        }

        // RHS was last assembled at the accepted solution: reuse its norm
        // instead of reassembling.
        recompute_rhs = false;
        norm_residual = norm_ls_residual;

        // Update present solution for the next line search
        present_solution = solver->evaluation_point;
      }
      else
      {
        // Line search disabled:
        // Increment solution and go to next iteration
        solver->local_evaluation_point = solver->evaluation_point;
        solver->local_evaluation_point.add(1., solver->newton_update);
        solver->distribute_nonzero_constraints();
        solver->evaluation_point = solver->local_evaluation_point;
        last_residual            = norm_residual;
      }

      ++iter;
    }

    // Update present solution
    present_solution = solver->evaluation_point;

    if (!solution_found && throw_on_failure)
      throw std::runtime_error("Nonlinear solver did not converge");

    time_handler.set_last_nonlinear_solve_status(solution_found);
  }

private:
  /**
   * An heuristic to determine whether the matrix should be assembled
   */
  bool reassembly_heuristic(const double current_residual_norm,
                            const double previous_residual_norm) const
  {
    const auto time_param = this->solver->get_time_parameters();

    // Always (re)assemble if solving for steady state
    if (time_param.is_steady())
      return true;

    // Assemble if, during the last solve, the residual norm did not decrease
    // enough w.r.t. the previous residual norm
    if (current_residual_norm >
        this->param.reassembly_decrease_tol * previous_residual_norm)
      return true;

    return false;
  }
};

#endif
