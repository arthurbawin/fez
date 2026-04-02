#ifndef NEWTON_SOLVER_H
#define NEWTON_SOLVER_H

#include <cmath>

/**
 * This structure and solve function are borrowed from the Lethe project.
 */

#include <nonlinear_solver.h>
#include <time_handler.h>

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
    bool         stop           = false;
    unsigned int iter           = 0;
    // double       norm_increment = 0;
    double norm_residual = 0.0;
    double last_residual = 0.0;
    bool   recompute_rhs = true;
    bool   assemble      = false;

    auto       solver = this->solver;
    const bool verbose =
      this->param.verbosity == Parameters::Verbosity::verbose;

    // Throw on failure for these cases:
    // - simulation is steady-state
    // - time adaptation is disabled
    // - this step or the previous was a BDF starting step
    const bool throw_on_failure =
      time_handler.is_steady() ||
      !(solver->get_time_parameters().adaptation.enable) ||
      time_handler.is_starting_step() ||
      time_handler.last_step_was_starting_step();

    solver->evaluation_point = solver->present_solution;

    while (!stop)
    {
      // Assemble residual and check if tolerance is reached
      // Linfty norm does not scale with the number of RHS entries
      if (recompute_rhs)
      {
        solver->assemble_rhs();
        norm_residual = solver->system_rhs.l2_norm();

        if (!std::isfinite(norm_residual))
        {
          stop = true;
          break;
        }
      }

      if (verbose)
      {
        solver->pcout
          << "Newton iter. " << std::setw(2) << iter << ": " << std::scientific
          << std::setprecision(8)
          // << "incr. = "  << norm_increment << " "
          << "         nonlinear residual = "
          << norm_residual
          // Print (M) if the matrix was assembled to obtain this residual norm
          << (assemble ? "\t(M)" : "") << std::endl;
      }

      // Abort if residual is too high
      if (iter > 0 && norm_residual > this->param.divergence_tolerance)
      {
        // throw std::runtime_error("Nonlinear solver diverged: " +
        // std::to_string(norm_residual) " > divergence tolerance = " +
        // std::to_string(this->param.divergence_tolerance));
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

      // Assemble matrix and solve
      assemble = this->reassembly_heuristic(norm_residual, last_residual);
      if (assemble)
        solver->assemble_matrix();
      solver->solve_linear_system();
      // norm_increment = solver->newton_update.linfty_norm();

      if (this->param.enable_line_search)
      {
        double norm_ls_residual = norm_residual;
        double best_ls_residual = norm_residual;
        double best_alpha       = 0.;
        bool   accepted_step    = false;
        last_residual           = norm_residual;
        unsigned int ls_iter    = 0;

        for (double alpha = 1.; alpha > 0.1; alpha /= 2., ++ls_iter)
        {
          // Compute NL(u + alpha * du) and check if residual decreases
          solver->local_evaluation_point = solver->present_solution;
          solver->local_evaluation_point.add(alpha, solver->newton_update);
          solver->distribute_nonzero_constraints();
          solver->evaluation_point = solver->local_evaluation_point;
          solver->assemble_rhs(); // NL(u + alpha * du)
          norm_ls_residual = solver->system_rhs.l2_norm();

          if (verbose)
          {
            solver->pcout << "\tLine search with alpha = " << std::fixed
                          << std::setprecision(3) << alpha << std::scientific
                          << std::setprecision(8)
                          << " : res = " << norm_ls_residual << std::endl;
          }

          // Reject non-finite residuals and keep searching with a smaller step.
          if (!std::isfinite(norm_ls_residual))
            continue;

          if (norm_ls_residual < best_ls_residual)
          {
            best_ls_residual = norm_ls_residual;
            best_alpha       = alpha;
          }

          // Exit if next residual is below tolerance
          if (norm_ls_residual <= this->param.tolerance)
          {
            if (verbose)
              solver->pcout << "Stopping because residual is below "
                               "prescribed tolerance ("
                            << std::setprecision(2) << this->param.tolerance
                            << ")" << std::endl;
            recompute_rhs  = false;
            norm_residual  = norm_ls_residual;
            accepted_step  = true;
            stop           = true;
            solution_found = true;
            break;
          }

          // Accept step if an Armijo-like sufficient condition is satisfied,
          // exit line search and continue Newton iterations.
          if (norm_ls_residual <= 0.1 * norm_residual)
          {
            // RHS was just computed, no need to recompute
            recompute_rhs = false;
            norm_residual = norm_ls_residual;
            accepted_step = true;
            break;
          }

          // If residual increased, backtrack and exit
          // Do not reject first iteration
          if (norm_ls_residual > best_ls_residual && ls_iter > 0)
          {
            if (verbose)
              solver->pcout << "\tRejecting last step and backtracking"
                            << std::endl;
            break;
          }
        }

        if (!stop && !accepted_step && best_alpha > 0.)
        {
          solver->local_evaluation_point = solver->present_solution;
          solver->local_evaluation_point.add(best_alpha, solver->newton_update);
          solver->distribute_nonzero_constraints();
          solver->evaluation_point = solver->local_evaluation_point;
          norm_residual            = best_ls_residual;
          recompute_rhs            = true;
          accepted_step            = true;
        }

        if (!stop && accepted_step)
          // Update present solution for the next Newton iteration
          solver->present_solution = solver->evaluation_point;
        else if (!stop)
        {
          if (verbose)
            solver->pcout << "\tLine search failed to find an acceptable "
                             "step"
                          << std::endl;
          solver->evaluation_point = solver->present_solution;
          stop                     = true;
        }
      }
      else
      {
        // Increment solution and go to next iteration
        solver->local_evaluation_point = solver->evaluation_point;
        solver->local_evaluation_point.add(1., solver->newton_update);
        solver->distribute_nonzero_constraints();
        solver->evaluation_point = solver->local_evaluation_point;
        last_residual            = norm_residual;
      }

      if (++iter > this->param.max_iterations)
        stop = true;
    }

    // Update present solution
    solver->present_solution = solver->evaluation_point;

    if (!solution_found && throw_on_failure)
      throw std::runtime_error("Nonlinear solver did not converge.");

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
