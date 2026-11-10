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
    bool         stop           = false;
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

    while (!stop)
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
      preconditioner_rebuilt = solver->preconditioner_rebuilt_at_last_solve;
      // norm_increment = solver->newton_update.linfty_norm();

      if (this->param.enable_line_search)
      {
        bool   accepted         = false;
        double norm_ls_residual = norm_residual;
        double alpha            = 1.;
        for (unsigned int trial = 0; trial < 20; ++trial, alpha *= 0.5)
        {
          solver->local_evaluation_point = present_solution;
          solver->local_evaluation_point.add(alpha, solver->newton_update);
          solver->distribute_nonzero_constraints();
          solver->evaluation_point = solver->local_evaluation_point;
          solver->assemble_rhs();
          norm_ls_residual = solver->system_rhs.l2_norm();

          if (verbose)
            solver->pcout << "\tLine search with alpha = " << std::scientific
                          << std::setprecision(8) << alpha
                          << " : res = " << norm_ls_residual << std::endl;

          // Compare every trial against the residual before the line search.
          if (std::isfinite(norm_ls_residual) &&
              (norm_ls_residual <= this->param.tolerance ||
               norm_ls_residual <= (1. - 1e-4 * alpha) * norm_residual))
          {
            accepted = true;
            break;
          }
        }

        if (!accepted)
        {
          // Restore the accepted state when all trial updates fail.
          solver->local_evaluation_point = present_solution;
          solver->evaluation_point       = present_solution;
          solver->assemble_rhs();
          time_handler.set_last_nonlinear_solve_status(false);
          if (throw_on_failure)
            throw std::runtime_error(
              "Newton line search failed to reduce residual");
          break;
        }

        last_residual    = norm_residual;
        norm_residual    = norm_ls_residual;
        recompute_rhs    = false;
        present_solution = solver->evaluation_point;
        if (norm_residual <= this->param.tolerance)
        {
          if (verbose)
            solver->pcout << "Stopping because residual is below prescribed "
                             "tolerance ("
                          << std::setprecision(2) << this->param.tolerance
                          << ")" << std::endl;
          stop           = true;
          solution_found = true;
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
    present_solution = solver->evaluation_point;

    if (iter > this->param.max_iterations &&
        norm_residual > this->param.tolerance)
      if (throw_on_failure)
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
