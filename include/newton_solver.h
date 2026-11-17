#ifndef NEWTON_SOLVER_H
#define NEWTON_SOLVER_H

/**
 * This structure and solve function are borrowed from the Lethe project.
 */

#include <nonlinear_solver.h>
#include <time_handler.h>

#include <algorithm>
#include <cmath>
#include <deque>

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

    // Small convenience lambda to clearly indicate that the previous iterate is
    // restored. Currently this simply re-assigns the last accepted iterate in
    // evaluation_point, but additional operations could be needed at some
    // point.
    const auto restore_last_accepted_iterate = [&]() {
      solver->evaluation_point = present_solution;
    };

    // History of accepted residuals, to allow the comparison to increase in
    // line search
    std::deque<double> residual_history;

    while (true)
    {
      // Assemble residual and check if tolerance is reached
      // FIXME: Maybe use Linfty norm, which does not scale with the number of
      // RHS entries
      if (recompute_rhs)
      {
        // Fail if this evaluation_point would yield an ill-formed RHS (i.e.,
        // non-finite or with NaNs)
        if (!solver->evaluation_point_is_valid())
        {
          restore_last_accepted_iterate();
          if (throw_on_failure)
            throw std::runtime_error(
              "Nonlinear solver: the Newton residual (RHS) was not computed "
              "because the current solution would yield a vector with "
              "non-finite vlues or NaNs. If the solver solves for a position "
              "field, this can indicate that the associated mapping is invalid "
              "in at least one mesh element.");
          else
            break;
        }
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

      // Abort if residual too high or not finite
      if (!std::isfinite(norm_residual) ||
          (iter > 0 && norm_residual > this->param.divergence_tolerance))
      {
        restore_last_accepted_iterate();
        if (throw_on_failure)
          throw std::runtime_error("Nonlinear solver diverged");
        else
          break;
      }

      // Iterate is accepted
      present_solution = solver->evaluation_point;
      if (this->param.line_search.enable)
      {
        residual_history.push_back(norm_residual);
        if (residual_history.size() > this->param.line_search.memory)
          residual_history.pop_front();
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

      if (this->param.line_search.enable)
      {
        // Fraction of the residual decrease predicted by the linearization
        // that a step must achieve to be accepted (Armijo condition)
        constexpr double armijo_coeff = 1e-4;

        double norm_ls_residual;
        bool   step_accepted = false;
        last_residual        = norm_residual;

        // Nonmonotone reference (Grippo, Lampariello and Lucidi, 1986): the
        // largest of the last accepted residual norms. It is fixed during
        // backtracking, and allows the residual to rise above norm_residual.
        const double reference_residual =
          *std::max_element(residual_history.begin(), residual_history.end());

        double alpha = 1.;
        for (unsigned int trial = 0;
             trial < this->param.line_search.max_backtrack_steps;
             ++trial, alpha /= 2.)
        {
          // Compute NL(u + alpha * du) and check if residual decreases
          solver->local_evaluation_point = present_solution;
          solver->local_evaluation_point.add(alpha, solver->newton_update);
          solver->distribute_nonzero_constraints();
          solver->evaluation_point = solver->local_evaluation_point;

          // If damped step would yield an invalid RHS, keep on backtracking.
          if (!solver->evaluation_point_is_valid())
          {
            if (verbose)
              solver->pcout << "\tLine search with alpha = " << std::fixed
                            << std::setprecision(3) << alpha << std::scientific
                            << " : invalid trial" << std::endl;
            continue;
          }
          solver->assemble_rhs(); // NL(u + alpha * du)
          norm_ls_residual = solver->system_rhs.l2_norm();

          // Accept step if the residual is below tolerance, or if the
          // nonmonotone Armijo condition is satisfied: the residual is below
          // the reference residual minus a fraction armijo_coeff of the
          // decrease alpha * norm_residual predicted by the linearization.
          // With a memory of 1, the reference is norm_residual and this is the
          // monotone Armijo condition.
          // Reject a non-finite (e.g., with NaNs) residual or one above the
          // divergence tolerance, and try again with a smaller alpha.
          const bool accepted =
            std::isfinite(norm_ls_residual) &&
            norm_ls_residual <= this->param.divergence_tolerance &&
            (norm_ls_residual <= this->param.tolerance ||
             norm_ls_residual <=
               reference_residual - armijo_coeff * alpha * norm_residual);

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
          // No valid trial step length decreases the residual enough, e.g.,
          // because the initial guess is too far from the solution.
          // Restore the last iterate and exit with nonconverged solution.
          if (verbose)
            solver->pcout << "\tLine search failed: no step length satisfies "
                             "the Armijo condition"
                          << std::endl;
          restore_last_accepted_iterate();
          if (throw_on_failure)
            throw std::runtime_error("Nonlinear solver line search failed");
          else
            break;
        }

        // RHS was last assembled at the accepted solution: reuse its norm
        // instead of reassembling.
        recompute_rhs = false;
        norm_residual = norm_ls_residual;
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
