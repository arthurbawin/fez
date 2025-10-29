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
    double global_res;
    double current_res;
    double last_res;
    unsigned int outer_iteration = 0;
    last_res              = 1e6;
    current_res           = 1e6;
    global_res            = 1e6;

    auto solver = this->solver;
    const bool verbose = this->param.verbosity == Parameters::Verbosity::verbose;

    // current_res and global_res are different as one is defined based on the l2
    // norm of the residual vector (current_res) and the other (global_res) is
    // defined by the physical solver and may differ from the l2_norm of the
    // residual vector. Only the global_res is compared to the tolerance in order
    // to evaluate if the nonlinear system is solved. Only current_res is used for
    // the alpha scheme as this scheme only monitors the convergence of the
    // non-linear system of equation (the matrix problem).

    while ((global_res > this->param.tolerance) &&
           outer_iteration < 50)
      {
        solver->evaluation_point = solver->present_solution;

        solver->assemble_matrix();

        // if (outer_iteration == 0)
          solver->assemble_rhs();

        current_res      = solver->system_rhs.l2_norm();
        if (outer_iteration == 0)
        {
          last_res         = current_res;
        }

        if (verbose)
        {
          solver->pcout << std::scientific << std::setprecision(16) << std::showpos;
          solver->pcout << "Newton iteration: " << outer_iteration << "  - Residual:  " << current_res << std::endl;
        }

        solver->solve_linear_system(first_step);
        double last_alpha_res = current_res;

        if(this->param.enable_line_search)
        {
          unsigned int alpha_iter = 0;
          for (double alpha = 1.0; alpha > 1e-1; alpha *= 0.5)
          {
            solver->local_evaluation_point       = solver->present_solution;
            solver->local_evaluation_point.add(alpha, solver->newton_update);
            solver->distribute_nonzero_constraints();
            solver->evaluation_point = solver->local_evaluation_point;
            solver->assemble_rhs();

            current_res      = solver->system_rhs.l2_norm();

            if (verbose)
            {
              solver->pcout << "\talpha = " << std::setw(6) << alpha
                            << std::setw(0) << " res = "
                            << std::setprecision(6)
                            << std::setw(6) << current_res << std::endl;
            }

            // If it's not the first iteration of alpha check if the residual is
            // smaller than the last alpha iteration. If it's not smaller, we fall
            // back to the last alpha iteration.
            if (current_res > last_alpha_res and alpha_iter != 0)
              {
                alpha                  = 2 * alpha;
                solver->local_evaluation_point = solver->present_solution;
                solver->local_evaluation_point.add(alpha, solver->newton_update);
                solver->distribute_nonzero_constraints();
                solver->evaluation_point = solver->local_evaluation_point;

                if (verbose)
                {
                  solver->pcout
                    << "\t\talpha value was kept at alpha = " << alpha
                    << " since alpha = " << alpha / 2
                    << " increased the residual" << std::endl;
                }
                current_res = last_alpha_res;
                break;
              }
            if (current_res < 0.1 * last_res ||
                last_res < this->param.tolerance)
              {
                break;
              }
            last_alpha_res = current_res;
            alpha_iter++;
          }
        }
        else
        {
          solver->local_evaluation_point       = solver->present_solution;
          solver->local_evaluation_point.add(1., solver->newton_update);
          solver->distribute_nonzero_constraints();
          solver->evaluation_point = solver->local_evaluation_point;
        }

        // global_res       = solver->get_current_residual();
        global_res       = current_res;
        solver->present_solution = solver->evaluation_point;
        last_res         = current_res;
        ++outer_iteration;
      }

    // If the non-linear solver has not converged abort simulation if
    // abort_at_convergence_failure=true
    if ((global_res > this->param.tolerance) &&
        outer_iteration >= 50)
      {
        throw(std::runtime_error(
          "Stopping simulation because the non-linear solver has failed to converge"));
      }
  }
};

#endif