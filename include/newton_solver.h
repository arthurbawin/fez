#ifndef NEWTON_SOLVER_H
#define NEWTON_SOLVER_H

#include <nonlinear_solver.h>

using namespace dealii;

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
  NewtonSolver(){};

public:
  void solve(const bool is_initial_step)
  {
    double global_res;
    double current_res;
    double last_res;
    bool   first_step     = is_initial_step;
    unsigned int outer_iteration = 0;
    last_res              = 1e6;
    current_res           = 1e6;
    global_res            = 1e6;

    // current_res and global_res are different as one is defined based on the l2
    // norm of the residual vector (current_res) and the other (global_res) is
    // defined by the physical solver and may differ from the l2_norm of the
    // residual vector. Only the global_res is compared to the tolerance in order
    // to evaluate if the nonlinear system is solved. Only current_res is used for
    // the alpha scheme as this scheme only monitors the convergence of the
    // non-linear system of equation (the matrix problem).

    // auto &evaluation_point = solver->get_evaluation_point();
    // auto &present_solution = solver->get_present_solution();

    while ((global_res > this->param.newton_tolerance) &&
           outer_iteration < 50)
      {
        evaluation_point = present_solution;

        this->assemble_matrix(false);

        // if (outer_iteration == 0)
          this->assemble_rhs(false);

        current_res      = this->system_rhs.l2_norm();
        if (outer_iteration == 0)
        {
          last_res         = current_res;
        }

        if (VERBOSE)
        {
          pcout << std::scientific << std::setprecision(16) << std::showpos;
          pcout << "Newton iteration: " << outer_iteration << "  - Residual:  " << current_res << std::endl;
        }

        this->solve_direct(first_step);
        double last_alpha_res = current_res;

        if(param.with_line_search)
        {
          unsigned int alpha_iter = 0;
          for (double alpha = 1.0; alpha > 1e-1; alpha *= 0.5)
          {
            local_evaluation_point       = present_solution;
            local_evaluation_point.add(alpha, newton_update);
            nonzero_constraints.distribute(local_evaluation_point);
            evaluation_point = local_evaluation_point;
            this->assemble_rhs(false);

            // auto &system_rhs = solver->get_system_rhs();
            current_res      = system_rhs.l2_norm();

            if (VERBOSE)
            {
              pcout << "\talpha = " << std::setw(6) << alpha
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
                local_evaluation_point = present_solution;
                local_evaluation_point.add(alpha, newton_update);
                nonzero_constraints.distribute(local_evaluation_point);
                evaluation_point = local_evaluation_point;

                if (VERBOSE)
                {
                  pcout
                    << "\t\talpha value was kept at alpha = " << alpha
                    << " since alpha = " << alpha / 2
                    << " increased the residual" << std::endl;
                }
                current_res = last_alpha_res;
                break;
              }
            if (current_res < 0.1 * last_res ||
                last_res < param.newton_tolerance)
              {
                break;
              }
            last_alpha_res = current_res;
            alpha_iter++;
          }
        }
        else
        {
          local_evaluation_point       = present_solution;
          local_evaluation_point.add(1., newton_update);
          nonzero_constraints.distribute(local_evaluation_point);
          evaluation_point = local_evaluation_point;
        }

        // global_res       = solver->get_current_residual();
        global_res       = current_res;
        present_solution = evaluation_point;
        last_res         = current_res;
        ++outer_iteration;
      }

    // If the non-linear solver has not converged abort simulation if
    // abort_at_convergence_failure=true
    if ((global_res > param.newton_tolerance) &&
        outer_iteration >= 50)
      {
        throw(std::runtime_error(
          "Stopping simulation because the non-linear solver has failed to converge"));
      }
  }
}

#endif