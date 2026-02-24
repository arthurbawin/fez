
#include <iostream>
#include <string>
#include <vector>

#include "../tests.h"

#include "boundary_conditions.h"
#include "incompressible_ns_solver.h"
#include "parameter_reader.h"
#include "parameters.h"

/**
 * Tests the checkpoint/restart functions with the incompressible N-S solver.
 * First, solve a manufactured solution problem from t = 0 to 0.5, checkpointing
 * once at t = 0.3. Then, restart from t = 0.3, finish the computation and
 * compare the errors on velocity and pressure.
 */
template <int dim>
void test_restart()
{
  // Manually create the parameters of the MMS file
  // incompressible_ns/mms_2d_bdf2.prm
  Parameters::BoundaryConditionsData bc_data;
  bc_data.n_fluid_bc = 4;

  ParameterHandler     prm;
  ParameterReader<dim> dummy_param(bc_data);
  dummy_param.declare(prm);

  // Dimension
  prm.enter_subsection("Dimension");
  prm.set("dimension", "2");
  prm.leave_subsection();

  // Timer
  prm.enter_subsection("Timer");
  prm.set("enable timer", "false");
  prm.leave_subsection();

  // Output
  prm.enter_subsection("Output");
  prm.set("write vtu results", "false");
  /**
   * The checkpoint files will be written in the current test directory
   */
  prm.set("output directory", "./");
  prm.set("output prefix", "solution");
  prm.leave_subsection();

  // Mesh
  prm.enter_subsection("Mesh");
  prm.set("verbosity", "quiet");
  prm.leave_subsection();

  // Time integration
  prm.enter_subsection("Time integration");
  prm.set("verbosity", "verbose");
  prm.set("dt", "0.1");
  prm.set("t_initial", "0");
  prm.set("t_end", "0.5");
  prm.set("scheme", "BDF2");
  prm.leave_subsection();

  // Nonlinear solver
  prm.enter_subsection("Nonlinear solver");
  prm.set("verbosity", "quiet");
  prm.set("tolerance", "1e-8");
  prm.set("divergence_tolerance", "1e+4");
  prm.set("max_iterations", "50");
  prm.set("enable_line_search", "true");
  prm.set("analytic_jacobian", "true");
  prm.leave_subsection();

  // Linear solver
  prm.enter_subsection("Linear solver");
  prm.enter_subsection("main physics");
  prm.set("verbosity", "quiet");
  prm.set("method", "direct_mumps");
  prm.leave_subsection();
  prm.leave_subsection();

  // FiniteElements
  prm.enter_subsection("FiniteElements");
  prm.set("Velocity degree", "2");
  prm.set("Pressure degree", "1");
  prm.leave_subsection();

  // Physical properties
  prm.enter_subsection("Physical properties");
  prm.set("number of fluids", "1");
  prm.enter_subsection("Fluid 0");
  prm.set("density", "1.2345");
  prm.set("kinematic viscosity", "9.8765");
  prm.leave_subsection();
  prm.leave_subsection();

  // Initial conditions
  prm.enter_subsection("Initial conditions");
  prm.set("to mms", "true");
  prm.leave_subsection();

  // Fluid boundary conditions
  prm.enter_subsection("Fluid boundary conditions");
  prm.set("number", "4");
  prm.set("fix pressure constant", "true");
  prm.enter_subsection("boundary 0");
  prm.set("id", "1");
  prm.set("name", "x_min");
  prm.set("type", "velocity_mms");
  prm.leave_subsection();
  prm.enter_subsection("boundary 1");
  prm.set("id", "2");
  prm.set("name", "x_max");
  prm.set("type", "velocity_mms");
  prm.leave_subsection();
  prm.enter_subsection("boundary 2");
  prm.set("id", "3");
  prm.set("name", "y_min");
  prm.set("type", "velocity_mms");
  prm.leave_subsection();
  prm.enter_subsection("boundary 3");
  prm.set("id", "4");
  prm.set("name", "y_max");
  prm.set("type", "velocity_mms");
  prm.leave_subsection();
  prm.leave_subsection();

  // Exact solution
  prm.enter_subsection("Exact solution");
  prm.enter_subsection("exact velocity");
  prm.set("Function expression",
          "(x*y + y^2) * cos(t); (x*y + x^2) * 1/(1 + t^2)");
  prm.leave_subsection();
  prm.enter_subsection("exact pressure");
  prm.set("Function expression", "x + y");
  prm.leave_subsection();
  prm.leave_subsection();

  // Manufactured solution
  prm.enter_subsection("Manufactured solution");
  prm.set("enable", "true");
  prm.set("type", "time");
  prm.set("convergence steps", "1");
  // Space convergence
  prm.enter_subsection("Space convergence");
  prm.set("use dealii cube mesh", "true");
  prm.set("norms to compute", "L2_norm, Linfty_norm, H1_seminorm");
  prm.leave_subsection();
  // Time convergence
  prm.enter_subsection("Time convergence");
  prm.set("norm", "Linfty");
  prm.set("use spatial mesh", "true");
  prm.set("spatial mesh index", "1");
  prm.set("time step reduction", "0.5");
  prm.leave_subsection();
  prm.leave_subsection();

  // Checkpoint Restart
  prm.enter_subsection("Checkpoint Restart");
  prm.set("enable checkpoint", "true");
  prm.set("restart", "false");
  prm.set("checkpoint file", "checkpoint");
  prm.set("checkpoint frequency", "3");
  prm.leave_subsection();

  std::vector<std::pair<double, double>> e_u, e_u_restart, e_p, e_p_restart;

  // Run the full computation from t = 0 to 0.5, checkpointing once at t = 0.3
  {
    ParameterReader<dim> param(bc_data);
    ParameterHandler     dummy_prm;
    param.declare(dummy_prm);
    param.read(prm);
    NSSolver<dim> problem(param);
    problem.run();
    problem.compute_errors();

    std::shared_ptr<ErrorHandler> eh =
      problem.get_error_handler(VectorTools::NormType::L2_norm);
    e_u = eh->get_unsteady_errors("u");
    e_p = eh->get_unsteady_errors("p");

    deallog << "Error without restart" << std::endl;
    for (unsigned int i = 0; i < e_u.size(); ++i)
    {
      const auto &eu = e_u[i];
      const auto &ep = e_p[i];
      deallog << "t = " << eu.first << " : e_u = " << eu.second
              << " - e_p = " << ep.second << std::endl;
    }
  }

  // Update the restart parameters, then
  // restart at t = 0.3 and continue to 0.5, then compare errors
  const unsigned int time_steps_shift = 3;
  {
    prm.enter_subsection("Checkpoint Restart");
    prm.set("enable checkpoint", "false");
    prm.set("restart", "true");
    prm.set("checkpoint file", "checkpoint");
    prm.set("checkpoint frequency", "3");
    prm.leave_subsection();

    ParameterReader<dim> param(bc_data);
    ParameterHandler     dummy_prm;
    param.declare(dummy_prm);
    param.read(prm);
    NSSolver<dim> problem(param);
    problem.run();
    problem.compute_errors();

    std::shared_ptr<ErrorHandler> eh =
      problem.get_error_handler(VectorTools::NormType::L2_norm);
    e_u_restart = eh->get_unsteady_errors("u");
    e_p_restart = eh->get_unsteady_errors("p");

    deallog << "Error with restart" << std::endl;
    for (unsigned int i = 0; i < e_u_restart.size(); ++i)
    {
      const auto &eu = e_u_restart[i];
      const auto &ep = e_p_restart[i];
      deallog << "t = " << eu.first << " : e_u = " << eu.second
              << " - e_p = " << ep.second << std::endl;
    }
  }

  AssertThrow(e_u.size() == 6, ExcMessage("Size mismatch"));
  AssertThrow(e_p.size() == 6, ExcMessage("Size mismatch"));
  AssertThrow(e_u_restart.size() == 6 - time_steps_shift,
              ExcMessage("Size mismatch"));
  AssertThrow(e_p_restart.size() == 6 - time_steps_shift,
              ExcMessage("Size mismatch"));
  for (unsigned int i = 0; i < time_steps_shift; ++i)
  {
    AssertThrow(std::abs(e_u[i + time_steps_shift].second -
                         e_u_restart[i].second) < 1e-12,
                ExcMessage("Error mismatch"));
    AssertThrow(std::abs(e_p[i + time_steps_shift].second -
                         e_p_restart[i].second) < 1e-12,
                ExcMessage("Error mismatch"));
  }
  deallog << "OK" << std::endl;
}

int main(int argc, char *argv[])
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    MPILogInitAll                    log;
    test_restart<2>();
  }
  catch (const std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
