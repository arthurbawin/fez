
#include <iostream>
#include <string>
#include <vector>

#include "../tests.h"

#include "boundary_conditions.h"
#include "fsi_solver.h"
#include "parameter_reader.h"
#include "parameters.h"

/**
 * Tests the checkpoint/restart functions for solvers with hp capabilities,
 * here the FSI solver. First, solve the problem from t = 0 to 0.5,
 * checkpointing once at t = 0.3. Then, restart from t = 0.3, finish the
 * computation and compare the errors on velocity and pressure.
 */
template <int dim>
void test_restart()
{
  // Manually create the parameters of the file
  // examples/fsi/flow_around_cylinder_on_spring_2d
  Parameters::BoundaryConditionsData bc_data;
  bc_data.n_fluid_bc       = 4;
  bc_data.n_pseudosolid_bc = 4;

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
  prm.set("output directory", "./");
  prm.leave_subsection();

  // Postprocessing
  prm.enter_subsection("Postprocessing");
  prm.enter_subsection("forces computation");
  prm.set("enable", "true");
  prm.set("boundary id", "4");
  prm.set("output prefix", "forces");
  prm.set("output frequency", "1");
  prm.set("computation method", "lagrange multiplier");
  prm.leave_subsection();
  prm.enter_subsection("structure position");
  prm.set("enable", "true");
  prm.set("boundary id", "4");
  prm.set("output prefix", "center");
  prm.set("output frequency", "1");
  prm.set("precision", "10");
  prm.leave_subsection();
  prm.leave_subsection();

  // Mesh
  prm.enter_subsection("Mesh");
  prm.set("mesh file", "../../cylinder.msh");
  // prm.set("mesh file", "tests/unit_tests/cylinder.msh");
  prm.leave_subsection();

  // Nonlinear solver
  prm.enter_subsection("Nonlinear solver");
  prm.set("verbosity", "quiet");
  prm.set("tolerance", "1e-10");
  prm.set("divergence_tolerance", "1e+4");
  prm.set("max_iterations", "20");
  prm.set("enable_line_search", "false");
  prm.enter_subsection("reassembly heuristic");
  prm.set("decrease tolerance", "0");
  prm.leave_subsection();
  prm.leave_subsection();

  // Linear solver
  prm.enter_subsection("Linear solver");
  prm.enter_subsection("main physics");
  prm.set("verbosity", "quiet");
  prm.set("method", "direct_mumps");
  prm.set("reuse", "true");
  prm.leave_subsection();
  prm.leave_subsection();

  // FiniteElements
  prm.enter_subsection("FiniteElements");
  prm.set("use quads", "false");
  prm.set("Velocity degree", "2");
  prm.set("Pressure degree", "1");
  prm.set("Mesh position degree", "1");
  prm.set("Lagrange multiplier degree", "2");
  prm.leave_subsection();

  // Physical properties
  prm.enter_subsection("Physical properties");
  prm.set("number of fluids", "1");
  prm.enter_subsection("Fluid 0");
  prm.set("density", "1");
  prm.set("kinematic viscosity", "0.005");
  prm.leave_subsection();
  prm.set("number of pseudosolids", "1");
  prm.enter_subsection("Pseudosolid 0");
  prm.enter_subsection("lame lambda");
  prm.set("Function expression", "1");
  prm.leave_subsection();
  prm.enter_subsection("lame mu");
  prm.set("Function expression", "1");
  prm.leave_subsection();
  prm.leave_subsection(); // Pseudosolid 0
  prm.leave_subsection(); // Physical properties

  // FSI
  prm.enter_subsection("FSI");
  prm.set("verbosity", "quiet");
  prm.set("enable coupling", "true");
  prm.set("spring constant", "1");
  prm.set("cylinder radius", "0.5");
  prm.leave_subsection();

  // Initial conditions
  prm.enter_subsection("Initial conditions");
  prm.enter_subsection("velocity");
  prm.set("Function expression", "1; 0");
  prm.leave_subsection();
  prm.leave_subsection();

  // Fluid boundary conditions
  prm.enter_subsection("Fluid boundary conditions");
  prm.set("number", "4");
  prm.enter_subsection("boundary 0");
  prm.set("id", "1");
  prm.set("name", "Outlet");
  prm.set("type", "outflow");
  prm.leave_subsection();
  prm.enter_subsection("boundary 1");
  prm.set("id", "2");
  prm.set("name", "Inlet");
  prm.set("type", "input_function");
  prm.enter_subsection("u");
  prm.set("Function expression", "1");
  prm.leave_subsection();
  prm.enter_subsection("v");
  prm.set("Function expression", "0");
  prm.leave_subsection();
  prm.leave_subsection();
  prm.enter_subsection("boundary 2");
  prm.set("id", "3");
  prm.set("name", "NoFlux");
  prm.set("type", "slip");
  prm.leave_subsection();
  prm.enter_subsection("boundary 3");
  prm.set("id", "4");
  prm.set("name", "InnerBoundary");
  prm.set("type", "weak_no_slip");
  prm.leave_subsection();
  prm.leave_subsection();

  // Pseudosolid boundary conditions
  prm.enter_subsection("Pseudosolid boundary conditions");
  prm.set("number", "4");
  prm.enter_subsection("boundary 0");
  prm.set("id", "1");
  prm.set("name", "Outlet");
  prm.set("type", "fixed");
  prm.leave_subsection();
  prm.enter_subsection("boundary 1");
  prm.set("id", "2");
  prm.set("name", "Inlet");
  prm.set("type", "fixed");
  prm.leave_subsection();
  prm.enter_subsection("boundary 2");
  prm.set("id", "3");
  prm.set("name", "NoFlux");
  prm.set("type", "fixed");
  prm.leave_subsection();
  prm.enter_subsection("boundary 3");
  prm.set("id", "4");
  prm.set("name", "InnerBoundary");
  prm.set("type", "coupled_to_fluid");
  prm.leave_subsection();
  prm.leave_subsection();

  // Debug
  prm.enter_subsection("Debug");
  prm.set("fsi_coupling_option", "1");
  prm.leave_subsection();

  /**
   * These parameters will be modified:
   */

  // Time integration
  prm.enter_subsection("Time integration");
  prm.set("verbosity", "verbose");
  prm.set("dt", "0.1");
  prm.set("t_initial", "0");
  prm.set("t_end", "0.3");
  prm.set("scheme", "BDF2");
  prm.set("bdf start method", "BDF1");
  // Adaptation
  prm.enter_subsection("Adaptation");
  prm.set("verbosity", "verbose");
  prm.set("enable", "true");
  prm.set("max timestep", "1.");
  prm.set("min timestep", "1e-3");
  prm.set("max timestep increase", "10");
  prm.set("max timestep reduction", "1e-1");
  prm.set("adaptation strategy", "cfl");
  prm.set("target cfl number", "1");
  prm.set("reject timestep with large cfl", "true");
  prm.set("cfl ratio to reject", "2");
  prm.leave_subsection();
  prm.leave_subsection(); // Time integration

  // Checkpoint Restart
  prm.enter_subsection("Checkpoint Restart");
  prm.set("enable checkpoint", "true");
  prm.set("restart", "false");
  prm.set("checkpoint file", "checkpoint");
  prm.set("checkpoint frequency", "4");
  prm.leave_subsection();

  // Run the full computation from t = 0 to 0.3, checkpointing once
  {
    ParameterReader<dim> param(bc_data);
    ParameterHandler     dummy_prm;
    param.declare(dummy_prm);
    param.read(prm);
    FSISolverLessLambda<dim> problem(param);
    problem.run();
    deallog << "Complete forces and positions tables:" << std::endl;
    problem.write_forces(deallog.get_file_stream());
    problem.write_structure_mean_position(deallog.get_file_stream());
  }

  // Update the restart parameters then restart
  // The forces and position must be compared visually in the output
  {
    prm.enter_subsection("Checkpoint Restart");
    prm.set("enable checkpoint", "false");
    prm.set("restart", "true");
    prm.leave_subsection();

    ParameterReader<dim> param(bc_data);
    ParameterHandler     dummy_prm;
    param.declare(dummy_prm);
    param.read(prm);
    FSISolverLessLambda<dim> problem(param);
    problem.run();
    deallog << "Forces and positions for last step after restart:" << std::endl;
    problem.write_forces(deallog.get_file_stream());
    problem.write_structure_mean_position(deallog.get_file_stream());
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
