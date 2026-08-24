
#include "../../tests.h"

#include "heat_solver.h"
#include "metric_field.h"
#include "parameter_reader.h"
#include "parameters.h"

/**
 * Tests the computation of a metric field to control the interpolation error
 * over a transient scalar field. The solution is a tanh travelling at velocity
 * c = 1. The metric field is the integral over the
 */
template <int dim>
void test()
{
  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  // Number of time subintervals
  const unsigned int n_intervals = 2;

  // Create dummy parameters to mesh a rectangle with deal.II's routines
  Parameters::BoundaryConditionsData dummy_bc;
  dummy_bc.n_heat_bc       = 2 * dim;
  dummy_bc.n_metric_fields = 1;
  ParameterHandler     prm;
  ParameterReader<dim> param(dummy_bc);
  param.declare(prm);

  // Time integration
  prm.enter_subsection("Time integration");
  prm.set("verbosity", "verbose");
  prm.set("dt", "0.1");
  prm.set("t_initial", "0");
  prm.set("t_end", "1");
  prm.set("scheme", "BDF1");
  prm.leave_subsection();

  // Nonlinear solver
  prm.enter_subsection("Nonlinear solver");
  prm.set("verbosity", "verbose");
  prm.set("enable_line_search", "false");
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
  prm.set("Temperature degree", "1");
  prm.leave_subsection();

  // Mesh
  prm.enter_subsection("Mesh");
  prm.set("verbosity", "quiet");

  prm.set("dealii preset mesh", "rectangle");
  if constexpr (dim == 2)
    prm.set("dealii mesh parameters", "4, 4 : -2., -1. : 2., 1. : true");
  else
    prm.set("dealii mesh parameters",
            "2, 2, 2 : -2, -1, -1 : 2., 1., 1. : true");

  prm.enter_subsection("Adaptation");
  prm.set("verbosity", "verbose");
  prm.set("enable", "false"); // <<<<< Computing metrics, but no adaptation
  prm.set("strategy", "riemannian metric");
  prm.enter_subsection("Metric");
  prm.set("n fixed point", "1");
  prm.set("n time intervals", std::to_string(n_intervals));
  prm.set("transfer solution", "true");
  prm.leave_subsection();
  prm.leave_subsection();
  prm.leave_subsection();

  // Metric parameters
  prm.enter_subsection("Metric tensor fields");
  prm.set("number", "1");
  prm.set("always compute", "true");
  prm.enter_subsection("Metric field 0");
  prm.set("type", "interpolation error");
  prm.set("variable", "temperature");
  prm.set("min mesh size", "1e-4");
  prm.set("max mesh size", "10");
  prm.enter_subsection("Multiscale optimal metric for interpolation error");
  prm.set("use analytical derivatives", "false");
  prm.leave_subsection();
  prm.leave_subsection();
  prm.leave_subsection();

  // Exact solution is a travelling tanh at c = 1
  prm.enter_subsection("Exact solution");
  prm.enter_subsection("exact temperature");
  prm.set("Function constants", "c=1");
  prm.set("Function expression", "tanh((2 * (x - c*t) - sin(5 * y)) / 0.2)");
  prm.leave_subsection();
  prm.leave_subsection();

  // Initial conditions
  prm.enter_subsection("Initial conditions");
  prm.set("to mms", "true");
  prm.leave_subsection();

  // Boundary conditions
  prm.enter_subsection("Heat boundary conditions");
  if constexpr (dim == 2)
    prm.set("number", "4");
  else
    prm.set("number", "6");
  prm.enter_subsection("boundary 0");
  prm.set("id", "0");
  prm.set("name", "x_min");
  prm.set("type", "dirichlet_mms");
  prm.leave_subsection();
  prm.enter_subsection("boundary 1");
  prm.set("id", "1");
  prm.set("name", "x_max");
  prm.set("type", "dirichlet_mms");
  prm.leave_subsection();
  prm.enter_subsection("boundary 2");
  prm.set("id", "2");
  prm.set("name", "y_min");
  prm.set("type", "dirichlet_mms");
  prm.leave_subsection();
  prm.enter_subsection("boundary 3");
  prm.set("id", "3");
  prm.set("name", "y_max");
  prm.set("type", "dirichlet_mms");
  prm.leave_subsection();

  if constexpr (dim == 3)
  {
    prm.enter_subsection("boundary 4");
    prm.set("id", "4");
    prm.set("name", "z_min");
    prm.set("type", "dirichlet_mms");
    prm.leave_subsection();
    prm.enter_subsection("boundary 5");
    prm.set("id", "5");
    prm.set("name", "z_max");
    prm.set("type", "dirichlet_mms");
    prm.leave_subsection();
  }

  prm.leave_subsection();

  // Manufactured solution
  prm.enter_subsection("Manufactured solution");
  prm.set("enable", "true");
  prm.set("type", "space");
  prm.set("convergence steps", "1");
  // Space convergence
  prm.enter_subsection("Space convergence");
  prm.set("norms to compute", "L2_norm");
  prm.set("initial target number of vertices", "100");
  prm.leave_subsection();
  prm.leave_subsection();

  param.read(prm);

  HeatSolver<dim> solver(param);
  solver.template run_convergence_loop<dim>();

  // Get the metrics for adaptation associated with each time interval
  for (unsigned int i = 0; i < n_intervals; ++i)
  {
    if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
      deallog << "Metrics for time subinterval " << i << ":" << std::endl;
    const auto &metrics = solver.get_metric_field(i);
    metrics.write_metrics(deallog.get_file_stream());
  }

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    deallog << "OK" << std::endl;
}

int main(int argc, char **argv)
{
  try
  {
    initlog();
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    test<2>();
    test<3>();
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
