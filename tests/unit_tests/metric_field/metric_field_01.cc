
#include <deal.II/distributed/fully_distributed_tria.h>

#include "../../tests.h"

#include "boundary_conditions.h"
#include "mesh.h"
#include "metric_field.h"
#include "parameter_reader.h"
#include "parameters.h"

/**
 * Create an optimal metric field for P1 interpolation error control.
 * Use the exact derivatives of a given scalar field.
 */
template <int dim>
void test()
{
  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  // Create dummy parameters to mesh a rectangle with deal.II's routines
  Parameters::BoundaryConditionsData dummy_bc;
  dummy_bc.n_metric_fields = 1;
  ParameterHandler     prm;
  ParameterReader<dim> param(dummy_bc);
  param.declare(prm);

  // Assign scalar field for the metric
  prm.enter_subsection("Metric tensor fields");
  prm.enter_subsection("Metric field 0");
  prm.set("min mesh size", "1e-4");
  prm.set("max mesh size", "10");
  prm.enter_subsection("Analytical scalar field");
  prm.set("Function expression", "tanh( (2*x - sin(5*y)) / 0.2)");
  prm.leave_subsection();
  prm.enter_subsection("Multiscale optimal metric for interpolation error");
  prm.set("use analytical derivatives", "true");
  prm.leave_subsection();
  prm.leave_subsection();
  prm.leave_subsection();

  param.read(prm);

  auto &mesh_param               = param.mesh;
  mesh_param.deal_ii_preset_mesh = "rectangle";

  if constexpr (dim == 2)
    mesh_param.deal_ii_mesh_param = "4, 4 : 0., 0. : 1., 1. : true";
  else
    mesh_param.deal_ii_mesh_param = "4, 4, 4 : 0., 0., 0. : 1., 1., 1. : true";

  mesh_param.refinement_level = 2;

  parallel::fullydistributed::Triangulation<dim> triangulation(
    mpi_communicator);
  MeshTools::read_mesh(triangulation, param);

  TimeHandler dummy_time_handler(param.time_integration);

  // Compute the optimal metric for interpolation error.
  // Do not apply gradation.
  MetricField<dim> metrics(0, param, triangulation);
  metrics.increment_anisotropic_measure(dummy_time_handler);
  metrics.apply_optimal_steady_multiscale_scaling();
  metrics.write_metrics(deallog.get_file_stream());

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    deallog << "OK" << std::endl;
}

int main(int argc, char **argv)
{
  try
  {
    initlog();
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    // MPILogInitAll                    log;
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
