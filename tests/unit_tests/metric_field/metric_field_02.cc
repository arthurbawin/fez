
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>

#include "../../tests.h"

#include "boundary_conditions.h"
#include "error_estimation/patches.h"
#include "error_estimation/solution_recovery.h"
#include "mesh.h"
#include "metric_field.h"
#include "parameter_reader.h"
#include "parameters.h"

template <int dim>
class ScalarField : public Function<dim>
{
public:
  ScalarField()
    : Function<dim>(1)
  {}

  virtual double value(const Point<dim> &p,
                       const unsigned int /* component = 0 */) const override
  {
    const double x = p[0];
    const double y = p[1];

    if constexpr (dim == 2)
      return tanh((2 * x - sin(5 * y)) / 0.2);
    else
      return sin(M_PI * p[0]) * cos(M_PI * p[1]) * sin(M_PI * p[2]);
  }
};

/**
 * Create an optimal metric field for P1 interpolation error control.
 * Use the reconstructed derivatives of an interpolated solution.
 */
template <int dim>
void test()
{
  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  // Linear interpolation
  constexpr unsigned int field_polynomial_degree = 1;

  MappingFE<dim> mapping(FE_SimplexP<dim>(1));
  FESystem<dim>  fe(FE_SimplexP<dim>(field_polynomial_degree), 1);

  // Create dummy parameters to mesh a rectangle with deal.II's routines
  Parameters::BoundaryConditionsData dummy_bc;
  dummy_bc.n_metric_fields = 1;
  ParameterHandler     prm;
  ParameterReader<dim> param(dummy_bc);
  param.declare(prm);

  // Assign scalar field for the metric
  prm.enter_subsection("Metric tensor fields");
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

  param.read(prm);

  auto &mesh_param               = param.mesh;
  mesh_param.deal_ii_preset_mesh = "rectangle";

  if constexpr (dim == 2)
    mesh_param.deal_ii_mesh_param = "4, 4 : 0., 0. : 1., 1. : true";
  else
    mesh_param.deal_ii_mesh_param = "2, 2, 2 : 0., 0., 0. : 1., 1., 1. : true";

  mesh_param.refinement_level = 2;

  parallel::fullydistributed::Triangulation<dim> triangulation(
    mpi_communicator);
  MeshTools::read_mesh(triangulation, param);

  // Initialize solution vectors
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
  IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);
  LA::ParVectorType solution, local_solution;
  solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  local_solution.reinit(locally_owned_dofs, mpi_communicator);
  VectorTools::interpolate(mapping,
                           dof_handler,
                           ScalarField<dim>(),
                           local_solution);
  solution = local_solution;

  const unsigned int derivatives_order = field_polynomial_degree + 1;

  // Create patches and reconstruct hessian
  ErrorEstimation::PatchHandler patch_handler(
    triangulation, mapping, dof_handler, solution, derivatives_order);

  patch_handler.build_patches();

  ErrorEstimation::SolutionRecovery::Scalar recovery(derivatives_order,
                                                     param,
                                                     patch_handler,
                                                     dof_handler,
                                                     solution,
                                                     fe,
                                                     mapping);

  recovery.reconstruct_fields(solution);

  TimeHandler dummy_time_handler(param.time_integration);

  // Compute the optimal metric for interpolation error.
  // Do not apply gradation.
  MetricField<dim> metrics(0, param, triangulation);
  metrics.increment_anisotropic_measure(recovery, dummy_time_handler);
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
