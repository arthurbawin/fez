#include <deal.II/base/tensor.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_description.h>

#include "../tests.h"

#include "error_estimation/patches.h"
#include "post_processing_tools.h"

int main(int argc, char *argv[])
{
  try
  {
    initlog();
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    Tensor<2, 3> grad_u;
    grad_u[0][0] = 1.;
    grad_u[0][1] = 2.;
    grad_u[0][2] = 3.;
    grad_u[1][0] = 4.;
    grad_u[1][1] = 5.;
    grad_u[1][2] = 6.;
    grad_u[2][0] = 7.;
    grad_u[2][1] = 8.;
    grad_u[2][2] = 9.;

    const Tensor<1, 3> vorticity =
      PostProcessingTools::compute_vorticity_from_velocity_gradient<3>(grad_u);
    const double qcriterion =
      PostProcessingTools::compute_qcriterion_from_velocity_gradient<3>(grad_u);

    deallog << "vorticity " << vorticity << std::endl;
    deallog << "Qcriterion " << qcriterion << std::endl;

    Triangulation<3> serial_triangulation;
    GridGenerator::subdivided_hyper_cube(serial_triangulation, 2);
    GridTools::partition_triangulation(
      Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD),
      serial_triangulation);
    const auto description = TriangulationDescription::Utilities::
      create_description_from_triangulation(serial_triangulation,
                                            MPI_COMM_WORLD);
    parallel::fullydistributed::Triangulation<3> triangulation(MPI_COMM_WORLD);
    triangulation.create_triangulation(description);

    FESystem<3>  fe(FE_Q<3>(2), 1);
    DoFHandler<3> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    const IndexSet locally_owned_dofs = dof_handler.locally_owned_dofs();
    const IndexSet locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(dof_handler);
    LA::ParVectorType solution;
    solution.reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    MPI_COMM_WORLD);
    solution = 0.;

    const MappingQ1<3> mapping;
    ErrorEstimation::PatchHandler<3> patch_handler(
      triangulation,
      mapping,
      dof_handler,
      solution,
      3,
      fe.component_mask(FEValuesExtractors::Scalar(0)));
    patch_handler.build_patches();

    deallog << "P3 recovery basis " << patch_handler.dim_recovery_basis
            << std::endl;
    deallog << "P3 patches full rank "
            << patch_handler.has_least_squares_matrices() << std::endl;
  }
  catch (const std::exception &exc)
  {
    std::cerr << exc.what() << std::endl;
    return 1;
  }
  catch (...)
  {
    return 1;
  }

  return 0;
}
