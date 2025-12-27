
#include "error_estimation/patches.h"

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/numerics/data_out.h>

#include "../tests.h"

#include "mesh.h"
#include "parameter_reader.h"

/**
 * This tests that the patches of dof support points, used for least-squares
 * recovery of more accurate solution, are identical in sequential and parallel.
 * We create a uniform rectangle mesh and define the patches of support points
 * to fit a polynomial of order field_polynomial_degree + 1 at each owned mesh
 * vertex.
 */

template <int dim>
void test_patches(const unsigned int field_polynomial_degree)
{
  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  MappingFE<dim> mapping(FE_SimplexP<dim>(1));
  FESystem<dim>  fe(FE_SimplexP<dim>(field_polynomial_degree), 1);

  // Create dummy parameters to mesh a rectangle with deal.II's routines
  Parameters::BoundaryConditionsData dummy_bc;
  ParameterHandler                   prm;
  ParameterReader<dim>               param(dummy_bc);
  param.declare(prm);
  param.read(prm);

  auto &mesh_param               = param.mesh;
  mesh_param.deal_ii_preset_mesh = "rectangle";
  mesh_param.deal_ii_mesh_param  = "8, 8 : 0., 0. : 1., 1. : true";
  mesh_param.refinement_level    = 1;

  parallel::fullydistributed::Triangulation<dim> triangulation(
    mpi_communicator);
  read_mesh(triangulation, param);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // Create the patches of dof support points and print the sorted patches
  // for each owned mesh vertex
  ErrorEstimation::Patches patches(triangulation,
                                   mapping,
                                   dof_handler,
                                   field_polynomial_degree + 1,
                                   fe.component_mask(
                                     FEValuesExtractors::Scalar(0)));

  deallog << "Patches" << std::endl;
  patches.write_support_points_patch(deallog.get_file_stream());
}

int main(int argc, char *argv[])
{
  try
  {
    // Initialize deallog for test output.
    // This also reroutes deallog output to a file "output".
    initlog();
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    test_patches<2>(1);
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