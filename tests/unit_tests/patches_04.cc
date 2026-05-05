
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/numerics/data_out.h>

#include "../tests.h"

#include "error_estimation/patches.h"
#include "mesh.h"
#include "parameter_reader.h"
#include "types.h"

/**
 * Same as patches_03.cc but in 3D.
 */

template <int dim>
class ScalarField : public Function<dim>
{
public:
  ScalarField()
    : Function<dim>(1)
  {}

  virtual double value(const Point<dim>  &p,
                       const unsigned int component = 0) const override
  {
    if constexpr (dim == 2)
      return sin(M_PI * p[0]) * cos(M_PI * p[1]);
    else
      return sin(M_PI * p[0]) * cos(M_PI * p[1]) * sin(M_PI * p[2]);
  }
};

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
  mesh_param.deal_ii_mesh_param  = "8, 8, 8 : 0., 0., 0. : 1., 1., 1. : true";
  mesh_param.refinement_level    = 1;

  parallel::fullydistributed::Triangulation<dim> triangulation(
    mpi_communicator);
  MeshTools::read_mesh(triangulation, param);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  // Initialize solution vectors
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

  // Create and print the patches.
  ErrorEstimation::PatchHandler patch_handler(triangulation,
                                              mapping,
                                              dof_handler,
                                              solution,
                                              field_polynomial_degree,
                                              fe.component_mask(
                                                FEValuesExtractors::Scalar(0)));

  const bool enforce_full_rank_least_squares_matrices = true;
  patch_handler.build_patches(enforce_full_rank_least_squares_matrices);

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
    deallog << "Patches for solution of degree " << field_polynomial_degree
            << std::endl;
  patch_handler.write_support_points_patch(".",
                                           solution,
                                           deallog.get_file_stream());
}

int main(int argc, char *argv[])
{
  try
  {
    initlog();
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    // Test for linear and quadratic solution
    const unsigned int max_solution_degree = 2;
    for (unsigned int d = 1; d <= max_solution_degree; ++d)
      test_patches<3>(d);
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
