
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/convergence_table.h>

#include "error_estimation/patches.h"
#include "error_estimation/solution_recovery.h"
#include "mesh.h"
#include "parameter_reader.h"
#include "types.h"

#include "../tests.h"

/**
 * 
 */
template <int dim>
class ScalarField : public Function<dim>
{
public:
  ScalarField()
    : Function<dim>(1)
  {}

  virtual double value(const Point<dim>  &p,
                       const unsigned int /* component = 0 */) const override
  {
    const double x = p[0];
    const double y = p[1];

    if constexpr (dim == 2)
      // return x*x;
      return sin(M_PI * x) * cos(M_PI * y);
    else
      return sin(M_PI * p[0]) * cos(M_PI * p[1]) * sin(M_PI * p[2]);
  }
};

template <int dim>
class ScalarFieldWithDerivatives : public Function<dim>
{
};


template <>
class ScalarFieldWithDerivatives<2> : public Function<2>
{
public:
  ScalarFieldWithDerivatives()
    : Function<2>(7)
  {}

  virtual double value(const Point<2>  &p,
                       const unsigned int component = 0) const override
  {
    const double x = p[0];
    const double y = p[1];

    // if (component == 0)      // f
    //   return x * x;
    // else if (component == 1) // df/dx
    //   return 2.0 * x;
    // else if (component == 2) // df/dy
    //   return 0.0;
    // else if (component == 3) // d2f/dx2
    //   return 2.0;
    // else if (component == 4) // d2f/dxdy
    //   return 0.0;
    // else if (component == 5) // d2f/dydx
    //   return 0.0;
    // else if (component == 6) // d2f/dy2
    //   return 0.0;

    if (component == 0)      // f
      return std::sin(M_PI * x) * std::cos(M_PI * y);
    else if (component == 1) // df/dx
      return M_PI * std::cos(M_PI * x) * std::cos(M_PI * y);
    else if (component == 2) // df/dy
      return -M_PI * std::sin(M_PI * x) * std::sin(M_PI * y);
    else if (component == 3) // d2f/dx2
      return -M_PI * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y);
    else if (component == 4) // d2f/dxdy
      return -M_PI * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y);
    else if (component == 5) // d2f/dydx
      return -M_PI * M_PI * std::cos(M_PI * x) * std::sin(M_PI * y);
    else if (component == 6) // d2f/dy2
      return -M_PI * M_PI * std::sin(M_PI * x) * std::cos(M_PI * y);

    DEAL_II_ASSERT_UNREACHABLE();
    return 0.;
  }
};

template <int dim>
void test_ppr(const unsigned int field_polynomial_degree)
{
  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  ConvergenceTable error_table;

  MappingFE<dim> mapping(FE_SimplexP<dim>(1));
  FESystem<dim>  fe(FE_SimplexP<dim>(field_polynomial_degree), 1);

  for (unsigned int i_conv = 0; i_conv < 4; ++i_conv)
  {
    // Create dummy parameters to mesh a rectangle with deal.II's routines
    Parameters::BoundaryConditionsData dummy_bc;
    ParameterHandler                   prm;
    ParameterReader<dim>               param(dummy_bc);
    param.declare(prm);
    param.read(prm);

    auto &mesh_param               = param.mesh;
    mesh_param.deal_ii_preset_mesh = "rectangle";

    if constexpr (dim == 2)
      mesh_param.deal_ii_mesh_param  = "4, 4 : 0., 0. : 1., 1. : true";
    else
      mesh_param.deal_ii_mesh_param  = "4, 4, 4 : 0., 0., 0. : 1., 1., 1. : true";

    mesh_param.refinement_level    = 1 + i_conv;

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

    // Create the patches of dof support points and print the sorted patches
    // for each owned mesh vertex
    ErrorEstimation::PatchHandler patch_handler(triangulation,
                                                mapping,
                                                dof_handler,
                                                solution,
                                                field_polynomial_degree + 1,
                                                fe.component_mask(
                                                  FEValuesExtractors::Scalar(0)));
    patch_handler.build_patches();

    const unsigned int                        highest_recovered_derivative = 2;
    ErrorEstimation::SolutionRecovery::Scalar recovery(
      highest_recovered_derivative,
      param,
      patch_handler,
      dof_handler,
      solution,
      fe,
      mapping,
      fe.component_mask(FEValuesExtractors::Scalar(0)));
    // recovery.compute_least_squares_matrices();
    recovery.reconstruct_fields();

    const ScalarFieldWithDerivatives<dim> exact_solution;
    const QGaussSimplex<dim> cell_quadrature(4);

    using Type = ErrorEstimation::SolutionRecovery::RecoveryType;
    using NormType = VectorTools::NormType;

    const double int_error_sol = recovery.compute_integral_error(Type::solution,
      NormType::Linfty_norm,
      exact_solution,
      cell_quadrature);
    const double nodal_error_sol = recovery.compute_nodal_error(Type::solution,
      NormType::Linfty_norm,
      exact_solution);

    const double int_error_grad = recovery.compute_integral_error(Type::gradient,
      NormType::Linfty_norm,
      exact_solution,
      cell_quadrature);
    const double nodal_error_grad = recovery.compute_nodal_error(Type::gradient,
      NormType::Linfty_norm,
      exact_solution);

    const double int_error_hess = recovery.compute_integral_error(Type::hessian,
      NormType::Linfty_norm,
      exact_solution,
      cell_quadrature);
    const double nodal_error_hess = recovery.compute_nodal_error(Type::hessian,
      NormType::Linfty_norm,
      exact_solution);

    error_table.add_value("n_elm", triangulation.n_global_active_cells());
    error_table.add_value("e_int_sol", int_error_sol);
    error_table.add_value("e_nodal_sol", nodal_error_sol);
    error_table.add_value("e_int_grad", int_error_grad);
    error_table.add_value("e_nodal_grad", nodal_error_grad);
    error_table.add_value("e_int_hess", int_error_hess);
    error_table.add_value("e_nodal_hess", nodal_error_hess);
  }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> errors = {"int", "nodal"};
    std::vector<std::string> fields = {"sol", "grad", "hess"};

    for (const auto &e : errors)
      for (const auto &f : fields)
      {
        const std::string key = "e_" + e + "_" + f;
        error_table.evaluate_convergence_rates(key, "n_elm", ConvergenceTable::reduction_rate_log2, dim);
        error_table.set_precision(key, 4);
        error_table.set_scientific(key, true);
      }

    deallog << "Convergence rates:" << std::endl;
    deallog << "Reconstructed solution and derivatives for solution of degree " << field_polynomial_degree << std::endl;
    error_table.write_text(deallog.get_file_stream());
  }
}

int main(int argc, char *argv[])
{
  try
  {
    initlog();
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    // MPILogInitAll                    log;
    test_ppr<2>(1);
    test_ppr<2>(2);
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
