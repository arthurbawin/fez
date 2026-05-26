
#include <deal.II/base/convergence_table.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/numerics/data_out.h>

#include "../tests.h"

#include "error_estimation/patches.h"
#include "error_estimation/solution_recovery.h"
#include "mesh.h"
#include "parameter_reader.h"
#include "types.h"

/**
 * Tests the convergence of reconstructed solution and derivatives with the 
 * PPR (polynomial preserving recovery) operator.
 */
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
      return sin(M_PI * x) * cos(M_PI * y);
    else
      return sin(M_PI * p[0]) * cos(M_PI * p[1]) * sin(M_PI * p[2]);
  }
};

template <int dim>
class ScalarFieldWithDerivatives : public Function<dim>
{};


template <>
class ScalarFieldWithDerivatives<2> : public Function<2>
{
public:
  ScalarFieldWithDerivatives(const unsigned int n_components)
    : Function<2>(n_components)
  {}

  virtual double value(const Point<2>    &p,
                       const unsigned int component = 0) const override
  {
    const double x = p[0];
    const double y = p[1];

    const double sx = std::sin(M_PI * x);
    const double cx = std::cos(M_PI * x);
    const double sy = std::sin(M_PI * y);
    const double cy = std::cos(M_PI * y);

    const double pi  = M_PI;
    const double pi2 = pi * pi;
    const double pi3 = pi2 * pi;

    if (component == 0) // f
      return sx * cy;

    // First derivatives
    else if (component == 1) // df/dx
      return pi * cx * cy;
    else if (component == 2) // df/dy
      return -pi * sx * sy;

    // Second derivatives
    else if (component == 3) // d2f/dx2
      return -pi2 * sx * cy;
    else if (component == 4) // d2f/dxdy
      return -pi2 * cx * sy;
    else if (component == 5) // d2f/dydx
      return -pi2 * cx * sy;
    else if (component == 6) // d2f/dy2
      return -pi2 * sx * cy;

    // Third derivatives
    else if (component == 7) // d3f/dx3
      return -pi3 * cx * cy;
    else if (component == 8) // d3f/dx2dy
      return pi3 * sx * sy;
    else if (component == 9) // d3f/dxdydx
      return pi3 * sx * sy;
    else if (component == 10) // d3f/dxdy2
      return -pi3 * cx * cy;
    else if (component == 11) // d3f/dydx2
      return pi3 * sx * sy;
    else if (component == 12) // d3f/dydxdy
      return -pi3 * cx * cy;
    else if (component == 13) // d3f/dy2dx
      return -pi3 * cx * cy;
    else if (component == 14) // d3f/dy3
      return pi3 * sx * sy;

    DEAL_II_ASSERT_UNREACHABLE();
    return 0.;
  }
};

template <>
class ScalarFieldWithDerivatives<3> : public Function<3>
{
public:
  ScalarFieldWithDerivatives(const unsigned int n_components)
    : Function<3>(n_components)
  {}

  virtual double value(const Point<3>    &p,
                       const unsigned int component = 0) const override
  {
    const double x = p[0];
    const double y = p[1];
    const double z = p[2];

    const double sx = std::sin(M_PI * x);
    const double cx = std::cos(M_PI * x);
    const double sy = std::sin(M_PI * y);
    const double cy = std::cos(M_PI * y);
    const double sz = std::sin(M_PI * z);
    const double cz = std::cos(M_PI * z);

    const double pi  = M_PI;
    const double pi2 = pi * pi;
    const double pi3 = pi2 * pi;

    if (component == 0) // f
      return sx * cy * sz;

    // First derivatives
    else if (component == 1)
      return pi * cx * cy * sz; // x
    else if (component == 2)
      return -pi * sx * sy * sz; // y
    else if (component == 3)
      return pi * sx * cy * cz; // z

    // Second derivatives
    else if (component == 4)
      return -pi2 * sx * cy * sz; // xx
    else if (component == 5)
      return -pi2 * cx * sy * sz; // xy
    else if (component == 6)
      return pi2 * cx * cy * cz; // xz
    else if (component == 7)
      return -pi2 * cx * sy * sz; // yx
    else if (component == 8)
      return -pi2 * sx * cy * sz; // yy
    else if (component == 9)
      return -pi2 * sx * sy * cz; // yz
    else if (component == 10)
      return pi2 * cx * cy * cz; // zx
    else if (component == 11)
      return -pi2 * sx * sy * cz; // zy
    else if (component == 12)
      return -pi2 * sx * cy * sz; // zz

    // Third derivatives
    else if (component == 13)
      return -pi3 * cx * cy * sz; // xxx
    else if (component == 14)
      return pi3 * sx * sy * sz; // xxy
    else if (component == 15)
      return -pi3 * sx * cy * cz; // xxz
    else if (component == 16)
      return pi3 * sx * sy * sz; // xyx
    else if (component == 17)
      return -pi3 * cx * cy * sz; // xyy
    else if (component == 18)
      return -pi3 * cx * sy * cz; // xyz
    else if (component == 19)
      return -pi3 * sx * cy * cz; // xzx
    else if (component == 20)
      return -pi3 * cx * sy * cz; // xzy
    else if (component == 21)
      return -pi3 * cx * cy * sz; // xzz

    else if (component == 22)
      return pi3 * sx * sy * sz; // yxx
    else if (component == 23)
      return -pi3 * cx * cy * sz; // yxy
    else if (component == 24)
      return -pi3 * cx * sy * cz; // yxz
    else if (component == 25)
      return -pi3 * cx * cy * sz; // yyx
    else if (component == 26)
      return pi3 * sx * sy * sz; // yyy
    else if (component == 27)
      return -pi3 * sx * cy * cz; // yyz
    else if (component == 28)
      return -pi3 * cx * sy * cz; // yzx
    else if (component == 29)
      return -pi3 * sx * cy * cz; // yzy
    else if (component == 30)
      return +pi3 * sx * sy * sz; // yzz

    else if (component == 31)
      return -pi3 * sx * cy * cz; // zxx
    else if (component == 32)
      return -pi3 * cx * sy * cz; // zxy
    else if (component == 33)
      return -pi3 * cx * cy * sz; // zxz
    else if (component == 34)
      return -pi3 * cx * sy * cz; // zyx
    else if (component == 35)
      return -pi3 * sx * cy * cz; // zyy
    else if (component == 36)
      return +pi3 * sx * sy * sz; // zyz
    else if (component == 37)
      return -pi3 * cx * cy * sz; // zzx
    else if (component == 38)
      return +pi3 * sx * sy * sz; // zzy
    else if (component == 39)
      return -pi3 * sx * cy * cz; // zzz

    DEAL_II_ASSERT_UNREACHABLE();
    return 0.;
  }
};

template <int dim>
void test_ppr(const unsigned int n_convergence_steps,
              const unsigned int field_polynomial_degree,
              const bool         isoparametric,
              const bool         single_reconstruction,
              const unsigned int highest_recovered_derivative)
{
  MPI_Comm mpi_communicator(MPI_COMM_WORLD);

  ConvergenceTable error_table;

  MappingFE<dim> mapping(FE_SimplexP<dim>(1));
  FESystem<dim>  fe(FE_SimplexP<dim>(field_polynomial_degree), 1);

  std::vector<std::string> fields = {"sol"};

  unsigned int n_components = 1;
  for (unsigned int i = 1; i <= highest_recovered_derivative; ++i)
    n_components += pow(dim, i);

  for (unsigned int i_conv = 0; i_conv < n_convergence_steps; ++i_conv)
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
      mesh_param.deal_ii_mesh_param = "4, 4 : 0., 0. : 1., 1. : true";
    else
      mesh_param.deal_ii_mesh_param =
        "4, 4, 4 : 0., 0., 0. : 1., 1., 1. : true";

    mesh_param.refinement_level = 1 + i_conv;

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
    solution.reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    mpi_communicator);
    local_solution.reinit(locally_owned_dofs, mpi_communicator);
    VectorTools::interpolate(mapping,
                             dof_handler,
                             ScalarField<dim>(),
                             local_solution);
    solution = local_solution;

    // Create the patches of dof support points and print the sorted patches
    // for each owned mesh vertex
    ErrorEstimation::PatchHandler patch_handler(
      triangulation,
      mapping,
      dof_handler,
      solution,
      field_polynomial_degree + 1,
      fe.component_mask(FEValuesExtractors::Scalar(0)));
    patch_handler.build_patches();

    ErrorEstimation::SolutionRecovery::Scalar recovery(
      highest_recovered_derivative,
      param,
      patch_handler,
      dof_handler,
      solution,
      fe,
      mapping,
      fe.component_mask(FEValuesExtractors::Scalar(0)),
      isoparametric,
      single_reconstruction);
    recovery.reconstruct_fields(solution);

    const ScalarFieldWithDerivatives<dim> exact_solution(n_components);
    const QGaussSimplex<dim>              cell_quadrature(4);

    using Type     = ErrorEstimation::SolutionRecovery::RecoveryType;
    using NormType = VectorTools::NormType;

    const double int_error_sol =
      recovery.compute_integral_error(Type::solution,
                                      NormType::Linfty_norm,
                                      mapping,
                                      exact_solution,
                                      cell_quadrature);
    const double nodal_error_sol = recovery.compute_nodal_error(
      Type::solution, NormType::Linfty_norm, mapping, exact_solution);

    error_table.add_value("n_elm", triangulation.n_global_active_cells());
    error_table.add_value("e_int_sol", int_error_sol);
    error_table.add_value("e_nodal_sol", nodal_error_sol);

    if (highest_recovered_derivative >= 1)
    {
      const double int_error_grad =
        recovery.compute_integral_error(Type::gradient,
                                        NormType::Linfty_norm,
                                        mapping,
                                        exact_solution,
                                        cell_quadrature);
      const double nodal_error_grad = recovery.compute_nodal_error(
        Type::gradient, NormType::Linfty_norm, mapping, exact_solution);

      error_table.add_value("e_int_grad", int_error_grad);
      error_table.add_value("e_nodal_grad", nodal_error_grad);
      if (i_conv == 0)
        fields.push_back("grad");
    }

    if (highest_recovered_derivative >= 2)
    {
      const double int_error_hess =
        recovery.compute_integral_error(Type::hessian,
                                        NormType::Linfty_norm,
                                        mapping,
                                        exact_solution,
                                        cell_quadrature);
      const double nodal_error_hess = recovery.compute_nodal_error(
        Type::hessian, NormType::Linfty_norm, mapping, exact_solution);

      error_table.add_value("e_int_hess", int_error_hess);
      error_table.add_value("e_nodal_hess", nodal_error_hess);
      if (i_conv == 0)
        fields.push_back("hess");
    }

    if (highest_recovered_derivative >= 3)
    {
      const double int_error_third =
        recovery.compute_integral_error(Type::third_derivatives,
                                        NormType::Linfty_norm,
                                        mapping,
                                        exact_solution,
                                        cell_quadrature);
      const double nodal_error_third =
        recovery.compute_nodal_error(Type::third_derivatives,
                                     NormType::Linfty_norm,
                                     mapping,
                                     exact_solution);

      error_table.add_value("e_int_third", int_error_third);
      error_table.add_value("e_nodal_third", nodal_error_third);
      if (i_conv == 0)
        fields.push_back("third");
    }
  }

  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::vector<std::string> errors = {"int", "nodal"};

    for (const auto &e : errors)
      for (const auto &f : fields)
      {
        const std::string key = "e_" + e + "_" + f;
        error_table.evaluate_convergence_rates(
          key, "n_elm", ConvergenceTable::reduction_rate_log2, dim);
        error_table.set_precision(key, 4);
        error_table.set_scientific(key, true);
      }

    deallog << std::endl;
    deallog << "Convergence rates:" << std::endl;
    deallog << "Reconstructions for solution of degree           : "
            << field_polynomial_degree << std::endl;
    deallog << "Isoparametric representation                     : "
            << (isoparametric ? "yes" : "no") << std::endl;
    deallog << "Computing derivatives from single reconstruction : "
            << (single_reconstruction ? "yes" : "no") << std::endl;
    error_table.write_text(deallog.get_file_stream());
    deallog << "OK" << std::endl;
  }
}