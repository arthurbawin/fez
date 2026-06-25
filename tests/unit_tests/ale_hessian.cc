#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

using namespace dealii;

namespace
{
  constexpr unsigned int dim = 2;

  // Mesh-position variation of a scalar Laplacian, with G and K the gradient
  // and hessian of the mesh displacement on the moving mesh.
  double scalar_laplacian_variation(const Tensor<2, dim> &hessian,
                                    const Tensor<1, dim> &gradient,
                                    const Tensor<2, dim> &G,
                                    const Tensor<3, dim> &K)
  {
    double variation = 0.;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int a = 0; a < dim; ++a)
        variation -= G[a][i] * hessian[a][i] + G[a][i] * hessian[i][a] +
                     K[a][i][i] * gradient[a];
    return variation;
  }

  Tensor<2, dim> deformation_gradient()
  {
    Tensor<2, dim> F;
    F[0][0] = 1.2;
    F[0][1] = 0.1;
    F[1][0] = 0.05;
    F[1][1] = 0.9;
    return F;
  }

  Tensor<1, dim> translation()
  {
    Tensor<1, dim> t;
    t[0] = 0.05;
    t[1] = 0.02;
    return t;
  }

  Tensor<1, dim> base_position(const Point<dim> &p)
  {
    return deformation_gradient() * static_cast<Tensor<1, dim>>(p) +
           translation();
  }

  Tensor<1, dim> mapping_perturbation(const Point<dim> &p)
  {
    Tensor<1, dim> dx;
    dx[0] = 0.09 * p[0] * p[0] + 0.04 * p[0] * p[1];
    dx[1] = -0.07 * p[1] * p[1] + 0.05 * p[0] * p[1];
    return dx;
  }

  class AleScalarField : public Function<dim>
  {
  public:
    AleScalarField(const double perturbation_scale = 0.)
      : Function<dim>(dim + 1)
      , perturbation_scale(perturbation_scale)
    {}

    double value(const Point<dim>  &p,
                 const unsigned int component = 0) const override
    {
      AssertIndexRange(component, dim + 1);

      const Tensor<1, dim> x =
        base_position(p) + perturbation_scale * mapping_perturbation(p);

      if (component < dim)
        return x[component];
      return x[0] * x[0];
    }

  private:
    const double perturbation_scale;
  };

  class AlePerturbationField : public Function<dim>
  {
  public:
    AlePerturbationField()
      : Function<dim>(dim + 1)
    {}

    double value(const Point<dim>  &p,
                 const unsigned int component = 0) const override
    {
      AssertIndexRange(component, dim + 1);

      if (component < dim)
        return mapping_perturbation(p)[component];
      return 0.;
    }
  };

  struct TestData
  {
    Triangulation<dim> triangulation;
    FESystem<dim>     fe;
    DoFHandler<dim>   dof_handler;
    const FEValuesExtractors::Vector position;
    const FEValuesExtractors::Scalar scalar;
    ComponentMask                    position_mask;
    Vector<double>                   solution;
    Vector<double>                   perturbation_solution;
    Vector<double>                   plus_solution;
    Vector<double>                   minus_solution;
    const double                     perturbation_epsilon = 1.e-6;

    TestData()
      : fe(FE_Q<dim>(2), dim, FE_Q<dim>(2), 1)
      , dof_handler(triangulation)
      , position(0)
      , scalar(dim)
    {
      GridGenerator::subdivided_hyper_rectangle(triangulation,
                                                {4, 4},
                                                Point<dim>(0., 0.),
                                                Point<dim>(1., 1.),
                                                true);

      dof_handler.distribute_dofs(fe);
      position_mask = fe.component_mask(position);

      solution.reinit(dof_handler.n_dofs());
      perturbation_solution.reinit(dof_handler.n_dofs());
      plus_solution.reinit(dof_handler.n_dofs());
      minus_solution.reinit(dof_handler.n_dofs());

      VectorTools::interpolate(dof_handler, AleScalarField(), solution);
      VectorTools::interpolate(dof_handler,
                               AlePerturbationField(),
                               perturbation_solution);
      VectorTools::interpolate(
        dof_handler,
        AleScalarField(perturbation_epsilon),
        plus_solution);
      VectorTools::interpolate(
        dof_handler,
        AleScalarField(-perturbation_epsilon),
        minus_solution);
    }
  };

  void test_ale_hessian(TestData &data)
  {
    MappingFEField<dim, dim, Vector<double>> moving_mapping(
      data.dof_handler, data.solution, data.position_mask);
    MappingQ<dim> fixed_mapping(1);

    QGauss<dim> quadrature(3);
    FEValues<dim> fe_values_moving(moving_mapping,
                                   data.fe,
                                   quadrature,
                                   update_hessians);
    FEValues<dim> fe_values_fixed(fixed_mapping,
                                  data.fe,
                                  quadrature,
                                  update_hessians);

    std::vector<Tensor<2, dim>> moving_hessians(quadrature.size());
    std::vector<Tensor<2, dim>> fixed_hessians(quadrature.size());

    double max_moving_error = 0.;
    double max_fixed_error  = 0.;

    for (const auto &cell : data.dof_handler.active_cell_iterators())
    {
      fe_values_moving.reinit(cell);
      fe_values_moving[data.scalar].get_function_hessians(data.solution,
                                                          moving_hessians);

      fe_values_fixed.reinit(cell);
      fe_values_fixed[data.scalar].get_function_hessians(data.solution,
                                                         fixed_hessians);

      for (unsigned int q = 0; q < quadrature.size(); ++q)
        for (unsigned int i = 0; i < dim; ++i)
          for (unsigned int j = 0; j < dim; ++j)
          {
            const double expected_moving = (i == 0 && j == 0) ? 2. : 0.;
            const double expected_fixed =
              2. * deformation_gradient()[0][i] *
              deformation_gradient()[0][j];

            max_moving_error =
              std::max(max_moving_error,
                       std::abs(moving_hessians[q][i][j] - expected_moving));
            max_fixed_error =
              std::max(max_fixed_error,
                       std::abs(fixed_hessians[q][i][j] - expected_fixed));
          }
    }

    deallog << "max ALE hessian error = " << max_moving_error << std::endl;
    deallog << "max fixed hessian error = " << max_fixed_error << std::endl;

    AssertThrow(max_moving_error < 1.e-12,
                ExcMessage("ALE Hessian of x^2 was not reconstructed as 2."));
    AssertThrow(max_fixed_error < 1.e-12,
                ExcMessage("Fixed-map Hessian of x^2 mismatch."));
  }

  void test_ale_laplacian_variation(TestData &data)
  {
    MappingFEField<dim, dim, Vector<double>> moving_mapping(
      data.dof_handler, data.solution, data.position_mask);
    MappingFEField<dim, dim, Vector<double>> plus_mapping(
      data.dof_handler, data.plus_solution, data.position_mask);
    MappingFEField<dim, dim, Vector<double>> minus_mapping(
      data.dof_handler, data.minus_solution, data.position_mask);

    QGauss<dim> quadrature(3);
    FEValues<dim> fe_values_moving(moving_mapping,
                                   data.fe,
                                   quadrature,
                                   update_gradients | update_hessians);
    FEValues<dim> fe_values_plus(plus_mapping,
                                 data.fe,
                                 quadrature,
                                 update_hessians);
    FEValues<dim> fe_values_minus(minus_mapping,
                                  data.fe,
                                  quadrature,
                                  update_hessians);

    std::vector<Tensor<1, dim>> scalar_gradients(quadrature.size());
    std::vector<Tensor<2, dim>> scalar_hessians(quadrature.size());
    std::vector<Tensor<2, dim>> plus_hessians(quadrature.size());
    std::vector<Tensor<2, dim>> minus_hessians(quadrature.size());
    std::vector<Tensor<2, dim>> perturbation_gradients(quadrature.size());
    std::vector<Tensor<3, dim>> perturbation_hessians(quadrature.size());

    double max_laplacian_variation_error = 0.;

    for (const auto &cell : data.dof_handler.active_cell_iterators())
    {
      fe_values_moving.reinit(cell);
      fe_values_moving[data.scalar].get_function_gradients(
        data.solution, scalar_gradients);
      fe_values_moving[data.scalar].get_function_hessians(data.solution,
                                                          scalar_hessians);
      fe_values_moving[data.position].get_function_gradients(
        data.perturbation_solution, perturbation_gradients);
      fe_values_moving[data.position].get_function_hessians(
        data.perturbation_solution, perturbation_hessians);

      fe_values_plus.reinit(cell);
      fe_values_plus[data.scalar].get_function_hessians(data.solution,
                                                        plus_hessians);
      fe_values_minus.reinit(cell);
      fe_values_minus[data.scalar].get_function_hessians(data.solution,
                                                         minus_hessians);

      for (unsigned int q = 0; q < quadrature.size(); ++q)
      {
        const double analytic_variation =
          scalar_laplacian_variation(scalar_hessians[q],
                                     scalar_gradients[q],
                                     perturbation_gradients[q],
                                     perturbation_hessians[q]);
        const double fd_variation =
          (trace(plus_hessians[q]) - trace(minus_hessians[q])) /
          (2. * data.perturbation_epsilon);

        max_laplacian_variation_error =
          std::max(max_laplacian_variation_error,
                   std::abs(analytic_variation - fd_variation));
      }
    }

    deallog << "max ALE laplacian variation error = "
            << max_laplacian_variation_error << std::endl;

    AssertThrow(max_laplacian_variation_error < 1.e-6,
                ExcMessage("ALE laplacian variation mismatch."));
  }
} // namespace

int main()
{
  initlog();
  deallog << std::setprecision(6);
  deallog << std::scientific;

  TestData data;
  test_ale_hessian(data);
  test_ale_laplacian_variation(data);

  deallog << "OK" << std::endl;
}
