#include <stabilization_utils.h>

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "../tests.h"

#include <cmath>
#include <iomanip>
#include <limits>
#include <vector>

using namespace dealii;

namespace
{
  constexpr unsigned int dim = 2;

  class CurvedPosition : public Function<dim>
  {
  public:
    CurvedPosition()
      : Function<dim>(dim)
    {}

    double value(const Point<dim>  &p,
                 const unsigned int component = 0) const override
    {
      AssertIndexRange(component, dim);

      if (component == 0)
        return 0.2 + 1.3 * p[0] + 0.15 * p[0] * p[1] +
               0.05 * p[1] * p[1];
      return -0.1 + 0.2 * p[0] + 0.8 * p[1] - 0.04 * p[0] * p[0] +
             0.08 * p[0] * p[1];
    }
  };

  template <typename MappingType>
  double compare_analytic_and_fevalues_lengths(
    const MappingType       &mapping,
    Triangulation<dim>      &triangulation,
    const FiniteElement<dim> &linear_geometry_fe,
    const Quadrature<dim>   &quadrature,
    const bool               use_quads,
    const Tensor<1, dim>    &advection_velocity,
    const double             fallback_length)
  {
    DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(linear_geometry_fe);

    FEValues<dim> fe_values(mapping,
                            linear_geometry_fe,
                            quadrature,
                            update_gradients | update_inverse_jacobians);

    double max_difference = 0.;
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
      {
        std::vector<Tensor<1, dim>> geometry_gradients;
        geometry_gradients.reserve(linear_geometry_fe.n_dofs_per_cell());
        for (unsigned int i = 0; i < linear_geometry_fe.n_dofs_per_cell(); ++i)
          geometry_gradients.push_back(fe_values.shape_grad(i, q));

        const double h_native =
          Stabilization::compute_streamline_length_from_geometry_gradients(
            advection_velocity, geometry_gradients, fallback_length);
        const double h_analytic = Stabilization::compute_streamline_length(
          advection_velocity,
          fe_values.inverse_jacobian(q),
          fe_values.get_quadrature().point(q),
          use_quads,
          fallback_length);

        max_difference =
          std::max(max_difference, std::abs(h_native - h_analytic));
      }
    }

    return max_difference;
  }

  double compute_min_fevalues_length_with_quadratic_geometry()
  {
    Triangulation<dim> triangulation;
    GridGenerator::subdivided_hyper_rectangle(triangulation,
                                              {2, 1},
                                              Point<dim>(0., 0.),
                                              Point<dim>(1., 0.4),
                                              true);

    FESystem<dim>                 position_fe(FE_Q<dim>(2), dim);
    DoFHandler<dim>               position_dof_handler(triangulation);
    FEValuesExtractors::Vector    position(0);
    const ComponentMask           position_mask =
      position_fe.component_mask(position);
    Vector<double> position_solution;

    position_dof_handler.distribute_dofs(position_fe);
    position_solution.reinit(position_dof_handler.n_dofs());
    VectorTools::interpolate(position_dof_handler,
                             CurvedPosition(),
                             position_solution,
                             position_mask);

    MappingFEField<dim, dim, Vector<double>> curved_mapping(
      position_dof_handler, position_solution, position_mask);

    Tensor<1, dim> advection_velocity;
    advection_velocity[0] = 1.;
    advection_velocity[1] = 0.25;

    FE_Q<dim>       linear_geometry_fe(1);
    DoFHandler<dim> geometry_dof_handler(triangulation);
    geometry_dof_handler.distribute_dofs(linear_geometry_fe);

    QGauss<dim> quadrature(3);
    FEValues<dim> fe_values(curved_mapping,
                            linear_geometry_fe,
                            quadrature,
                            update_gradients);

    double min_length = std::numeric_limits<double>::max();
    for (const auto &cell : geometry_dof_handler.active_cell_iterators())
    {
      fe_values.reinit(cell);
      for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
      {
        std::vector<Tensor<1, dim>> geometry_gradients;
        geometry_gradients.reserve(linear_geometry_fe.n_dofs_per_cell());
        for (unsigned int i = 0; i < linear_geometry_fe.n_dofs_per_cell(); ++i)
          geometry_gradients.push_back(fe_values.shape_grad(i, q));

        min_length = std::min(
          min_length,
          Stabilization::compute_streamline_length_from_geometry_gradients(
            advection_velocity, geometry_gradients, cell->diameter()));
      }
    }

    return min_length;
  }

  void test_tau_polynomial_degree_scaling()
  {
    const double convective_tau_p1 =
      Stabilization::compute_tau(1., true, 3., 0., 2., 1);
    const double convective_tau_p2 =
      Stabilization::compute_tau(1., true, 3., 0., 2., 2);

    const double diffusive_tau_p1 =
      Stabilization::compute_tau(1., true, 0., 1., 2., 1);
    const double diffusive_tau_p2 =
      Stabilization::compute_tau(1., true, 0., 1., 2., 2);
    const double transient_tau =
      Stabilization::compute_tau(2., false, 0., 0., 2., 1);

    AssertThrow(std::abs(convective_tau_p2 / convective_tau_p1 - 0.5) <
                  1.e-14,
                ExcMessage("Convective tau scaling should be proportional to "
                           "1/p."));
    AssertThrow(std::abs(diffusive_tau_p2 / diffusive_tau_p1 - 0.25) <
                  1.e-14,
                ExcMessage("Diffusive tau scaling should be proportional to "
                           "1/p^2."));
    AssertThrow(std::abs(transient_tau - 2.) < 1.e-14,
                ExcMessage("Transient tau should use 1/dt^2."));
  }

  void test_streamline_lengths()
  {
    Tensor<1, dim> advection_velocity;
    advection_velocity[0] = 1.;
    advection_velocity[1] = 0.3;

    Triangulation<dim> affine_triangulation;
    GridGenerator::subdivided_hyper_rectangle(affine_triangulation,
                                              {2, 1},
                                              Point<dim>(0., 0.),
                                              Point<dim>(1., 0.25),
                                              true);

    MappingQ<dim> affine_mapping(1);
    FE_Q<dim>     quad_geometry_fe(1);
    QGauss<dim>   quad_quadrature(3);
    const double  affine_quad_difference =
      compare_analytic_and_fevalues_lengths(affine_mapping,
                                            affine_triangulation,
                                            quad_geometry_fe,
                                            quad_quadrature,
                                            true,
                                            advection_velocity,
                                            1.);

    Triangulation<dim> simplex_triangulation;
    GridGenerator::subdivided_hyper_rectangle(simplex_triangulation,
                                              {2, 1},
                                              Point<dim>(0., 0.),
                                              Point<dim>(1., 0.25),
                                              true);
    GridGenerator::convert_hypercube_to_simplex_mesh(simplex_triangulation,
                                                     simplex_triangulation,
                                                     2);

    FE_SimplexP<dim> simplex_geometry_fe(1);
    MappingFE<dim>   simplex_mapping(simplex_geometry_fe);
    QGaussSimplex<dim> simplex_quadrature(3);
    const double       affine_simplex_difference =
      compare_analytic_and_fevalues_lengths(simplex_mapping,
                                            simplex_triangulation,
                                            simplex_geometry_fe,
                                            simplex_quadrature,
                                            false,
                                            advection_velocity,
                                            1.);

    Triangulation<dim> curved_triangulation;
    GridGenerator::subdivided_hyper_rectangle(curved_triangulation,
                                              {2, 1},
                                              Point<dim>(0., 0.),
                                              Point<dim>(1., 0.4),
                                              true);

    FESystem<dim>              position_fe(FE_Q<dim>(2), dim);
    DoFHandler<dim>            position_dof_handler(curved_triangulation);
    FEValuesExtractors::Vector position(0);
    const ComponentMask        position_mask =
      position_fe.component_mask(position);
    Vector<double> position_solution;

    position_dof_handler.distribute_dofs(position_fe);
    position_solution.reinit(position_dof_handler.n_dofs());
    VectorTools::interpolate(position_dof_handler,
                             CurvedPosition(),
                             position_solution,
                             position_mask);

    MappingFEField<dim, dim, Vector<double>> curved_mapping(
      position_dof_handler, position_solution, position_mask);
    const double curved_difference =
      compare_analytic_and_fevalues_lengths(curved_mapping,
                                            curved_triangulation,
                                            quad_geometry_fe,
                                            quad_quadrature,
                                            true,
                                            advection_velocity,
                                            1.);

    const double curved_min_length =
      compute_min_fevalues_length_with_quadratic_geometry();

    AssertThrow(affine_quad_difference < 1.e-12,
                ExcMessage("Analytic and FEValues Q1 lengths differ."));
    AssertThrow(affine_simplex_difference < 1.e-12,
                ExcMessage("Analytic and FEValues P1 simplex lengths differ."));
    AssertThrow(curved_difference < 1.e-12,
                ExcMessage("Analytic and FEValues Q1 lengths differ with a "
                           "quadratic mapping."));
    AssertThrow(std::isfinite(curved_min_length) && curved_min_length > 0.,
                ExcMessage("Invalid streamline length with quadratic mapping."));
    test_tau_polynomial_degree_scaling();

    deallog << std::scientific << std::setprecision(3);
    deallog << "affine Q1 analytic/native max diff = "
            << affine_quad_difference << std::endl;
    deallog << "affine P1 simplex analytic/native max diff = "
            << affine_simplex_difference << std::endl;
    deallog << "quadratic mapping Q1 analytic/native max diff = "
            << curved_difference << std::endl;
    deallog << "quadratic mapping min h_stream = " << curved_min_length
            << std::endl;
    deallog << "tau polynomial degree scaling OK" << std::endl;
    deallog << "OK" << std::endl;
  }
} // namespace

int main()
{
  initlog();
  test_streamline_lengths();

  return 0;
}
