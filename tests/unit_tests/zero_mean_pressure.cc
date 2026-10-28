#include <boundary_conditions.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/vector.h>

#include <cmath>

#include "../tests.h"

using namespace dealii;

void test(const unsigned int degree)
{
  Triangulation<2> triangulation;
  GridGenerator::subdivided_hyper_cube(triangulation, 2);
  GridGenerator::convert_hypercube_to_simplex_mesh(triangulation,
                                                   triangulation,
                                                   2);
  const FESystem<2>      fe(FE_SimplexP<2>(degree + 1),
                       2,
                       FE_SimplexP<2>(degree),
                       1);
  const MappingFE<2>     mapping(FE_SimplexP<2>(1));
  const QGaussSimplex<2> quadrature(4);
  DoFHandler<2>          dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);
  IndexSet relevant = DoFTools::extract_locally_relevant_dofs(dof_handler);
  std::vector<unsigned char> components;
  types::global_dof_index    pivot = numbers::invalid_dof_index;
  std::vector<std::pair<types::global_dof_index, double>> constraint_weights;
  BoundaryConditions::create_zero_mean_pressure_constraints_data(
    triangulation,
    dof_handler,
    relevant,
    components,
    mapping,
    quadrature,
    2,
    pivot,
    constraint_weights);

  const auto initial_pivot = pivot;
  for (unsigned int update = 0; update < 2; ++update)
  {
    if (update == 1)
    {
      GridTools::transform(
        [](const Point<2> &point) {
          Point<2> result = point;
          result[0] += 0.2 * point[0] * (1. - point[0]) * point[1];
          return result;
        },
        triangulation);
      BoundaryConditions::update_zero_mean_pressure_constraint_weights(
        triangulation,
        dof_handler,
        relevant,
        mapping,
        quadrature,
        2,
        pivot,
        constraint_weights);
      AssertThrow(pivot == initial_pivot,
                  ExcMessage("The pressure pivot changed on a mesh update."));
    }

    // Integrate the pressure basis independently of the constraint weights.
    Vector<double>                       integrals(dof_handler.n_dofs());
    FEValues<2>                          values(mapping,
                       fe,
                       quadrature,
                       update_values | update_JxW_values);
    std::vector<types::global_dof_index> dofs(fe.n_dofs_per_cell());
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      values.reinit(cell);
      cell->get_dof_indices(dofs);
      for (unsigned int i = 0; i < dofs.size(); ++i)
        if (fe.system_to_component_index(i).first == 2)
          for (unsigned int q = 0; q < quadrature.size(); ++q)
            integrals[dofs[i]] += values.shape_value(i, q) * values.JxW(q);
    }

    const auto pressure_dofs =
      DoFTools::extract_dofs(dof_handler,
                             fe.component_mask(FEValuesExtractors::Scalar(2)));
    const auto first_pressure = *pressure_dofs.begin();
    if (update == 0 && degree == 1)
      AssertThrow(pivot == first_pressure,
                  ExcMessage("The safe legacy linear-pressure pivot changed."));
    if (update == 0 && degree == 2)
    {
      AssertThrow(std::abs(integrals[first_pressure]) < 1e-14,
                  ExcMessage("The quadratic test must start with a zero "
                             "vertex integration weight."));
      AssertThrow(std::abs(integrals[pivot]) > 1e-3,
                  ExcMessage("A zero quadratic-vertex weight was used as "
                             "pressure pivot."));
      for (const auto &[index, weight] : constraint_weights)
        AssertThrow(std::isfinite(weight) && std::abs(weight) <= 1. + 1e-12,
                    ExcMessage(
                      "The quadratic pressure weights are unbounded."));
    }

    AffineConstraints<double> constraints(relevant);
    BoundaryConditions::add_zero_mean_pressure_constraints(constraints,
                                                           relevant,
                                                           pivot,
                                                           constraint_weights);
    constraints.close();
    Vector<double> pressure(dof_handler.n_dofs());
    for (unsigned int i = 0; i < pressure.size(); ++i)
      pressure[i] = std::cos(0.7 * (i + 1.));
    constraints.distribute(pressure);
    AssertThrow(std::isfinite(pressure.linfty_norm()) &&
                  std::abs(pressure * integrals) < 1e-12,
                ExcMessage(
                  "The distributed pressure does not have zero mean."));
  }
  deallog << "P" << degree
          << " triangular pressure: finite gauge, zero mean and fixed pivot "
             "after mesh update OK"
          << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  test(1);
  test(2);
}
