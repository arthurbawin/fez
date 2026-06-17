
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include "../../tests.h"

#include "parameters.h"
#include "stabilization_tools.h"
// #include "time_handler.h"

#include <deal.II/grid/grid_out.h>

using namespace dealii;

template <int dim>
Triangulation<dim> make_grid(const Point<dim> a, const Point<dim> b)
{
  Triangulation<dim>        tria;
  std::vector<unsigned int> repetitions(dim, 1.);
  GridGenerator::subdivided_hyper_rectangle(tria, repetitions, a, b);
  GridGenerator::convert_hypercube_to_simplex_mesh(tria,
                                                   tria,
                                                   (dim == 2) ? 2u : 6u);
  // GridOut grid_out;
  // grid_out.write_msh(tria, "simplicial_mesh.msh");
  return tria;
}

template <int dim>
void compute_cell_lengths(const Triangulation<dim> &tria,
                          FEValues<dim>            &fe_values,
                          const Tensor<1, dim>     &convective_velocity)
{
  deallog << "Convective velocity " << convective_velocity << std::endl;
  const auto                      &fe            = fe_values.get_fe();
  const auto                      &quadrature    = fe_values.get_quadrature();
  const auto                       dofs_per_cell = fe.n_dofs_per_cell();
  const FEValuesExtractors::Vector velocity(0);
  std::vector<Tensor<1, dim>> grad_phi_u_first_component(fe.n_dofs_per_cell());

  unsigned int i_cell = 0;
  for (const auto &cell : tria.active_cell_iterators())
  {
    fe_values.reinit(cell);

    for (unsigned int q = 0; q < quadrature.size(); ++q)
    {
      // Use the gradients of the shape functions associated with the first
      // velocity component to compute the cell length with Verdier's formula.
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
        grad_phi_u_first_component[k] = fe_values[velocity].gradient(k, q)[0];

      const double h_tau_squared = StabilizationTools::compute_h_tau_squared(
        dofs_per_cell,
        cell->diameter(),
        convective_velocity,
        convective_velocity.norm_square(),
        grad_phi_u_first_component);

      deallog << "h_tau on cell " << i_cell << " = " << std::sqrt(h_tau_squared)
              << std::endl;
    }
    i_cell++;
  }
}

template <int dim>
void test_velocities(const Triangulation<dim> &tria, FEValues<dim> &fe_values)
{
  Tensor<1, dim> convective_velocity;
  if constexpr (dim == 3)
    convective_velocity[2] = 0;
  {
    // Zero velocity. Cell length is the cell diameter.
    convective_velocity[0] = 0;
    convective_velocity[1] = 0;
    compute_cell_lengths(tria, fe_values, convective_velocity);
  }
  {
    // Velocity along x. Cell length for stabilization is the length of the side
    // aligned with the velocity vector.
    convective_velocity[0] = 1;
    convective_velocity[1] = 0;
    compute_cell_lengths(tria, fe_values, convective_velocity);
  }
  {
    // Velocity along x.
    convective_velocity[0] = 5;
    convective_velocity[1] = 0;
    compute_cell_lengths(tria, fe_values, convective_velocity);
  }
  {
    // Velocity along y.
    convective_velocity[0] = 0;
    convective_velocity[1] = 1;
    compute_cell_lengths(tria, fe_values, convective_velocity);
  }
  {
    // Velocity along y.
    convective_velocity[0] = 0;
    convective_velocity[1] = 5;
    compute_cell_lengths(tria, fe_values, convective_velocity);
  }
  {
    // Velocity along x+y.
    convective_velocity[0] = 1;
    convective_velocity[1] = 1;
    compute_cell_lengths(tria, fe_values, convective_velocity);
  }
  {
    // Velocity along x+y.
    convective_velocity[0] = 5;
    convective_velocity[1] = 5;
    compute_cell_lengths(tria, fe_values, convective_velocity);
  }
  {
    // Velocity along x-y.
    convective_velocity[0] = -1;
    convective_velocity[1] = 1;
    compute_cell_lengths(tria, fe_values, convective_velocity);
  }

  if constexpr (dim == 3)
  {
    {
      // Velocity along z.
      convective_velocity[0] = 0;
      convective_velocity[1] = 0;
      convective_velocity[2] = 1;
      compute_cell_lengths(tria, fe_values, convective_velocity);
    }
    {
      // Velocity along x + z.
      convective_velocity[0] = 1;
      convective_velocity[1] = 0;
      convective_velocity[2] = 1;
      compute_cell_lengths(tria, fe_values, convective_velocity);
    }
    {
      // Velocity along y + z.
      convective_velocity[0] = 0;
      convective_velocity[1] = 1;
      convective_velocity[2] = 1;
      compute_cell_lengths(tria, fe_values, convective_velocity);
    }
  }
}

/**
 * Tests the definition of the cell length used to define the stabilization
 * parameter tau_SUPG.
 *
 * Create a few isotropic and anisotropic triangulations with 2 triangles/6
 * tetrahedra each, then compare the cell length for various velocity fields.
 */
template <int dim>
void test_cell_length(const unsigned int velocity_degree)
{
  deallog << std::endl;
  deallog << "Dimension      : " << dim << std::endl;
  deallog << "Velocity degree: " << velocity_degree << std::endl;
  deallog << std::endl;
  FESystem<dim> fe(FE_SimplexP<dim>(velocity_degree), dim);

  MappingFE<dim>     mapping(FE_SimplexP<dim>(1));
  QGaussSimplex<dim> quadrature((dim == 2) ? 2 : 1);
  FEValues<dim>      fe_values(mapping, fe, quadrature, update_gradients);

  /**
   * Isotropic square mesh [0,1]^d.
   */
  {
    Point<dim> corner_a;
    Point<dim> corner_b;
    corner_b[0] = 1.;
    corner_b[1] = 1.;
    if constexpr (dim == 3)
      corner_b[2] = 1.;
    auto tria = make_grid(corner_a, corner_b);
    deallog << "Square mesh [0,1] ^ " << dim << " (isotropic)" << std::endl;
    test_velocities(tria, fe_values);
  }

  /**
   * Anisotropic rectangular mesh [0, 0.1] x [0, 1] x [0,1].
   */
  {
    Point<dim> corner_a;
    Point<dim> corner_b;
    corner_b[0] = 0.1;
    corner_b[1] = 1;
    if constexpr (dim == 2)
      deallog << "Rectangular mesh [0,0.1] x [0,1] (anisotropic)" << std::endl;
    else
    {
      deallog << "Rectangular mesh [0,0.1] x [0,1] x [0,1] (anisotropic)"
              << std::endl;
      corner_b[2] = 1.;
    }
    auto tria = make_grid(corner_a, corner_b);
    test_velocities(tria, fe_values);
  }

  /**
   * Anisotropic rectangular mesh [0,1] x [0, 0.1] x [0,1].
   */
  {
    Point<dim> corner_a;
    Point<dim> corner_b;
    corner_b[0] = 1.;
    corner_b[1] = 0.1;
    if constexpr (dim == 2)
      deallog << "Rectangular mesh [0,1] x [0,0.1] (anisotropic)" << std::endl;
    else
    {
      deallog << "Rectangular mesh [0,1] x [0,0.1] x [0,1] (anisotropic)"
              << std::endl;
      corner_b[2] = 1.;
    }
    auto tria = make_grid(corner_a, corner_b);
    test_velocities(tria, fe_values);
  }

  /**
   * Anisotropic rectangular mesh [0,1] x [0, 1] x [0, 0.1].
   */
  if constexpr (dim == 3)
  {
    Point<dim> corner_a;
    Point<dim> corner_b;
    corner_b[0] = 1.;
    corner_b[1] = 1;
    corner_b[2] = 0.1;
    deallog << "Rectangular mesh [0,1] x [0, 1] x [0,0.1] (anisotropic)"
            << std::endl;
    auto tria = make_grid(corner_a, corner_b);
    test_velocities(tria, fe_values);
  }
}

int main()
{
  initlog();
  test_cell_length<2>(1);
  test_cell_length<2>(2);

  // Definition of h_tau in 3D does not depend on the degree of the solution,
  // so test only for linear velocity.
  test_cell_length<3>(1);

  return 0;
}
