#include <boundary_conditions.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/vector.h>

#include <array>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "../tests.h"

// The angular velocity is scalar (2+t) in 2D and a non-axis-aligned vector
// (1+t, 2-t, 3+2t) in 3D. The center is (0.25,-0.5[,0.75]). Expected velocities
// below are expanded polynomials, independent of the production cross product.
template <int dim>
BoundaryConditions::FluidBC<dim> make_bc(const std::string &type,
                                         const bool         enabled = true,
                                         const bool zero_rotation   = false,
                                         const int  mask            = -1)
{
  BoundaryConditions::FluidBC<dim> bc;
  ParameterHandler                 prm;
  bc.declare_parameters(prm);
  prm.set("id", "0");
  prm.set("type", type);
  if (mask >= 0)
  {
    prm.set("constrain_u", mask & 1 ? "true" : "false");
    prm.set("constrain_v", mask & 2 ? "true" : "false");
    prm.set("constrain_w", mask & 4 ? "true" : "false");
  }
  const std::array<std::string, 3> names = {{"u", "v", "w"}};
  for (unsigned int d = 0; d < names.size(); ++d)
  {
    prm.enter_subsection(names[d]);
    prm.set("Function expression",
            std::to_string(11 * (d + 1)) + "+" + std::to_string(d + 1) + "*t");
    prm.leave_subsection();
  }
  prm.enter_subsection("rigid body rotation");
  prm.set("enable", enabled ? "true" : "false");
  prm.set("center of rotation", dim == 2 ? "0.25, -0.5" : "0.25, -0.5, 0.75");
  prm.enter_subsection("angular velocity");
  if constexpr (dim == 2)
    prm.set("Function expression", zero_rotation ? "0" : "2+t");
  else
    prm.set("Function expression",
            zero_rotation ? "0; 0; 0" : "1+t; 2-t; 3+2*t");
  prm.leave_subsection();
  prm.leave_subsection();
  bc.read_parameters(prm);
  return bc;
}

template <int dim>
double
expected_rotation(const Point<dim> &p, const unsigned int d, const double t)
{
  if constexpr (dim == 2)
    return d == 0 ? -2. * p[1] - t * p[1] - 1. - 0.5 * t :
                    2. * p[0] + t * p[0] - 0.5 - 0.25 * t;
  else
  {
    if (d == 0)
      return 2. * p[2] - t * p[2] - 3. * p[1] - 2. * t * p[1] - 3. - 0.25 * t;
    if (d == 1)
      return 3. * p[0] + 2. * t * p[0] - p[2] - t * p[2] + 0.25 * t;
    return p[1] + t * p[1] - 2. * p[0] + t * p[0] + 1. + 0.25 * t;
  }
}

void assert_value(const double       actual,
                  const double       expected,
                  const std::string &message)
{
  AssertThrow(std::isfinite(actual) && std::abs(actual - expected) < 1e-12,
              ExcMessage(message));
}

template <int dim>
void check_function()
{
  auto       bc = make_bc<dim>("no_slip");
  Point<dim> sample;
  for (unsigned int d = 0; d < dim; ++d)
    sample[d] = 0.6 + 0.2 * d;
  Point<dim> center;
  center[0] = 0.25;
  center[1] = -0.5;
  if constexpr (dim == 3)
    center[2] = 0.75;
  for (unsigned int u_lower : {0u, 1u})
  {
    const RigidBodyVelocity<dim> rotation(u_lower,
                                          dim + 2,
                                          bc.center_of_rotation,
                                          *bc.angular_velocity);
    for (const double time : {0., 0.5, 2.})
    {
      bc.set_time(time);
      for (unsigned int c = 0; c < dim + 2; ++c)
      {
        const bool velocity = c >= u_lower && c < u_lower + dim;
        assert_value(rotation.value(sample, c),
                     velocity ? expected_rotation(sample, c - u_lower, time) :
                                0.,
                     "Rigid-body function value or scalar component mismatch");
        assert_value(rotation.value(center, c),
                     0.,
                     "Velocity must vanish at the rotation center");
      }
    }
  }
  deallog << dim << "d rotation function, center and time OK" << std::endl;
}

template <int dim>
void check_constraints(const std::string &type,
                       const bool         homogeneous,
                       const bool         enabled,
                       const bool         zero_rotation,
                       const unsigned int u_lower)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0., 1., true);
  const FESystem<dim> fe(FE_Q<dim>(1), dim + 2);
  DoFHandler<dim>     dofs(tria);
  dofs.distribute_dofs(fe);
  const MappingQ1<dim> mapping;
  const auto points = DoFTools::map_dofs_to_support_points(mapping, dofs);
  std::vector<unsigned int>            components(dofs.n_dofs());
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  for (const auto &cell : dofs.active_cell_iterators())
  {
    cell->get_dof_indices(indices);
    for (unsigned int i = 0; i < indices.size(); ++i)
      components[indices[i]] = fe.system_to_component_index(i).first;
  }
  std::map<types::boundary_id, BoundaryConditions::FluidBC<dim>> boundaries = {
    {0, make_bc<dim>(type, enabled, zero_rotation)}};
  const Functions::ZeroFunction<dim> exact(dim + 2), exact_velocity(dim);
  for (const double time : {0., 0.5, 2.})
  {
    boundaries.at(0).set_time(time);
    AffineConstraints<double> constraints;
    BoundaryConditions::apply_velocity_boundary_conditions(homogeneous,
                                                           u_lower,
                                                           dim + 2,
                                                           dofs,
                                                           mapping,
                                                           boundaries,
                                                           exact,
                                                           exact_velocity,
                                                           constraints);
    constraints.close();
    Vector<double> values(dofs.n_dofs());
    for (unsigned int i = 0; i < values.size(); ++i)
      values[i] = -7. - i;
    constraints.distribute(values);
    unsigned int expected_count = 0;
    for (unsigned int i = 0; i < values.size(); ++i)
    {
      const unsigned int c        = components[i];
      const bool         velocity = c >= u_lower && c < u_lower + dim;
      const bool         should_constrain =
        velocity && std::abs(points.at(i)[0]) < 1e-12;
      AssertThrow(constraints.is_constrained(i) == should_constrain,
                  ExcMessage(
                    type +
                    ": unexpected constraint or scalar-field constraint"));
      double expected = -7. - i;
      if (should_constrain)
      {
        ++expected_count;
        const unsigned int d = c - u_lower;
        if (homogeneous)
          expected = 0.;
        else if (enabled)
          expected =
            zero_rotation ? 0. : expected_rotation(points.at(i), d, time);
        else
          expected = type == "no_slip" ? 0. : 11. * (d + 1) + (d + 1) * time;
      }
      assert_value(values[i],
                   expected,
                   type +
                     ": incorrect rotating, homogeneous or disabled value");
    }
    AssertThrow(expected_count > 0 &&
                  constraints.n_constraints() == expected_count,
                ExcInternalError());
  }
}

template <int dim>
void check_parameter_validation()
{
  for (const std::string type : {"no_slip", "input_function"})
    for (unsigned int d = 0; d < dim; ++d)
    {
      // A nonempty mask accepted by the parent PR must be rejected when
      // rotation needs the missing velocity component.
      const unsigned int partial_mask = ((1u << dim) - 1) & ~(1u << d);
      bool               rejected     = false;
      try
      {
        (void)make_bc<dim>(type, true, false, partial_mask);
      }
      catch (const ExceptionBase &error)
      {
        rejected = std::string(error.what())
                     .find("rigid body rotation requires all velocity") !=
                   std::string::npos;
      }
      AssertThrow(rejected,
                  ExcMessage("Partial rotating velocity mask was accepted"));
      (void)make_bc<dim>(type, false, false, partial_mask);
    }
  // Keep the existing weak-rotation path compatible with ignored component
  // masks; the new restriction concerns only the two strong boundary types.
  (void)make_bc<dim>("weak_no_slip", true, false, 0);
  deallog << dim << "d rotating-mask validation OK" << std::endl;
}

template <int dim>
void check_dimension()
{
  check_function<dim>();
  check_parameter_validation<dim>();
  for (const std::string type : {"no_slip", "input_function"})
  {
    for (const bool homogeneous : {false, true})
      for (const bool enabled : {false, true})
        for (const bool zero_rotation : {false, true})
          for (const unsigned int u_lower : {0u, 1u})
            check_constraints<dim>(
              type, homogeneous, enabled, zero_rotation, u_lower);
    deallog << dim << "d " << type
            << " rotating, homogeneous and disabled constraints OK"
            << std::endl;
  }
}

int main()
{
  deal_II_exceptions::disable_abort_on_exception();
  initlog();
  check_dimension<2>();
  check_dimension<3>();
}
