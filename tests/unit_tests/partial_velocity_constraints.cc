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

// Test the constraints actually assembled on a mixed finite element, rather
// than only checking the component mask. The sentinel field exposes accidental
// changes to unconstrained velocity, pressure and temperature components.
template <int dim>
BoundaryConditions::FluidBC<dim> read_bc(const std::string &type,
                                         const int          mask = -1)
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
    prm.set("Function expression", std::to_string(11 * (d + 1)));
    prm.leave_subsection();
  }
  bc.read_parameters(prm);
  return bc;
}

template <int dim>
class ManufacturedField : public Function<dim>
{
public:
  ManufacturedField()
    : Function<dim>(dim + 2)
  {}

  double value(const Point<dim> &p, const unsigned int c = 0) const override
  {
    double result = 100. + c;
    for (unsigned int d = 0; d < dim; ++d)
      result += (d + 1) * p[d];
    return result;
  }
};

template <int dim>
void check_constraints(const std::string &type,
                       const int          mask,
                       const bool         homogeneous,
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

  const auto bc = read_bc<dim>(type, mask);
  const std::map<types::boundary_id, BoundaryConditions::FluidBC<dim>>
                                     boundaries = {{0, bc}};
  const ManufacturedField<dim>       exact;
  const Functions::ZeroFunction<dim> exact_velocity(dim);
  AffineConstraints<double>          constraints;
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
    const unsigned int c           = components[i];
    const bool         is_velocity = c >= u_lower && c < u_lower + dim;
    const bool         at_boundary = std::abs(points.at(i)[0]) < 1e-12;
    const bool         selected =
      is_velocity && (mask < 0 || (mask & (1u << (c - u_lower))));
    const bool should_constrain = at_boundary && selected;
    AssertThrow(constraints.is_constrained(i) == should_constrain,
                ExcMessage(type + ": unexpected constraint on component " +
                           std::to_string(c)));
    double expected = -7. - i;
    if (should_constrain)
    {
      ++expected_count;
      expected = homogeneous || type == "no_slip" ? 0. :
                 type == "input_function"         ? 11. * (c - u_lower + 1) :
                                                    exact.value(points.at(i), c);
    }
    AssertThrow(
      std::isfinite(values[i]) && std::abs(values[i] - expected) < 1e-12,
      ExcMessage(type + ": incorrect value on component " + std::to_string(c)));
  }
  AssertThrow(expected_count > 0 &&
                constraints.n_constraints() == expected_count,
              ExcMessage("The test did not exercise the selected boundary"));
}

template <int dim>
void check_parameter_validation()
{
  const std::array<std::string, 3> strong_types = {
    {"input_function", "no_slip", "velocity_mms"}};
  for (const auto &type : strong_types)
  {
    const auto defaults = read_bc<dim>(type);
    AssertThrow(defaults.constrain_u && defaults.constrain_v &&
                  defaults.constrain_w == (dim == 3),
                ExcInternalError());
    const auto expect_rejected = [&](const int mask) {
      bool rejected = false;
      try
      {
        (void)read_bc<dim>(type, mask);
      }
      catch (const ExceptionBase &error)
      {
        rejected =
          std::string(error.what()).find("at least one velocity component") !=
          std::string::npos;
      }
      AssertThrow(rejected,
                  ExcMessage("Empty strong-velocity mask was accepted"));
    };
    expect_rejected(0);
    if constexpr (dim == 2)
      expect_rejected(4); // constrain_w alone cannot constrain a 2D velocity.
    else
    {
      const auto w_only = read_bc<dim>(type, 4);
      AssertThrow(!w_only.constrain_u && !w_only.constrain_v &&
                    w_only.constrain_w,
                  ExcInternalError());
    }
  }

  // Types not using strong component masks must not acquire a new validation
  // restriction. Their existing slip/traction/weak enforcement stays in place.
  for (const std::string type :
       {"outflow", "slip", "weak_pressure", "weak_no_slip"})
  {
    const auto bc = read_bc<dim>(type, 0);
    AssertThrow(!bc.constrain_u && !bc.constrain_v && !bc.constrain_w,
                ExcInternalError());
  }
  deallog << dim << "d parameter validation OK" << std::endl;
}

template <int dim>
void check_dimension()
{
  check_parameter_validation<dim>();
  for (const std::string type : {"input_function", "no_slip", "velocity_mms"})
  {
    for (unsigned int u_lower : {0u, 1u})
      for (bool homogeneous : {false, true})
      {
        check_constraints<dim>(type, -1, homogeneous, u_lower);
        for (unsigned int mask = 1; mask < (1u << dim); ++mask)
          check_constraints<dim>(type, mask, homogeneous, u_lower);
      }
    deallog << dim << "d " << type << " masks and values OK" << std::endl;
  }
}

int main()
{
  deal_II_exceptions::disable_abort_on_exception();
  initlog();
  check_dimension<2>();
  check_dimension<3>();
}
