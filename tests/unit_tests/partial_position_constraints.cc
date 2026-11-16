#include <boundary_conditions.h>
#include <deal.II/dofs/dof_tools.h>
#ifdef DEAL_II_WITH_P4EST
#  include <deal.II/distributed/tria.h>
#  include <deal.II/lac/la_parallel_vector.h>
#endif
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/vector.h>

#include <array>
#include <cmath>
#include <map>
#include <string>
#include <vector>

#include "../tests.h"

// Test the constraints actually assembled on a mixed finite element, rather
// than only checking the component mask. The sentinel field exposes accidental
// changes to free position components and other fields in the mixed system.
// Adapted from the partial_velocity_constraints regression test in PR120.
template <int dim>
BoundaryConditions::PseudosolidBC<dim> read_bc(const std::string &type,
                                               const int          mask = -1)
{
  BoundaryConditions::PseudosolidBC<dim> bc;
  ParameterHandler                       prm;
  bc.declare_parameters(prm);
  prm.set("id", "0");
  prm.set("type", type);
  if (mask >= 0)
  {
    prm.set("constrain_x", mask & 1 ? "true" : "false");
    prm.set("constrain_y", mask & 2 ? "true" : "false");
    prm.set("constrain_z", mask & 4 ? "true" : "false");
  }
  const std::array<std::string, 3> names = {{"x", "y", "z"}};
  for (unsigned int d = 0; d < names.size(); ++d)
  {
    prm.enter_subsection(names[d]);
    prm.set("Function expression", std::to_string(11 * (d + 1)) + "+x+2*y+t");
    prm.leave_subsection();
  }
  bc.read_parameters(prm);
  bc.set_time(0.5);
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
                       const unsigned int x_lower)
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
  const std::map<types::boundary_id, BoundaryConditions::PseudosolidBC<dim>>
                                     boundaries = {{0, bc}};
  const ManufacturedField<dim>       exact;
  const Functions::ZeroFunction<dim> exact_position(dim);
  AffineConstraints<double>          constraints;
  BoundaryConditions::apply_mesh_position_boundary_conditions(homogeneous,
                                                              x_lower,
                                                              dim + 2,
                                                              dofs,
                                                              mapping,
                                                              boundaries,
                                                              exact,
                                                              exact_position,
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
    const bool         is_position = c >= x_lower && c < x_lower + dim;
    const bool         at_boundary = std::abs(points.at(i)[0]) < 1e-12;
    const bool         selected =
      is_position &&
      (type != "input_function" || mask < 0 || (mask & (1u << (c - x_lower))));
    const bool should_constrain = at_boundary && selected;
    AssertThrow(constraints.is_constrained(i) == should_constrain,
                ExcMessage(type + ": unexpected constraint on component " +
                           std::to_string(c)));
    double expected = -7. - i;
    if (should_constrain)
    {
      ++expected_count;
      expected = homogeneous     ? 0. :
                 type == "fixed" ? points.at(i)[c - x_lower] :
                 type == "input_function" ?
                                   11. * (c - x_lower + 1) + points.at(i)[0] +
                                     2. * points.at(i)[1] + 0.5 :
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
  const auto expect_rejected = [&](const int mask) {
    bool rejected = false;
    try
    {
      (void)read_bc<dim>("input_function", mask);
    }
    catch (const ExceptionBase &error)
    {
      rejected =
        std::string(error.what()).find("at least one position component") !=
        std::string::npos;
    }
    AssertThrow(rejected,
                ExcMessage("Empty prescribed-position mask was accepted"));
  };
  expect_rejected(0);
  if constexpr (dim == 2)
    expect_rejected(4); // constrain_z alone cannot constrain a 2D position.

  // Component flags apply only to input_function, as in PR101.
  for (const std::string type : {"fixed",
                                 "position_mms",
                                 "no_flux",
                                 "coupled_to_fluid",
                                 "position_flux_mms"})
    (void)read_bc<dim>(type, 0);
  deallog << dim << "d parameter validation OK" << std::endl;
}

// Check the assembled position values, including corners where different
// boundary equations initially choose the same constrained component.
template <int dim>
void check_flux_intersection(const double       slope,
                             const bool         homogeneous,
                             const double       translation   = 0.,
                             const bool         reverse_order = false,
                             const std::string &full_type     = "",
                             const std::string &flux_type     = "no_flux",
                             const bool         second_wall   = false,
                             const bool         incompatible  = false)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0., 1., true);
  tria.refine_global(1);
  GridTools::transform(
    [&](const Point<dim> &p) {
      Point<dim> q = p;
      q[0] += translation;
      q[1] += slope * p[0] + 2. * translation;
      if constexpr (dim == 3)
        q[2] += 3. * p[0] + translation;
      return q;
    },
    tria);
  // Reverse the boundary IDs without changing their geometry.
  if (reverse_order)
    for (const auto &cell : tria.active_cell_iterators())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary())
          face->set_boundary_id(2 * dim - 1 - face->boundary_id());
  const auto boundary_id = [&](const unsigned int id) {
    return reverse_order ? 2 * dim - 1 - id : id;
  };
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

  const bool prescribed_field = full_type == "prescribed_field";
  auto       prescribed       = read_bc<dim>(
    full_type.empty() || prescribed_field ? "input_function" : full_type,
    full_type.empty() || prescribed_field ? (incompatible ? 2 : 1) : -1);
  prescribed.id = boundary_id(0);
  auto wall     = read_bc<dim>(flux_type, 0);
  wall.id       = boundary_id(2);
  std::map<types::boundary_id, BoundaryConditions::PseudosolidBC<dim>>
    boundaries = {{prescribed.id, prescribed}, {wall.id, wall}};
  if (!full_type.empty())
  {
    // Activate partial constraints on the opposite side as well, so full
    // Dirichlet precedence is checked while using the combined algorithm.
    auto partial = read_bc<dim>("input_function", 1);
    partial.id   = boundary_id(1);
    boundaries.emplace(partial.id, partial);
  }
  if constexpr (dim == 3)
    if (second_wall)
    {
      wall.id = boundary_id(4);
      boundaries.emplace(wall.id, wall);
    }
  const ManufacturedField<dim>           exact;
  const Functions::ConstantFunction<dim> exact_position(
    std::vector<double>(dim, 4.));
  AffineConstraints<double> constraints;
  if (prescribed_field)
    for (unsigned int i = 0; i < dofs.n_dofs(); ++i)
      if (components[i] >= 1 && components[i] <= dim)
      {
        constraints.add_line(i);
        constraints.set_inhomogeneity(
          i, homogeneous ? 0. : exact.value(points.at(i), components[i]));
      }
  bool rejected = false;
  try
  {
    BoundaryConditions::apply_mesh_position_boundary_conditions(homogeneous,
                                                                1,
                                                                dim + 2,
                                                                dofs,
                                                                mapping,
                                                                boundaries,
                                                                exact,
                                                                exact_position,
                                                                constraints);
    constraints.close();
  }
  catch (const ExceptionBase &error)
  {
    if (!incompatible)
      throw;
    rejected = std::string(error.what()).find("Incompatible flux conditions") !=
               std::string::npos;
  }
  AssertThrow(rejected == incompatible,
              ExcMessage(
                "Incompatible partial position/flux was not rejected"));
  if (incompatible)
    return;

  Vector<double> values(dofs.n_dofs());
  for (unsigned int i = 0; i < values.size(); ++i)
    values[i] = -7. - i;
  constraints.distribute(values);
  std::map<Point<dim>, Tensor<1, dim>, PointComparator<dim>> positions;
  unsigned int                                               checked = 0;
  for (unsigned int i = 0; i < values.size(); ++i)
  {
    const auto &p           = points.at(i);
    const auto  c           = components[i];
    const bool  is_position = c >= 1 && c <= dim;
    if (is_position)
    {
      positions[p][c - 1] = values[i];
      if (prescribed_field)
        AssertThrow(std::abs(values[i] -
                             (homogeneous ? 0. : exact.value(p, c))) < 1e-10,
                    ExcMessage("An incoming position constraint was changed"));
    }
    else
      AssertThrow(values[i] == -7. - i && !constraints.is_constrained(i),
                  ExcMessage("A non-position field was constrained"));
    const bool at_left   = std::abs(p[0] - translation) < 1e-12;
    const bool at_bottom = std::abs(p[1] - 2. * translation) < 1e-12;
    if (at_left && at_bottom)
    {
      const double x            = homogeneous ? 0. : 11.5 + p[0] + 2. * p[1];
      const double normal_value = homogeneous ? 0. :
                                  flux_type == "no_flux" ?
                                                (2. - slope) * translation :
                                                4. * (1. - slope);
      double       expected     = c == 1 ? x :
                                  c == 2 ? slope * x + normal_value :
                                           -7. - i;
      if constexpr (dim == 3)
        if (second_wall && c == 3 && std::abs(p[2] - translation) < 1e-12)
          expected = 3. * x + (homogeneous            ? 0. :
                               flux_type == "no_flux" ? -2. * translation :
                                                        -8.);
      if (!full_type.empty() && is_position)
        expected = homogeneous          ? 0. :
                   full_type == "fixed" ? p[c - 1] :
                   (full_type == "position_mms" || prescribed_field) ?
                                          exact.value(p, c) :
                                          11. * c + p[0] + 2. * p[1] + 0.5;
      AssertThrow(
        std::isfinite(values[i]) && std::abs(values[i] - expected) < 1e-10,
        ExcMessage("Incorrect partial position/flux intersection value: dim=" +
                   std::to_string(dim) + ", component=" + std::to_string(c) +
                   ", slope=" + std::to_string(slope) +
                   ", translation=" + std::to_string(translation) +
                   ", homogeneous=" + std::to_string(homogeneous) +
                   ", actual=" + std::to_string(values[i]) +
                   ", expected=" + std::to_string(expected)));
      ++checked;
    }
  }
  AssertThrow(checked > 0, ExcInternalError());
  for (const auto &[p, value] : positions)
  {
    if (prescribed_field ||
        (!full_type.empty() && std::abs(p[0] - translation) < 1e-12))
      continue; // Fully prescribed boundaries retain their existing priority.
    if (std::abs(p[1] - slope * (p[0] - translation) - 2. * translation) <
        1e-12)
    {
      const double target = homogeneous ? 0. :
                            flux_type == "no_flux" ?
                                          (2. - slope) * translation :
                                          4. * (1. - slope);
      AssertThrow(std::abs(value[1] - slope * value[0] - target) < 1e-10,
                  ExcMessage("The wall-normal position was not preserved"));
    }
    if constexpr (dim == 3)
      if (second_wall &&
          std::abs(p[2] - 3. * (p[0] - translation) - translation) < 1e-12)
      {
        const double target = homogeneous            ? 0. :
                              flux_type == "no_flux" ? -2. * translation :
                                                       -8.;
        AssertThrow(std::abs(value[2] - 3. * value[0] - target) < 1e-10,
                    ExcMessage(
                      "The second wall-normal position was not preserved"));
      }
  }
}

#ifdef DEAL_II_WITH_P4EST
// Exercise locally owned/relevant constraints on a genuinely partitioned mesh.
// This also checks the full-boundary DoF selection across MPI partitions.
template <int dim>
void check_parallel_flux_intersection(const bool homogeneous,
                                      const bool full_boundary)
{
  const MPI_Comm                            comm = MPI_COMM_WORLD;
  parallel::distributed::Triangulation<dim> tria(comm);
  GridGenerator::hyper_cube(tria, 0., 1., true);
  tria.refine_global(2);
  GridTools::transform(
    [](const Point<dim> &p) {
      Point<dim> q = p;
      q[0] += 3.;
      q[1] += 2. * p[0] + 7.;
      return q;
    },
    tria);
  const FESystem<dim> fe(FE_Q<dim>(1), dim + 2);
  DoFHandler<dim>     dofs(tria);
  dofs.distribute_dofs(fe);
  const MappingQ1<dim> mapping;
  const auto points = DoFTools::map_dofs_to_support_points(mapping, dofs);
  std::vector<unsigned int>            components(dofs.n_dofs());
  std::vector<types::global_dof_index> indices(fe.n_dofs_per_cell());
  for (const auto &cell : dofs.active_cell_iterators())
    if (!cell->is_artificial())
    {
      cell->get_dof_indices(indices);
      for (unsigned int i = 0; i < indices.size(); ++i)
        components[indices[i]] = fe.system_to_component_index(i).first;
    }
  auto prescribed = read_bc<dim>("input_function", full_boundary ? -1 : 1);
  auto wall       = read_bc<dim>("no_flux", 0);
  wall.id         = 2;
  std::map<types::boundary_id, BoundaryConditions::PseudosolidBC<dim>>
    boundaries = {{0, prescribed}, {2, wall}};
  if (full_boundary)
  {
    auto partial = read_bc<dim>("input_function", 1);
    partial.id   = 1;
    boundaries.emplace(1, partial);
  }
  const ManufacturedField<dim>       exact;
  const Functions::ZeroFunction<dim> exact_position(dim);
  const IndexSet                     owned = dofs.locally_owned_dofs();
  IndexSet                           relevant;
  DoFTools::extract_locally_relevant_dofs(dofs, relevant);
  AffineConstraints<double> constraints(owned, relevant);
  BoundaryConditions::apply_mesh_position_boundary_conditions(homogeneous,
                                                              1,
                                                              dim + 2,
                                                              dofs,
                                                              mapping,
                                                              boundaries,
                                                              exact,
                                                              exact_position,
                                                              constraints);
  constraints.close();
  LinearAlgebra::distributed::Vector<double> values(owned, comm);
  for (const auto i : owned)
    values[i] = -7. - i;
  constraints.distribute(values);
  unsigned int checked = 0;
  for (const auto i : owned)
  {
    const auto  c = components[i];
    const auto &p = points.at(i);
    if (c == 0 || c > dim)
      AssertThrow(values[i] == -7. - i,
                  ExcMessage("An unrelated MPI field changed"));
    if (std::abs(p[0] - 3.) < 1e-12 && std::abs(p[1] - 7.) < 1e-12)
    {
      double expected = c == 1 ? (homogeneous ? 0. : 28.5) :
                        c == 2 ? (homogeneous ? 0. : 58.) :
                                 -7. - i;
      if (full_boundary && c >= 1 && c <= dim)
        expected = homogeneous ? 0. : 11. * c + p[0] + 2. * p[1] + 0.5;
      AssertThrow(std::abs(values[i] - expected) < 1e-10,
                  ExcMessage(
                    "Incorrect distributed partial position/flux value"));
      ++checked;
    }
  }
  AssertThrow(Utilities::MPI::sum(checked, comm) > 0, ExcInternalError());
}
#endif

template <int dim>
void check_dimension()
{
  for (const std::string type : {"input_function", "fixed", "position_mms"})
  {
    for (unsigned int x_lower : {0u, 1u})
      for (bool homogeneous : {false, true})
      {
        check_constraints<dim>(type, -1, homogeneous, x_lower);
        for (unsigned int mask = (type == "input_function" ? 1 : 0);
             mask < (1u << dim);
             ++mask)
          check_constraints<dim>(type, mask, homogeneous, x_lower);
      }
    deallog << dim << "d " << type << " masks and values OK" << std::endl;
  }
  check_parameter_validation<dim>();
  for (const double slope : {0., 0.5, 2., -2.})
    for (const bool homogeneous : {false, true})
      for (const double translation : {0., 3.})
        for (const bool reverse_order : {false, true})
          check_flux_intersection<dim>(slope,
                                       homogeneous,
                                       translation,
                                       reverse_order);
  for (const bool homogeneous : {false, true})
  {
    for (const std::string type :
         {"fixed", "input_function", "position_mms", "prescribed_field"})
      check_flux_intersection<dim>(2., homogeneous, 3., false, type);
    check_flux_intersection<dim>(
      2., homogeneous, 3., false, "", "position_flux_mms");
    if constexpr (dim == 3)
      check_flux_intersection<dim>(
        2., homogeneous, 3., false, "", "no_flux", true);
  }
  check_flux_intersection<dim>(
    0., false, 3., false, "", "no_flux", false, true);
#ifdef DEAL_II_WITH_P4EST
  for (const bool homogeneous : {false, true})
    for (const bool full_boundary : {false, true})
      check_parallel_flux_intersection<dim>(homogeneous, full_boundary);
#endif
  deallog << dim << "d combined partial position and flux constraints OK"
          << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  deallog.depth_console(0);
  mpi_initlog();
  check_dimension<2>();
  check_dimension<3>();
}
