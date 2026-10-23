#include <boundary_conditions.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/vector.h>

#include <array>
#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "../tests.h"

enum class Geometry
{
  orthogonal,
  oblique,
  changing_pivot,
  nearly_parallel
};

// Each inner wall of the L-shaped mesh belongs to a different cell. Their
// normals must remain independent at the shared corner and, in 3D, edge.
template <int dim>
bool check_flux_constraints(const Geometry     geometry,
                            const bool         position,
                            const unsigned int n_walls,
                            const std::string &case_name,
                            const bool         reverse_order)
{
  Triangulation<dim> tria;
  GridGenerator::hyper_L(tria, -1., 1., false);
  for (const auto &cell : tria.active_cell_iterators())
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary())
        for (unsigned int d = 0; d < dim; ++d)
          if (std::abs(face->center()[d]) < 1e-12)
            face->set_boundary_id(
              10 + (reverse_order && d < n_walls ? n_walls - 1 - d : d));

  FESystem<dim>   fe(FE_Q<dim>(1), dim);
  DoFHandler<dim> dofs(tria);
  dofs.distribute_dofs(fe);
  MappingQ1<dim> mapping;
  const auto     reference_points =
    DoFTools::map_dofs_to_support_points(mapping, dofs);
  std::vector<unsigned int>            components(dofs.n_dofs());
  std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
  for (const auto &cell : dofs.active_cell_iterators())
  {
    cell->get_dof_indices(indices);
    for (unsigned int i = 0; i < indices.size(); ++i)
      components[indices[i]] = fe.system_to_component_index(i).first;
  }

  const std::array<double, 3> translation = {
    {.4, geometry == Geometry::nearly_parallel ? 0. : .2, .3}};
  const double delta = 1e-8;
  GridTools::transform(
    [&](const Point<dim> &p) {
      auto q = p;
      if (geometry == Geometry::oblique)
      {
        // Both wall normals select the y component as their initial pivot.
        q[0] = p[0] + p[1];
        q[1] = .2 * p[0] + .6 * p[1];
      }
      if (geometry == Geometry::nearly_parallel)
      {
        // The two walls differ by a small, nonzero slope. Both conditions
        // must still fix the corner, even though their normals nearly agree.
        q[0] = p[0] + p[1];
        q[1] = 5e-13 * p[1];
      }
      if constexpr (dim == 3)
        if (geometry == Geometry::changing_pivot)
        {
          // The normals (1,.5,-(1-delta)) and (-1,.5,1) initially
          // select x and z. Elimination must switch the second pivot to y:
          // y + delta*z = 0, rather than z = -y/delta.
          q[0] = .5 * p[0] - .5 * p[1] + (1. - delta / 2.) * p[2];
          q[1] = p[0] + p[1] - delta * p[2];
        }
      for (unsigned int d = 0; d < dim; ++d)
        q[d] += translation[d];
      return q;
    },
    tria);

  Functions::ZeroFunction<dim> zero(dim);
  AffineConstraints<double>    constraints;
  if (position)
  {
    std::map<types::boundary_id, BoundaryConditions::PseudosolidBC<dim>> bcs;
    for (unsigned int d = 0; d < n_walls; ++d)
    {
      auto &bc = bcs[10 + d];
      bc.id    = 10 + d;
      bc.type  = BoundaryConditions::Type::no_flux;
    }
    BoundaryConditions::apply_mesh_position_boundary_conditions<dim>(
      false, 0, dim, dofs, mapping, bcs, zero, zero, constraints);
  }
  else
  {
    std::map<types::boundary_id, BoundaryConditions::FluidBC<dim>> bcs;
    for (unsigned int d = 0; d < n_walls; ++d)
    {
      auto &bc = bcs[10 + d];
      bc.id    = 10 + d;
      bc.type  = BoundaryConditions::Type::slip;
    }
    BoundaryConditions::apply_velocity_boundary_conditions<dim>(
      false, 0, dim, dofs, mapping, bcs, zero, zero, constraints);
  }
  constraints.close();

  Vector<double> values(dofs.n_dofs());
  values = 7.;
  if (geometry == Geometry::changing_pivot)
    // Start from the same tangent coordinate whether x or z is left free.
    // The y component still violates the walls before distributing constraints.
    for (unsigned int i = 0; i < values.size(); ++i)
      if (components[i] == 0)
        values[i] = (position ? translation[0] : 0.) +
                    (1. - delta / 2.) * (7. - (position ? translation[2] : 0.));
  constraints.distribute(values);

  bool success = true;
  if (geometry == Geometry::changing_pivot)
  {
    double largest_coefficient = 0.;
    for (const auto &line : constraints.get_lines())
      for (const auto &[index, weight] : line.entries)
      {
        (void)index;
        if (std::abs(weight) > largest_coefficient)
          largest_coefficient = std::abs(weight);
      }
    if (!(largest_coefficient <= 2.))
    {
      std::cerr << case_name << ": coefficient " << largest_coefficient
                << " amplifies roundoff despite independent wall normals"
                << std::endl;
      success = false;
    }
  }
  unsigned int checked_dofs = 0;
  for (unsigned int i = 0; i < values.size(); ++i)
  {
    const auto &p      = reference_points.at(i);
    const bool  corner = p.norm_square() < 1e-24;
    bool        edge   = false;
    if constexpr (dim == 3)
      edge = n_walls == 2 && (p - Point<dim>(0., 0., 1.)).norm_square() < 1e-24;
    if (corner || edge)
    {
      ++checked_dofs;
      const auto component = components[i];
      // In the first two geometries, two normals leave the z tangent free;
      // three normals fix the corner.
      // Position no_flux preserves the nonzero reference position.
      double expected =
        component >= n_walls ? 7. : (position ? translation[component] : 0.);
      if (geometry == Geometry::changing_pivot && component < 2)
      {
        // These values satisfy both original normal equations, including
        // their nonzero position data, while leaving the edge tangent free.
        const double free_position = 7. - (position ? translation[2] : 0.);
        expected += (component == 0 ? 1. - delta / 2. : -delta) * free_position;
      }
      if (!(std::abs(values[i] - expected) <= 1e-12))
      {
        std::cerr << case_name << ": " << (corner ? "corner" : "edge")
                  << " component " << component << " expected " << expected
                  << ", got " << values[i] << std::endl;
        success = false;
      }
    }
  }
  AssertThrow(checked_dofs == dim * (dim == 3 && n_walls == 2 ? 2 : 1),
              ExcMessage("Missing corner or edge support points"));
  return success;
}

template <int dim>
bool check_dimension()
{
  bool                                                   success    = true;
  const std::array<std::pair<Geometry, const char *>, 4> geometries = {{
    {Geometry::orthogonal, "orthogonal"},
    {Geometry::oblique, "oblique"},
    {Geometry::changing_pivot, "changing pivot"},
    {Geometry::nearly_parallel, "nearly parallel"},
  }};
  for (const auto &[geometry, geometry_name] : geometries)
    for (const bool position : {false, true})
      for (unsigned int n_walls = 2; n_walls <= dim; ++n_walls)
        for (const bool reverse_order : {false, true})
        {
          if ((geometry == Geometry::changing_pivot &&
               (dim != 3 || n_walls != 2)) ||
              (geometry == Geometry::nearly_parallel && dim != 2))
            continue;
          const std::string case_name =
            std::to_string(dim) + "D " + geometry_name + " " +
            (position ? "position " : "fluid ") + std::to_string(n_walls) +
            " walls" + (reverse_order ? " reversed" : "");
          try
          {
            if (!check_flux_constraints<dim>(
                  geometry, position, n_walls, case_name, reverse_order))
              success = false;
          }
          catch (const std::exception &e)
          {
            std::cerr << case_name << ": " << e.what() << std::endl;
            success = false;
          }
        }
  if (success)
    deallog << dim << "D boundary flux constraints OK" << std::endl;
  return success;
}

// Two adjacent faces on a straight wall prescribe the same normal position.
// Shifting only one prescription must be rejected at their shared vertex.
class ShiftedPosition : public Function<2>
{
public:
  explicit ShiftedPosition(const double shift)
    : Function<2>(2)
    , shift(shift)
  {}
  double value(const Point<2> &p, const unsigned int component) const override
  {
    return p[component] + (component == 1 ? shift : 0.);
  }

private:
  const double shift;
};

bool check_redundant_conditions()
{
  bool success = true;
  for (const double length : {1e-6, 1e3, 1e6})
    for (const bool translated : {false, true})
      for (const bool incompatible : {false, true})
      {
        // These translated coordinates reproduce cancellation in normal dot
        // position before the equations reach the constraint-combining helper.
        const double     slope   = translated ? .2931265934486439 : .3;
        const double     shift_x = translated ? 1.5253538488765903 : 0.;
        const double     shift_y = translated ? slope * shift_x : .2;
        Triangulation<2> tria;
        GridGenerator::subdivided_hyper_rectangle(tria,
                                                  std::vector<unsigned int>{2,
                                                                            1},
                                                  Point<2>(-1., 0.),
                                                  Point<2>(1., 1.));
        for (const auto &cell : tria.active_cell_iterators())
          for (const auto &face : cell->face_iterators())
            if (face->at_boundary() && face->center()[1] == 0.)
              face->set_boundary_id(face->center()[0] < 0. ? 10 : 11);
        GridTools::transform(
          [=](const Point<2> &p) {
            return Point<2>(length * (p[0] + shift_x),
                            length * (slope * p[0] + p[1] + shift_y));
          },
          tria);
        FESystem<2>   fe(FE_Q<2>(1), 2);
        DoFHandler<2> dofs(tria);
        dofs.distribute_dofs(fe);
        MappingQ1<2>               mapping;
        Functions::ZeroFunction<2> zero(2);
        ShiftedPosition            position(incompatible ? length * 1e-7 : 0.);
        std::map<types::boundary_id, BoundaryConditions::PseudosolidBC<2>> bcs;
        bcs[10].id   = 10;
        bcs[10].type = BoundaryConditions::Type::no_flux;
        bcs[11].id   = 11;
        bcs[11].type = BoundaryConditions::Type::position_flux_mms;
        AffineConstraints<double> constraints;
        try
        {
          BoundaryConditions::apply_mesh_position_boundary_conditions<2>(
            false, 0, 2, dofs, mapping, bcs, zero, position, constraints);
          constraints.close();
          if (incompatible)
          {
            std::cerr << "Contradictory normal positions accepted at scale "
                      << length << std::endl;
            success = false;
          }
          else
          {
            Vector<double> values(dofs.n_dofs());
            values = length;
            constraints.distribute(values);
            const auto points =
              DoFTools::map_dofs_to_support_points(mapping, dofs);
            std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
            unsigned int                         checked = 0;
            for (const auto &cell : dofs.active_cell_iterators())
            {
              cell->get_dof_indices(indices);
              for (unsigned int i = 0; i < indices.size(); ++i)
                if (points.at(indices[i])
                      .distance(Point<2>(shift_x * length, shift_y * length)) <
                    1e-12 * length)
                {
                  ++checked;
                  const double expected =
                    fe.system_to_component_index(i).first == 0 ?
                      length :
                      (slope + shift_y - slope * shift_x) * length;
                  if (std::abs(values[indices[i]] - expected) > 1e-12 * length)
                  {
                    std::cerr << "Incorrect redundant constraint at scale "
                              << length << std::endl;
                    success = false;
                  }
                }
            }
            AssertThrow(checked == 4, ExcMessage("Missing shared wall vertex"));
          }
        }
        catch (const ExceptionBase &e)
        {
          if (!incompatible ||
              std::string(e.what()).find("Incompatible flux conditions") ==
                std::string::npos)
          {
            std::cerr << "Redundant conditions at scale " << length
                      << (translated ? " translated: " : ": ") << e.what()
                      << std::endl;
            success = false;
          }
        }
      }
  return success;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  initlog();
  const bool success_2d = check_dimension<2>();
  const bool success_3d = check_dimension<3>();
  const bool redundant  = check_redundant_conditions();
  return success_2d && success_3d && redundant ? 0 : 1;
}
