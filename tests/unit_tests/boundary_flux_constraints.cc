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
#include <functional>
#include <iomanip>
#include <map>
#include <set>
#include <sstream>
#include <string>
#include <vector>

#include "../tests.h"

// Note: this test was mostly AI-generated, but was reviewed, edited and tested.

// This test compares three ways of building the normal flux constraints at the
// corners and edges where slip (velocity) or no_flux (mesh position) walls
// meet:
//
//  - all walls: a single deal.II call with all wall ids (behaviour before the
//               fix). deal.II averages the normals of walls that belong to
//               different cells. At the interface of e.g. two slip conditions,
//               this yields a relation on the vector components
//               instead of setting these to zero.
//
//  - per wall:  one deal.II call per wall into the same constraints. deal.II
//               skips a line whose pivot is already constrained, so a second
//               slip constraint will be silently ignored.
//
//  - fix:       The current implementation of
//  BoundaryConditions::apply_*_boundary_conditions.
//               When adding a flux condition, the function looks for existing
//               constraints and substitute the new one into it, resolving
//               chains if needed. It never creates any cycle, as the pivot
//               is chosen among free DoFs.
//
// For each corner or edge node, the table reports the number of free vector
// components, the largest residual of the exact wall equations n_k.(u - g_k)
// after distributing a constant vector, and the largest constraint weight.
//
// Debug and release give the same table: none of deal.II's debug-only checks
// fire on these geometries, and cycles in the constraints are detected here
// (see has_cycle) rather than in close(), which checks them only in debug.

// All geometries start from the reference L-shaped domain of hyper_L, then
// apply q = A p + t (see linear_map). The drawings omit the translation t.
// Normals n are outward, i.e. they point into the removed region, and
// "pivot" is the component deal.II picks when it treats each wall on its own.
//
// Reference, 2D:
//
//        y
//      1 +-------+
//        |       |
//        |       | 10       removed
//        |       |
//      0 |       *-------+
//        |          11   |
//        |               |
//     -1 +---------------+ x
//       -1       0       1
//
//   Wall 10 (x = 0) and wall 11 (y = 0) are faces of two different cells, so
//   a single deal.II call averages their normals at the corner *.
//   reverse_order swaps the two ids.
//
// Reference, 3D, looking into the corner from the removed octant [0,1]^3:
//
//                   z
//                   |
//                   + (0,0,1): edge node, checked with two walls
//         11        |        10
//       (y = 0)     |      (x = 0)
//                   |
//                   * (0,0,0): corner node
//                 /   \
//               /  12   \
//             /  (z = 0)  \
//            x             y
//
//   With two walls only 10 and 11 are slip, and they share the edge along z.
//   With three walls, 12 is also slip and all three meet at the corner.
//
// orthogonal: A = I. The drawings above, as is.
//   n10 = (1, 0, 0) pivot x, n11 = (0, 1, 0) pivot y, n12 = (0, 0, 1) pivot z.
//   Distinct pivots, so the per-wall calls do not conflict.
//
// oblique: x' = x + y, y' = .2 x + .6 y (z unchanged in 3D). In 2D, and in
// every z = const section in 3D:
//
//                       (1, .6)
//                      /
//                 10  /
//                    /    removed       ___ (1, .2)
//      domain       /          ___---'''
//                  /   ___---'''   11
//                 *---'
//                       domain
//
//   The removed wedge is about 20 degrees wide; the domain wraps around it.
//   n10 ~ (.6, -1) pivot y,  n11 ~ (-.2, 1) pivot y,  n12 = (0, 0, 1) pivot z.
//   Walls 10 and 11 pick the same pivot, so the per-wall calls drop one.
//
// nearly_parallel (2D only): x' = x + y, y' = 5e-13 y. The whole domain is
// flattened to a height of about 1e-12 and both walls run along +x:
//
//                    10: (0, 0) -> (1, 5e-13)
//            ____________________________
//   domain  *____________________________   removed sliver (angle 5e-13 rad,
//                                            exaggerated here)
//                    11: (0, 0) -> (1, 0)
//
//   n10 ~ (5e-13, -1) pivot y,  n11 = (0, 1) pivot y. The outward normals are
//   nearly opposite, so their average is close to zero. The two conditions
//   are still independent and must fix both components at the corner.
//
// changing_pivot (3D, two walls): the edge along z becomes the direction
// (1 - delta/2, -delta, 1), with delta = 1e-8. Section normal to the edge,
// with u = (1, 0, -1)/sqrt(2) and y in the plane of the page:
//
//                 y
//           10    ^    11
//             \   |   /
//              \  |  /     removed wedge, about 39 degrees
//               \ | /
//                \|/
//      domain     *      domain
//
//   n10 = (1, .5, -(1 - delta)) pivot x, since |x| beats |z| by delta.
//   n11 = (-1, .5, 1) pivot z, since x and z tie.
//   Substituting wall 10 into wall 11 leaves y + delta z = 0, which must use
//   pivot y. Keeping pivot z would give z = -y / delta.

enum class Geometry
{
  orthogonal,
  oblique,
  changing_pivot,
  nearly_parallel
};

enum class Method
{
  all_walls,
  per_wall,
  fix
};

const char *method_name(const Method method)
{
  switch (method)
  {
    case Method::all_walls:
      return "all walls";
    case Method::per_wall:
      return "per wall ";
    default:
      return "fix      ";
  }
}

std::string format(const double value, const double scale = 1.)
{
  std::ostringstream stream;
  stream << std::scientific << std::setprecision(1)
         << (std::abs(value) < 1e-10 * scale ? 0. : value);
  return stream.str();
}

// Affine maps to transform the mesh, written as q = A*p + t.
template <int dim>
Tensor<2, dim> linear_map(const Geometry geometry)
{
  const double   delta = 1e-8;
  Tensor<2, dim> A;
  for (unsigned int d = 0; d < dim; ++d)
    A[d][d] = 1.;
  if (geometry == Geometry::oblique)
  {
    A[0][0] = 1.;
    A[0][1] = 1.;
    A[1][0] = .2;
    A[1][1] = .6;
  }
  if (geometry == Geometry::nearly_parallel)
  {
    A[0][0] = 1.;
    A[0][1] = 1.;
    A[1][0] = 0.;
    A[1][1] = 5e-13;
  }
  if constexpr (dim == 3)
    if (geometry == Geometry::changing_pivot)
    {
      A[0][0] = .5;
      A[0][1] = -.5;
      A[0][2] = 1. - delta / 2.;
      A[1][0] = 1.;
      A[1][1] = 1.;
      A[1][2] = -delta;
    }
  return A;
}

template <int dim>
void apply_flux_constraints(const Method                        method,
                            const bool                          position,
                            const std::set<types::boundary_id> &walls,
                            const DoFHandler<dim>              &dofs,
                            const Mapping<dim>                 &mapping,
                            AffineConstraints<double>          &constraints)
{
  Functions::ZeroFunction<dim> zero(dim);
  if (method == Method::fix)
  {
    if (position)
    {
      std::map<types::boundary_id, BoundaryConditions::PseudosolidBC<dim>> bcs;
      for (const auto id : walls)
      {
        bcs[id].id   = id;
        bcs[id].type = BoundaryConditions::Type::no_flux;
      }
      BoundaryConditions::apply_mesh_position_boundary_conditions<dim>(
        false, 0, dim, dofs, mapping, bcs, zero, zero, constraints);
    }
    else
    {
      std::map<types::boundary_id, BoundaryConditions::FluidBC<dim>> bcs;
      for (const auto id : walls)
      {
        bcs[id].id   = id;
        bcs[id].type = BoundaryConditions::Type::slip;
      }
      BoundaryConditions::apply_velocity_boundary_conditions<dim>(
        false, 0, dim, dofs, mapping, bcs, zero, zero, constraints);
    }
    return;
  }

  FixedMeshPosition<dim> fixed_mesh(0, dim);

  /**
   * Add to @p constraints the flux constraints associated with the boundary/ies
   * stored in @p ids
   */
  const auto add = [&](const std::set<types::boundary_id> &ids) {
    if (position)
    {
      std::map<types::boundary_id, const Function<dim> *> functions;
      for (const auto id : ids)
        functions[id] = &fixed_mesh;
      VectorTools::compute_nonzero_normal_flux_constraints(
        dofs, 0, ids, functions, constraints, mapping, false);
    }
    else
      VectorTools::compute_no_normal_flux_constraints(
        dofs, 0, ids, constraints, mapping, false);
  };

  if (method == Method::all_walls)
    add(walls);
  else
    for (const auto id : walls)
      add({id});
}

// deal.II detects cycles in the constraints only in debug mode. In release
// mode, close() keeps substituting them forever. Check beforehand, so that
// both modes report the same thing.
bool has_cycle(const AffineConstraints<double> &constraints)
{
  // 1: on the current path, 2: no cycle reachable from this line.
  std::map<types::global_dof_index, unsigned int>    state;
  const std::function<bool(types::global_dof_index)> visit =
    [&](const types::global_dof_index index) {
      unsigned int &index_state = state[index];
      if (index_state != 0)
        return index_state == 1;
      index_state = 1;
      for (const auto &[other, weight] :
           *constraints.get_constraint_entries(index))
      {
        (void)weight;
        if (constraints.is_constrained(other) && visit(other))
          return true;
      }
      state[index] = 2;
      return false;
    };
  for (const auto &line : constraints.get_lines())
    if (visit(line.index))
      return true;
  return false;
}

/**
 * Tests the application of the flux constraints for each geometry and each
 * method, and writes a summary.
 */
template <int dim>
void compare_geometry(const Geometry     geometry,
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

  const std::array<double, 3> translation_values = {
    {.4, geometry == Geometry::nearly_parallel ? 0. : .2, .3}};
  const Tensor<2, dim> A = linear_map<dim>(geometry);
  Tensor<1, dim>       translation;
  for (unsigned int d = 0; d < dim; ++d)
    translation[d] = translation_values[d];
  GridTools::transform(
    [&](const Point<dim> &p) { return Point<dim>(A * p + translation); }, tria);

  // Exact unit normal of the reference plane p_k = 0 after the map: row k of
  // A^{-1}, normalized. The sign does not matter for the residuals.
  const Tensor<2, dim>        A_inverse = invert(A);
  std::vector<Tensor<1, dim>> normals(n_walls);
  for (unsigned int k = 0; k < n_walls; ++k)
  {
    for (unsigned int j = 0; j < dim; ++j)
      normals[k][j] = A_inverse[k][j];
    normals[k] /= normals[k].norm();
  }

  // Corner, and in 3D with two walls, the far end of the shared edge.
  std::vector<std::pair<std::string, Point<dim>>> nodes = {
    {"corner", Point<dim>()}};
  if constexpr (dim == 3)
    if (n_walls == 2)
      nodes.emplace_back("edge  ", Point<dim>(0., 0., 1.));

  std::set<types::boundary_id> walls;
  for (unsigned int d = 0; d < n_walls; ++d)
    walls.insert(10 + d);

  deallog << case_name << std::endl;
  for (const Method method : {Method::all_walls, Method::per_wall, Method::fix})
  {
    AffineConstraints<double> constraints;
    try
    {
      apply_flux_constraints<dim>(
        method, position, walls, dofs, mapping, constraints);
      if (has_cycle(constraints))
      {
        for (const auto &[node_name, node] : nodes)
          deallog << "  " << method_name(method) << " " << node_name
                  << " cycle in constraints" << std::endl;
        continue;
      }
      constraints.close();
    }
    catch (const ExceptionBase &e)
    {
      for (const auto &[node_name, node] : nodes)
        deallog << "  " << method_name(method) << " " << node_name << " throws "
                << e.get_exc_name() << std::endl;
      continue;
    }

    Vector<double> values(dofs.n_dofs());
    values = 7.;
    constraints.distribute(values);

    for (const auto &[node_name, node] : nodes)
    {
      Tensor<1, dim> u;
      unsigned int   n_found        = 0;
      unsigned int   n_free         = 0;
      double         largest_weight = 0.;
      for (unsigned int i = 0; i < values.size(); ++i)
        if ((reference_points.at(i) - node).norm_square() < 1e-24)
        {
          ++n_found;
          u[components[i]] = values[i];
          if (!constraints.is_constrained(i))
            ++n_free;
          else
            for (const auto &[index, weight] :
                 *constraints.get_constraint_entries(i))
            {
              (void)index;
              largest_weight = std::max(largest_weight, std::abs(weight));
            }
        }
      // Position no_flux keeps the node on its wall: g is the node position.
      const Tensor<1, dim> g =
        position ? Tensor<1, dim>(A * node + translation) : Tensor<1, dim>();
      double largest_residual = 0.;
      for (const auto &normal : normals)
        largest_residual =
          std::max(largest_residual, std::abs(normal * (u - g)));

      // Otherwise u = 0 would look like a correct fluid result.
      if (n_found != dim)
      {
        deallog << "  " << method_name(method) << " " << node_name << " found "
                << n_found << " of " << dim << " DoFs" << std::endl;
        continue;
      }
      deallog << "  " << method_name(method) << " " << node_name
              << " free=" << n_free << " (expected " << dim - n_walls
              << ") residual=" << format(largest_residual)
              << " weight=" << format(largest_weight) << std::endl;
    }
  }
}

template <int dim>
void compare_dimension()
{
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
          compare_geometry<dim>(
            geometry, position, n_walls, case_name, reverse_order);
        }
}

// Mesh position shifted by `shift` along y, used as the position_flux_mms data.
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

// Two boundary ids that impose the same wall equation at a shared node. The
// equation must be accepted when it is redundant and rejected when it
// contradicts the first one.
//
// Reference, before (x, y) -> length (x + shift_x, slope x + y + shift_y):
//
//   (-1, 1) +-------+-------+ (1, 1)
//           |       |       |
//           |       |       |
//   (-1, 0) +-------*-------+ (1, 0)
//              10       11
//           no_flux   position_flux_mms
//
// After the map, both faces lie on one straight line of slope `slope`, with
// normal n ~ (-slope, 1), and * lands at vertex = length (shift_x, shift_y).
// At *, both walls constrain the same normal direction and pick pivot y:
//
//   10: n.(x - p) = 0              p is the node position (FixedMeshPosition)
//   11: n.(x - p - (0, shift)) = 0 (ShiftedPosition)
//
// The equations coincide when shift = 0 and contradict each other otherwise.
// The 12 cases combine:
//  - length in {1e-6, 1e3, 1e6}: the tolerance must scale with the problem.
//  - translated: * lies on the line of slope `slope` through the origin, so
//    n.vertex is zero but is computed as the difference of two terms of size
//    ~.45 length. The roundoff left over must not be taken as a
//    contradiction. This case needs prescribed_value_scale.
//  - incompatible: shift = 1e-7 length, far above roundoff at every scale.
//
// Two methods are compared:
//  - per wall: one deal.II call for {10}, then one for {11}, exactly as
//    the code did before the fix. The second call finds y already
//    constrained and silently skips its line.
//  - fix: apply_mesh_position_boundary_conditions. Wall 11 reduces to an
//    equation without coefficients after substituting wall 10, so it is
//    skipped if its right-hand side is within roundoff. Otherwise the call
//    throws "Incompatible flux conditions".
//
// Each case logs either the exception, or the residuals of both equations at
// * divided by length, after distributing a constant vector.
// Expected: compatible cases give zero residuals for both methods. In
// incompatible cases, per wall leaves mms ~ -n_y 1e-7, i.e. wall 11 is
// ignored, while fix throws.
void compare_redundant_conditions()
{
  for (const double length : {1e-6, 1e3, 1e6})
    for (const bool translated : {false, true})
      for (const bool incompatible : {false, true})
      {
        const double     slope   = translated ? .2931265934486439 : .3;
        const double     shift_x = translated ? 1.5253538488765903 : 0.;
        const double     shift_y = translated ? slope * shift_x : .2;
        const double     shift   = incompatible ? length * 1e-7 : 0.;
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
        FESystem<2>   fe(FE_Q<2>(1), 2);
        DoFHandler<2> dofs(tria);
        dofs.distribute_dofs(fe);
        MappingQ1<2> mapping;
        const auto   reference_points =
          DoFTools::map_dofs_to_support_points(mapping, dofs);
        std::vector<unsigned int>            components(dofs.n_dofs());
        std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
        for (const auto &cell : dofs.active_cell_iterators())
        {
          cell->get_dof_indices(indices);
          for (unsigned int i = 0; i < indices.size(); ++i)
            components[indices[i]] = fe.system_to_component_index(i).first;
        }
        GridTools::transform(
          [=](const Point<2> &p) {
            return Point<2>(length * (p[0] + shift_x),
                            length * (slope * p[0] + p[1] + shift_y));
          },
          tria);

        const Point<2> vertex(length * shift_x, length * shift_y);
        Tensor<1, 2>   normal;
        normal[0] = -slope;
        normal[1] = 1.;
        normal /= normal.norm();
        Tensor<1, 2> shifted_vertex = vertex;
        shifted_vertex[1] += shift;

        deallog << "redundant scale " << format(length)
                << (translated ? " translated" : "")
                << (incompatible ? " incompatible" : " compatible")
                << std::endl;
        for (const Method method : {Method::per_wall, Method::fix})
        {
          Functions::ZeroFunction<2> zero(2);
          ShiftedPosition            position(shift);
          AffineConstraints<double>  constraints;
          try
          {
            if (method == Method::fix)
            {
              std::map<types::boundary_id, BoundaryConditions::PseudosolidBC<2>>
                bcs;
              bcs[10].id   = 10;
              bcs[10].type = BoundaryConditions::Type::no_flux;
              bcs[11].id   = 11;
              bcs[11].type = BoundaryConditions::Type::position_flux_mms;
              BoundaryConditions::apply_mesh_position_boundary_conditions<2>(
                false, 0, 2, dofs, mapping, bcs, zero, position, constraints);
            }
            else
            {
              FixedMeshPosition<2> fixed_mesh(0, 2);
              VectorTools::compute_nonzero_normal_flux_constraints(
                dofs,
                0,
                {10},
                {{10, &fixed_mesh}},
                constraints,
                mapping,
                false);
              VectorTools::compute_nonzero_normal_flux_constraints(
                dofs, 0, {11}, {{11, &position}}, constraints, mapping, false);
            }
            if (has_cycle(constraints))
            {
              deallog << "  " << method_name(method)
                      << " vertex cycle in constraints" << std::endl;
              continue;
            }
            constraints.close();
          }
          catch (const ExceptionBase &e)
          {
            deallog << "  " << method_name(method) << " vertex throws "
                    << e.get_exc_name() << std::endl;
            continue;
          }

          Vector<double> values(dofs.n_dofs());
          values = length;
          constraints.distribute(values);
          Tensor<1, 2> u;
          unsigned int n_found = 0;
          for (unsigned int i = 0; i < values.size(); ++i)
            if (reference_points.at(i).norm_square() < 1e-24)
            {
              ++n_found;
              u[components[i]] = values[i];
            }
          if (n_found != 2)
          {
            deallog << "  " << method_name(method) << " vertex found "
                    << n_found << " of 2 DoFs" << std::endl;
            continue;
          }
          // Residuals of the no_flux and of the shifted mms wall equations.
          deallog << "  " << method_name(method) << " vertex residual no_flux="
                  << format(normal * (u - vertex) / length)
                  << " mms=" << format(normal * (u - shifted_vertex) / length)
                  << std::endl;
        }
      }
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  initlog();
  compare_dimension<2>();
  compare_dimension<3>();
  compare_redundant_conditions();
  return 0;
}
