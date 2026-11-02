#include <boundary_conditions.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/vector.h>

#include "../tests.h"

template <int dim>
void check_masks()
{
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0., 1., true);
  FESystem<dim>   fe(FE_Q<dim>(1), dim);
  DoFHandler<dim> dofs(tria);
  dofs.distribute_dofs(fe);
  MappingQ1<dim> mapping;
  const auto     points = DoFTools::map_dofs_to_support_points(mapping, dofs);
  std::vector<unsigned int>            components(dofs.n_dofs());
  std::vector<types::global_dof_index> indices(fe.dofs_per_cell);
  for (const auto &cell : dofs.active_cell_iterators())
  {
    cell->get_dof_indices(indices);
    for (unsigned int i = 0; i < indices.size(); ++i)
      components[indices[i]] = fe.system_to_component_index(i).first;
  }
  Functions::ZeroFunction<dim>     zero(dim);
  BoundaryConditions::FluidBC<dim> fluid;
  ParameterHandler                 fp;
  fluid.declare_parameters(fp);
  fp.parse_input_from_string(
    "set id=0\nset type=input_function\nset constrain_v=false\nset "
    "constrain_w=false\nsubsection u\nset Function expression=3\nend\n");
  fluid.read_parameters(fp);
  BoundaryConditions::PseudosolidBC<dim> mesh;
  ParameterHandler                       mp;
  mesh.declare_parameters(mp);
  mp.parse_input_from_string(
    "set id=0\nset type=input_function\nsubsection x\nset Function "
    "expression=x+0.1\nend\nsubsection y\nset type=no_flux\nend\nsubsection "
    "z\nset type=no_flux\nend\n");
  mesh.read_parameters(mp);
  for (bool homogeneous : {false, true})
    for (bool position : {false, true})
    {
      AffineConstraints<double> constraints;
      if (position)
        BoundaryConditions::apply_mesh_position_boundary_conditions<dim>(
          homogeneous,
          0,
          dim,
          dofs,
          mapping,
          {{0, mesh}},
          zero,
          zero,
          constraints);
      else
        BoundaryConditions::apply_velocity_boundary_conditions<dim>(
          homogeneous,
          0,
          dim,
          dofs,
          mapping,
          {{0, fluid}},
          zero,
          zero,
          constraints);
      constraints.close();
      Vector<double> values(dofs.n_dofs());
      values = 7.;
      constraints.distribute(values);
      for (unsigned int i = 0; i < values.size(); ++i)
      {
        const bool   constrained = points.at(i)[0] == 0. && components[i] == 0;
        const double expected =
          constrained ? (homogeneous ? 0. : (position ? .1 : 3.)) : 7.;
        AssertThrow(std::abs(values[i] - expected) < 1e-12,
                    ExcMessage("Component mask constrained the wrong DoF"));
      }
    }
  // Two oblique slip walls must impose both normals at their intersection.
  // Both normals select the same dominant component in this geometry.
  GridTools::transform(
    [](const Point<dim> &p) {
      auto q = p;
      q[0]   = .4 + p[0] + p[1];
      q[1]   = .2 + .2 * p[0] + .6 * p[1];
      return q;
    },
    tria);
  BoundaryConditions::FluidBC<dim> left, bottom;
  left.id   = 0;
  bottom.id = 2;
  left.type = bottom.type = BoundaryConditions::Type::slip;
  for (bool position : {false, true})
  {
    AffineConstraints<double> constraints;
    if (position)
    {
      BoundaryConditions::PseudosolidBC<dim> mesh_left, mesh_bottom;
      mesh_left.id   = 0;
      mesh_bottom.id = 2;
      mesh_left.type = mesh_bottom.type = BoundaryConditions::Type::no_flux;
      BoundaryConditions::apply_mesh_position_boundary_conditions<dim>(
        false,
        0,
        dim,
        dofs,
        mapping,
        {{0, mesh_left}, {2, mesh_bottom}},
        zero,
        zero,
        constraints);
    }
    else
      BoundaryConditions::apply_velocity_boundary_conditions<dim>(false,
                                                                  0,
                                                                  dim,
                                                                  dofs,
                                                                  mapping,
                                                                  {{0, left},
                                                                   {2, bottom}},
                                                                  zero,
                                                                  zero,
                                                                  constraints);
    constraints.close();
    Vector<double> values(dofs.n_dofs());
    values = 7.;
    constraints.distribute(values);
    for (unsigned int i = 0; i < values.size(); ++i)
      if (points.at(i)[0] == 0. && points.at(i)[1] == 0.)
      {
        const double expected =
          components[i] >= 2 ? 7. :
                               (position ? (components[i] == 0 ? .4 : .2) : 0.);
        AssertThrow(std::abs(values[i] - expected) < 1e-12,
                    ExcMessage("Slip corner does not satisfy both walls"));
      }
  }
  deallog << dim << "D component masks and slip corners OK" << std::endl;
}
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  initlog();
  try
  {
    check_masks<2>();
    check_masks<3>();
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
