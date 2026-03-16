// ------------------------------------------------------------------------
//
// SPDX-License-Identifier: LGPL-2.1-or-later
// Copyright (C) 2015 - 2020 by the deal.II authors
//
// This file is part of the deal.II library.
//
// Part of the source code is dual licensed under Apache-2.0 WITH
// LLVM-exception OR LGPL-2.1-or-later. Detailed license information
// governing the source code and code contributions can be found in
// LICENSE.md and CONTRIBUTING.md at the top level directory of deal.II.
//
// ------------------------------------------------------------------------

#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/fe_bernstein.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_cartesian.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/fe/mapping_q1.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/reference_cell.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/fe_values.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <fe_simplex_p_with_3d_hp.h>
#include <mapping_fe_field_hp2.h>

#include <sstream>
#include <string>
#include <vector>

#include "../tests.h"

#define PRECISION 2

/**
 * Tests the very ad-hoc modification to MappingFEField to handle a hp
 * DoFHandler
 *
 * Compare the behavior with and without hp capabilities, on quads and simplices
 */

template <int dim>
class FixedMeshPosition : public Function<dim>
{
public:
  // Lower bound of the mesh position variable (first component)
  const unsigned int x_lower;

public:
  FixedMeshPosition(const unsigned int x_lower, const unsigned int n_components)
    : Function<dim>(n_components)
    , x_lower(x_lower)
  {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double>   &values) const override
  {
    for (unsigned int d = 0; d < dim; ++d)
      values[x_lower + d] = p[d];
  }
};

template <int dim>
class DeformedMeshPosition : public Function<dim>
{
public:
  // Lower bound of the mesh position variable (first component)
  const unsigned int x_lower;

public:
  DeformedMeshPosition(const unsigned int x_lower,
                       const unsigned int n_components)
    : Function<dim>(n_components)
    , x_lower(x_lower)
  {}

  virtual void vector_value(const Point<dim> &p,
                            Vector<double>   &values) const override
  {
    for (unsigned int d = 0; d < dim; ++d)
      values[x_lower + d] = p[d] + 0.5 * p[d] * (p[d] - 1.);
  }
};

template <int dim>
class LinearFunction : public Function<dim>
{
public:
  LinearFunction(const unsigned int n_components)
    : Function<dim>(n_components)
  {}

  double value(const Point<dim> &p, const unsigned int) const
  {
    return p[0] + 2;
  }
};

template <int dim>
void test(const bool with_hp, const bool with_simplices)
{
  deallog << "dim = " << dim << std::endl;
  deallog << "With hp : " << with_hp << std::endl;

  TimerOutput computing_timer(std::cout,
                              TimerOutput::summary,
                              TimerOutput::wall_times);

  TimerOutput::Scope t(computing_timer, "Run");

  /**
   * hp part : create an FE collection
   */
  Triangulation<dim> tria;
  GridGenerator::hyper_cube(tria, 0, 1);
  tria.refine_global(2);

  if (with_simplices)
  {
    const unsigned int n_divisions = (dim == 2) ? 2u : 6u;
    GridGenerator::convert_hypercube_to_simplex_mesh(tria, tria, n_divisions);
  }

  // Create a collection with two FESystems:
  // first field for both is the position variable,
  // second field for one is e.g. temperature, for the other it is a dummy.
  hp::FECollection<dim>      fe_collection;
  hp::MappingCollection<dim> mapping_collection;
  hp::MappingCollection<dim> moving_mapping_collection;
  hp::QCollection<dim>       q_collection;
  hp::QCollection<dim - 1>   face_q_collection;

  std::shared_ptr<FESystem<dim>> fe0;
  std::shared_ptr<FESystem<dim>> fe1;

  if (with_simplices)
  {
    fe0 = std::make_shared<FESystem<dim>>(FE_SimplexP_3D_hp<dim>(1) ^ dim,
                                          FE_SimplexP_3D_hp<dim>(1));
    fe1 =
      std::make_shared<FESystem<dim>>(FE_SimplexP_3D_hp<dim>(1) ^ dim,
                                      FE_Nothing<dim>(
                                        ReferenceCells::get_simplex<dim>()));
    mapping_collection.push_back(MappingFE<dim>(FE_SimplexP_3D_hp<dim>(1)));
    mapping_collection.push_back(MappingFE<dim>(FE_SimplexP_3D_hp<dim>(1)));
    q_collection.push_back(QGaussSimplex<dim>(4));
    q_collection.push_back(QGaussSimplex<dim>(4));
    face_q_collection.push_back(QGaussSimplex<dim - 1>(4));
    face_q_collection.push_back(QGaussSimplex<dim - 1>(4));
  }
  else
  {
    fe0 = std::make_shared<FESystem<dim>>(FE_Q<dim>(1) ^ dim, FE_Q<dim>(1));
    fe1 =
      std::make_shared<FESystem<dim>>(FE_Q<dim>(1) ^ dim, FE_Nothing<dim>());
    mapping_collection.push_back(MappingQ<dim>(1));
    mapping_collection.push_back(MappingQ<dim>(1));
    q_collection.push_back(QGauss<dim>(2));
    q_collection.push_back(QGauss<dim>(2));
    face_q_collection.push_back(QGauss<dim - 1>(2));
    face_q_collection.push_back(QGauss<dim - 1>(2));
  }

  fe_collection.push_back(*fe0);
  fe_collection.push_back(*fe1);

  deallog << fe_collection[0].n_dofs_per_cell() << " dofs per cell for system 0"
          << std::endl;
  deallog << fe_collection[1].n_dofs_per_cell() << " dofs per cell for system 1"
          << std::endl;

  DoFHandler<dim> dof_handler(tria);

  if (with_hp)
  {
    // Cells inside circle have fe1, others have fe_system
    typename DoFHandler<dim>::active_cell_iterator cell =
                                                     dof_handler.begin_active(),
                                                   endc = dof_handler.end();

    for (; cell != endc; ++cell)
    {
      Point<dim> center = cell->center();
      if (std::sqrt(center.square()) < 0.5)
        cell->set_active_fe_index(1);
      else
        cell->set_active_fe_index(0);
    }

    dof_handler.distribute_dofs(fe_collection);
  }
  else
    dof_handler.distribute_dofs(*fe0);

  deallog << "   Number of active cells:       " << tria.n_active_cells()
          << std::endl
          << "   Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

  /**
   * Now create a MappingFEField with the dof handler
   */
  // FE_Bernstein<dim, spacedim> fe(degree);
  // // FE_Q<dim> fe(degree);
  // FESystem<dim, spacedim> fe_sys(fe, spacedim);

  // // DoFHandler<dim> dof(tria);
  // DoFHandler<dim, spacedim> dof_sys(tria);
  // // dof.distribute_dofs(fe);
  // dof_sys.distribute_dofs(fe_sys);

  Vector<double> solution;
  solution.reinit(dof_handler.n_dofs());

  // Ordering is :
  // x : [0, dim)
  // T : dim
  FEValuesExtractors::Vector position(0);
  const ComponentMask position_mask = fe_collection.component_mask(position);
  FEValuesExtractors::Scalar field(dim);
  const ComponentMask        field_mask = fe_collection.component_mask(field);

  std::shared_ptr<Mapping<dim>> mapping;

  const unsigned int x_lower      = 0;
  const unsigned int n_components = dim + 1;

  if (with_hp)
  {
    // Get the position vector in hp context
    VectorTools::interpolate(mapping_collection,
                             dof_handler,
                             DeformedMeshPosition<dim>(x_lower, n_components),
                             solution,
                             position_mask);
    deallog << "Solution with hp:" << std::endl;
    // solution.print(std::cout, 3, true, false);
    mapping = std::make_shared<MappingFEFieldHp2<dim>>(dof_handler,
                                                       mapping_collection,
                                                       q_collection,
                                                       face_q_collection,
                                                       solution,
                                                       position_mask);
    moving_mapping_collection.push_back(*mapping);
    moving_mapping_collection.push_back(*mapping);

    // Set the non-position field
    VectorTools::interpolate(moving_mapping_collection,
                             dof_handler,
                             LinearFunction<dim>(dim + 1),
                             solution,
                             field_mask);

    // Get the shape functions and their gradients
    hp::FEValues     hp_fe_values(moving_mapping_collection,
                              fe_collection,
                              q_collection,
                              update_values | update_gradients |
                                update_JxW_values);
    hp::FEFaceValues hp_fe_face_values(moving_mapping_collection,
                                       fe_collection,
                                       face_q_collection,
                                       update_values | update_gradients |
                                         update_JxW_values);

    // std::ofstream outfile_f("values_f_" + std::to_string(with_hp) + ".txt");
    // std::ofstream outfile_x("values_x_" + std::to_string(with_hp) + ".txt");

    double         I_f = 0., A = 0., I_f_bord = 0., A_bord = 0.;
    Tensor<1, dim> I_grad_f, I_grad_f_bord;
    Tensor<1, dim> I_x;
    Tensor<2, dim> I_grad_x;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      const unsigned int fe_index = cell->active_fe_index();
      if (fe_index == 1)
        continue;

      hp_fe_values.reinit(cell);
      const auto &fe_values = hp_fe_values.get_present_fe_values();

      const auto &fe = fe_values.get_fe();
      // outfile_f << "======== Partition " << fe_index
      //           << " ===========" << std::endl;
      // outfile_x << "======== Partition " << fe_index
      //           << " ===========" << std::endl;

      // outfile_f << "Cell with vertices" << std::endl;
      // outfile_x << "Cell with vertices" << std::endl;
      // for (unsigned int iv = 0; iv < cell->n_vertices(); ++iv)
      // {
      //   outfile_f << cell->vertex(iv) << std::endl;
      //   outfile_x << cell->vertex(iv) << std::endl;
      // }

      {
        std::vector<double>         f_val(q_collection[fe_index].size());
        std::vector<Tensor<1, dim>> f_grad(q_collection[fe_index].size()),
          x_val(q_collection[fe_index].size());
        std::vector<Tensor<2, dim>> x_grad(q_collection[fe_index].size());

        fe_values[field].get_function_values(solution, f_val);
        fe_values[field].get_function_gradients(solution, f_grad);
        // fe_values[position].get_function_values(solution, x_val);
        // fe_values[position].get_function_gradients(solution, x_grad);

        for (unsigned int q = 0; q < q_collection[fe_index].size(); ++q)
        {
          const double JxW = fe_values.JxW(q);

          A += JxW;
          I_f += f_val[q] * JxW;
          // I_x += x_val[q] * JxW;
          I_grad_f += f_grad[q] * JxW;
          // I_grad_x += x_grad[q] * JxW;

          // outfile_f << "JxW " << q << " = " << JxW << std::endl;

          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          {
            const auto  field_value = fe_values[field].value(i, q);
            const auto &grad_field  = fe_values[field].gradient(i, q);
            // outfile_f << "f Value = " << field_value << std::endl;
            // outfile_f << "f Grad  = " << grad_field << std::endl;

            // const auto &position_value = fe_values[position].value(i, q);
            // const auto &grad_position  = fe_values[position].gradient(i, q);
            // outfile_x << "x Value = " << position_value << std::endl;
            // outfile_x << "x Grad  = " << grad_position << std::endl;
          }
        }
      }

      // Faces
      if (cell->at_boundary())
      {
        for (const auto i_face : cell->face_indices())
        {
          const auto &face = cell->face(i_face);
          if (face->at_boundary())
          {
            hp_fe_face_values.reinit(cell, i_face);
            const unsigned int fe_index = cell->active_fe_index();
            const auto        &fe_face_values =
              hp_fe_face_values.get_present_fe_values();
            const auto &fe = fe_face_values.get_fe();

            std::vector<double> f_val(face_q_collection[fe_index].size());
            std::vector<Tensor<1, dim>> f_grad(
              face_q_collection[fe_index].size());

            fe_face_values[field].get_function_values(solution, f_val);
            fe_face_values[field].get_function_gradients(solution, f_grad);

            for (unsigned int q = 0; q < face_q_collection[fe_index].size();
                 ++q)
            {
              const double face_JxW = fe_face_values.JxW(q);

              A_bord += face_JxW;
              I_f_bord += f_val[q] * face_JxW;
              I_grad_f_bord += f_grad[q] * face_JxW;

              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              {
                const auto  field_value = fe_face_values[field].value(i, q);
                const auto &grad_field  = fe_face_values[field].gradient(i, q);
                // outfile_f << "f Value Bord = " << field_value << std::endl;
                // outfile_f << "f Grad  Bord = " << grad_field << std::endl;
              }
            }
          }
        }
      }
    }

    std::cout << "Domain area                      = " << A << std::endl;
    std::cout << "Integral over domain of      f   = " << I_f << std::endl;
    std::cout << "Integral over domain of grad f   = " << I_grad_f << std::endl;
    std::cout << "Boundary area                    = " << A_bord << std::endl;
    std::cout << "Integral over boundary of      f = " << I_f_bord << std::endl;
    std::cout << "Integral over boundary of grad f = " << I_grad_f_bord
              << std::endl;

    // outfile_x << "Integral over domain of      x = " << I_x << std::endl;
    // outfile_x << "Integral over domain of grad x = " << I_grad_x <<
    // std::endl;

    for (const auto &m : moving_mapping_collection)
    {
      Assert((dynamic_cast<const typename dealii::MappingFEFieldHp2<dim> *>(
                &m) != nullptr),
             ExcInternalError());
      const typename dealii::MappingFEFieldHp2<dim> &my_mapping =
        static_cast<const typename dealii::MappingFEFieldHp2<dim> &>(m);
      std::cout << "Realloced cell data: " << my_mapping.n_realloc_cell_data
                << std::endl;
      std::cout << "Realloced face data: " << my_mapping.n_realloc_face_data
                << std::endl;
      std::cout << "Kept      cell data: " << my_mapping.n_kept_cell_data
                << std::endl;
      std::cout << "Kept      face data: " << my_mapping.n_kept_face_data
                << std::endl;
    }
  }
  else
  {
    // VectorTools::get_position_vector(dof_handler, solution, position_mask);
    VectorTools::interpolate(mapping_collection[0],
                             dof_handler,
                             DeformedMeshPosition<dim>(x_lower, n_components),
                             solution,
                             position_mask);

    deallog << "Solution without hp:" << std::endl;
    // solution.print(std::cout, 3, true, false);
    mapping = std::make_shared<MappingFEField<dim>>(dof_handler,
                                                    solution,
                                                    position_mask);

    // Set the non-position field
    VectorTools::interpolate(*mapping,
                             dof_handler,
                             LinearFunction<dim>(dim + 1),
                             solution,
                             field_mask);

    const auto &fe              = *fe0;
    const auto &quadrature      = q_collection[0];
    const auto &face_quadrature = face_q_collection[0];

    // Get the shape functions and their gradients
    FEValues     fe_values(*mapping,
                       fe,
                       quadrature,
                       update_values | update_gradients | update_JxW_values);
    FEFaceValues fe_face_values(*mapping,
                                fe,
                                face_quadrature,
                                update_values | update_gradients |
                                  update_JxW_values);

    // Compute integrals and fields, gradients
    // std::ofstream outfile_f("values_f_" + std::to_string(with_hp) + ".txt");
    // std::ofstream outfile_x("values_x_" + std::to_string(with_hp) + ".txt");

    double         I_f = 0., A = 0., I_f_bord = 0., A_bord = 0.;
    Tensor<1, dim> I_grad_f, I_grad_f_bord;
    Tensor<1, dim> I_x;
    Tensor<2, dim> I_grad_x;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      Point<dim> center = cell->center();
      if (std::sqrt(center.square()) < 0.5)
        continue;

      fe_values.reinit(cell);

      // outfile_f << "======== Partition 0 ===========" << std::endl;
      // outfile_x << "======== Partition 0 ===========" << std::endl;

      // outfile_f << "Cell with vertices" << std::endl;
      // outfile_x << "Cell with vertices" << std::endl;
      // for (unsigned int iv = 0; iv < cell->n_vertices(); ++iv)
      // {
      //   outfile_f << cell->vertex(iv) << std::endl;
      //   outfile_x << cell->vertex(iv) << std::endl;
      // }

      // Volume
      {
        std::vector<double>         f_val(quadrature.size());
        std::vector<Tensor<1, dim>> f_grad(quadrature.size()),
          x_val(quadrature.size());
        std::vector<Tensor<2, dim>> x_grad(quadrature.size());

        fe_values[field].get_function_values(solution, f_val);
        fe_values[field].get_function_gradients(solution, f_grad);
        // fe_values[position].get_function_values(solution, x_val);
        // fe_values[position].get_function_gradients(solution, x_grad);

        for (unsigned int q = 0; q < quadrature.size(); ++q)
        {
          const double JxW = fe_values.JxW(q);

          A += JxW;
          I_f += f_val[q] * JxW;
          // I_x += x_val[q] * JxW;
          I_grad_f += f_grad[q] * JxW;
          // I_grad_x += x_grad[q] * JxW;

          // outfile_f << "JxW " << q << " = " << JxW << std::endl;

          for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
          {
            const auto  field_value = fe_values[field].value(i, q);
            const auto &grad_field  = fe_values[field].gradient(i, q);
            // outfile_f << "f Value = " << field_value << std::endl;
            // outfile_f << "f Grad  = " << grad_field << std::endl;

            // const auto &position_value = fe_values[position].value(i, q);
            // const auto &grad_position  = fe_values[position].gradient(i, q);
            // outfile_x << "x Value = " << position_value << std::endl;
            // outfile_x << "x Grad  = " << grad_position << std::endl;
          }
        }
      }

      // Faces
      if (cell->at_boundary())
      {
        for (const auto i_face : cell->face_indices())
        {
          const auto &face = cell->face(i_face);
          if (face->at_boundary())
          {
            fe_face_values.reinit(cell, i_face);

            std::vector<double>         f_val(face_quadrature.size());
            std::vector<Tensor<1, dim>> f_grad(face_quadrature.size());

            fe_face_values[field].get_function_values(solution, f_val);
            fe_face_values[field].get_function_gradients(solution, f_grad);

            for (unsigned int q = 0; q < face_quadrature.size(); ++q)
            {
              const double face_JxW = fe_face_values.JxW(q);

              A_bord += face_JxW;
              I_f_bord += f_val[q] * face_JxW;
              I_grad_f_bord += f_grad[q] * face_JxW;

              for (unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
              {
                const auto  field_value = fe_face_values[field].value(i, q);
                const auto &grad_field  = fe_face_values[field].gradient(i, q);
                // outfile_f << "f Value Bord = " << field_value << std::endl;
                // outfile_f << "f Grad  Bord = " << grad_field << std::endl;
              }
            }
          }
        }
      }
    }

    std::cout << "Domain area                      = " << A << std::endl;
    std::cout << "Integral over domain of      f   = " << I_f << std::endl;
    std::cout << "Integral over domain of grad f   = " << I_grad_f << std::endl;
    std::cout << "Boundary area                    = " << A_bord << std::endl;
    std::cout << "Integral over boundary of      f = " << I_f_bord << std::endl;
    std::cout << "Integral over boundary of grad f = " << I_grad_f_bord
              << std::endl;

    // outfile_x << "Integral over domain of      x = " << I_x << std::endl;
    // outfile_x << "Integral over domain of grad x = " << I_grad_x <<
    // std::endl;
  }

  std::vector<std::string> solution_names(dim, "position");
  solution_names.push_back("field");

  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation(
      dim, DataComponentInterpretation::component_is_part_of_vector);
  data_component_interpretation.push_back(
    DataComponentInterpretation::component_is_scalar);

  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(solution,
                           solution_names,
                           DataOut<dim>::type_dof_data,
                           data_component_interpretation);

  // Active fe indices
  Vector<float> fe_indices(tria.n_active_cells());
  for (const auto &cell : dof_handler.active_cell_iterators())
    fe_indices(cell->id().get_coarse_cell_id()) = cell->active_fe_index();
  data_out.add_data_vector(fe_indices, "fe_indices");

  data_out.build_patches(*mapping, 2);

  std::string filename;
  if (with_simplices)
    filename = with_hp ?
                 "solution" + std::to_string(dim) + "d_hp_simplices.vtk" :
                 "solution" + std::to_string(dim) + "d_simplices.vtk";
  else
    filename = with_hp ? "solution" + std::to_string(dim) + "d_hp.vtk" :
                         "solution" + std::to_string(dim) + "d.vtk";

  std::ofstream output(filename);
  data_out.write_vtk(output);
}

int main()
{
  initlog();
  // // test<2>(false, false);
  // test<2>(false, true);
  // // test<2>(true, false);
  // test<2>(true, true);

  // test<3>(false, false);
  test<3>(false, true);
  // test<3>(true, false);
  test<3>(true, true);

  // for (unsigned int d = 1; d < 4; ++d)
  //   {
  //     test<2, 2>(d);
  //     // test<2, 3>(d);
  //     // test<3, 3>(d);
  //   }
}
