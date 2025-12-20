#ifndef COMPARE_MATRIX_H
#define COMPARE_MATRIX_H

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <types.h>
#include <utilities.h>


namespace Verification
{
  using namespace dealii;

  /**
   * Compute the analytic Jacobian matrix on each mesh element
   * and compare it to its finite differences approximation by
   * perturing the Newton residual.
   *
   * The FSI solver has the mesh position as unknown, so it is
   * my understanding that we cannot use automatic differentiation (AD)
   * in that case, although this would be the preferred way.
   */
  template <int dim,
            typename MainClass,
            typename Iterator,
            typename ScratchData,
            typename CopyData,
            typename VectorType>
  std::pair<double, double> compare_analytical_matrix_with_fd(
    const DoFHandler<dim> &dof_handler,
    const unsigned int     n_dofs_per_cell,
    MainClass             &main_object,
    void (MainClass::*assemble_local_matrix)(const Iterator &,
                                             ScratchData &,
                                             CopyData &),
    void (MainClass::*assemble_local_rhs)(const Iterator &,
                                          ScratchData &,
                                          CopyData &),
    ScratchData       &scratch_data,
    CopyData          &copy_data,
    VectorType        &present_solution,
    VectorType        &evaluation_point,
    VectorType        &local_evaluation_point,
    MPI_Comm           mpi_communicator,
    const std::string &output_dir         = "",
    const bool         print_matrices     = false,
    const double       absolute_tolerance = 1e-6,
    const double       relative_tolerance = 1e-3)
  {
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
    FullMatrix<double> local_matrix(n_dofs_per_cell, n_dofs_per_cell);
    FullMatrix<double> local_matrix_fd(n_dofs_per_cell, n_dofs_per_cell);
    FullMatrix<double> error_matrix(n_dofs_per_cell, n_dofs_per_cell);
    Vector<double>     ref_local_rhs(n_dofs_per_cell);
    Vector<double>     perturbed_local_rhs(n_dofs_per_cell);

    ////////////////////////////////////////////////////////////////////
    // const FEValuesExtractors::Vector position(dim + 1);
    // std::shared_ptr<Mapping<dim>>    mapping =
    //   std::make_shared<MappingFEField<dim, dim, LA::ParVectorType>>(
    //     dof_handler,
    //     evaluation_point,
    //     dof_handler.get_fe().component_mask(position));
    ////////////////////////////////////////////////////////////////////

    // Write problematic elements to a Gmsh pos file
    std::ofstream outfile(output_dir + "elements_with_mismatched_matrix.pos");
    if (print_matrices)
      outfile << "View \"elements_with_mismatched_matrix\"{" << std::endl;

    local_evaluation_point = present_solution;
    evaluation_point       = present_solution;

    double max_absolute_error_over_all_elements = 0.;
    double max_relative_error_over_all_elements = 0.;

    // Loop over mesh elements
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      // std::cout << std::endl;
      // std::cout << "On element " << cell->index() << std::endl;
      // std::cout << std::endl;

      ///////////////////////////////////////
      // Point<dim>              ref0(0., 0.);
      // Point<dim>              ref1(1., 0.);
      // Point<dim>              ref2(0., 1.);
      // std::vector<Point<dim>> refs = {ref0, ref1, ref2};
      // std::vector<Point<dim>> vertices(cell->n_vertices());
      // for (unsigned int iv = 0; iv < cell->n_vertices(); ++iv)
      // {
      //   vertices[iv] = mapping->transform_unit_to_real_cell(cell, refs[iv]);
      //   std::cout << "With current vertex " << vertices[iv] << std::endl;
      // }
      ///////////////////////////////////////

      if (!cell->is_locally_owned())
        continue;

      cell->get_dof_indices(local_dof_indices);

      {
        //
        // Compute analytic matrix
        //
        // std::cout << "Assembling matrix" << std::endl;
        (main_object.*assemble_local_matrix)(cell, scratch_data, copy_data);
        local_matrix = copy_data.local_matrix;
        // std::cout << "Done matrix" << std::endl;
      }

      {
        //
        // Compute matrix with finite differences
        //
        const double h  = 1.e-8;
        local_matrix_fd = 0.;

        // Compute non-perturbed RHS
        (main_object.*assemble_local_rhs)(cell, scratch_data, copy_data);
        ref_local_rhs = copy_data.local_rhs;

        // std::cout << "Non-perturbed residual is " << std::endl;
        // ref_local_rhs.print(std::cout, 12, 3);

        for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
        {
          // std::cout << "Perturbing component " << j << std::endl;
          evaluation_point      = local_evaluation_point;
          const double og_value = evaluation_point[local_dof_indices[j]];

          local_evaluation_point[local_dof_indices[j]] += h;
          local_evaluation_point.compress(VectorOperation::add);
          evaluation_point = local_evaluation_point;

          // Compute perturbed RHS
          (main_object.*assemble_local_rhs)(cell, scratch_data, copy_data);
          perturbed_local_rhs = copy_data.local_rhs;

          // std::cout << "Perturbed residual is " << std::endl;
          // perturbed_local_rhs.print(std::cout, 12, 3);

          // Finite differences (with sign change as residual is -NL(u))
          for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
          {
            local_matrix_fd(i, j) =
              -(perturbed_local_rhs(i) - ref_local_rhs(i)) / h;
          }

          // Restore solution
          local_evaluation_point[local_dof_indices[j]] = og_value;
          local_evaluation_point.compress(VectorOperation::insert);
          evaluation_point = local_evaluation_point;
        }
      }

      error_matrix.equ(1.0, local_matrix);
      error_matrix.add(-1.0, local_matrix_fd);

      // Compute absolute and relative error
      double max_abs_err = 0., max_rel_err = 0.;
      for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      {
        for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
        {
          auto         error_pair = compute_relative_error(local_matrix(i, j),
                                                   local_matrix_fd(i, j),
                                                   absolute_tolerance);
          const double abs_err    = error_pair.first;
          const double rel_err    = error_pair.second;
          max_abs_err             = std::max(max_abs_err, abs_err);
          max_rel_err             = std::max(max_rel_err, rel_err);

          if (abs_err > absolute_tolerance && rel_err > relative_tolerance)
          {
            const unsigned int ci =
              dof_handler.get_fe().system_to_component_index(i).first;
            const unsigned int cj =
              dof_handler.get_fe().system_to_component_index(j).first;
            std::cout << "Entry (" << i << ", " << j << ") - components (" << ci
                      << ", " << cj << ") : an: " << local_matrix(i, j)
                      << " fd: " << local_matrix_fd(i, j)
                      << " abs_err: " << abs_err << " - rel_err: " << rel_err
                      << std::endl;
          }
        }
      }

      max_absolute_error_over_all_elements =
        std::max(max_absolute_error_over_all_elements, max_abs_err);
      // Update the overall max relative error only if the associated absolute
      // error is large enough, to prevent false positives.
      if (max_abs_err > absolute_tolerance)
        max_relative_error_over_all_elements =
          std::max(max_relative_error_over_all_elements, max_rel_err);

      if (max_abs_err > absolute_tolerance && max_rel_err > relative_tolerance)
      {
        std::cout << "Max abs error on elem : " << max_abs_err
                  << " - max rel error : " << max_rel_err << std::endl;

        std::cout << "Analytic Jacobian  matrix is " << std::endl;
        local_matrix.print(std::cout, 12, 3);
        std::cout << "Finite differences matrix is " << std::endl;
        local_matrix_fd.print(std::cout, 12, 3);

        std::cout << "Error matrix is " << std::endl;
        error_matrix.print(std::cout, 12, 3);
        std::cout << "Non-perturbed residual (2) is " << std::endl;
        ref_local_rhs.print(std::cout, 12, 3);
        // std::cout << "Max difference on element is " << diff << std::endl;
        std::cout << "On element with measure " << cell->measure() << std::endl;

        // Draw the problematic element
        outfile << ((dim == 2) ? "ST(" : "SS(");
        for (unsigned int iv = 0; iv < cell->n_vertices(); ++iv)
        {
          if constexpr (dim == 2)
            outfile << cell->vertex(iv)[0] << "," << cell->vertex(iv)[1]
                    << ",0." << ((iv == cell->n_vertices() - 1) ? "" : ",");
          else
            outfile << cell->vertex(iv)[0] << "," << cell->vertex(iv)[1] << ","
                    << cell->vertex(iv)[2]
                    << ((iv == cell->n_vertices() - 1) ? "" : ",");
        }
        outfile << ((dim == 2) ? "){1., 1., 1.};" : "){1., 1., 1., 1.};")
                << std::endl;

        ////////////////////////////////////////////////////////
        // // Draw its modified position
        // outfile << ((dim == 2) ? "ST(" : "SS(");
        // for (unsigned int iv = 0; iv < cell->n_vertices(); ++iv)
        // {
        //   const auto &v = vertices[iv];
        //   if constexpr (dim == 2)
        //     outfile << v[0] << "," << v[1] << ",0."
        //             << ((iv == cell->n_vertices() - 1) ? "" : ",");
        //   else
        //     outfile << v[0] << "," << v[1] << "," << v[2]
        //             << ((iv == cell->n_vertices() - 1) ? "" : ",");
        // }
        // outfile << ((dim == 2) ? "){1., 1., 1.};" : "){1., 1., 1., 1.};")
        //         << std::endl;
        ////////////////////////////////////////////////////////
      }
    }

    max_absolute_error_over_all_elements =
      Utilities::MPI::max(max_absolute_error_over_all_elements,
                          mpi_communicator);
    max_relative_error_over_all_elements =
      Utilities::MPI::max(max_relative_error_over_all_elements,
                          mpi_communicator);

    if (print_matrices)
      outfile << "};" << std::endl;
    outfile.close();

    return {max_absolute_error_over_all_elements,
            max_relative_error_over_all_elements};
  }
} // namespace Verification

#endif