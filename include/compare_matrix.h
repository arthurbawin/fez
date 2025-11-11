#ifndef COMPARE_MATRIX_H
#define COMPARE_MATRIX_H

#include <deal.II/dofs/dof_handler.h>

namespace Verification
{
  using namespace dealii;
  
  template <int dim,
            typename MainClass,
            typename Iterator,
            typename ScratchData,
            typename CopyData,
            typename VectorType>
  void compare_analytical_matrix_with_fd(
    const DoFHandler<dim> &dof_handler,
    const unsigned int     n_dofs_per_cell,
    MainClass             &main_object,
    void (MainClass::*assemble_local_matrix)(const Iterator &,
                                             ScratchData &,
                                             CopyData &),
    void (MainClass::*assemble_local_rhs)(const Iterator &,
                                          ScratchData &,
                                          CopyData &),
    ScratchData &scratch_data,
    CopyData    &copy_data,
    VectorType  &present_solution,
    VectorType  &evaluation_point,
    VectorType  &local_evaluation_point,
    MPI_Comm     mpi_communicator,
    double      &max_error_over_all_elements)
  {
    std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);
    FullMatrix<double> local_matrix(n_dofs_per_cell, n_dofs_per_cell);
    FullMatrix<double> local_matrix_fd(n_dofs_per_cell, n_dofs_per_cell);
    FullMatrix<double> error_matrix(n_dofs_per_cell, n_dofs_per_cell);
    Vector<double>     ref_local_rhs(n_dofs_per_cell);
    Vector<double>     perturbed_local_rhs(n_dofs_per_cell);

    local_evaluation_point = present_solution;
    evaluation_point       = present_solution;

    double max_diff = 0.;

    // Loop over mesh elements
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      cell->get_dof_indices(local_dof_indices);

      {
        //
        // Compute analytic matrix
        //
        (main_object.*assemble_local_matrix)(cell, scratch_data, copy_data);
        local_matrix = copy_data.local_matrix;
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

        for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
        {
          const double og_value = evaluation_point[local_dof_indices[j]];

          local_evaluation_point[local_dof_indices[j]] += h;
          local_evaluation_point.compress(VectorOperation::add);
          evaluation_point = local_evaluation_point;

          // Compute perturbed RHS
          (main_object.*assemble_local_rhs)(cell, scratch_data, copy_data);
          perturbed_local_rhs = copy_data.local_rhs;

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

      // std::cout << "An matrix is " << std::endl;
      // local_matrix.print(std::cout, 12, 3);
      // std::cout << "Fd matrix is " << std::endl;
      // local_matrix_fd.print(std::cout, 12, 3);

      error_matrix.equ(1.0, local_matrix);
      error_matrix.add(-1.0, local_matrix_fd);
      max_diff = std::max(max_diff, error_matrix.linfty_norm());

      // std::cout << "Error matrix is " << std::endl;
      // error_matrix.print(std::cout, 12, 3);
      // std::cout << "Max difference is " << error_matrix.linfty_norm()
      //           << std::endl;
    }

    max_error_over_all_elements =
      Utilities::MPI::max(max_diff, mpi_communicator);

    // std::cout << "Max difference over all elements is " << global_max_diff
    //           << std::endl;
  }
} // namespace Verification

#endif