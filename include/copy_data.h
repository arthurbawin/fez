#ifndef COPY_DATA_H
#define COPY_DATA_H

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>

using namespace dealii;

/**
 * 
 * 
 */
class CopyData
{
public:
  /**
   * Constructor
   */
  CopyData(const unsigned int n_dofs_per_cell)
  : local_matrix(n_dofs_per_cell, n_dofs_per_cell)
  , local_rhs(n_dofs_per_cell)
  , local_dof_indices(n_dofs_per_cell)
  {}

  void reinit_matrix()
  {
    local_matrix = 0;
  }
  void reinit_rhs()
  {
    local_rhs    = 0;
  }

public:
  FullMatrix<double> local_matrix;
  Vector<double>     local_rhs;

  std::vector<types::global_dof_index> local_dof_indices;
  
  bool cell_is_locally_owned;
};

#endif