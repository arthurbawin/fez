#ifndef COPY_DATA_H
#define COPY_DATA_H

#include <deal.II/hp/fe_collection.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/vector.h>
#include <deal.II/meshworker/copy_data.h>

using namespace dealii;

/**
 * A CopyData is a helper structure used to copy the local matrix and RHS
 * contributions to the global system, see for instance the deal.II doc of
 * meshworker/copy_data.h and the run() functions in work_stream.h.
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

public:
  FullMatrix<double> local_matrix;
  Vector<double>     local_rhs;

  std::vector<types::global_dof_index> local_dof_indices;

  bool cell_is_locally_owned;
};

/**
 * CopyData with hp capabilities.
 *
 * This is basically a CopyData as above with "n_hp_partitions" copies of
 * the local matrix, local rhs and local dof indices, which is exactly what a
 * MeshWorker::CopyData does.
 *
 * FIXME: Ideally there would be a single class template, maybe with
 * specializations.
 */
template <int dim, unsigned int n_hp_partitions>
class MyCopyData : public MeshWorker::CopyData<n_hp_partitions>
{
public:
  /**
   * Constructor with hp FECollection
   */
  MyCopyData(const hp::FECollection<dim> &fe_collection)
    : MeshWorker::CopyData<n_hp_partitions>()
  {
    AssertDimension(n_hp_partitions, fe_collection.size());
    for (unsigned int i = 0; i < n_hp_partitions; ++i)
      this->reinit(i, fe_collection[i].n_dofs_per_cell());
  }

public:
  bool         cell_is_locally_owned;
  unsigned int last_active_fe_index;
};

#endif