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

 * This class describes a CopyData which can be used with solvers where one
 * FESystem is defined on a part of the domain, and one or more other systems
 * are defined on some other partitions, using the hp tools from deal.II in the
 * spirit of the step-46 examples. To this end, this class is templated on the
 * number of partitions, the regular CopyData structure being simply the case
 * n_hp_partitions = 1.
 *
 * This class basically holds "n_hp_partitions" copies of
 * the local matrix, local rhs and local dof indices, which is exactly what a
 * MeshWorker::CopyData does, and is thus derived from that class.
 *
 * This class's name is CopyDataBase, so that we can use an alias in the
 * definition of the various derived solvers, for instance:
 *
 * using CopyData = CopyDataBase<1>;
 *
 * This is not possible if this class's name is CopyData, as otherwise the
 * compiler says that this typedef changes the meaning of CopyData. Thus, we're
 * simply using any other name that CopyData, which is then used as the local
 * alias in most solvers.
 */
template <unsigned int n_hp_partitions = 1>
class CopyDataBase : public MeshWorker::CopyData<n_hp_partitions>
{
public:
  /**
   * Constructor with non-hp FiniteElement simply calls the hp-constructor
   * below.
   */
  template <int dim>
  CopyDataBase(const FiniteElement<dim> &fe)
    : CopyDataBase(hp::FECollection<dim>(fe))
  {}

  /**
   * Constructor with hp FECollection.
   */
  template <int dim>
  CopyDataBase(const hp::FECollection<dim> &fe_collection)
    : MeshWorker::CopyData<n_hp_partitions>()
  {
    AssertDimension(n_hp_partitions, fe_collection.size());
    for (unsigned int i = 0; i < n_hp_partitions; ++i)
      this->reinit(i, fe_collection[i].n_dofs_per_cell());
  }

  /**
   * Return a reference to the @p fe_index-th local right-hand side vector.
   */
  Vector<double> &local_rhs(const unsigned int fe_index = 0);

  /**
   * Const version of the function above
   */
  const Vector<double> &local_rhs(const unsigned int fe_index = 0) const;

  /**
   * Return a reference to the @p fe_index-th local matrix.
   */
  FullMatrix<double> &local_matrix(const unsigned int fe_index = 0);

  /**
   * Const version of the function above
   */
  const FullMatrix<double> &local_matrix(const unsigned int fe_index = 0) const;

  /**
   * Return a reference to the @p fe_index-th local vector of dof indices.
   */
  std::vector<types::global_dof_index> &
  dof_indices(const unsigned int fe_index = 0);

  /**
   * Const version of the function above
   */
  const std::vector<types::global_dof_index> &
  dof_indices(const unsigned int fe_index = 0) const;

public:
  bool         cell_is_locally_owned;
  bool         cell_is_at_boundary;
  bool         cell_has_lagrange_multiplier = false;
  unsigned int last_active_fe_index;
};

/* --------------- inline functions ------------------- */

template <unsigned int n_hp_partitions>
inline Vector<double> &
CopyDataBase<n_hp_partitions>::local_rhs(const unsigned int fe_index)
{
  AssertIndexRange(fe_index, n_hp_partitions);
  return this->vectors[fe_index];
}

template <unsigned int n_hp_partitions>
inline const Vector<double> &
CopyDataBase<n_hp_partitions>::local_rhs(const unsigned int fe_index) const
{
  AssertIndexRange(fe_index, n_hp_partitions);
  return this->vectors[fe_index];
}

template <unsigned int n_hp_partitions>
inline FullMatrix<double> &
CopyDataBase<n_hp_partitions>::local_matrix(const unsigned int fe_index)
{
  AssertIndexRange(fe_index, n_hp_partitions);
  return this->matrices[fe_index];
}

template <unsigned int n_hp_partitions>
inline const FullMatrix<double> &
CopyDataBase<n_hp_partitions>::local_matrix(const unsigned int fe_index) const
{
  AssertIndexRange(fe_index, n_hp_partitions);
  return this->matrices[fe_index];
}

template <unsigned int n_hp_partitions>
inline std::vector<types::global_dof_index> &
CopyDataBase<n_hp_partitions>::dof_indices(const unsigned int fe_index)
{
  AssertIndexRange(fe_index, n_hp_partitions);
  return this->local_dof_indices[fe_index];
}

template <unsigned int n_hp_partitions>
inline const std::vector<types::global_dof_index> &
CopyDataBase<n_hp_partitions>::dof_indices(const unsigned int fe_index) const
{
  AssertIndexRange(fe_index, n_hp_partitions);
  return this->local_dof_indices[fe_index];
}

#endif
