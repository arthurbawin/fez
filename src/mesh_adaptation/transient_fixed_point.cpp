
#include <deal.II/distributed/fully_distributed_tria.h>
#include <mesh_adaptation/transient_fixed_point.h>

template <int dim>
TransientFixedPointData<dim>::TransientFixedPointData(
  const unsigned int n_time_intervals,
  const MPI_Comm     mpi_communicator)
  : n_time_intervals(n_time_intervals)
  , triangulations(n_time_intervals)
  , dof_handlers(n_time_intervals)
  , present_solutions(n_time_intervals)
  , previous_solutions(n_time_intervals)
  , postproc_handlers(n_time_intervals)
{
  for (unsigned int i = 0; i < n_time_intervals; ++i)
  {
    triangulations[i] =
      std::make_unique<parallel::fullydistributed::Triangulation<dim>>(
        mpi_communicator);
    dof_handlers[i] = std::make_unique<DoFHandler<dim>>(*triangulations[i]);
    // postproc_handlers[i] = std::make_unique()
  }
}

template <int dim>
parallel::fullydistributed::Triangulation<dim> *
TransientFixedPointData<dim>::get_triangulation(
  const unsigned int interval_index)
{
  return triangulations[interval_index].get();
}

template <int dim>
DoFHandler<dim> *
TransientFixedPointData<dim>::get_dof_handler(const unsigned int interval_index)
{
  return dof_handlers[interval_index].get();
}

template <int dim>
void TransientFixedPointData<dim>::set_interval_data(
  const unsigned int                               interval_index,
  parallel::fullydistributed::Triangulation<dim> *&triangulation,
  DoFHandler<dim>                                *&dof_handler
  // ,
  // LA::ParVectorType &present_solution,
  // std::vector<LA::ParVectorType> &interval_previous_solutions,
  // PostProcessingHandler<dim> *postproc_handler
)
{
  triangulation = this->get_triangulation(interval_index);
  dof_handler   = this->get_dof_handler(interval_index);
  // present_solution = present_solutions[interval_index];
  // interval_previous_solutions = previous_solutions[interval_index];
  // postproc_handler = postproc_handlers[interval_index].get();
}

template <int dim>
void TransientFixedPointData<dim>::clear()
{
  for (unsigned int i = 0; i < n_time_intervals; ++i)
  {
    triangulations[i]->clear();
    dof_handlers[i]->clear();
  }
  // present_solution = present_solutions[interval_index];
  // interval_previous_solutions = previous_solutions[interval_index];
  // postproc_handler = postproc_handlers[interval_index].get();
}

template class TransientFixedPointData<2>;
template class TransientFixedPointData<3>;
