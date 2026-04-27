#ifndef TRANSIENT_FIXED_POINT_CPP
#define TRANSIENT_FIXED_POINT_CPP

#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <post_processing_handler.h>
#include <types.h>

/**
 * A class storing the data that need to be duplicated for each time
 * sub-interval when using the transient fixed point mesh adaptation method for
 * unsteady simulations.
 *
 * This class stores and owns vectors of triangulations, of dof handlers, etc.
 * The solvers using it should only store "observer" pointers, in the form of
 * raw, non-owning pointers to the index-th interval data.
 *  */
template <int dim>
class TransientFixedPointData
{
public:
  /**
   * Constructor. Initializes the data for @p n subintervals.
   */
  TransientFixedPointData(const unsigned int n_time_intervals,
                          const MPI_Comm     mpi_communicator);

  /**
   *
   */
  parallel::fullydistributed::Triangulation<dim> *
  get_triangulation(const unsigned int interval_index);

  /**
   *
   */
  DoFHandler<dim> *get_dof_handler(const unsigned int interval_index);

  /**
   *
   */
  void set_interval_data(
    const unsigned int                               interval_index,
    parallel::fullydistributed::Triangulation<dim> *&triangulation,
    DoFHandler<dim>                                *&dof_handler);
  // ,
  // LA::ParVectorType &present_solution,
  // std::vector<LA::ParVectorType> &interval_previous_solutions,
  // PostProcessingHandler<dim> *postproc_handler
  // );

  /**
   *
   */
  void clear();

public:
  /**
   *
   */
  const unsigned int n_time_intervals;

  /**
   *
   */
  std::vector<std::unique_ptr<parallel::fullydistributed::Triangulation<dim>>>
    triangulations;

  /**
   *
   */
  std::vector<std::unique_ptr<DoFHandler<dim>>> dof_handlers;

  /**
   *
   */
  std::vector<LA::ParVectorType> present_solutions;

  /**
   *
   */
  std::vector<std::vector<LA::ParVectorType>> previous_solutions;

  /**
   * Is it required, or can we re-create a single postproc per interval?
   */
  std::vector<std::unique_ptr<PostProcessingHandler<dim>>> postproc_handlers;

  // Also add patch handlers, recoveries, metric fields
  // Actually maybe only the metric fields, since the patches and recoveries are
  // needed to increment the metrics, but we don't need the complete
  // patches/recoveries at any point, unlike the metrics (which need a global
  // scaling to adapt their own subinterval mesh).
};

#endif
