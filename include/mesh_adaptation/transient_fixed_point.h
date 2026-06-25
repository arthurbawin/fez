#ifndef TRANSIENT_FIXED_POINT_CPP
#define TRANSIENT_FIXED_POINT_CPP

#include <deal.II/base/timer.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/solution_transfer.h>
#include <metric_field.h>
#include <parameter_reader.h>
#include <time_handler.h>
#include <types.h>

/**
 * A class storing the data that need to be duplicated for each time
 * subinterval when using the transient fixed point mesh adaptation method for
 * unsteady simulations. To use a common interface, this structure is also used
 * for non-adaptive simulations, either steady or unsteady, in which case
 * this class stores the data for the single, full simulation time interval.
 *
 * This class stores and owns (in the sense of memory management) the
 * triangulations and dof handlers, present and previous solutions vectors, as
 * well as the Riemannian metrics needed for anisotropic mesh adaptation.
 *
 * The intended use of this class is to have each solver own one such object,
 * and store "observer" pointers, in the form of
 * raw, non-owning pointers, to the triangulation, dof handler, etc. associated
 * with the index-th time subinterval.
 *
 * A more fitting name for this class would maybe be "SolverDataCollection".
 */
template <int dim>
class TransientFixedPointData
{
public:
  /**
   * Constructor. Initializes the data for @p n subintervals.
   */
  TransientFixedPointData(
    const ParameterReader<dim>                      &param,
    TimerOutput                                     &timer,
    const unsigned int                               n_time_intervals,
    const MPI_Comm                                   mpi_communicator,
    parallel::DistributedTriangulationBase<dim> *&triangulation,
    DoFHandler<dim>                                *&dof_handler,
    LA::ParVectorType                              *&present_solution,
    std::vector<LA::ParVectorType>                 *&solver_previous_solutions,
    MetricField<dim>                               *&metric_for_adaptation);

  /**
   * Reinitialize this object to hold data for @p n_time_intervals intervals.
   */
  void reinit(const unsigned int                               n_time_intervals,
              parallel::DistributedTriangulationBase<dim> *&triangulation,
              DoFHandler<dim>                                *&dof_handler,
              LA::ParVectorType                              *&present_solution,
              std::vector<LA::ParVectorType> *&solver_previous_solutions,
              MetricField<dim>               *&metric_for_adaptation);

  /**
   * Assign the data associated with the interval_index-th time interval to the
   * given pointers.
   */
  void set_interval_data(
    const unsigned int                               interval_index,
    parallel::DistributedTriangulationBase<dim> *&triangulation,
    DoFHandler<dim>                                *&dof_handler,
    LA::ParVectorType                              *&present_solution,
    std::vector<LA::ParVectorType>                 *&solver_previous_solutions,
    MetricField<dim>                               *&metric_for_adaptation);

  /**
   * Return the number of time intervals stored in this object.
   */
  unsigned int get_n_time_intervals() const;

  /**
   * Get the raw pointer to the @p interval_index-th triangulation.
   */
  parallel::DistributedTriangulationBase<dim> *
  get_triangulation(const unsigned int interval_index);

  /**
   * Get the raw pointer to the @p interval_index-th dof handler.
   */
  DoFHandler<dim> *get_dof_handler(const unsigned int interval_index);

  /**
   * Get the raw pointer to the @p interval_index-th solution vector.
   */
  LA::ParVectorType *get_present_solution(const unsigned int interval_index);

  /**
   * Get the raw pointer to the @p interval_index-th vector of previous solution
   * vectors.
   */
  std::vector<LA::ParVectorType> *
  get_previous_solutions(const unsigned int interval_index);

  /**
   * Get the raw pointer to the @p interval_index-th Riemannian metric.
   */
  MetricField<dim> *get_metric_field(const unsigned int interval_index);

  /**
   * Const version of the above
   */
  const MetricField<dim> *
  get_metric_field(const unsigned int interval_index) const;

  /**
   * Return the name of the mesh file to be used for this time interval.
   * If this is the first fixed-point iteration, this is the initial mesh file.
   * Otherwise, this is the name of the adapted mesh on this interval.
   */
  std::string get_meshfile_name(const unsigned int interval_index) const;

  /**
   * Return the sum of the active cells among the triangulations of all time
   * intervals.
   */
  unsigned int get_sum_of_active_cells() const;

  /**
   * Return the sum of the mesh vertices among the triangulations of all time
   * intervals.
   */
  unsigned int get_sum_of_vertices() const;

  /**
   * Return the sum of all dofs among the dof handlers of all time intervals.
   */
  unsigned int get_sum_of_dofs() const;

  /**
   * Return the effective space-time complexity associated with the collection
   * of triangulations stored in object and the given @p time_handler.
   * This quantity is defined by:
   *
   *   N_st := sum_intervals n_mesh_vertices_i * n_time_steps_i,
   *
   * where n_time_steps_i is the number of time steps spent in the i-th time
   * interval.
   */
  unsigned int
  get_effective_space_time_complexity(const TimeHandler &time_handler) const;

  /**
   * Transfer the current and previous solution from the (interval_index - 1)-th
   * interval to the @p interval_index-th interval.
   */
  void transfer_solution_between_intervals(
    const unsigned int                interval_index,
    const Mapping<dim>               &mapping,
    Function<dim>                    &exact_solution,
    const TimeHandler                &time_handler,
    const IndexSet                   &locally_relevant_dofs,
    const std::vector<unsigned char> &dofs_to_component);

  /**
   * Transfer the solutions associated with the previous state of refinement of
   * the triangulation to the solutions associated with its current state. This
   * function is used when using the deal.II refinement and coarsening routines,
   * and thus expects a single time interval (i.e., a single triangulation) for
   * now.
   *
   * This functions interpolates the data stored in the SolutionTransfer object,
   * which must have been previously (re-)initialized in adapt_meshes().
   */
  void transfer_solution_between_refinements(
    const IndexSet                  &locally_relevant_dofs,
    const AffineConstraints<double> &nonzero_constraints);

  /**
   * Apply a local and global scaling to all metric fields.
   */
  void scale_metrics(const unsigned int metric_index,
                     const TimeHandler &time_handler);

  /**
   * Apply gradation to all metric fields.
   */
  void apply_gradation_to_metrics();

  /**
   * Adapt the meshes on all subintervals.
   */
  void adapt_meshes(const TimeHandler   &time_handler,
                    const Vector<float> &criteria);

  /**
   * Clear the data for each subinterval.
   */
  void clear();

  /**
   * Write a summary of the data stored in this object, such as the number of
   * intervals, their starting and ending times, the number of time steps spent
   * in each interval, the number of mesh vertices, of dofs, etc.
   */
  void write_summary(const TimeHandler &time_handler,
                     std::ostream      &out = std::cout) const;

private:
  /**
   * Function that actually performs the solution transfer in parallel.
   * It uses VectorTools::point_values() which is templated on the number of
   * components in the FESystem, so this function is templated similarly.
   */
  template <int n_components>
  void
  do_solution_transfer(const unsigned int                interval_index,
                       const Mapping<dim>               &mapping,
                       Function<dim>                    &exact_solution,
                       const TimeHandler                &time_handler,
                       const IndexSet                   &locally_relevant_dofs,
                       const std::vector<unsigned char> &dofs_to_component);

  /**
   * Adapt the mesh for each time interval with the mmg library.
   */
  void adapt_meshes_with_mmg();

  /**
   * Adapt the mesh using deal.II's adaptation routines.
   * This function assumes a single time interval, and adapts the mesh
   * associated with the first interval.
   *
   * FIXME: maybe template on Number instead of float.
   */
  void adapt_mesh_with_dealii_routines(const Vector<float> &criteria);

private:
  /**
   *
   */
  const ParameterReader<dim> &param;

  /**
   *
   */
  TimerOutput &timer;

  /**
   *
   */
  MPI_Comm mpi_communicator;

  /**
   *
   */
  unsigned int n_time_intervals;

  /**
   *
   */
  std::vector<std::unique_ptr<parallel::DistributedTriangulationBase<dim>>>
    triangulations;

  /**
   *
   */
  std::vector<std::unique_ptr<DoFHandler<dim>>> dof_handlers;

  /**
   *
   */
  std::vector<std::unique_ptr<LA::ParVectorType>> present_solutions;

  /**
   *
   */
  std::vector<std::unique_ptr<std::vector<LA::ParVectorType>>>
    previous_solutions;

  /**
   *
   */
  std::vector<std::unique_ptr<MetricField<dim>>> metrics_for_adaptation;

  /**
   *
   */
  std::unique_ptr<SolutionTransfer<dim, LA::ParVectorType>> solution_transfer;
};

#endif
