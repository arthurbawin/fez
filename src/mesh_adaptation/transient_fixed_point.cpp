
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <mesh_adaptation/transient_fixed_point.h>
#include <mesh_adaptation_tools.h>
#include <metric_field.h>
#include <parameter_reader.h>
#include <time_handler.h>
#include <types.h>

template <int dim>
TransientFixedPointData<dim>::TransientFixedPointData(
  const ParameterReader<dim> &param,
  const unsigned int          n_time_intervals,
  const MPI_Comm              mpi_communicator)
  : param(param)
  , mpi_communicator(mpi_communicator)
  , n_time_intervals(n_time_intervals)
  , triangulations(n_time_intervals)
  , dof_handlers(n_time_intervals)
  , present_solutions(n_time_intervals)
  , previous_solutions(n_time_intervals)
  , metrics_for_adaptation(n_time_intervals)
{
  for (unsigned int i = 0; i < n_time_intervals; ++i)
  {
    triangulations[i] =
      std::make_unique<parallel::fullydistributed::Triangulation<dim>>(
        mpi_communicator);
    dof_handlers[i] = std::make_unique<DoFHandler<dim>>(*triangulations[i]);
    present_solutions[i]  = std::make_unique<LA::ParVectorType>();
    previous_solutions[i] = std::make_unique<std::vector<LA::ParVectorType>>();
    metrics_for_adaptation[i] = std::make_unique<MetricField<dim>>();
  }
}

template <int dim>
parallel::fullydistributed::Triangulation<dim> *
TransientFixedPointData<dim>::get_triangulation(
  const unsigned int interval_index)
{
  AssertIndexRange(interval_index, n_time_intervals);
  return triangulations[interval_index].get();
}

template <int dim>
DoFHandler<dim> *
TransientFixedPointData<dim>::get_dof_handler(const unsigned int interval_index)
{
  AssertIndexRange(interval_index, n_time_intervals);
  return dof_handlers[interval_index].get();
}

template <int dim>
LA::ParVectorType *TransientFixedPointData<dim>::get_present_solution(
  const unsigned int interval_index)
{
  AssertIndexRange(interval_index, n_time_intervals);
  return present_solutions[interval_index].get();
}

template <int dim>
std::vector<LA::ParVectorType> *
TransientFixedPointData<dim>::get_previous_solutions(
  const unsigned int interval_index)
{
  AssertIndexRange(interval_index, n_time_intervals);
  return previous_solutions[interval_index].get();
}

template <int dim>
MetricField<dim> *TransientFixedPointData<dim>::get_metric_field(
  const unsigned int interval_index)
{
  AssertIndexRange(interval_index, n_time_intervals);
  return metrics_for_adaptation[interval_index].get();
}

template <int dim>
std::string TransientFixedPointData<dim>::get_meshfile_name(
  const unsigned int interval_index) const
{
  const unsigned int fixed_point_iteration =
    param.mesh.adaptation.metric.current_fixed_point_iteration;

  // If this is the first fixed-point iteration, the input mesh file is the
  // initial mesh for all intervals. Otherwise, read the last adapted mesh for
  // this interval.
  if (fixed_point_iteration == 0)
    return param.mesh.filename;

  AssertIndexRange(interval_index, n_time_intervals);

  const std::string adapt_dir =
    param.output.output_dir + param.mesh.adaptation.adapt_dir;

  const std::string adapted_meshfile_on_interval =
    adapt_dir + param.mesh.adaptation.adapted_mesh_extension + "_" +
    std::to_string(interval_index) + ".msh";

  return adapted_meshfile_on_interval;
}

template <int dim>
unsigned int TransientFixedPointData<dim>::get_sum_of_active_cells() const
{
  unsigned int sum = 0;
  for (const auto &tria_ptr : triangulations)
    sum += tria_ptr->n_global_active_cells();
  return sum;
}

template <int dim>
unsigned int TransientFixedPointData<dim>::get_sum_of_vertices() const
{
  unsigned int sum = 0;
  for (const auto &metric_ptr : metrics_for_adaptation)
    sum += metric_ptr->get_n_owned_vertices();
  return Utilities::MPI::sum(sum, mpi_communicator);
}

template <int dim>
unsigned int TransientFixedPointData<dim>::get_sum_of_dofs() const
{
  unsigned int sum = 0;
  for (const auto &dh_ptr : dof_handlers)
    sum += dh_ptr->n_dofs();
  return sum;
}

template <int dim>
unsigned int TransientFixedPointData<dim>::get_effective_space_time_complexity(
  const TimeHandler &time_handler) const
{
  unsigned int sum = 0;
  for (unsigned int i = 0; i < n_time_intervals; ++i)
  {
    const auto n_vertices_i =
      Utilities::MPI::sum(metrics_for_adaptation[i]->get_n_owned_vertices(),
                          mpi_communicator);
    sum += n_vertices_i * time_handler.n_steps_on_each_interval[i];
  }
  return sum;
}

template <int dim>
void TransientFixedPointData<dim>::set_interval_data(
  const unsigned int                               interval_index,
  parallel::fullydistributed::Triangulation<dim> *&triangulation,
  DoFHandler<dim>                                *&dof_handler,
  LA::ParVectorType                              *&present_solution,
  std::vector<LA::ParVectorType>                 *&solver_previous_solutions,
  MetricField<dim>                               *&metric_for_adaptation)
{
  AssertIndexRange(interval_index, n_time_intervals);

  triangulation             = get_triangulation(interval_index);
  dof_handler               = get_dof_handler(interval_index);
  present_solution          = get_present_solution(interval_index);
  solver_previous_solutions = get_previous_solutions(interval_index);
  metric_for_adaptation     = get_metric_field(interval_index);
}

template <int dim>
void TransientFixedPointData<dim>::transfer_solution(
  const unsigned int   interval_index,
  const Mapping<dim>  &mapping,
  const Function<dim> &exact_solution)
{
  AssertIndexRange(interval_index, n_time_intervals);
  Assert(interval_index > 0, ExcInternalError());

  // Transfer solution from previous interval
  // Possible steps:
  /**
   * - get the support points of the new dofs on this partition
   * - locate on which partition in the old mesh they are located
   * - for each proc, create a list of support points whose values (for each
   * component) are requested
   * - send and receive the lists of requests
   * - fulfill the requests and send back the values
   */

  // For now and for prototyping: simply evaluate re-interpolate the exact
  // solution at the beginning of this interval.
  auto &present_solution = *present_solutions[interval_index];
  auto &dof_handler      = *dof_handlers[interval_index];

  LA::ParVectorType distributed_present_solution(
    dof_handler.locally_owned_dofs(), mpi_communicator);

  // Interpolate present solution
  VectorTools::interpolate(mapping,
                           dof_handler,
                           exact_solution,
                           distributed_present_solution);
  present_solution = distributed_present_solution;

  // Copy/interpolate previous solutions
  auto &previous_solutions_this_interval = *previous_solutions[interval_index];
  auto &previous_solutions_previous_interval =
    *previous_solutions[interval_index - 1];
  for (unsigned int i = 0; i < previous_solutions_this_interval.size(); ++i)
    previous_solutions_this_interval[i] =
      previous_solutions_previous_interval[i];
}

template <int dim>
void TransientFixedPointData<dim>::scale_metrics(
  const unsigned int metric_index,
  const TimeHandler &time_handler)
{
  if (param.time_integration.is_steady())
  {
    // Scaling for steady-state adaptation.
    AssertThrow(n_time_intervals == 1,
                ExcMessage(
                  "When solving for steady-state solution, a single time "
                  "subinterval is expected."));
    metrics_for_adaptation[0]->apply_optimal_steady_multiscale_scaling();
  }
  else
  {
    // Scaling for the transient method.

    // Space-time complexity N_st.
    // If this is a convergence study with anisotropic adaptation, get this
    // value from the MMS parameters.
    const double N =
      param.mms_param.enable ?
        (double)param.mms_param.n_target_vertices :
        (double)param.metric_fields[metric_index].multiscale.n_target_vertices;

    // FIXME: use general exponents for higher order solutions
    const double p   = (double)param.metric_fields[metric_index].multiscale.p;
    const double d   = (double)dim;
    const double den = 2. * p + d;
    const double exponent                  = p / den;
    const double exponent_int_steps        = 2. * p / den;
    const double exponent_local_scaling    = -1. / den;
    const double exponent_local_scaling_dt = -2. / den;

    // Integral of det(Q)^exponent
    // std::vector<double> integral_determinants(n_time_intervals);

    const auto &n_steps_on_each_interval =
      time_handler.n_steps_on_each_interval;

    AssertDimension(n_steps_on_each_interval.size(), n_time_intervals);

    // Compute the global scaling factors from the collection of metrics
    double global_scaling = 0.;
    for (unsigned int i = 0; i < n_time_intervals; ++i)
    {
      auto &metric = *metrics_for_adaptation[i];

      // Update the FE solution to compute integral of determinant
      metric.metrics_to_tensor_solution();
      const double Ki = metric.compute_integral_determinant(exponent);
      global_scaling +=
        Ki * std::pow(n_steps_on_each_interval[i], exponent_int_steps);

      // Apply local scaling by (det Q) ^ (-1 / (2 * p + d)) * (int_interval
      // (dt)^-1) ^(- 2 / (2 * p + d))
      for (auto &m : metric.metrics)
        m *= std::pow(determinant(m), exponent_local_scaling) *
             std::pow(n_steps_on_each_interval[i], exponent_local_scaling_dt);
    }

    // Apply global scaling (operator *= includes update of FE solution and
    // ghosts)
    for (const auto &m_ptr : metrics_for_adaptation)
      (*m_ptr) *= std::pow(N / global_scaling, 2. / d);
  }
}

template <int dim>
void TransientFixedPointData<dim>::apply_gradation_to_metrics()
{
  for (const auto &m_ptr : metrics_for_adaptation)
    m_ptr->apply_gradation();
}

template <int dim>
void TransientFixedPointData<dim>::adapt_meshes()
{
  const bool verbose =
    Utilities::MPI::this_mpi_process(mpi_communicator) == 0 &&
    param.mesh.adaptation.verbosity == Parameters::Verbosity::verbose;

  if (verbose && !param.time_integration.is_steady())
  {
    std::cout << std::endl;
    std::cout << "-- Adapting meshes on all time intervals :" << std::endl;
  }

  const std::string adapt_dir =
    param.output.output_dir + param.mesh.adaptation.adapt_dir;

  for (unsigned int i = 0; i < n_time_intervals; ++i)
  {
    const std::string input_meshfile = get_meshfile_name(i);
    const std::string output_meshfile =
      adapt_dir + param.mesh.adaptation.adapted_mesh_extension + "_" +
      std::to_string(i) + ".msh";

    MeshTools::adapt_with_mmg(param,
                              *metrics_for_adaptation[i],
                              adapt_dir,
                              input_meshfile,
                              output_meshfile,
                              i);
  }
}

template <int dim>
void TransientFixedPointData<dim>::clear()
{
  for (unsigned int i = 0; i < n_time_intervals; ++i)
  {
    triangulations[i]->clear();
    dof_handlers[i]->clear();
    present_solutions[i]->clear();
    previous_solutions[i]->clear();
    metrics_for_adaptation[i]->clear();
  }
}

template <int dim>
void TransientFixedPointData<dim>::write_summary(
  const TimeHandler &time_handler,
  std::ostream      &out) const
{
  TableHandler intervals_summary;

  for (unsigned int i = 0; i < n_time_intervals; ++i)
  {
    intervals_summary.add_value("Interval", i);
    intervals_summary.add_value("from", time_handler.initial_times[i]);
    intervals_summary.add_value("to", time_handler.final_times[i]);
    intervals_summary.add_value("Time steps",
                                time_handler.n_steps_on_each_interval[i]);
    intervals_summary.add_value(
      "# of vertices",
      Utilities::MPI::sum(metrics_for_adaptation[i]->get_n_owned_vertices(),
                          mpi_communicator));
    intervals_summary.add_value("# of elements",
                                triangulations[i]->n_global_active_cells());
    intervals_summary.add_value("# of dofs", dof_handlers[i]->n_dofs());
  }

  if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
  {
    std::cout << "Summary of all time sub-intervals" << std::endl;
    intervals_summary.write_text(
      out, TableHandler::TextOutputFormat::org_mode_table);
  }
}

template class TransientFixedPointData<2>;
template class TransientFixedPointData<3>;
