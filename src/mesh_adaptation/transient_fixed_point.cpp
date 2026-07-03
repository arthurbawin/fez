
#include <deal.II/base/function.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_remote_point_evaluation.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/distributed/tria_base.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <mesh_adaptation/transient_fixed_point.h>
#include <mesh_adaptation_tools.h>
#include <metric_field.h>
#include <parameter_reader.h>
#include <time_handler.h>
#include <types.h>
#include <utilities.h>

template <int dim>
TransientFixedPointData<dim>::TransientFixedPointData(
  const ParameterReader<dim>                   &param,
  TimerOutput                                  &timer,
  const unsigned int                            n_time_intervals,
  const MPI_Comm                                mpi_communicator,
  parallel::DistributedTriangulationBase<dim> *&triangulation,
  DoFHandler<dim>                             *&dof_handler,
  LA::ParVectorType                           *&present_solution,
  std::vector<LA::ParVectorType>              *&solver_previous_solutions,
  MetricField<dim>                            *&metric_for_adaptation)
  : param(param)
  , timer(timer)
  , mpi_communicator(mpi_communicator)
  , n_time_intervals(n_time_intervals)
{
  reinit(n_time_intervals,
         triangulation,
         dof_handler,
         present_solution,
         solver_previous_solutions,
         metric_for_adaptation);
}

template <int dim>
void TransientFixedPointData<dim>::reinit(
  const unsigned int                            new_n_time_intervals,
  parallel::DistributedTriangulationBase<dim> *&triangulation,
  DoFHandler<dim>                             *&dof_handler,
  LA::ParVectorType                           *&present_solution,
  std::vector<LA::ParVectorType>              *&solver_previous_solutions,
  MetricField<dim>                            *&metric_for_adaptation)
{
  // Clear and reallocate for the new number of intervals
  if (triangulations.size() > 0)
    clear();

  n_time_intervals = new_n_time_intervals;

  triangulations.resize(n_time_intervals);
  dof_handlers.resize(n_time_intervals);
  present_solutions.resize(n_time_intervals);
  previous_solutions.resize(n_time_intervals);
  metrics_for_adaptation.resize(n_time_intervals);

  for (unsigned int i = 0; i < n_time_intervals; ++i)
  {
    // Use a parallel::distributed::Triangulation if tree-based AMR is enabled.
    // Otherwise, use a parallel::fullydistributed::Triangulation.
    if (param.mesh.adaptation.strategy ==
        Parameters::Mesh::Adaptation::Strategy::LocalRefinement)
#if defined(DEAL_II_WITH_P4EST)
      triangulations[i] =
        std::make_unique<parallel::distributed::Triangulation<dim>>(
          mpi_communicator);
#else
      AssertThrow(false, ExcDealiiNeedsP4EST());
#endif
    else
      triangulations[i] =
        std::make_unique<parallel::fullydistributed::Triangulation<dim>>(
          mpi_communicator);

    dof_handlers[i] = std::make_unique<DoFHandler<dim>>(*triangulations[i]);
    present_solutions[i]  = std::make_unique<LA::ParVectorType>();
    previous_solutions[i] = std::make_unique<std::vector<LA::ParVectorType>>();
    metrics_for_adaptation[i] = std::make_unique<MetricField<dim>>();
  }

  // Assign the data at interval 0 to the passed non-owning pointers
  this->set_interval_data(0,
                          triangulation,
                          dof_handler,
                          present_solution,
                          solver_previous_solutions,
                          metric_for_adaptation);
}

template <int dim>
void TransientFixedPointData<dim>::set_interval_data(
  const unsigned int                            interval_index,
  parallel::DistributedTriangulationBase<dim> *&triangulation,
  DoFHandler<dim>                             *&dof_handler,
  LA::ParVectorType                           *&present_solution,
  std::vector<LA::ParVectorType>              *&solver_previous_solutions,
  MetricField<dim>                            *&metric_for_adaptation)
{
  AssertIndexRange(interval_index, n_time_intervals);

  triangulation             = get_triangulation(interval_index);
  dof_handler               = get_dof_handler(interval_index);
  present_solution          = get_present_solution(interval_index);
  solver_previous_solutions = get_previous_solutions(interval_index);
  metric_for_adaptation     = get_metric_field(interval_index);
}

template <int dim>
unsigned int TransientFixedPointData<dim>::get_n_time_intervals() const
{
  return n_time_intervals;
}

template <int dim>
parallel::DistributedTriangulationBase<dim> *
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
const MetricField<dim> *TransientFixedPointData<dim>::get_metric_field(
  const unsigned int interval_index) const
{
  AssertIndexRange(interval_index, n_time_intervals);
  return metrics_for_adaptation[interval_index].get();
}

template <int dim>
std::string TransientFixedPointData<dim>::get_meshfile_name(
  const unsigned int interval_index) const
{
  // If not adapting with metrics, keep provided mesh file
  if (!param.mesh.adaptation.with_metric_based_adaptation())
    return param.mesh.filename;

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
template <int n_components>
void TransientFixedPointData<dim>::do_solution_transfer(
  const unsigned int                interval_index,
  const Mapping<dim>               &mapping,
  Function<dim>                    &exact_solution,
  const TimeHandler                &time_handler,
  const IndexSet                   &locally_relevant_dofs,
  const std::vector<unsigned char> &dofs_to_component)
{
  if (param.mesh.adaptation.metric.transfer_solution)
  {
    /**
     * Transfer solution from the previous interval in parallel, using the
     * Utilities::MPI::RemotePointEvaluation tools.
     *
     * This is (a priori) the worst possible case for solution transfer, as we
     * use different fully_distributed triangulations across intervals, whose
     * partitions are completely unrelated.
     *
     * This function is not expected to scale well, but it is only called
     * N_intervals - 1 times, where N_intervals is typically small compared to
     * the number of time steps per interval.
     */

    // Data on the current interval, to be transferred into
    const auto &dh                   = *dof_handlers[interval_index];
    auto       &solution             = *present_solutions[interval_index];
    auto &previous_solutions_current = *previous_solutions[interval_index];

    // Data on the previous interval, to be transferred from
    const auto &dh_prev       = *dof_handlers[interval_index - 1];
    const auto &solution_prev = *present_solutions[interval_index - 1];
    const auto &previous_solutions_prev =
      *previous_solutions[interval_index - 1];

    Assert(previous_solutions_current.size() == previous_solutions_prev.size(),
           ExcInternalError());

    LA::ParVectorType distributed_solution(dh.locally_owned_dofs(),
                                           mpi_communicator);

    /**
     * The VectorTools::point_values() function takes a vector of evaluation
     * points and returns the interpolated values in the same order, then we
     * need to assign them in the solution vector.
     *
     * Here we create the vectors of evaluation points and of dofs at these
     * points for each component from the map of (locally relevant) support
     * points.
     *
     * For FESystems with more than one component, *all* components will be
     * interpolated at *all* evaluation points, but not all these components
     * have corresponding dofs (e.g., with a P2-P1 Taylor-Hood element, pressure
     * dofs will be interpolated at P2 dofs support points even if there is no
     * pressure dof there). Set the dofs to invalid by default for all
     * components, then overwrite and only transfer in the solution vector if
     * the dof is valid for that component.
     */

    // The (locally relevant) support points at which the solutions on the
    // previous interval are evaluated.
    const std::map<types::global_dof_index, Point<dim>> support_points =
      DoFTools::map_dofs_to_support_points(mapping, dh);

    // Convert into a map [support_point : dofs]
    std::map<Point<dim>,
             std::vector<types::global_dof_index>,
             PointComparator<dim>>
      points_to_dofs;
    for (const auto &[dof, pt] : support_points)
      points_to_dofs[pt].push_back(dof);

    const unsigned int n_pts = points_to_dofs.size();

    // Convert map into vector
    std::vector<std::vector<types::global_dof_index>> local_dofs(
      n_pts,
      std::vector<types::global_dof_index>(n_components,
                                           numbers::invalid_unsigned_int));
    std::vector<Point<dim>> evaluation_points(n_pts);

    unsigned int i = 0;
    for (const auto &[pt, dofs] : points_to_dofs)
    {
      evaluation_points[i] = pt;

      for (const auto dof : dofs)
      {
        const unsigned char comp =
          dofs_to_component[locally_relevant_dofs.index_within_set(dof)];

        Assert(comp != static_cast<unsigned char>(-1), ExcInternalError());
        Assert(local_dofs[i][comp] == numbers::invalid_unsigned_int,
               ExcInternalError());

        local_dofs[i][comp] = dof;
      }
      ++i;
    }

    // This guy then does the heavy-lifting
    Utilities::MPI::RemotePointEvaluation<dim, dim> cache;

    // Transfer current solution from the previous interval onto this interval.
    // Call point_values() overload with 5 arguments to initialize the cache.
    // The subsequent calls use 3 arguments, which does not reinitialize it.
    // See also the comments in vector_tools_evaluate.h in deal.II.
    const auto transferred_solution = VectorTools::point_values<n_components>(
      mapping, dh_prev, solution_prev, evaluation_points, cache);

    Assert(cache.all_points_found(), ExcInternalError());

    for (unsigned int i = 0; i < n_pts; ++i)
    {
      // For 1 component, transferred_solution is a vector of Tensor<1,0>,
      // thus a vector of double which are not subscriptable.
      if constexpr (n_components == 1)
        distributed_solution[local_dofs[i][0]] = transferred_solution[i];
      else
        for (unsigned int c = 0; c < n_components; ++c)
          if (local_dofs[i][c] != numbers::invalid_unsigned_int)
            distributed_solution[local_dofs[i][c]] = transferred_solution[i][c];
    }
    distributed_solution.compress(VectorOperation::insert);
    solution = distributed_solution;

    // Then transfer the previous solutions defined on the previous interval.
    // Call point_values() overload with 3 arguments, which does *not*
    // reinitialize the cache.
    for (unsigned int k = 0; k < previous_solutions_current.size(); ++k)
    {
      const auto &previous_sol_prev = previous_solutions_prev[k];
      const auto  transferred_solution =
        VectorTools::point_values<n_components>(cache,
                                                dh_prev,
                                                previous_sol_prev);

      Assert(cache.all_points_found(), ExcInternalError());

      for (unsigned int i = 0; i < n_pts; ++i)
      {
        if constexpr (n_components == 1)
          distributed_solution[local_dofs[i][0]] = transferred_solution[i];
        else
          for (unsigned int c = 0; c < n_components; ++c)
            if (local_dofs[i][c] != numbers::invalid_unsigned_int)
              distributed_solution[local_dofs[i][c]] =
                transferred_solution[i][c];
      }
      distributed_solution.compress(VectorOperation::insert);
      previous_solutions_current[k] = distributed_solution;
    }
  }
  else
  {
    // Interpolate the exact solutions at the beginning of this interval.
    auto &dof_handler             = *dof_handlers[interval_index];
    auto &present_solution        = *present_solutions[interval_index];
    auto &this_previous_solutions = *previous_solutions[interval_index];

    LA::ParVectorType distributed_solution(dof_handler.locally_owned_dofs(),
                                           mpi_communicator);

    // Interpolate present solution
    VectorTools::interpolate(mapping,
                             dof_handler,
                             exact_solution,
                             distributed_solution);
    present_solution = distributed_solution;

    // Interpolate previous solutions, at previous times
    for (unsigned int k = 0; k < this_previous_solutions.size(); ++k)
    {
      exact_solution.set_time(time_handler.simulation_times[1 + k]);
      VectorTools::interpolate(mapping,
                               dof_handler,
                               exact_solution,
                               distributed_solution);
      this_previous_solutions[k] = distributed_solution;
    }

    // Restore time
    exact_solution.set_time(time_handler.simulation_times[0]);
  }
}

template <int dim>
void TransientFixedPointData<dim>::transfer_solution_between_intervals(
  const unsigned int                interval_index,
  const Mapping<dim>               &mapping,
  Function<dim>                    &exact_solution,
  const TimeHandler                &time_handler,
  const IndexSet                   &locally_relevant_dofs,
  const std::vector<unsigned char> &dofs_to_component)
{
  TimerOutput::Scope t(timer, "Transfer solutions between intervals");

  AssertIndexRange(interval_index, n_time_intervals);
  Assert(interval_index > 0, ExcInternalError());
  Assert(dofs_to_component.size() > 0,
         ExcMessage(
           "Assign each dof to its FE component before calling this function"));

  const unsigned int n_components =
    dof_handlers[interval_index]->get_fe().n_components();

  /**
   * VectorTools::point_values<n_components> needs the number of components
   * at compile time, which we currently do not have access to from the NS
   * base class.
   *
   * FIXME: would be cleaner to have a way to get it, but that probably
   * involves adding a template parameter to the NS base class...
   */
  switch (n_components)
  {
    case 1:
      do_solution_transfer<1>(interval_index,
                              mapping,
                              exact_solution,
                              time_handler,
                              locally_relevant_dofs,
                              dofs_to_component);
      break;
    case 2:
      do_solution_transfer<2>(interval_index,
                              mapping,
                              exact_solution,
                              time_handler,
                              locally_relevant_dofs,
                              dofs_to_component);
      break;
    case 3:
      do_solution_transfer<3>(interval_index,
                              mapping,
                              exact_solution,
                              time_handler,
                              locally_relevant_dofs,
                              dofs_to_component);
      break;
    case 4:
      do_solution_transfer<4>(interval_index,
                              mapping,
                              exact_solution,
                              time_handler,
                              locally_relevant_dofs,
                              dofs_to_component);
      break;
    case 5:
      do_solution_transfer<5>(interval_index,
                              mapping,
                              exact_solution,
                              time_handler,
                              locally_relevant_dofs,
                              dofs_to_component);
      break;
    case 6:
      do_solution_transfer<6>(interval_index,
                              mapping,
                              exact_solution,
                              time_handler,
                              locally_relevant_dofs,
                              dofs_to_component);
      break;
    case 7:
      do_solution_transfer<7>(interval_index,
                              mapping,
                              exact_solution,
                              time_handler,
                              locally_relevant_dofs,
                              dofs_to_component);
      break;
    case 8:
      do_solution_transfer<8>(interval_index,
                              mapping,
                              exact_solution,
                              time_handler,
                              locally_relevant_dofs,
                              dofs_to_component);
      break;
    case 9:
      do_solution_transfer<9>(interval_index,
                              mapping,
                              exact_solution,
                              time_handler,
                              locally_relevant_dofs,
                              dofs_to_component);
      break;
    case 10:
      do_solution_transfer<10>(interval_index,
                               mapping,
                               exact_solution,
                               time_handler,
                               locally_relevant_dofs,
                               dofs_to_component);
      break;
    default:
      // If not enough, simply add the case when needed (or a better solution
      // will have been found by then (-: )
      AssertThrow(
        false,
        ExcMessage(
          "Solution transfer is not implemented for this number of FESystem "
          "components (" +
          std::to_string(n_components) +
          "). This is very easy to solve however, as you only need to add a "
          "call to do_solution_transfer with this number of components."));
  }
}

template <int dim>
void TransientFixedPointData<dim>::transfer_solution_between_refinements(
  const IndexSet                  &locally_relevant_dofs,
  const AffineConstraints<double> &nonzero_constraints)
{
  TimerOutput::Scope t(timer, "Transfer solution between refinements");

  // Adapt only the first interval for now
  const auto &dh               = *dof_handlers[0];
  auto       &present_solution = *present_solutions[0];
  auto       &previous_sols    = *previous_solutions[0];

  LA::ParVectorType completely_distributed_solution(dh.locally_owned_dofs(),
                                                    mpi_communicator);

  std::vector<LA::ParVectorType> all_out(previous_sols.size() + 1);
  for (unsigned int i = 0; i < all_out.size(); ++i)
    all_out[i] = completely_distributed_solution;

  Assert(solution_transfer, ExcInternalError());

  // Interpolate the (ghosted) solutions stored in the SolutionTransfer object.
  solution_transfer->interpolate(all_out);

  // Apply the nonhomogeneous and hanging nodes constraints to the current sol
  nonzero_constraints.distribute(all_out[0]);
  present_solution = all_out[0];

  /**
   * At this point, we only have the nonzero_constraints for the current time
   * available, but not for the previous times, see also the discussion in
   * step-86.cc (transfer_solution_vectors_to_new_mesh).
   *
   * What we can do, however, is create the hanging node constraints for the
   * current mesh, and apply them to the previous solutions.
   */
  AffineConstraints<double> hanging_node_constraints(dh.locally_owned_dofs(),
                                                     locally_relevant_dofs);
  DoFTools::make_hanging_node_constraints(dh, hanging_node_constraints);
  hanging_node_constraints.close();

  for (unsigned int i = 0; i < previous_sols.size(); ++i)
  {
    hanging_node_constraints.distribute(all_out[i + 1]);
    previous_sols[i] = all_out[i + 1];
  }
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

    // Target average spatial complexity on each time interval.
    // If this is a convergence study with anisotropic adaptation, get this
    // value from the MMS parameters.
    const double Navg =
      param.mms_param.enable ?
        (double)param.mms_param.n_target_vertices :
        (double)param.metric_fields[metric_index].multiscale.n_target_vertices;

    const auto &n_steps_on_each_interval =
      time_handler.n_steps_on_each_interval;

    AssertDimension(n_steps_on_each_interval.size(), n_time_intervals);

    // Space-time complexity N_st.
    // If we want Navg vertices for each mesh on which "n_steps" time steps
    // were performed, the total space-time complexity should be the sum
    // of Navg * n_steps over each interval.

    // FIXME: quadruple-check this during the convergence study
    // const double Nst = Navg * n_time_intervals;

    double Nst = 0;
    for (auto n_steps : n_steps_on_each_interval)
    {
      Assert(n_steps > 0, ExcInternalError());
      Nst += Navg * n_steps;
    }

    // FIXME: use general exponents for higher order solutions
    const double p   = (double)param.metric_fields[metric_index].multiscale.p;
    const double d   = (double)dim;
    const double den = 2. * p + d;
    const double exponent                  = p / den;
    const double exponent_int_steps        = 2. * p / den;
    const double exponent_local_scaling    = -1. / den;
    const double exponent_local_scaling_dt = -2. / den;

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
    {
      (*m_ptr) *= std::pow(Nst / global_scaling, 2. / d);
      m_ptr->is_scaled = true;
    }
  }
}

template <int dim>
void TransientFixedPointData<dim>::apply_gradation_to_metrics()
{
  for (const auto &m_ptr : metrics_for_adaptation)
    m_ptr->apply_gradation();
}

template <int dim>
void TransientFixedPointData<dim>::adapt_meshes(const Vector<float> &criteria)
{
  if (!param.mesh.adaptation.enable)
    return;

  switch (param.mesh.adaptation.strategy)
  {
    case Parameters::Mesh::Adaptation::Strategy::RiemannianMetric:
    {
      /**
       * For each time interval, adapt the associated mesh with mmg, using
       * the designated Riemannian metric.
       */
      this->adapt_meshes_with_mmg();
      break;
    }
    case Parameters::Mesh::Adaptation::Strategy::LocalRefinement:
    {
      /**
       * Adapt the triangulation with the routines from deal.II.
       */
      this->adapt_mesh_with_dealii_routines(criteria);
      break;
    }
    default:
      DEAL_II_ASSERT_UNREACHABLE();
  }
}

template <int dim>
void TransientFixedPointData<dim>::adapt_meshes_with_mmg()
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
    Assert(metrics_for_adaptation[i]->is_scaled, ExcInternalError());
    if (param.metric_fields[param.metrics.metric_for_adaptation]
          .gradation.enable)
      Assert(metrics_for_adaptation[i]->is_graded, ExcInternalError());

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
void TransientFixedPointData<dim>::adapt_mesh_with_dealii_routines(
  const Vector<float> &criteria)
{
  const bool verbose =
    Utilities::MPI::this_mpi_process(mpi_communicator) == 0 &&
    param.mesh.adaptation.verbosity == Parameters::Verbosity::verbose;

  if (verbose)
  {
    std::cout << std::endl;
    std::cout << "-- Adapting tree-based mesh" << std::endl;
  }

#if defined(DEAL_II_WITH_P4EST)

  // Adapt only the first interval for now
  auto       &triangulation    = *triangulations[0];
  const auto &dh               = *dof_handlers[0];
  const auto &present_solution = *present_solutions[0];
  const auto &previous_sols    = *previous_solutions[0];

  const auto &tree_amr = param.mesh.adaptation.tree_amr;

  // Mark cells for refinement and coarsening
  switch (tree_amr.refinement_strategy)
  {
    case Parameters::Mesh::Adaptation::TreeAMR::RefinementStrategy::FixedNumber:
      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_number(
        triangulation,
        criteria,
        tree_amr.fraction_to_refine,
        tree_amr.fraction_to_coarsen,
        tree_amr.max_n_cells);
      break;
    case Parameters::Mesh::Adaptation::TreeAMR::RefinementStrategy::
      FixedFraction:
      parallel::distributed::GridRefinement::refine_and_coarsen_fixed_fraction(
        triangulation,
        criteria,
        tree_amr.fraction_to_refine,
        tree_amr.fraction_to_coarsen);
      break;
    default:
      DEAL_II_ASSERT_UNREACHABLE();
  }

  // Bound refinement according to prescribed min and max allowed levels
  if (triangulation.n_levels() > tree_amr.max_level)
    for (const auto &cell :
         triangulation.active_cell_iterators_on_level(tree_amr.max_level))
      cell->clear_refine_flag();
  for (const auto &cell :
       triangulation.active_cell_iterators_on_level(tree_amr.min_level))
    cell->clear_coarsen_flag();

  // (Re-)initialize the solution transfer object, and adapt the mesh
  solution_transfer =
    std::make_unique<SolutionTransfer<dim, LA::ParVectorType>>(dh);

  // Create the vector of pointers to give the solution_transfer.
  // Each entry must point to a *ghosted* parallel vector.
  std::vector<const LA::ParVectorType *> all_in(previous_sols.size() + 1);
  all_in[0] = &present_solution;
  for (unsigned int i = 0; i < previous_sols.size(); ++i)
    all_in[i + 1] = &previous_sols[i];

  for (auto vec : all_in)
  {
    Assert(vec->size() == dh.n_dofs(), ExcInternalError());
    Assert(vec->has_ghost_elements(), ExcInternalError());
  }

  triangulation.prepare_coarsening_and_refinement();
  solution_transfer->prepare_for_coarsening_and_refinement(all_in);

  // Now actually refine the mesh
  triangulation.execute_coarsening_and_refinement();

  if (verbose)
  {
    std::cout << "-- \tNumber of cells after adaptation: "
              << triangulation.n_global_active_cells() << std::endl;
  }

#else
  AssertThrow(false, ExcDealiiNeedsP4EST());
#endif
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
