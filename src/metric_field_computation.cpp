
#include <error_estimation/solution_recovery.h>
#include <metric_field.h>
#include <metric_tensor_tools.h>
#include <time_handler.h>

template <int dim>
void MetricField<dim>::increment_anisotropic_measure(
  const TimeHandler &time_handler)
{
  AssertThrow(param->metric_fields[index].multiscale.use_analytical_derivatives,
              ExcMessage(
                "This function can only be used if the metric field is to be "
                "computed from the derivatives of an analytical scalar field. "
                "To use the derivatives from a numerical solution, use the "
                "function taking a SolutionRecovery operator instead."));

  if (time_handler.is_steady())
  {
    // Metric for steady-state adaptation.
    // Simply set this metric field to the anisotropic measure.
    // The transient fixed point data will scale the metric before adaptation.
    const bool add = false;
    set_to_or_add_anisotropic_measure(1., add);
  }
  else
  {
    // Metric for unsteady simulation, using the transient fixed point method.
    // This metric field is the metric on a subinterval of the simulation, and
    // we have to compute the tine integral of the anisotropic measure on this
    // interval. Increment the time integral using the trapeze rule.
    // If this is the first time step in this time interval, set the field to
    // dt/2. * Q, otherwise increment by dt * Q. If this is the last step on
    // this subinterval, increment by dt/2. * Q. Use is_finished() to check if
    // this is the last step, as it applies to subintervals.
    const bool first_step =
      time_handler.current_time_iteration_in_interval == 0;
    const bool last_step = time_handler.is_finished();

    // FIXME: trapeze rule only works for constant time step
    AssertThrow(!time_handler.with_adaptive_timestep,
                ExcMessage("Adjust metric time integration rule for "
                           "non-constant time step (remove trapeze rule)"));

    const bool add    = !first_step;
    double     factor = time_handler.get_current_timestep();
    if (first_step || last_step)
      factor /= 2.;

    set_to_or_add_anisotropic_measure(factor, add);
  }
}

template void
MetricField<2>::increment_anisotropic_measure(const TimeHandler &);
template void
MetricField<3>::increment_anisotropic_measure(const TimeHandler &);

template <int dim>
void MetricField<dim>::increment_anisotropic_measure(
  const ErrorEstimation::SolutionRecovery::Base<dim> &recovery,
  const TimeHandler                                  &time_handler,
  const unsigned int                                  component)
{
  if (time_handler.is_steady())
  {
    // Metric for steady-state adaptation. See comments above.
    const bool add = false;
    set_to_or_add_anisotropic_measure(1., add, &recovery, component);
  }
  else
  {
    // Metric for unsteady simulation. See comments above.
    const bool first_step =
      time_handler.current_time_iteration_in_interval == 0;
    const bool last_step = time_handler.is_finished();

    // FIXME: trapeze rule only works for constant time step
    AssertThrow(!time_handler.with_adaptive_timestep,
                ExcMessage("Adjust metric time integration rule for "
                           "non-constant time step (remove trapeze rule)"));

    const bool add    = !first_step;
    double     factor = time_handler.get_current_timestep();
    if (first_step || last_step)
      factor /= 2.;

    set_to_or_add_anisotropic_measure(factor, add, &recovery, component);
  }
}

template void MetricField<2>::increment_anisotropic_measure(
  const ErrorEstimation::SolutionRecovery::Base<2> &,
  const TimeHandler &,
  const unsigned int);
template void MetricField<3>::increment_anisotropic_measure(
  const ErrorEstimation::SolutionRecovery::Base<3> &,
  const TimeHandler &,
  const unsigned int);

template <int dim>
void MetricField<dim>::set_to_or_add_anisotropic_measure(
  const double                                        factor,
  const bool                                          add,
  const ErrorEstimation::SolutionRecovery::Base<dim> *recovery,
  const unsigned int                                  component)
{
  Assert(solution_polynomial_degree > 0, ExcInternalError());

  const auto metric_param = param->metric_fields[index];

  if (!metric_param.multiscale.use_analytical_derivatives)
  {
    Assert(recovery, ExcInternalError());
    AssertThrow(recovery->get_highest_stored_derivative() >=
                  solution_polynomial_degree + 1,
                ExcMessage("You are trying to compute a metric field which "
                           "requires the derivatives of order p + 1 "
                           "of a scalar field of order p, but "
                           "the provided reconstruction does not store such "
                           "reconstructed derivatives. "
                           "The SolutionRecovery must be created with at least "
                           "highest_recovered_derivatives = p + 1 to store "
                           "smoothed derivatives."));
  }

  if (solution_polynomial_degree == 1)
    set_to_or_add_anisotropic_measure_P1(
      metric_param.multiscale.use_analytical_derivatives ?
        std::vector<Tensor<2, dim>>() :
        recovery->get_reconstructed_hessian(component),
      factor,
      add);
  else if (solution_polynomial_degree == 2)
    set_to_or_add_anisotropic_measure_P2();
  else
    set_to_or_add_anisotropic_measure_Pn();
}

template void MetricField<2>::set_to_or_add_anisotropic_measure(
  const double,
  const bool,
  const ErrorEstimation::SolutionRecovery::Base<2> *,
  const unsigned int);
template void MetricField<3>::set_to_or_add_anisotropic_measure(
  const double,
  const bool,
  const ErrorEstimation::SolutionRecovery::Base<3> *,
  const unsigned int);

template <int dim>
void MetricField<dim>::set_to_or_add_anisotropic_measure_P1(
  const std::vector<Tensor<2, dim>> &solution_hessians,
  const double                       factor,
  const bool                         add)
{
  const auto metric_param = param->metric_fields[index];

  if (metric_param.multiscale.use_analytical_derivatives)
  {
    // Get the exact hessians at all mesh vertices
    std::vector<SymmetricTensor<2, dim>> hessians(n_vertices);
    metric_param.analytical_field->hessian_list(triangulation->get_vertices(),
                                                hessians);

    // TODO: this is "embarrassingly parallel" and can be multithreaded
    if (add)
    {
      for (unsigned int v = 0; v < n_vertices; ++v)
        if (owned_vertices[v])
          metrics[v] +=
            factor * MetricTensorTools::anisotropic_measure_P1(metric_param,
                                                               hessians[v]);
    }
    else
    {
      for (unsigned int v = 0; v < n_vertices; ++v)
        if (owned_vertices[v])
          metrics[v] =
            factor * MetricTensorTools::anisotropic_measure_P1(metric_param,
                                                               hessians[v]);
    }
  }
  else
  {
    AssertDimension(solution_hessians.size(), n_vertices);

    // Use the provided reconstructed hessian
    if (add)
    {
      for (unsigned int v = 0; v < n_vertices; ++v)
        if (owned_vertices[v])
          metrics[v] +=
            factor * MetricTensorTools::anisotropic_measure_P1(
                       metric_param, symmetrize(solution_hessians[v]));
    }
    else
    {
      for (unsigned int v = 0; v < n_vertices; ++v)
        if (owned_vertices[v])
          metrics[v] =
            factor * MetricTensorTools::anisotropic_measure_P1(
                       metric_param, symmetrize(solution_hessians[v]));
    }
  }
}

template void MetricField<2>::set_to_or_add_anisotropic_measure_P1(
  const std::vector<Tensor<2, 2>> &,
  const double,
  const bool);
template void MetricField<3>::set_to_or_add_anisotropic_measure_P1(
  const std::vector<Tensor<2, 3>> &,
  const double,
  const bool);

template <int dim>
void MetricField<dim>::set_to_or_add_anisotropic_measure_P2()
{
  // Compute with Mirebeau's analytical solution
  DEAL_II_NOT_IMPLEMENTED();
}

template void MetricField<2>::set_to_or_add_anisotropic_measure_P2();
template void MetricField<3>::set_to_or_add_anisotropic_measure_P2();

template <int dim>
void MetricField<dim>::set_to_or_add_anisotropic_measure_Pn()
{
  // Compute with the log-simplex method
  DEAL_II_NOT_IMPLEMENTED();
}

template void MetricField<2>::set_to_or_add_anisotropic_measure_Pn();
template void MetricField<3>::set_to_or_add_anisotropic_measure_Pn();

template <int dim>
void MetricField<dim>::apply_optimal_steady_multiscale_scaling()
{
  using MultiscaleMetric =
    typename Parameters::MetricField<dim>::MultiscaleMetric;
  const auto target_norm = param->metric_fields[index].multiscale.target_norm;

  // // Compute the anisotropic measure Q
  // compute_anisotropic_measure(recovery, component);

  // Add the global and local scaling coefficients
  double N = (double)param->metric_fields[index].multiscale.n_target_vertices;
  const double n = (double)dim; // space dimension

  // If this is a convergence study with anisotropic adaptation, overwrite the
  // target number of vertices by the one from the MMS parameters.
  if (param->mms_param.enable)
    N = (double)param->mms_param.n_target_vertices;

  if (mpi_rank == 0 &&
      param->metric_fields[index].verbosity == Parameters::Verbosity::verbose)
  {
    const std::string norm = MultiscaleMetric::to_string(target_norm);
    std::cout << std::endl;
    std::cout << "-- Computing optimal Riemannian metric..." << std::endl;
    std::cout << "\tTarget number of mesh vertices                   : " << N
              << std::endl;
    std::cout << "\tTarget norm for interpolation error minimization : " << norm
              << std::endl;
    std::cout << "\tPolynomial degree of the solution                : "
              << solution_polynomial_degree << std::endl;
  }

  // Need to update the FE solution to compute integral of determinant
  metrics_to_tensor_solution();

  double det_field;
  if (target_norm == MultiscaleMetric::TargetNorm::Linfty_norm)
  {
    // Special treatment for the Linfty norm, where the exponents are 1/2 and 0
    det_field = compute_integral_determinant(0.5);
  }
  else
  {
    const double s =
      (double)param->metric_fields[index].multiscale.s; // W^{s,p} norm
    const double p = (double)param->metric_fields[index].multiscale.p;
    const double m = (double)solution_polynomial_degree + 1;

    const double exponent_for_integral =
      (p * (m - s)) / (2. * (p * (m - s) + n));
    const double exponent_for_determinant = -1. / (p * (m - s) + n);

    det_field = compute_integral_determinant(exponent_for_integral);

    // Local scaling by (det Q) ^ -(tau / (2 * p))
    for (auto &metric : metrics)
      metric *= std::pow(determinant(metric), exponent_for_determinant);
  }

  // Global scaling (operator *= includes update of the FE solution and ghosts)
  (*this) *= std::pow(N / det_field, 2. / n);

  this->is_scaled = true;
}

template void MetricField<2>::apply_optimal_steady_multiscale_scaling();
template void MetricField<3>::apply_optimal_steady_multiscale_scaling();
