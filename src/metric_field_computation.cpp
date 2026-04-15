
#include <error_estimation/solution_recovery.h>
#include <metric_field.h>
#include <metric_tensor_tools.h>

template <int dim>
void MetricField<dim>::compute_optimal_multiscale_metric(
  const ErrorEstimation::SolutionRecovery::Base<dim> &recovery,
  const unsigned int                                  component)
{
  using MultiscaleMetric =
    typename Parameters::MetricField<dim>::MultiscaleMetric;
  const auto target_norm = param.metric_fields[index].multiscale.target_norm;

  // Compute the anisotropic measure Q
  compute_anisotropic_measure(recovery, component);

  // Add the global and local scaling coefficients
  double N = (double)param.metric_fields[index].multiscale.n_target_vertices;
  const double n = (double)dim; // space dimension

  // If this is a convergence study with anisotropic adaptation, overwrite the
  // target number of vertices by the one from the MMS parameters.
  if (param.mms_param.enable)
    N = (double)param.mms_param.n_target_vertices;

  if (mpi_rank == 0 &&
      param.metric_fields[index].verbosity == Parameters::Verbosity::verbose)
  {
    const std::string norm = MultiscaleMetric::to_string(target_norm);
    std::cout << std::endl;
    std::cout << "Computing optimal Riemannian metric..." << std::endl;
    std::cout << "Target number of mesh vertices                   : " << N
              << std::endl;
    std::cout << "Target norm for interpolation error minimization : " << norm
              << std::endl;
    std::cout << "Polynomial degree of the solution                : "
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
      (double)param.metric_fields[index].multiscale.s; // W^{s,p} norm
    const double p = (double)param.metric_fields[index].multiscale.p;
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
}

template void MetricField<2>::compute_optimal_multiscale_metric(
  const ErrorEstimation::SolutionRecovery::Base<2> &,
  const unsigned int);
template void MetricField<3>::compute_optimal_multiscale_metric(
  const ErrorEstimation::SolutionRecovery::Base<3> &,
  const unsigned int);

template <int dim>
void MetricField<dim>::compute_anisotropic_measure(
  const ErrorEstimation::SolutionRecovery::Base<dim> &recovery,
  const unsigned int                                  component)
{
  Assert(solution_polynomial_degree > 0, ExcInternalError());

  const auto metric_param = param.metric_fields[index];

  if (!metric_param.multiscale.use_analytical_derivatives)
    AssertThrow(recovery.get_highest_stored_derivative() >=
                  solution_polynomial_degree + 1,
                ExcMessage("You are trying to compute a metric field which "
                           "requires the derivatives of order p + 1 "
                           "of a scalar field of order p, but "
                           "the provided reconstruction does not store such "
                           "reconstructed derivatives. "
                           "The SolutionRecovery must be created with at least "
                           "highest_recovered_derivatives = p + 1 to store "
                           "smoothed derivatives."));

  if (solution_polynomial_degree == 1)
    compute_anisotropic_measure_P1(
      recovery.get_reconstructed_hessian(component));
  else if (solution_polynomial_degree == 2)
    compute_anisotropic_measure_P2();
  else
    compute_anisotropic_measure_Pn();
}

template void MetricField<2>::compute_anisotropic_measure(
  const ErrorEstimation::SolutionRecovery::Base<2> &,
  const unsigned int);
template void MetricField<3>::compute_anisotropic_measure(
  const ErrorEstimation::SolutionRecovery::Base<3> &,
  const unsigned int);

template <int dim>
void MetricField<dim>::compute_anisotropic_measure_P1(
  const std::vector<Tensor<2, dim>> &solution_hessians)
{
  const auto metric_param = param.metric_fields[index];

  if (metric_param.multiscale.use_analytical_derivatives)
  {
    // Get the exact hessians at all mesh vertices
    std::vector<SymmetricTensor<2, dim>> hessians(n_vertices);
    metric_param.analytical_field->hessian_list(triangulation.get_vertices(),
                                                hessians);

    // TODO: this is "embarrassingly parallel" and can be multithreaded
    for (unsigned int v = 0; v < n_vertices; ++v)
      if (owned_vertices[v])
        metrics[v] = MetricTensorTools::absolute_value(
          hessians[v],
          param.metric_fields[index].min_eigenvalue,
          param.metric_fields[index].max_eigenvalue);
  }
  else
  {
    AssertDimension(solution_hessians.size(), n_vertices);

    // Use the provided reconstructed hessian
    for (unsigned int v = 0; v < n_vertices; ++v)
      if (owned_vertices[v])
        metrics[v] = MetricTensorTools::absolute_value(
          symmetrize(solution_hessians[v]),
          param.metric_fields[index].min_eigenvalue,
          param.metric_fields[index].max_eigenvalue);
  }
}

template void MetricField<2>::compute_anisotropic_measure_P1(
  const std::vector<Tensor<2, 2>> &);
template void MetricField<3>::compute_anisotropic_measure_P1(
  const std::vector<Tensor<2, 3>> &);

template <int dim>
void MetricField<dim>::compute_anisotropic_measure_P2()
{
  // Compute with Mirebeau's analytical solution
  DEAL_II_NOT_IMPLEMENTED();
}

template void MetricField<2>::compute_anisotropic_measure_P2();
template void MetricField<3>::compute_anisotropic_measure_P2();

template <int dim>
void MetricField<dim>::compute_anisotropic_measure_Pn()
{
  // Compute with the log-simplex method
  DEAL_II_NOT_IMPLEMENTED();
}

template void MetricField<2>::compute_anisotropic_measure_Pn();
template void MetricField<3>::compute_anisotropic_measure_Pn();
