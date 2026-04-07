
#include <metric_field.h>
#include <metric_tensor_tools.h>

template <int dim>
void MetricField<dim>::compute_optimal_multiscale_metric()
{
  using MultiscaleMetric =
    typename Parameters::MetricField<dim>::MultiscaleMetric;
  const auto target_norm = param.metric_fields[0].multiscale.target_norm;

  // Compute the anisotropic measure Q
  compute_anisotropic_measure();

  // Add the global and local scaling coefficients
  double       N = (double)param.metric_fields[0].multiscale.n_target_vertices;
  const double n = (double)dim; // space dimension

  // If this is a convergence study with anisotropic adaptation, overwrite the
  // target number of vertices by the one from the MMS parameters.
  if (param.mms_param.enable)
    N = (double)param.mms_param.n_target_vertices;

  if (mpi_rank == 0 &&
      param.metric_fields[0].verbosity == Parameters::Verbosity::verbose)
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

  double det_field;
  if (target_norm == MultiscaleMetric::TargetNorm::Linfty_norm)
  {
    // Special treatment for the Linfty norm, where the exponents are 1/2 and 0
    det_field = compute_integral_determinant(0.5);
  }
  else
  {
    const double s =
      (double)param.metric_fields[0].multiscale.s; // W^{s,p} norm
    const double p = (double)param.metric_fields[0].multiscale.p;
    const double m = (double)solution_polynomial_degree + 1;

    const double exponent_for_integral =
      (p * (m - s)) / (2. * (p * (m - s) + n));
    const double exponent_for_determinant = -1. / (p * (m - s) + n);

    det_field = compute_integral_determinant(exponent_for_integral);

    // Local scaling by (det Q) ^ -(tau / (2 * p))
    multiply_each_metric_by_determinant_power(exponent_for_determinant);
  }

  // Global scaling
  (*this) *= std::pow(N / det_field, 2. / n);
}

template void MetricField<2>::compute_optimal_multiscale_metric();
template void MetricField<3>::compute_optimal_multiscale_metric();

template <int dim>
void MetricField<dim>::compute_anisotropic_measure()
{
  Assert(solution_polynomial_degree > 0, ExcInternalError());

  if (solution_polynomial_degree == 1)
    compute_anisotropic_measure_P1();
  else if (solution_polynomial_degree == 2)
    compute_anisotropic_measure_P2();
  else
    compute_anisotropic_measure_Pn();
}

template void MetricField<2>::compute_anisotropic_measure();
template void MetricField<3>::compute_anisotropic_measure();

template <int dim>
void MetricField<dim>::compute_anisotropic_measure_P1()
{
  std::vector<SymmetricTensor<2, dim>> hessians(n_vertices);
  const std::vector<Point<dim>>       &vertices = triangulation.get_vertices();

  // Get the hessians at all mesh vertices
  param.metric_fields[0].analytical_field->hessian_list(vertices, hessians);

  // TODO: this is "embarrassingly parallel" and can be multithreaded
  for (unsigned int v = 0; v < n_vertices; ++v)
    if (owned_vertices[v])
      metrics[v] = MetricTensorTools::absolute_value(
        hessians[v],
        param.metric_fields[0].min_eigenvalue,
        param.metric_fields[0].max_eigenvalue);
}

template void MetricField<2>::compute_anisotropic_measure_P1();
template void MetricField<3>::compute_anisotropic_measure_P1();

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
