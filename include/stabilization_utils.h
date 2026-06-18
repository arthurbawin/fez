#ifndef STABILIZATION_UTILS_H
#define STABILIZATION_UTILS_H

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

/**
 * Pre-computation utilities for SUPG/PSPG stabilization.
 * Called exclusively from ScratchData::reinit*(). Not part of the assembly
 * layer — see assembly/stabilization_forms.h for the actual contributions.
 */
namespace Stabilization
{
  using namespace dealii;

  inline double
  direction_norm_threshold()
  {
    return std::sqrt(std::numeric_limits<double>::epsilon());
  }

  inline double
  direction_norm_threshold_from_bdf(const double bdf_c0)
  {
    return direction_norm_threshold() * std::max(1., std::abs(bdf_c0));
  }

  template <int dim>
  inline Tensor<1, dim>
  compute_reference_geometry_shape_gradient(const unsigned int vertex,
                                            const Point<dim>  &reference_point,
                                            const bool         use_quads)
  {
    Tensor<1, dim> grad;

    if (use_quads)
    {
      AssertIndexRange(vertex, GeometryInfo<dim>::vertices_per_cell);

      const Point<dim> vertex_point = GeometryInfo<dim>::unit_cell_vertex(vertex);
      for (unsigned int d = 0; d < dim; ++d)
      {
        grad[d] = vertex_point[d] > 0.5 ? 1. : -1.;
        for (unsigned int e = 0; e < dim; ++e)
          if (e != d)
            grad[d] *=
              vertex_point[e] > 0.5 ? reference_point[e] : 1. - reference_point[e];
      }
    }
    else
    {
      AssertIndexRange(vertex, dim + 1);

      if (vertex == 0)
        for (unsigned int d = 0; d < dim; ++d)
          grad[d] = -1.;
      else
        grad[vertex - 1] = 1.;
    }

    return grad;
  }

  template <int dim>
  inline Tensor<1, dim>
  transform_reference_gradient_to_real(
    const Tensor<1, dim>                    &reference_gradient,
    const DerivativeForm<1, dim, dim>       &inverse_jacobian)
  {
    Tensor<1, dim> real_gradient;
    for (unsigned int real_d = 0; real_d < dim; ++real_d)
      for (unsigned int ref_d = 0; ref_d < dim; ++ref_d)
        real_gradient[real_d] += reference_gradient[ref_d] *
                                 inverse_jacobian[ref_d][real_d];

    return real_gradient;
  }

  template <int dim>
  inline double
  compute_streamline_length_from_geometry_gradients(
    const Tensor<1, dim>              &advection_velocity,
    const std::vector<Tensor<1, dim>> &geometry_shape_gradients,
    const double                       fallback_length,
    const double direction_threshold = direction_norm_threshold())
  {
    const double norm = advection_velocity.norm();
    if (norm <= direction_threshold)
      return fallback_length;

    const Tensor<1, dim> s           = advection_velocity / norm;
    double               denominator = 0.;

    for (const Tensor<1, dim> &gradient : geometry_shape_gradients)
      denominator += std::abs(s * gradient);

    return (denominator <= std::numeric_limits<double>::epsilon()) ?
             fallback_length :
             2. / denominator;
  }

  template <int dim>
  inline void
  compute_geometry_shape_gradients(
    const DerivativeForm<1, dim, dim> &inverse_jacobian,
    const Point<dim>                  &reference_point,
    const bool                         use_quads,
    std::vector<Tensor<1, dim>>       &geometry_shape_gradients)
  {
    const unsigned int n_geometry_shapes =
      use_quads ? GeometryInfo<dim>::vertices_per_cell : dim + 1;

    geometry_shape_gradients.resize(n_geometry_shapes);
    for (unsigned int v = 0; v < n_geometry_shapes; ++v)
    {
      const Tensor<1, dim> reference_gradient =
        compute_reference_geometry_shape_gradient<dim>(v,
                                                       reference_point,
                                                       use_quads);
      geometry_shape_gradients[v] =
        transform_reference_gradient_to_real<dim>(reference_gradient,
                                                  inverse_jacobian);
    }
  }

  template <int dim>
  inline double
  compute_streamline_length(const Tensor<1, dim> &advection_velocity,
                            const DerivativeForm<1, dim, dim> &inverse_jacobian,
                            const Point<dim>                  &reference_point,
                            const bool                         use_quads,
                            const double       fallback_length,
                            const double direction_threshold =
                              direction_norm_threshold())
  {
    const double norm = advection_velocity.norm();
    if (norm <= direction_threshold)
      return fallback_length;

    std::vector<Tensor<1, dim>> geometry_shape_gradients;
    compute_geometry_shape_gradients<dim>(inverse_jacobian,
                                          reference_point,
                                          use_quads,
                                          geometry_shape_gradients);

    return compute_streamline_length_from_geometry_gradients(
      advection_velocity,
      geometry_shape_gradients,
      fallback_length,
      direction_threshold);
  }

  template <int dim>
  inline double
  compute_streamline_length_variation(
    const Tensor<1, dim>              &advection_velocity,
    const Tensor<1, dim>              &advection_variation,
    const std::vector<Tensor<1, dim>> &geometry_shape_gradients,
    const double                       h_tau,
    const Tensor<2, dim>              *geometry_variation = nullptr,
    const double direction_threshold = direction_norm_threshold())
  {
    const double norm = advection_velocity.norm();
    if (norm <= direction_threshold ||
        h_tau <= std::numeric_limits<double>::epsilon())
      return 0.;

    const Tensor<1, dim> s = advection_velocity / norm;
    Tensor<1, dim>       ds = advection_variation;
    ds -= (s * advection_variation) * s;
    ds /= norm;

    double d_denominator = 0.;
    for (const Tensor<1, dim> &gradient : geometry_shape_gradients)
    {
      const double directional_gradient = s * gradient;
      if (std::abs(directional_gradient) <=
          std::numeric_limits<double>::epsilon())
        continue;

      Tensor<1, dim> gradient_variation;
      if (geometry_variation != nullptr)
        gradient_variation = -transpose(*geometry_variation) * gradient;

      const double d_directional_gradient =
        ds * gradient + s * gradient_variation;
      d_denominator += (directional_gradient > 0. ? 1. : -1.) *
                       d_directional_gradient;
    }

    return -0.5 * h_tau * h_tau * d_denominator;
  }

  inline double compute_tau(const double dt,
                            const bool   is_steady,
                            const double advection_norm,
                            const double diffusivity,
                            const double h_tau,
                            const unsigned int polynomial_degree)
  {
    if (h_tau <= std::numeric_limits<double>::epsilon())
      return 0.;

    const double p  = polynomial_degree > 0 ? polynomial_degree : 1.;
    const double p2 = p * p;
    const double p4 = p2 * p2;

    const double temporal =
      (is_steady || dt <= std::numeric_limits<double>::epsilon()) ?
        0. :
        1. / (dt * dt);
    const double convective =
      4. * p2 * advection_norm * advection_norm / (h_tau * h_tau);
    const double diffusive =
      144. * p4 * diffusivity * diffusivity / std::pow(h_tau, 4);
    const double denom = std::sqrt(temporal + convective + diffusive);

    return (denom <= std::numeric_limits<double>::epsilon()) ? 0. : 1. / denom;
  }

  template <int dim>
  inline double
  compute_tau_variation(const double          tau,
                        const Tensor<1, dim> &advection_velocity,
                        const Tensor<1, dim> &advection_variation,
                        const double          diffusivity,
                        const double          diffusivity_variation,
                        const double          h_tau,
                        const double          h_tau_variation,
                        const unsigned int    polynomial_degree)
  {
    if (tau <= std::numeric_limits<double>::epsilon() ||
        h_tau <= std::numeric_limits<double>::epsilon())
      return 0.;

    const double p     = polynomial_degree > 0 ? polynomial_degree : 1.;
    const double alpha = 4. * p * p;
    const double beta  = 144. * p * p * p * p;

    const double h2 = h_tau * h_tau;
    const double h3 = h2 * h_tau;
    const double h4 = h2 * h2;
    const double h5 = h4 * h_tau;

    const double advection_norm_square =
      advection_velocity * advection_velocity;
    const double bracket =
      alpha * (advection_velocity * advection_variation) / h2 +
      beta * diffusivity * diffusivity_variation / h4 -
      (alpha * advection_norm_square / h3 +
       2. * beta * diffusivity * diffusivity / h5) *
        h_tau_variation;

    return -tau * tau * tau * bracket;
  }

} // namespace Stabilization

#endif
