#ifndef STABILIZATION_UTILS_H
#define STABILIZATION_UTILS_H

#include <deal.II/base/derivative_form.h>
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

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
    const double                       fallback_length)
  {
    const double norm = advection_velocity.norm();
    if (norm <= std::numeric_limits<double>::epsilon())
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
  inline double
  compute_streamline_length(const Tensor<1, dim> &advection_velocity,
                            const DerivativeForm<1, dim, dim> &inverse_jacobian,
                            const Point<dim>                  &reference_point,
                            const bool                         use_quads,
                            const double       fallback_length)
  {
    const double norm = advection_velocity.norm();
    if (norm <= std::numeric_limits<double>::epsilon())
      return fallback_length;

    const Tensor<1, dim> s = advection_velocity / norm;
    const unsigned int n_geometry_shapes =
      use_quads ? GeometryInfo<dim>::vertices_per_cell : dim + 1;

    double denominator = 0.;
    for (unsigned int v = 0; v < n_geometry_shapes; ++v)
    {
      const Tensor<1, dim> reference_gradient =
        compute_reference_geometry_shape_gradient<dim>(v,
                                                       reference_point,
                                                       use_quads);
      const Tensor<1, dim> real_gradient =
        transform_reference_gradient_to_real<dim>(reference_gradient,
                                                  inverse_jacobian);
      denominator += std::abs(s * real_gradient);
    }

    return (denominator <= std::numeric_limits<double>::epsilon()) ?
             fallback_length :
             2. / denominator;
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
        4. / (dt * dt);
    const double convective =
      4. * p2 * advection_norm * advection_norm / (h_tau * h_tau);
    const double diffusive =
      144. * p4 * diffusivity * diffusivity / std::pow(h_tau, 4);
    const double denom = std::sqrt(temporal + convective + diffusive);

    return (denom <= std::numeric_limits<double>::epsilon()) ? 0. : 1. / denom;
  }

} // namespace Stabilization

#endif
