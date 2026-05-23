#ifndef ASSEMBLY_ALE_GEOMETRY_H
#define ASSEMBLY_ALE_GEOMETRY_H

#include <deal.II/base/tensor.h>

using namespace dealii;

namespace Assembly::ALE
{
  // ALE differential identities used by local Jacobian assembly.
  // G is grad(delta x) in the current configuration.
  template <int dim>
  inline double
  jacobian_trace(const Tensor<2, dim> &G)
  {
    return trace(G);
  }

  template <int dim>
  inline Tensor<1, dim>
  gradient_variation(const Tensor<1, dim> &gradient,
                     const Tensor<2, dim> &G)
  {
    return -transpose(G) * gradient;
  }

  template <int dim>
  inline Tensor<2, dim>
  vector_gradient_variation(const Tensor<2, dim> &gradient,
                            const Tensor<2, dim> &G)
  {
    return -gradient * G;
  }

  template <int dim>
  inline Tensor<1, dim>
  mesh_velocity_variation(const double          bdf_c0,
                          const Tensor<1, dim> &shape_position)
  {
    return -bdf_c0 * shape_position;
  }

  template <int dim>
  inline Tensor<1, dim>
  convective_direction_variation(const double          bdf_c0,
                                 const Tensor<1, dim> &shape_position,
                                 const Tensor<2, dim> &G,
                                 const Tensor<1, dim> &convective_velocity)
  {
    return mesh_velocity_variation(bdf_c0, shape_position) -
           G * convective_velocity;
  }

  template <int dim>
  inline double
  jacobian_weighted_value_variation(const double          value,
                                    const Tensor<2, dim> &G)
  {
    return value * jacobian_trace(G);
  }

  template <int dim>
  inline double
  gradient_inner_product_jacobian_variation(
    const Tensor<1, dim> &test_gradient,
    const Tensor<1, dim> &field_gradient,
    const Tensor<2, dim> &G)
  {
    return scalar_product(gradient_variation(test_gradient, G), field_gradient) +
           scalar_product(test_gradient,
                          gradient_variation(field_gradient, G)) +
           scalar_product(test_gradient, field_gradient) * jacobian_trace(G);
  }

  template <int dim>
  inline Tensor<2, dim>
  scalar_hessian_variation(const Tensor<2, dim> &hessian,
                           const Tensor<1, dim> &gradient,
                           const Tensor<2, dim> &G,
                           const Tensor<3, dim> &K)
  {
    Tensor<2, dim> variation;
    for (unsigned int i = 0; i < dim; ++i)
      for (unsigned int j = 0; j < dim; ++j)
        for (unsigned int a = 0; a < dim; ++a)
          variation[i][j] -= G[a][i] * hessian[a][j] +
                             G[a][j] * hessian[i][a] +
                             K[a][i][j] * gradient[a];
    return variation;
  }

  template <int dim>
  inline double
  scalar_laplacian_variation(const Tensor<2, dim> &hessian,
                             const Tensor<1, dim> &gradient,
                             const Tensor<2, dim> &G,
                             const Tensor<3, dim> &K)
  {
    return trace(scalar_hessian_variation(hessian, gradient, G, K));
  }

  template <int dim>
  inline Tensor<1, dim>
  vector_lap_plus_graddiv_variation(const Tensor<3, dim> &hessian,
                                    const Tensor<2, dim> &gradient,
                                    const Tensor<2, dim> &G,
                                    const Tensor<3, dim> &K)
  {
    Tensor<1, dim> variation;
    for (unsigned int c = 0; c < dim; ++c)
      for (unsigned int i = 0; i < dim; ++i)
        for (unsigned int j = 0; j < dim; ++j)
        {
          double dH_cij = 0.;
          for (unsigned int a = 0; a < dim; ++a)
            dH_cij -= G[a][i] * hessian[c][a][j] +
                      G[a][j] * hessian[c][i][a] +
                      K[a][i][j] * gradient[c][a];

          if (i == j)
            variation[c] += dH_cij;
          if (i == c)
            variation[j] += dH_cij;
        }
    return variation;
  }
} // namespace Assembly::ALE

#endif
