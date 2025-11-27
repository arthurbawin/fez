
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <preset_mms.h>

#include <iomanip>

namespace ManufacturedSolutions
{
  using namespace dealii;

  double S_quintic(const double x)
  {
    return 6. * pow(x, 5) - 15. * pow(x, 4) + 10 * pow(x, 3);
  }

  double dS_quintic(const double x)
  {
    return 30. * (x - 1.) * (x - 1.) * x * x;
  }

  double d2S_quintic(const double x)
  {
    return 60. * x * (2. * x * x - 3. * x + 1.);
  }

  /**
   * Radial kernel
   */
  template <int dim>
  double kernel_fun(const Point<dim> &p,
                    const Point<dim> &center,
                    const double      R0,
                    const double      R1)
  {
    const double r = (p - center).norm();
    if (r <= R0)
      return 1.;
    if (R0 < r && r <= R1)
      return 1. - S_quintic((r - R0) / (R1 - R0));
    else
      return 0.;
  }

  template <int dim>
  double dr_kernel(const Point<dim> &p,
                   const Point<dim> &center,
                   const double      R0,
                   const double      R1)
  {
    const double r = (p - center).norm();
    if (R0 < r && r <= R1)
      return -dS_quintic((r - R0) / (R1 - R0)) / (R1 - R0);
    else
      return 0.;
  }

  template <int dim>
  double d2r_kernel(const Point<dim> &p,
                    const Point<dim> &center,
                    const double      R0,
                    const double      R1)
  {
    const double r = (p - center).norm();
    if (R0 < r && r <= R1)
      return -d2S_quintic((r - R0) / (R1 - R0)) / (R1 - R0) / (R1 - R0);
    else
      return 0.;
  }

  template <int dim>
  double dxi_kernel(const Point<dim>  &p,
                    const Point<dim>  &center,
                    const double       R0,
                    const double       R1,
                    const unsigned int component)
  {
    const double r = (p - center).norm();
    if (r < 1e-14)
      return 0.;
    return dr_kernel(p, center, R0, R1) * (p[component] - center[component]) /
           r;
  }

  template <int dim>
  double d2xij_kernel(const Point<dim>  &p,
                      const Point<dim>  &center,
                      const double       R0,
                      const double       R1,
                      const unsigned int comp_i,
                      const unsigned int comp_j)
  {
    const double r = (p - center).norm();
    if (r < 1e-14)
      return 0.;

    const double delta_ci = p[comp_i] - center[comp_i];
    const double delta_cj = p[comp_j] - center[comp_j];
    const double dij      = (comp_i == comp_j) ? 1. : 0.;
    return d2r_kernel(p, center, R0, R1) * (delta_ci * delta_cj / (r * r)) +
           dr_kernel(p, center, R0, R1) *
             (dij / r - delta_ci * delta_cj / (r * r * r));
  }

  template <int dim>
  double RigidMeshPosition2<dim>::value(const Point<dim>  &p,
                                        const unsigned int component) const
  {
    return p[component] + time_function->value(p) * translation[component] *
                            kernel_fun(p, center, R0, R1);
  }

  template <int dim>
  double
  RigidMeshPosition2<dim>::time_derivative(const Point<dim>  &p,
                                           const unsigned int component) const
  {
    return time_function->time_derivative(p) * translation[component] *
           kernel_fun(p, center, R0, R1);
  }

  template <int dim>
  Tensor<1, dim>
  RigidMeshPosition2<dim>::gradient(const Point<dim>  &p,
                                    const unsigned int component) const
  {
    Tensor<1, dim> grad;
    for (unsigned int d = 0; d < dim; ++d)
      grad[d] = dxi_kernel(p, center, R0, R1, d);
    grad *= time_function->value(p) * translation[component];

    // Add delta_ij (gradient of mesh position, not displacement)
    grad[component] += 1.;
    return grad;
  }

  template <int dim>
  SymmetricTensor<2, dim>
  RigidMeshPosition2<dim>::hessian(const Point<dim>  &p,
                                   const unsigned int component) const
  {
    SymmetricTensor<2, dim> hess;
    for (unsigned int di = 0; di < dim; ++di)
      for (unsigned int dj = di; dj < dim; ++dj)
        hess[di][dj] = d2xij_kernel(p, center, R0, R1, di, dj);
    hess *= time_function->value(p) * translation[component];
    return hess;
  }

  template class RigidMeshPosition2<2>;
  template class RigidMeshPosition2<3>;

  template <int dim>
  double MovingRadialKernel<dim>::value(const Point<dim>  &p,
                                        const unsigned int component) const
  {
    const Point<dim> current_center =
      center + translation * time_function->value(p);
    return time_function->time_derivative(p) * translation[component] *
           kernel_fun(p, current_center, R0, R1);
  }

  template <int dim>
  double
  MovingRadialKernel<dim>::time_derivative(const Point<dim>  &p,
                                           const unsigned int component) const
  {
    const double     f              = time_function->value(p);
    const double     fdot           = time_function->time_derivative(p);
    const double     fddot          = time_function->time_second_derivative(p);
    const Point<dim> current_center = center + translation * f;
    const double     phi_u          = kernel_fun(p, current_center, R0, R1);
    const double     dphidr         = dr_kernel(p, current_center, R0, R1);
    const double     r              = (p - current_center).norm();
    if (r < 1e-14)
      return 0.;
    const Tensor<1, dim> drdx      = (p - current_center) / r;
    const Tensor<1, dim> dxdt      = -translation * fdot;
    const double         phi_u_dot = dphidr * drdx * dxdt;
    return translation[component] * (fddot * phi_u + fdot * phi_u_dot);
  }

  template <int dim>
  Tensor<1, dim>
  MovingRadialKernel<dim>::gradient(const Point<dim>  &p,
                                    const unsigned int component) const
  {
    const double     f              = time_function->value(p);
    const double     fdot           = time_function->time_derivative(p);
    const Point<dim> current_center = center + translation * f;
    const double     r              = (p - current_center).norm();
    if (r < 1e-14)
      return Tensor<1, dim>();
    Tensor<1, dim> grad;
    for (unsigned int d = 0; d < dim; ++d)
      grad[d] = fdot * translation[component] *
                dxi_kernel(p, current_center, R0, R1, d);
    return grad;
  }

  template <int dim>
  SymmetricTensor<2, dim>
  MovingRadialKernel<dim>::hessian(const Point<dim>  &p,
                                   const unsigned int component) const
  {
    const double     f              = time_function->value(p);
    const double     fdot           = time_function->time_derivative(p);
    const Point<dim> current_center = center + translation * f;
    const double     r              = (p - current_center).norm();
    if (r < 1e-14)
      return SymmetricTensor<2, dim>();
    const double         dphidr  = dr_kernel(p, current_center, R0, R1);
    const double         dphidrr = d2r_kernel(p, current_center, R0, R1);
    const Tensor<1, dim> xrel    = p - current_center;
    const Tensor<1, dim> drdx    = xrel / r;

    // Hess(r) = 1/r * (I - x^T*x/r^2)
    const Tensor<2, dim> drdxx =
      (unit_symmetric_tensor<dim>() - outer_product(xrel, xrel) / (r * r)) / r;

    SymmetricTensor<2, dim> hess;
    for (unsigned int di = 0; di < dim; ++di)
      for (unsigned int dj = di; dj < dim; ++dj)
        hess[di][dj] = (dphidrr * drdx[di] * drdx[dj] + dphidr * drdxx[di][dj]);
    hess *= fdot * translation[component];
    return hess;
  }

  template class MovingRadialKernel<2>;
  template class MovingRadialKernel<3>;

  template <int dim>
  double NormalRadialKernel<dim>::value(const Point<dim> &p,
                                        const unsigned int /*component*/) const
  {
    const double         ft             = time_function->value(p);
    const Point<dim>     current_center = center + translation * ft;
    const Tensor<1, dim> x_rel          = p - current_center;
    const double         r              = x_rel.norm();
    if (r < 1e-14)
      return 0.;
    // Can be a different kernel from u_MMS and/or x_MMS
    const double phi_p = kernel_fun(p, current_center, R0, R1);
    return -ft * a * translation * x_rel / r * phi_p;
  }

  template <int dim>
  Tensor<1, dim>
  NormalRadialKernel<dim>::gradient(const Point<dim> &p,
                                    const unsigned int /*component*/) const
  {
    const double         ft             = time_function->value(p);
    const Point<dim>     current_center = center + translation * ft;
    const Tensor<1, dim> x_rel          = p - current_center;
    const double         r              = x_rel.norm();
    if (r < 1e-14)
      return Tensor<1, dim>();

    const double phi    = kernel_fun(p, current_center, R0, R1);
    const double dphidr = dr_kernel(p, current_center, R0, R1);

    Tensor<1, dim> grad;
    for (unsigned int d = 0; d < dim; ++d)
      grad[d] =
        -ft * a *
        (phi / r * translation[d] + (dphidr / (r * r) - phi / (r * r * r)) *
                                      (translation * x_rel) * x_rel[d]);
    return grad;
  }

  template class NormalRadialKernel<2>;
  template class NormalRadialKernel<3>;

} // namespace ManufacturedSolutions