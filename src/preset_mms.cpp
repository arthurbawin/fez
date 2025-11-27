
#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>
#include <preset_mms.h>


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
  double RigidMeshPosition2<dim>::value(const Point<dim>  &p,
                                        const unsigned int component) const
  {
    return p[component] + time_function.value(p) * translation[component] *
                            kernel_fun(p, center, R0, R1);
  }

  template <int dim>
  double
  RigidMeshPosition2<dim>::time_derivative(const Point<dim>  &p,
                                           const unsigned int component) const
  {
    return time_function.value(p) * translation[component] *
           kernel_fun(p, center, R0, R1);
  }

  template <int dim>
  Tensor<1, dim>
  RigidMeshPosition2<dim>::gradient(const Point<dim>  &p,
                                    const unsigned int component) const
  {
    Tensor<1, dim> grad;
    // for (unsigned int d = 0; d < dim; ++d)
    //   grad[d] = grad_function_object[component]->value(p, d);
    return grad;
  }

  template <int dim>
  double RigidMeshPosition2<dim>::divergence(const Point<dim> &p) const
  {
    return 0.;
  }

  template class RigidMeshPosition2<2>;
  template class RigidMeshPosition2<3>;
} // namespace ManufacturedSolutions