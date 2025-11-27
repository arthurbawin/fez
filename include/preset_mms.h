#ifndef PRESET_MMS_H
#define PRESET_MMS_H

#include <deal.II/base/function.h>
#include <parsed_function_symengine.h>

namespace ManufacturedSolutions
{
  using namespace dealii;

  /**
   * x_MMS(X, t) = X_0 + f(t) * kernel(|X - center|) * translation,
   *
   * where kernel is a C^2 bell-shaped function
   * and translation is the final translation vector.
   */
  template <int dim>
  class RigidMeshPosition2 : public MMSFunction<dim>
  {
  public:
    RigidMeshPosition2(const MMSFunction<dim>   &time_function,
                      const Point<dim>         &center,
                      const double              R0,
                      const double              R1,
                      const Tensor<1, dim>     &translation,
                      const double              spring_constant)
      : MMSFunction<dim>(dim)
      , time_function(time_function)
      , center(center)
      , R0(R0)
      , R1(R1)
      , translation(translation)
      , spring_constant(spring_constant)
    {
      // this->check_spatial_derivatives();
    }

  public:
    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override;
    virtual double time_derivative(const Point<dim>  &p,
                                   const unsigned int component = 0) const override;
    virtual Tensor<1, dim>
    gradient(const Point<dim>  &p,
             const unsigned int component = 0) const override;
    virtual double divergence(const Point<dim> &p) const override;

  private:
    const MMSFunction<dim>   &time_function;
    const Point<dim>     center;
    const double         R0;
    const double         R1;
    const Tensor<1, dim> translation;
    const double         spring_constant;
  };


} // namespace ManufacturedSolutions

#endif