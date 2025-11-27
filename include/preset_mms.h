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
    RigidMeshPosition2(std::shared_ptr<ParsedFunctionSDBase<dim>> time_function,
                       const Point<dim>                          &center,
                       const double                               R0,
                       const double                               R1,
                       const Tensor<1, dim>                      &translation)
      : MMSFunction<dim>(dim)
      , time_function(time_function)
      , center(center)
      , R0(R0)
      , R1(R1)
      , translation(translation)
    {
      this->check_derivatives();
    }

  public:
    virtual void set_time(const double newtime) override
    {
      FunctionTime<double>::set_time(newtime);
      time_function->set_time(newtime);
    }
    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override;
    virtual double
    time_derivative(const Point<dim>  &p,
                    const unsigned int component = 0) const override;
    virtual Tensor<1, dim>
    gradient(const Point<dim>  &p,
             const unsigned int component = 0) const override;
    virtual SymmetricTensor<2, dim>
    hessian(const Point<dim>  &p,
            const unsigned int component = 0) const override;

  private:
    std::shared_ptr<ParsedFunctionSDBase<dim>> time_function;
    const Point<dim>                           center;
    const double                               R0;
    const double                               R1;
    const Tensor<1, dim>                       translation;
  };

  /**
   * Identical to above, but here the centers moves according to
   *  c(t) = c0 + translation * f(t)
   */
  template <int dim>
  class MovingRadialKernel : public MMSFunction<dim>
  {
  public:
    MovingRadialKernel(std::shared_ptr<ParsedFunctionSDBase<dim>> time_function,
                       const Point<dim>                          &center,
                       const double                               R0,
                       const double                               R1,
                       const Tensor<1, dim>                      &translation)
      : MMSFunction<dim>(dim)
      , time_function(time_function)
      , center(center)
      , R0(R0)
      , R1(R1)
      , translation(translation)
    {
      this->check_derivatives();
    }

  public:
    virtual void set_time(const double newtime) override
    {
      FunctionTime<double>::set_time(newtime);
      time_function->set_time(newtime);
    }
    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override;
    virtual double
    time_derivative(const Point<dim>  &p,
                    const unsigned int component = 0) const override;
    virtual Tensor<1, dim>
    gradient(const Point<dim>  &p,
             const unsigned int component = 0) const override;
    virtual SymmetricTensor<2, dim>
    hessian(const Point<dim>  &p,
            const unsigned int component = 0) const override;

  private:
    std::shared_ptr<ParsedFunctionSDBase<dim>> time_function;
    const Point<dim>                           center;
    const double                               R0;
    const double                               R1;
    const Tensor<1, dim>                       translation;
  };

  /**
   * A scalar field with prescribed dot product with normal :
   *
   * g(x,t) = f(t) * (a*dir cdot (x - x_0(t))/||x - x_0(t)||) * kernel(r(x,t)),
   *
   * where a is a scalar, dir is a (non necessarily unit) translation vector and
   * x0(t) = c_0 + dir*f(t) is the current location of the center of the kernel.
   * Denoting n := (x - x_0(t))/||x - x_0(t)||) the unit vector along x-x_0,
   * this also writes:
   *
   *        = f(t) * (a*dir cdot n) * kernel(r(x,t))
   *
   * This function is used to build a manufacture pressure for the FSI solver.
   *
   * Note: neither the time derivative or Hessian matrix are implemented for
   * this function.
   */
  template <int dim>
  class NormalRadialKernel : public MMSFunction<dim>
  {
  public:
    NormalRadialKernel(std::shared_ptr<ParsedFunctionSDBase<dim>> time_function,
                       const Point<dim>                          &center,
                       const double                               R0,
                       const double                               R1,
                       const Tensor<1, dim>                      &translation,
                       const double                               a)
      : MMSFunction<dim>(dim,
                         /*ignore_time_derivative =*/true,
                         /*ignore_hessian =*/true)
      , time_function(time_function)
      , center(center)
      , R0(R0)
      , R1(R1)
      , translation(translation)
      , a(a)
    {
      this->check_derivatives();
    }

  public:
    virtual void set_time(const double newtime) override
    {
      FunctionTime<double>::set_time(newtime);
      time_function->set_time(newtime);
    }
    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override;
    virtual double
    time_derivative(const Point<dim> & /*p*/,
                    const unsigned int /*component*/ = 0) const override
    {
      Assert(!this->ignore_time_derivative, TimeDerivativeIsIgnored());
      return 0.;
    }
    virtual Tensor<1, dim>
    gradient(const Point<dim>  &p,
             const unsigned int component = 0) const override;
    virtual SymmetricTensor<2, dim>
    hessian(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override
    {
      Assert(!this->ignore_hessian, HessianIsIgnored());
      return SymmetricTensor<2, dim>();
    }

  private:
    std::shared_ptr<ParsedFunctionSDBase<dim>> time_function;
    const Point<dim>                           center;
    const double                               R0;
    const double                               R1;
    const Tensor<1, dim>                       translation;
    const double                               a;
  };

  /**
   * A spatially constant vector field which only depends on time.
   * FIXME: Not very useful, can be easily done with simple function parsers.
   */
  template <int dim>
  class TimeDependentVector : public MMSFunction<dim>
  {
  public:
    TimeDependentVector(
      std::shared_ptr<ParsedFunctionSDBase<dim>> time_function,
      const Tensor<1, dim>                      &constant_vector)
      : MMSFunction<dim>(dim)
      , time_function(time_function)
      , constant_vector(constant_vector)
    {
      this->check_derivatives();
    }

  public:
  public:
    virtual void set_time(const double newtime) override
    {
      FunctionTime<double>::set_time(newtime);
      time_function->set_time(newtime);
    }
    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      return constant_vector[component] * time_function->value(p);
    }
    virtual double
    time_derivative(const Point<dim>  &p,
                    const unsigned int component = 0) const override
    {
      return constant_vector[component] * time_function->time_derivative(p);
    }
    virtual Tensor<1, dim>
    gradient(const Point<dim> & /*p*/,
             const unsigned int /*component*/ = 0) const override
    {
      return Tensor<1, dim>();
    }
    virtual SymmetricTensor<2, dim>
    hessian(const Point<dim> & /*p*/,
            const unsigned int /*component*/ = 0) const override
    {
      return SymmetricTensor<2, dim>();
    }

  private:
    std::shared_ptr<ParsedFunctionSDBase<dim>> time_function;
    const Tensor<1, dim>                       constant_vector;
  };


} // namespace ManufacturedSolutions

#endif