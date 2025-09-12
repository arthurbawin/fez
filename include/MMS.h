
#include <deal.II/base/point.h>

#include <cmath>
#include <vector>
#include <iomanip>

namespace ManufacturedSolution
{
  using namespace dealii;

  //
  // Base class for the time dependence of a manufactured solution
  //
  class TimeDependenceBase
  {
  public:
    TimeDependenceBase() {}

    /**
     * The value f(t)
     */
    virtual double value(const double t) const = 0;

    /**
     * The time derivative f'(t)
     */
    virtual double value_dot(const double t) const = 0;

    /**
     * The second time derivative f''(t)
     */
    virtual double value_ddot(const double t) const = 0;

    /**
     * Check value_dot by comparing with finite differences
     * FIXME: This is with centered FD, this may not work if the
     * time function is not defined for t < 0.
     */
    void check_time_derivatives(const double tol_order_1 = 1e-7,
                                const double tol_order_2 = 1e-5) const
    {
      const double h_first  = 1e-8;
      const double h_second = 1e-5;

      std::vector<double> test_points = {0., 0.1, 0.345, 1., 10.};

      auto finite_diff_first = [&](auto fun, const double &t) {
        double t_plus = t + h_first, t_minus = t - h_first;
        return (fun(t_plus) - fun(t_minus)) / (2.0 * h_first);
      };

      auto finite_diff_second = [&](auto fun, const double &t) {
        double t_plus = t + h_second, t_minus = t - h_second;
        return (fun(t_plus) - 2.0 * fun(t) + fun(t_minus)) / (h_second * h_second);
      };

      auto check_relative_error = [&](double exact, double numerical, unsigned int order, double tol)
      {
        const double err = std::abs(exact - numerical);

        // Do not check relative error for values close to zero
        if (std::abs(exact) < 1e-14 && err < tol)
          return;

        if (err < tol)
          return;

        const double relative_err = err / std::abs(numerical);
        AssertThrow(relative_err < tol,
                    ExcMessage(
                      "Time derivative check for order " + std::to_string(order) + " failed:"
                      " exact = " +
                      std::to_string(exact) +
                      ", FD = " + std::to_string(numerical) +
                      ", absolute error = " + std::to_string(err) + 
                      ", relative error = " + std::to_string(relative_err)));
      };

      for (const auto &t : test_points)
      {
        // Check f'(t)
        const double ddt_exact = value_dot(t);
        const double ddt_fd = finite_diff_first([&](auto q) { return value(q); }, t);
        check_relative_error(ddt_exact, ddt_fd, 1, tol_order_1);

        // Check f''(t)
        const double d2t_exact = value_ddot(t);
        const double d2t_fd = finite_diff_second([&](auto q) { return value(q); }, t);
        check_relative_error(d2t_exact, d2t_fd, 2, tol_order_2);
      }
    }

    /**
     * Check if `other` is the time derivative of this.
     */
    void check_dependency(const TimeDependenceBase &other,
                          const double tol = 1e-8,
                          const std::vector<double> &test_points = {0.0, 0.1, 0.5, 1.0}) const
    {
      for (double t : test_points)
      {
        double f1 = this->value_dot(t);
        double g0 = other.value(t);

        AssertThrow(std::abs(f1 - g0) < tol,
                    ExcMessage(
                      "Mismatch at t = " + std::to_string(t) +
                      " : this.f'(t) = " + std::to_string(f1) +
                      " vs other.f(t) = " + std::to_string(g0)));

        double f2 = this->value_ddot(t);
        double g1 = other.value_dot(t);

        AssertThrow(std::abs(f2 - g1) < tol,
                    ExcMessage(
                      "Mismatch at t = " + std::to_string(t) +
                      " : this.f''(t) = " + std::to_string(f1) +
                      " vs other.f'(t) = " + std::to_string(g0)));
      }
    }
  };

  /**
   *  G(t) = 1
   */
  class ConstantTimeDep : public TimeDependenceBase
  {
  public:
    ConstantTimeDep()
      : TimeDependenceBase()
    {
      this->check_time_derivatives();
    }

    double value(const double /*t*/) const override { return 1.; }
    double value_dot(const double /*t*/) const override { return 0.; }
    double value_ddot(const double /*t*/) const override { return 0.; }
  };

  /**
   *  G(t) = t^p
   */
  class PowerTimeDep : public TimeDependenceBase
  {
  public:
    const unsigned int exponent;

  public:
    PowerTimeDep(const unsigned int exponent)
      : TimeDependenceBase()
      , exponent(exponent)
    {
      if (exponent == 0)
        throw std::runtime_error("Cannot create PowerTimeDep with p = 0, use "
                                 "ConstantTimeDep instead.");
      this->check_time_derivatives();
    }
    double value(const double t) const override
    {
      return pow(t, exponent);
    }
    double value_dot(const double t) const override
    {
      return exponent * pow(t, exponent - 1);
    }
    double value_ddot(const double t) const override
    {
      if(exponent == 1)
        return 0.;
      else
        return exponent * (exponent - 1) * pow(t, exponent - 2);
    }
  };

  /**
   *  G(t) = C * sin(2. * M_PI * t)
   */
  class SineTimeDep : public TimeDependenceBase
  {
  public:
    const double C;

  public:
    SineTimeDep(const double C)
      : TimeDependenceBase()
      , C(C)
    {
      this->check_time_derivatives();
    }

    double value(const double t) const override
    {
      return C * sin(2. * M_PI * t);
    }
    double value_dot(const double t) const override
    {
      return C * 2 * M_PI * cos(2. * M_PI * t);
    }
    double value_ddot(const double t) const override
    {
      return - C * 4 * M_PI * M_PI * sin(2. * M_PI * t);
    }
  };

  /**
   *  G(t) = C * cos(2. * M_PI * t)
   */
  class CosineTimeDep : public TimeDependenceBase
  {
  public:
    const double C;

  public:
    CosineTimeDep(const double C)
      : TimeDependenceBase()
      , C(C)
    {
      this->check_time_derivatives();
    }

    double value(const double t) const override
    {
      return C * cos(2. * M_PI * t);
    }
    double value_dot(const double t) const override
    {
      return - C * 2 * M_PI * sin(2. * M_PI * t);
    }
    double value_ddot(const double t) const override
    {
      return - C * 4 * M_PI * M_PI * cos(2. * M_PI * t);
    }
  };

  /**
   * Abstract base class for a manufactured solution
   * with separable variables of the form:
   *
   *  field(x,y,z,t) = f(t) * g(x,y,z).
   *
   * Stores the time dependency f(t) and checks
   * the derivatives.
   */
  template <int dim>
  class ManufacturedSolutionBase
  {
  public:
    const TimeDependenceBase &time_function;

  public:
    /**
     * Constructor
     */
    ManufacturedSolutionBase(const TimeDependenceBase &time_function)
      : time_function(time_function)
    {}

    // Virtual destructor, making this an abstract class
    virtual ~ManufacturedSolutionBase() = 0;

  protected:
    /**
     * Checks that the derivatives implemented are consistent
     * with their finite differences approximation
     */
    virtual void check_spatial_derivatives(const double tol_order_1 = 1e-7,
                                           const double tol_order_2 = 1e-5) const = 0;
  };

  template <int dim>
  ManufacturedSolutionBase<dim>::~ManufacturedSolutionBase() = default;

  /**
   * Base class for the spatial part of a manufactured flow.
   * This class is abstract, and actual manufactured solutions
   * should be defined through a derived class, which overrides
   * the functions in u_fun, v_fun, w_fun, p_fun as well as
   * their derivatives, as functions of space only.
   *
   * The public interface returns the velocity, its gradient,
   * its laplacian, etc, and same for pressure.
   *
   * The callbacks implemented in the derived classes can be
   * checked by calling check_spatial_derivatives() at construct time.
   */
  template <int dim>
  class FlowManufacturedSolutionBase : public ManufacturedSolutionBase<dim>
  {
  public:
    FlowManufacturedSolutionBase(const TimeDependenceBase &time_function)
      : ManufacturedSolutionBase<dim>(time_function)
    {}

  public:
    // The component-th component of the manufactured velocity vector
    double velocity(const double       t,
                    const Point<dim>  &p,
                    const unsigned int component) const
    {
      const double ft = this->time_function.value(t);

      if (component == 0)
        return ft * u_fun(p);
      if (component == 1)
        return ft * v_fun(p);
      if (component == 2)
        return ft * w_fun(p);
      DEAL_II_ASSERT_UNREACHABLE();
    }

    void velocity(const double t, const Point<dim> &p, Tensor<1, dim> &u) const
    {
      for (unsigned int d = 0; d < dim; ++d)
        u[d] = velocity(t, p, d);
    }

    double velocity_time_derivative(const double       t,
                                    const Point<dim>  &p,
                                    const unsigned int component) const
    {
      const double fdot = this->time_function.value_dot(t);

      if (component == 0)
        return fdot * u_fun(p);
      if (component == 1)
        return fdot * v_fun(p);
      if (component == 2)
        return fdot * w_fun(p);
      DEAL_II_ASSERT_UNREACHABLE();
    }

    void velocity_time_derivative(const double      t,
                                  const Point<dim> &p,
                                  Tensor<1, dim>   &dudt) const
    {
      for (unsigned int d = 0; d < dim; ++d)
        dudt[d] = velocity_time_derivative(t, p, d);
    }

    //
    // Convention : gradu_ij := du_i/dx_j
    //
    void grad_velocity_ui_xj(const double      t,
                             const Point<dim> &p,
                             Tensor<2, dim>   &grad_u) const
    {
      const double ft = this->time_function.value(t);

      if constexpr (dim == 2)
      {
        grad_u[0][0] = ft * ux_fun(p);
        grad_u[0][1] = ft * uy_fun(p);
        grad_u[1][0] = ft * vx_fun(p);
        grad_u[1][1] = ft * vy_fun(p);
      }
      else
      {
        grad_u[0][0] = ft * ux_fun(p);
        grad_u[0][1] = ft * uy_fun(p);
        grad_u[0][2] = ft * uz_fun(p);
        grad_u[1][0] = ft * vx_fun(p);
        grad_u[1][1] = ft * vy_fun(p);
        grad_u[1][2] = ft * vz_fun(p);
        grad_u[2][0] = ft * wx_fun(p);
        grad_u[2][1] = ft * wy_fun(p);
        grad_u[2][2] = ft * wz_fun(p);
      }
    }

    //
    // Convention : gradu_ij := du_j/dx_i
    //
    void grad_velocity_uj_xi(const double      t,
                             const Point<dim> &p,
                             Tensor<2, dim>   &grad_u) const
    {
      grad_velocity_ui_xj(t, p, grad_u);
      grad_u = transpose(grad_u);
    }

    void grad_velocity_ui_xj_time_derivative(const double      t,
                                             const Point<dim> &p,
                                             Tensor<2, dim>   &res) const
    {
      const double fdot = this->time_function.value_dot(t);

      if constexpr (dim == 2)
      {
        res[0][0] = fdot * ux_fun(p);
        res[0][1] = fdot * uy_fun(p);
        res[1][0] = fdot * vx_fun(p);
        res[1][1] = fdot * vy_fun(p);
      }
      else
      {
        res[0][0] = fdot * ux_fun(p);
        res[0][1] = fdot * uy_fun(p);
        res[0][2] = fdot * uz_fun(p);
        res[1][0] = fdot * vx_fun(p);
        res[1][1] = fdot * vy_fun(p);
        res[1][2] = fdot * vz_fun(p);
        res[2][0] = fdot * wx_fun(p);
        res[2][1] = fdot * wy_fun(p);
        res[2][2] = fdot * wz_fun(p);
      }
    }

    void grad_velocity_uj_xi_time_derivative(const double      t,
                                             const Point<dim> &p,
                                             Tensor<2, dim>   &res) const
    {
      grad_velocity_ui_xj_time_derivative(t, p, res);
      res = transpose(res);
    }

    void laplacian_velocity(const double      t,
                            const Point<dim> &p,
                            Tensor<1, dim>   &lap_u) const
    {
      const double ft = this->time_function.value(t);

      if constexpr (dim == 2)
      {
        lap_u[0] = ft * (uxx_fun(p) + uyy_fun(p));
        lap_u[1] = ft * (vxx_fun(p) + vyy_fun(p));
      }
      else
      {
        lap_u[0] = ft * (uxx_fun(p) + uyy_fun(p) + uzz_fun(p));
        lap_u[1] = ft * (vxx_fun(p) + vyy_fun(p) + vzz_fun(p));
        lap_u[2] = ft * (wxx_fun(p) + wyy_fun(p) + wzz_fun(p));
      }
    }

    // The manufactured pressure
    double pressure(const double t, const Point<dim> &p) const
    {
      const double ft = this->time_function.value(t);
      return ft * p_fun(p);
    }

    void grad_pressure(const double      t,
                       const Point<dim> &p,
                       Tensor<1, dim>   &grad_p) const
    {
      const double ft = this->time_function.value(t);
      grad_p[0]       = ft * px_fun(p);
      grad_p[1]       = ft * py_fun(p);
      if constexpr (dim == 3)
        grad_p[2] = ft * pz_fun(p);
    }

    void newtonian_stress(const double      t,
                          const Point<dim> &p,
                          const double      mu,
                          Tensor<2, dim>   &sigma) const
    {
      sigma                 = 0;
      const double pressure = this->pressure(t, p);
      for (unsigned int d = 0; d < dim; ++d)
        sigma[d][d] = -pressure;

      Tensor<2, dim> grad_u;
      this->grad_velocity_ui_xj(t, p, grad_u);
      sigma += mu * (grad_u + transpose(grad_u));
    }

  protected:
    //
    // The spatial derivatives that each derived class must overload
    //
    virtual double u_fun(const Point<dim> &) const { return 0.; };
    virtual double ux_fun(const Point<dim> &) const { return 0.; };
    virtual double uy_fun(const Point<dim> &) const { return 0.; };
    virtual double uz_fun(const Point<dim> &) const { return 0.; };
    virtual double uxx_fun(const Point<dim> &) const { return 0.; };
    virtual double uyy_fun(const Point<dim> &) const { return 0.; };
    virtual double uzz_fun(const Point<dim> &) const { return 0.; };

    virtual double v_fun(const Point<dim> &) const { return 0.; };
    virtual double vx_fun(const Point<dim> &) const { return 0.; };
    virtual double vy_fun(const Point<dim> &) const { return 0.; };
    virtual double vz_fun(const Point<dim> &) const { return 0.; };
    virtual double vxx_fun(const Point<dim> &) const { return 0.; };
    virtual double vyy_fun(const Point<dim> &) const { return 0.; };
    virtual double vzz_fun(const Point<dim> &) const { return 0.; };

    virtual double w_fun(const Point<dim> &) const { return 0.; };
    virtual double wx_fun(const Point<dim> &) const { return 0.; };
    virtual double wy_fun(const Point<dim> &) const { return 0.; };
    virtual double wz_fun(const Point<dim> &) const { return 0.; };
    virtual double wxx_fun(const Point<dim> &) const { return 0.; };
    virtual double wyy_fun(const Point<dim> &) const { return 0.; };
    virtual double wzz_fun(const Point<dim> &) const { return 0.; };

    virtual double p_fun(const Point<dim> &) const { return 0.; };
    virtual double px_fun(const Point<dim> &) const { return 0.; };
    virtual double py_fun(const Point<dim> &) const { return 0.; };
    virtual double pz_fun(const Point<dim> &) const { return 0.; };

    virtual void
    check_spatial_derivatives(const double tol_order_1 = 1e-7,
                              const double tol_order_2 = 1e-5) const override;
  };

  template <int dim>
  class Poiseuille : public FlowManufacturedSolutionBase<dim>
  {
  public:
    const double dpdx;
    const double mu;

  public:
    Poiseuille(const TimeDependenceBase &time_function,
               const double              dpdx,
               const double              mu)
      : FlowManufacturedSolutionBase<dim>(time_function)
      , dpdx(dpdx)
      , mu(mu)
    {
      this->check_spatial_derivatives();
    }

  private:
    // velocity u(x,y)
    double u_fun(const Point<dim> &p) const override
    {
      return -dpdx / (2 * mu) * p[1] * (1. - p[1]);
    }
    double uy_fun(const Point<dim> &p) const override
    {
      return -dpdx / (2 * mu) * (1. - 2 * p[1]);
    }
    double uyy_fun(const Point<dim> &p) const override { return dpdx / mu; }

    // pressure
    double p_fun(const Point<dim> &p) const override
    {
      return -dpdx * (1. - p[0]);
    }
    double px_fun(const Point<dim> &p) const override { return dpdx; }
  };

  /**
   * This velocity is the time derivative of the divergence-free
   * mesh displacement field given by
   *
   *  chi_x = phi(t) / C * (  sin^2(pi*x) * sin(2*pi*y) )
   *  chi_y = phi(t) / C * (- sin(2*pi*x) * sin^2(pi*y) ),
   *
   * so that we can enforce u = dxdt on boundaries without
   * additional source term.
   *
   * The TimeFunction must be dphi/dt.
   *
   */
  template <int dim>
  class DisplacementTimeDerivative : public FlowManufacturedSolutionBase<dim>
  {
  public:
    DisplacementTimeDerivative(const TimeDependenceBase &dphi_dt)
      : FlowManufacturedSolutionBase<dim>(dphi_dt)
    {
      this->check_spatial_derivatives();
    }

    private:
      // velocity u(x,y)
      double u_fun(const Point<dim> &p) const override
      {
        return sin(M_PI*p[0])*sin(M_PI*p[0]) * sin(2.0*M_PI*p[1]);
      }

      double ux_fun(const Point<dim> &p) const override
      {
        return M_PI * sin(2.0*M_PI*p[0]) * sin(2.0*M_PI*p[1]);
      }

      double uy_fun(const Point<dim> &p) const override
      {
        const double sX = sin(M_PI*p[0]);
        return 2.0*M_PI * (sX*sX) * cos(2.0*M_PI*p[1]);
      }

      double uxx_fun(const Point<dim> &p) const override
      {
        return 2.0 * M_PI * M_PI * cos(2.0*M_PI*p[0]) * sin(2.0*M_PI*p[1]);
      }

      double uyy_fun(const Point<dim> &p) const override
      {
        const double sX = sin(M_PI*p[0]);
        return -4.0 * M_PI * M_PI * (sX*sX) * sin(2.0*M_PI*p[1]);
      }


      double v_fun(const Point<dim> &p) const override
      {
        return - sin(2.0*M_PI*p[0]) * sin(M_PI*p[1]) * sin(M_PI*p[1]);
      }

      double vx_fun(const Point<dim> &p) const override
      {
        const double sY = sin(M_PI*p[1]);
        return -2.0 * M_PI * cos(2.0*M_PI*p[0]) * (sY*sY);
      }

      double vy_fun(const Point<dim> &p) const override
      {
        return - M_PI * sin(2.0*M_PI*p[0]) * sin(2.0*M_PI*p[1]);
      }

      double vxx_fun(const Point<dim> &p) const override
      {
        const double sY = sin(M_PI*p[1]);
        return 4.0 * M_PI * M_PI * sin(2.0*M_PI*p[0]) * (sY*sY);
      }

      double vyy_fun(const Point<dim> &p) const override
      {
        return -2.0 * M_PI * M_PI * sin(2.0*M_PI*p[0]) * cos(2.0*M_PI*p[1]);
      }

      // pressure
      double p_fun(const Point<dim> &p) const override { return p[0] + p[1]; }
      double px_fun(const Point<dim> &) const override { return 1.; }
      double py_fun(const Point<dim> &) const override { return 1.; }
  };

  template <int dim>
  class FlowA : public FlowManufacturedSolutionBase<dim>
  {
  public:
    FlowA(const TimeDependenceBase &time_function)
      : FlowManufacturedSolutionBase<dim>(time_function)
    {
      this->check_spatial_derivatives();
    }

  private:
    // velocity u(x,y)
    double u_fun(const Point<dim> &p) const override
    {
      return -cos(M_PI * p[0]) * sin(M_PI * p[1]);
    }
    double ux_fun(const Point<dim> &p) const override
    {
      return M_PI * sin(M_PI * p[0]) * sin(M_PI * p[1]);
    }
    double uy_fun(const Point<dim> &p) const override
    {
      return -M_PI * cos(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double uz_fun(const Point<dim> &) const override { return 0.0; }
    double uxx_fun(const Point<dim> &p) const override
    {
      return M_PI * M_PI * cos(M_PI * p[0]) * sin(M_PI * p[1]);
    }
    double uyy_fun(const Point<dim> &p) const override
    {
      return M_PI * M_PI * cos(M_PI * p[0]) * sin(M_PI * p[1]);
    }
    double uzz_fun(const Point<dim> &) const override { return 0.0; }

    // velocity v(x,y)
    double v_fun(const Point<dim> &p) const override
    {
      return sin(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double vx_fun(const Point<dim> &p) const override
    {
      return M_PI * cos(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double vy_fun(const Point<dim> &p) const override
    {
      return -M_PI * sin(M_PI * p[0]) * sin(M_PI * p[1]);
    }
    double vz_fun(const Point<dim> &) const override { return 0.0; }
    double vxx_fun(const Point<dim> &p) const override
    {
      return -M_PI * M_PI * sin(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double vyy_fun(const Point<dim> &p) const override
    {
      return -M_PI * M_PI * sin(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double vzz_fun(const Point<dim> &) const override { return 0.0; }

    // velocity w (2D => 0)
    double w_fun(const Point<dim> &) const override { return 0.0; }
    double wx_fun(const Point<dim> &) const override { return 0.0; }
    double wy_fun(const Point<dim> &) const override { return 0.0; }
    double wz_fun(const Point<dim> &) const override { return 0.0; }
    double wxx_fun(const Point<dim> &) const override { return 0.0; }
    double wyy_fun(const Point<dim> &) const override { return 0.0; }
    double wzz_fun(const Point<dim> &) const override { return 0.0; }

    // pressure
    double p_fun(const Point<dim> &p) const override
    {
      return cos(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double px_fun(const Point<dim> &p) const override
    {
      return -M_PI * sin(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double py_fun(const Point<dim> &p) const override
    {
      return -M_PI * cos(M_PI * p[0]) * sin(M_PI * p[1]);
    }
    double pz_fun(const Point<dim> &) const override { return 0.0; }
  };

  /**
   * u = y (y-1) + x
   * v = x (x-1) - y
   * p = x + y
   */
  template <int dim>
  class FlowB : public FlowManufacturedSolutionBase<dim>
  {
  public:
    FlowB(const TimeDependenceBase &time_function)
      : FlowManufacturedSolutionBase<dim>(time_function)
    {
      this->check_spatial_derivatives();
    }

  private:
    // u = y(y-1) + x
    double u_fun(const Point<dim> &p) const override
    {
      return p[1] * (p[1] - 1.0) + p[0];
    }
    double ux_fun(const Point<dim> &) const override { return 1.0; }
    double uy_fun(const Point<dim> &p) const override
    {
      return 2.0 * p[1] - 1.0;
    }
    double uyy_fun(const Point<dim> &) const override { return 2.0; }

    // v = x(x-1) - y
    double v_fun(const Point<dim> &p) const override
    {
      return p[0] * (p[0] - 1.0) - p[1];
    }
    double vx_fun(const Point<dim> &p) const override
    {
      return 2.0 * p[0] - 1.0;
    }
    double vy_fun(const Point<dim> &) const override { return -1.0; }
    double vxx_fun(const Point<dim> &) const override { return 2.0; }

    // w = 0 in 2D

    // p = x + y
    double p_fun(const Point<dim> &p) const override { return p[0] + p[1]; }
    double px_fun(const Point<dim> &) const override { return 1.0; }
    double py_fun(const Point<dim> &) const override { return 1.0; }
  };

  /**
   * Constant or linear
   * u =
   * v =
   * p =
   */
  template <int dim>
  class FlowC : public FlowManufacturedSolutionBase<dim>
  {
  public:
    FlowC(const TimeDependenceBase &time_function)
      : FlowManufacturedSolutionBase<dim>(time_function)
    {
      this->check_spatial_derivatives();
    }

  private:
    // u-component = y
    double u_fun(const Point<dim> &p) const override { return 0; }
    double uy_fun(const Point<dim> &) const override { return 0; }
    // v-component = 0
    double v_fun(const Point<dim> &p) const override { return 2 * p[0]; }
    double vx_fun(const Point<dim> &) const override { return 2; }
    // w-component = 0
    // pressure = x
    double p_fun(const Point<dim> &p) const override { return p[0] + p[1]; }
    double px_fun(const Point<dim> &) const override { return 1; }
    double py_fun(const Point<dim> &) const override { return 1; }
  };

  /**
   * u = C.
   * v = C.
   * p = C.
   */
  template <int dim>
  class FlowD : public FlowManufacturedSolutionBase<dim>
  {
  public:
    FlowD(const TimeDependenceBase &time_function)
      : FlowManufacturedSolutionBase<dim>(time_function)
    {
      this->check_spatial_derivatives();
    }

  private:
    double u_fun(const Point<dim> &) const override { return 1.; }
    double v_fun(const Point<dim> &) const override { return 1.; }
    double w_fun(const Point<dim> &) const override
    {
      return (dim == 3) ? 1. : 0.;
    }
    double p_fun(const Point<dim> &) const override { return 1.; }
  };

  /**
   * Checks that the derivatives provided match
   * the ones obtained with finite differences.
   */
  template <int dim>
  void FlowManufacturedSolutionBase<dim>::check_spatial_derivatives(
    const double tol_order_1,
                                           const double tol_order_2) const
  {
    const double h_first  = 1e-8;
    const double h_second = 1e-5;

    std::vector<Point<dim>> test_points;
    if constexpr (dim == 2)
      test_points = {{0.3, 0.7}, {0.5, 0.5}, {0.9, 0.1}};
    else if constexpr (dim == 3)
      test_points = {{0.3, 0.7, 0.2}, {0.5, 0.5, 0.5}, {0.9, 0.1, 0.4}};

    auto finite_diff = [&](auto fun, const Point<dim> &p, unsigned int d) {
      Point<dim> p_plus = p, p_minus = p;
      p_plus[d] += h_first;
      p_minus[d] -= h_first;
      return (fun(p_plus) - fun(p_minus)) / (2.0 * h_first);
    };

    auto finite_diff2 = [&](auto fun, const Point<dim> &p, unsigned int d) {
      Point<dim> p_plus = p, p_minus = p;
      p_plus[d] += h_second;
      p_minus[d] -= h_second;
      return (fun(p_plus) - 2.0 * fun(p) + fun(p_minus)) /
             (h_second * h_second);
    };

    auto check_order_1 = [&](double             exact,
                     double             numerical,
                     const std::string &name) {
      const double err = std::abs(exact - numerical);
      AssertThrow(err < tol_order_1,
                  ExcMessage("Derivative check failed for " + name +
                             ": exact = " + std::to_string(exact) +
                             ", FD = " + std::to_string(numerical) +
                             ", error = " + std::to_string(err)));
    };

    auto check_order_2 = [&](double             exact,
                     double             numerical,
                     const std::string &name) {
      const double err = std::abs(exact - numerical);
      AssertThrow(err < tol_order_2,
                  ExcMessage("Derivative check failed for " + name +
                             ": exact = " + std::to_string(exact) +
                             ", FD = " + std::to_string(numerical) +
                             ", error = " + std::to_string(err)));
    };

    for (const auto &p : test_points)
    {
      // --- u ---
      check_order_1(ux_fun(p),
            finite_diff([&](auto q) { return u_fun(q); }, p, 0),
            "ux");
      if (dim > 1)
        check_order_1(uy_fun(p),
              finite_diff([&](auto q) { return u_fun(q); }, p, 1),
              "uy");
      if (dim > 2)
        check_order_1(uz_fun(p),
              finite_diff([&](auto q) { return u_fun(q); }, p, 2),
              "uz");

      check_order_2(uxx_fun(p),
            finite_diff2([&](auto q) { return u_fun(q); }, p, 0),
            "uxx");
      if (dim > 1)
        check_order_2(uyy_fun(p),
              finite_diff2([&](auto q) { return u_fun(q); }, p, 1),
              "uyy");
      if (dim > 2)
        check_order_2(uzz_fun(p),
              finite_diff2([&](auto q) { return u_fun(q); }, p, 2),
              "uzz");

      // --- v ---
      if (dim >= 2)
      {
        check_order_1(vx_fun(p),
              finite_diff([&](auto q) { return v_fun(q); }, p, 0),
              "vx");
        check_order_1(vy_fun(p),
              finite_diff([&](auto q) { return v_fun(q); }, p, 1),
              "vy");
        if (dim > 2)
          check_order_1(vz_fun(p),
                finite_diff([&](auto q) { return v_fun(q); }, p, 2),
                "vz");

        check_order_2(vxx_fun(p),
              finite_diff2([&](auto q) { return v_fun(q); }, p, 0),
              "vxx");
        check_order_2(vyy_fun(p),
              finite_diff2([&](auto q) { return v_fun(q); }, p, 1),
              "vyy");
        if (dim > 2)
          check_order_2(vzz_fun(p),
                finite_diff2([&](auto q) { return v_fun(q); }, p, 2),
                "vzz");
      }

      // --- w ---
      if (dim == 3)
      {
        check_order_1(wx_fun(p),
              finite_diff([&](auto q) { return w_fun(q); }, p, 0),
              "wx");
        check_order_1(wy_fun(p),
              finite_diff([&](auto q) { return w_fun(q); }, p, 1),
              "wy");
        check_order_1(wz_fun(p),
              finite_diff([&](auto q) { return w_fun(q); }, p, 2),
              "wz");

        check_order_2(wxx_fun(p),
              finite_diff2([&](auto q) { return w_fun(q); }, p, 0),
              "wxx");
        check_order_2(wyy_fun(p),
              finite_diff2([&](auto q) { return w_fun(q); }, p, 1),
              "wyy");
        check_order_2(wzz_fun(p),
              finite_diff2([&](auto q) { return w_fun(q); }, p, 2),
              "wzz");
      }

      // --- p ---
      check_order_1(px_fun(p),
            finite_diff([&](auto q) { return p_fun(q); }, p, 0),
            "px");
      if (dim > 1)
        check_order_1(py_fun(p),
              finite_diff([&](auto q) { return p_fun(q); }, p, 1),
              "py");
      if (dim > 2)
        check_order_1(pz_fun(p),
              finite_diff([&](auto q) { return p_fun(q); }, p, 2),
              "pz");
    }
  }

  /**
   * Base class for the spatial part of a manufactured mesh position field:
   *
   *                    x(X,Y,Z) = (x(X,Y,Z), y(X,Y,Z), z(X,Y,Z)).
   *
   * This field represents the current position (x,y,z) of the points at initial
   * position (X,Y,Z). This class is abstract, and actual manufactured solutions
   * should be defined through a derived class, which overrides
   * the functions in x_fun, y_fun, z_fun as well as their derivatives, as
   * functions of space only.
   *
   * The public interface returns the ingredients necessary to form a source
   * term for the linear elasticity equation, namely the symmetric gradient, its
   * divergence, the strain tensor and also the field itself.
   *
   * The callbacks implemented in the derived classes can be
   * checked by calling check_spatial_derivatives() at construct time.
   */
  template <int dim>
  class MeshPositionMMSBase : public ManufacturedSolutionBase<dim>
  {
  public:
    MeshPositionMMSBase(const TimeDependenceBase &time_function)
      : ManufacturedSolutionBase<dim>(time_function)
    {}

  public:
    // The component-th component of the manufactured position vector
    double position(const double       t,
                    const Point<dim>  &p,
                    const unsigned int component) const
    {
      const double ft = this->time_function.value(t);

      if (component == 0)
        return p[0] + ft * x_fun(p);
      if (component == 1)
        return p[1] + ft * y_fun(p);
      if (component == 2)
        return p[2] + ft * z_fun(p);
      DEAL_II_ASSERT_UNREACHABLE();
    }

    void position(const double t, const Point<dim> &p, Tensor<1, dim> &x) const
    {
      for (unsigned int d = 0; d < dim; ++d)
        x[d] = position(t, p, d);
    }

    double mesh_velocity(const double       t,
                    const Point<dim>  &p,
                    const unsigned int component) const
    {
      const double dfdt = this->time_function.value_dot(t);

      if (component == 0)
        return dfdt * x_fun(p);
      if (component == 1)
        return dfdt * y_fun(p);
      if (component == 2)
        return dfdt * z_fun(p);
      DEAL_II_ASSERT_UNREACHABLE();
    }

    void mesh_velocity(const double t, const Point<dim> &p, Tensor<1, dim> &dxdt) const
    {
      for (unsigned int d = 0; d < dim; ++d)
        dxdt[d] = mesh_velocity(t, p, d);
    }

  private:
    /**
     * The position or displacement gradient only appears through
     * its symmetrized version in the strain tensor, so we pick
     * either convention for the gradient and hide its implementation.
     */
    void grad_position_ui_xj(const double      t,
                             const Point<dim> &p,
                             Tensor<2, dim>   &grad_x) const
    {
      const double ft = this->time_function.value(t);

      if constexpr (dim == 2)
      {
        grad_x[0][0] = ft * xX_fun(p);
        grad_x[0][1] = ft * xY_fun(p);
        grad_x[1][0] = ft * yX_fun(p);
        grad_x[1][1] = ft * yY_fun(p);
      }
      else
      {
        grad_x[0][0] = ft * xX_fun(p);
        grad_x[0][1] = ft * xY_fun(p);
        grad_x[0][2] = ft * xZ_fun(p);
        grad_x[1][0] = ft * yX_fun(p);
        grad_x[1][1] = ft * yY_fun(p);
        grad_x[1][2] = ft * yZ_fun(p);
        grad_x[2][0] = ft * zX_fun(p);
        grad_x[2][1] = ft * zY_fun(p);
        grad_x[2][2] = ft * zZ_fun(p);
      }
    }

  public:
    /**
     * Returns epsilon = (grad_X(x) + grad_X(x)^T) / 2.
     */
    void strain_tensor(const double      t,
                       const Point<dim> &p,
                       Tensor<2, dim>   &strain) const
    {
      Tensor<2, dim> grad_x;
      this->grad_position_ui_xj(t, p, grad_x);
      strain = 0.5 * (grad_x + transpose(grad_x));
    }

    /**
     * Returns the linear stress sigma = 2*mu*epsilon + lambda*tr(epsilon)*I
     */
    void stress_tensor(const double      t,
                       const Point<dim> &p,
                       const double      mu,
                       const double      lambda,
                       Tensor<2, dim>   &sigma) const
    {
      this->strain_tensor(t, p, sigma);
      const double tr_epsilon = trace(sigma);
      sigma *= 2. * mu;
      for (unsigned int d = 0; d < dim; ++d)
        sigma[d][d] += lambda * tr_epsilon;
    }

    /**
     * Returns the divergence of the linear stress tensor:
     *
     * div_X(sigma(x))= mu * lap_X(x) + (mu + lambda) * grad_X (div_X(x))
     */
    void divergence_stress_tensor(const double      t,
                                  const Point<dim> &p,
                                  const double      mu,
                                  const double      lambda,
                                  Tensor<1, dim>   &div_sigma) const
    {
      const double ft = this->time_function.value(t);

      if constexpr (dim == 2)
      {
        // mu * lap(x)
        div_sigma[0] = mu * (xXX_fun(p) + xYY_fun(p));
        div_sigma[1] = mu * (yXX_fun(p) + yYY_fun(p));

        // (mu + lambda) * grad(div(x))
        div_sigma[0] += (mu + lambda) * (xXX_fun(p) + yXY_fun(p));
        div_sigma[1] += (mu + lambda) * (xXY_fun(p) + yYY_fun(p));

        div_sigma[0] *= ft;
        div_sigma[1] *= ft;
      }
      else
      {
        div_sigma[0] = mu * (xXX_fun(p) + xYY_fun(p) + xZZ_fun(p));
        div_sigma[1] = mu * (yXX_fun(p) + yYY_fun(p) + yZZ_fun(p));
        div_sigma[2] = mu * (zXX_fun(p) + zYY_fun(p) + zZZ_fun(p));

        div_sigma[0] += (mu + lambda) * (xXX_fun(p) + yXY_fun(p) + zXZ_fun(p));
        div_sigma[1] += (mu + lambda) * (xXY_fun(p) + yYY_fun(p) + zYZ_fun(p));
        div_sigma[2] += (mu + lambda) * (xXZ_fun(p) + yZY_fun(p) + zZZ_fun(p));

        div_sigma[0] *= ft;
        div_sigma[1] *= ft;
        div_sigma[2] *= ft;
      }
    }

  protected:
    //
    // The spatial derivatives that each derived class must overload
    //
    virtual double x_fun(const Point<dim> &) const { return 0.; };
    virtual double xX_fun(const Point<dim> &) const { return 0.; };
    virtual double xY_fun(const Point<dim> &) const { return 0.; };
    virtual double xZ_fun(const Point<dim> &) const { return 0.; };
    virtual double xXX_fun(const Point<dim> &) const { return 0.; };
    virtual double xXY_fun(const Point<dim> &) const { return 0.; };
    virtual double xXZ_fun(const Point<dim> &) const { return 0.; };
    virtual double xYY_fun(const Point<dim> &) const { return 0.; };
    virtual double xZZ_fun(const Point<dim> &) const { return 0.; };

    virtual double y_fun(const Point<dim> &) const { return 0.; };
    virtual double yX_fun(const Point<dim> &) const { return 0.; };
    virtual double yY_fun(const Point<dim> &) const { return 0.; };
    virtual double yZ_fun(const Point<dim> &) const { return 0.; };
    virtual double yXX_fun(const Point<dim> &) const { return 0.; };
    virtual double yXY_fun(const Point<dim> &) const { return 0.; };
    virtual double yYY_fun(const Point<dim> &) const { return 0.; };
    virtual double yYZ_fun(const Point<dim> &) const { return 0.; };
    virtual double yZZ_fun(const Point<dim> &) const { return 0.; };

    virtual double z_fun(const Point<dim> &) const { return 0.; };
    virtual double zX_fun(const Point<dim> &) const { return 0.; };
    virtual double zY_fun(const Point<dim> &) const { return 0.; };
    virtual double zZ_fun(const Point<dim> &) const { return 0.; };
    virtual double zXX_fun(const Point<dim> &) const { return 0.; };
    virtual double zYY_fun(const Point<dim> &) const { return 0.; };
    virtual double zXZ_fun(const Point<dim> &) const { return 0.; };
    virtual double zYZ_fun(const Point<dim> &) const { return 0.; };
    virtual double zZZ_fun(const Point<dim> &) const { return 0.; };

    virtual void
    check_spatial_derivatives(const double tol_order_1 = 1e-7,
                                           const double tol_order_2 = 1e-5) const override;
  };

  /**
   * x = X * (X - 1.)
   * y = Y * (Y - 1.)
   */
  template <int dim>
  class QuadraticMeshPosition : public MeshPositionMMSBase<dim>
  {
  public:
    QuadraticMeshPosition(const TimeDependenceBase &time_function)
      : MeshPositionMMSBase<dim>(time_function)
    {
      this->check_spatial_derivatives();
    }

  private:
    double x_fun(const Point<dim> &p) const override
    {
      return p[0] * (p[0] - 1.0);
    };
    double xX_fun(const Point<dim> &p) const override
    {
      return 2.0 * p[0] - 1.0;
    };
    double xXX_fun(const Point<dim> &) const override { return 2.0; };

    double y_fun(const Point<dim> &p) const override
    {
      return p[1] * (p[1] - 1.0);
    };
    double yY_fun(const Point<dim> &p) const override
    {
      return 2.0 * p[1] - 1.0;
    };
    double yYY_fun(const Point<dim> &) const override { return 2.0; };
  };

  /**
   * 
   */
  template <int dim>
  class DivergenceFreeMeshPosition : public MeshPositionMMSBase<dim>
  {
  public:
    DivergenceFreeMeshPosition(const TimeDependenceBase &time_function)
      : MeshPositionMMSBase<dim>(time_function)
    {
      this->check_spatial_derivatives();
    }

  private:
    double x_fun(const Point<dim> &p) const override
    {
      return sin(M_PI*p[0])*sin(M_PI*p[0]) * sin(2.0*M_PI*p[1]);
    }

    double xX_fun(const Point<dim> &p) const override
    {
      return M_PI * sin(2.0*M_PI*p[0]) * sin(2.0*M_PI*p[1]);
    }

    double xY_fun(const Point<dim> &p) const override
    {
      const double sX = sin(M_PI*p[0]);
      return 2.0*M_PI * (sX*sX) * cos(2.0*M_PI*p[1]);
    }

    double xXX_fun(const Point<dim> &p) const override
    {
      return 2.0 * M_PI * M_PI * cos(2.0*M_PI*p[0]) * sin(2.0*M_PI*p[1]);
    }

    double xXY_fun(const Point<dim> &p) const override
    {
      return 2.0 * M_PI * M_PI * sin(2.0*M_PI*p[0]) * cos(2.0*M_PI*p[1]);
    }

    double xYY_fun(const Point<dim> &p) const override
    {
      const double sX = sin(M_PI*p[0]);
      return -4.0 * M_PI * M_PI * (sX*sX) * sin(2.0*M_PI*p[1]);
    }


    double y_fun(const Point<dim> &p) const override
    {
      return - sin(2.0*M_PI*p[0]) * sin(M_PI*p[1]) * sin(M_PI*p[1]);
    }

    double yX_fun(const Point<dim> &p) const override
    {
      const double sY = sin(M_PI*p[1]);
      return -2.0 * M_PI * cos(2.0*M_PI*p[0]) * (sY*sY);
    }

    double yY_fun(const Point<dim> &p) const override
    {
      return - M_PI * sin(2.0*M_PI*p[0]) * sin(2.0*M_PI*p[1]);
    }

    double yXX_fun(const Point<dim> &p) const override
    {
      const double sY = sin(M_PI*p[1]);
      return 4.0 * M_PI * M_PI * sin(2.0*M_PI*p[0]) * (sY*sY);
    }

    double yXY_fun(const Point<dim> &p) const override
    {
      return -2.0 * M_PI * M_PI * cos(2.0*M_PI*p[0]) * sin(2.0*M_PI*p[1]);
    }

    double yYY_fun(const Point<dim> &p) const override
    {
      return -2.0 * M_PI * M_PI * sin(2.0*M_PI*p[0]) * cos(2.0*M_PI*p[1]);
    }
  };

  /**
   * Checks that the derivatives provided match
   * the ones obtained with finite differences.
   */
  template <int dim>
  void
  MeshPositionMMSBase<dim>::check_spatial_derivatives(const double tol_order_1,
                                           const double tol_order_2) const
  {
    const double h_first  = 1e-8;
    const double h_second = 1e-5;

    std::vector<Point<dim>> test_points;
    if constexpr (dim == 2)
      test_points = {{0.3, 0.7}, {0.5, 0.5}, {0.9, 0.1}};
    else if constexpr (dim == 3)
      test_points = {{0.3, 0.7, 0.2}, {0.5, 0.5, 0.5}, {0.9, 0.1, 0.4}};

    auto finite_diff = [&](auto fun, const Point<dim> &p, unsigned int d) {
      Point<dim> p_plus = p, p_minus = p;
      p_plus[d] += h_first;
      p_minus[d] -= h_first;
      return (fun(p_plus) - fun(p_minus)) / (2.0 * h_first);
    };

    auto finite_diff2 = [&](auto fun, const Point<dim> &p, unsigned int d) {
      Point<dim> p_plus = p, p_minus = p;
      p_plus[d] += h_second;
      p_minus[d] -= h_second;
      return (fun(p_plus) - 2.0 * fun(p) + fun(p_minus)) /
             (h_second * h_second);
    };

    auto finite_diff_mixed =
      [&](auto fun, const Point<dim> &p, unsigned int d1, unsigned int d2) {
        Point<dim> p_pp = p, p_pm = p, p_mp = p, p_mm = p;

        p_pp[d1] += h_second;
        p_pp[d2] += h_second;
        p_pm[d1] += h_second;
        p_pm[d2] -= h_second;
        p_mp[d1] -= h_second;
        p_mp[d2] += h_second;
        p_mm[d1] -= h_second;
        p_mm[d2] -= h_second;

        return (fun(p_pp) - fun(p_pm) - fun(p_mp) + fun(p_mm)) /
               (4.0 * h_second * h_second);
      };


    auto check_order_1 = [&](double exact, double numerical, const std::string &name)
    {
      const double err = std::abs(exact - numerical);
      AssertThrow(err < tol_order_1,
                  ExcMessage("Derivative check failed for " + name +
                             ": exact = " + std::to_string(exact) +
                             ", FD = " + std::to_string(numerical) +
                             ", error = " + std::to_string(err)));
    };

    auto check_order_2 = [&](double exact, double numerical, const std::string &name)
    {
      const double err = std::abs(exact - numerical);
      AssertThrow(err < tol_order_2,
                  ExcMessage("Derivative check failed for " + name +
                             ": exact = " + std::to_string(exact) +
                             ", FD = " + std::to_string(numerical) +
                             ", error = " + std::to_string(err)));
    };

    for (const auto &p : test_points)
    {
      // --- x ---
      check_order_1(xX_fun(p),
            finite_diff([&](auto q) { return x_fun(q); }, p, 0),
            "xX");
      check_order_1(xY_fun(p),
            finite_diff([&](auto q) { return x_fun(q); }, p, 1),
            "xY");
      if constexpr (dim == 3)
        check_order_1(xZ_fun(p),
              finite_diff([&](auto q) { return x_fun(q); }, p, 2),
              "xZ");

      check_order_2(xXX_fun(p),
            finite_diff2([&](auto q) { return x_fun(q); }, p, 0),
            "xXX");
      check_order_2(xYY_fun(p),
            finite_diff2([&](auto q) { return x_fun(q); }, p, 1),
            "xYY");
      if constexpr (dim == 3)
        check_order_2(xZZ_fun(p),
              finite_diff2([&](auto q) { return x_fun(q); }, p, 2),
              "xZZ");

      check_order_2(xXY_fun(p),
            finite_diff_mixed([&](auto q) { return x_fun(q); }, p, 0, 1),
            "xXY");
      if constexpr (dim == 3)
        check_order_2(xXZ_fun(p),
              finite_diff_mixed([&](auto q) { return x_fun(q); }, p, 0, 2),
              "xXZ");

      // --- y ---
      check_order_1(yX_fun(p),
            finite_diff([&](auto q) { return y_fun(q); }, p, 0),
            "yX");
      check_order_1(yY_fun(p),
            finite_diff([&](auto q) { return y_fun(q); }, p, 1),
            "yY");
      if constexpr (dim == 3)
        check_order_1(yZ_fun(p),
              finite_diff([&](auto q) { return y_fun(q); }, p, 2),
              "yZ");

      check_order_2(yXX_fun(p),
            finite_diff2([&](auto q) { return y_fun(q); }, p, 0),
            "yXX");
      check_order_2(yYY_fun(p),
            finite_diff2([&](auto q) { return y_fun(q); }, p, 1),
            "yYY");
      if constexpr (dim == 3)
        check_order_2(yZZ_fun(p),
              finite_diff2([&](auto q) { return y_fun(q); }, p, 2),
              "yZZ");

      check_order_2(yXY_fun(p),
            finite_diff_mixed([&](auto q) { return y_fun(q); }, p, 0, 1),
            "yXY");
      if constexpr (dim == 3)
        check_order_2(yYZ_fun(p),
              finite_diff_mixed([&](auto q) { return y_fun(q); }, p, 1, 2),
              "yYZ");

      if constexpr (dim == 3)
      {
        // --- z ---
        check_order_1(zX_fun(p),
              finite_diff([&](auto q) { return z_fun(q); }, p, 0),
              "zX");
        check_order_1(zY_fun(p),
              finite_diff([&](auto q) { return z_fun(q); }, p, 1),
              "zY");
        check_order_1(zZ_fun(p),
              finite_diff([&](auto q) { return z_fun(q); }, p, 2),
              "zZ");

        check_order_2(zXX_fun(p),
              finite_diff2([&](auto q) { return z_fun(q); }, p, 0),
              "zXX");
        check_order_2(zYY_fun(p),
              finite_diff2([&](auto q) { return z_fun(q); }, p, 1),
              "zYY");
        check_order_2(zZZ_fun(p),
              finite_diff2([&](auto q) { return z_fun(q); }, p, 2),
              "zZZ");

        check_order_2(zXZ_fun(p),
              finite_diff_mixed([&](auto q) { return z_fun(q); }, p, 0, 2),
              "zXZ");
        check_order_2(zYZ_fun(p),
              finite_diff_mixed([&](auto q) { return z_fun(q); }, p, 1, 2),
              "zYZ");
      }
    }
  }
} // namespace ManufacturedSolution