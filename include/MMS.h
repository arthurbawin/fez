
#include <deal.II/base/point.h>

#include <cmath>
#include <vector>

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
    virtual double value(const double t) const     = 0;
    virtual double value_dot(const double t) const = 0;
  };

  /**
   *  G(t) = 1
   */
  class ConstantTimeDep : public TimeDependenceBase
  {
  public:
    ConstantTimeDep()
      : TimeDependenceBase()
    {}

    virtual double value(const double t) const override { return 1.; }
    virtual double value_dot(const double t) const override { return 0.; }
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
    }
    virtual double value(const double t) const override
    {
      return pow(t, exponent);
    }
    virtual double value_dot(const double t) const override
    {
      return exponent * pow(t, exponent - 1);
    }
  };

  /**
   *  G(t) = -sin(2. * M_PI * t)
   */
  class SineTimeDep : public TimeDependenceBase
  {
  public:
    SineTimeDep()
      : TimeDependenceBase()
    {}
    virtual double value(const double t) const override
    {
      return -sin(2. * M_PI * t);
    }
    virtual double value_dot(const double t) const override
    {
      return -2 * M_PI * cos(2. * M_PI * t);
    }
  };

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
   * checked by calling check_derivatives() at construct time.
   */
  template <int dim>
  class FlowManufacturedSolutionBase
  {
  public:
    const TimeDependenceBase &time_function;

  public:
    FlowManufacturedSolutionBase(const TimeDependenceBase &time_function)
      : time_function(time_function)
    {}

    // Virtual desctructor, making this an abstract class
    virtual ~FlowManufacturedSolutionBase() = 0;

  public:
    // The component-th component of the manufactured velocity vector
    double velocity(const double       t,
                    const Point<dim>  &p,
                    const unsigned int component) const
    {
      const double ft = time_function.value(t);

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
      const double fdot = time_function.value_dot(t);

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
      const double ft = time_function.value(t);

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
      const double fdot = time_function.value_dot(t);

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
      const double ft = time_function.value(t);

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
      const double ft = time_function.value(t);
      return ft * p_fun(p);
    }

    void grad_pressure(const double      t,
                       const Point<dim> &p,
                       Tensor<1, dim>   &grad_p) const
    {
      const double ft = time_function.value(t);
      grad_p[0]       = ft * px_fun(p);
      grad_p[1]       = ft * py_fun(p);
      if constexpr (dim == 3)
        grad_p[2] = ft * pz_fun(p);
    }

  protected:
    //
    // The spatial derivatives that each derived class must overload
    //
    virtual double u_fun(const Point<dim> &p) const { return 0.; };
    virtual double ux_fun(const Point<dim> &p) const { return 0.; };
    virtual double uy_fun(const Point<dim> &p) const { return 0.; };
    virtual double uz_fun(const Point<dim> &p) const { return 0.; };
    virtual double uxx_fun(const Point<dim> &p) const { return 0.; };
    virtual double uyy_fun(const Point<dim> &p) const { return 0.; };
    virtual double uzz_fun(const Point<dim> &p) const { return 0.; };

    virtual double v_fun(const Point<dim> &p) const { return 0.; };
    virtual double vx_fun(const Point<dim> &p) const { return 0.; };
    virtual double vy_fun(const Point<dim> &p) const { return 0.; };
    virtual double vz_fun(const Point<dim> &p) const { return 0.; };
    virtual double vxx_fun(const Point<dim> &p) const { return 0.; };
    virtual double vyy_fun(const Point<dim> &p) const { return 0.; };
    virtual double vzz_fun(const Point<dim> &p) const { return 0.; };

    virtual double w_fun(const Point<dim> &p) const { return 0.; };
    virtual double wx_fun(const Point<dim> &p) const { return 0.; };
    virtual double wy_fun(const Point<dim> &p) const { return 0.; };
    virtual double wz_fun(const Point<dim> &p) const { return 0.; };
    virtual double wxx_fun(const Point<dim> &p) const { return 0.; };
    virtual double wyy_fun(const Point<dim> &p) const { return 0.; };
    virtual double wzz_fun(const Point<dim> &p) const { return 0.; };

    virtual double p_fun(const Point<dim> &p) const { return 0.; };
    virtual double px_fun(const Point<dim> &p) const { return 0.; };
    virtual double py_fun(const Point<dim> &p) const { return 0.; };
    virtual double pz_fun(const Point<dim> &p) const { return 0.; };

    void check_derivatives() const;
  };

  template <int dim>
  FlowManufacturedSolutionBase<dim>::~FlowManufacturedSolutionBase() = default;

  template <int dim>
  class Poiseuille : public FlowManufacturedSolutionBase<dim>
  {
  public:
    const double dpdx;
    const double mu;
  public:
    Poiseuille(const TimeDependenceBase &time_function,
               const double dpdx,
               const double mu)
      : FlowManufacturedSolutionBase<dim>(time_function)
      , dpdx(dpdx)
      , mu(mu)
    {
      this->check_derivatives();
    }

  private:
    // velocity u(x,y)
    double u_fun(const Point<dim> &p) const override
    {
      return - dpdx / (2*mu) * p[1] * (1. - p[1]);
    }
    double uy_fun(const Point<dim> &p) const override
    {
      return - dpdx / (2*mu) * (1. - 2*p[1]);
    }
    double uyy_fun(const Point<dim> &p) const override
    {
      return dpdx / mu;
    }

    // pressure
    double p_fun(const Point<dim> &p) const override
    {
      return - dpdx * (1. - p[0]);
    }
    double px_fun(const Point<dim> &p) const override
    {
      return dpdx;
    }
  };

  template <int dim>
  class FlowA : public FlowManufacturedSolutionBase<dim>
  {
  public:
    FlowA(const TimeDependenceBase &time_function)
      : FlowManufacturedSolutionBase<dim>(time_function)
    {
      this->check_derivatives();
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
      this->check_derivatives();
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
      this->check_derivatives();
    }

  private:
    // u-component = y
    double u_fun(const Point<dim> &p) const override { return 0; }
    double uy_fun(const Point<dim> &) const override { return 0; }
    // v-component = 0
    double v_fun(const Point<dim> &p) const override { return 2*p[0]; }
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
      this->check_derivatives();
    }

  private:
    double u_fun(const Point<dim> &) const override { return 1.; }
    double v_fun(const Point<dim> &) const override { return 1.; }
    double p_fun(const Point<dim> &) const override { return 1.; }
  };

  /**
   * Checks that the derivatives provided match
   * the ones obtained with finite differences.
   */
  template <int dim>
  void FlowManufacturedSolutionBase<dim>::check_derivatives() const
  {
    const double h_first  = 1e-8;
    const double h_second = 1e-5;
    const double tol      = 5e-5;

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

    auto check = [&](double             exact,
                     double             numerical,
                     const std::string &name,
                     const Point<dim>  &p) {
      const double err = std::abs(exact - numerical);
      AssertThrow(err < tol,
                  ExcMessage("Derivative check failed for " + name +
                             ": exact = " + std::to_string(exact) +
                             ", FD = " + std::to_string(numerical) +
                             ", error = " + std::to_string(err)));
    };

    for (const auto &p : test_points)
    {
      // --- u ---
      check(ux_fun(p),
            finite_diff([&](auto q) { return u_fun(q); }, p, 0),
            "ux",
            p);
      if (dim > 1)
        check(uy_fun(p),
              finite_diff([&](auto q) { return u_fun(q); }, p, 1),
              "uy",
              p);
      if (dim > 2)
        check(uz_fun(p),
              finite_diff([&](auto q) { return u_fun(q); }, p, 2),
              "uz",
              p);

      check(uxx_fun(p),
            finite_diff2([&](auto q) { return u_fun(q); }, p, 0),
            "uxx",
            p);
      if (dim > 1)
        check(uyy_fun(p),
              finite_diff2([&](auto q) { return u_fun(q); }, p, 1),
              "uyy",
              p);
      if (dim > 2)
        check(uzz_fun(p),
              finite_diff2([&](auto q) { return u_fun(q); }, p, 2),
              "uzz",
              p);

      // --- v ---
      if (dim >= 2)
      {
        check(vx_fun(p),
              finite_diff([&](auto q) { return v_fun(q); }, p, 0),
              "vx",
              p);
        check(vy_fun(p),
              finite_diff([&](auto q) { return v_fun(q); }, p, 1),
              "vy",
              p);
        if (dim > 2)
          check(vz_fun(p),
                finite_diff([&](auto q) { return v_fun(q); }, p, 2),
                "vz",
                p);

        check(vxx_fun(p),
              finite_diff2([&](auto q) { return v_fun(q); }, p, 0),
              "vxx",
              p);
        check(vyy_fun(p),
              finite_diff2([&](auto q) { return v_fun(q); }, p, 1),
              "vyy",
              p);
        if (dim > 2)
          check(vzz_fun(p),
                finite_diff2([&](auto q) { return v_fun(q); }, p, 2),
                "vzz",
                p);
      }

      // --- w ---
      if (dim == 3)
      {
        check(wx_fun(p),
              finite_diff([&](auto q) { return w_fun(q); }, p, 0),
              "wx",
              p);
        check(wy_fun(p),
              finite_diff([&](auto q) { return w_fun(q); }, p, 1),
              "wy",
              p);
        check(wz_fun(p),
              finite_diff([&](auto q) { return w_fun(q); }, p, 2),
              "wz",
              p);

        check(wxx_fun(p),
              finite_diff2([&](auto q) { return w_fun(q); }, p, 0),
              "wxx",
              p);
        check(wyy_fun(p),
              finite_diff2([&](auto q) { return w_fun(q); }, p, 1),
              "wyy",
              p);
        check(wzz_fun(p),
              finite_diff2([&](auto q) { return w_fun(q); }, p, 2),
              "wzz",
              p);
      }

      // --- p ---
      check(px_fun(p),
            finite_diff([&](auto q) { return p_fun(q); }, p, 0),
            "px",
            p);
      if (dim > 1)
        check(py_fun(p),
              finite_diff([&](auto q) { return p_fun(q); }, p, 1),
              "py",
              p);
      if (dim > 2)
        check(pz_fun(p),
              finite_diff([&](auto q) { return p_fun(q); }, p, 2),
              "pz",
              p);
    }
  }
} // namespace ManufacturedSolution