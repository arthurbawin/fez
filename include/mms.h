
#include <deal.II/base/point.h>

#include <cmath>
#include <iomanip>
#include <vector>

namespace ManufacturedSolution
{
  using namespace dealii;

  /**
   * Functions used to define the rigid displacement MMS
   *
   */
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
                                const double tol_order_2 = 1e-4) const
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
        return (fun(t_plus) - 2.0 * fun(t) + fun(t_minus)) /
               (h_second * h_second);
      };

      auto check_relative_error =
        [&](double exact, double numerical, unsigned int order, double tol) {
          const double err = std::abs(exact - numerical);

          // Do not check relative error for values close to zero
          if (std::abs(exact) < 1e-14 && err < tol)
            return;

          if (err < tol)
            return;

          const double relative_err = err / std::abs(numerical);
          AssertThrow(
            relative_err < tol,
            ExcMessage(
              "Time derivative check for order " + std::to_string(order) +
              " failed:"
              " exact = " +
              std::to_string(exact) + ", FD = " + std::to_string(numerical) +
              ", absolute error = " + std::to_string(err) +
              ", relative error = " + std::to_string(relative_err)));
        };

      for (const auto &t : test_points)
      {
        // Check f'(t)
        const double ddt_exact = value_dot(t);
        const double ddt_fd =
          finite_diff_first([&](auto q) { return value(q); }, t);
        check_relative_error(ddt_exact, ddt_fd, 1, tol_order_1);

        // Check f''(t)
        const double d2t_exact = value_ddot(t);
        const double d2t_fd =
          finite_diff_second([&](auto q) { return value(q); }, t);
        check_relative_error(d2t_exact, d2t_fd, 2, tol_order_2);
      }
    }

    /**
     * Check if `other` is the time derivative of this.
     */
    void check_dependency(
      const TimeDependenceBase  &other,
      const double               tol         = 1e-8,
      const std::vector<double> &test_points = {0.0, 0.1, 0.5, 1.0}) const
    {
      for (double t : test_points)
      {
        double f1 = this->value_dot(t);
        double g0 = other.value(t);

        AssertThrow(std::abs(f1 - g0) < tol,
                    ExcMessage("Mismatch at t = " + std::to_string(t) +
                               " : this.f'(t) = " + std::to_string(f1) +
                               " vs other.f(t) = " + std::to_string(g0)));

        double f2 = this->value_ddot(t);
        double g1 = other.value_dot(t);

        AssertThrow(std::abs(f2 - g1) < tol,
                    ExcMessage("Mismatch at t = " + std::to_string(t) +
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
    const double C;

  public:
    ConstantTimeDep(const double C)
      : TimeDependenceBase()
      , C(C)
    {
      this->check_time_derivatives();
    }

    double value(const double /*t*/) const override { return C; }
    double value_dot(const double /*t*/) const override { return 0.; }
    double value_ddot(const double /*t*/) const override { return 0.; }
  };

  /**
   *  G(t) = C * t^p
   */
  class PowerTimeDep : public TimeDependenceBase
  {
  public:
    const double       C;
    const unsigned int exponent;

  public:
    PowerTimeDep(const double C, const unsigned int exponent)
      : TimeDependenceBase()
      , C(C)
      , exponent(exponent)
    {
      if (exponent == 0)
        throw std::runtime_error("Cannot create PowerTimeDep with p = 0, use "
                                 "ConstantTimeDep instead.");
      this->check_time_derivatives();
    }
    double value(const double t) const override { return C * pow(t, exponent); }
    double value_dot(const double t) const override
    {
      return C * exponent * pow(t, exponent - 1);
    }
    double value_ddot(const double t) const override
    {
      if (exponent == 1)
        return 0.;
      else
        return C * exponent * (exponent - 1) * pow(t, exponent - 2);
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
      return -C * 4 * M_PI * M_PI * sin(2. * M_PI * t);
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
      return -C * 2 * M_PI * sin(2. * M_PI * t);
    }
    double value_ddot(const double t) const override
    {
      return -C * 4 * M_PI * M_PI * cos(2. * M_PI * t);
    }
  };

  /**
   * DEPRECATED? Used by MeshPositionMMS.
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
    virtual void
    check_spatial_derivatives(const double tol_order_1 = 1e-7,
                              const double tol_order_2 = 1e-4) const = 0;
  };

  template <int dim>
  ManufacturedSolutionBase<dim>::~ManufacturedSolutionBase() = default;

  /**
   * Abstract base class for a velocity-pressure manufactured solution:
   *
   *  u(x,y,z,t)
   *  p(x,y,z,t)
   *
   */
  template <int dim>
  class FlowManufacturedSolutionBase
  {
  public:
    FlowManufacturedSolutionBase() {}

  public:
    // The component-th component of the manufactured velocity vector
    double velocity(const double       t,
                    const Point<dim>  &p,
                    const unsigned int component) const
    {
      if (component == 0)
        return u_fun(p, t);
      if (component == 1)
        return v_fun(p, t);
      if (component == 2)
        return w_fun(p, t);
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
      if (component == 0)
        return ut_fun(p, t);
      if (component == 1)
        return vt_fun(p, t);
      if (component == 2)
        return wt_fun(p, t);
      DEAL_II_ASSERT_UNREACHABLE();
    }

    void velocity_time_derivative(const double      t,
                                  const Point<dim> &p,
                                  Tensor<1, dim>   &dudt) const
    {
      for (unsigned int d = 0; d < dim; ++d)
        dudt[d] = velocity_time_derivative(t, p, d);
    }

    double velocity_divergence(const double t, const Point<dim> &p) const
    {
      if constexpr (dim == 2)
        return (ux_fun(p, t) + vy_fun(p, t));
      else
        return (ux_fun(p, t) + vy_fun(p, t) + wz_fun(p, t));
    }

    //
    // Convention : gradu_ij := du_i/dx_j
    //
    void grad_velocity_ui_xj(const double      t,
                             const Point<dim> &p,
                             Tensor<2, dim>   &grad_u) const
    {
      if constexpr (dim == 2)
      {
        grad_u[0][0] = ux_fun(p, t);
        grad_u[0][1] = uy_fun(p, t);
        grad_u[1][0] = vx_fun(p, t);
        grad_u[1][1] = vy_fun(p, t);
      }
      else
      {
        grad_u[0][0] = ux_fun(p, t);
        grad_u[0][1] = uy_fun(p, t);
        grad_u[0][2] = uz_fun(p, t);
        grad_u[1][0] = vx_fun(p, t);
        grad_u[1][1] = vy_fun(p, t);
        grad_u[1][2] = vz_fun(p, t);
        grad_u[2][0] = wx_fun(p, t);
        grad_u[2][1] = wy_fun(p, t);
        grad_u[2][2] = wz_fun(p, t);
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

    void laplacian_velocity(const double      t,
                            const Point<dim> &p,
                            Tensor<1, dim>   &lap_u) const
    {
      if constexpr (dim == 2)
      {
        lap_u[0] = (uxx_fun(p, t) + uyy_fun(p, t));
        lap_u[1] = (vxx_fun(p, t) + vyy_fun(p, t));
      }
      else
      {
        lap_u[0] = (uxx_fun(p, t) + uyy_fun(p, t) + uzz_fun(p, t));
        lap_u[1] = (vxx_fun(p, t) + vyy_fun(p, t) + vzz_fun(p, t));
        lap_u[2] = (wxx_fun(p, t) + wyy_fun(p, t) + wzz_fun(p, t));
      }
    }

    // The manufactured pressure
    double pressure(const double t, const Point<dim> &p) const
    {
      return p_fun(p, t);
    }

    void grad_pressure(const double      t,
                       const Point<dim> &p,
                       Tensor<1, dim>   &grad_p) const
    {
      grad_p[0] = px_fun(p, t);
      grad_p[1] = py_fun(p, t);
      if constexpr (dim == 3)
        grad_p[2] = pz_fun(p, t);
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
    // The fields and derivatives that each derived class must overload
    //
#define DECLARE_PURE_VIRTUAL_FUN(fun_name)                              \
  virtual double fun_name(const Point<dim> & /*p*/, const double /*t*/) \
    const = 0

    DECLARE_PURE_VIRTUAL_FUN(u_fun);
    DECLARE_PURE_VIRTUAL_FUN(ut_fun);
    DECLARE_PURE_VIRTUAL_FUN(ux_fun);
    DECLARE_PURE_VIRTUAL_FUN(uy_fun);
    DECLARE_PURE_VIRTUAL_FUN(uz_fun);
    DECLARE_PURE_VIRTUAL_FUN(uxx_fun);
    DECLARE_PURE_VIRTUAL_FUN(uyy_fun);
    DECLARE_PURE_VIRTUAL_FUN(uzz_fun);

    DECLARE_PURE_VIRTUAL_FUN(v_fun);
    DECLARE_PURE_VIRTUAL_FUN(vt_fun);
    DECLARE_PURE_VIRTUAL_FUN(vx_fun);
    DECLARE_PURE_VIRTUAL_FUN(vy_fun);
    DECLARE_PURE_VIRTUAL_FUN(vz_fun);
    DECLARE_PURE_VIRTUAL_FUN(vxx_fun);
    DECLARE_PURE_VIRTUAL_FUN(vyy_fun);
    DECLARE_PURE_VIRTUAL_FUN(vzz_fun);

    DECLARE_PURE_VIRTUAL_FUN(w_fun);
    DECLARE_PURE_VIRTUAL_FUN(wt_fun);
    DECLARE_PURE_VIRTUAL_FUN(wx_fun);
    DECLARE_PURE_VIRTUAL_FUN(wy_fun);
    DECLARE_PURE_VIRTUAL_FUN(wz_fun);
    DECLARE_PURE_VIRTUAL_FUN(wxx_fun);
    DECLARE_PURE_VIRTUAL_FUN(wyy_fun);
    DECLARE_PURE_VIRTUAL_FUN(wzz_fun);

    DECLARE_PURE_VIRTUAL_FUN(p_fun);
    DECLARE_PURE_VIRTUAL_FUN(px_fun);
    DECLARE_PURE_VIRTUAL_FUN(py_fun);
    DECLARE_PURE_VIRTUAL_FUN(pz_fun);

#undef DECLARE_PURE_VIRTUAL_FUN

    void check_time_derivatives(const double tol = 1e-7) const
    {
      const double h = 1e-8;

      std::vector<double> test_times = {0., 0.1, 0.345, 1., 10.};

      std::vector<Point<dim>> test_points;
      if constexpr (dim == 2)
        test_points = {{0., 0.}, {0.3, 0.7}, {0.52, 0.52}, {0.9, 0.1}};
      else if constexpr (dim == 3)
        test_points = {{0., 0., 0.},
                       {0.3, 0.7, 0.2},
                       {0.5, 0.5, 0.5},
                       {0.9, 0.1, 0.4}};

      auto finite_diff_first =
        [&](auto fun, const Point<dim> &p, const double &t) {
          double t_plus = t + h, t_minus = t - h;
          return (fun(p, t_plus) - fun(p, t_minus)) / (2.0 * h);
        };

      auto check_relative_error = [&](double      t,
                                      double      exact,
                                      double      numerical,
                                      double      tol,
                                      std::string fun_name) {
        const double err = std::abs(exact - numerical);

        if (err < tol)
        {
          // std::cout << std::scientific <<
          //   "Time derivative check for " + fun_name + " at " +
          //   std::to_string(t) + " passed:" " exact = " << exact << ", FD = "
          //   << numerical << ", absolute error = " << err << std::endl;
          return;
        }

        const double relative_err = err / std::abs(numerical);
        AssertThrow(relative_err < tol,
                    ExcMessage(
                      "Time derivative check for " + fun_name + " at " +
                      std::to_string(t) +
                      " failed:"
                      " exact = " +
                      std::to_string(exact) +
                      ", FD = " + std::to_string(numerical) +
                      ", absolute error = " + std::to_string(err) +
                      ", relative error = " + std::to_string(relative_err)));
      };

      for (const auto &t : test_times)
      {
        for (const auto &p : test_points)
        {
          // Check all dfdt(p, t)
          const double ut_fd = finite_diff_first(
            [&](const Point<dim> &pt, const double time) {
              return u_fun(pt, time);
            },
            p,
            t);
          check_relative_error(t, ut_fun(p, t), ut_fd, tol, "ut_fun");
          const double vt_fd = finite_diff_first(
            [&](const Point<dim> &pt, const double time) {
              return v_fun(pt, time);
            },
            p,
            t);
          check_relative_error(t, vt_fun(p, t), vt_fd, tol, "vt_fun");
          const double wt_fd = finite_diff_first(
            [&](const Point<dim> &pt, const double time) {
              return w_fun(pt, time);
            },
            p,
            t);
          check_relative_error(t, wt_fun(p, t), wt_fd, tol, "wt_fun");
        }
      }
    }

    /**
     * Checks that the derivatives provided match
     * the ones obtained with finite differences.
     */
    void check_spatial_derivatives(const double tol_order_1 = 1e-7,
                                   const double tol_order_2 = 1e-4) const
    {
      const double h_first  = 1e-8;
      const double h_second = 1e-5;

      std::vector<double> test_times = {0., 0.1, 0.345, 1.};

      std::vector<Point<dim>> test_points;
      if constexpr (dim == 2)
        test_points = {{0., 0.}, {0.3, 0.7}, {0.52, 0.52}, {0.9, 0.1}};
      else if constexpr (dim == 3)
        test_points = {{0., 0., 0.},
                       {0.3, 0.7, 0.2},
                       {0.5, 0.5, 0.5},
                       {0.9, 0.1, 0.4}};

      auto finite_diff =
        [&](auto fun, const Point<dim> &p, const double t, unsigned int d) {
          Point<dim> p_plus = p, p_minus = p;
          p_plus[d] += h_first;
          p_minus[d] -= h_first;
          return (fun(p_plus, t) - fun(p_minus, t)) / (2.0 * h_first);
        };

      auto finite_diff2 =
        [&](auto fun, const Point<dim> &p, const double t, unsigned int d) {
          Point<dim> p_plus = p, p_minus = p;
          p_plus[d] += h_second;
          p_minus[d] -= h_second;
          return (fun(p_plus, t) - 2.0 * fun(p, t) + fun(p_minus, t)) /
                 (h_second * h_second);
        };

      auto check_order_1 =
        [&](double exact, double numerical, const std::string &name) {
          const double err = std::abs(exact - numerical);
          if(err < tol_order_1) return;
          const double relative_err = err / std::abs(numerical);
          AssertThrow(relative_err < tol_order_1,
                      ExcMessage("Derivative check failed for " + name +
                                 ": exact = " + std::to_string(exact) +
                                 ", FD = " + std::to_string(numerical) +
                                 ", error = " + std::to_string(err)));
        };

      auto check_order_2 =
        [&](double exact, double numerical, const std::string &name) {
          const double err = std::abs(exact - numerical);
          if(err < tol_order_2) return;
          const double relative_err = err / std::abs(numerical);
          AssertThrow(relative_err < tol_order_2,
                      ExcMessage("Derivative check failed for " + name +
                                 ": exact = " + std::to_string(exact) +
                                 ", FD = " + std::to_string(numerical) +
                                 ", error = " + std::to_string(err)));
        };

      for (const auto t : test_times)
      {
        for (const auto &p : test_points)
        {
          // --- u ---
          check_order_1(
            ux_fun(p, t),
            finite_diff([&](auto q, auto tt) { return u_fun(q, tt); }, p, t, 0),
            "ux");
          if (dim > 1)
            check_order_1(uy_fun(p, t),
                          finite_diff([&](auto q,
                                          auto tt) { return u_fun(q, tt); },
                                      p,
                                      t,
                                      1),
                          "uy");
          if (dim > 2)
            check_order_1(uz_fun(p, t),
                          finite_diff([&](auto q,
                                          auto tt) { return u_fun(q, tt); },
                                      p,
                                      t,
                                      2),
                          "uz");

          check_order_2(uxx_fun(p, t),
                        finite_diff2([&](auto q,
                                         auto tt) { return u_fun(q, tt); },
                                     p,
                                     t,
                                     0),
                        "uxx");
          if (dim > 1)
            check_order_2(uyy_fun(p, t),
                          finite_diff2([&](auto q,
                                           auto tt) { return u_fun(q, tt); },
                                       p,
                                       t,
                                       1),
                          "uyy");
          if (dim > 2)
            check_order_2(uzz_fun(p, t),
                          finite_diff2([&](auto q,
                                           auto tt) { return u_fun(q, tt); },
                                       p,
                                       t,
                                       2),
                          "uzz");

          // --- v ---
          if (dim >= 2)
          {
            check_order_1(vx_fun(p, t),
                          finite_diff([&](auto q,
                                          auto tt) { return v_fun(q, tt); },
                                      p,
                                      t,
                                      0),
                          "vx");
            check_order_1(vy_fun(p, t),
                          finite_diff([&](auto q,
                                          auto tt) { return v_fun(q, tt); },
                                      p,
                                      t,
                                      1),
                          "vy");
            if (dim > 2)
              check_order_1(vz_fun(p, t),
                            finite_diff([&](auto q,
                                            auto tt) { return v_fun(q, tt); },
                                        p,
                                        t,
                                        2),
                            "vz");

            check_order_2(vxx_fun(p, t),
                          finite_diff2([&](auto q,
                                           auto tt) { return v_fun(q, tt); },
                                       p,
                                       t,
                                       0),
                          "vxx");
            check_order_2(vyy_fun(p, t),
                          finite_diff2([&](auto q,
                                           auto tt) { return v_fun(q, tt); },
                                       p,
                                       t,
                                       1),
                          "vyy");
            if (dim > 2)
              check_order_2(vzz_fun(p, t),
                            finite_diff2([&](auto q,
                                             auto tt) { return v_fun(q, tt); },
                                         p,
                                         t,
                                         2),
                            "vzz");
          }

          // --- w ---
          if (dim == 3)
          {
            check_order_1(wx_fun(p, t),
                          finite_diff([&](auto q,
                                          auto tt) { return w_fun(q, tt); },
                                      p,
                                      t,
                                      0),
                          "wx");
            check_order_1(wy_fun(p, t),
                          finite_diff([&](auto q,
                                          auto tt) { return w_fun(q, tt); },
                                      p,
                                      t,
                                      1),
                          "wy");
            check_order_1(wz_fun(p, t),
                          finite_diff([&](auto q,
                                          auto tt) { return w_fun(q, tt); },
                                      p,
                                      t,
                                      2),
                          "wz");

            check_order_2(wxx_fun(p, t),
                          finite_diff2([&](auto q,
                                           auto tt) { return w_fun(q, tt); },
                                       p,
                                       t,
                                       0),
                          "wxx");
            check_order_2(wyy_fun(p, t),
                          finite_diff2([&](auto q,
                                           auto tt) { return w_fun(q, tt); },
                                       p,
                                       t,
                                       1),
                          "wyy");
            check_order_2(wzz_fun(p, t),
                          finite_diff2([&](auto q,
                                           auto tt) { return w_fun(q, tt); },
                                       p,
                                       t,
                                       2),
                          "wzz");
          }

          // --- p ---
          check_order_1(
            px_fun(p, t),
            finite_diff([&](auto q, auto tt) { return p_fun(q, tt); }, p, t, 0),
            "px");
          if (dim > 1)
            check_order_1(py_fun(p, t),
                          finite_diff([&](auto q,
                                          auto tt) { return p_fun(q, tt); },
                                      p,
                                      t,
                                      1),
                          "py");
          if (dim > 2)
            check_order_1(pz_fun(p, t),
                          finite_diff([&](auto q,
                                          auto tt) { return p_fun(q, tt); },
                                      p,
                                      t,
                                      2),
                          "pz");
        }
      }
    }
  };

  /**
   * Particular case of a space-time separable solution:
   *
   * u = f(t) * u_space(x,y,z)
   * p = f(t) * p_space(x,y,z) with a unique f(t).
   *
   */
  template <int dim>
  class SeparableFlowMMS : public FlowManufacturedSolutionBase<dim>
  {
  public:
    const TimeDependenceBase &time_function;

  public:
    SeparableFlowMMS(const TimeDependenceBase &time_function)
      : FlowManufacturedSolutionBase<dim>()
      , time_function(time_function)
    {}

  protected:
    double value_time(const double t) const
    {
      return this->time_function.value(t);
    }
    double value_time_dot(const double t) const
    {
      return this->time_function.value_dot(t);
    }

    double
    space_time_wrapper(const Point<dim> &p,
                       double            t,
                       double (SeparableFlowMMS::*space_fun)(const Point<dim> &)
                         const,
                       double (SeparableFlowMMS::*time_fun)(double) const) const
    {
      return (this->*time_fun)(t) * (this->*space_fun)(p);
    }

    // These functions are final.
    // Only the spatial part can be overriden in derived classes.
#define DEFINE_SPACE_TIME_FUN(fun_name, spatial_fun, time_fun)              \
  double fun_name(const Point<dim> &p, const double t) const override final \
  {                                                                         \
    return space_time_wrapper(p,                                            \
                              t,                                            \
                              &SeparableFlowMMS::spatial_fun,               \
                              &SeparableFlowMMS::time_fun);                 \
  }

    DEFINE_SPACE_TIME_FUN(u_fun, u_space, value_time);
    DEFINE_SPACE_TIME_FUN(ut_fun, u_space, value_time_dot);
    DEFINE_SPACE_TIME_FUN(ux_fun, ux_space, value_time);
    DEFINE_SPACE_TIME_FUN(uy_fun, uy_space, value_time);
    DEFINE_SPACE_TIME_FUN(uz_fun, uz_space, value_time);
    DEFINE_SPACE_TIME_FUN(uxx_fun, uxx_space, value_time);
    DEFINE_SPACE_TIME_FUN(uyy_fun, uyy_space, value_time);
    DEFINE_SPACE_TIME_FUN(uzz_fun, uzz_space, value_time);

    DEFINE_SPACE_TIME_FUN(v_fun, v_space, value_time);
    DEFINE_SPACE_TIME_FUN(vt_fun, v_space, value_time_dot);
    DEFINE_SPACE_TIME_FUN(vx_fun, vx_space, value_time);
    DEFINE_SPACE_TIME_FUN(vy_fun, vy_space, value_time);
    DEFINE_SPACE_TIME_FUN(vz_fun, vz_space, value_time);
    DEFINE_SPACE_TIME_FUN(vxx_fun, vxx_space, value_time);
    DEFINE_SPACE_TIME_FUN(vyy_fun, vyy_space, value_time);
    DEFINE_SPACE_TIME_FUN(vzz_fun, vzz_space, value_time);

    DEFINE_SPACE_TIME_FUN(w_fun, w_space, value_time);
    DEFINE_SPACE_TIME_FUN(wt_fun, w_space, value_time_dot);
    DEFINE_SPACE_TIME_FUN(wx_fun, wx_space, value_time);
    DEFINE_SPACE_TIME_FUN(wy_fun, wy_space, value_time);
    DEFINE_SPACE_TIME_FUN(wz_fun, wz_space, value_time);
    DEFINE_SPACE_TIME_FUN(wxx_fun, wxx_space, value_time);
    DEFINE_SPACE_TIME_FUN(wyy_fun, wyy_space, value_time);
    DEFINE_SPACE_TIME_FUN(wzz_fun, wzz_space, value_time);

    DEFINE_SPACE_TIME_FUN(p_fun, p_space, value_time);
    DEFINE_SPACE_TIME_FUN(px_fun, px_space, value_time);
    DEFINE_SPACE_TIME_FUN(py_fun, py_space, value_time);
    DEFINE_SPACE_TIME_FUN(pz_fun, pz_space, value_time);

#undef DEFINE_SPACE_TIME_FUN

  protected:
    /**
     * The spatial parts only, with same name as space-time function.
     */
#define DEFINE_SPACE_FUN(fun_name)                        \
  virtual double fun_name(const Point<dim> & /*p*/) const \
  {                                                       \
    return 0.;                                            \
  }

    DEFINE_SPACE_FUN(u_space);
    DEFINE_SPACE_FUN(ux_space);
    DEFINE_SPACE_FUN(uy_space);
    DEFINE_SPACE_FUN(uz_space);
    DEFINE_SPACE_FUN(uxx_space);
    DEFINE_SPACE_FUN(uyy_space);
    DEFINE_SPACE_FUN(uzz_space);

    DEFINE_SPACE_FUN(v_space);
    DEFINE_SPACE_FUN(vx_space);
    DEFINE_SPACE_FUN(vy_space);
    DEFINE_SPACE_FUN(vz_space);
    DEFINE_SPACE_FUN(vxx_space);
    DEFINE_SPACE_FUN(vyy_space);
    DEFINE_SPACE_FUN(vzz_space);

    DEFINE_SPACE_FUN(w_space);
    DEFINE_SPACE_FUN(wx_space);
    DEFINE_SPACE_FUN(wy_space);
    DEFINE_SPACE_FUN(wz_space);
    DEFINE_SPACE_FUN(wxx_space);
    DEFINE_SPACE_FUN(wyy_space);
    DEFINE_SPACE_FUN(wzz_space);

    DEFINE_SPACE_FUN(p_space);
    DEFINE_SPACE_FUN(px_space);
    DEFINE_SPACE_FUN(py_space);
    DEFINE_SPACE_FUN(pz_space);

#undef DEFINE_SPACE_FUN
  };

  template <int dim>
  class NonSeparableFlowMMS : public FlowManufacturedSolutionBase<dim>
  {
  public:
    NonSeparableFlowMMS()
      : FlowManufacturedSolutionBase<dim>()
    {}

  protected:
    // These are set to return 0, but can be overriden on derived classes
#define DEFINE_NONSEPARABLE_FUN(fun_name)                                      \
  double fun_name(const Point<dim> & /*p*/, const double /*t*/) const override \
  {                                                                            \
    return 0.;                                                                 \
  }

    DEFINE_NONSEPARABLE_FUN(u_fun);
    DEFINE_NONSEPARABLE_FUN(ut_fun);
    DEFINE_NONSEPARABLE_FUN(ux_fun);
    DEFINE_NONSEPARABLE_FUN(uy_fun);
    DEFINE_NONSEPARABLE_FUN(uz_fun);
    DEFINE_NONSEPARABLE_FUN(uxx_fun);
    DEFINE_NONSEPARABLE_FUN(uyy_fun);
    DEFINE_NONSEPARABLE_FUN(uzz_fun);

    DEFINE_NONSEPARABLE_FUN(v_fun);
    DEFINE_NONSEPARABLE_FUN(vt_fun);
    DEFINE_NONSEPARABLE_FUN(vx_fun);
    DEFINE_NONSEPARABLE_FUN(vy_fun);
    DEFINE_NONSEPARABLE_FUN(vz_fun);
    DEFINE_NONSEPARABLE_FUN(vxx_fun);
    DEFINE_NONSEPARABLE_FUN(vyy_fun);
    DEFINE_NONSEPARABLE_FUN(vzz_fun);

    DEFINE_NONSEPARABLE_FUN(w_fun);
    DEFINE_NONSEPARABLE_FUN(wt_fun);
    DEFINE_NONSEPARABLE_FUN(wx_fun);
    DEFINE_NONSEPARABLE_FUN(wy_fun);
    DEFINE_NONSEPARABLE_FUN(wz_fun);
    DEFINE_NONSEPARABLE_FUN(wxx_fun);
    DEFINE_NONSEPARABLE_FUN(wyy_fun);
    DEFINE_NONSEPARABLE_FUN(wzz_fun);

    DEFINE_NONSEPARABLE_FUN(p_fun);
    DEFINE_NONSEPARABLE_FUN(px_fun);
    DEFINE_NONSEPARABLE_FUN(py_fun);
    DEFINE_NONSEPARABLE_FUN(pz_fun);

#undef DEFINE_NONSEPARABLE_FUN
  };

  template <int dim>
  class ConstantFlow : public SeparableFlowMMS<dim>
  {
  public:
    const Tensor<1, dim> &C;

  public:
    ConstantFlow(const TimeDependenceBase &time_function,
                 const Tensor<1, dim>     &C)
      : SeparableFlowMMS<dim>(time_function)
      , C(C)
    {
      this->check_time_derivatives();
      this->check_spatial_derivatives();
    }

  private:
    double u_space(const Point<dim> &) const override { return C[0]; }
    double v_space(const Point<dim> &) const override { return C[1]; }
    double p_space(const Point<dim> &) const override { return 0.; }
  };

  template <int dim>
  class Poiseuille : public SeparableFlowMMS<dim>
  {
  public:
    const double dpdx;
    const double mu;

  public:
    Poiseuille(const TimeDependenceBase &time_function,
               const double              dpdx,
               const double              mu)
      : SeparableFlowMMS<dim>(time_function)
      , dpdx(dpdx)
      , mu(mu)
    {
      this->check_time_derivatives();
      this->check_spatial_derivatives();
    }

  private:
    // velocity u(x,y)
    double u_space(const Point<dim> &p) const override
    {
      return -dpdx / (2 * mu) * p[1] * (1. - p[1]);
    }
    double uy_space(const Point<dim> &p) const override
    {
      return -dpdx / (2 * mu) * (1. - 2 * p[1]);
    }
    double uyy_space(const Point<dim> &) const override { return dpdx / mu; }

    // pressure
    double p_space(const Point<dim> &p) const override
    {
      return -dpdx * (1. - p[0]);
    }
    double px_space(const Point<dim> &) const override { return dpdx; }
  };

  /**
   * Non separable MMS:
   *
   * u_MMS(X, t) = translation * dfdt(t) * kernel(|x - x_center(t)|),
   *
   * where kernel is a C^2 bell-shaped function
   * and translation is the final (constant) translation vector.
   */
  template <int dim>
  class RigidFlow : public NonSeparableFlowMMS<dim>
  {
  public:
    const Point<dim>     center;
    const double         R0;
    const double         R1;
    const Tensor<1, dim> translation;

    /* If this is true, then the pressure is set so that
     *
     *     lambda_MMS = - sigma(u_MMS, p_MMS) dot n
     *
     * on the cylinder is compatible with the mesh position
     * boundary condition:
     *
     *    x_C = X_C + 1/k * int_Gamma lambda_MMS dx,
     *
     * while also ensuring u_MMS = d/dt(x_MMS) on the cylinder.
     *
     * That is, x_MMS, u_MMS/p_MMS and lambda_MMS are compatible
     * and describe a fully coupled fluid-structure problem,
     * without additional source terms.
     *
     * If this is false, then the pressure is arbitrary (e.g., linear).
     */
    const bool           coupled_pressure;
    const double         spring_constant;

    const TimeDependenceBase &flow_time_function;
    // Need to know where the object is
    const TimeDependenceBase &mesh_time_function;

  public:
    RigidFlow(const TimeDependenceBase &flow_time_function,
              const TimeDependenceBase &mesh_time_function,
              const Point<dim>         &center,
              const double              R0,
              const double              R1,
              const Tensor<1, dim>     &translation,
              const bool                coupled_pressure,
              const double              spring_constant)
      : NonSeparableFlowMMS<dim>()
      , center(center)
      , R0(R0)
      , R1(R1)
      , translation(translation)
      , coupled_pressure(coupled_pressure)
      , spring_constant(spring_constant)
      , flow_time_function(flow_time_function)
      , mesh_time_function(mesh_time_function)
    {
      this->check_time_derivatives();
      this->check_spatial_derivatives();
    }

  private:
    // u
    double u_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * kernel_fun(p, current_center, R0, R1) * translation[0];
    };
    double ut_fun(const Point<dim> &p, const double t) const override
    {
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      const double r = (p - current_center).norm();
      if (r < 1e-14)
        return 0.;
      const double fdot   = mesh_time_function.value_dot(t);
      const double fddot  = mesh_time_function.value_ddot(t);
      const double drdt   = -translation * (p - current_center) / r * fdot;
      const double phi    = kernel_fun(p, current_center, R0, R1);
      const double dphidr = (R0 < r && r <= R1) ?
                            -dS_quintic((r - R0) / (R1 - R0)) / (R1 - R0) :
                            0.;

      return translation[0] * (fddot * phi + fdot * dphidr * drdt);
    };
    double ux_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * dxi_kernel(p, current_center, R0, R1, 0) * translation[0];
    }
    double uy_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * dxi_kernel(p, current_center, R0, R1, 1) * translation[0];
    }
    double uz_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * dxi_kernel(p, current_center, R0, R1, 2) * translation[0];
    }
    double uxx_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * d2xij_kernel(p, current_center, R0, R1, 0, 0) *
             translation[0];
    }
    double uyy_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * d2xij_kernel(p, current_center, R0, R1, 1, 1) *
             translation[0];
    }
    double uzz_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * d2xij_kernel(p, current_center, R0, R1, 2, 2) *
             translation[0];
    }

    // y
    double v_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * kernel_fun(p, current_center, R0, R1) * translation[1];
    };
    double vt_fun(const Point<dim> &p, const double t) const override
    {
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      const double r = (p - current_center).norm();
      if (r < 1e-14)
        return 0.;
      const double fdot   = mesh_time_function.value_dot(t);
      const double fddot  = mesh_time_function.value_ddot(t);
      const double drdt   = -translation * (p - current_center) / r * fdot;
      const double phi    = kernel_fun(p, current_center, R0, R1);
      const double dphidr = (R0 < r && r <= R1) ?
                            -dS_quintic((r - R0) / (R1 - R0)) / (R1 - R0) :
                            0.;

      return translation[1] * (fddot * phi + fdot * dphidr * drdt);
    };
    double vx_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * dxi_kernel(p, current_center, R0, R1, 0) * translation[1];
    }
    double vy_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * dxi_kernel(p, current_center, R0, R1, 1) * translation[1];
    }
    double vz_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * dxi_kernel(p, current_center, R0, R1, 2) * translation[1];
    }
    double vxx_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * d2xij_kernel(p, current_center, R0, R1, 0, 0) *
             translation[1];
    }
    double vyy_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * d2xij_kernel(p, current_center, R0, R1, 1, 1) *
             translation[1];
    }
    double vzz_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      const Point<dim> current_center =
        center + translation * mesh_time_function.value(t);
      return fdot * d2xij_kernel(p, current_center, R0, R1, 2, 2) *
             translation[1];
    }

    // pressure
    double p_fun(const Point<dim> &p, const double t) const override
    {
      if(coupled_pressure)
      {
        const double R0_p = 1.;
        const double     f = mesh_time_function.value(t);
        const Tensor<1, dim> A = translation * f;
        const Point<dim> current_center = center + translation * t;
        const Tensor<1, dim> x_rel = p - current_center;
        const double r = x_rel.norm();
        if (r < 1e-14)
          return 0.;

        // Can be a different kernel from u_MMS and/or x_MMS
        const double phi_p = kernel_fun(p, current_center, R0_p, R1);

        return A * x_rel / (M_PI * R0 * r) * phi_p;
      }
      else
      {
        return p[0] + p[1];
      }
    }
    double coupled_grad_p_fun(const Point<dim> &p, const double t, const unsigned int component) const
    {
      const double R0_p = 1.;
      const double     f = mesh_time_function.value(t);
      const Tensor<1, dim> A = translation * f;
      const Point<dim> current_center = center + translation * f;
      const Tensor<1, dim> x_rel = p - current_center;
      const double r = x_rel.norm();
      if (r < 1e-14)
          return 0.;

      // Can be a different kernel from u_MMS and/or x_MMS
      const double phi    = kernel_fun(p, current_center, R0_p, R1);
      const double dphidr = dr_kernel(p, current_center, R0_p, R1);

      return (phi/r * A[component] + (dphidr/(r*r) - phi/(r*r*r)) * (A * x_rel) * x_rel[component]) / (M_PI * R0);
    }
    double px_fun(const Point<dim> &p, const double t) const override
    {
      if(coupled_pressure)
        return coupled_grad_p_fun(p, t, 0);
      else
        return 1.;
    }
    double py_fun(const Point<dim> &p, const double t) const override
    {
      if(coupled_pressure)
        return coupled_grad_p_fun(p, t, 1);
      else
        return 1.;
    }
  };

  template <int dim>
  class ConstantFlowCoupledPressure : public NonSeparableFlowMMS<dim>
  {
  public:
    const Point<dim>     center;
    const double         R0;
    const double         R1;
    const Tensor<1, dim> translation;
    const double         spring_constant;

    const TimeDependenceBase &flow_time_function;
    const TimeDependenceBase &mesh_time_function;

  public:
    ConstantFlowCoupledPressure(const TimeDependenceBase &flow_time_function,
              const TimeDependenceBase &mesh_time_function,
              const Point<dim>         &center,
              const double              R0,
              const double              R1,
              const Tensor<1, dim>     &translation,
              const double              spring_constant)
      : NonSeparableFlowMMS<dim>()
      , center(center)
      , R0(R0)
      , R1(R1)
      , translation(translation)
      , spring_constant(spring_constant)
      , flow_time_function(flow_time_function)
      , mesh_time_function(mesh_time_function)
    {
      this->check_time_derivatives();
      this->check_spatial_derivatives();
    }

  private:
    // u
    double u_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      return translation[0] * fdot;
    };
    double ut_fun(const Point<dim> &p, const double t) const override
    {
      const double fddot  = mesh_time_function.value_ddot(t);
      return translation[0] * fddot;
    };

    // y
    double v_fun(const Point<dim> &p, const double t) const override
    {
      const double     fdot = mesh_time_function.value_dot(t);
      return translation[1] * fdot;
    };
    double vt_fun(const Point<dim> &p, const double t) const override
    {
      const double fddot  = mesh_time_function.value_ddot(t);
      return translation[1] * fddot;
    };

    // // u
    // double u_fun(const Point<dim> &p, const double t) const override
    // {
    //   // return flow_time_function.value(t);
    //   return 1.;
    // };
    // double ut_fun(const Point<dim> &p, const double t) const override
    // {
    //   // return flow_time_function.value_dot(t);
    //   return 0.;
    // };

    // // y
    // double v_fun(const Point<dim> &p, const double t) const override
    // {
    //   // return flow_time_function.value(t);
    //   return 1.;
    // };
    // double vt_fun(const Point<dim> &p, const double t) const override
    // {
    //   // return flow_time_function.value_dot(t);
    //   return 0.;
    // };

    // pressure
    double p_fun(const Point<dim> &p, const double t) const override
    {
      const double R0_p = 1.;
      const double     f = mesh_time_function.value(t);
      const Tensor<1, dim> A = - translation * f;
      const Point<dim> current_center = center + translation * f;
      const Tensor<1, dim> x_rel = p - current_center;
      const double r = x_rel.norm();
      if (r < 1e-14)
        return 0.;

      // std::cout << std::setprecision(8);
      // std::cout << "Printing" << std::endl;
      // std::cout << "f      :" << f << std::endl;
      // std::cout << "d * f  :" << translation * f << std::endl;
      // std::cout << "A      :" << A << std::endl;
      // std::cout << "center :" << center << std::endl;
      // std::cout << "current:" << current_center << std::endl;
      // std::cout << "x      :" << p << std::endl;
      // std::cout << "x_rel  :" << x_rel << std::endl;
      // std::cout << "r      :" << r << std::endl;
      // std::cout << "x_rel/r:" << x_rel/r << std::endl;

      // Can be a different kernel from u_MMS and/or x_MMS
      const double phi_p = kernel_fun(p, current_center, R0_p, R1);

      return spring_constant * A * x_rel / (M_PI * R0 * r) * phi_p;
    }
    double coupled_grad_p_fun(const Point<dim> &p, const double t, const unsigned int component) const
    {
      const double R0_p = 1.;
      const double     f = mesh_time_function.value(t);
      const Tensor<1, dim> A = - translation * f;
      const Point<dim> current_center = center + translation * f;
      const Tensor<1, dim> x_rel = p - current_center;
      const double r = x_rel.norm();
      if (r < 1e-14)
          return 0.;

      // Can be a different kernel from u_MMS and/or x_MMS
      const double phi    = kernel_fun(p, current_center, R0_p, R1);
      const double dphidr = dr_kernel(p, current_center, R0_p, R1);

      return spring_constant * (phi/r * A[component] + (dphidr/(r*r) - phi/(r*r*r)) * (A * x_rel) * x_rel[component]) / (M_PI * R0);
    }
    double px_fun(const Point<dim> &p, const double t) const override
    {
      return coupled_grad_p_fun(p, t, 0);
    }
    double py_fun(const Point<dim> &p, const double t) const override
    {
      return coupled_grad_p_fun(p, t, 1);
    }
  };

  /**
   * Non divergence free flow:
   * x = X * (X - 1.)
   * y = Y * (Y - 1.)
   */
  template <int dim>
  class QuadraticFlow : public SeparableFlowMMS<dim>
  {
  public:
    QuadraticFlow(const TimeDependenceBase &time_function)
      : SeparableFlowMMS<dim>(time_function)
    {
      this->check_time_derivatives();
      this->check_spatial_derivatives();
    }

  private:
    double u_space(const Point<dim> &p) const override
    {
      return p[0] * (p[0] - 1.0);
    };
    double ux_space(const Point<dim> &p) const override
    {
      return 2.0 * p[0] - 1.0;
    };
    double uxx_space(const Point<dim> &) const override { return 2.0; };

    double v_space(const Point<dim> &p) const override
    {
      return p[1] * (p[1] - 1.0);
    };
    double vy_space(const Point<dim> &p) const override
    {
      return 2.0 * p[1] - 1.0;
    };
    double vyy_space(const Point<dim> &) const override { return 2.0; };

    double p_space(const Point<dim> &p) const override { return p[0] + p[1]; }
    double px_space(const Point<dim> &) const override { return 1.0; }
    double py_space(const Point<dim> &) const override { return 1.0; }
  };

  template <int dim>
  class FlowA : public SeparableFlowMMS<dim>
  {
  public:
    FlowA(const TimeDependenceBase &time_function)
      : SeparableFlowMMS<dim>(time_function)
    {
      this->check_spatial_derivatives();
    }

  private:
    // velocity u(x,y)
    double u_space(const Point<dim> &p) const override
    {
      return -cos(M_PI * p[0]) * sin(M_PI * p[1]);
    }
    double ux_space(const Point<dim> &p) const override
    {
      return M_PI * sin(M_PI * p[0]) * sin(M_PI * p[1]);
    }
    double uy_space(const Point<dim> &p) const override
    {
      return -M_PI * cos(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double uz_space(const Point<dim> &) const override { return 0.0; }
    double uxx_space(const Point<dim> &p) const override
    {
      return M_PI * M_PI * cos(M_PI * p[0]) * sin(M_PI * p[1]);
    }
    double uyy_space(const Point<dim> &p) const override
    {
      return M_PI * M_PI * cos(M_PI * p[0]) * sin(M_PI * p[1]);
    }
    double uzz_space(const Point<dim> &) const override { return 0.0; }

    // velocity v(x,y)
    double v_space(const Point<dim> &p) const override
    {
      return sin(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double vx_space(const Point<dim> &p) const override
    {
      return M_PI * cos(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double vy_space(const Point<dim> &p) const override
    {
      return -M_PI * sin(M_PI * p[0]) * sin(M_PI * p[1]);
    }
    double vz_space(const Point<dim> &) const override { return 0.0; }
    double vxx_space(const Point<dim> &p) const override
    {
      return -M_PI * M_PI * sin(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double vyy_space(const Point<dim> &p) const override
    {
      return -M_PI * M_PI * sin(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double vzz_space(const Point<dim> &) const override { return 0.0; }

    // pressure
    double p_space(const Point<dim> &p) const override
    {
      return cos(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double px_space(const Point<dim> &p) const override
    {
      return -M_PI * sin(M_PI * p[0]) * cos(M_PI * p[1]);
    }
    double py_space(const Point<dim> &p) const override
    {
      return -M_PI * cos(M_PI * p[0]) * sin(M_PI * p[1]);
    }
    double pz_space(const Point<dim> &) const override { return 0.0; }
  };

  /**
   * u = y (y-1) + x
   * v = x (x-1) - y
   * p = x + y
   */
  // template <int dim>
  // class FlowB : public SeparableFlowMMS<dim>
  // {
  // public:
  //   FlowB(const TimeDependenceBase &time_function)
  //     : SeparableFlowMMS<dim>(time_function)
  //   {
  //     this->check_spatial_derivatives();
  //   }

  // private:
  //   // u = y(y-1) + x
  //   double u_fun(const Point<dim> &p) const override
  //   {
  //     return p[1] * (p[1] - 1.0) + p[0];
  //   }
  //   double ux_fun(const Point<dim> &) const override { return 1.0; }
  //   double uy_fun(const Point<dim> &p) const override
  //   {
  //     return 2.0 * p[1] - 1.0;
  //   }
  //   double uyy_fun(const Point<dim> &) const override { return 2.0; }

  //   // v = x(x-1) - y
  //   double v_fun(const Point<dim> &p) const override
  //   {
  //     return p[0] * (p[0] - 1.0) - p[1];
  //   }
  //   double vx_fun(const Point<dim> &p) const override
  //   {
  //     return 2.0 * p[0] - 1.0;
  //   }
  //   double vy_fun(const Point<dim> &) const override { return -1.0; }
  //   double vxx_fun(const Point<dim> &) const override { return 2.0; }

  //   // w = 0 in 2D

  //   // p = x + y
  //   double p_fun(const Point<dim> &p) const override { return p[0] + p[1]; }
  //   double px_fun(const Point<dim> &) const override { return 1.0; }
  //   double py_fun(const Point<dim> &) const override { return 1.0; }
  // };

  // /**
  //  * Constant or linear
  //  * u =
  //  * v =
  //  * p =
  //  */
  // template <int dim>
  // class FlowC : public SeparableFlowMMS<dim>
  // {
  // public:
  //   FlowC(const TimeDependenceBase &time_function)
  //     : SeparableFlowMMS<dim>(time_function)
  //   {
  //     this->check_spatial_derivatives();
  //   }

  // private:
  //   // u-component = y
  //   double u_fun(const Point<dim> &p) const override { return 0; }
  //   double uy_fun(const Point<dim> &) const override { return 0; }
  //   // v-component = 0
  //   double v_fun(const Point<dim> &p) const override { return 2 * p[0]; }
  //   double vx_fun(const Point<dim> &) const override { return 2; }
  //   // w-component = 0
  //   // pressure = x
  //   double p_fun(const Point<dim> &p) const override { return p[0] + p[1]; }
  //   double px_fun(const Point<dim> &) const override { return 1; }
  //   double py_fun(const Point<dim> &) const override { return 1; }
  // };

  // /**
  //  * u = C.
  //  * v = C.
  //  * p = C.
  //  */
  // template <int dim>
  // class FlowD : public SeparableFlowMMS<dim>
  // {
  // public:
  //   FlowD(const TimeDependenceBase &time_function)
  //     : SeparableFlowMMS<dim>(time_function)
  //   {
  //     this->check_spatial_derivatives();
  //   }

  // private:
  //   double u_fun(const Point<dim> &) const override { return 1.; }
  //   double v_fun(const Point<dim> &) const override { return 1.; }
  //   double w_fun(const Point<dim> &) const override
  //   {
  //     return (dim == 3) ? 1. : 0.;
  //   }
  //   double p_fun(const Point<dim> &) const override { return 1.; }
  // };

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

    void mesh_velocity(const double      t,
                       const Point<dim> &p,
                       Tensor<1, dim>   &dxdt) const
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
                              const double tol_order_2 = 1e-4) const override;
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
   * x_MMS(X, t) = X_0 + f(t) * kernel(|X - center|) * translation,
   *
   * where kernel is a C^2 bell-shaped function
   * and translation is the final translation vector.
   */
  template <int dim>
  class RigidMeshPosition : public MeshPositionMMSBase<dim>
  {
  public:
    const Point<dim>     center;
    const double         R0;
    const double         R1;
    const Tensor<1, dim> translation;
    const double spring_constant;

  public:
    RigidMeshPosition(const TimeDependenceBase &time_function,
                      const Point<dim>         &center,
                      const double              R0,
                      const double              R1,
                      const Tensor<1, dim>     &translation,
                      const double              spring_constant)
      : MeshPositionMMSBase<dim>(time_function)
      , center(center)
      , R0(R0)
      , R1(R1)
      , translation(translation)
      , spring_constant(spring_constant)
    {
      this->check_spatial_derivatives();
    }

  private:
    // x
    double x_fun(const Point<dim> &p) const override
    {
      return kernel_fun(p, center, R0, R1) * translation[0];
    };
    double xX_fun(const Point<dim> &p) const override
    {
      return dxi_kernel(p, center, R0, R1, 0) * translation[0];
    }
    double xY_fun(const Point<dim> &p) const override
    {
      return dxi_kernel(p, center, R0, R1, 1) * translation[0];
    }
    double xZ_fun(const Point<dim> &p) const override
    {
      return dxi_kernel(p, center, R0, R1, 2) * translation[0];
    }

    double xXX_fun(const Point<dim> &p) const override
    {
      return d2xij_kernel(p, center, R0, R1, 0, 0) * translation[0];
    }
    double xXY_fun(const Point<dim> &p) const override
    {
      return d2xij_kernel(p, center, R0, R1, 0, 1) * translation[0];
    }
    double xXZ_fun(const Point<dim> &p) const override
    {
      return d2xij_kernel(p, center, R0, R1, 0, 2) * translation[0];
    }
    double xYY_fun(const Point<dim> &p) const override
    {
      return d2xij_kernel(p, center, R0, R1, 1, 1) * translation[0];
    }
    double xZZ_fun(const Point<dim> &p) const override
    {
      return d2xij_kernel(p, center, R0, R1, 2, 2) * translation[0];
    }

    // y
    double y_fun(const Point<dim> &p) const override
    {
      return kernel_fun(p, center, R0, R1) * translation[1];
    };
    double yX_fun(const Point<dim> &p) const override
    {
      return dxi_kernel(p, center, R0, R1, 0) * translation[1];
    }
    double yY_fun(const Point<dim> &p) const override
    {
      return dxi_kernel(p, center, R0, R1, 1) * translation[1];
    }
    double yZ_fun(const Point<dim> &p) const override
    {
      return dxi_kernel(p, center, R0, R1, 2) * translation[1];
    }

    double yXX_fun(const Point<dim> &p) const override
    {
      return d2xij_kernel(p, center, R0, R1, 0, 0) * translation[1];
    }
    double yXY_fun(const Point<dim> &p) const override
    {
      return d2xij_kernel(p, center, R0, R1, 0, 1) * translation[1];
    }
    double yYZ_fun(const Point<dim> &p) const override
    {
      return d2xij_kernel(p, center, R0, R1, 1, 2) * translation[1];
    }
    double yYY_fun(const Point<dim> &p) const override
    {
      return d2xij_kernel(p, center, R0, R1, 1, 1) * translation[1];
    }
    double yZZ_fun(const Point<dim> &p) const override
    {
      return d2xij_kernel(p, center, R0, R1, 2, 2) * translation[1];
    }
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
      return sin(M_PI * p[0]) * sin(M_PI * p[0]) * sin(2.0 * M_PI * p[1]);
    }

    double xX_fun(const Point<dim> &p) const override
    {
      return M_PI * sin(2.0 * M_PI * p[0]) * sin(2.0 * M_PI * p[1]);
    }

    double xY_fun(const Point<dim> &p) const override
    {
      const double sX = sin(M_PI * p[0]);
      return 2.0 * M_PI * (sX * sX) * cos(2.0 * M_PI * p[1]);
    }

    double xXX_fun(const Point<dim> &p) const override
    {
      return 2.0 * M_PI * M_PI * cos(2.0 * M_PI * p[0]) *
             sin(2.0 * M_PI * p[1]);
    }

    double xXY_fun(const Point<dim> &p) const override
    {
      return 2.0 * M_PI * M_PI * sin(2.0 * M_PI * p[0]) *
             cos(2.0 * M_PI * p[1]);
    }

    double xYY_fun(const Point<dim> &p) const override
    {
      const double sX = sin(M_PI * p[0]);
      return -4.0 * M_PI * M_PI * (sX * sX) * sin(2.0 * M_PI * p[1]);
    }

    double y_fun(const Point<dim> &p) const override
    {
      return -sin(2.0 * M_PI * p[0]) * sin(M_PI * p[1]) * sin(M_PI * p[1]);
    }

    double yX_fun(const Point<dim> &p) const override
    {
      const double sY = sin(M_PI * p[1]);
      return -2.0 * M_PI * cos(2.0 * M_PI * p[0]) * (sY * sY);
    }

    double yY_fun(const Point<dim> &p) const override
    {
      return -M_PI * sin(2.0 * M_PI * p[0]) * sin(2.0 * M_PI * p[1]);
    }

    double yXX_fun(const Point<dim> &p) const override
    {
      const double sY = sin(M_PI * p[1]);
      return 4.0 * M_PI * M_PI * sin(2.0 * M_PI * p[0]) * (sY * sY);
    }

    double yXY_fun(const Point<dim> &p) const override
    {
      return -2.0 * M_PI * M_PI * cos(2.0 * M_PI * p[0]) *
             sin(2.0 * M_PI * p[1]);
    }

    double yYY_fun(const Point<dim> &p) const override
    {
      return -2.0 * M_PI * M_PI * sin(2.0 * M_PI * p[0]) *
             cos(2.0 * M_PI * p[1]);
    }
  };

  /**
   * Checks that the derivatives provided match
   * the ones obtained with finite differences.
   */
  template <int dim>
  void MeshPositionMMSBase<dim>::check_spatial_derivatives(
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


    auto check_order_1 =
      [&](double exact, double numerical, const std::string &name) {
        const double err = std::abs(exact - numerical);
        if(err < tol_order_1) return;
        const double relative_err = err / std::abs(numerical);
        AssertThrow(relative_err < tol_order_1,
                    ExcMessage("Derivative check failed for " + name +
                               ": exact = " + std::to_string(exact) +
                               ", FD = " + std::to_string(numerical) +
                               ", error = " + std::to_string(err)));
      };

    auto check_order_2 =
      [&](double exact, double numerical, const std::string &name) {
        const double err = std::abs(exact - numerical);
        if(err < tol_order_2) return;
        const double relative_err = err / std::abs(numerical);
        AssertThrow(relative_err < tol_order_2,
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

      check_order_2(
        xXY_fun(p),
        finite_diff_mixed([&](auto q) { return x_fun(q); }, p, 0, 1),
        "xXY");
      if constexpr (dim == 3)
        check_order_2(
          xXZ_fun(p),
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

      check_order_2(
        yXY_fun(p),
        finite_diff_mixed([&](auto q) { return y_fun(q); }, p, 0, 1),
        "yXY");
      if constexpr (dim == 3)
        check_order_2(
          yYZ_fun(p),
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

        check_order_2(
          zXZ_fun(p),
          finite_diff_mixed([&](auto q) { return z_fun(q); }, p, 0, 2),
          "zXZ");
        check_order_2(
          zYZ_fun(p),
          finite_diff_mixed([&](auto q) { return z_fun(q); }, p, 1, 2),
          "zYZ");
      }
    }
  }
} // namespace ManufacturedSolution