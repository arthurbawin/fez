
#include <deal.II/base/symmetric_tensor.h>
#include <manufactured_solution.h>
#include <parsed_function_symengine.h>
#include <preset_mms.h>
#include <utilities.h>

namespace ManufacturedSolutions
{
  /**
   * Declare the parameters for all the possible preset MMS.
   * Declare the spatial and time component.
   */
  template <int dim>
  void declare_preset_manufactured_solutions(ParameterHandler &prm)
  {
    const std::string default_point = (dim == 2) ? "0, 0" : "0, 0, 0";
    std::shared_ptr<ParsedFunctionSDBase<dim>> dummy_time_fun =
      std::make_shared<ParsedFunctionSDBase<dim>>(1);
    prm.enter_subsection("space component");
    {
      prm.declare_entry(
        "preset",
        "none",
        Patterns::Selection(
          "none|time_dependent_vector|rigid_motion_kernel|moving radial kernel|"
          "normal_radial_kernel|vector radial kernel|vector one minus radial "
          "kernel"),
        "");

      //
      // Time dependent vector field
      //
      prm.enter_subsection("time_dependent_vector");
      {
        prm.declare_entry("constant vector",
                          default_point,
                          Patterns::List(Patterns::Double(), dim, dim, ","),
                          "Constant vector");
      }
      prm.leave_subsection();

      //
      // Vector-valued radial kernel
      //
      prm.enter_subsection("vector radial kernel");
      {
        prm.declare_entry("V",
                          default_point,
                          Patterns::List(Patterns::Double(), dim, dim, ","),
                          "Constant vector");
        prm.declare_entry("r0", "0.1", Patterns::Double(), "R0");
        prm.declare_entry("r1", "0.2", Patterns::Double(), "R1");
        prm.declare_entry("center",
                          default_point,
                          Patterns::List(Patterns::Double(), dim, dim, ","),
                          "Center of the kernel");
        prm.declare_entry("cylindrical", "false", Patterns::Bool(), "");
      }
      prm.leave_subsection();

      //
      // Vector-valued (1 - radial kernel)
      //
      prm.enter_subsection("vector one minus radial kernel");
      {
        prm.declare_entry("V",
                          default_point,
                          Patterns::List(Patterns::Double(), dim, dim, ","),
                          "Constant vector");
        prm.declare_entry("r0", "0.1", Patterns::Double(), "R0");
        prm.declare_entry("r1", "0.2", Patterns::Double(), "R1");
        prm.declare_entry("center",
                          default_point,
                          Patterns::List(Patterns::Double(), dim, dim, ","),
                          "Center of the kernel");
        prm.declare_entry("cylindrical", "false", Patterns::Bool(), "");
      }
      prm.leave_subsection();

      //
      // Radial basis kernel
      //
      prm.enter_subsection("rigid_motion_kernel");
      {
        prm.declare_entry("translation",
                          default_point,
                          Patterns::List(Patterns::Double(), dim, dim, ","),
                          "Translation vector");
        prm.declare_entry("r0", "0.1", Patterns::Double(), "R0");
        prm.declare_entry("r1", "0.2", Patterns::Double(), "R1");
        prm.declare_entry("center",
                          default_point,
                          Patterns::List(Patterns::Double(), dim, dim, ","),
                          "Initial center of the kernel");
        prm.declare_entry("cylindrical",
                          "false",
                          Patterns::Bool(),
                          "If true, the kernel is cylindrical (z-aligned) "
                          "instead of spherical.");
      }
      prm.leave_subsection();

      //
      // Radial basis kernel with moving center
      //
      prm.enter_subsection("moving radial kernel");
      {
        prm.declare_entry("translation",
                          default_point,
                          Patterns::List(Patterns::Double(), dim, dim, ","),
                          "Translation vector");
        prm.declare_entry("r0", "0.1", Patterns::Double(), "R0");
        prm.declare_entry("r1", "0.2", Patterns::Double(), "R1");
        prm.declare_entry("center",
                          default_point,
                          Patterns::List(Patterns::Double(), dim, dim, ","),
                          "Initial center of the kernel");
        prm.declare_entry("cylindrical", "false", Patterns::Bool(), "");
      }
      prm.leave_subsection();

      //
      // Radial basis kernel with product with normal
      //
      prm.enter_subsection("normal_radial_kernel");
      {
        prm.declare_entry("translation",
                          default_point,
                          Patterns::List(Patterns::Double(), dim, dim, ","),
                          "Translation vector");
        prm.declare_entry("r0", "0.1", Patterns::Double(), "R0");
        prm.declare_entry("r1", "0.2", Patterns::Double(), "R1");
        prm.declare_entry("a", "1.", Patterns::Double(), "Coefficient a");
        prm.declare_entry("center",
                          default_point,
                          Patterns::List(Patterns::Double(), dim, dim, ","),
                          "Initial center of the kernel");
        prm.declare_entry("cylindrical", "false", Patterns::Bool(), "");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
    prm.enter_subsection("time component");
    {
      dummy_time_fun->declare_parameters(prm, 1);
    }
    prm.leave_subsection();
  }

  /**
   *
   */
  template <int dim>
  void parse_preset_manufactured_solution(
    ParameterHandler                  &prm,
    PresetMMS                         &preset_field_mms,
    std::shared_ptr<MMSFunction<dim>> &preset_mms)
  {
    auto time_function = std::make_shared<ParsedFunctionSDBase<dim>>(1);

    // Parse the time function first to populate the pointer
    prm.enter_subsection("time component");
    time_function->parse_parameters(prm);
    AssertThrow(
      time_function->is_function_of_time_only(),
      ExcMessage("You tried to create a preset manufactured solution by "
                 "provided a function in a \"time component\" subsection which "
                 "is not a function of time only. The given function is : " +
                 time_function->get_function_expression()));

    prm.leave_subsection();

    // Then parse the spatial component and create the MMS object
    prm.enter_subsection("space component");
    {
      const std::string parsed_preset = prm.get("preset");
      if (parsed_preset == "none")
        preset_field_mms = PresetMMS::none;
      else if (parsed_preset == "time_dependent_vector")
        preset_field_mms = PresetMMS::time_dependent_vector;
      else if (parsed_preset == "rigid_motion_kernel")
        preset_field_mms = PresetMMS::rigid_motion_kernel;
      else if (parsed_preset == "moving radial kernel")
        preset_field_mms = PresetMMS::moving_radial_kernel;
      else if (parsed_preset == "normal_radial_kernel")
        preset_field_mms = PresetMMS::normal_radial_kernel;
      else if (parsed_preset == "vector radial kernel")
        preset_field_mms = PresetMMS::vector_radial_kernel;
      else if (parsed_preset == "vector one minus radial kernel")
        preset_field_mms = PresetMMS::vector_one_minus_radial_kernel;
      else
        AssertThrow(false, ExcMessage("Unknown preset MMS: " + parsed_preset));

      prm.enter_subsection("time_dependent_vector");
      {
        const Tensor<1, dim> constant_vector =
          parse_rank_1_tensor<dim>(prm.get("constant vector"));

        if (preset_field_mms == PresetMMS::time_dependent_vector)
          preset_mms =
            std::make_shared<TimeDependentVector<dim>>(time_function,
                                                       constant_vector);
      }
      prm.leave_subsection();

      prm.enter_subsection("vector radial kernel");
      {
        const Tensor<1, dim> V  = parse_rank_1_tensor<dim>(prm.get("V"));
        const double         r0 = prm.get_double("r0");
        const double         r1 = prm.get_double("r1");
        const Point<dim> center(parse_rank_1_tensor<dim>(prm.get("center")));
        const bool       cylindrical = prm.get_bool("cylindrical");
        if (preset_field_mms == PresetMMS::vector_radial_kernel)
          preset_mms = std::make_shared<VectorRadialKernel<dim>>(
            time_function, center, r0, r1, V, cylindrical);
      }
      prm.leave_subsection();

      prm.enter_subsection("vector one minus radial kernel");
      {
        const Tensor<1, dim> V  = parse_rank_1_tensor<dim>(prm.get("V"));
        const double         r0 = prm.get_double("r0");
        const double         r1 = prm.get_double("r1");
        const Point<dim> center(parse_rank_1_tensor<dim>(prm.get("center")));
        const bool       cylindrical = prm.get_bool("cylindrical");
        if (preset_field_mms == PresetMMS::vector_one_minus_radial_kernel)
          preset_mms = std::make_shared<VectorOneMinusRadialKernel<dim>>(
            time_function, center, r0, r1, V, cylindrical);
      }
      prm.leave_subsection();

      prm.enter_subsection("rigid_motion_kernel");
      {
        const Tensor<1, dim> translation =
          parse_rank_1_tensor<dim>(prm.get("translation"));
        const double     r0 = prm.get_double("r0");
        const double     r1 = prm.get_double("r1");
        const Point<dim> center(parse_rank_1_tensor<dim>(prm.get("center")));
        const bool       cylindrical = prm.get_bool("cylindrical");

        if (preset_field_mms == PresetMMS::rigid_motion_kernel)
          preset_mms = std::make_shared<PositionRadialKernel<dim>>(
            time_function, center, r0, r1, translation, cylindrical);
      }
      prm.leave_subsection();

      prm.enter_subsection("moving radial kernel");
      {
        const Tensor<1, dim> translation =
          parse_rank_1_tensor<dim>(prm.get("translation"));
        const double     r0 = prm.get_double("r0");
        const double     r1 = prm.get_double("r1");
        const Point<dim> center(parse_rank_1_tensor<dim>(prm.get("center")));
        const bool       cylindrical = prm.get_bool("cylindrical");

        if (preset_field_mms == PresetMMS::moving_radial_kernel)
          preset_mms = std::make_shared<MovingRadialKernel<dim>>(
            time_function, center, r0, r1, translation, cylindrical);
      }
      prm.leave_subsection();

      prm.enter_subsection("normal_radial_kernel");
      {
        const Tensor<1, dim> translation =
          parse_rank_1_tensor<dim>(prm.get("translation"));
        const double     r0 = prm.get_double("r0");
        const double     r1 = prm.get_double("r1");
        const double     a  = prm.get_double("a");
        const Point<dim> center(parse_rank_1_tensor<dim>(prm.get("center")));
        const bool       cylindrical = prm.get_bool("cylindrical");

        if (preset_field_mms == PresetMMS::normal_radial_kernel)
          preset_mms = std::make_shared<NormalRadialKernel<dim>>(
            time_function, center, r0, r1, translation, a, cylindrical);
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <int dim>
  void ManufacturedSolution<dim>::declare_parameters(ParameterHandler &prm)
  {
    std::shared_ptr<ParsedFunctionSDBase<dim>> dummy_scalar_fun =
      std::make_shared<ParsedFunctionSDBase<dim>>(1);
    std::shared_ptr<ParsedFunctionSDBase<dim>> dummy_vector_fun =
      std::make_shared<ParsedFunctionSDBase<dim>>(dim);

    prm.enter_subsection("Exact solution");
    {
      prm.enter_subsection("exact velocity");
      prm.declare_entry("as solution", "false", Patterns::Bool(), "");
      dummy_vector_fun->declare_parameters(prm, dim);
      declare_preset_manufactured_solutions<dim>(prm);
      prm.leave_subsection();
      prm.enter_subsection("exact pressure");
      prm.declare_entry("as solution", "false", Patterns::Bool(), "");
      dummy_scalar_fun->declare_parameters(prm, 1);
      declare_preset_manufactured_solutions<dim>(prm);
      prm.leave_subsection();
      prm.enter_subsection("exact mesh displacement");
      prm.declare_entry("as solution", "false", Patterns::Bool(), "");
      dummy_vector_fun->declare_parameters(prm, dim);
      declare_preset_manufactured_solutions<dim>(prm);
      prm.leave_subsection();
      prm.enter_subsection("exact cahn hilliard tracer");
      prm.declare_entry("as solution", "false", Patterns::Bool(), "");
      dummy_scalar_fun->declare_parameters(prm, 1);
      declare_preset_manufactured_solutions<dim>(prm);
      prm.leave_subsection();
      prm.enter_subsection("exact cahn hilliard potential");
      prm.declare_entry("as solution", "false", Patterns::Bool(), "");
      dummy_scalar_fun->declare_parameters(prm, 1);
      declare_preset_manufactured_solutions<dim>(prm);
      prm.leave_subsection();
      prm.enter_subsection("exact temperature");
      prm.declare_entry("as solution", "false", Patterns::Bool(), "");
      dummy_scalar_fun->declare_parameters(prm, 1);
      declare_preset_manufactured_solutions<dim>(prm);
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <int dim>
  void ManufacturedSolution<dim>::read_parameters(ParameterHandler &prm)
  {
    // For each unknown variable (velocity, pressure, ...),
    // initialize pointers for both a function with symbolic derivatives
    // and a preset manufactured solution, then assign the chosen one at the
    // end. Symbolic parsed function must be initialized to parse, but preset
    // mms pointers are created in parse_preset_manufactured_solution().

    auto sym_velocity      = std::make_shared<ParsedFunctionSDBase<dim>>(dim);
    auto sym_pressure      = std::make_shared<ParsedFunctionSDBase<dim>>(1);
    auto sym_mesh_position = std::make_shared<ParsedFunctionSDBase<dim>>(dim);
    auto sym_tracer        = std::make_shared<ParsedFunctionSDBase<dim>>(1);
    auto sym_potential     = std::make_shared<ParsedFunctionSDBase<dim>>(1);
    auto sym_temperature   = std::make_shared<ParsedFunctionSDBase<dim>>(1);

    std::shared_ptr<MMSFunction<dim>> preset_velocity;
    std::shared_ptr<MMSFunction<dim>> preset_pressure;
    std::shared_ptr<MMSFunction<dim>> preset_mesh_position;
    std::shared_ptr<MMSFunction<dim>> preset_tracer;
    std::shared_ptr<MMSFunction<dim>> preset_potential;
    std::shared_ptr<MMSFunction<dim>> preset_temperature;

    prm.enter_subsection("Exact solution");
    {
      prm.enter_subsection("exact velocity");
      set_field_as_solution["velocity"] = prm.get_bool("as solution");
      sym_velocity->parse_parameters(prm);
      parse_preset_manufactured_solution(prm,
                                         preset_velocity_type,
                                         preset_velocity);
      prm.leave_subsection();
      prm.enter_subsection("exact pressure");
      set_field_as_solution["pressure"] = prm.get_bool("as solution");
      sym_pressure->parse_parameters(prm);
      parse_preset_manufactured_solution(prm,
                                         preset_pressure_type,
                                         preset_pressure);
      prm.leave_subsection();
      prm.enter_subsection("exact mesh displacement");
      set_field_as_solution["mesh position"] = prm.get_bool("as solution");
      sym_mesh_position->parse_parameters(prm);
      parse_preset_manufactured_solution(prm,
                                         preset_mesh_position_type,
                                         preset_mesh_position);
      prm.leave_subsection();
      prm.enter_subsection("exact cahn hilliard tracer");
      set_field_as_solution["tracer"] = prm.get_bool("as solution");
      sym_tracer->parse_parameters(prm);
      parse_preset_manufactured_solution(prm,
                                         preset_tracer_type,
                                         preset_tracer);
      prm.leave_subsection();
      prm.enter_subsection("exact cahn hilliard potential");
      set_field_as_solution["potential"] = prm.get_bool("as solution");
      sym_potential->parse_parameters(prm);
      parse_preset_manufactured_solution(prm,
                                         preset_potential_type,
                                         preset_potential);
      prm.leave_subsection();
      prm.enter_subsection("exact temperature");
      set_field_as_solution["temperature"] = prm.get_bool("as solution");
      sym_temperature->parse_parameters(prm);
      parse_preset_manufactured_solution(prm,
                                         preset_temperature_type,
                                         preset_temperature);
      prm.leave_subsection();
    }
    prm.leave_subsection();

    // Assign the final choice based on the preset_type (none or preset)
    exact_velocity      = (preset_velocity_type == PresetMMS::none) ?
                            sym_velocity :
                            preset_velocity;
    exact_pressure      = (preset_pressure_type == PresetMMS::none) ?
                            sym_pressure :
                            preset_pressure;
    exact_mesh_position = (preset_mesh_position_type == PresetMMS::none) ?
                            sym_mesh_position :
                            preset_mesh_position;
    exact_tracer =
      (preset_tracer_type == PresetMMS::none) ? sym_tracer : preset_tracer;
    exact_potential   = (preset_potential_type == PresetMMS::none) ?
                          sym_potential :
                          preset_potential;
    exact_temperature = (preset_temperature_type == PresetMMS::none) ?
                          sym_temperature :
                          preset_temperature;

    exact_solution["velocity"]      = exact_velocity;
    exact_solution["pressure"]      = exact_pressure;
    exact_solution["mesh position"] = exact_mesh_position;
    exact_solution["pressure"]      = exact_tracer;
    exact_solution["potential"]     = exact_potential;
    exact_solution["temperature"]   = exact_temperature;
  }

  // Explicit instantiation
  template class ManufacturedSolution<2>;
  template class ManufacturedSolution<3>;

  template <int dim>
  Tensor<1, dim>
  MMSFunction<dim>::divergence_linear_elastic_stress_variable_coefficients(
    const Point<dim>                          &p,
    std::shared_ptr<ParsedFunctionSDBase<dim>> lame_mu,
    std::shared_ptr<ParsedFunctionSDBase<dim>> lame_lambda) const
  {
    const double         mu          = lame_mu->value(p);
    const double         lambda      = lame_lambda->value(p);
    const Tensor<1, dim> grad_mu     = lame_mu->gradient(p);
    const Tensor<1, dim> grad_lambda = lame_lambda->gradient(p);
    const Tensor<2, dim> grad_x      = gradient_vi_xj(p);
    const Tensor<1, dim> grad_div_x  = grad_div(p);

    const SymmetricTensor<2, dim> strain =
      symmetrize(grad_x) - unit_symmetric_tensor<dim>();
    const Tensor<1, dim> div_strain = 0.5 * (vector_laplacian(p) + grad_div_x);

    return 2. * (mu * div_strain + grad_mu * strain) + lambda * grad_div_x +
           grad_lambda * trace(strain);
  }

  template <int dim>
  void MMSFunction<dim>::check_derivatives(const double tol_order_1,
                                           const double tol_order_2)
  {
    const double h_first  = 1e-8;
    const double h_second = 1e-5;

    std::vector<double> test_times = {0., 0.1, 0.345, 1., 10.};

    std::vector<Point<dim>> test_points;
    if constexpr (dim == 2)
      test_points = {{0., 0.}, {0.3, 0.7}, {0.52, 0.52}, {0.9, 0.1}};
    else if constexpr (dim == 3)
      test_points = {{0., 0., 0.},
                     {0.3, 0.7, 0.2},
                     {0.5, 0.5, 0.5},
                     {0.9, 0.1, 0.4}};

    auto check_relative_error = [&](double       t,
                                    double       exact,
                                    double       numerical,
                                    double       tol,
                                    unsigned int comp,
                                    std::string  name,
                                    std::string  entry) {
      const double err = std::abs(exact - numerical);

      if (err < tol)
        return;

      const double relative_err = err / std::abs(numerical);
      AssertThrow(relative_err < tol,
                  ExcMessage(
                    "Derivative check failed for " + name + " (entry " + entry +
                    ") and component " + std::to_string(comp) + " at " +
                    std::to_string(t) + ": exact = " + std::to_string(exact) +
                    ", FD = " + std::to_string(numerical) +
                    ", absolute error = " + std::to_string(err) +
                    ", relative error = " + std::to_string(relative_err)));
    };

    for (const auto &t : test_times)
      for (const auto &p : test_points)
        for (unsigned int i_comp = 0; i_comp < this->n_components; ++i_comp)
        {
          // For each vector component:
          // Check time derivatives
          if (!ignore_time_derivative)
          {
            this->set_time(t + h_first);
            const double val_plus = this->value(p, i_comp);
            this->set_time(t - h_first);
            const double val_minus = this->value(p, i_comp);
            const double fdot_fd   = (val_plus - val_minus) / (2.0 * h_first);
            this->set_time(t);
            check_relative_error(t,
                                 this->time_derivative(p, i_comp),
                                 fdot_fd,
                                 tol_order_1,
                                 i_comp,
                                 "time derivative",
                                 "0");
          }

          // Check gradient at time t
          this->set_time(t);
          const Tensor<1, dim> grad = this->gradient(p, i_comp);
          for (unsigned int d = 0; d < dim; ++d)
          {
            Point<dim> p_plus = p, p_minus = p;
            p_plus[d] += h_first;
            p_minus[d] -= h_first;
            const double val_plus  = this->value(p_plus, i_comp);
            const double val_minus = this->value(p_minus, i_comp);
            const double grad_fd   = (val_plus - val_minus) / (2.0 * h_first);
            check_relative_error(t,
                                 grad[d],
                                 grad_fd,
                                 tol_order_1,
                                 i_comp,
                                 "gradient",
                                 std::to_string(d));
          }

          // Check hessian at time t
          if (!ignore_hessian)
          {
            this->set_time(t);
            const SymmetricTensor<2, dim> hess = this->hessian(p, i_comp);
            for (unsigned int di = 0; di < dim; ++di)
              for (unsigned int dj = 0; dj < dim; ++dj)
              {
                double d2_fd;
                if (di == dj)
                {
                  // Centered 2nd order finite differences
                  Point<dim> p_p = p, p_m = p;
                  p_p[di] += h_second;
                  p_m[di] -= h_second;
                  const double val_p = this->value(p_p, i_comp);
                  const double val_m = this->value(p_m, i_comp);
                  const double val   = this->value(p, i_comp);
                  d2_fd = (val_p - 2.0 * val + val_m) / (h_second * h_second);
                }
                else
                {
                  Point<dim> p_pp = p, p_pm = p, p_mp = p, p_mm = p;
                  p_pp[di] += h_second;
                  p_pp[dj] += h_second;
                  p_pm[di] += h_second;
                  p_pm[dj] -= h_second;
                  p_mp[di] -= h_second;
                  p_mp[dj] += h_second;
                  p_mm[di] -= h_second;
                  p_mm[dj] -= h_second;
                  const double val_pp = this->value(p_pp, i_comp);
                  const double val_pm = this->value(p_pm, i_comp);
                  const double val_mp = this->value(p_mp, i_comp);
                  const double val_mm = this->value(p_mm, i_comp);
                  d2_fd               = (val_pp - val_pm - val_mp + val_mm) /
                          (4.0 * h_second * h_second);
                }
                check_relative_error(t,
                                     hess[di][dj],
                                     d2_fd,
                                     tol_order_2,
                                     i_comp,
                                     "hessian",
                                     std::to_string(di) + " - " +
                                       std::to_string(dj));
              }
          }
        }
  }

  template class MMSFunction<2>;
  template class MMSFunction<3>;

} // namespace ManufacturedSolutions
