
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
          "normal_radial_kernel"),
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

    prm.enter_subsection("Manufactured solution");
    {
      prm.enter_subsection("exact velocity");
      dummy_vector_fun->declare_parameters(prm, dim);
      declare_preset_manufactured_solutions<dim>(prm);
      prm.leave_subsection();
      prm.enter_subsection("exact pressure");
      dummy_scalar_fun->declare_parameters(prm, 1);
      declare_preset_manufactured_solutions<dim>(prm);
      prm.leave_subsection();
      prm.enter_subsection("exact mesh displacement");
      dummy_vector_fun->declare_parameters(prm, dim);
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

    std::shared_ptr<MMSFunction<dim>> preset_velocity;
    std::shared_ptr<MMSFunction<dim>> preset_pressure;
    std::shared_ptr<MMSFunction<dim>> preset_mesh_position;

    prm.enter_subsection("Manufactured solution");
    {
      prm.enter_subsection("exact velocity");
      sym_velocity->parse_parameters(prm);
      parse_preset_manufactured_solution(prm,
                                         preset_velocity_type,
                                         preset_velocity);
      prm.leave_subsection();
      prm.enter_subsection("exact pressure");
      sym_pressure->parse_parameters(prm);
      parse_preset_manufactured_solution(prm,
                                         preset_pressure_type,
                                         preset_pressure);
      prm.leave_subsection();
      prm.enter_subsection("exact mesh displacement");
      sym_mesh_position->parse_parameters(prm);
      parse_preset_manufactured_solution(prm,
                                         preset_mesh_position_type,
                                         preset_mesh_position);
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
  }

  // Explicit instantiation
  template class ManufacturedSolution<2>;
  template class ManufacturedSolution<3>;

  template <int dim>
  Tensor<1, dim>
  MMSFunction<dim>::divergence_linear_elastic_stress_variable_coefficients(const Point<dim> &p,
      std::shared_ptr<ParsedFunctionSDBase<dim>> lame_mu,
      std::shared_ptr<ParsedFunctionSDBase<dim>> lame_lambda) const
  {
    const double mu     = lame_mu->value(p);
    const double lambda = lame_lambda->value(p);
    const Tensor<1, dim> grad_mu = lame_mu->gradient(p);
    const Tensor<1, dim> grad_lambda = lame_lambda->gradient(p);
    const Tensor<2, dim> grad_x = this->gradient_vi_xj(p);
    const Tensor<2, dim> grad_x_sym = grad_x + transpose(grad_x);

    return mu * this->vector_laplacian(p) +
           (mu + lambda) * this->grad_div(p) + 
           grad_mu * grad_x_sym + grad_lambda * this->divergence(p);
  }

  template class MMSFunction<2>;
  template class MMSFunction<3>;

} // namespace ManufacturedSolutions