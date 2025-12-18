
#include <parameters.h>

#define DECLARE_VERBOSITY_PARAM(prm, default_verbosity)                        \
  (prm).declare_entry("verbosity",                                             \
                      std::string(default_verbosity),                          \
                      Patterns::Selection("quiet|verbose"),                    \
                      "Level of message display in console: quiet or verbose " \
                      "(default: " +                                           \
                        std::string(default_verbosity) + ")");

#define READ_VERBOSITY_PARAM(prm)                                \
  {                                                              \
    const std::string parsed_verbosity = (prm).get("verbosity"); \
    if (parsed_verbosity == "quiet")                             \
      verbosity = Verbosity::quiet;                              \
    if (parsed_verbosity == "verbose")                           \
      verbosity = Verbosity::verbose;                            \
  }

namespace Parameters
{
  /**
   * These should agree with the declare_entry in utilities.h
   */
  void DummyDimension::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Dimension");
    {
      // Read in utilities, declared here but not parsed
      prm.declare_entry("dimension",
                        "2",
                        Patterns::Integer(),
                        "Problem dimension (2 or 3)");
    }
    prm.leave_subsection();
  }

  void DummyDimension::read_parameters(ParameterHandler &prm)
  {
    // Nothing to do, dimension was read in utilities.h
  }

  void Timer::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Timer");
    {
      prm.declare_entry("enable timer",
                        "false",
                        Patterns::Bool(),
                        "Enable summary of elapsed time in solver components");
    }
    prm.leave_subsection();
  }

  void Timer::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Timer");
    {
      enable_timer = prm.get_bool("enable timer");
    }
    prm.leave_subsection();
  }

  void BoundaryConditionsData::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid boundary conditions");
    {
      prm.declare_entry(
        "fix pressure constant",
        "false",
        Patterns::Bool(),
        "Fix pressure nullspace by pinning a single pressure point");
      prm.declare_entry(
        "enforce zero mean pressure",
        "false",
        Patterns::Bool(),
        "Fix pressure nullspace by enforcing zero-mean pressure");
    }
    prm.leave_subsection();
  }

  void BoundaryConditionsData::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Fluid boundary conditions");
    {
      fix_pressure_constant      = prm.get_bool("fix pressure constant");
      enforce_zero_mean_pressure = prm.get_bool("enforce zero mean pressure");
    }
    prm.leave_subsection();
  }

  void Mesh::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Mesh");
    {
      prm.declare_entry("mesh file",
                        "",
                        Patterns::FileName(),
                        "Mesh file in .msh format (Gmsh msh4)");
      prm.declare_entry("use dealii cube mesh",
                        "false",
                        Patterns::Bool(),
                        "Use cube mesh from deal.II's routines");
      prm.declare_entry("dealii preset mesh",
                        "none",
                        Patterns::Selection("none|cube|rectangle|holed plate"),
                        "Use dealii meshing routines for specified geometry");
      prm.declare_entry("dealii mesh parameters",
                        "2, 2 : 0., 0. : 1., 1. : false");
      prm.declare_entry(
        "refinement level",
        "1",
        Patterns::Integer(),
        "Level of uniform refinement if using deal.II's meshing routines");
      DECLARE_VERBOSITY_PARAM(prm, "verbose")
    }
    prm.leave_subsection();
  }

  void Mesh::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Mesh");
    {
      filename              = prm.get("mesh file");
      use_deal_ii_cube_mesh = prm.get_bool("use dealii cube mesh");
      deal_ii_preset_mesh   = prm.get("dealii preset mesh");
      deal_ii_mesh_param    = prm.get("dealii mesh parameters");
      refinement_level      = prm.get_integer("refinement level");
      READ_VERBOSITY_PARAM(prm)
    }
    prm.leave_subsection();
  }

  void Output::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Output");
    {
      prm.declare_entry("write results",
                        "true",
                        Patterns::Bool(),
                        "Enable/disable output writing");
      prm.declare_entry("output directory",
                        "./",
                        Patterns::FileName(),
                        "Output directory");
      prm.declare_entry("output prefix",
                        "solution",
                        Patterns::FileName(),
                        "Prefix to attach to the output files");
    }
    prm.leave_subsection();
  }

  void Output::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Output");
    {
      write_results = prm.get_bool("write results");
      output_dir    = prm.get("output directory");
      output_prefix = prm.get("output prefix");
    }
    prm.leave_subsection();
  }

  void FiniteElements::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("FiniteElements");
    {
      prm.declare_entry("Velocity degree",
                        "2",
                        Patterns::Integer(),
                        "Polynomial degree of the velocity interpolant");
      prm.declare_entry("Pressure degree",
                        "1",
                        Patterns::Integer(),
                        "Polynomial degree of the pressure interpolant");
      prm.declare_entry("Mesh position degree",
                        "1",
                        Patterns::Integer(),
                        "Polynomial degree of the mesh position interpolant");
      prm.declare_entry(
        "Lagrange multiplier degree",
        "2",
        Patterns::Integer(),
        "Polynomial degree of the no-slip Lagrange multiplier interpolant");
      prm.declare_entry("Tracer degree",
                        "1",
                        Patterns::Integer(),
                        "Polynomial degree of the CHNS tracer interpolant");
      prm.declare_entry(
        "Potential degree",
        "1",
        Patterns::Integer(),
        "Polynomial degree of the CHNS chemical potential interpolant");
    }
    prm.leave_subsection();
  }

  void FiniteElements::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("FiniteElements");
    {
      velocity_degree      = prm.get_integer("Velocity degree");
      pressure_degree      = prm.get_integer("Pressure degree");
      mesh_position_degree = prm.get_integer("Mesh position degree");
      no_slip_lagrange_mult_degree =
        prm.get_integer("Lagrange multiplier degree");
      tracer_degree    = prm.get_integer("Tracer degree");
      potential_degree = prm.get_integer("Potential degree");
    }
    prm.leave_subsection();
  }

  void Fluid::declare_parameters(ParameterHandler &prm, unsigned int index)
  {
    prm.enter_subsection("Fluid " + std::to_string(index));
    {
      prm.declare_entry("density", "1", Patterns::Double(), "Fluid density");
      prm.declare_entry("kinematic viscosity",
                        "1",
                        Patterns::Double(),
                        "Fluid kinematic viscosity");
    }
    prm.leave_subsection();
  }

  void Fluid::read_parameters(ParameterHandler &prm, unsigned int index)
  {
    prm.enter_subsection("Fluid " + std::to_string(index));
    {
      density             = prm.get_double("density");
      kinematic_viscosity = prm.get_double("kinematic viscosity");
    }
    prm.leave_subsection();
  }

  template <int dim>
  void PseudoSolid<dim>::declare_parameters(ParameterHandler &prm,
                                            unsigned int      index)
  {
    lame_lambda_fun =
      std::make_shared<ManufacturedSolutions::ParsedFunctionSDBase<dim>>(1);
    lame_mu_fun =
      std::make_shared<ManufacturedSolutions::ParsedFunctionSDBase<dim>>(1);

    prm.enter_subsection("Pseudosolid " + std::to_string(index));
    {
      prm.enter_subsection("lame lambda");
      lame_lambda_fun->declare_parameters(prm);
      prm.leave_subsection();
      prm.enter_subsection("lame mu");
      lame_mu_fun->declare_parameters(prm);
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <int dim>
  void PseudoSolid<dim>::read_parameters(ParameterHandler &prm,
                                         unsigned int      index)
  {
    prm.enter_subsection("Pseudosolid " + std::to_string(index));
    {
      prm.enter_subsection("lame lambda");
      lame_lambda_fun->parse_parameters(prm);
      prm.leave_subsection();
      prm.enter_subsection("lame mu");
      lame_mu_fun->parse_parameters(prm);
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template class PseudoSolid<2>;
  template class PseudoSolid<3>;

  template <int dim>
  void PhysicalProperties<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Physical properties");
    {
      // Declare the fluid subsections
      prm.declare_entry("number of fluids",
                        "1",
                        Patterns::Integer(),
                        "Number of fluids");

      fluids.resize(max_fluids);
      for (unsigned int i = 0; i < max_fluids; ++i)
        fluids[i].declare_parameters(prm, i);

      // Declare the pseudosolid subsections
      prm.declare_entry(
        "number of pseudosolids",
        "0",
        Patterns::Integer(),
        "Number of pseudosolids (linear elastic analogy for mesh movement)");

      pseudosolids.resize(max_pseudosolids);
      for (unsigned int i = 0; i < max_pseudosolids; ++i)
        pseudosolids[i].declare_parameters(prm, i);
    }
    prm.leave_subsection();
  }

  template <int dim>
  void PhysicalProperties<dim>::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Physical properties");
    {
      n_fluids = prm.get_integer("number of fluids");
      AssertThrow(n_fluids <= max_fluids,
                  ExcMessage("More than " + std::to_string(max_fluids) +
                             " fluids are specified, which is not supported"));

      for (unsigned int i = 0; i < n_fluids; ++i)
        fluids[i].read_parameters(prm, i);

      n_pseudosolids = prm.get_integer("number of pseudosolids");
      AssertThrow(n_pseudosolids <= max_pseudosolids,
                  ExcMessage("More than " + std::to_string(max_pseudosolids) +
                             " pseudo-solids (mesh analogy) are specified, "
                             "which is not supported"));

      for (unsigned int i = 0; i < n_pseudosolids; ++i)
        pseudosolids[i].read_parameters(prm, i);
    }
    prm.leave_subsection();
  }

  template class PhysicalProperties<2>;
  template class PhysicalProperties<3>;

  void NonLinearSolver::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Nonlinear solver");
    {
      prm.declare_entry("tolerance",
                        "1e-10",
                        Patterns::Double(),
                        "Stopping tolerance for the Newton solver");
      prm.declare_entry("divergence_tolerance",
                        "1e4",
                        Patterns::Double(),
                        "Stop nonlinear solver if residual exceeds this value");
      prm.declare_entry("max_iterations",
                        "1",
                        Patterns::Integer(),
                        "Maximum number of Newton iterations");
      prm.declare_entry("enable_line_search",
                        "true",
                        Patterns::Bool(),
                        "Enable line search");
      prm.declare_entry(
        "analytic_jacobian",
        "true",
        Patterns::Bool(),
        "Compute exact Jacobian matrix. If false, use finite differences.");
      DECLARE_VERBOSITY_PARAM(prm, "verbose")
    }
    prm.leave_subsection();
  }

  void NonLinearSolver::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Nonlinear solver");
    {
      tolerance            = prm.get_double("tolerance");
      divergence_tolerance = prm.get_double("divergence_tolerance");
      max_iterations       = prm.get_integer("max_iterations");
      enable_line_search   = prm.get_bool("enable_line_search");
      analytic_jacobian    = prm.get_bool("analytic_jacobian");
      READ_VERBOSITY_PARAM(prm);
    }
    prm.leave_subsection();
  }

  void LinearSolver::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Linear solver");
    {
      prm.declare_entry("method",
                        "direct_mumps",
                        Patterns::Selection("direct_mumps|gmres"),
                        "Method");
      prm.declare_entry("tolerance",
                        "1e-6",
                        Patterns::Double(),
                        "Tolerance for iterative solver");
      prm.declare_entry("max iterations",
                        "100",
                        Patterns::Integer(),
                        "Max number of outer iterations for iterative solver");
      prm.declare_entry("ilu fill level",
                        "0",
                        Patterns::Integer(),
                        "Max lelve of fill-in for ILU preconditioner");
      prm.declare_entry("renumber", "false", Patterns::Bool(), "");
      prm.declare_entry("reuse", "false", Patterns::Bool(), "");
      DECLARE_VERBOSITY_PARAM(prm, "verbose")
    }
    prm.leave_subsection();
  }

  void LinearSolver::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Linear solver");
    {
      const std::string parsed_method = prm.get("method");
      if (parsed_method == "direct_mumps")
        method = Method::direct_mumps;
      else if (parsed_method == "gmres")
        method = Method::gmres;
      tolerance      = prm.get_double("tolerance");
      max_iterations = prm.get_integer("max iterations");
      ilu_fill_level = prm.get_integer("ilu fill level");
      renumber       = prm.get_bool("renumber");
      reuse          = prm.get_bool("reuse");
      READ_VERBOSITY_PARAM(prm);
    }
    prm.leave_subsection();
  }

  void TimeIntegration::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Time integration");
    {
      prm.declare_entry("dt", "1", Patterns::Double(), "Time step");
      prm.declare_entry("t_initial",
                        "1",
                        Patterns::Double(),
                        "Beginning of the simulation time interval");
      prm.declare_entry("t_end",
                        "1",
                        Patterns::Double(),
                        "End of the simulation time interval");
      prm.declare_entry("scheme",
                        "stationary",
                        Patterns::Selection("stationary|BDF1|BDF2"),
                        "Time stepping scheme (default is stationary)");
      prm.declare_entry("bdf start method",
                        "initial condition",
                        Patterns::Selection("initial condition|BDF1"),
                        "Starting method for BDF schemes of order > 1.");
      DECLARE_VERBOSITY_PARAM(prm, "verbose")
    }
    prm.leave_subsection();
  }

  void TimeIntegration::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Time integration");
    {
      dt        = prm.get_double("dt");
      t_initial = prm.get_double("t_initial");
      t_end     = prm.get_double("t_end");

      // Set the number of (constant) time steps.
      // For now, we only consider an integer number of time steps.
      // const double n_timesteps_estimate = (t_end - t_initial) / dt;
      // n_constant_timesteps              = std::floor(n_timesteps_estimate);
      // AssertThrow(
      //   std::abs(n_timesteps_estimate - n_constant_timesteps) < 1e-2,
      //   ExcMessage(
      //     "The prescribed (constant) time step does not yield an integer
      //     number " "of steps for the given time interval. The given time step
      //     yields "
      //     + std::to_string(n_timesteps_estimate) + " time steps. For now, we
      //     only consider an integer number of constant " "time steps."));

      const std::string parsed_scheme = prm.get("scheme");
      if (parsed_scheme == "stationary")
        scheme = Scheme::stationary;
      else if (parsed_scheme == "BDF1")
        scheme = Scheme::BDF1;
      else if (parsed_scheme == "BDF2")
        scheme = Scheme::BDF2;
      else
        throw std::runtime_error("Unknown time intergation scheme : " +
                                 parsed_scheme);
      const std::string parsed_startup = prm.get("bdf start method");
      if (parsed_startup == "initial condition")
        bdfstart = BDFStart::initial_condition;
      else if (parsed_startup == "BDF1")
        bdfstart = BDFStart::BDF1;
      else
        throw std::runtime_error("Unknown BDF starting method : " +
                                 parsed_startup);
      READ_VERBOSITY_PARAM(prm)
    }
    prm.leave_subsection();
  }

  void CahnHilliard::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Cahn Hilliard");
    {
      prm.declare_entry("mobility model",
                        "constant",
                        Patterns::Selection("constant"),
                        "Model for the mobility tensor");
      prm.declare_entry("mobility",
                        "1.",
                        Patterns::Double(),
                        "Mobility value if constant");
      prm.declare_entry("surface tension",
                        "1.",
                        Patterns::Double(),
                        "Fluid-fluid surface tension");
      prm.declare_entry("interface thickness",
                        "1e-2",
                        Patterns::Double(),
                        "Interface thickness (epsilon)");
    }
    prm.leave_subsection();
  }

  void CahnHilliard::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Cahn Hilliard");
    {
      const std::string parsed_mobility_model = prm.get("mobility model");
      if (parsed_mobility_model == "linear")
        mobility_model = MobilityModel::constant;
      mobility          = prm.get_double("mobility");
      surface_tension   = prm.get_double("surface tension");
      epsilon_interface = prm.get_double("interface thickness");
    }
    prm.leave_subsection();
  }

  void MMS::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Manufactured solution");
    {
      prm.declare_entry("enable",
                        "false",
                        Patterns::Bool(),
                        "Enable convergence study for the prescribed "
                        "manufactured solution. This "
                        "overrides the given mesh filename.");
      prm.declare_entry("type",
                        "space",
                        Patterns::Selection("space|time|spacetime"),
                        "Choose between space and/or time convergence study.");
      prm.declare_entry(
        "subtract mean pressure",
        "false",
        Patterns::Bool(),
        "Subtract mean pressure for L2 error computation. If disabled and if "
        "the "
        "pressure solution is not zero-mean, an error is thrown to avoid "
        "returning erroneous convergence rates.");
      prm.declare_entry("force source term",
                        "false",
                        Patterns::Bool(),
                        "Use the provided source term instead of the one "
                        "computed with symbolic differentiation");
      prm.declare_entry("convergence steps",
                        "1",
                        Patterns::Integer(),
                        "Number of steps in the convergence study");
      prm.declare_entry(
        "run only step",
        "-1",
        Patterns::Integer(),
        "If specified, run only this convergence step (in [0, n_steps])");
      prm.enter_subsection("Space convergence");
      {
        prm.declare_entry("use dealii cube mesh",
                          "false",
                          Patterns::Bool(),
                          "Use cube mesh from deal.II's routines");
        prm.declare_entry("use dealii holed plate mesh",
                          "false",
                          Patterns::Bool(),
                          "Use plate with hole mesh from deal.II's routines");
        prm.declare_entry("mesh prefix",
                          "",
                          Patterns::Anything(),
                          "Prefix (including full path) for the meshes to use "
                          "for the spatial convergence study");
        prm.declare_entry("first mesh",
                          "0",
                          Patterns::Integer(),
                          "Index of the first mesh, which will be (mesh "
                          "prefix)_(first mesh).msh");
        prm.declare_entry("compute L2 norm", "false", Patterns::Bool(), "");
        prm.declare_entry("compute Li norm", "false", Patterns::Bool(), "");
        prm.declare_entry("compute H1 norm", "false", Patterns::Bool(), "");
      }
      prm.leave_subsection();
      prm.enter_subsection("Time convergence");
      {
        prm.declare_entry("norm",
                          "L1",
                          Patterns::Selection("L1|L2|Linfty"),
                          "Lp norm to use for the temporal error.");
        prm.declare_entry(
          "use spatial mesh",
          "false",
          Patterns::Bool(),
          "If true, use the mesh provided for the spatial convergence study. "
          "If "
          "false, use the provided mesh in the Mesh subsection.");
        prm.declare_entry("spatial mesh index",
                          "0",
                          Patterns::Integer(),
                          "If use spatial mesh is true, set the index "
                          "(refinement level) of the used mesh.");
        prm.declare_entry("time step reduction",
                          "0.5",
                          Patterns::Double(),
                          "Reduction factor between two time steps in a time "
                          "convergence study.");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  void MMS::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Manufactured solution");
    {
      enable                        = prm.get_bool("enable");
      const std::string parsed_type = prm.get("type");
      if (parsed_type == "space")
        type = Type::space;
      else if (parsed_type == "time")
        type = Type::time;
      else if (parsed_type == "spacetime")
        type = Type::spacetime;
      subtract_mean_pressure = prm.get_bool("subtract mean pressure");
      force_source_term      = prm.get_bool("force source term");
      n_convergence          = prm.get_integer("convergence steps");
      run_only_step          = prm.get_integer("run only step");
      prm.enter_subsection("Space convergence");
      {
        use_deal_ii_cube_mesh = prm.get_bool("use dealii cube mesh");
        use_deal_ii_holed_plate_mesh =
          prm.get_bool("use dealii holed plate mesh");
        mesh_prefix             = prm.get("mesh prefix");
        first_mesh_index        = prm.get_integer("first mesh");
        compute_L2_spatial_norm = prm.get_bool("compute L2 norm");
        if (compute_L2_spatial_norm)
          norms_to_compute.push_back(VectorTools::L2_norm);
        compute_Li_spatial_norm = prm.get_bool("compute Li norm");
        if (compute_Li_spatial_norm)
          norms_to_compute.push_back(VectorTools::Linfty_norm);
        compute_H1_spatial_norm = prm.get_bool("compute H1 norm");
        if (compute_H1_spatial_norm)
          norms_to_compute.push_back(VectorTools::H1_seminorm);
      }
      prm.leave_subsection();
      prm.enter_subsection("Time convergence");
      {
        const std::string parsed_norm = prm.get("norm");
        if (parsed_norm == "L1")
          time_norm = TimeLpNorm::L1;
        else if (parsed_norm == "L2")
          time_norm = TimeLpNorm::L2;
        else if (parsed_norm == "Linfty")
          time_norm = TimeLpNorm::Linfty;
        use_space_convergence_mesh = prm.get_bool("use spatial mesh");
        spatial_mesh_index         = prm.get_integer("spatial mesh index");
        time_step_reduction_factor = prm.get_double("time step reduction");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  void FSI::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("FSI");
    {
      DECLARE_VERBOSITY_PARAM(prm, "verbose")
      prm.declare_entry("enable coupling",
                        "false",
                        Patterns::Bool(),
                        "Enable coupling between fluid and solid obstacle");
      prm.declare_entry("spring constant",
                        "1",
                        Patterns::Double(),
                        "Spring stiffness constant attached to solid object");
      prm.declare_entry("damping",
                        "1",
                        Patterns::Double(),
                        "Damping coefficient of the studied system");
      prm.declare_entry("mass",
                        "1",
                        Patterns::Double(),
                        "Mass of the studied system");
      prm.declare_entry("cylinder radius", "1", Patterns::Double(), "");
      prm.declare_entry("cylinder length", "1", Patterns::Double(), "");
      prm.declare_entry("fix z component", "true", Patterns::Bool(), "");
    }
    prm.leave_subsection();
  }

  void FSI::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("FSI");
    {
      READ_VERBOSITY_PARAM(prm)
      enable_coupling = prm.get_bool("enable coupling");
      spring_constant = prm.get_double("spring constant");
      damping         = prm.get_double("damping");
      mass            = prm.get_double("mass");
      cylinder_radius = prm.get_double("cylinder radius");
      cylinder_length = prm.get_double("cylinder length");
      fix_z_component = prm.get_bool("fix z component");
    }
    prm.leave_subsection();
  }

  void Debug::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Debug");
    {
      DECLARE_VERBOSITY_PARAM(prm, "quiet")
      prm.declare_entry("apply exact solution", "false", Patterns::Bool(), "");
      prm.declare_entry("compare jacobian matrix with fd",
                        "false",
                        Patterns::Bool(),
                        "");
      prm.declare_entry("analytical_jacobian_absolute_tolerance",
                        "1e-3",
                        Patterns::Double(),
                        "");
      prm.declare_entry("analytical_jacobian_relative_tolerance",
                        "1e-3",
                        Patterns::Double(),
                        "");
      prm.declare_entry("fsi_apply_erroneous_coupling",
                        "false",
                        Patterns::Bool(),
                        "");
      prm.declare_entry("fsi_check_mms_on_boundary",
                        "false",
                        Patterns::Bool(),
                        "");
    }
    prm.leave_subsection();
  }

  void Debug::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Debug");
    {
      READ_VERBOSITY_PARAM(prm)
      apply_exact_solution = prm.get_bool("apply exact solution");
      compare_analytical_jacobian_with_fd =
        prm.get_bool("compare jacobian matrix with fd");
      analytical_jacobian_absolute_tolerance =
        prm.get_double("analytical_jacobian_absolute_tolerance");
      analytical_jacobian_relative_tolerance =
        prm.get_double("analytical_jacobian_relative_tolerance");
      fsi_apply_erroneous_coupling =
        prm.get_bool("fsi_apply_erroneous_coupling");
      fsi_check_mms_on_boundary = prm.get_bool("fsi_check_mms_on_boundary");
    }
    prm.leave_subsection();
  }
} // namespace Parameters

#undef DECLARE_VERBOSITY_PARAM
#undef READ_VERBOSITY_PARAM