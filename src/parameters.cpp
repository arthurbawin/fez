
#include <parameters.h>

using namespace Parameters;

#define DECLARE_VERBOSITY_PARAM(prm)                                           \
  (prm).declare_entry("verbosity",                                             \
                      "verbose",                                               \
                      Patterns::Selection("quiet|verbose"),                    \
                      "Level of message display in console: quiet or verbose " \
                      "(default: verbose)");

#define READ_VERBOSITY_PARAM(prm)                                \
  {                                                              \
    const std::string parsed_verbosity = (prm).get("verbosity"); \
    if (parsed_verbosity == "quiet")                             \
      verbosity = Verbosity::quiet;                              \
    if (parsed_verbosity == "verbose")                           \
      verbosity = Verbosity::verbose;                            \
  }

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
  // prm.enter_subsection("Dimension");
  // {
  // }
  // prm.leave_subsection();
}

void Mesh::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Mesh");
  {
    prm.declare_entry("mesh file",
                      "",
                      Patterns::FileName(),
                      "Mesh file in .msh format (Gmsh msh4)");
    DECLARE_VERBOSITY_PARAM(prm)
  }
  prm.leave_subsection();
}

void Mesh::read_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Mesh");
  {
    filename = prm.get("mesh file");
    READ_VERBOSITY_PARAM(prm)
  }
  prm.leave_subsection();
}

void Output::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Output");
  {
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
  }
  prm.leave_subsection();
}

void Fluid::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Fluid");
  {
    prm.declare_entry("density", "1", Patterns::Double(), "Fluid density");
    prm.declare_entry("kinematic viscosity",
                      "1",
                      Patterns::Double(),
                      "Fluid kinematic viscosity");
  }
  prm.leave_subsection();
}

void Fluid::read_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Fluid");
  {
    density             = prm.get_double("density");
    kinematic_viscosity = prm.get_double("kinematic viscosity");
  }
  prm.leave_subsection();
}

void PseudoSolid::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Pseudosolid");
  {
    prm.declare_entry("lame lambda",
                      "1",
                      Patterns::Double(),
                      "First Lamé coefficient lambda");
    prm.declare_entry("lame mu",
                      "1",
                      Patterns::Double(),
                      "Second Lamé coefficient mu");
  }
  prm.leave_subsection();
}

void PseudoSolid::read_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Pseudosolid");
  {
    lame_lambda = prm.get_double("lame lambda");
    lame_mu     = prm.get_double("lame mu");
  }
  prm.leave_subsection();
}

void PhysicalProperties::declare_parameters(ParameterHandler &prm)
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
      fluids[i].declare_parameters(prm);

    // Declare the pseudosolid subsections
    prm.declare_entry(
      "number of pseudosolids",
      "1",
      Patterns::Integer(),
      "Number of pseudosolids (linear elastic analogy for mesh movement)");

    pseudosolids.resize(max_pseudosolids);
    for (unsigned int i = 0; i < max_pseudosolids; ++i)
      pseudosolids[i].declare_parameters(prm);
  }
  prm.leave_subsection();
}

void PhysicalProperties::read_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("Physical properties");
  {
    n_fluids = prm.get_integer("number of fluids");
    AssertThrow(n_fluids <= max_fluids,
                ExcMessage("More than " + std::to_string(max_fluids) +
                           " fluids are specified, which is not supported"));

    for (unsigned int i = 0; i < n_fluids; ++i)
      fluids[i].read_parameters(prm);

    n_pseudosolids = prm.get_integer("number of pseudosolids");
    AssertThrow(
      n_pseudosolids <= max_pseudosolids,
      ExcMessage(
        "More than " + std::to_string(max_pseudosolids) +
        " pseudo-solids (mesh analogy) are specified, which is not supported"));

    for (unsigned int i = 0; i < n_pseudosolids; ++i)
      pseudosolids[i].read_parameters(prm);
  }
  prm.leave_subsection();
}

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
    const double n_timesteps_estimate = (t_end - t_initial) / dt;
    n_constant_timesteps              = std::floor(n_timesteps_estimate);
    AssertThrow(
      std::abs(n_timesteps_estimate - n_constant_timesteps) < 1e-2,
      ExcMessage(
        "The prescribed (constant) time step does not yield an integer number "
        "of steps for the given time interval. The given time step yields " +
        std::to_string(n_timesteps_estimate) +
        " time steps. For now, we only consider an integer number of constant "
        "time steps."));

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
  }
  prm.leave_subsection();
}

void FSI::declare_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("FSI");
  {
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
  }
  prm.leave_subsection();
}

void FSI::read_parameters(ParameterHandler &prm)
{
  prm.enter_subsection("FSI");
  {
    enable_coupling = prm.get_bool("enable coupling");
    spring_constant = prm.get_double("spring constant");
    damping         = prm.get_double("damping");
    mass            = prm.get_double("mass");
  }
  prm.leave_subsection();
}

#undef DECLARE_VERBOSITY_PARAM
#undef READ_VERBOSITY_PARAM