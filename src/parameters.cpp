
#include <parameters.h>
#include <utilities.h>

#define DECLARE_VERBOSITY_PARAM(prm, default_verbosity)                        \
  (prm).declare_entry("verbosity",                                             \
                      std::string(default_verbosity),                          \
                      Patterns::Selection("quiet|verbose"),                    \
                      "Level of message display in console: quiet or verbose " \
                      "(default: " +                                           \
                        std::string(default_verbosity) + ")");

#define READ_VERBOSITY_PARAM(prm, verbosity)                     \
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
      DECLARE_VERBOSITY_PARAM(prm, "quiet")
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
      READ_VERBOSITY_PARAM(prm, fluid_verbosity)
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
      READ_VERBOSITY_PARAM(prm, verbosity)
    }
    prm.leave_subsection();
  }

  void Output::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Output");
    {
      prm.declare_entry("write vtu results",
                        "false",
                        Patterns::Bool(),
                        "Enable/disable vtu output writing.");
      prm.declare_entry("output directory",
                        "./",
                        Patterns::FileName(),
                        "Output directory.");
      prm.declare_entry("output prefix",
                        "solution",
                        Patterns::FileName(),
                        "Prefix for the output files.");
      prm.declare_entry(
        "vtu output frequency",
        "1",
        Patterns::Integer(1),
        "Frequency (in time steps) for the standard vtu export.");
      prm.enter_subsection("skin");
      {
        prm.declare_entry(
          "write vtu results",
          "false",
          Patterns::Bool(),
          "Enable extraction of vtu files on prescribed boundary.");
        prm.declare_entry("boundary id",
                          "0",
                          Patterns::Integer(0),
                          "Boundary id used to define the skin.");
        prm.declare_entry("output prefix",
                          "skin",
                          Patterns::FileName(),
                          "Prefix for the output files on the skin");
        prm.declare_entry("output frequency",
                          "1",
                          Patterns::Integer(1),
                          "Frequency (in time steps) for the exportation of "
                          "skin-related files");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  void Output::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Output");
    {
      write_results        = prm.get_bool("write vtu results");
      output_dir           = prm.get("output directory");
      output_prefix        = prm.get("output prefix");
      vtu_output_frequency = prm.get_integer("vtu output frequency");
      prm.enter_subsection("skin");
      {
        skin.write_results    = prm.get_bool("write vtu results");
        skin.boundary_id      = prm.get_integer("boundary id");
        skin.output_prefix    = prm.get("output prefix");
        skin.output_frequency = prm.get_integer("output frequency");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  void declare_postprocessing_base(ParameterHandler &prm)
  {
    DECLARE_VERBOSITY_PARAM(prm, "verbose")
    prm.declare_entry("enable",
                      "false",
                      Patterns::Bool(),
                      "Enable/disable this postprocessing");
    prm.declare_entry("write results",
                      "true",
                      Patterns::Bool(),
                      "Write result of this postprocessing to file");
    prm.declare_entry("output prefix",
                      "forces",
                      Patterns::FileName(),
                      "Prefix for the postprocessing output files");
    prm.declare_entry(
      "output frequency",
      "1",
      Patterns::Integer(1),
      "Frequency (in time steps) for the exportation of postprocessing files");
    prm.declare_entry("precision",
                      "6",
                      Patterns::Integer(1),
                      "Number of significant digits to print");
  }

  void declare_postprocessing_boundary(ParameterHandler &prm)
  {
    declare_postprocessing_base(prm);
    prm.declare_entry(
      "boundary id",
      "0",
      Patterns::Integer(0),
      "Boundary id on which this postprocessing should be applied");
  }

  void PostProcessing::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Postprocessing");
    {
      prm.enter_subsection("forces computation");
      {
        declare_postprocessing_boundary(prm);
        prm.declare_entry("computation method",
                          "stress vector",
                          Patterns::Selection(
                            "stress vector|lagrange multiplier"),
                          "Method used to evaluate the hydrodynamic forces");
      }
      prm.leave_subsection();
      prm.enter_subsection("structure position");
      {
        declare_postprocessing_boundary(prm);
      }
      prm.leave_subsection();
      prm.enter_subsection("slicing");
      {
        declare_postprocessing_boundary(prm);
        prm.declare_entry("along which axis",
                          "z",
                          Patterns::Selection("x|y|z"),
                          "Axis along which the slices are defined");
        prm.declare_entry("number of slices",
                          "1",
                          Patterns::Integer(1),
                          "Number of slices along the chosen direction");
        prm.declare_entry(
          "compute forces",
          "false",
          Patterns::Bool(),
          "Compute and write the hydrodynamic forces on each slice");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  void read_postprocessing_base(ParameterHandler                   &prm,
                                PostProcessing::PostProcessingBase &pp_base)
  {
    READ_VERBOSITY_PARAM(prm, pp_base.verbosity)
    pp_base.enable           = prm.get_bool("enable");
    pp_base.write_results    = prm.get_bool("write results");
    pp_base.output_prefix    = prm.get("output prefix");
    pp_base.output_frequency = prm.get_integer("output frequency");
    pp_base.precision        = prm.get_integer("precision");
  }

  void read_postprocessing_boundary(
    ParameterHandler                           &prm,
    PostProcessing::PostProcessingBaseBoundary &pp_boundary)
  {
    read_postprocessing_base(prm, pp_boundary);
    pp_boundary.boundary_id = prm.get_integer("boundary id");
  }

  void PostProcessing::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Postprocessing");
    {
      prm.enter_subsection("forces computation");
      {
        read_postprocessing_boundary(prm, forces);
        const std::string parsed_method = prm.get("computation method");
        if (parsed_method == "stress vector")
          forces.method = Forces::ComputationMethod::stress_vector;
        else if (parsed_method == "lagrange multiplier")
          forces.method = Forces::ComputationMethod::lagrange_multiplier;
      }
      prm.leave_subsection();
      prm.enter_subsection("structure position");
      {
        read_postprocessing_boundary(prm, structure_position);
      }
      prm.leave_subsection();
      prm.enter_subsection("slicing");
      {
        read_postprocessing_boundary(prm, slices);
        slices.along_which_axis         = prm.get("along which axis");
        slices.n_slices                 = prm.get_integer("number of slices");
        slices.compute_forces_on_slices = prm.get_bool("compute forces");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  // Declare the parameters for a quadrature rule.
  // The default parameters are different for the rule to compute the numerical
  // solution, and for the rule used to compute error norms.
  // If default_for_error = true, then this declares the default parameters for
  // the rule used to compute errors.
  template <int dim>
  void declare_quadrature_rule(ParameterHandler &prm,
                               const bool        default_for_error = false)
  {
    prm.declare_entry("rule for simplices",
                      default_for_error ? "witherden_vincent" : "gauss",
                      Patterns::Selection("gauss|witherden_vincent"),
                      "Gauss or Witherden-Vincent (odd order) rule.");
    prm.declare_entry("number of 1d nodes for cell quadrature",
                      default_for_error ? ((dim == 2) ? "6" : "5") : "4",
                      Patterns::Integer(),
                      "Number of nodes in the base 1d quadrature. Will "
                      "integrate a polynomial of degree 2n-1.");
    prm.declare_entry("number of 1d nodes for face quadrature",
                      default_for_error ? ((dim == 2) ? "6" : "5") : "4",
                      Patterns::Integer(),
                      "Number of nodes in the base 1d quadrature. Will "
                      "integrate a polynomial of degree 2n-1.");
  }

  template <int dim>
  void FiniteElements<dim>::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("FiniteElements");
    {
      prm.declare_entry("use quads",
                        "false",
                        Patterns::Bool(),
                        "If true, use quads/hexes instead of simplices");
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
      prm.declare_entry("Temperature degree",
                        "1",
                        Patterns::Integer(),
                        "Polynomial degree of the temperature interpolant");

      prm.enter_subsection("Quadrature rule");
      {
        declare_quadrature_rule<dim>(prm);
      }
      prm.leave_subsection();
      prm.enter_subsection("Quadrature rule for error");
      {
        declare_quadrature_rule<dim>(prm, true);
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <int dim>
  void read_quadrature_rule(ParameterHandler                             &prm,
                            typename FiniteElements<dim>::QuadratureRule &rule)
  {
    const std::string parsed_rule = prm.get("rule for simplices");
    if (parsed_rule == "gauss")
      rule.type = FiniteElements<dim>::QuadratureRule::Type::GaussSimplex;
    else if (parsed_rule == "witherden_vincent")
      rule.type = FiniteElements<dim>::QuadratureRule::Type::WitherdenVincent;
    else
      throw std::runtime_error("Unknown quadrature rule : " + parsed_rule);

    rule.n_pts_1D_simplex_cell_quad =
      prm.get_integer("number of 1d nodes for cell quadrature");
    rule.n_pts_1D_simplex_face_quad =
      prm.get_integer("number of 1d nodes for face quadrature");

    const unsigned int nc = rule.n_pts_1D_simplex_cell_quad,
                       nf = rule.n_pts_1D_simplex_face_quad;

    // Test input parameters. For simplices, default is QGaussSimplex with
    // n_points_1d = 4 (maximum available)
    if (rule.type == FiniteElements<dim>::QuadratureRule::Type::GaussSimplex)
    {
      // Max implemented in deal.ii is n_points_1d = 4
      AssertThrow(1 <= nc and nc <= 4,
                  ExcMessage("Gauss quadrature rule on simplicial cells "
                             "expects n_points_1d in [1,4]."));
      AssertThrow(1 <= nf and nf <= 4,
                  ExcMessage("Gauss quadrature rule on simplicial faces "
                             "expects n_points_1d in [1,4]."));
    }
    if (rule.type ==
        FiniteElements<dim>::QuadratureRule::Type::WitherdenVincent)
    {
      // Max implemented in deal.ii is n_points_1d = 7 in 2D and = 5 in 3D
      if constexpr (dim == 2)
      {
        // Max for cells is 7, no max for edges (QGauss 1D rules)
        AssertThrow(1 <= nc and nc <= 7,
                    ExcMessage("WitherdenVincent quadrature rule on simplicial "
                               "cells in 2D expects n_points_1d in [1,7]."));
        AssertThrow(1 <= nf,
                    ExcMessage("Quadrature rule on simplicial faces in 1D "
                               "expects n_points_1d greater than 0."));
      }
      else
      {
        // Max for cells is 5, max for faces is 7
        AssertThrow(1 <= nc and nc <= 5,
                    ExcMessage("WitherdenVincent quadrature rule on simplicial "
                               "cells in 3D expects n_points_1d in [1,5]."));
        AssertThrow(1 <= nf and nf <= 7,
                    ExcMessage("WitherdenVincent quadrature rule on simplicial "
                               "cells in 2D expects n_points_1d in [1,7]."));
      }
    }
  }

  template <int dim>
  void FiniteElements<dim>::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("FiniteElements");
    {
      use_quads            = prm.get_bool("use quads");
      velocity_degree      = prm.get_integer("Velocity degree");
      pressure_degree      = prm.get_integer("Pressure degree");
      mesh_position_degree = prm.get_integer("Mesh position degree");
      no_slip_lagrange_mult_degree =
        prm.get_integer("Lagrange multiplier degree");
      tracer_degree      = prm.get_integer("Tracer degree");
      potential_degree   = prm.get_integer("Potential degree");
      temperature_degree = prm.get_integer("Temperature degree");

      prm.enter_subsection("Quadrature rule");
      {
        read_quadrature_rule<dim>(prm, rule);
      }
      prm.leave_subsection();
      prm.enter_subsection("Quadrature rule for error");
      {
        read_quadrature_rule<dim>(prm, rule_for_error);
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template struct FiniteElements<2>;
  template struct FiniteElements<3>;

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
      dynamic_viscosity   = density * kinematic_viscosity;
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
      prm.enter_subsection("reassembly heuristic");
      {
        prm.declare_entry(
          "decrease tolerance",
          "0.",
          Patterns::Double(),
          "If the norm of the current residual is higher than this value times "
          "the previous residual norm, reassemble the matrix.");
      }
      prm.leave_subsection();
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
      prm.enter_subsection("reassembly heuristic");
      {
        reassembly_decrease_tol = prm.get_double("decrease tolerance");
      }
      prm.leave_subsection();
      READ_VERBOSITY_PARAM(prm, verbosity)
    }
    prm.leave_subsection();
  }

  void LinearSolver::declare_parameters(ParameterHandler  &prm,
                                        const std::string &solver_type)
  {
    prm.enter_subsection("Linear solver");
    {
      prm.enter_subsection(solver_type);
      prm.declare_entry("method",
                        "direct_mumps",
                        Patterns::Selection("direct_mumps|cg|gmres"),
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
                        "Max level of fill-in for ILU preconditioner");
      prm.declare_entry("reuse", "false", Patterns::Bool(), "");
      DECLARE_VERBOSITY_PARAM(prm, "verbose")
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  void LinearSolver::read_parameters(ParameterHandler  &prm,
                                     const std::string &solver_type)
  {
    prm.enter_subsection("Linear solver");
    {
      prm.enter_subsection(solver_type);
      {
        const std::string parsed_method = prm.get("method");
        if (parsed_method == "direct_mumps")
          method = Method::direct_mumps;
        else if (parsed_method == "cg")
          method = Method::cg;
        else if (parsed_method == "gmres")
          method = Method::gmres;
        tolerance      = prm.get_double("tolerance");
        max_iterations = prm.get_integer("max iterations");
        ilu_fill_level = prm.get_integer("ilu fill level");
        reuse          = prm.get_bool("reuse");
        READ_VERBOSITY_PARAM(prm, verbosity)
      }
      prm.leave_subsection();
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

      prm.declare_entry("adaptative_dt","false",Patterns::Bool(),
                        "Enable adaptative time step");
      prm.declare_entry("dt min factor",
                  "0.1",
                  Patterns::Double(0.0),
                  "Minimum time step as a factor of dt (dt_min = dt * factor).");

      prm.declare_entry("dt max factor",
                  "10.0",
                  Patterns::Double(0.0),
                  "Maximum time step as a factor of dt (dt_max = dt * factor).");
      

      prm.declare_entry("reject factor", "2", Patterns::Double(), "Factor of eps for the reject of the adaptive dt");
      prm.declare_entry("max reject step", "8", Patterns::Double(), "Maximum number of rejection of the same step");
      prm.declare_entry("safety", "0.9", Patterns::Double(), "Safety factor for adaptive dt");
      
      prm.declare_entry("eps_u", "1e-3", Patterns::Double(), "Target error for velocity (adaptive dt)");
      prm.declare_entry("eps_p", "1e-2", Patterns::Double(), "Target error for pressure (adaptive dt)");
      prm.declare_entry("eps_x", "1e-3", Patterns::Double(), "Target error for position (adaptive dt)");
      prm.declare_entry("eps_t",
                  "-1",
                  Patterns::Double(),
                  "Target error for temperature (adaptive dt). If <= 0, fallback to eps_u");
      prm.declare_entry("eps_l",
                        "-1",
                        Patterns::Double(),
                        "Target error for Lagrange multiplier (adaptive dt). If <= 0, fallback to eps_u");
      prm.declare_entry("eps_phi",
                        "-1",
                        Patterns::Double(),
                        "Target error for tracer phi (adaptive dt). If <= 0, fallback to eps_u");
      prm.declare_entry("eps_mu",
                        "-1",
                        Patterns::Double(),
                        "Target error for potential mu (adaptive dt). If <= 0, fallback to eps_u");

      prm.declare_entry("mms_scale_eps_with_dt",
      "false",
      Patterns::Bool(),
      "If true, scale eps for MMS when dt is refined: eps <- eps*(dt/dt_ref)^p.");
      
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

      adaptative_dt = prm.get_bool("adaptative_dt");
      dt_max_factor  = prm.get_double("dt max factor");
      dt_min_factor  = prm.get_double("dt min factor");
      
      reject_factor        = prm.get_double("reject factor");
      max_rejects_per_step = prm.get_double("max reject step"); 
      safety               = prm.get_double("safety");


      eps_u        = prm.get_double("eps_u");
      eps_p        = prm.get_double("eps_p");
      eps_x        = prm.get_double("eps_x");
      eps_t        = prm.get_double("eps_t");
      eps_l        = prm.get_double("eps_l");
      eps_phi      = prm.get_double("eps_phi");
      eps_mu       = prm.get_double("eps_mu");

      mms_scale_eps_with_dt = prm.get_bool("mms_scale_eps_with_dt");

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
      READ_VERBOSITY_PARAM(prm, verbosity)
    }
    prm.leave_subsection();
  }

  template <int dim>
  void CahnHilliard<dim>::declare_parameters(ParameterHandler &prm)
  {
    const std::string default_point = (dim == 2) ? "0, 0" : "0, 0, 0";
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
      prm.declare_entry("body force",
                        default_point,
                        Patterns::List(Patterns::Double(), dim, dim, ","),
                        "Body force vector (e.g., gravity acceleration)");
      prm.declare_entry("enable tracer limiter",
                        "false",
                        Patterns::Bool(),
                        "Enable limiter for the tracer (phase field marker)");
      // Mesh forcing parameters
      prm.declare_entry(
        "alpha",
        "0.0",
        Patterns::Double(),
        "Coefficient of pseudosolid source term alpha * phi * grad(phi).");
      prm.declare_entry("beta",
                        "0.0",
                        Patterns::Double(),
                        "Coefficient of pseudosolid source term beta * (u_ALE "
                        "* grad(phi)) * grad(phi).");
    }
    prm.leave_subsection();
  }

  template <int dim>
  void CahnHilliard<dim>::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Cahn Hilliard");
    {
      const std::string parsed_mobility_model = prm.get("mobility model");
      if (parsed_mobility_model == "linear")
        mobility_model = MobilityModel::constant;
      mobility            = prm.get_double("mobility");
      surface_tension     = prm.get_double("surface tension");
      epsilon_interface   = prm.get_double("interface thickness");
      body_force          = parse_rank_1_tensor<dim>(prm.get("body force"));
      with_tracer_limiter = prm.get_bool("enable tracer limiter");
      // mesh forcing parameters
      alpha = prm.get_double("alpha");
      beta  = prm.get_double("beta");
    }
    prm.leave_subsection();
  }

  template class CahnHilliard<2>;
  template class CahnHilliard<3>;

  void LinearElasticity::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Linear elasticity");
    {
      prm.enter_subsection("current mesh source term");
      {
        prm.declare_entry(
          "enable",
          "false",
          Patterns::Bool(),
          "Enable the evaluation of the given position source term on the "
          "current mesh (and not on the reference mesh as usual)");
        prm.declare_entry("min multiplier",
                          "1.",
                          Patterns::Double(1.),
                          "Minimum coefficient multiplying the source term "
                          "evaluated on the current mesh");
        prm.declare_entry("max multiplier",
                          "1.",
                          Patterns::Double(1.),
                          "Maximum coefficient multiplying the source term "
                          "evaluated on the current mesh");
        prm.declare_entry("continuation steps",
                          "1",
                          Patterns::Integer(1),
                          "Number of steps to use in the continuation method");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  void LinearElasticity::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Linear elasticity");
    {
      prm.enter_subsection("current mesh source term");
      {
        enable_source_term_on_current_mesh = prm.get_bool("enable");
        min_current_mesh_source_term_multiplier =
          prm.get_double("min multiplier");
        max_current_mesh_source_term_multiplier =
          prm.get_double("max multiplier");
        AssertThrow(max_current_mesh_source_term_multiplier >=
                      min_current_mesh_source_term_multiplier,
                    ExcMessage("Max source term multiplier should be greater "
                               "than the min multiplier"));
        n_continuation_steps = prm.get_integer("continuation steps");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  void CheckpointRestart::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Checkpoint Restart");
    {
      prm.declare_entry("enable checkpoint",
                        "false",
                        Patterns::Bool(),
                        "Save data periodically to allow restart?");
      prm.declare_entry("restart",
                        "false",
                        Patterns::Bool(),
                        "Restart simulation from given checkpoint file?");
      prm.declare_entry(
        "checkpoint file",
        "checkpoint",
        Patterns::Anything(),
        "Name of the file to write to and read from the checkpoint");
      prm.declare_entry("checkpoint frequency",
                        "10",
                        Patterns::Integer(),
                        "Write a checkpoint every N time steps");
    }
    prm.leave_subsection();
  }

  void CheckpointRestart::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Checkpoint Restart");
    {
      enable_checkpoint    = prm.get_bool("enable checkpoint");
      restart              = prm.get_bool("restart");
      filename             = prm.get("checkpoint file");
      checkpoint_frequency = prm.get_integer("checkpoint frequency");
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
        prm.declare_entry(
          "norms to compute",
          "L2_norm",
          Patterns::List(
            *Patterns::Tools::Convert<VectorTools::NormType>::to_pattern()),
          "A comma-separated list of norms (e.g., L2_norm, H1_norm)");
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
        mesh_prefix      = prm.get("mesh prefix");
        first_mesh_index = prm.get_integer("first mesh");
        const auto parsed_norms =
          Utilities::split_string_list(prm.get("norms to compute"));
        for (const auto &s : parsed_norms)
          norms_to_compute.push_back(
            Patterns::Tools::Convert<VectorTools::NormType>::to_value(s));
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

  template <int dim>
  void FSI<dim>::declare_parameters(ParameterHandler &prm)
  {
    const std::string default_point = (dim == 2) ? "0, 0" : "0, 0, 0";
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
      prm.declare_entry("cylinder center",
                        default_point,
                        Patterns::List(Patterns::Double(), dim, dim, ","),
                        "Center of the cylinder");
      prm.declare_entry("fix z component", "true", Patterns::Bool(), "");
      prm.declare_entry("compute error on forces",
                        "false",
                        Patterns::Bool(),
                        "");
    }
    prm.leave_subsection();
  }

  template <int dim>
  void FSI<dim>::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("FSI");
    {
      READ_VERBOSITY_PARAM(prm, verbosity)
      enable_coupling = prm.get_bool("enable coupling");
      spring_constant = prm.get_double("spring constant");
      damping         = prm.get_double("damping");
      mass            = prm.get_double("mass");
      cylinder_radius = prm.get_double("cylinder radius");
      cylinder_length = prm.get_double("cylinder length");
      cylinder_center = parse_rank_1_tensor<dim>(prm.get("cylinder center"));
      fix_z_component = prm.get_bool("fix z component");
      compute_error_on_forces = prm.get_bool("compute error on forces");
    }
    prm.leave_subsection();
  }

  template struct FSI<2>;
  template struct FSI<3>;

  void Debug::declare_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Debug");
    {
      DECLARE_VERBOSITY_PARAM(prm, "quiet")
      prm.declare_entry(
        "write dealii mesh as msh",
        "false",
        Patterns::Bool(),
        "If using deal.II meshing routines, write the mesh as a .msh file");
      prm.declare_entry("write partition gmsh",
                        "false",
                        Patterns::Bool(),
                        "Write the mesh partitions as a Gmsh .pos file.");
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
      prm.declare_entry("fsi_coupling_option", "1", Patterns::Integer(), "");
    }
    prm.leave_subsection();
  }

  void Debug::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Debug");
    {
      READ_VERBOSITY_PARAM(prm, verbosity)
      write_dealii_mesh_as_msh = prm.get_bool("write dealii mesh as msh");
      write_partition_pos_gmsh = prm.get_bool("write partition gmsh");
      apply_exact_solution     = prm.get_bool("apply exact solution");
      compare_analytical_jacobian_with_fd =
        prm.get_bool("compare jacobian matrix with fd");
      analytical_jacobian_absolute_tolerance =
        prm.get_double("analytical_jacobian_absolute_tolerance");
      analytical_jacobian_relative_tolerance =
        prm.get_double("analytical_jacobian_relative_tolerance");
      fsi_apply_erroneous_coupling =
        prm.get_bool("fsi_apply_erroneous_coupling");
      fsi_check_mms_on_boundary = prm.get_bool("fsi_check_mms_on_boundary");
      fsi_coupling_option       = prm.get_integer("fsi_coupling_option");
    }
    prm.leave_subsection();
  }
} // namespace Parameters

#undef DECLARE_VERBOSITY_PARAM
#undef READ_VERBOSITY_PARAM
