
#include <parameter_reader.h>

#include <sstream>

namespace
{
  void enter_path(ParameterHandler                &prm,
                  const std::vector<std::string> &path)
  {
    for (const auto &subsection : path)
      prm.enter_subsection(subsection);
  }

  void leave_path(ParameterHandler                &prm,
                  const std::vector<std::string> &path)
  {
    for (unsigned int i = 0; i < path.size(); ++i)
      prm.leave_subsection();
  }

  void append_parameter(std::ostringstream              &out,
                        ParameterHandler               &prm,
                        const std::vector<std::string> &path,
                        const std::string              &entry)
  {
    enter_path(prm, path);
    out << "/";
    for (const auto &subsection : path)
      out << subsection << "/";
    out << entry << "=" << prm.get(entry) << "\n";
    leave_path(prm, path);
  }
} // namespace

template <int dim>
std::string
ParameterReader<dim>::make_presolved_mesh_position_fingerprint(
  ParameterHandler &prm) const
{
  std::ostringstream out;

  out << "format=1\n";
  out << "template_dimension=" << dim << "\n";

  append_parameter(out, prm, {"Dimension"}, "dimension");

  append_parameter(out, prm, {"Mesh"}, "mesh file");
  append_parameter(out, prm, {"Mesh"}, "use dealii cube mesh");
  append_parameter(out, prm, {"Mesh"}, "dealii preset mesh");
  append_parameter(out, prm, {"Mesh"}, "dealii mesh parameters");
  append_parameter(out, prm, {"Mesh"}, "refinement level");

  append_parameter(out, prm, {"FiniteElements"}, "use quads");
  append_parameter(out, prm, {"FiniteElements"}, "Mesh position degree");
  append_parameter(out, prm, {"FiniteElements"}, "Tracer degree");

  append_parameter(out, prm, {"Linear elasticity", "source term"},
                   "Function expression");
  append_parameter(out, prm, {"Linear elasticity", "source term"},
                   "Function constants");
  append_parameter(out, prm, {"Linear elasticity", "current mesh source term"},
                   "enable");
  append_parameter(out, prm, {"Linear elasticity", "current mesh source term"},
                   "min multiplier");
  append_parameter(out, prm, {"Linear elasticity", "current mesh source term"},
                   "max multiplier");
  append_parameter(out, prm, {"Linear elasticity", "current mesh source term"},
                   "continuation steps");
  append_parameter(out, prm, {"Linear elasticity", "CHNS presolver"},
                   "enable");
  append_parameter(out, prm, {"Linear elasticity", "CHNS presolver"},
                   "initial compression multiplier");
  append_parameter(out, prm, {"Linear elasticity", "CHNS presolver"},
                   "continuation steps");

  append_parameter(out, prm, {"Cahn Hilliard"}, "CHNS model");
  append_parameter(out, prm, {"Cahn Hilliard"}, "interface thickness");
  append_parameter(out, prm, {"Cahn Hilliard"}, "tanh mixing steepness");
  append_parameter(out, prm, {"Cahn Hilliard"}, "psi interface width factor");
  append_parameter(out, prm, {"Cahn Hilliard"},
                   "mff_enlarged_compression_factor");
  append_parameter(out, prm, {"Cahn Hilliard"},
                   "mff_physics_compression_factor");
  append_parameter(out, prm, {"Cahn Hilliard"}, "mff_regularization_gamma");
  append_parameter(out,
                   prm,
                   {"Cahn Hilliard"},
                   "mff_enlarged_factor_equalization_exponent");
  append_parameter(out, prm, {"Cahn Hilliard"}, "psi mu correction factor");

  append_parameter(out, prm, {"Initial conditions"}, "use enlarged psi");
  append_parameter(out,
                   prm,
                   {"Initial conditions", "cahn hilliard tracer"},
                   "Function expression");
  append_parameter(out,
                   prm,
                   {"Initial conditions", "cahn hilliard tracer"},
                   "Function constants");
  append_parameter(out,
                   prm,
                   {"Initial conditions", "enlarged psi"},
                   "Function expression");
  append_parameter(out,
                   prm,
                   {"Initial conditions", "enlarged psi"},
                   "Function constants");

  append_parameter(out, prm, {"Physical properties"}, "number of pseudosolids");
  for (unsigned int i = 0; i < physical_properties.n_pseudosolids; ++i)
  {
    const std::string subsection = "Pseudosolid " + std::to_string(i);
    append_parameter(out,
                     prm,
                     {"Physical properties", subsection},
                     "constitutive model");
    append_parameter(out, prm, {"Physical properties", subsection},
                     "ogden beta");
    append_parameter(out,
                     prm,
                     {"Physical properties", subsection, "lame lambda"},
                     "Function expression");
    append_parameter(out,
                     prm,
                     {"Physical properties", subsection, "lame lambda"},
                     "Function constants");
    append_parameter(out,
                     prm,
                     {"Physical properties", subsection, "lame mu"},
                     "Function expression");
    append_parameter(out,
                     prm,
                     {"Physical properties", subsection, "lame mu"},
                     "Function constants");
  }

  append_parameter(out, prm, {"Pseudosolid boundary conditions"}, "number");
  for (unsigned int i = 0; i < bc_data.n_pseudosolid_bc; ++i)
  {
    const std::string subsection = "boundary " + std::to_string(i);
    append_parameter(out,
                     prm,
                     {"Pseudosolid boundary conditions", subsection},
                     "id");
    append_parameter(out,
                     prm,
                     {"Pseudosolid boundary conditions", subsection},
                     "name");
    append_parameter(out,
                     prm,
                     {"Pseudosolid boundary conditions", subsection},
                     "type");
    for (const std::string component : {"x", "y", "z"})
    {
      append_parameter(out,
                       prm,
                       {"Pseudosolid boundary conditions",
                        subsection,
                        component},
                       "type");
      append_parameter(out,
                       prm,
                       {"Pseudosolid boundary conditions",
                        subsection,
                        component},
                       "Function expression");
      append_parameter(out,
                       prm,
                       {"Pseudosolid boundary conditions",
                        subsection,
                        component},
                       "Function constants");
    }
  }

  append_parameter(out, prm, {"CahnHilliard boundary conditions"}, "number");
  for (unsigned int i = 0; i < bc_data.n_cahn_hilliard_bc; ++i)
  {
    const std::string subsection = "boundary " + std::to_string(i);
    append_parameter(out,
                     prm,
                     {"CahnHilliard boundary conditions", subsection},
                     "id");
    append_parameter(out,
                     prm,
                     {"CahnHilliard boundary conditions", subsection},
                     "name");
    append_parameter(out,
                     prm,
                     {"CahnHilliard boundary conditions", subsection},
                     "type");
    append_parameter(out,
                     prm,
                     {"CahnHilliard boundary conditions", subsection},
                     "contact angle");
  }

  append_parameter(out, prm, {"Nonlinear solver"}, "tolerance");
  append_parameter(out, prm, {"Nonlinear solver"}, "divergence_tolerance");
  append_parameter(out, prm, {"Nonlinear solver"}, "max_iterations");
  append_parameter(out, prm, {"Nonlinear solver"}, "enable_line_search");
  append_parameter(out, prm, {"Nonlinear solver"}, "analytic_jacobian");
  append_parameter(out,
                   prm,
                   {"Nonlinear solver", "reassembly heuristic"},
                   "decrease tolerance");

  append_parameter(out, prm, {"Linear solver", "linear elasticity"}, "method");
  append_parameter(out,
                   prm,
                   {"Linear solver", "linear elasticity"},
                   "tolerance");
  append_parameter(out,
                   prm,
                   {"Linear solver", "linear elasticity"},
                   "max iterations");
  append_parameter(out,
                   prm,
                   {"Linear solver", "linear elasticity"},
                   "ilu fill level");
  append_parameter(out, prm, {"Linear solver", "linear elasticity"}, "reuse");

  return out.str();
}

template <int dim>
void ParameterReader<dim>::check_parameters() const
{
  // Pressure nullspace
  AssertThrow(
    !(bc_data.fix_pressure_constant && bc_data.enforce_zero_mean_pressure),
    ExcMessage(
      "\n Both fixing a pressure DoF *and* enforcing zero-mean pressure "
      "may be ill-posed. Please choose one or the other."));

  bool has_strong_pressure_bc = false;
  for (const auto &[id, bc] : fluid_bc)
    if (bc.type == BoundaryConditions::Type::dirichlet_pressure)
    {
      has_strong_pressure_bc = true;
      break;
    }

  AssertThrow(
    !(has_strong_pressure_bc && bc_data.fix_pressure_constant),
    ExcMessage(
      "Incompatible pressure constraints: a Dirichlet pressure boundary "
      "condition is prescribed while 'fix pressure constant = true'. "
      "Disable 'fix pressure constant' when pressure is imposed "
      "on a boundary with a Dirichlet condition."));
  AssertThrow(
    !(has_strong_pressure_bc && bc_data.enforce_zero_mean_pressure),
    ExcMessage(
      "Incompatible pressure constraints: a Dirichlet pressure boundary "
      "condition is prescribed while 'enforce zero mean pressure = true'. "
      "Disable 'enforce zero mean pressure' when pressure is imposed "
      "on a boundary with a Dirichlet condition."));

  /**
   * Do not allow to apply both an exact pressure field and a zero-mean
   * constraint, as in general, these conditions won't agree. As an alternative,
   * we could *not* enforce the zero-mean constraint and apply only the exact
   * pressure field, but that would bypass an option, which seems ill advised.
   */
  if (mms.set_field_as_solution.count("pressure") > 0)
    AssertThrow(
      !(bc_data.enforce_zero_mean_pressure &&
        mms.set_field_as_solution.at("pressure")),
      ExcMessage(
        "\n You are trying to enforce zero mean pressure, while also setting "
        "an "
        "exact pressure field. This is not compatible in general, so this is "
        "currently disallowed."));

  // Initial conditions
  AssertThrow(
    !(initial_conditions.set_to_mms && !mms_param.enable),
    ExcMessage(
      "\n The initial conditions should be prescribed by the manufactured "
      "solution, but either no manufactured solution was provided or it was "
      "not enabled."));

  if (!time_integration.is_steady() && !initial_conditions.set_to_mms &&
      mms_param.enable)
  {
    throw std::runtime_error(
      "\n A manufactured solution is prescribed, but the initial conditions "
      "for "
      "this unsteady problem are "
      "not set to be prescribed by this solution. Set \"set to mms = true\" "
      "for the initial conditions.");
  }

  // Postprocessing
  if (postprocessing.slices.enable)
  {
    AssertThrow(dim == 3,
                ExcMessage("Boundary slicing is only available in 3D"));
    if (postprocessing.slices.compute_forces_on_slices)
      AssertThrow(postprocessing.forces.enable,
                  ExcMessage("Forces computation must be enabled to compute "
                             "forces on slices of a given boundary"));
  }

  // FSI
  if (!fsi.enable_coupling)
  {
    for (const auto &[id, bc] : pseudosolid_bc)
      AssertThrow(
        bc.type != BoundaryConditions::Type::coupled_to_fluid,
        ExcMessage(
          "\n A pseudosolid boundary condition is set to \"coupled_to_fluid\", "
          "but the fluid-structure interaction coupling was not enabled."));
  }
  if (fsi.enable_coupling)
  {
    bool at_least_one_coupled_boundary = false;
    for (const auto &[id, bc] : pseudosolid_bc)
      if (bc.type == BoundaryConditions::Type::coupled_to_fluid)
      {
        at_least_one_coupled_boundary = true;
        break;
      }
    AssertThrow(at_least_one_coupled_boundary,
                ExcMessage(
                  "\n Fluid-structure interaction coupling is enabled, but no "
                  "pseudosolid "
                  "boundary condition is set to \"coupled_to_fluid\"."));
  }

  // Linear elasticity
  AssertThrow(
    !(linear_elasticity.enable_source_term_on_current_mesh && mms_param.enable),
    ExcMessage(
      "The parameter file specifies that the linear elasticity solver should "
      "evaluate the given source term on the current mesh (not the reference "
      "mesh), but a convergence study with a manufactured solution should also "
      "be run. This is not compatible, as the source term for the linear "
      "elasticity equation and based on the manufactured solution is expected "
      "to be evaluated on the reference mesh."));

  const auto cache_mode =
    linear_elasticity.presolved_mesh_position_cache.mode;
  const bool cache_enabled =
    cache_mode !=
    Parameters::LinearElasticity::PresolvedMeshPositionCache::Mode::off;

  AssertThrow(
    !(cache_enabled && checkpoint_restart.restart),
    ExcMessage("The presolved mesh position cache is an initial-condition "
               "cache and cannot be combined with a full checkpoint restart."));
  AssertThrow(
    !(cache_enabled && !linear_elasticity.use_as_presolver &&
      !linear_elasticity.chns_presolver_enable),
    ExcMessage("The presolved mesh position cache is enabled, but neither "
               "'use as presolver' nor 'CHNS presolver/enable' is true. "
               "Enable a presolver and let the cache mode decide whether it "
               "is loaded or recomputed."));
  AssertThrow(
    !(cache_enabled &&
      linear_elasticity.presolved_mesh_position_cache.filename.empty()),
    ExcMessage("The presolved mesh position cache filename cannot be empty."));

  // Mesh adaptation
  if (mesh.adaptation.with_metric_based_adaptation())
    if (time_integration.is_steady())
      AssertThrow(time_integration.n_time_intervals == 1,
                  ExcMessage(
                    "When solving for steady-state solution, a single time "
                    "subinterval is expected."));
}

template class ParameterReader<2>;
template class ParameterReader<3>;
