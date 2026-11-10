
#include <parameter_reader.h>

template <int dim>
void ParameterReader<dim>::check_parameters() const
{
  // Pressure nullspace
  // FIXME: these bools will be combined in a single parameter
  AssertThrow(
    !(bc_data.fix_pressure_constant && bc_data.enforce_zero_mean_pressure),
    ExcMessage(
      "\n Both fixing a pressure DoF *and* enforcing zero-mean pressure "
      "may be ill-posed. Please choose one or the other."));

  AssertThrow(
    !(bc_data.pressure_null_space &&
      (bc_data.fix_pressure_constant || bc_data.enforce_zero_mean_pressure)),
    ExcMessage("\n 'pressure null space' replaces 'fix pressure constant' and "
               "'enforce zero mean pressure': please choose only one way of "
               "handling the pressure constant."));

  // A pinned pressure dof leaves a near-null mode in the system, whose
  // eigenvalue decreases with the mesh size and which iterative solvers
  // resolve poorly: on fine meshes, the pressure converges erratically, or not
  // at all. Iterative solvers should use the pressure null space instead.
  if (bc_data.fix_pressure_constant &&
      linear_solver.count(SolverInfo::SolverType::main_physics) > 0)
    AssertThrow(
      !linear_solver.at(SolverInfo::SolverType::main_physics).is_iterative(),
      ExcMessage(
        "\n 'fix pressure constant' is only allowed with the direct_mumps "
        "linear solver. With an iterative solver, set 'pressure null space = "
        "true' instead."));

  // With the pressure null space, the system matrix is singular
  if (bc_data.pressure_null_space &&
      linear_solver.count(SolverInfo::SolverType::main_physics) > 0)
    AssertThrow(
      linear_solver.at(SolverInfo::SolverType::main_physics).is_iterative(),
      ExcMessage(
        "\n 'pressure null space' requires an iterative linear solver, as "
        "the system matrix is singular. With direct_mumps, set 'fix pressure "
        "constant = true' instead."));

  // The field split Schur preconditioner assumes a zero pressure block, which
  // the PSPG stabilization makes nonzero: stabilized formulations should use
  // a monolithic preconditioner (amg)
  if (stabilization.enable_supg &&
      linear_solver.count(SolverInfo::SolverType::main_physics) > 0)
    AssertThrow(
      !linear_solver.at(SolverInfo::SolverType::main_physics).use_fieldsplit(),
      ExcMessage("\n The fieldsplit_schur_cc preconditioner requires 'enable "
                 "supg = false': it assumes a zero pressure-pressure block, "
                 "which the PSPG stabilization makes nonzero. Use the amg "
                 "preconditioner with stabilization."));

  // The constant pressure is in the null space only if the normal velocity is
  // prescribed on the whole boundary
  // FIXME: should maybe use a function that returns whether a BC sets the
  // normal velocity
  if (bc_data.pressure_null_space)
    for (const auto &[id, bc] : fluid_bc)
    {
      using Type = BoundaryConditions::Type;
      const bool prescribes_normal_velocity =
        bc.type == Type::input_function || bc.type == Type::dirichlet_mms ||
        bc.type == Type::no_slip || bc.type == Type::slip ||
        bc.type == Type::velocity_mms || bc.type == Type::velocity_flux_mms;
      AssertThrow(prescribes_normal_velocity,
                  ExcMessage(
                    "'pressure null space = true' requires the normal "
                    "velocity to be prescribed (strongly) on the whole "
                    "boundary, but boundary " +
                    std::to_string(id) +
                    " has a boundary condition that does not, and which "
                    "already determines the pressure constant."));
    }

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
  if (mms.set_field_as_solution.count("pressure") > 0)
    AssertThrow(!(bc_data.pressure_null_space &&
                  mms.set_field_as_solution.at("pressure")),
                ExcMessage("\n 'pressure null space' is incompatible with "
                           "setting the exact pressure field, which "
                           "constrains all pressure dofs."));

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

  // Mesh adaptation
  if (mesh.adaptation.with_metric_based_adaptation())
  {
    if (time_integration.is_steady())
      AssertThrow(time_integration.n_time_intervals == 1,
                  ExcMessage(
                    "When solving for steady-state solution, a single time "
                    "subinterval is expected."));


    // If adapting with a fixed-point loop and modifying some parameters every
    // few iterations, check that these parameters exist in the relevant
    // function objects before running any computation.
    const auto &fpu = mesh.adaptation.metric.fixed_point_updates;

    if (fpu.chns_interface_thickness.enable)
    {
      // Check for existence of interface thickness variable in tracer initial
      // condition, which will need to be updated.
      const auto &constants =
        initial_conditions.initial_chns_tracer_callback->get_constants();
      const auto &s = fpu.chns_interface_thickness.constant_name;
      if (constants.count(s) == 0)
      {
        std::ostringstream oss;
        oss << "The prescribed updates in the fixed-point mesh adaptation loop "
               "require replacing the constant \""
            << s
            << "\" in the initial condition of the CHNS tracer, but that "
               "function does not contain this constant. Instead, it contains "
               "the following constants:\n";
        for (const auto &[name, value] : constants)
          oss << "  " << name << " = " << value << '\n';
        AssertThrow(false, ExcMessage(oss.str()));
      }
    }
  }

  if (mesh.adaptation.with_metric_based_adaptation() || metrics.always_compute)
    AssertThrow(
      bc_data.n_metric_fields > 0,
      ExcMessage(
        "A Riemannian metric for mesh adaptation should be computed, but no "
        "metric field parameters were provided (set number = 0)."));
  if (mesh.adaptation.with_tree_based_adaptation())
  {
    // Only available for quads/hexes
    AssertThrow(finite_elements.use_quads,
                ExcMessage(
                  "Mesh adaptation using p4est and deal.II's routines is only "
                  "available for quad/hex finite elements. Howwver, mesh "
                  "adaptation with remeshing is available for simplices using "
                  "the \"riemannian metric\" strategy."));

    // Not available with the transient-fixed point method for now: limit to a
    // single time subinterval
    AssertThrow(time_integration.n_time_intervals == 1,
                ExcMessage(
                  "The transient fixed-point method is not implemented for "
                  "tree-based meshes. Please use a single time interval."));
  }
}

template class ParameterReader<2>;
template class ParameterReader<3>;
