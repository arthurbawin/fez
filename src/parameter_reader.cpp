
#include <parameter_reader.h>

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

  if (postprocessing.flow_diagnostics.enable)
  {
    AssertThrow(postprocessing.flow_diagnostics.compute_vorticity ||
                  postprocessing.flow_diagnostics.compute_qcriterion,
                ExcMessage("Flow diagnostics are enabled, but neither "
                          "vorticity nor Q criterion is requested."));
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

  // Mesh concentration
  if (mesh_concentration.enable)
  {
    AssertThrow(mesh_concentration.h_min > 0.0,
                ExcMessage("Mesh concentration: h min must be > 0."));

    AssertThrow(mesh_concentration.h_max > 0.0,
                ExcMessage("Mesh concentration: h max must be > 0."));

    AssertThrow(mesh_concentration.h_min <= mesh_concentration.h_max,
                ExcMessage("Mesh concentration: h min must be <= h max."));

    AssertThrow(mesh_concentration.G0 > 0.0,
                ExcMessage("Mesh concentration: G0 must be > 0."));

    AssertThrow(mesh_concentration.velocity_ref >
                  mesh_concentration.velocity_min,
                ExcMessage("Mesh concentration: velocity ref must be > velocity min."));

    AssertThrow(mesh_concentration.velocity_max >
                  mesh_concentration.velocity_ref,
                ExcMessage("Mesh concentration: velocity max must be > velocity ref."));

    AssertThrow(mesh_concentration.velocity_gradient_ref >
                  mesh_concentration.velocity_gradient_min,
                ExcMessage("Mesh concentration: velocity gradient ref must be "
                           "> velocity gradient min."));

    AssertThrow(mesh_concentration.velocity_gradient_max >
                  mesh_concentration.velocity_gradient_ref,
                ExcMessage("Mesh concentration: velocity gradient max must be "
                           "> velocity gradient ref."));

    AssertThrow(mesh_concentration.exponent > 0.0,
                ExcMessage("Mesh concentration: exponent must be > 0."));

    AssertThrow(mesh_concentration.eps > 0.0,
                ExcMessage("Mesh concentration: epsilon must be > 0."));

    AssertThrow(mesh_concentration.max_pressure >= 0.0,
                ExcMessage("Mesh concentration: max pressure must be >= 0."));

    AssertThrow(mesh_concentration.normal_weight >= 0.0,
                ExcMessage("Mesh concentration: normal weight must be >= 0."));

    AssertThrow(mesh_concentration.tangential_weight >= 0.0,
                ExcMessage("Mesh concentration: tangential weight must be >= 0."));

    AssertThrow(mesh_concentration.ramp_time >= 0.0,
                ExcMessage("Mesh concentration: ramp time must be >= 0."));
  }
}

template class ParameterReader<2>;
template class ParameterReader<3>;
