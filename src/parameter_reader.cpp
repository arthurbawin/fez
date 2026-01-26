
#include <parameter_reader.h>

template <int dim>
void ParameterReader<dim>::check_parameters(ParameterHandler &prm) const
{
  // Pressure nullspace
  AssertThrow(
    !(bc_data.fix_pressure_constant && bc_data.enforce_zero_mean_pressure),
    ExcMessage(
      "\n Both fixing a pressure DoF *and* enforcing zero-mean pressure "
      "may be ill-posed. Please choose one or the other."));

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
}

template class ParameterReader<2>;
template class ParameterReader<3>;
