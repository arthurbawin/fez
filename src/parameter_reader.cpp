
#include <parameter_reader.h>

template <int dim>
void ParameterReader<dim>::check_parameters() const
{
  // Pressure nullspace
  AssertThrow(
    !(bc_data.fix_pressure_constant && bc_data.enforce_zero_mean_pressure),
    ExcMessage("Both fixing a pressure DoF *and* enforcing zero-mean pressure "
               "may be ill-posed. Please choose one or the other."));

  // Initial conditions
  if (initial_conditions.set_to_mms && !mms_param.enable)
  {
    throw std::runtime_error(
      "The initial conditions should be prescribed by the manufactured "
      "solution, but either no manufactured solution was provided or it was "
      "not enabled.");
  }
  if (time_integration.scheme !=
        Parameters::TimeIntegration::Scheme::stationary &&
      !initial_conditions.set_to_mms && mms_param.enable)
  {
    throw std::runtime_error(
      "A manufactured solution is prescribed, but the initial conditions for "
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
          "A pseudosolid boundary condition is set to \"coupled_to_fluid\", "
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
    AssertThrow(
      at_least_one_coupled_boundary,
      ExcMessage(
        "Fluid-structure interaction coupling is enabled, but no pseudosolid "
        "boundary condition is set to \"coupled_to_fluid\"."));
  }

  // MMS
  // if constexpr (dim == 3)
  //   if (mms_param.enable)
  //     AssertThrow(mms_param.use_deal_ii_cube_mesh ||
  //                   mms_param.use_deal_ii_holed_plate_mesh,
  //                 ExcMessage(
  //                   "There seems to be a bug when deal.II's function parses a
  //                   " "transfinite cube mesh from Gmsh. Until this is figured
  //                   " "out, 3D convergence studies should be run with \"use "
  //                   "dealii cube mesh = true\"."));
}

template class ParameterReader<2>;
template class ParameterReader<3>;
