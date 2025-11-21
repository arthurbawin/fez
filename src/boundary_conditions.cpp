
#include <boundary_conditions.h>

namespace BoundaryConditions
{
  void BoundaryCondition::declare_parameters(ParameterHandler &prm)
  {
    prm.declare_entry(
      "id",
      "-1",
      Patterns::Integer(),
      "Gmsh tag of the physical entity associated to this boundary");
    prm.declare_entry(
      "name",
      "",
      Patterns::Anything(),
      "Name of the Gmsh physical entity associated to this boundary");
  }

  void BoundaryCondition::read_parameters(ParameterHandler &prm)
  {
    id        = prm.get_integer("id");
    gmsh_name = prm.get("name");
  }

  template <int dim>
  void FluidBC<dim>::declare_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::declare_parameters(prm);
    prm.declare_entry("type",
                      "none",
                      Patterns::Selection(
                        "none|input_function|outflow|no_slip|weak_no_slip|slip|"
                        "velocity_mms|velocity_flux_mms|open_mms"),
                      "Type of fluid boundary condition");

    // Imposed functions, if any
    prm.enter_subsection("u");
    u->declare_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("v");
    v->declare_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("w");
    w->declare_parameters(prm);
    prm.leave_subsection();
  }

  template <int dim>
  void FluidBC<dim>::read_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::read_parameters(prm);
    physics_type                  = PhysicsType::fluid;
    physics_str                   = "fluid";
    const std::string parsed_type = prm.get("type");
    if (parsed_type == "input_function")
      type = Type::input_function;
    if (parsed_type == "outflow")
      type = Type::outflow;
    if (parsed_type == "no_slip")
      type = Type::no_slip;
    if (parsed_type == "weak_no_slip")
      type = Type::weak_no_slip;
    if (parsed_type == "slip")
      type = Type::slip;
    if (parsed_type == "velocity_mms")
      type = Type::velocity_mms;
    if (parsed_type == "velocity_flux_mms")
      type = Type::velocity_flux_mms;
    if (parsed_type == "open_mms")
      type = Type::open_mms;
    if (parsed_type == "none")
      throw std::runtime_error(
        "Fluid boundary condition for boundary " + std::to_string(this->id) +
        " is set to \"none\".\n"
        "Either you specified this type by mistake, or the number of \n"
        "prescribed fluid boundary conditions is smaller than "
        "the specified \"number\" field.");

    prm.enter_subsection("u");
    u->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("v");
    v->parse_parameters(prm);
    prm.leave_subsection();

    prm.enter_subsection("w");
    w->parse_parameters(prm);
    prm.leave_subsection();
  }

  template <int dim>
  void PseudosolidBC<dim>::declare_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::declare_parameters(prm);
    prm.declare_entry("type",
                      "none",
                      Patterns::Selection(
                        "none|fixed|coupled_to_fluid|no_flux|input_function|position_mms"),
                      "Type of pseudosolid boundary condition");
  }

  template <int dim>
  void PseudosolidBC<dim>::read_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::read_parameters(prm);
    physics_type                  = PhysicsType::pseudosolid;
    physics_str                   = "pseudosolid";
    const std::string parsed_type = prm.get("type");
    if (parsed_type == "fixed")
      type = Type::fixed;
    if (parsed_type == "coupled_to_fluid")
      type = Type::coupled_to_fluid;
    if (parsed_type == "no_flux")
      type = Type::no_flux;
    if (parsed_type == "input_function")
      type = Type::input_function;
    if (parsed_type == "position_mms")
      type = Type::position_mms;
    if (parsed_type == "none")
      throw std::runtime_error(
        "Pseudosolid boundary condition for boundary " +
        std::to_string(this->id) +
        " is set to \"none\".\n"
        "Either you specified this type by mistake, or the number of \n"
        "prescribed pseudosolid boundary conditions is smaller than "
        "the specified \"number\" field.");
  }

  template <int dim>
  void CahnHilliardBC<dim>::declare_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::declare_parameters(prm);
    prm.declare_entry("type",
                      "none",
                      Patterns::Selection("none|no_flux"),
                      "Type of Cahn-Hilliard boundary condition");
  }

  template <int dim>
  void CahnHilliardBC<dim>::read_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::read_parameters(prm);
    physics_type                  = PhysicsType::cahn_hilliard;
    physics_str                   = "cahn_hilliard";
    const std::string parsed_type = prm.get("type");
    if (parsed_type == "no_flux")
      type = Type::no_flux;
    if (parsed_type == "none")
      throw std::runtime_error(
        "Cahn-Hilliard boundary condition for boundary " +
        std::to_string(this->id) +
        " is set to \"none\".\n"
        "Either you specified this type by mistake, or the number of \n"
        "prescribed pseudosolid boundary conditions is smaller than "
        "the specified \"number\" field.");
  }

  // Explicit instantiation
  template class FluidBC<2>;
  template class FluidBC<3>;
  template class PseudosolidBC<2>;
  template class PseudosolidBC<3>;
  template class CahnHilliardBC<2>;
  template class CahnHilliardBC<3>;
} // namespace BoundaryConditions