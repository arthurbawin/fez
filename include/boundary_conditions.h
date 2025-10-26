#ifndef BOUNDARY_CONDITIONS_H
#define BOUNDARY_CONDITIONS_H

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

/**
 *
 */
namespace BoundaryConditions
{
  /**
   * Physics for which boundary conditions are available
   */
  enum class PhysicsType
  {
    fluid,
    pseudosolid
  };

  /**
   * Available boundary conditions
   */
  enum class Type
  {
    none,

    // Flow
    input_function,
    outflow,      // Do nothing
    no_slip,      // Enforce given functions
    weak_no_slip, // Check that lagrange mult is defined, couple
    slip,         // enforce no_flux

    // Pseudo_solid
    fixed,            // Enforce 0
    coupled_to_fluid, // Couple to lagrange mult
    no_flux
  };

  /**
   * Base class for boundary conditions
   */
  class BoundaryCondition
  {
  public:
    // Physics associated to this boundary condition (type and string)
    PhysicsType physics_type;
    std::string physics_str;

    // Type of boundary condition
    Type type;

    // Id of the associated boundary
    types::boundary_id id;

    // The Gmsh name of the boundary entity
    std::string gmsh_name;

    /**
     * Declare the parameters common to all boundary conditions
     */
    virtual void declare_parameters(ParameterHandler &prm);
    virtual void read_parameters(ParameterHandler  &prm,
                                 const unsigned int boundary_id);
  };

  template <int dim>
  class FluidBC : public BoundaryCondition
  {
  public:
    void declare_parameters(ParameterHandler &prm) override;
    void read_parameters(ParameterHandler  &prm,
                         const unsigned int boundary_id) override;
  };

  template <int dim>
  class PseudosolidBC : public BoundaryCondition
  {
  public:
    void declare_parameters(ParameterHandler &prm) override;
    void read_parameters(ParameterHandler  &prm,
                         const unsigned int boundary_id) override;
  };

  // FIXME: template the declare and read functions below
  template <int dim>
  void declare_fluid_boundary_conditions(ParameterHandler          &prm,
                                         std::vector<FluidBC<dim>> &fluid_bc)
  {
    prm.enter_subsection("Fluid boundary conditions");
    {
      // The number was already parsed in a first dry run
      prm.declare_entry(
        "number",
        "0", // Utilities::int_to_string(number_of_boundary_conditions),
        Patterns::Integer(),
        "Number of fluid boundary conditions");

      for (unsigned int i = 0; i < fluid_bc.size(); ++i)
      {
        prm.enter_subsection("boundary " + std::to_string(i));
        {
          fluid_bc[i].declare_parameters(prm);
        }
        prm.leave_subsection();
      }
    }
    prm.leave_subsection();
  }

  template <int dim>
  void read_fluid_boundary_conditions(ParameterHandler          &prm,
                                      std::vector<FluidBC<dim>> &fluid_bc)
  {
    /**
     * FIXME: If the number of bc actually added in the parameter file is
     * greater than the specified number, deal.ii throws a parse error.
     Ideally,
     * we should be able to return an error here if the user specified too
     few
     * boundary conditions.
     */
    prm.enter_subsection("Fluid boundary conditions");
    {
      for (unsigned int i = 0; i < fluid_bc.size(); ++i)
      {
        prm.enter_subsection("boundary " + std::to_string(i));
        {
          fluid_bc[i].read_parameters(prm, i);
        }
        prm.leave_subsection();
      }
    }
    prm.leave_subsection();
  }

  template <int dim>
  void FluidBC<dim>::declare_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::declare_parameters(prm);
    prm.declare_entry(
      "type",
      "none",
      Patterns::Selection(
        "none|input_function|outflow|no_slip|weak_no_slip|slip"),
      "Type of fluid boundary condition");
  }

  template <int dim>
  void FluidBC<dim>::read_parameters(ParameterHandler  &prm,
                                     const unsigned int boundary_id)
  {
    BoundaryCondition::read_parameters(prm, boundary_id);
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
    if (parsed_type == "none")
      throw std::runtime_error(
        "Fluid boundary condition for boundary " + std::to_string(boundary_id) +
        " is set to \"none\".\n"
        "Either you specified this type by mistake, or the number of \n"
        "prescribed fluid boundary conditions is smaller than "
        "the specified \"number\" field.");
  }

  template <int dim>
  void declare_pseudosolid_boundary_conditions(ParameterHandler          &prm,
                                         std::vector<PseudosolidBC<dim>> &pseudosolid_bc)
  {
    prm.enter_subsection("Pseudosolid boundary conditions");
    {
      prm.declare_entry(
        "number",
        "0", // Utilities::int_to_string(number_of_boundary_conditions),
        Patterns::Integer(),
        "Number of pseudosolid boundary conditions");

      for (unsigned int i = 0; i < pseudosolid_bc.size(); ++i)
      {
        prm.enter_subsection("boundary " + std::to_string(i));
        {
          pseudosolid_bc[i].declare_parameters(prm);
        }
        prm.leave_subsection();
      }
    }
    prm.leave_subsection();
  }

  template <int dim>
  void read_pseudosolid_boundary_conditions(ParameterHandler          &prm,
                                      std::vector<PseudosolidBC<dim>> &pseudosolid_bc)
  {
    prm.enter_subsection("Pseudosolid boundary conditions");
    {
      for (unsigned int i = 0; i < pseudosolid_bc.size(); ++i)
      {
        prm.enter_subsection("boundary " + std::to_string(i));
        {
          pseudosolid_bc[i].read_parameters(prm, i);
        }
        prm.leave_subsection();
      }
    }
    prm.leave_subsection();
  }

  template <int dim>
  void PseudosolidBC<dim>::declare_parameters(ParameterHandler &prm)
  {
    BoundaryCondition::declare_parameters(prm);
    prm.declare_entry("type",
                      "none",
                      Patterns::Selection(
                        "none|fixed|coupled_to_fluid|no_flux"),
                      "Type of pseudosolid boundary condition");
  }

  template <int dim>
  void PseudosolidBC<dim>::read_parameters(ParameterHandler  &prm,
                                           const unsigned int boundary_id)
  {
    BoundaryCondition::read_parameters(prm, boundary_id);
    physics_type                  = PhysicsType::pseudosolid;
    physics_str                   = "pseudosolid";
    const std::string parsed_type = prm.get("type");
    if (parsed_type == "fixed")
      type = Type::fixed;
    if (parsed_type == "coupled_to_fluid")
      type = Type::coupled_to_fluid;
    if (parsed_type == "no_flux")
      type = Type::no_flux;
    if (parsed_type == "none")
      throw std::runtime_error(
        "Pseudosolid boundary condition for boundary " +
        std::to_string(boundary_id) +
        " is set to \"none\".\n"
        "Either you specified this type by mistake, or the number of \n"
        "prescribed pseudosolid boundary conditions is smaller than "
        "the specified \"number\" field.");
  }
} // namespace BoundaryConditions

#endif