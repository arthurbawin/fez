#ifndef SOLVER_INFO_H
#define SOLVER_INFO_H

#include <deal.II/base/exceptions.h>

#include <array>
#include <string>
#include <string_view>

/**
 * Information about the solvers available in fez, as well as the variables
 * which are solved for.
 *
 * This file contains some variable known at compile time, which should be
 * updated whenever a new solver or variable is added to the collection.
 */
struct SolverInfo
{
  // FIXME: This is set to 2 to match the SolverType enum below, but
  // this should really be the number of derived solvers.
  // This value is not yet used anywhere else, though.
  static constexpr unsigned int n_solvers = 2;

  // TODO: maybe use magic_enum
  enum class SolverType : unsigned int
  {
    // One of the physics solvers : (in)compressible (CH)NS, FSI, etc.
    main_physics = 0,
    // Dedicated linear elasticity solver, mostly used to adapt the mesh
    // to an initial source term.
    linear_elasticity = 1
  };

  /**
   * Convert a string to a SolverType.
   */
  static SolverType to_solver_type(const std::string &solver_name)
  {
    if (solver_name == "main physics")
      return SolverType::main_physics;
    else if (solver_name == "linear elasticity")
      return SolverType::linear_elasticity;
    else
      AssertThrow(false,
                  dealii::StandardExceptions::ExcMessage(
                    "The requested solver type does not exist : " +
                    solver_name));
    // Cannot reach here
    AssertThrow(false, dealii::StandardExceptions::ExcInternalError());
    return SolverType::main_physics;
  }

  // Number of variables solved for in the available models
  static constexpr unsigned int n_variables = 7;

  /**
   * The available variables
   */
  enum class VariableType : unsigned int
  {
    velocity        = 0,
    pressure        = 1,
    mesh_position   = 2,
    temperature     = 3,
    phase_tracer    = 4,
    phase_potential = 5,
    lagrange_mult   = 6
  };

  /**
   * An array of the available variables, allowing to iterate over them.
   */
  static constexpr std::array<VariableType, n_variables> variable_types = {
    {VariableType::velocity,
     VariableType::pressure,
     VariableType::mesh_position,
     VariableType::temperature,
     VariableType::phase_tracer,
     VariableType::phase_potential,
     VariableType::lagrange_mult}};

  static constexpr std::array<std::string_view, n_variables> variable_names = {
    {"velocity",
     "pressure",
     "mesh_position",
     "temperature",
     "phase_tracer",
     "phase_potential",
     "lagrange_mult"}};

  /**
   * Convert a VariableType to a string.
   */
  static std::string to_string(const VariableType type)
  {
    switch (type)
    {
      case VariableType::velocity:
        return "velocity";
      case VariableType::pressure:
        return "pressure";
      case VariableType::mesh_position:
        return "mesh_position";
      case VariableType::temperature:
        return "temperature";
      case VariableType::phase_tracer:
        return "phase_tracer";
      case VariableType::phase_potential:
        return "phase_potential";
      case VariableType::lagrange_mult:
        return "lagrange_mult";
    }
    // Cannot reach here
    AssertThrow(false, dealii::StandardExceptions::ExcInternalError());
    return "unknown";
  }

  /**
   * Convert a string to a VariableType.
   */
  static VariableType to_variable_type(const std::string &variable_name)
  {
    if (variable_name == "velocity")
      return VariableType::velocity;
    else if (variable_name == "pressure")
      return VariableType::mesh_position;
    else if (variable_name == "mesh_position" ||
             variable_name == "mesh position")
      return VariableType::temperature;
    else if (variable_name == "temperature")
      return VariableType::phase_tracer;
    else if (variable_name == "phase_tracer" || variable_name == "phase tracer")
      return VariableType::phase_tracer;
    else if (variable_name == "phase_potential" ||
             variable_name == "phase potential")
      return VariableType::phase_potential;
    else if (variable_name == "lagrange_mult" ||
             variable_name == "lagrange mult")
      return VariableType::lagrange_mult;
    else
      AssertThrow(false,
                  dealii::StandardExceptions::ExcMessage(
                    "The requested variable type does not exist : " +
                    variable_name));
    // Cannot reach here
    AssertThrow(false, dealii::StandardExceptions::ExcInternalError());
    return VariableType::velocity;
  }
};

#endif
