#ifndef UTILITIES_H
#define UTILITIES_H

#include <deal.II/base/parameter_handler.h>

using namespace dealii;

/**
 * Perform a dry run to read the run-time problem dimension set in the
 * "Dimension" block of the given parameter file.
 */
unsigned int read_problem_dimension(const std::string &parameter_file)
{
  ParameterHandler prm;

  // Declare
  prm.enter_subsection("Dimension");
  prm.declare_entry("dimension",
                    "2",
                    Patterns::Integer(2, 3),
                    "Problem dimension (2 or 3)",
                    true);
  prm.leave_subsection();

  // Read only this structure from the file
  // Parse will fail if the "Dimension" block is not specified
  prm.parse_input(parameter_file,
                  /*last_line=*/"",
                  /*skip_undefined=*/true,
                  /*assert_mandatory_entries_are_found=*/true);

  // Parse
  prm.enter_subsection("Dimension");
  unsigned int dim = prm.get_integer("dimension");
  prm.leave_subsection();

  return dim;
}

/**
 * Perform a dry run to read the number of boundary conditions of each type.
 */
void read_number_of_boundary_conditions(
  const std::string                   &parameter_file,
  Parameters::BoundaryConditionsData &bc_data)
{
  ParameterHandler prm;

  // Declare all possible boundary conditions.
  // They do not all need to be present in the parameter file.
  prm.enter_subsection("Fluid boundary conditions");
  {
    prm.declare_entry("number",
                      "0",
                      Patterns::Integer(),
                      "Number of boundary conditions for the flow problem "
                      "(Navier-Stokes equations)");
  }
  prm.leave_subsection();

  prm.enter_subsection("Pseudosolid boundary conditions");
  {
    prm.declare_entry("number",
                      "0",
                      Patterns::Integer(),
                      "Number of boundary conditions for the pseudosolid mesh "
                      "movement problem");
  }
  prm.leave_subsection();

  prm.enter_subsection("CahnHilliard boundary conditions");
  {
    prm.declare_entry("number",
                      "0",
                      Patterns::Integer(),
                      "Number of boundary conditions for two-phase flows with the Cahn-Hilliard Navier-Stokes model");
  }
  prm.leave_subsection();

  // Read only these structures from the file
  prm.parse_input(parameter_file, /*last_line=*/"", /*skip_undefined=*/true);

  // Parse
  prm.enter_subsection("Fluid boundary conditions");
  bc_data.n_fluid_bc = prm.get_integer("number");
  prm.leave_subsection();

  prm.enter_subsection("Pseudosolid boundary conditions");
  bc_data.n_pseudosolid_bc = prm.get_integer("number");
  prm.leave_subsection();

  prm.enter_subsection("CahnHilliard boundary conditions");
  bc_data.n_cahn_hilliard_bc = prm.get_integer("number");
  prm.leave_subsection();
}

#endif