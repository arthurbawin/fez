
#include <boundary_conditions.h>

using namespace BoundaryConditions;

void BoundaryCondition::declare_parameters(ParameterHandler &prm)
{
  prm.declare_entry("id",
                    "-1",
                    Patterns::Integer(),
                    "Gmsh tag of the physical entity associated to this boundary");
  prm.declare_entry("name",
                    "",
                    Patterns::Anything(),
                    "Name of the Gmsh physical entity associated to this boundary");
}

void BoundaryCondition::read_parameters(ParameterHandler &prm, const unsigned int /*boundary_id*/)
{
  id        = prm.get_integer("id");;
  gmsh_name = prm.get("name");
}