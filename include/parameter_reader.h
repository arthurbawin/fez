#ifndef PARAMETER_READER_H
#define PARAMETER_READER_H

#include <boundary_conditions.h>
#include <parameters.h>

using namespace dealii;

/**
 *
 */
template <int dim>
class ParameterReader
{
public:
  //
  // Parameters
  //
  Parameters::DummyDimension            dummy_dimension;
  Parameters::Mesh               mesh;
  Parameters::Output             output;
  Parameters::FiniteElements     finite_elements;
  Parameters::PhysicalProperties physical_properties;
  Parameters::FSI                fsi;
  Parameters::TimeIntegration    time_integration;
  Parameters::NonLinearSolver    nonlinear_solver;

  //
  // Boundary conditions
  //
  Parameters::BoundaryConditionsCount                 bc_count;
  std::vector<BoundaryConditions::FluidBC<dim>>       fluid_bc;
  std::vector<BoundaryConditions::PseudosolidBC<dim>> pseudosolid_bc;

public:
  /**
   * Constructor
   */
  ParameterReader(const Parameters::BoundaryConditionsCount &bc_count)
    : bc_count(bc_count)
  {
    fluid_bc.resize(bc_count.n_fluid_bc);
    pseudosolid_bc.resize(bc_count.n_pseudosolid_bc);
  }

public:
  void declare(ParameterHandler &prm)
  {
    dummy_dimension.declare_parameters(prm);
    mesh.declare_parameters(prm);
    output.declare_parameters(prm);
    finite_elements.declare_parameters(prm);
    physical_properties.declare_parameters(prm);
    fsi.declare_parameters(prm);
    time_integration.declare_parameters(prm);
    nonlinear_solver.declare_parameters(prm);
    BoundaryConditions::declare_fluid_boundary_conditions(prm, fluid_bc);
    BoundaryConditions::declare_pseudosolid_boundary_conditions(prm,
                                                                pseudosolid_bc);
  }

  void read(ParameterHandler &prm)
  {
    dummy_dimension.read_parameters(prm);
    mesh.read_parameters(prm);
    output.read_parameters(prm);
    finite_elements.read_parameters(prm);
    physical_properties.read_parameters(prm);
    fsi.read_parameters(prm);
    time_integration.read_parameters(prm);
    nonlinear_solver.read_parameters(prm);
    BoundaryConditions::read_fluid_boundary_conditions(prm, fluid_bc);
    BoundaryConditions::read_pseudosolid_boundary_conditions(prm,
                                                             pseudosolid_bc);
  }
};

#endif