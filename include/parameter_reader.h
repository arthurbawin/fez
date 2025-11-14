#ifndef PARAMETER_READER_H
#define PARAMETER_READER_H

#include <boundary_conditions.h>
#include <initial_conditions.h>
#include <manufactured_solution.h>
#include <parameters.h>
#include <source_terms.h>

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
  Parameters::DummyDimension     dummy_dimension;
  Parameters::Timer              timer;
  Parameters::Mesh               mesh;
  Parameters::Output             output;
  Parameters::FiniteElements     finite_elements;
  Parameters::PhysicalProperties physical_properties;
  Parameters::FSI                fsi;
  Parameters::TimeIntegration    time_integration;
  Parameters::LinearSolver       linear_solver;
  Parameters::NonLinearSolver    nonlinear_solver;
  Parameters::MMS                mms_param;

  //
  // Initial and boundary conditions
  //
  Parameters::InitialConditions<dim> initial_conditions;
  Parameters::BoundaryConditionsData bc_data;
  std::map<types::boundary_id, BoundaryConditions::FluidBC<dim>> fluid_bc;
  std::map<types::boundary_id, BoundaryConditions::PseudosolidBC<dim>>
    pseudosolid_bc;

  //
  // Source terms
  //
  Parameters::SourceTerms<dim> source_terms;

  //
  // Manufactured solution
  //
  ManufacturedSolution::ManufacturedSolution<dim> mms;

public:
  /**
   * Constructor
   */
  ParameterReader(const Parameters::BoundaryConditionsData &bc_data)
    : bc_data(bc_data)
  {}

public:
  void check_parameters() const;

  void declare(ParameterHandler &prm)
  {
    dummy_dimension.declare_parameters(prm);
    timer.declare_parameters(prm);
    mesh.declare_parameters(prm);
    output.declare_parameters(prm);
    finite_elements.declare_parameters(prm);
    physical_properties.declare_parameters(prm);
    fsi.declare_parameters(prm);
    time_integration.declare_parameters(prm);
    linear_solver.declare_parameters(prm);
    nonlinear_solver.declare_parameters(prm);
    initial_conditions.declare_parameters(prm);
    bc_data.declare_parameters(prm);
    BoundaryConditions::declare_fluid_boundary_conditions<dim>(
      prm, bc_data.n_fluid_bc);
    BoundaryConditions::declare_pseudosolid_boundary_conditions<dim>(
      prm, bc_data.n_pseudosolid_bc);
    source_terms.declare_parameters(prm);
    mms_param.declare_parameters(prm);
    mms.declare_parameters(prm);
  }

  void read(ParameterHandler &prm)
  {
    dummy_dimension.read_parameters(prm);
    timer.read_parameters(prm);
    mesh.read_parameters(prm);
    output.read_parameters(prm);
    finite_elements.read_parameters(prm);
    physical_properties.read_parameters(prm);
    fsi.read_parameters(prm);
    time_integration.read_parameters(prm);
    linear_solver.read_parameters(prm);
    nonlinear_solver.read_parameters(prm);
    initial_conditions.read_parameters(prm);
    bc_data.read_parameters(prm);
    BoundaryConditions::read_fluid_boundary_conditions(prm,
                                                       bc_data.n_fluid_bc,
                                                       fluid_bc);
    BoundaryConditions::read_pseudosolid_boundary_conditions(
      prm, bc_data.n_pseudosolid_bc, pseudosolid_bc);
    source_terms.read_parameters(prm);
    mms_param.read_parameters(prm);
    mms.read_parameters(prm);

    check_parameters();
  }
};

#endif