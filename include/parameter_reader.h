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
  Parameters::DummyDimension          dummy_dimension;
  Parameters::Timer                   timer;
  Parameters::Mesh                    mesh;
  Parameters::Output                  output;
  Parameters::FiniteElements          finite_elements;
  Parameters::PhysicalProperties<dim> physical_properties;
  Parameters::FSI                     fsi;
  Parameters::TimeIntegration         time_integration;
  Parameters::CheckpointRestart       checkpoint_restart;
  Parameters::LinearSolver            linear_solver;
  Parameters::NonLinearSolver         nonlinear_solver;
  Parameters::CahnHilliard<dim>       cahn_hilliard;
  Parameters::MMS                     mms_param;
  Parameters::Debug                   debug;

  //
  // Initial and boundary conditions
  //
  Parameters::InitialConditions<dim> initial_conditions;
  Parameters::BoundaryConditionsData bc_data;
  std::map<types::boundary_id, BoundaryConditions::FluidBC<dim>> fluid_bc;
  std::map<types::boundary_id, BoundaryConditions::PseudosolidBC<dim>>
    pseudosolid_bc;
  std::map<types::boundary_id, BoundaryConditions::CahnHilliardBC<dim>>
    cahn_hilliard_bc;
  std::map<types::boundary_id, BoundaryConditions::HeatBC<dim>> heat_bc;

  //
  // Source terms
  //
  Parameters::SourceTerms<dim> source_terms;

  //
  // Manufactured solution
  //
  ManufacturedSolutions::ManufacturedSolution<dim> mms;

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
    checkpoint_restart.declare_parameters(prm);
    linear_solver.declare_parameters(prm);
    nonlinear_solver.declare_parameters(prm);
    initial_conditions.declare_parameters(prm);
    bc_data.declare_parameters(prm);
    BoundaryConditions::declare_boundary_conditions<
      BoundaryConditions::FluidBC<dim>>(prm, bc_data.n_fluid_bc, "Fluid");
    BoundaryConditions::declare_boundary_conditions<
      BoundaryConditions::PseudosolidBC<dim>>(prm,
                                              bc_data.n_pseudosolid_bc,
                                              "Pseudosolid");
    BoundaryConditions::declare_boundary_conditions<
      BoundaryConditions::CahnHilliardBC<dim>>(prm,
                                               bc_data.n_cahn_hilliard_bc,
                                               "CahnHilliard");
    BoundaryConditions::declare_boundary_conditions<
      BoundaryConditions::HeatBC<dim>>(prm, bc_data.n_heat_bc, "Heat");
    cahn_hilliard.declare_parameters(prm);
    source_terms.declare_parameters(prm);
    mms_param.declare_parameters(prm);
    mms.declare_parameters(prm);
    debug.declare_parameters(prm);
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
    checkpoint_restart.read_parameters(prm);
    linear_solver.read_parameters(prm);
    nonlinear_solver.read_parameters(prm);
    initial_conditions.read_parameters(prm);
    bc_data.read_parameters(prm);
    BoundaryConditions::read_boundary_conditions<
      BoundaryConditions::FluidBC<dim>>(prm,
                                        bc_data.n_fluid_bc,
                                        "Fluid",
                                        fluid_bc);
    BoundaryConditions::read_boundary_conditions(prm,
                                                 bc_data.n_pseudosolid_bc,
                                                 "Pseudosolid",
                                                 pseudosolid_bc);
    BoundaryConditions::read_boundary_conditions(prm,
                                                 bc_data.n_cahn_hilliard_bc,
                                                 "CahnHilliard",
                                                 cahn_hilliard_bc);
    BoundaryConditions::read_boundary_conditions(prm,
                                                 bc_data.n_heat_bc,
                                                 "Heat",
                                                 heat_bc);
    cahn_hilliard.read_parameters(prm);
    source_terms.read_parameters(prm);
    mms_param.read_parameters(prm);
    mms.read_parameters(prm);
    debug.read_parameters(prm);

    check_parameters();
  }
};

#endif