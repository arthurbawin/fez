#ifndef PARAMETER_READER_H
#define PARAMETER_READER_H

#include <parameters.h>

using namespace dealii;

/**
 *
 */
template <int dim>
class ParameterReader
{
public:
  Parameters::DummyDimension     dummy_dimension;
  Parameters::Mesh               mesh;
  Parameters::Output             output;
  Parameters::FiniteElements     finite_elements;
  Parameters::PhysicalProperties physical_properties;
  Parameters::FSI                fsi;
  Parameters::TimeIntegration    time_integration;
  Parameters::NonLinearSolver    nonlinear_solver;

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
  }
};

#endif