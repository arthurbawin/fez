#ifndef FEZ_TEST_PRESOLVER_PARAMETERS_H
#define FEZ_TEST_PRESOLVER_PARAMETERS_H

#include <parameter_reader.h>

#include <sstream>

// Shared small physical problem for the real-solver presolver tests.
template <int dim>
ParameterReader<dim>
presolver_test_parameters(const std::string &overrides = "")
{
  Parameters::BoundaryConditionsData boundaries{};
  boundaries.n_fluid_bc           = boundaries.n_pseudosolid_bc =
    boundaries.n_cahn_hilliard_bc = 2 * dim;
  ParameterReader<dim> param(boundaries);
  ParameterHandler     prm;
  param.declare(prm);
  std::ostringstream input;
  input << "subsection Dimension\nset dimension = " << dim << "\nend\n";
  input << R"(subsection Timer
 set enable timer = false
end
subsection Output
 set write vtu results = false
end
subsection Mesh
 set dealii preset mesh = rectangle
 set refinement level = 1
 set dealii mesh parameters = )";
  input << (dim == 2 ? "3,3:0,0:1,1:true" : "3,3,3:0,0,0:1,1,1:true");
  input << R"(
end
subsection Time integration
 set verbosity = quiet
 set t_initial = 0
 set t_end = 0.03
 set dt = 0.01
 set scheme = BDF2
 set bdf start method = initial condition
end
subsection FiniteElements
 set use quads = )";
  // Pseudosolid face geometry is currently implemented on simplices in 3D.
  input << (dim == 2 ? "true" : "false");
  input << R"(
 set Velocity degree = 2
 set Pressure degree = 1
 set Mesh position degree = 1
 set Tracer degree = 1
 set Potential degree = 1
end
subsection Nonlinear solver
 set verbosity = quiet
 set tolerance = 1e-9
 set max_iterations = 25
 set enable_line_search = true
end
subsection Linear solver
 subsection main physics
  set verbosity = quiet
  set method = direct_mumps
 end
 subsection elasticity
  set verbosity = quiet
  set method = direct_mumps
 end
end
subsection Physical properties
 set number of fluids = 2
 subsection Fluid 0
  set density = 1
  set kinematic viscosity = 1
 end
 subsection Fluid 1
  set density = 1
  set kinematic viscosity = 1
 end
 set number of pseudosolids = 1
 subsection Pseudosolid 0
  set constitutive model = neo hookean
  subsection lame lambda
   set Function expression = 1
  end
  subsection lame mu
   set Function expression = 1
  end
 end
end
subsection Cahn Hilliard
 set mobility = 0.1
 set surface tension = 1
 set interface thickness = 0.2
 set mff source term = chns form
 set mff physics compression factor = 0.5
 set mff transport factor = 0.1
 set mff regularization gamma = 0.8
 set use presolver = true
end
subsection Elasticity
 subsection presolver
  set initial compression multiplier = 0.1
  set continuation steps = 3
 end
end
subsection Initial conditions
 subsection cahn hilliard tracer
  set Function expression = tanh((x-0.4)/0.2)
 end
 subsection velocity
  set Function expression = )";
  input << (dim == 2 ? "0.1*x*(1-x)*y*(1-y);0" :
                       "0.1*x*(1-x)*y*(1-y)*z*(1-z);0;0");
  input << "\nend\nend\n";
  for (const std::string section : {"Fluid boundary conditions",
                                    "Pseudosolid boundary conditions",
                                    "CahnHilliard boundary conditions"})
  {
    input << "subsection " << section << "\nset number = " << 2 * dim << '\n';
    if (section == "Fluid boundary conditions")
      input << "set fix pressure constant = false\nset enforce zero mean "
               "pressure = true\n";
    for (unsigned int b = 0; b < 2 * dim; ++b)
      input << "subsection boundary " << b << "\nset id = " << b
            << "\nset name = "
            << std::vector<std::string>{"x_min",
                                        "x_max",
                                        "y_min",
                                        "y_max",
                                        "z_min",
                                        "z_max"}[b]
            << "\nset type = "
            << (section == "Fluid boundary conditions" ?
                  "no_slip" :
                  (section == "Pseudosolid boundary conditions" ? "fixed" :
                                                                  "no_flux"))
            << "\nend\n";
    input << "end\n";
  }
  prm.parse_input_from_string(input.str());
  if (!overrides.empty())
    prm.parse_input_from_string(overrides);
  param.read(prm);
  return param;
}

#endif
