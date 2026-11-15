#include <compressible_ns_solver.h>
#include <deal.II/numerics/vector_tools.h>

#include <array>

#include "../tests.h"

namespace
{
  class InspectableSolver : public CompressibleNSSolver<2>
  {
  public:
    using CompressibleNSSolver<2>::CompressibleNSSolver;
    unsigned int residual_assemblies = 0;

    void assemble_rhs() override
    {
      ++residual_assemblies;
      CompressibleNSSolver<2>::assemble_rhs();
    }

    void check(const double expected_u) const
    {
      Functions::ConstantFunction<2> expected(
        std::vector<double>{expected_u, 0., 0., 0.});
      LA::ParVectorType reference;
      reference.reinit(this->locally_owned_dofs, MPI_COMM_WORLD);
      VectorTools::interpolate(*this->fixed_mapping,
                               *this->dof_handler,
                               expected,
                               reference);
      double error = 0.;
      for (const auto i : this->locally_owned_dofs)
      {
        const double value = (*this->present_solution)[i];
        AssertThrow(std::isfinite(value), ExcMessage("Nonfinite solution"));
        error = std::max(error, std::abs(value - reference[i]));
      }
      AssertThrow(Utilities::MPI::max(error, MPI_COMM_WORLD) < 1e-9,
                  ExcMessage("Uniform relaxation mismatch: " +
                             std::to_string(error)));
      AssertThrow(std::abs(this->time_handler.current_time - .1) < 1e-12 &&
                    this->time_handler.current_time_iteration == 1 &&
                    residual_assemblies > 0,
                  ExcMessage("The solver did not execute one BDF1 step"));
      AssertThrow(Utilities::MPI::min(this->locally_owned_dofs.n_elements(),
                                      MPI_COMM_WORLD) > 0,
                  ExcMessage("Each MPI rank must own DoFs"));
    }
  };

  ParameterReader<2> make_parameters(const bool enabled)
  {
    Parameters::BoundaryConditionsData bc{};
    bc.n_fluid_bc = bc.n_heat_bc = 4;
    ParameterReader<2> param(bc);
    ParameterHandler   prm;
    param.declare(prm);
    prm.parse_input_from_string(R"(
subsection Dimension
  set dimension = 2
end
subsection Timer
  set enable timer = false
end
subsection Output
  set write vtu results = false
  set output directory = ./
end
subsection Mesh
  set verbosity = quiet
  set use dealii cube mesh = true
  set refinement level = 2
end
subsection Time integration
  set verbosity = quiet
  set scheme = BDF1
  set dt = 0.1
  set t_initial = 0
  set t_end = 0.1
end
subsection Nonlinear solver
  set verbosity = quiet
  set tolerance = 1e-11
  set max_iterations = 20
  set analytic_jacobian = true
end
subsection Linear solver
  subsection main physics
    set verbosity = quiet
    set method = direct_mumps
  end
end
subsection FiniteElements
  set Velocity degree = 2
  set Pressure degree = 1
  set Temperature degree = 2
end
subsection Physical properties
  set number of fluids = 1
  subsection Fluid 0
    set density = 2
    set dynamic viscosity = 0.2
    set thermal conductivity = 0.3
    set heat capacity at constant pressure = 20
    set pressure reference = 100
    set temperature reference = 10
  end
end
subsection Initial conditions
  set to mms = false
  subsection velocity
    set Function expression = 1; 0
  end
  subsection pressure
    set Function expression = 0
  end
  subsection temperature
    set Function expression = 0
  end
end
subsection Fluid boundary conditions
  set number = 4
  set fix pressure constant = false
end
subsection Heat boundary conditions
  set number = 4
end
subsection Sponge Layer
  subsection outflow
    set x_start = -2
    set x_end = -1
    set sigma_max = 2
    set u = 0
  end
end
)");
    // The whole unit cube lies on the constant-sigma plateau.
    prm.enter_subsection("Sponge Layer");
    prm.enter_subsection("outflow");
    prm.set("enable", enabled ? "true" : "false");
    prm.leave_subsection();
    prm.leave_subsection();
    const std::array<std::string, 4> names{
      {"x_min", "x_max", "y_min", "y_max"}};
    prm.enter_subsection("Fluid boundary conditions");
    for (unsigned int i = 0; i < 4; ++i)
    {
      prm.enter_subsection("boundary " + std::to_string(i));
      prm.set("id", std::to_string(i + 1));
      prm.set("name", names[i]);
      prm.set("type", "input_function");
      prm.enter_subsection("u");
      prm.set("Function expression", enabled ? "1/(1+2*t)" : "1");
      prm.leave_subsection();
      prm.leave_subsection();
    }
    prm.leave_subsection();
    prm.enter_subsection("Heat boundary conditions");
    for (unsigned int i = 0; i < 4; ++i)
    {
      prm.enter_subsection("boundary " + std::to_string(i));
      prm.set("id", std::to_string(i + 1));
      prm.set("name", names[i]);
      prm.set("type", "no_flux");
      prm.leave_subsection();
    }
    prm.leave_subsection();
    param.read(prm);
    return param;
  }

  void test(const bool enabled)
  {
    InspectableSolver solver(make_parameters(enabled));
    solver.run();
    // du/dt + sigma*u = 0: backward Euler gives u1 = u0/(1+sigma*dt).
    solver.check(enabled ? 5. / 6. : 1.);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      deallog << "uniform BDF1 " << (enabled ? "enabled" : "disabled") << " OK"
              << std::endl;
  }
} // namespace

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  try
  {
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      initlog();
    test(false);
    test(true);
  }
  catch (const std::exception &error)
  {
    std::cerr << error.what() << std::endl;
    return 1;
  }
}
