#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <incompressible_chns_solver.h>

#include "../tests.h"

namespace
{
  class State : public Function<2>
  {
  public:
    explicit State(const bool direction = false)
      : Function<2>(5)
      , direction(direction)
    {}

    double value(const Point<2> &p, const unsigned int component) const override
    {
      if (direction)
        return component == 3 ? p[0] * (1. - p[0]) * p[1] * (1. - p[1]) : 0.;
      switch (component)
      {
        case 0:
          return 0.2 + p[1];
        case 1:
          return -0.1 * p[0];
        case 2:
          return p[0] * p[1];
        // Include pure phases and overshoots in the nodal data.
        case 3:
          return 1.01 * std::tanh((p[0] + 0.3 * p[1] - 0.53) / 0.16);
        case 4:
          return p[0] * p[0] + 0.4 * p[1];
      }
      return 0.;
    }

  private:
    const bool direction;
  };

  class TestSolver : public CHNSSolver<2>
  {
  public:
    using CHNSSolver<2>::CHNSSolver;

    void check()
    {
      initialize();
      set_time();
      GridGenerator::subdivided_hyper_cube(*triangulation, 2);
      for (const auto &cell : triangulation->active_cell_iterators())
        if (cell->is_locally_owned() && cell->center()[0] < 0.5 &&
            cell->center()[1] < 0.5)
          cell->set_refine_flag();
      triangulation->execute_coarsening_and_refinement();
      setup_dofs();
      setup_mappings();
      create_scratch_data();
      create_solver_specific_constraints_data();
      create_zero_constraints();
      create_nonzero_constraints();
      create_sparsity_pattern();
      AssertThrow(Utilities::MPI::sum(zero_constraints.n_constraints(),
                                      mpi_communicator) > 0,
                  ExcMessage(
                    "The fixture must contain hanging-node constraints"));

      VectorTools::interpolate(*fixed_mapping,
                               *dof_handler,
                               State(),
                               local_evaluation_point);
      nonzero_constraints.distribute(local_evaluation_point);
      *present_solution = local_evaluation_point;
      evaluation_point  = local_evaluation_point;
      for (auto &previous : *previous_solutions)
        previous = local_evaluation_point;
      prepare_timestep();
      time_handler.advance(pcout);
      set_time();

      LA::ParVectorType base(locally_owned_dofs, mpi_communicator);
      LA::ParVectorType independent(locally_owned_dofs, mpi_communicator);
      LA::ParVectorType physical(locally_owned_dofs, mpi_communicator);
      LA::ParVectorType analytic(locally_owned_dofs, mpi_communicator);
      LA::ParVectorType finite_difference(locally_owned_dofs, mpi_communicator);
      base = local_evaluation_point;
      VectorTools::interpolate(*fixed_mapping,
                               *dof_handler,
                               State(true),
                               independent);
      zero_constraints.set_zero(independent);
      physical = independent;
      zero_constraints.distribute(physical);
      assemble_matrix();
      system_matrix.vmult(analytic, independent);
      zero_constraints.set_zero(analytic);

      const double step      = 1e-6;
      local_evaluation_point = base;
      local_evaluation_point.add(step, physical);
      evaluation_point = local_evaluation_point;
      assemble_rhs();
      finite_difference      = system_rhs;
      local_evaluation_point = base;
      local_evaluation_point.add(-step, physical);
      evaluation_point = local_evaluation_point;
      assemble_rhs();
      finite_difference -= system_rhs;
      finite_difference *= -0.5 / step; // RHS is minus the residual.
      zero_constraints.set_zero(finite_difference);
      finite_difference -= analytic;
      const double relative_error =
        finite_difference.l2_norm() / std::max(1., analytic.l2_norm());
      AssertThrow(relative_error < 2e-7,
                  ExcMessage("Constrained reconstructed Jacobian mismatch: " +
                             std::to_string(relative_error)));
      deallog << "Constrained PC/FC Jacobian on a hanging-node mesh OK"
              << std::endl;
    }
  };
} // namespace

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  initlog();
  Parameters::BoundaryConditionsData bc;
  ParameterHandler                   prm;
  ParameterReader<2>                 param(bc);
  param.declare(prm);
  std::istringstream input(R"(
subsection Timer
  set enable timer = false
end
subsection Output
  set write vtu results = false
end
subsection Mesh
  subsection Adaptation
    set enable = true
    set strategy = local refinement
  end
end
subsection Time integration
  set scheme = BDF1
  set dt = 0.01
  set t_end = 0.02
  set verbosity = quiet
end
subsection FiniteElements
  set use quads = true
  set Velocity degree = 2
  set Pressure degree = 1
  set Tracer degree = 2
  set Potential degree = 2
end
subsection Cahn Hilliard
  set CHNS model = abels
  set interface profile correction = profile_flux
  set interface thickness = 0.1
  set surface tension = 1
  set mobility model = adaptative_mobility
end
subsection Physical properties
  set number of fluids = 2
  subsection Fluid 0
    set density = 1
  end
  subsection Fluid 1
    set density = 3
  end
end
)");
  prm.parse_input(input);
  param.read(prm);
  param.bc_data.fix_pressure_constant      = false;
  param.bc_data.enforce_zero_mean_pressure = false;
  TestSolver(param).check();
}
