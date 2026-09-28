#include <deal.II/grid/grid_generator.h>
#include <deal.II/numerics/vector_tools.h>
#include <incompressible_chns_solver.h>

#include "../tests.h"

namespace
{
  template <bool moving>
  class State : public Function<2>
  {
  public:
    State(const unsigned int components,
          const double       velocity_factor,
          const double       position_factor)
      : Function<2>(components)
      , velocity_factor(velocity_factor)
      , position_factor(position_factor)
    {}

    double value(const Point<2> &p, const unsigned int c) const override
    {
      if (c == 0)
        return velocity_factor * (1. + p[0] + 2. * p[1]);
      if constexpr (moving)
      {
        if (c == 3)
          return position_factor * p[0];
        if (c == 4)
          return p[1];
      }
      if (c == (moving ? 5 : 3))
        return 0.1 + 0.2 * p[0];
      return 0.;
    }

  private:
    const double velocity_factor;
    const double position_factor;
  };

  template <bool moving, bool enlarged = false>
  class TestSolver : public CHNSSolver<2, moving, enlarged>
  {
  public:
    using CHNSSolver<2, moving, enlarged>::CHNSSolver;

    void check()
    {
      this->initialize();
      GridGenerator::subdivided_hyper_cube(*this->triangulation, 4);
      this->setup_dofs();
      this->setup_mappings();
      this->create_scratch_data();
      this->create_zero_constraints();
      this->create_nonzero_constraints();
      this->create_sparsity_pattern();

      // The accepted geometry stretches x by two. The Newton iterate has a
      // different geometry and velocity, so using its mapping or fields fails.
      set_present(1., 2.);
      for (auto &previous : *this->previous_solutions)
        previous = *this->present_solution;
      set_present(7., 5.);
      this->prepare_timestep();
      const double expected = expected_maximum(1., moving ? 2. : 1.);
      check_reference(expected);

      // WorkStream copies must retain the runtime value, rather than reset it
      // to the default used when a new scratch object is constructed.
      const auto copied_scratch = *this->scratch_data;
      AssertThrow(std::abs(
                    copied_scratch.profile_correction_reference_mobility -
                    expected) < 1e-12,
                  ExcMessage("Scratch copy lost the frozen mobility"));

      ConditionalOStream quiet(std::cout, false);
      this->time_handler.advance(quiet);
      set_present(11., 3.);
      this->assemble_rhs();
      check_reference(expected);
      this->time_handler.last_nonlinear_solver_converged = false;
      AssertThrow(!this->time_handler.is_timestep_accepted(
                    *this->present_solution, *this->previous_solutions),
                  ExcMessage("The fixture must reject its first attempt"));
      this->time_handler.advance(quiet);
      set_present(13., 4.);
      this->assemble_rhs();
      check_reference(expected);

      // A newly accepted state updates the reference at the next step only.
      set_present(2., 3.);
      this->time_handler.rotate_solutions(*this->present_solution,
                                          *this->previous_solutions);
      check_reference(expected);
      this->prepare_timestep();
      check_reference(expected_maximum(2., moving ? 3. : 1.));

      // A signed parsed mobility is reduced in absolute value; zero remains
      // zero, without a positive floor for the profile correction.
      this->param.cahn_hilliard.mobility_model =
        Parameters::CahnHilliard<2>::MobilityModel::degenerate;
      this->prepare_timestep();
      check_reference(0.75);
      this->param.cahn_hilliard.mobility_model =
        Parameters::CahnHilliard<2>::MobilityModel::constant;
      this->param.cahn_hilliard.mobility = 0.;
      this->prepare_timestep();
      check_reference(0.);
    }

    void check_stationary()
    {
      this->initialize();
      GridGenerator::subdivided_hyper_cube(*this->triangulation, 4);
      this->setup_dofs();
      this->setup_mappings();
      this->create_scratch_data();
      AssertThrow(this->previous_solutions->empty(), ExcInternalError());
      set_present(1., 2.);
      this->prepare_timestep();
      check_reference(expected_maximum(1., moving ? 2. : 1.));
    }

  private:
    void set_present(const double velocity_factor, const double position_factor)
    {
      VectorTools::interpolate(
        *this->fixed_mapping,
        *this->dof_handler,
        State<moving>(this->dof_handler->get_fe().n_components(),
                      velocity_factor,
                      position_factor),
        this->local_evaluation_point);
      *this->present_solution = this->local_evaluation_point;
      this->evaluation_point  = this->local_evaluation_point;
    }

    double expected_maximum(const double velocity_factor,
                            const double position_factor) const
    {
      // For QGauss(4), the global maximum of u_x=1+x+2y is in the upper
      // right cell. This expectation uses the prescribed affine fields, not
      // the mobility evaluator or the mapping used by the production hook.
      double largest_q = 0.;
      for (const auto &point : this->quadrature->get_points())
        largest_q = std::max(largest_q, point[0]);
      const double coordinate = (3. + largest_q) / 4.;
      const double raw =
        velocity_factor * (1. + 3. * coordinate) * 0.2 / position_factor;
      return std::sqrt(raw * raw + 0.01 * 0.01);
    }

    void check_reference(const double expected) const
    {
      AssertThrow(
        std::abs(this->scratch_data->profile_correction_reference_mobility -
                 expected) < 1e-12,
        ExcMessage(
          "Profile reference must be the global accepted-state mobility"));
    }
  };
} // namespace

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize   mpi(argc, argv, 1);
  Parameters::BoundaryConditionsData bc;
  ParameterHandler                   prm;
  ParameterReader<2>                 param(bc);
  param.declare(prm);
  std::istringstream input(R"(
subsection Mesh
  subsection Adaptation
    set enable = true
    set strategy = local refinement
  end
end
subsection Time integration
  set scheme = BDF1
  set t_initial = 0
  set t_end = 1
  set dt = 0.1
  subsection Adaptation
    set enable = true
  end
end
subsection Output
  set write vtu results = false
end
subsection FiniteElements
  set use quads = true
end
subsection Physical properties
  set number of pseudosolids = 1
  subsection Pseudosolid 0
    subsection lame lambda
      set Function expression = 1
    end
    subsection lame mu
      set Function expression = 1
    end
  end
end
subsection Cahn Hilliard
  set mobility model = adaptative_mobility
  set adaptive mobility n = 0.75
  set adaptive mobility m = 0
  set adaptive mobility delta = 0.01
  set surface tension = 1
  set interface thickness = 1
  set interface profile correction = profile
  subsection degenerate mobility
    set Function expression = -0.75
  end
end
)");
  prm.parse_input(input);
  param.read(prm);
  param.bc_data.fix_pressure_constant      = false;
  param.bc_data.enforce_zero_mean_pressure = false;
  TestSolver<false>(param).check();
  TestSolver<true>(param).check();
  TestSolver<true, true>(param).check();
  param.time_integration.scheme =
    Parameters::TimeIntegration::Scheme::stationary;
  param.time_integration.adaptation.enable = false;
  TestSolver<true, true>(param).check_stationary();
  initlog();
  deallog << "Global profile mobility uses accepted ALE state and survives "
             "scratch copies and retries"
          << std::endl;
}
