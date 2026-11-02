#include <generic_solver.h>

#include <cfenv>
#include <functional>
#include <limits>

#include "../tests.h"

// Exercise the real Newton controller with a prescribed correction and scalar
// residuals. Each rank owns one identical unknown; norms are independent of
// the MPI size. No finite-element assembly is needed to test line-search logic.
class ScalarProblem : public GenericSolver<LA::ParVectorType>
{
public:
  ScalarProblem(const Parameters::NonLinearSolver   &nonlinear,
                const Parameters::TimeIntegration   &time,
                const Parameters::Output            &output,
                const std::function<double(double)> &residual)
    : GenericSolver(output,
                    nonlinear,
                    Parameters::Timer{},
                    Parameters::Mesh{},
                    time,
                    Parameters::MMS{},
                    SolverInfo::SolverType::elasticity)
    , residual(residual)
  {
    IndexSet owned(mpi_size);
    owned.add_index(mpi_rank);
    for (auto *v : {&solution,
                    &evaluation_point,
                    &local_evaluation_point,
                    &newton_update,
                    &system_rhs})
      v->reinit(owned, mpi_communicator);
    constraints.close();
    pcout.set_condition(false);
  }

  void run() override {}
  void assemble_matrix() override {}
  void assemble_rhs() override
  {
    const double value =
      residual(evaluation_point[mpi_rank]) / std::sqrt(mpi_size);
    // Set deliberately invalid residuals through PETSc, bypassing deal.II's
    // finite-value assertion in its vector element accessor in Debug builds.
    AssertThrow(VecSet(system_rhs, value) == 0, ExcInternalError());
  }
  void               solve_linear_system() override { newton_update = 1.; }
  LA::ParVectorType &get_present_solution() override { return solution; }
  AffineConstraints<double> &get_nonzero_constraints() override
  {
    return constraints;
  }

  void check_state(const double expected) const
  {
    for (const auto *v :
         {&solution, &evaluation_point, &local_evaluation_point})
      AssertThrow(std::abs((*v)[mpi_rank] - expected) < 1e-14,
                  ExcMessage(
                    "Newton left an unexpected solution/evaluation point"));
  }

private:
  LA::ParVectorType                   solution;
  AffineConstraints<double>           constraints;
  const std::function<double(double)> residual;
};

struct Case
{
  const char                   *name;
  std::function<double(double)> residual;
  double                        expected;
  bool                          converges;
  bool                          line_search          = true;
  unsigned int                  max_iterations       = 0;
  double                        divergence_tolerance = 1e10;
};

void check_case(const Case &test, const bool adaptive)
{
  Parameters::NonLinearSolver nonlinear{};
  nonlinear.tolerance            = 1e-10;
  nonlinear.divergence_tolerance = test.divergence_tolerance;
  nonlinear.max_iterations =
    test.line_search ? test.max_iterations : 1; // One correction by default.
  nonlinear.enable_line_search      = test.line_search;
  nonlinear.verbosity               = Parameters::Verbosity::quiet;
  nonlinear.reassembly_decrease_tol = 0.1;

  Parameters::TimeIntegration time{};
  time.scheme           = adaptive ? Parameters::TimeIntegration::Scheme::BDF1 :
                                     Parameters::TimeIntegration::Scheme::stationary;
  time.n_time_intervals = 1;
  time.dt               = 0.1;
  time.t_end            = 1.;
  time.adaptation.enable = adaptive;
  time.adaptation.strategy =
    Parameters::TimeIntegration::Adaptation::AdaptationStrategy::CFL;
  time.adaptation.target_cfl             = 1.;
  time.adaptation.min_timestep           = 1e-6;
  time.adaptation.max_timestep           = 1.;
  time.adaptation.max_timestep_increase  = 2.;
  time.adaptation.max_timestep_reduction = 0.1;
  Parameters::Output output{};
  ScalarProblem      solver(nonlinear, time, output, test.residual);
  TimeHandler        handler(time);
  if (adaptive)
  {
    handler.set_time_interval(0);
    handler.advance(solver.pcout);
    handler.set_max_cfl(1.);
  }

  std::vector<LA::ParVectorType> previous(1);
  previous[0] = solver.get_present_solution();
  bool threw  = false;
  try
  {
    NewtonSolver<LA::ParVectorType> newton(nonlinear, &solver);
    newton.solve(handler);
  }
  catch (const std::runtime_error &)
  {
    threw = true;
  }
  AssertThrow(threw == (!test.converges && !adaptive),
              ExcMessage("Unexpected Newton failure/return"));
  solver.check_state(test.expected);

  if (adaptive)
  {
    AssertThrow(handler.is_timestep_accepted(solver.get_present_solution(),
                                             previous) == test.converges,
                ExcMessage("Newton reported an incorrect convergence status"));
    if (!test.converges)
    {
      AssertThrow(solver.get_present_solution().l2_norm() == 0.,
                  ExcInternalError());
      AssertThrow(handler.get_n_rejected_steps() == 1, ExcInternalError());
      handler.advance(solver.pcout);
      AssertThrow(std::abs(handler.get_current_timestep() - 0.05) < 1e-14,
                  ExcMessage("A failed Newton step must halve the time step"));
    }
  }
  if (solver.mpi_rank == 0)
    deallog << test.name << (adaptive ? " adaptive" : " steady") << " OK"
            << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
#ifdef DEAL_II_HAVE_FP_EXCEPTIONS
  // Invalid residuals are intentional inputs to the controller in this test.
  fedisableexcept(FE_INVALID | FE_DIVBYZERO);
#endif
  deal_II_exceptions::disable_abort_on_exception();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  const double      nan   = std::numeric_limits<double>::quiet_NaN();
  const double      inf   = std::numeric_limits<double>::infinity();
  std::vector<Case> cases = {
    {"NaN then root",
     [=](double x) { return x > .5 ? nan : x - .5; },
     .5,
     true},
    {"two Inf then root",
     [=](double x) { return x >= .5 ? inf : x - .25; },
     .25,
     true},
    {"finite backtrack",
     [](double x) { return x == 0. ? 10. : (x == 1. ? 2. : 3.); },
     1.,
     false},
    {"invalid gap before backtrack",
     [=](double x) {
       return x == 0. ? 10. : (x == 1. ? 2. : (x == .5 ? nan : 3.));
     },
     1.,
     false},
    {"finite decreasing trials",
     [](double x) { return x == 0. ? 10. : 4. + 4. * x; },
     .125,
     false},
    {"finite trial then invalid trials",
     [=](double x) { return x == 0. ? 10. : (x == 1. ? 2. : nan); },
     1.,
     false},
    {"all NaN trials", [=](double x) { return x == 0. ? 1. : nan; }, 0., false},
    {"all Inf trials", [=](double x) { return x == 0. ? 1. : inf; }, 0., false},
    {"invalid initial residual", [=](double) { return inf; }, 0., false},
    {"invalid undamped update",
     [=](double x) { return x == 0. ? 1. : nan; },
     0.,
     false,
     false}};
  // After an Armijo step, exhausted finite trials must refresh the residual.
  // The true norm 15 exceeds the divergence threshold; a stale norm of 5
  // would incorrectly permit another correction to the artificial root.
  cases.push_back({"fresh residual after fallback",
                   [](double x) {
                     if (x == 0.)
                       return 100.;
                     if (x == 1.)
                       return 5.;
                     if (x == 2.)
                       return 40.;
                     if (x == 1.5)
                       return 30.;
                     if (x == 1.25)
                       return 20.;
                     if (x == 1.125)
                       return 15.;
                     return 0.;
                   },
                   1.125,
                   false,
                   true,
                   3,
                   10.});
  unsigned int failures = 0;
  for (const auto &test : cases)
    for (const bool adaptive : {false, true})
      try
      {
        check_case(test, adaptive);
      }
      catch (const std::exception &e)
      {
        ++failures;
        if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cerr << test.name << " adaptive=" << adaptive << ": " << e.what()
                    << std::endl;
      }
  return failures == 0 ? 0 : 1;
}
