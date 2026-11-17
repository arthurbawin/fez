#include <generic_solver.h>

#include <cfenv>
#include <functional>
#include <limits>

#include "../tests.h"

/**
 * Test the safeguards of the Newton solver on a scalar problem with a
 * prescribed residual:
 *
 * - the nonmonotone line search, which accepts a trial step if its residual
 *   satisfies the Armijo condition relative to the largest of the last
 *   "memory" accepted residuals,
 * - the maximum number of trial steps in the line search,
 * - the rejection of non-finite residuals,
 * - the restoration of the last accepted iterate on failure, both when the
 *   solver throws (steady solve) and when it returns (adaptive time step).
 */

/**
 * Scalar problem solved by the actual Newton solver. Each rank owns one
 * unknown x, and all unknowns have the same value. The residual is a
 * prescribed function r(x), scaled by 1/sqrt(n_ranks) so that its l2 norm is
 * |r(x)| for any number of ranks. The Newton update is always 1, so that the
 * trial steps from an accepted iterate x are x + alpha, with alpha = 1, 1/2,
 * 1/4, etc.
 */
class ScalarProblem : public GenericSolver<LA::ParVectorType>
{
public:
  ScalarProblem(const Parameters::NonLinearSolver &nonlinear,
                const Parameters::TimeIntegration &time,
                std::function<double(double)>      residual)
    : GenericSolver(Parameters::Output{},
                    nonlinear,
                    Parameters::Timer{},
                    Parameters::Mesh{},
                    time,
                    Parameters::MMS{},
                    SolverInfo::SolverType::linear_elasticity)
    , residual(std::move(residual))
  {
    IndexSet owned(mpi_size);
    owned.add_index(mpi_rank);
    for (auto *v : {&local_evaluation_point, &newton_update, &system_rhs})
      v->reinit(owned, mpi_communicator);
    IndexSet relevant(mpi_size);
    relevant.add_range(0, mpi_size);
    solution.reinit(owned, relevant, mpi_communicator);
    evaluation_point.reinit(owned, relevant, mpi_communicator);
    constraints.close();
    pcout.set_condition(false);
  }

  void run() override {}
  void assemble_matrix() override {}
  void assemble_rhs() override
  {
    ++rhs_calls;
    // PETSc permits intentional NaN/Inf inputs even in a Debug build.
    const double r = residual(evaluation_point[mpi_rank]) / std::sqrt(mpi_size);
    AssertThrow(VecSet(system_rhs, r) == 0, ExcInternalError());
  }
  void               solve_linear_system() override { newton_update = 1.; }
  LA::ParVectorType &get_present_solution() override { return solution; }
  AffineConstraints<double> &get_nonzero_constraints() override
  {
    return constraints;
  }

  // Check that the solution and the evaluation point both store the expected
  // value, i.e., the converged or the last accepted iterate. All entries are
  // checked: the owned one and the ghosts, which are the other ranks' unknowns.
  void check_state(const double expected) const
  {
    for (const auto *v : {&solution, &evaluation_point})
      for (unsigned int i = 0; i < mpi_size; ++i)
        AssertThrow(std::abs((*v)[i] - expected) < 1e-12,
                    ExcMessage("Unexpected solution/evaluation point"));
  }

  // Check the norm of the last assembled RHS
  void check_rhs(const double expected) const
  {
    AssertThrow(std::abs(system_rhs.l2_norm() - expected) < 1e-12,
                ExcMessage("Unexpected RHS"));
  }

  LA::ParVectorType             solution;
  AffineConstraints<double>     constraints;
  std::function<double(double)> residual;

  // Number of RHS assemblies
  unsigned int rhs_calls = 0;
};

/**
 * A test case: a prescribed residual, the Newton solver parameters, and the
 * expected outcome of a solve starting from x = 0.
 */
struct Case
{
  const char *name;
  // Prescribed residual r(x)
  std::function<double(double)> residual;
  // Final value of x: the converged iterate on success, or the last accepted
  // iterate on failure
  double expected;
  // Whether the solve is expected to converge
  bool success;
  // Line search parameters
  unsigned int memory          = 1;
  unsigned int backtrack_steps = 4;
  // Expected number of RHS assemblies, not checked if negative
  int  expected_rhs = -1;
  bool line_search  = true;
};

/**
 * Solve @p test from x = 0 and check its outcome. If @p adaptive is false, the
 * problem is steady and the Newton solver throws on failure. Otherwise, it is
 * one step of an adaptive BDF1 time integration: the Newton solver returns on
 * failure, and the time handler must then reject the time step.
 */
void check_case(const Case &test, const bool adaptive)
{
  Parameters::NonLinearSolver nonlinear{};
  nonlinear.tolerance                       = 1e-10;
  nonlinear.divergence_tolerance            = 1e4;
  nonlinear.max_iterations                  = 6;
  nonlinear.line_search.enable              = test.line_search;
  nonlinear.line_search.memory              = test.memory;
  nonlinear.line_search.max_backtrack_steps = test.backtrack_steps;
  nonlinear.verbosity                       = Parameters::Verbosity::quiet;
  Parameters::TimeIntegration time{};
  time.scheme           = adaptive ? Parameters::TimeIntegration::Scheme::BDF1 :
                                     Parameters::TimeIntegration::Scheme::stationary;
  time.dt               = .1;
  time.t_end            = 1.;
  time.n_time_intervals = 1;
  time.adaptation.enable = adaptive;
  time.adaptation.strategy =
    Parameters::TimeIntegration::Adaptation::AdaptationStrategy::CFL;
  time.adaptation.target_cfl             = 1.;
  time.adaptation.min_timestep           = 1e-6;
  time.adaptation.max_timestep           = 1.;
  time.adaptation.max_timestep_increase  = 2.;
  time.adaptation.max_timestep_reduction = .1;

  ScalarProblem solver(nonlinear, time, test.residual);
  TimeHandler   handler(time);
  if (adaptive)
  {
    handler.set_time_interval(0);
    handler.advance(solver.pcout);
    handler.set_max_cfl(1.);
  }
  handler.set_last_nonlinear_solve_status(true);

  // Solve, and return whether the solver threw
  NewtonSolver<LA::ParVectorType> newton(nonlinear, &solver);
  const auto                      solve_throws = [&]() {
    try
    {
      newton.solve(handler);
      return false;
    }
    catch (const std::runtime_error &)
    {
      return true;
    }
  };

  // First solve: the solver throws on failure only for a steady problem
  AssertThrow(solve_throws() == (!test.success && !adaptive),
              ExcMessage("Unexpected Newton failure/return"));

  // The solver must exit with the converged or last accepted iterate
  solver.check_state(test.expected);
  // On success, the RHS is the residual at the converged iterate. On failure,
  // it is the last assembled residual, which is not restored.
  if (test.success)
    solver.check_rhs(std::abs(test.residual(test.expected)));
  if (test.expected_rhs >= 0)
    AssertThrow(solver.rhs_calls == unsigned(test.expected_rhs),
                ExcMessage("Unexpected number of RHS assemblies"));
  // With adaptive time stepping, the time step is accepted only if the Newton
  // solver converged
  if (adaptive)
  {
    std::vector<LA::ParVectorType> previous(1);
    previous[0] = solver.solution;
    AssertThrow(handler.is_timestep_accepted(solver.solution, previous) ==
                  test.success,
                ExcMessage("Incorrect nonlinear convergence status"));
  }

  // After a failure, the same Newton object must be able to solve a new
  // problem, r(x) = root - x, from the restored iterate. The local evaluation
  // point is not restored on failure, so this checks that the solver does not
  // depend on its value.
  if (!test.success)
  {
    const double root = test.expected + 1.;
    solver.residual   = [root](double x) { return root - x; };
    newton.solve(handler);
    solver.check_state(root);
    solver.check_rhs(0.);
  }

  // Reuse the very same Newton object: a previous maximum must not survive
  // into the next nonlinear solve (time step or continuation stage).
  // From x = 0, r = 1 and every trial step yields r = 2. This is rejected
  // with a fresh history, but would be accepted if the last residuals of the
  // previous solve (up to 8 for the chain residual) were still in it.
  if (test.success && test.memory > 1)
  {
    solver.get_local_evaluation_point() = 0.;
    solver.solution                     = solver.get_local_evaluation_point();
    solver.residual = [](double x) { return x == 0. ? 1. : 2.; };
    AssertThrow(solve_throws() == !adaptive,
                ExcMessage("Unexpected failure on second solve"));
    solver.check_state(0.);
  }
  if (solver.mpi_rank == 0)
    deallog << test.name << (test.line_search ? " damped" : " undamped")
            << (adaptive ? " adaptive" : " steady") << " OK" << std::endl;
}

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
#ifdef DEAL_II_HAVE_FP_EXCEPTIONS
  fedisableexcept(FE_INVALID | FE_DIVBYZERO);
#endif
  deal_II_exceptions::disable_abort_on_exception();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();

  // Check the default line search parameters, that they are read from the
  // "Line search" subsection, and that a zero memory is rejected.
  Parameters::NonLinearSolver parameters{};
  ParameterHandler            prm;
  parameters.declare_parameters(prm);
  parameters.read_parameters(prm);
  AssertThrow(parameters.line_search.memory == 3 &&
                parameters.line_search.max_backtrack_steps == 4,
              ExcInternalError());
  prm.enter_subsection("Nonlinear solver");
  prm.enter_subsection("Line search");
  prm.set("memory", "1");
  prm.set("max backtrack steps", "11");
  prm.set("enable", "false");
  prm.leave_subsection();
  prm.leave_subsection();
  parameters.read_parameters(prm);
  AssertThrow(parameters.line_search.memory == 1 &&
                parameters.line_search.max_backtrack_steps == 11 &&
                !parameters.line_search.enable,
              ExcInternalError());
  bool rejected_zero = false;
  prm.enter_subsection("Nonlinear solver");
  try
  {
    prm.enter_subsection("Line search");
    prm.set("memory", "0");
  }
  catch (const std::exception &)
  {
    rejected_zero = true;
  }
  prm.leave_subsection();
  prm.leave_subsection();
  AssertThrow(rejected_zero, ExcMessage("Zero-length memory must be rejected"));
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << "memory parameter OK" << std::endl;
  // Residual equal to values[k] at the full Newton steps x = k, and large at
  // all other points, i.e., at damped steps.
  const auto full_steps = [](const std::vector<double> values) {
    return [values](double x) {
      const auto k = static_cast<std::size_t>(std::round(x));
      return (x == k && k < values.size()) ? values[k] : 100.;
    };
  };

  // The residual rises from 2 to 8 at x = 2 before reaching 0 at x = 4
  const auto chain = full_steps({10., 2., 8., 1., 0.});

  // The residual vanishes only at x = 1/1024, i.e., at the 11th trial step
  const auto tiny_step = [](double x) {
    return x == 0. ? 1. : (x == 1. / 1024. ? 0. : 2.);
  };

  const std::vector<Case> cases = {
    // Monotone line search: the step from x = 1 to 2 raises the residual from
    // 2 to 8 and is rejected, as are all damped steps. Fails at x = 1.
    {"memory 1 rejects increase", chain, 1., false},
    // The reference max(10, 2) = 10 accepts the rise to 8, and the solver
    // reaches the root at x = 4.
    {"memory 3 permits bounded increase", chain, 4., true, 3},
    // Same as the chain, but r(4) = 9. At x = 3, the residual 10 at x = 0 has
    // left the window {2, 8, 1}, so 9 > 8 is rejected. Fails at x = 3.
    {"expired maximum", full_steps({10., 2., 8., 1., 9.}), 3., false, 3},
    // The trials alpha = 1, 1/2, 1/4 yield 100, 90 and 0. The reference stays
    // at the last accepted residual 10 during backtracking: 90 would be
    // accepted if the rejected residual 100 had raised it. Converges at 1/4.
    {"fixed reference",
     [](double x) {
       return x == 0. ? 10. : (x == .25 ? 0. : (x == 1. ? 100. : 90.));
     },
     .25,
     true},
    // With 4 trials (alpha down to 1/8), the line search fails after 1 + 4 RHS
    // assemblies.
    {"four trials stop before tiny step", tiny_step, 0., false, 1, 4, 5},
    // With 11 trials, the solver converges at x = 1/1024 after 1 + 11 RHS
    // assemblies.
    {"eleven trials reach tiny step", tiny_step, 1. / 1024., true, 1, 11, 12}};
  unsigned int failures = 0;
  const auto   run      = [&](const Case &test, const bool adaptive) {
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
  };
  for (const auto &test : cases)
    run(test, false);

  // One injected non-finite residual, independent of mesh validity: fail only
  // after a valid Newton update, and check restoration on both exit paths.
  // The first step reaches x = 1, after which every residual is NaN. Without
  // line search, the NaN residual stops the solver. With line search, all
  // trials are rejected. In both cases, the solver fails at x = 1, either by
  // throwing (steady) or by returning (adaptive).
  const double nan = std::numeric_limits<double>::quiet_NaN();
  Case         nonfinite{"nonfinite last valid iterate",
                 [=](double x) { return x == 0. ? 10. : (x == 1. ? 2. : nan); },
                 1.,
                 false};
  for (const bool line_search : {false, true})
  {
    nonfinite.line_search = line_search;
    for (const bool adaptive : {false, true})
      run(nonfinite, adaptive);
  }
  return failures == 0 ? 0 : 1;
}
