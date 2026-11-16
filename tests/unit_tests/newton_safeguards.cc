#include <generic_solver.h>

#include <cfenv>
#include <functional>
#include <limits>

#include "../tests.h"

// Use the actual Newton controller with prescribed corrections/residuals.
// One unknown per rank, scaled so that the residual norm is MPI-independent.
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
  void solve_linear_system() override
  {
    ++linear_calls;
    newton_update = 1.;
  }
  LA::ParVectorType &get_present_solution() override { return solution; }
  AffineConstraints<double> &get_nonzero_constraints() override
  {
    return constraints;
  }

  void check_state(const double expected) const
  {
    for (const auto *v : {&solution, &evaluation_point})
      AssertThrow(std::abs((*v)[mpi_rank] - expected) < 1e-12,
                  ExcMessage("Unexpected solution/evaluation point"));
  }

  void check_rhs(const double expected) const
  {
    AssertThrow(std::abs(system_rhs.l2_norm() - expected) < 1e-12,
                ExcMessage("RHS not restored"));
  }

  void check_ghosts(const double expected) const
  {
    for (const auto *v : {&solution, &evaluation_point})
      for (unsigned int i = 0; i < mpi_size; ++i)
        AssertThrow(std::abs((*v)[i] - expected) < 1e-12,
                    ExcMessage("Solution/evaluation ghosts not restored"));
  }

  LA::ParVectorType             solution;
  AffineConstraints<double>     constraints;
  std::function<double(double)> residual;
  unsigned int                  rhs_calls = 0, linear_calls = 0;
};

struct Case
{
  const char                   *name;
  std::function<double(double)> residual;
  double                        expected;
  bool                          success;
  unsigned int                  memory          = 1;
  bool                          line_search     = true;
  unsigned int                  budget          = 6;
  int                           expected_rhs    = -1;
  unsigned int                  backtrack_steps = 4;
};

void check_case(const Case &test, const bool adaptive)
{
  Parameters::NonLinearSolver nonlinear{};
  nonlinear.tolerance            = 1e-10;
  nonlinear.divergence_tolerance = 1e4;
  nonlinear.max_iterations       = test.budget;
  nonlinear.enable_line_search   = test.line_search;
  nonlinear.line_search_memory   = test.memory;
  nonlinear.max_backtrack_steps  = test.backtrack_steps;
  nonlinear.verbosity            = Parameters::Verbosity::quiet;
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
  NewtonSolver<LA::ParVectorType> newton(nonlinear, &solver);
  bool                            threw = false;
  try
  {
    newton.solve(handler);
  }
  catch (const std::runtime_error &)
  {
    threw = true;
  }
  AssertThrow(threw == (!test.success && !adaptive),
              ExcMessage("Unexpected Newton failure/return"));
  solver.check_state(test.expected);
  solver.check_ghosts(test.expected);
  const double r = test.residual(test.expected);
  solver.check_rhs(std::isfinite(r) ? std::abs(r) : 0.);
  AssertThrow(solver.linear_calls <= test.budget,
              ExcMessage("Iteration budget exceeded"));
  if (test.expected_rhs >= 0)
    AssertThrow(solver.rhs_calls == unsigned(test.expected_rhs),
                ExcMessage("Unexpected number of RHS assemblies"));
  if (adaptive)
  {
    std::vector<LA::ParVectorType> previous(1);
    previous[0] = solver.solution;
    AssertThrow(handler.is_timestep_accepted(solver.solution, previous) ==
                  test.success,
                ExcMessage("Incorrect nonlinear convergence status"));
  }

  // Restart after failure without restoring the local scratch vector.
  if (!test.success)
  {
    const double root = test.expected + 1.;
    solver.residual   = [root](double x) { return root - x; };
    newton.solve(handler);
    solver.check_state(root);
    solver.check_ghosts(root);
    solver.check_rhs(0.);
  }

  // Reuse the very same Newton object: a previous maximum must not survive
  // into the next nonlinear solve (time step or continuation stage).
  if (test.success && test.memory > 1)
  {
    solver.get_local_evaluation_point() = 0.;
    solver.solution                     = solver.get_local_evaluation_point();
    solver.residual = [](double x) { return x == 0. ? 1. : 2.; };
    threw           = false;
    try
    {
      newton.solve(handler);
    }
    catch (const std::runtime_error &)
    {
      threw = true;
    }
    AssertThrow(threw == !adaptive,
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
  Parameters::NonLinearSolver parameters{};
  ParameterHandler            prm;
  parameters.declare_parameters(prm);
  parameters.read_parameters(prm);
  AssertThrow(parameters.line_search_memory == 3 &&
                parameters.max_backtrack_steps == 4,
              ExcInternalError());
  prm.enter_subsection("Nonlinear solver");
  prm.enter_subsection("Line search");
  prm.set("memory", "1");
  prm.set("max backtrack steps", "11");
  prm.set("enable", "false");
  prm.leave_subsection();
  prm.leave_subsection();
  parameters.read_parameters(prm);
  AssertThrow(parameters.line_search_memory == 1 &&
                parameters.max_backtrack_steps == 11 &&
                !parameters.enable_line_search,
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
  const double nan   = std::numeric_limits<double>::quiet_NaN();
  const auto   chain = [](double x) {
    if (x == 0.)
      return 10.;
    if (x == 1.)
      return 2.;
    if (x == 2.)
      return 8.;
    if (x == 3.)
      return 1.;
    if (x == 4.)
      return 0.;
    return 100.;
  };
  const std::vector<Case> cases = {
    {"memory 1 rejects increase", chain, 1., false},
    {"memory 3 permits bounded increase", chain, 4., true, 3},
    {"expired maximum",
     [](double x) {
       if (x == 0.)
         return 10.;
       if (x == 1.)
         return 2.;
       if (x == 2.)
         return 8.;
       if (x == 3.)
         return 1.;
       if (x == 4.)
         return 9.;
       return 100.;
     },
     3.,
     false,
     3},
    {"fixed reference",
     [](double x) {
       return x == 0. ? 10. : (x == .25 ? 0. : (x == 1. ? 100. : 90.));
     },
     .25,
     true},
    {"four trials stop before tiny step",
     [](double x) { return x == 0. ? 1. : (x == 1. / 1024. ? 0. : 2.); },
     0.,
     false,
     1,
     true,
     6,
     5,
     4},
    {"eleven trials reach tiny step",
     [](double x) { return x == 0. ? 1. : (x == 1. / 1024. ? 0. : 2.); },
     1. / 1024.,
     true,
     1,
     true,
     6,
     12,
     11}};
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
  for (const bool line_search : {false, true})
    for (const bool adaptive : {false, true})
      run({"nonfinite last valid iterate",
           [=](double x) { return x == 0. ? 10. : (x == 1. ? 2. : nan); },
           1.,
           false,
           1,
           line_search},
          adaptive);
  return failures == 0 ? 0 : 1;
}
