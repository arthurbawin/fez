#include <elasticity_solver.h>
#include <incompressible_chns_solver.h>

#include "../tests.h"

#include "presolver_test_parameters.h"

class QuietElasticity : public ElasticitySolver<2>
{
public:
  QuietElasticity(const ParameterReader<2> &p)
    : ElasticitySolver<2>(p)
  {
    pcout.set_condition(false);
  }
  void output_results(const Mapping<2> * = nullptr) override {}
};
class HandoffProblem : public CHNSSolver<2, true>
{
public:
  HandoffProblem(const ParameterReader<2> &p)
    : CHNSSolver<2, true>(p)
  {
    pcout.set_condition(false);
  }
  void         solver_specific_post_processing() override { ++handoffs; }
  unsigned int handoffs = 0;
};
void check_continuation()
{
  const std::string single       = R"(subsection Time integration
 set scheme=stationary
end
subsection Elasticity
 subsection presolver
  set initial compression multiplier=0.1
  set continuation steps=1
 end
end
)";
  auto              one_param    = presolver_test_parameters<2>(single);
  auto              direct_param = presolver_test_parameters<2>(
    single + "subsection Elasticity\nsubsection presolver\nset initial "
                          "compression multiplier=1\nend\nend\n");
  QuietElasticity one(one_param), direct(direct_param);
  one.run();
  direct.run();
  double error = 0.;
  for (const auto i : one.get_dof_handler().locally_owned_dofs())
    error = std::max(error,
                     std::abs(one.get_present_solution()[i] -
                              direct.get_present_solution()[i]));
  AssertThrow(Utilities::MPI::max(error, MPI_COMM_WORLD) < 1e-12,
              ExcMessage(
                "A single continuation step did not reach full compression"));

  auto zero = presolver_test_parameters<2>(
    "subsection Elasticity\nsubsection presolver\nset initial compression "
    "multiplier=0\nset continuation steps=3\nend\nend\n");
  bool rejected = false;
  try
  {
    QuietElasticity problem(zero);
    problem.run();
  }
  catch (const std::exception &e)
  {
    rejected =
      std::string(e.what()).find("strictly positive") != std::string::npos;
  }
  AssertThrow(rejected,
              ExcMessage("Invalid geometric continuation was accepted"));

  auto fail = presolver_test_parameters<2>(
    "subsection Nonlinear solver\nset max_iterations=0\nend\n");
  HandoffProblem problem(fail);
  bool           failed = false;
  try
  {
    problem.run();
  }
  catch (const std::runtime_error &)
  {
    failed = true;
  }
  AssertThrow(failed && problem.handoffs == 0,
              ExcMessage("CHNS started with a nonconverged presolved mesh"));
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    deallog << "Single-step target, invalid ramp and failed handoff OK"
            << std::endl;
}
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  try
  {
    check_continuation();
  }
  catch (const std::exception &e)
  {
    return 1;
  }
}
