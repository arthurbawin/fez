#include <elasticity_solver.h>

#include <filesystem>
#include <fstream>

#include "../tests.h"

#include "presolver_test_parameters.h"

class CachedElasticity : public ElasticitySolver<2>
{
public:
  CachedElasticity(const ParameterReader<2> &p)
    : ElasticitySolver<2>(p)
  {
    pcout.set_condition(false);
  }
  void solve_linear_system() override
  {
    ++linear_solves;
    ElasticitySolver<2>::solve_linear_system();
  }
  void         output_results(const Mapping<2>         * = nullptr) override {}
  unsigned int linear_solves = 0;
};

std::string cache_contents()
{
  std::ifstream file("presolver-test.cache");
  return {std::istreambuf_iterator<char>(file),
          std::istreambuf_iterator<char>()};
}
void restore_cache(const std::string &contents)
{
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::ofstream file("presolver-test.cache");
    file << contents;
  }
  MPI_Barrier(MPI_COMM_WORLD);
}
ParameterReader<2> parameters(const std::string &extra = "")
{
  return presolver_test_parameters<2>(R"(subsection Time integration
 set scheme=stationary
end
subsection Initial conditions
 subsection cahn hilliard tracer
  set Function expression=tanh((x-xc)/0.2)
  set Function constants=xc=0.4
 end
end
subsection Elasticity
 subsection presolved mesh position
  set mode=reuse
  set file=presolver-test.cache
 end
end
)" + extra);
}
void check_cache()
{
  restore_cache("absent or invalid cache");
  auto             p = parameters();
  CachedElasticity fresh(p);
  fresh.run();
  AssertThrow(fresh.linear_solves > 0,
              ExcMessage("Missing cache must trigger a solve"));
  const auto       original = cache_contents();
  CachedElasticity loaded(p);
  loaded.run();
  AssertThrow(loaded.linear_solves == 0,
              ExcMessage("Valid cache was not reused"));
  double difference = 0.;
  for (const auto i : fresh.get_dof_handler().locally_owned_dofs())
    difference = std::max(difference,
                          std::abs(fresh.get_present_solution()[i] -
                                   loaded.get_present_solution()[i]));
  AssertThrow(Utilities::MPI::max(difference, MPI_COMM_WORLD) < 1e-12,
              ExcMessage(
                "Cached positions differ from the converged solution"));

  const std::vector<std::string> changes = {
    "subsection Cahn Hilliard\nset interface thickness=0.25\nend\n",
    "subsection Cahn Hilliard\nset mff physics compression factor=0.7\nend\n",
    "subsection Initial conditions\nsubsection cahn hilliard tracer\nset "
    "Function constants=xc=0.45\nend\nend\n",
    "subsection Pseudosolid boundary conditions\nsubsection boundary 0\nset "
    "type=input_function\nsubsection x\nset Function "
    "expression=x\nend\nsubsection y\nset type=no_flux\nend\nend\nend\n",
    "subsection Mesh\nset dealii mesh parameters=3,3:0,0:1.1,1:true\nend\n",
    "subsection Elasticity\nsubsection presolved mesh position\nset "
    "mode=force_recompute\nend\nend\n"};
  for (const auto &change : changes)
  {
    restore_cache(original);
    auto             changed = parameters(change);
    CachedElasticity solver(changed);
    solver.run();
    AssertThrow(solver.linear_solves > 0,
                ExcMessage("Stale cache was incorrectly reused"));
  }
  restore_cache(original);
  auto updated_parameters = parameters();
  updated_parameters.initial_conditions.initial_chns_tracer_callback
    ->update_constants({{"xc", .45}});
  CachedElasticity updated(updated_parameters);
  updated.run();
  AssertThrow(updated.linear_solves > 0,
              ExcMessage("Runtime constants did not invalidate the cache"));
  restore_cache(original);
  auto transport_parameters =
    parameters("subsection Cahn Hilliard\nset mff transport factor=9\nend\n");
  CachedElasticity transport(transport_parameters);
  transport.run();
  AssertThrow(transport.linear_solves == 0,
              ExcMessage(
                "Inactive transport needlessly invalidated the cache"));

  restore_cache("truncated");
  auto             truncated_parameters = parameters();
  CachedElasticity truncated(truncated_parameters);
  truncated.run();
  AssertThrow(truncated.linear_solves > 0,
              ExcMessage("Corrupt cache must be recomputed"));
  restore_cache(original);
  auto failure_parameters =
    parameters("subsection Nonlinear solver\nset max_iterations=0\nend\n");
  bool failed = false;
  try
  {
    CachedElasticity failure(failure_parameters);
    failure.run();
  }
  catch (const std::runtime_error &)
  {
    failed = true;
  }
  AssertThrow(failed,
              ExcMessage("This intentionally under-iterated solve must fail"));
  AssertThrow(cache_contents() == original,
              ExcMessage("A failed solve replaced a valid cache"));

  auto invalid_path_parameters = parameters(
    "subsection Elasticity\nsubsection presolved mesh position\nset "
    "mode=force_recompute\nset file=missing-cache-parent/position\nend\nend\n");
  failed = false;
  try
  {
    CachedElasticity failure(invalid_path_parameters);
    failure.run();
  }
  catch (const std::exception &e)
  {
    failed = std::string(e.what()).find(
               "Could not write presolved mesh cache") != std::string::npos;
  }
  AssertThrow(Utilities::MPI::min(failed ? 1 : 0, MPI_COMM_WORLD) == 1,
              ExcMessage("All ranks must observe a cache write failure"));
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    deallog << "Cache preserves positions, invalidates stale inputs and "
               "handles failures"
            << std::endl;
    std::filesystem::remove("presolver-test.cache");
  }
}
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    initlog();
  try
  {
    check_cache();
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
