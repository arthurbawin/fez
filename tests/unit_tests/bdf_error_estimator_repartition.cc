#include <components_ordering.h>
#include <time_handler.h>

#include "../tests.h"

// AMR can preserve the global DoF count while changing the MPI partition.
// With the BDF truncation error strategy, the error estimator stores an
// additional solution, whose layout is updated when the solver reinitializes
// the error estimator after redistributing the dofs.

// This tests that the additional solution layout is indeed updated when needed,
// e.g. after mesh adaptation.
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  const auto                       comm = MPI_COMM_WORLD;
  const auto rank = Utilities::MPI::this_mpi_process(comm);
  AssertThrow(Utilities::MPI::n_mpi_processes(comm) == 2,
              ExcMessage("This test requires two MPI ranks"));
  if (rank == 0)
    initlog();

  Parameters::TimeIntegration parameters{};
  ParameterHandler            prm;
  parameters.declare_parameters(prm);
  prm.enter_subsection("Time integration");
  prm.set("scheme", "BDF2");
  prm.enter_subsection("Adaptation");
  prm.set("enable", "true");
  prm.set("adaptation strategy", "bdf truncation error");
  prm.leave_subsection();
  prm.leave_subsection();
  parameters.read_parameters(prm);
  parameters.n_time_intervals = 1;

  TimeHandler                time(parameters);
  ComponentOrdering          ordering;
  std::vector<unsigned char> dofs_to_component;

  // Keep eight global entries, but move the ownership boundary both ways.
  for (const unsigned int split : {4u, 3u, 5u, 4u})
  {
    IndexSet owned(8), relevant(8);
    owned.add_range(rank == 0 ? 0 : split, rank == 0 ? split : 8);
    relevant.add_range(0, 8);
    LA::ParVectorType values(owned, comm);
    for (const auto i : owned)
      values[i] = i + split;
    values.compress(VectorOperation::insert);

    // Model the solutions already transferred to the new partition by AMR.
    LA::ParVectorType present(owned, relevant, comm);
    present = values;
    std::vector<LA::ParVectorType> previous(2);
    for (auto &v : previous)
    {
      v.reinit(owned, relevant, comm);
      v = values;
    }

    // As in the solvers' setup_dofs(), reinit the error estimator, whose
    // additional solution has the preceding partition.
    time.reinit_error_estimator(
      ordering, owned, relevant, dofs_to_component, comm);
    time.rotate_solutions(present, previous);
    for (const auto i : relevant)
      AssertThrow(previous[0][i] == i + split && previous[1][i] == i + split,
                  ExcMessage(
                    "Parallel layout of the additional solution in BDF error "
                    "estimator was not updated after repartition"));
  }
  if (rank == 0)
    deallog << "Error estimator survives same-size MPI repartition"
            << std::endl;
}
