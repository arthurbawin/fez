
#include <iterative_fsi_solver.h>
#include <parameter_reader.h>
#include <utilities.h>

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    ConditionalOStream               pcout(
      std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

    char petsc_version[256] = {};
    AssertThrow(PetscGetVersion(petsc_version, sizeof(petsc_version)) == 0,
                ExcMessage("Cannot query the PETSc version."));
    char mpi_version[MPI_MAX_LIBRARY_VERSION_STRING] = {};
    int mpi_version_length = 0;
    AssertThrow(MPI_Get_library_version(mpi_version, &mpi_version_length) ==
                  MPI_SUCCESS,
                ExcMessage("Cannot query the MPI version."));
    pcout << "deal.II " << DEAL_II_PACKAGE_VERSION << '\n'
          << petsc_version << '\n'
          << std::string(mpi_version, mpi_version_length) << std::endl;

    if (argc < 2)
    {
      std::cerr << "Usage: " << argv[0] << " <parameter_file>" << std::endl;
      return 1;
    }

    const std::string parameter_file = argv[1];

    pcout << "Number of MPI processes: "
          << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;

    const unsigned int dim = read_problem_dimension(parameter_file);
    Parameters::BoundaryConditionsData bc_data;
    read_number_of_boundary_conditions(parameter_file, bc_data);

    if (dim == 2)
    {
      ParameterHandler   prm;
      ParameterReader<2> param(bc_data);
      param.declare(prm);

      prm.parse_input(parameter_file);
      param.read(prm);

      IterativeFSISolver<2> problem(param);
      if (param.mms_param.enable)
        problem.run_convergence_loop<2>();
      else
        problem.run();
    }
    else if (dim == 3)
    {
      ParameterHandler   prm;
      ParameterReader<3> param(bc_data);
      param.declare(prm);

      prm.parse_input(parameter_file);
      param.read(prm);

      IterativeFSISolver<3> problem(param);
      if (param.mms_param.enable)
        problem.run_convergence_loop<3>();
      else
        problem.run();
    }
    else
    {
      throw std::runtime_error("Can only run solver for dim = 2 or 3.");
    }
  }
  catch (const std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
