
#include <MMS.h>
#include <mesh.h>
#include <scratch_data.h>
#include <parameter_reader.h>
#include <utilities.h>
#include <boundary_conditions.h>
#include <monolithic_fsi_solver.h>

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    ConditionalOStream pcout(
        std::cout, (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0));

    std::string parameter_file = "../examples/test.prm";

    const unsigned int dim = read_problem_dimension(parameter_file);
    Parameters::BoundaryConditionsCount bc_count;
    read_number_of_boundary_conditions(parameter_file, bc_count);

    pcout << "Running solver with dim = " << dim << std::endl;

    if(dim == 2)
    {
      ParameterHandler prm;
      ParameterReader<2>  param(bc_count);
      param.declare(prm);

      prm.parse_input(parameter_file);
      param.read(prm);

      MonolithicFSISolver<2> problem(param);
      problem.run();
    }
    else if(dim == 3)
    {
      ParameterHandler prm;
      ParameterReader<3>  param(bc_count);
      param.declare(prm);

      prm.parse_input(parameter_file);
      param.read(prm);

      MonolithicFSISolver<3> problem(param);
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
