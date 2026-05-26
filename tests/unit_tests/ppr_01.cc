
#include "ppr.h"

int main(int argc, char *argv[])
{
  try
  {
    initlog();
    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    const unsigned int highest_recovered_derivative = 3;

    // Linear field
    // Without and with isoparametric representation of the recovery operator
    test_ppr<2>(4, 1, false, false, highest_recovered_derivative);
    test_ppr<2>(4, 1, true, false, highest_recovered_derivative);
    test_ppr<2>(4, 1, true, true, highest_recovered_derivative);

    // Quadratic field
    test_ppr<2>(4, 2, false, false, highest_recovered_derivative);
    test_ppr<2>(4, 2, true, false, highest_recovered_derivative);
    test_ppr<2>(4, 2, true, true, highest_recovered_derivative);
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
