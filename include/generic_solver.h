#ifndef GENERIC_SOLVER_H
#define GENERIC_SOLVER_H

#include <parameter_reader.h>
#include <types.h>

#include <deal.II/base/timer.h>
#include <deal.II/base/conditional_ostream.h>

using namespace dealii;

/**
 * Abstract base class for a generic solver with the following common members:
 *
 * - simulation parameters
 * - mpi communicator
 * - nonlinear solver
 * - linear system solver
 *
 * TODO: the template dim only affects the ParameterReader, maybe it's best
 * to select what is needed from the parameters to avoid templating the whole
 * class.
 */
template <int dim, typename VectorType>
class GenericSolver
{
public:
  GenericSolver(const ParameterReader<dim> &param)
    : param(param)
  , mpi_communicator(MPI_COMM_WORLD)
  , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
  , mpi_size(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , pcout(std::cout, (mpi_rank == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  {}

public:
  virtual ~GenericSolver()
  {

  }

public:
  /**
   * Solve the problem
   */
  virtual void run() = 0;

  /**
   *
   */
  virtual void assemble_matrix() = 0;

  /**
   *
   */
  virtual void assemble_rhs() = 0;

  /**
   *
   */
  virtual void solve_linear_system() = 0;

  /**
   *
   */
  virtual void solve_nonlinear_problem()
	{
		pcout << "Solving nonlinear problem" << std::endl;
	}

protected:
  ParameterReader<dim> param;

  MPI_Comm           mpi_communicator;
  const unsigned int mpi_rank;
  const unsigned int mpi_size;

  ConditionalOStream pcout;
  TimerOutput        computing_timer;

  // If parallel vector type, these are vectors with ghost entries (read only)
  VectorType present_solution;
  VectorType evaluation_point;

  // If parallel vector type, these are vectors w/o ghost entries (owned)
  VectorType local_evaluation_point;
  VectorType newton_update;
  VectorType system_rhs;
};

#endif