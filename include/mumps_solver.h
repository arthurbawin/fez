
#ifndef MUMPS_SOLVER_H
#define MUMPS_SOLVER_H

#include <deal.II/lac/petsc_solver.h>

DEAL_II_NAMESPACE_OPEN

namespace PETScWrappers
{
  /**
   * A variant of SparseDirectMUMPS which keeps its KSP across solves, so that
   * MUMPS performs the analysis (symbolic factorization) only once, and only
   * the numerical factorization for subsequent solves with matrices sharing
   * the same sparsity pattern.
   *
   * The KSP is created and configured in the constructor, and the command
   * line options (with the optional prefix) are applied there, so solve() only
   * attaches the matrix, calls MUMPS and checks for failure. The object must be
   * recreated when the matrix sizes change, e.g., after mesh adaptation.
   */
  class SparseDirectMUMPSReuse : public SparseDirectMUMPS
  {
  public:
    /**
     * Constructor. Creates the KSP on @p mpi_communicator, which must be the
     * communicator of the matrices passed to solve().
     */
    SparseDirectMUMPSReuse(SolverControl        &cn,
                           const MPI_Comm        mpi_communicator,
                           const AdditionalData &data = AdditionalData());

    /**
     * Solve the linear system and check MUMPS's return code.
     *
     * If the factorization fails because the MUMPS workspace is too small,
     * ICNTL(14) is increased in the options database, and the analysis and
     * factorization are redone, up to a few times. The workspace may grow in
     * one solve by a factor of at most the value of the command line option
     * -<prefix>mat_mumps_workspace_growth_max (4 by default), and ICNTL(14) is
     * at most the value of -<prefix>mat_mumps_icntl_14_max (1000 by default).
     * The increased ICNTL(14) is kept for the next factorizations. Throws if
     * the factorization failed otherwise, reporting the MUMPS error codes
     * INFOG(1) and INFOG(2).
     */
    void solve(const MatrixBase &A, VectorBase &x, const VectorBase &b);

    /**
     * Set the command line options prefix, allowing to specify runtime
     * PETSc/Mumps options through flags of the form e.g.
     * -<prefix>mat_mumps_icntl_4 2. Re-applies the command line options to the
     * KSP.
     */
    void set_prefix(const std::string &prefix);

    /**
     * Select a Cholesky (symmetric) or LU factorization and re-apply the
     * command line options to the KSP.
     */
    void set_symmetric_mode(const bool matrix_is_symmetric);

    /**
     * Set the verbosity flag.
     */
    void set_verbose(const bool is_verbose);

  private:
    /**
     * Set the PC to a MUMPS LU or Cholesky factorization, then apply the
     * options prefix and the command line options.
     */
    void configure_ksp();

    /**
     * Maximum factor by which the workspace retries in one solve() may
     * increase the workspace, read from
     * -<prefix>mat_mumps_workspace_growth_max.
     */
    PetscReal max_workspace_growth;

    /**
     * Largest ICNTL(14) the workspace retries in solve() may set, read from
     * -<prefix>mat_mumps_icntl_14_max.
     */
    PetscInt max_icntl14;

    /**
     * Verbosity flag.
     */
    bool verbose = false;
  };
} // namespace PETScWrappers

DEAL_II_NAMESPACE_CLOSE

#endif
