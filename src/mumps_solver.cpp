
#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/petsc_compatibility.h>
#include <deal.II/lac/petsc_matrix_base.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <mumps_solver.h>

#if defined(FEZ_WITH_PETSC)
#include <petscmat.h>
#endif

// Shorthand notation for PETSc error codes.
#define AssertPETSc(code)                        \
  do                                             \
  {                                              \
    PetscErrorCode ierr = (code);                \
    AssertThrow(ierr == 0, ExcPETScError(ierr)); \
  }                                              \
  while (false)

DEAL_II_NAMESPACE_OPEN

namespace PETScWrappers
{
  SparseDirectMUMPSReuse::SparseDirectMUMPSReuse(SolverControl        &cn,
                                                 const AdditionalData &data)
    : SparseDirectMUMPS(cn, data)
  {
    initialize_ksp_with_comm(MPI_COMM_WORLD);
    AssertPETSc(KSPSetType(ksp, KSPPREONLY));
    AssertPETSc(KSPGetPC(ksp, &pc));
    AssertPETSc(PCSetType(pc, PCLU));
    AssertPETSc(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
  }

  void SparseDirectMUMPSReuse::solve(const MatrixBase &A,
                                     VectorBase       &x,
                                     const VectorBase &b)
  // {
  //   AssertPETSc(KSPSetOperators(ksp, A, A));
  //   AssertPETSc(PCFactorSetUpMatSolverType(pc));
  //   AssertPETSc(PCFactorGetMatrix(pc, &factored_matrix));

  //   // Sequential and parallel reordering strategy
  //   // MatMumpsSetIcntl(factored_matrix, 7, 2);
  //   // MatMumpsSetIcntl(factored_matrix, 28, 2);
  //   // MatMumpsSetIcntl(factored_matrix, 29, 1);

  //   // Override with command line options
  //   AssertPETSc(KSPSetFromOptions(ksp));

  //   // Solve
  //   AssertPETSc(KSPSolve(ksp, b, x));

  //   // Check MUMPS error code
  //   PetscInt info_entry = 1, ret_value;
  //   AssertPETSc(MatMumpsGetInfo(factored_matrix, info_entry, &ret_value));

  //   AssertThrow(ret_value == 0,
  //     ExcMessage("Error"));
  // }
  {
#  ifdef DEAL_II_PETSC_WITH_MUMPS
    /*
     * creating a solver object if this is necessary
     */
    if (ksp == nullptr)
      {
        initialize_ksp_with_comm(A.get_mpi_communicator());

        /*
         * setting the solver type
         */
        set_solver_type(ksp);

        /*
         * set the matrices involved. the last argument is irrelevant here,
         * since we use the solver only once anyway
         */
        AssertPETSc(KSPSetOperators(ksp, A, A));

        /*
         * getting the associated preconditioner context
         */
        PC pc;
        AssertPETSc(KSPGetPC(ksp, &pc));

        /*
         * build PETSc PC for particular PCLU or PCCHOLESKY preconditioner
         * depending on whether the symmetric mode has been set
         */
        if (symmetric_mode)
          AssertPETSc(PCSetType(pc, PCCHOLESKY));
        else
          AssertPETSc(PCSetType(pc, PCLU));

          /*
           * set the software that is to be used to perform the lu
           * factorization here we start to see differences with the base
           * class solve function
           */
#    if DEAL_II_PETSC_VERSION_LT(3, 9, 0)
        AssertPETSc(PCFactorSetMatSolverPackage(pc, MATSOLVERMUMPS));
#    else
        AssertPETSc(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));
#    endif

        /*
         * set up the package to call for the factorization
         */
#    if DEAL_II_PETSC_VERSION_LT(3, 9, 0)
        AssertPETSc(PCFactorSetUpMatSolverPackage(pc));
#    else
        AssertPETSc(PCFactorSetUpMatSolverType(pc));
#    endif

        /*
         * get the factored matrix F from the preconditioner context.
         */
        // Mat F;
        AssertPETSc(PCFactorGetMatrix(pc, &factored_matrix));

        /*
         * pass control parameters to MUMPS.
         * Setting entry 7 of MUMPS ICNTL array to a value
         * of 2. This sets use of Approximate Minimum Fill (AMF)
         */
        AssertPETSc(MatMumpsSetIcntl(factored_matrix, 7, 2));

        /*
         * by default we set up the preconditioner only once.
         * this can be overridden by command line.
         */
        AssertPETSc(KSPSetReusePreconditioner(ksp, PETSC_TRUE));
      }

    /*
     * set the matrices involved. the last argument is irrelevant here,
     * since we use the solver only once anyway
     */
    AssertPETSc(KSPSetOperators(ksp, A, A));

    /*
     * set the command line option prefix name
     */
    AssertPETSc(KSPSetOptionsPrefix(ksp, prefix_name.c_str()));

    /*
     * set the command line options provided by the user to override
     * the defaults
     */
    AssertPETSc(KSPSetFromOptions(ksp));

    /*
     * solve the linear system
     */
    AssertPETSc(KSPSolve(ksp, b, x));

    // Check MUMPS error code
    AssertPETSc(PCFactorGetMatrix(pc, &factored_matrix));
    PetscInt error_code;
    PetscInt info1 = 1;
    AssertPETSc(MatMumpsGetInfo(factored_matrix, info1, &error_code));

    // != 0 means error, and -1 means the error happened on another proc
    PetscInt global_error = 0;
    MPI_Allreduce(&error_code, &global_error, 1, MPIU_INT, MPI_MIN, PETSC_COMM_WORLD);

    if (global_error != 0 && global_error != -1)
      AssertThrow(
        false,
        ExcMessage(
          "MUMPS failed with error code " + std::to_string(global_error)));

    // /*
    //  * in case of failure throw exception
    //  */
    // if (solver_control &&
    //     solver_control->last_check() != SolverControl::success)
    //   {
    //     AssertThrow(false,
    //                 SolverControl::NoConvergence(solver_control->last_step(),
    //                                              solver_control->last_value()));
    //   }

#  else // DEAL_II_PETSC_WITH_MUMPS
    Assert(
      false,
      ExcMessage(
        "Your PETSc installation does not include a copy of "
        "the MUMPS package necessary for this solver. You will need to configure "
        "PETSc so that it includes MUMPS, recompile it, and then re-configure "
        "and recompile deal.II as well."));
    (void)A;
    (void)x;
    (void)b;
#  endif
  }

} // namespace PETScWrappers

DEAL_II_NAMESPACE_CLOSE