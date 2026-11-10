
#include <deal.II/base/mpi.h>
#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/petsc_compatibility.h>
#include <deal.II/lac/petsc_matrix_base.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <mumps_solver.h>

#if defined(FEZ_WITH_PETSC)
#  include <petscmat.h>
#endif

#include <algorithm>
#include <cmath>

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
  namespace
  {
    /**
     * Maximum number of times the analysis and factorization are redone with
     * an increased ICNTL(14) after MUMPS reports a too small workspace.
     */
    constexpr unsigned int max_workspace_retries = 5;

    /**
     * Default maximum factor by which the retries of one solve may increase
     * the workspace, which is (100 + ICNTL(14)) % of the analysis estimate,
     * relative to the workspace at the start of the solve.
     */
    constexpr PetscReal default_max_workspace_growth = 4.;

    /**
     * Default absolute ceiling for ICNTL(14) set by the retries, i.e., a
     * workspace of at most 11 times the analysis estimate.
     */
    constexpr PetscInt default_max_icntl14 = 1000;

    /**
     * Return the factored matrix of @p pc if MUMPS performs the factorization,
     * and nullptr otherwise, e.g., if another package was selected from the
     * command line.
     */
    Mat get_mumps_factor(const PC pc)
    {
      MatSolverType solver_type;
      AssertPETSc(PCFactorGetMatSolverType(pc, &solver_type));
      PetscBool is_mumps = PETSC_FALSE;
      if (solver_type)
        AssertPETSc(PetscStrcmp(solver_type, MATSOLVERMUMPS, &is_mumps));

      Mat F = nullptr;
      if (is_mumps)
        AssertPETSc(PCFactorGetMatrix(pc, &F));
      return F;
    }

    /**
     * After a factorization of @p F failed because of a too small workspace,
     * compute a larger ICNTL(14), the percentage increase of the workspace over
     * the analysis estimate, capped at @p icntl14_limit, and store it in the
     * options database. Returns the new ICNTL(14).
     */
    PetscInt increase_workspace_relaxation(const Mat      F,
                                           const PetscInt icntl14_limit)
    {
      PetscInt infog1, icntl14;
      AssertPETSc(MatMumpsGetInfog(F, 1, &infog1));
      AssertPETSc(MatMumpsGetIcntl(F, 14, &icntl14));

      // Factor applied to the workspace size, which is (100 + ICNTL(14)) % of
      // the analysis estimate. If the error is -9, MUMPS also reports the
      // additional memory needed, so that value is used instead of "growth".
      double growth = 2.;
      if (infog1 == -9)
      {
        // On the rank which raised the error, INFO(2) is the number of entries
        // missing in the workarray S, of size INFO(39), when the factorization
        // stopped. Negative values count millions of entries. The rest of the
        // factorization may need more, hence the safety factor of 1.5.
        PetscInt info1, info2, info39;
        AssertPETSc(MatMumpsGetInfo(F, 1, &info1));
        AssertPETSc(MatMumpsGetInfo(F, 2, &info2));
        AssertPETSc(MatMumpsGetInfo(F, 39, &info39));
        const auto n_entries = [](const PetscInt info) {
          return info >= 0 ? static_cast<double>(info) :
                             -1e6 * static_cast<double>(info);
        };

        double ratio = 0.;
        if (info1 == -9 && info39 != 0)
          ratio = (n_entries(info39) + n_entries(info2)) / n_entries(info39);
        ratio = Utilities::MPI::max(ratio, PetscObjectComm((PetscObject)F));
        if (ratio > 0.)
          growth = 1.5 * ratio;
      }

      const PetscInt new_icntl14 =
        std::min(icntl14_limit,
                 static_cast<PetscInt>(
                   std::ceil((100. + icntl14) * growth - 100.)));

      // The analysis reads -mat_mumps_icntl_14 from the options database, which
      // would override a value set with MatMumpsSetIcntl if the option was
      // given on the command line, so set it in the database instead. Later
      // analyses with the same prefix, e.g. after mesh adaptation, also use it.
      const char *prefix;
      AssertPETSc(MatGetOptionsPrefix(F, &prefix));
      const std::string option =
        "-" + std::string(prefix ? prefix : "") + "mat_mumps_icntl_14";
      AssertPETSc(PetscOptionsSetValue(nullptr,
                                       option.c_str(),
                                       std::to_string(new_icntl14).c_str()));
      return new_icntl14;
    }
  } // namespace

  SparseDirectMUMPSReuse::SparseDirectMUMPSReuse(
    SolverControl        &cn,
    const MPI_Comm        mpi_communicator,
    const AdditionalData &data)
    : SparseDirectMUMPS(cn, data)
  {
#ifndef DEAL_II_PETSC_WITH_MUMPS
    AssertThrow(
      false,
      ExcMessage(
        "Your PETSc installation does not include a copy of "
        "the MUMPS package necessary for this solver. You will need to "
        "configure PETSc so that it includes MUMPS, recompile it, and then "
        "re-configure and recompile deal.II as well."));
#endif

    initialize_ksp_with_comm(mpi_communicator);

    // Call the base class function.
    // Set ksp to KSPPREONLY with a zero initial guess, and a successful state
    // for the SolverControl, which KSPPREONLY never updates.
    set_solver_type(ksp);

    configure_ksp();
  }

  void SparseDirectMUMPSReuse::configure_ksp()
  {
    PC pc;
    AssertPETSc(KSPGetPC(ksp, &pc));
    AssertPETSc(PCSetType(pc, symmetric_mode ? PCCHOLESKY : PCLU));
    AssertPETSc(PCFactorSetMatSolverType(pc, MATSOLVERMUMPS));

    // Let the command line options override the defaults above. MUMPS options
    // such as -mat_mumps_icntl_14 are stored in the options database and read
    // by PETSc at each analysis phase.
    AssertPETSc(KSPSetOptionsPrefix(ksp, prefix_name.c_str()));
    AssertPETSc(KSPSetFromOptions(ksp));

    // Limits for the workspace retries, with the same prefix as the MUMPS
    // options.
    max_workspace_growth = default_max_workspace_growth;
    AssertPETSc(PetscOptionsGetReal(nullptr,
                                    prefix_name.c_str(),
                                    "-mat_mumps_workspace_growth_max",
                                    &max_workspace_growth,
                                    nullptr));
    AssertThrow(max_workspace_growth >= 1.,
                ExcMessage(
                  "-mat_mumps_workspace_growth_max must be at least 1."));

    max_icntl14 = default_max_icntl14;
    AssertPETSc(PetscOptionsGetInt(nullptr,
                                   prefix_name.c_str(),
                                   "-mat_mumps_icntl_14_max",
                                   &max_icntl14,
                                   nullptr));
    AssertThrow(max_icntl14 >= 0,
                ExcMessage("-mat_mumps_icntl_14_max must be non-negative."));
  }

  void SparseDirectMUMPSReuse::set_prefix(const std::string &prefix)
  {
    SparseDirectMUMPS::set_prefix(prefix);
    configure_ksp();
  }

  void
  SparseDirectMUMPSReuse::set_symmetric_mode(const bool matrix_is_symmetric)
  {
    SparseDirectMUMPS::set_symmetric_mode(matrix_is_symmetric);
    configure_ksp();
  }

  void SparseDirectMUMPSReuse::set_verbose(const bool is_verbose)
  {
    verbose = is_verbose;
  }

  void SparseDirectMUMPSReuse::solve(const MatrixBase &A,
                                     VectorBase       &x,
                                     const VectorBase &b)
  {
    Assert(ksp != nullptr,
           ExcMessage("The KSP of this solver was destroyed by reset(). "
                      "Create a new SparseDirectMUMPSReuse instead."));

    PC pc;
    AssertPETSc(KSPGetPC(ksp, &pc));

    // PETSc redoes the analysis only if the sparsity pattern of A changed since
    // the last solve, and the numerical factorization otherwise.
    AssertPETSc(KSPSetOperators(ksp, A, A));

    // Largest ICNTL(14) the retries may set, computed at the first workspace
    // error from the ICNTL(14) at the start of this solve.
    PetscInt icntl14_limit = -1;

    for (unsigned int retry = 0;; ++retry)
    {
      AssertPETSc(KSPSolve(ksp, b, x));

      // A failed analysis or factorization does not raise a PETSc error, but
      // is reported through the KSP converged reason, which is the same on all
      // ranks. Errors in the MUMPS solve phase raise a PETSc error caught by
      // AssertPETSc above.
      KSPConvergedReason reason;
      AssertPETSc(KSPGetConvergedReason(ksp, &reason));
      if (reason >= 0)
        return;

      // INFOG holds the global MUMPS error codes, identical on all ranks.
      const Mat F      = get_mumps_factor(pc);
      PetscInt  infog1 = 0, infog2 = 0;
      if (F)
      {
        AssertPETSc(MatMumpsGetInfog(F, 1, &infog1));
        AssertPETSc(MatMumpsGetInfog(F, 2, &infog2));
      }

      // Errors for which the MUMPS user guide recommends increasing ICNTL(14).
      // They typically occur when numerical pivoting creates more fill-in
      // than estimated by the analysis.
      const bool workspace_too_small =
        infog1 == -8 || infog1 == -9 || infog1 == -17 || infog1 == -20;

      PetscInt icntl14 = 0;
      if (workspace_too_small)
      {
        AssertPETSc(MatMumpsGetIcntl(F, 14, &icntl14));
        if (icntl14_limit < 0)
          icntl14_limit =
            std::min(max_icntl14,
                     static_cast<PetscInt>(std::floor(
                       max_workspace_growth * (100. + icntl14) - 100.)));
      }

      if (workspace_too_small && retry < max_workspace_retries &&
          icntl14 < icntl14_limit)
      {
        const PetscInt new_icntl14 =
          increase_workspace_relaxation(F, icntl14_limit);
        if (verbose)
        {
          const std::string info =
            "MUMPS workspace too small (INFOG(1) = " + std::to_string(infog1) +
            "), redoing the analysis and factorization with ICNTL(14) = " +
            std::to_string(new_icntl14) + "\n";
          AssertPETSc(
            PetscPrintf(PetscObjectComm((PetscObject)ksp), "%s", info.c_str()));
        }

        // PETSc does not refactor a factored matrix whose last factorization
        // failed, so destroy it. The next KSPSolve then redoes the analysis,
        // which reads the new ICNTL(14) from the options database, and the
        // factorization.
        AssertPETSc(PCReset(pc));
        AssertPETSc(KSPSetOperators(ksp, A, A));
        continue;
      }

      PCFailedReason pc_reason;
      AssertPETSc(PCGetFailedReason(pc, &pc_reason));

      std::string message =
        "MUMPS direct solver failed: KSP converged reason " +
        std::string(KSPConvergedReasons[reason]) + ", PC failed reason " +
        std::string(PCFailedReasons[pc_reason]);
      if (F)
        message += ", MUMPS INFOG(1) = " + std::to_string(infog1) +
                   ", INFOG(2) = " + std::to_string(infog2) +
                   ". See the MUMPS user guide for the meaning of these codes.";
      if (workspace_too_small)
      {
        message += " The workspace was still too small with ICNTL(14) = " +
                   std::to_string(icntl14);
        if (icntl14 >= icntl14_limit)
          message += ", the largest value allowed: one solve may increase "
                     "the workspace by a factor of at most " +
                     Utilities::to_string(max_workspace_growth) +
                     " (set with -mat_mumps_workspace_growth_max)"
                     ", and ICNTL(14) may not exceed " +
                     std::to_string(max_icntl14) +
                     " (set with -mat_mumps_icntl_14_max).";
        else
          message +=
            ", after " + std::to_string(max_workspace_retries) + " retries.";
      }

      AssertThrow(false, ExcMessage(message));
    }
  }

} // namespace PETScWrappers

DEAL_II_NAMESPACE_CLOSE
