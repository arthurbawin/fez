
#include <petscis.h>
#include <petscksp.h>
#include <petscmat.h>
#include <petscpc.h>

#include <deal.II/lac/exceptions.h>
#include <deal.II/lac/petsc_compatibility.h>
#include <precondition_fieldsplit.h>

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
     * Create the matrix L = A10 D^{-1} A01, where D is the diagonal matrix
     * with entries @p velocity_diagonal (on the velocity dofs, zero entries
     * are skipped), with a nonzero diagonal added on its empty rows. The
     * caller is responsible for destroying it.
     */
    Mat create_scaled_product(const Mat A,
                              const IS  is_u,
                              const IS  is_p,
                              const Vec velocity_diagonal)
    {
      Mat A01, A10, L;
      Vec scale, diagonal;
      AssertPETSc(MatCreateSubMatrix(A, is_u, is_p, MAT_INITIAL_MATRIX, &A01));
      AssertPETSc(MatCreateSubMatrix(A, is_p, is_u, MAT_INITIAL_MATRIX, &A10));

      AssertPETSc(VecDuplicate(velocity_diagonal, &scale));
      AssertPETSc(VecCopy(velocity_diagonal, scale));
      AssertPETSc(VecReciprocal(scale));
      AssertPETSc(MatDiagonalScale(A01, scale, nullptr));
#if DEAL_II_PETSC_VERSION_LT(3, 22, 0)
      AssertPETSc(MatMatMult(A10, A01, MAT_INITIAL_MATRIX, PETSC_DEFAULT, &L));
#else
      AssertPETSc(MatMatMult(A10, A01, MAT_INITIAL_MATRIX, PETSC_CURRENT, &L));
#endif

      // Rows of constrained pressure dofs are empty in A10, so L has no
      // diagonal there, which makes it singular. Add the mean of the nonzero
      // diagonal entries on these rows.
      AssertPETSc(MatCreateVecs(L, &diagonal, nullptr));
      AssertPETSc(MatGetDiagonal(L, diagonal));
      PetscInt     n;
      PetscScalar *values;
      AssertPETSc(VecGetLocalSize(diagonal, &n));
      AssertPETSc(VecGetArray(diagonal, &values));
      double       sum       = 0.;
      unsigned int n_nonzero = 0;
      for (PetscInt i = 0; i < n; ++i)
        if (values[i] != 0.)
        {
          sum += std::abs(values[i]);
          ++n_nonzero;
        }
      const double mean = n_nonzero > 0 ? sum / n_nonzero : 1.;
      for (PetscInt i = 0; i < n; ++i)
        values[i] = (values[i] == 0.) ? mean : 0.;
      AssertPETSc(VecRestoreArray(diagonal, &values));

      // These rows have no allocated entry in L, and inserting the entries one
      // by one (e.g., with MatDiagonalSet()) would reallocate L for each row.
      // Add a preallocated diagonal matrix instead: MatAXPY() then rebuilds L
      // once, with the merged sparsity pattern.
      Mat fill;
      AssertPETSc(
        MatCreateAIJ(PetscObjectComm(reinterpret_cast<PetscObject>(L)),
                     n,
                     n,
                     PETSC_DETERMINE,
                     PETSC_DETERMINE,
                     1,
                     nullptr,
                     0,
                     nullptr,
                     &fill));
      AssertPETSc(MatDiagonalSet(fill, diagonal, INSERT_VALUES));
      AssertPETSc(MatAXPY(L, 1., fill, DIFFERENT_NONZERO_PATTERN));
      AssertPETSc(MatDestroy(&fill));

      AssertPETSc(VecDestroy(&diagonal));
      AssertPETSc(VecDestroy(&scale));
      AssertPETSc(MatDestroy(&A01));
      AssertPETSc(MatDestroy(&A10));
      return L;
    }

    /**
     * Create the matrix -A10 D^{-1} A01 of the Cahouet-Chabard
     * preconditioner, where D is given by the velocity entries of
     * @p time_derivative_diagonal (see AdditionalData). The caller is
     * responsible for destroying it.
     */
    Mat create_cc_laplacian(const Mat A,
                            const IS  is_u,
                            const IS  is_p,
                            const Vec time_derivative_diagonal)
    {
      Vec diagonal;
      AssertPETSc(VecGetSubVector(time_derivative_diagonal, is_u, &diagonal));
      // The diagonal added on empty rows is positive, so negate afterwards
      Mat L = create_scaled_product(A, is_u, is_p, diagonal);
      AssertPETSc(
        VecRestoreSubVector(time_derivative_diagonal, is_u, &diagonal));
      AssertPETSc(MatScale(L, -1.));
      return L;
    }
  } // namespace

  PreconditionFieldSplitSchur::PreconditionFieldSplitSchur(
    const MatrixBase     &matrix,
    const IndexSet       &velocity_dofs,
    const IndexSet       &pressure_dofs,
    const AdditionalData &data)
    : PreconditionBase()
  {
    create_pc_with_mat(matrix);
    const MPI_Comm comm = get_mpi_communicator();

    AssertPETSc(PCSetType(pc, PCFIELDSPLIT));
    AssertPETSc(PCSetOptionsPrefix(pc, "fez_fs_"));
    // The type must be set now, and not through the options database:
    // PCFieldSplitSetSchurPre() is silently ignored for other types
    AssertPETSc(PCFieldSplitSetType(pc, PC_COMPOSITE_SCHUR));

    const bool use_cc =
      data.schur_approximation == SchurApproximation::cahouet_chabard;
    AssertThrow(data.scaled_pressure_mass_matrix != nullptr,
                ExcMessage("The mass and Cahouet-Chabard approximations of "
                           "the Schur complement require the scaled pressure "
                           "mass matrix."));
    AssertThrow(!use_cc || data.time_derivative_diagonal != nullptr,
                ExcMessage("The Cahouet-Chabard approximation of the Schur "
                           "complement requires the diagonal of the time "
                           "derivative term of the velocity block."));
    // User matrix preconditioning S (mass variants) and Laplacian part of the
    // Cahouet-Chabard preconditioner
    Mat schur_pre    = nullptr;
    Mat cc_laplacian = nullptr;

    IS is_u = velocity_dofs.make_petsc_is(comm);
    IS is_p = pressure_dofs.make_petsc_is(comm);
    AssertPETSc(PCFieldSplitSetIS(pc, "u", is_u));
    AssertPETSc(PCFieldSplitSetIS(pc, "p", is_p));

    // Extract the pressure rows and columns, in the same ordering as the
    // pressure field of the PC, and use them to precondition S.
    AssertPETSc(MatCreateSubMatrix(static_cast<const Mat &>(
                                     *data.scaled_pressure_mass_matrix),
                                   is_p,
                                   is_p,
                                   MAT_INITIAL_MATRIX,
                                   &schur_pre));
    AssertPETSc(
      PCFieldSplitSetSchurPre(pc, PC_FIELDSPLIT_SCHUR_PRE_USER, schur_pre));
    // With Cahouet-Chabard, both parts are kept until they are given to the
    // composite preconditioner of S, below
    if (use_cc)
      cc_laplacian = create_cc_laplacian(static_cast<const Mat &>(matrix),
                                         is_u,
                                         is_p,
                                         static_cast<const Vec &>(
                                           *data.time_derivative_diagonal));
    // The PC holds its own reference to the index sets
    AssertPETSc(ISDestroy(&is_u));
    AssertPETSc(ISDestroy(&is_p));

    // Set an option unless it was already given, e.g. on the command line
    auto set_default = [](const std::string &name, const std::string &value) {
      PetscBool already_set = PETSC_FALSE;
      AssertPETSc(
        PetscOptionsHasName(nullptr, nullptr, name.c_str(), &already_set));
      if (!already_set)
        set_option_value(name, value);
    };

    set_default("-fez_fs_pc_fieldsplit_schur_fact_type", "upper");

    // Preconditioner of A00, see the documentation of the constructor.
    // TODO: with Cahouet-Chabard, choose between block_jacobi and amg from
    // r = c0 h^2 / nu, the ratio of the time derivative and viscous terms,
    // minimized over the cells: block Jacobi with ILU becomes too weak when
    // A00 is dominated by viscosity in some cells (r below about 0.1, e.g. at
    // low Reynolds numbers or with local refinement), where AMG is needed.
    VelocityPreconditioner u_preconditioner = data.velocity_preconditioner;
#if DEAL_II_PETSC_VERSION_LT(3, 22, 0)
    // BoomerAMG with ILU smoothing requires PETSc 3.22 onward
    AssertThrow(u_preconditioner != VelocityPreconditioner::amg_ilu,
                ExcMessage("PETSc 3.22 onward is required to use BoomerAMG "
                           "with ILU smoother (amg_ilu)"));
#endif
    if (u_preconditioner == VelocityPreconditioner::automatic)
      u_preconditioner = use_cc ? VelocityPreconditioner::block_jacobi :
                                  VelocityPreconditioner::amg;

#if !defined(DEAL_II_PETSC_WITH_HYPRE)
    // BoomerAMG is used for the velocity block (amg, amg_ilu) and for the
    // Laplacian part of Cahouet-Chabard
    AssertThrow(u_preconditioner == VelocityPreconditioner::block_jacobi &&
                  !use_cc,
                ExcMessage("PETSc must be configured with hypre to use "
                           "BoomerAMG, which is required for the amg and "
                           "amg_ilu velocity preconditioners and "
                           "Cahouet-Chabard. Without hypre, only the mass "
                           "approximation of the Schur complement with the "
                           "block_jacobi velocity preconditioner is "
                           "available."));
#endif

    const std::string ilu_levels = std::to_string(data.ilu_fill_level);

    // Without pinned pressure dof, the pressure Laplacian-like matrices are
    // singular: avoid the direct solve of BoomerAMG on the coarsest level
    auto set_coarse_smoother = [&](const std::string &amg_prefix) {
      if (data.pressure_null_space)
        set_default(amg_prefix + "pc_hypre_boomeramg_relax_type_coarse",
                    "symmetric-SOR/Jacobi");
    };

    // One application of the inner preconditioner for A00 and S
    const std::string u_prefix = "-fez_fs_fieldsplit_u_";
    const std::string p_prefix = "-fez_fs_fieldsplit_p_";
    set_default(u_prefix + "ksp_type", "preonly");
    set_default(p_prefix + "ksp_type", "preonly");

    // Velocity block A00, see above
    if (u_preconditioner == VelocityPreconditioner::block_jacobi)
    {
      set_default(u_prefix + "pc_type", "bjacobi");
      set_default(u_prefix + "sub_pc_type", "ilu");
      set_default(u_prefix + "sub_pc_factor_levels", ilu_levels);
    }
    else
    {
      // amg or amg_ilu
      set_default(u_prefix + "pc_type", "hypre");
      set_default(u_prefix + "pc_hypre_type", "boomeramg");
      // ILU smoothing (hypre with PETSc 3.22 onward), for a non-symmetric,
      // convection-dominated A00 (large CFL numbers), for which the
      // Gauss-Seidel smoother is not sufficient and the outer iterations can
      // stagnate
      if (u_preconditioner == VelocityPreconditioner::amg_ilu)
      {
        set_default(u_prefix + "pc_hypre_boomeramg_smooth_type", "ILU");
        set_default(u_prefix + "pc_hypre_boomeramg_ilu_level", ilu_levels);
      }
    }

    // Schur complement S
    if (use_cc)
    {
      // Sum of the approximate inverses of -Mp/nu (sub_0, block Jacobi with
      // ILU(0) as in the mass variant) and -A10 D^{-1} A01 (sub_1, AMG).
      // The sub-PCs are added after the setup, see below, and must not be
      // given with -pc_composite_pcs.
      set_default(p_prefix + "pc_type", "composite");
      set_default(p_prefix + "pc_composite_type", "additive");
      set_default(p_prefix + "sub_0_sub_pc_type", "ilu");
      set_default(p_prefix + "sub_1_pc_hypre_type", "boomeramg");
      set_coarse_smoother(p_prefix + "sub_1_");
    }
    else
    {
      // Mass matrix, for which AMG is not suited (a mass matrix is not an
      // M-matrix): block Jacobi with ILU(0), as a single point Jacobi
      // application is too inaccurate
      set_default(p_prefix + "pc_type", "bjacobi");
      set_default(p_prefix + "sub_pc_type", "ilu");
    }

    AssertPETSc(PCSetFromOptions(pc));
    AssertPETSc(PCSetUp(pc));

    if (use_cc)
    {
      // The Schur complement KSP is created by PCSetUp(), but its PC is only
      // set up at the first solve. Add the sub-PCs of the composite now, each
      // with its own operator: PCCOMPOSITE only passes its operators (here
      // -Mp/nu) to the sub-PCs which have none. They are not given through
      // -pc_composite_pcs, because PCSetFromOptions() is called again on the
      // Schur KSP at each solve once it has iterated, and every call appends
      // the listed sub-PCs again, without operators. If the PC type was
      // overridden on the command line, the mass variant is used instead.
      KSP     *sub_ksp;
      PetscInt n_splits;
      PC       pc_schur;
      AssertPETSc(PCFieldSplitSchurGetSubKSP(pc, &n_splits, &sub_ksp));
      AssertPETSc(KSPGetPC(sub_ksp[1], &pc_schur));
      AssertPETSc(PetscFree(sub_ksp));

      PetscBool is_composite = PETSC_FALSE;
      AssertPETSc(PetscObjectTypeCompare(
        reinterpret_cast<PetscObject>(pc_schur), PCCOMPOSITE, &is_composite));
      if (is_composite)
      {
        PetscInt n_sub_pcs = 0;
        AssertPETSc(PCCompositeGetNumberPC(pc_schur, &n_sub_pcs));
        AssertThrow(n_sub_pcs == 0,
                    ExcMessage("Do not set -fez_fs_fieldsplit_p_pc_composite_"
                               "pcs: the sub-PCs of the Cahouet-Chabard "
                               "preconditioner are added in the code."));
        AssertPETSc(PCCompositeAddPCType(pc_schur, PCBJACOBI));
        AssertPETSc(PCCompositeAddPCType(pc_schur, PCHYPRE));

        PC sub_pc;
        AssertPETSc(PCCompositeGetPC(pc_schur, 0, &sub_pc));
        AssertPETSc(PCSetOperators(sub_pc, schur_pre, schur_pre));
        AssertPETSc(PCSetFromOptions(sub_pc));
        AssertPETSc(PCCompositeGetPC(pc_schur, 1, &sub_pc));
        AssertPETSc(PCSetOperators(sub_pc, cc_laplacian, cc_laplacian));
        AssertPETSc(PCSetFromOptions(sub_pc));
      }
    }

    // The PCs hold their own references (cc_laplacian is null without
    // Cahouet-Chabard, which MatDestroy() accepts)
    AssertPETSc(MatDestroy(&schur_pre));
    AssertPETSc(MatDestroy(&cc_laplacian));
  }
} // namespace PETScWrappers

DEAL_II_NAMESPACE_CLOSE
