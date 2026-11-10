#ifndef PRECONDITION_FIELDSPLIT_H
#define PRECONDITION_FIELDSPLIT_H

#include <deal.II/base/index_set.h>
#include <deal.II/lac/petsc_matrix_base.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_vector_base.h>
#include <parameters.h>

DEAL_II_NAMESPACE_OPEN

namespace PETScWrappers
{
  /**
   * Block Schur complement preconditioner for a (velocity, pressure) saddle
   * point system, based on PETSc's PCFIELDSPLIT. It acts on the monolithic
   * system matrix, whose rows are split in two fields given by index sets.
   *
   * This preconditioner is intended for the nonstabilized Navier-Stokes system,
   * with a zero pressure block: stabilized formulations (PSPG) should use a
   * monolithic preconditioner (e.g., AMG) instead.

   * The system matrix J and the preconditioner P (upper triangular block
   * factorization) are
   *
   *     J = [ A00  A01 ]      P = [ A00  A01 ]
   *         [ A10   0  ],         [  0    S  ],
   *
   * with S = -A10 A00^{-1} A01 the Schur complement. For the incompressible
   * Navier-Stokes equations, A00 = c0 Mu + nu K + N(u) gathers the time
   * derivative (c0 the leading BDF coefficient, 0 for steady problems, and Mu
   * the velocity mass matrix), the viscous and the linearized convection terms,
   * and A01 = A10^T = B^T, with B_ij = -∫Ω q_i ∇·φ_j dx. Applying P^{-1} to a
   * residual (r_u, r_p) amounts to
   *
   *     p = S^{-1} r_p,
   *     u = A00^{-1} (r_u - A01 p),
   *
   * where A00^{-1} and S^{-1} are each replaced by a single application of a
   * preconditioner, so that P^{-1} is a fixed linear operator, suitable for
   * GMRES.
   *
   * The Schur complement is approximated with the scaled pressure mass matrix
   * -Mp/nu (steady problems at low Reynolds numbers), the Cahouet-Chabard
   * preconditioner (unsteady problems, see below) or the least-squares
   * commutator (steady problems at higher Reynolds numbers).
   *
   * The Cahouet-Chabard preconditioner is derived from the generalized Stokes
   * operator, i.e., neglecting the convection N(u) in S. In the continuous
   * setting, and assuming that the operators commute,
   *
   *     -S = div (c0 - nu Δ)^{-1} ∇ ≈ (-Δ_p) (c0 + nu (-Δ_p))^{-1},
   *
   * so that (-S)^{-1} ≈ nu I + c0 (-Δ_p)^{-1}. Once discretized in weak form,
   * as S maps pressure coefficients to integrals against the pressure test
   * functions, the identity becomes the pressure mass matrix Mp, so that nu I
   * becomes (Mp/nu)^{-1}, and -Δ_p becomes B diag(Mu)^{-1} B^T, a pressure
   * Laplacian with the boundary conditions induced by those of the velocity,
   * where the lumped velocity mass matrix
   * keeps the product sparse. With c0 moved inside the inverse, and the sign of
   * S, this gives
   *
   *     S^{-1} ≈ (-Mp/nu)^{-1} + (-A10 (c0 diag(Mu))^{-1} A01)^{-1},
   *
   * which is the implemented approximation. For steady problems (c0 = 0), this
   * reduces to the well-known approximation
   *
   *     S^{-1} ≈ (-Mp/nu)^{-1},
   *
   * suited for viscous-dominated flows, and used in several deal.II tutorial
   * steps. For small time steps, the Laplacian term dominates, as in projection
   * methods. The sum covers all the ratios c0 h^2 / nu with a single
   * preconditioner. As convection is neglected in the approximation of S (but
   * not in A00), it degrades at high cell Reynolds numbers with large time
   steps,
   * where LSC is the alternative.
   *
   * The least-squares commutator (LSC) approximates
   *
   *     (A10 A00^{-1} A01)^{-1} ≈ L^{-1} A10 Q^{-1} A00 Q^{-1} A01 L^{-1},
   *
   * with L = A10 Q^{-1} A01 and Q = diag(Mu), the diagonal of the velocity mass
   * matrix, which defines the norm of the least-squares problem. The diagonal
   * of A00 is not used for Q: in the Newton Jacobian at higher Reynolds
   * numbers, the convection and reaction terms make some of its entries small
   * or negative, so that L is indefinite or nearly singular, and both its AMG
   * approximation and the scaling by Q^{-1} become unreliable. PETSc's PCLSC
   * omits the minus sign of S in this approximation, which is restored by
   * using a single Richardson step scaled by -1 as the solver of S.
   *
   * All the options passed to PETSc are prefixed with "fez_fs_", so that they
   * do not affect other PETSc preconditioners, and may be overridden from the
   * command line, e.g. -fez_fs_fieldsplit_p_pc_type gamg.
   */
  class PreconditionFieldSplitSchur : public PreconditionBase
  {
  public:
    /**
     * Preconditioner of the velocity block A00, applied once per outer
     * iteration. automatic selects block_jacobi for Cahouet-Chabard, amg_ilu
     * for LSC and amg otherwise.
     */
    using VelocityPreconditioner =
      ::Parameters::LinearSolver::VelocityBlockPreconditioner;

    /**
     * Approximation of the Schur complement S (see the class documentation).
     */
    enum class SchurApproximation
    {
      // (-Mp/nu)^{-1}, for steady problems at low Reynolds numbers
      mass,
      // (-Mp/nu)^{-1} + (-A10 D^{-1} A01)^{-1}, for unsteady problems
      cahouet_chabard,
      // Least-squares commutator, which accounts for convection, for steady
      // problems at higher Reynolds numbers
      lsc
    };

    /**
     * Additional data and settings of the inner preconditioners.
     */
    struct AdditionalData
    {
      SchurApproximation schur_approximation = SchurApproximation::mass;

      // Scaled pressure mass matrix -Mp/nu, required by the mass and
      // cahouet_chabard approximations. It has the size of the system, and
      // only its rows and columns of the pressure dofs are read.
      const MatrixBase *scaled_pressure_mass_matrix = nullptr;

      // Diagonal D = c0 diag(Mu) of the time derivative term of A00, required
      // by the cahouet_chabard approximation. It has the size of the system,
      // and only its velocity entries are read.
      const VectorBase *time_derivative_diagonal = nullptr;

      // Diagonal Q = diag(Mu) of the velocity mass matrix, required by the lsc
      // approximation. It has the size of the system, and only its velocity
      // entries are read. They must be positive.
      const VectorBase *velocity_mass_diagonal = nullptr;

      // Preconditioner of A00 (see the constructor)
      VelocityPreconditioner velocity_preconditioner =
        VelocityPreconditioner::automatic;

      // Levels of fill of the ILU of the velocity preconditioner
      unsigned int ilu_fill_level = 0;

      // Normalized constant pressure mode, if the pressure is only determined
      // up to a constant, nullptr otherwise (see the constructor). It has the
      // size of the system, and only its pressure entries are read.
      const VectorBase *pressure_null_vector = nullptr;
    };

    /**
     * Constructor. @p velocity_dofs and @p pressure_dofs are the locally owned
     * dofs (global indices, as in the matrix rows) of each field.
     *
     * The Schur complement S is approximated as chosen by
     * @p data.schur_approximation, which requires the matrix and vector of
     * @p data listed in AdditionalData. With lsc and cahouet_chabard, the
     * matrices L = A10 Q^{-1} A01 (LSC) and A10 D^{-1} A01 (Laplacian
     * of Cahouet-Chabard) are assembled here, to give them a nonzero diagonal
     * on the constrained pressure dofs (hanging nodes, Dirichlet pressure
     * conditions), for which the rows of A10 are empty. With cahouet_chabard,
     * each of the two inverses is approximated by its own preconditioner
     * (PCCOMPOSITE, additive), with options prefixes
     * fez_fs_fieldsplit_p_sub_0_ (mass) and fez_fs_fieldsplit_p_sub_1_
     * (Laplacian).
     *
     * @p data.velocity_preconditioner is the preconditioner of A00: one
     * BoomerAMG V-cycle with hypre's default smoother (amg) or with ILU
     * smoothing (amg_ilu), or block Jacobi with ILU on each process
     * (block_jacobi), the ILU having @p data.ilu_fill_level levels of fill. If
     * automatic:
     * - block_jacobi with Cahouet-Chabard: A00 is then dominated by the time
     *   derivative term, and AMG barely coarsens it (BoomerAMG reduces to its
     *   smoother), while one ILU application is several times cheaper,
     * - amg_ilu with LSC, whose convergence depends on the accuracy of the
     *   A00 solve, for which Gauss-Seidel smoothing is not sufficient once
     *   convection matters. At high cell Peclet numbers |u| h / nu, the
     *   V-cycle diverges on A00, and block_jacobi with at least ILU(1) should
     *   be chosen instead,
     * - amg with mass (for steady problems with a viscous-dominated A00).
     *
     * If @p data.pressure_null_vector is given, the pressure is determined up
     * to a constant and the system is singular (no pinned pressure dof). The
     * matrices A10 D^{-1} A01 (Cahouet-Chabard, LSC) are then singular too,
     * and their BoomerAMG uses a smoother instead of a direct solve on the
     * coarsest level. With LSC, the pressure part of the null vector is also
     * attached to L as its null space, and to its transpose: the solves with
     * L, of which BoomerAMG ignores the null space, then remove the constant
     * mode from their right-hand side and from their solution.
     */
    PreconditionFieldSplitSchur(const MatrixBase     &matrix,
                                const IndexSet       &velocity_dofs,
                                const IndexSet       &pressure_dofs,
                                const AdditionalData &data);
  };
} // namespace PETScWrappers

DEAL_II_NAMESPACE_CLOSE

#endif
