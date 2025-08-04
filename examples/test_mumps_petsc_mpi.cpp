#include <petscksp.h>  // PETSc Krylov Subspace methods (includes Mat, Vec, KSP)

int main(int argc, char **argv)
{
  PetscErrorCode ierr;
  ierr = PetscInitialize(&argc, &argv, nullptr, nullptr); CHKERRQ(ierr);

  MPI_Comm comm = PETSC_COMM_WORLD;

  // Problem size (small)
  PetscInt n = 100;

  // Create matrix (AIJ sparse)
  Mat A;
  ierr = MatCreate(comm, &A); CHKERRQ(ierr);
  ierr = MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, n, n); CHKERRQ(ierr);
  ierr = MatSetFromOptions(A); CHKERRQ(ierr);
  ierr = MatSetUp(A); CHKERRQ(ierr);

  // Insert values into matrix (simple 2D Poisson stencil)
  // Here: diagonal 2, off-diagonal -1 for adjacent entries
  for (PetscInt i = 0; i < n; ++i)
  {
    if (i > 0)
      ierr = MatSetValue(A, i, i - 1, -1.0, INSERT_VALUES); CHKERRQ(ierr);
    ierr = MatSetValue(A, i, i, 2.0, INSERT_VALUES); CHKERRQ(ierr);
    if (i < n - 1)
      ierr = MatSetValue(A, i, i + 1, -1.0, INSERT_VALUES); CHKERRQ(ierr);
  }

  ierr = MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);
  ierr = MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY); CHKERRQ(ierr);

  // Create vectors x and b
  Vec x, b;
  ierr = VecCreate(comm, &x); CHKERRQ(ierr);
  ierr = VecSetSizes(x, PETSC_DECIDE, n); CHKERRQ(ierr);
  ierr = VecSetFromOptions(x); CHKERRQ(ierr);
  ierr = VecDuplicate(x, &b); CHKERRQ(ierr);

  // Set right-hand side vector b = [1, 2, 3, 4]^T
  for (PetscInt i = 0; i < n; ++i)
    ierr = VecSetValue(b, i, (PetscScalar)(i + 1), INSERT_VALUES); CHKERRQ(ierr);

  ierr = VecAssemblyBegin(b); CHKERRQ(ierr);
  ierr = VecAssemblyEnd(b); CHKERRQ(ierr);

  // Create KSP solver context
  KSP ksp;
  ierr = KSPCreate(comm, &ksp); CHKERRQ(ierr);

  ierr = KSPSetOperators(ksp, A, A); CHKERRQ(ierr);

  // Set solver type to direct MUMPS if available
  ierr = KSPSetType(ksp, KSPPREONLY); CHKERRQ(ierr);

  PC pc;
  ierr = KSPGetPC(ksp, &pc); CHKERRQ(ierr);
  ierr = PCSetType(pc, PCLU); CHKERRQ(ierr);
  ierr = PCFactorSetMatSolverType(pc, MATSOLVERMUMPS); CHKERRQ(ierr);

  ierr = KSPSetFromOptions(ksp); CHKERRQ(ierr);

  // Solve the system
  ierr = KSPSolve(ksp, b, x); CHKERRQ(ierr);

  // Output solution vector
  PetscInt start, end;
  ierr = VecGetOwnershipRange(x, &start, &end); CHKERRQ(ierr);
  PetscScalar *x_array;
  ierr = VecGetArray(x, &x_array); CHKERRQ(ierr);

  for (PetscInt i = start; i < end; ++i)
    // PetscPrintf(comm, "x[%d] = %g\n", i, PetscRealPart(x_array[i - start]));
    printf("x[%d] = %g\n", i, PetscRealPart(x_array[i - start]));

  ierr = VecRestoreArray(x, &x_array); CHKERRQ(ierr);

  // Clean up
  ierr = KSPDestroy(&ksp); CHKERRQ(ierr);
  ierr = VecDestroy(&x); CHKERRQ(ierr);
  ierr = VecDestroy(&b); CHKERRQ(ierr);
  ierr = MatDestroy(&A); CHKERRQ(ierr);

  ierr = PetscFinalize(); CHKERRQ(ierr);

  return 0;
}
