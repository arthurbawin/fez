#include <petscksp.h>

int main(int argc, char **args)
{
  Vec            x, b;
  Mat            A;
  KSP            ksp;
  PetscErrorCode ierr;

  PetscInitialize(&argc, &args, NULL, NULL);

  // Build 5x5 identity matrix
  MatCreate(PETSC_COMM_WORLD, &A);
  MatSetSizes(A, PETSC_DECIDE, PETSC_DECIDE, 5, 5);
  MatSetFromOptions(A);
  MatSetUp(A);
  for (int i = 0; i < 5; i++)
    MatSetValue(A, i, i, 1.0, INSERT_VALUES);
  MatAssemblyBegin(A, MAT_FINAL_ASSEMBLY);
  MatAssemblyEnd(A, MAT_FINAL_ASSEMBLY);

  VecCreate(PETSC_COMM_WORLD, &b);
  VecSetSizes(b, PETSC_DECIDE, 5);
  VecSetFromOptions(b);
  VecSet(b, 1.0);

  VecDuplicate(b, &x);

  KSPCreate(PETSC_COMM_WORLD, &ksp);
  KSPSetOperators(ksp, A, A);
  KSPSetType(ksp, KSPPREONLY);

  PC pc;
  KSPGetPC(ksp, &pc);
  PCSetType(pc, PCLU);
  PCFactorSetMatSolverType(pc, MATSOLVERMUMPS);

  KSPSetFromOptions(ksp);
  KSPSolve(ksp, b, x);

  VecView(x, PETSC_VIEWER_STDOUT_WORLD);

  KSPDestroy(&ksp);
  VecDestroy(&x);
  VecDestroy(&b);
  MatDestroy(&A);
  PetscFinalize();
  return 0;
}