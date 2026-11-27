#ifndef ITERATIVE_FSI_SOLVER_H
#define ITERATIVE_FSI_SOLVER_H

#include <monolithic_fsi_solver.h>
#include <petscksp.h>

#include <array>
#include <vector>

/**
 * FGMRES on the unchanged, assembled FSI Jacobian. MUMPS remains available
 * through the inherited direct_mumps selection. Experimental reductions must
 * first pass the global Jacobian and mechanical checks provided here.
 */
template <int dim>
class IterativeFSISolver : public FSISolver<dim>
{
public:
  explicit IterativeFSISolver(const ParameterReader<dim> &param);
  ~IterativeFSISolver() override;

  void create_sparsity_pattern() override;
  void reset_solver_specific_data() override;
  void assemble_matrix() override;
  void solver_specific_post_processing() override;

protected:
  void solve_linear_system_iterative() override;

private:
  enum Role : unsigned int
  {
    velocity,
    pressure,
    mesh_interior,
    mesh_boundary,
    multiplier,
    force,
    body_velocity,
    angle,
    constrained,
    n_roles
  };

  void clear_linear_solver();
  void create_role_index_sets();
  void initialize_ksp();
  void diagnostic_residual(LA::ParVectorType &residual);
  void check_global_jacobian();

  KSP ksp = nullptr;
  std::array<IS, n_roles> role_is{};
  std::vector<Role> owned_roles;
  LA::ParVectorType reference_positions;
  bool pc_ready = false;
  double pc_bdf_c0 = 0.;
  PetscInt baseline_iterations = 0;
  PetscBool compare_mumps = PETSC_FALSE;
  PetscBool check_jacobian = PETSC_FALSE;
  PetscBool mechanical_diagnostics = PETSC_FALSE;
  PetscReal comparison_tolerance = 1e-7;
  PetscReal fd_step = 1e-6;
};

#endif
