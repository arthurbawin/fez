#include <iterative_fsi_solver.h>

#include <deal.II/lac/petsc_compatibility.h>
#include <deal.II/lac/exceptions.h>
#include <deal.II/numerics/vector_tools.h>

#include <algorithm>
#include <cmath>
#include <limits>

namespace
{
  void check_petsc(const PetscErrorCode error)
  {
    AssertThrow(error == 0, dealii::ExcPETScError(error));
  }

  constexpr const char *role_names[] = {"velocity", "pressure", "mesh_interior",
                                        "mesh_boundary", "multiplier", "force",
                                        "body_velocity", "angle", "constrained"};
}

template <int dim>
IterativeFSISolver<dim>::IterativeFSISolver(const ParameterReader<dim> &param)
  : FSISolver<dim>(param)
{
  using Method = Parameters::LinearSolver::Method;
  const auto &linear = param.linear_solver.at(this->solver_type);
  AssertThrow(linear.method == Method::direct_mumps ||
                linear.method == Method::fgmres,
              ExcMessage("iterative_fsi supports direct_mumps or fgmres."));
  AssertThrow(linear.method != Method::fgmres || linear.right_preconditioning,
              ExcMessage("FGMRES requires right preconditioning."));
  using Coupling = typename Parameters::FSI<dim>::CouplingStrategy;
  AssertThrow(!param.fsi.enable_coupling || param.fsi.zero_mass_model ||
                param.fsi.coupling ==
                  Coupling::local_position_master_to_lambda_accumulators,
              ExcMessage("The validated dynamic reference uses coupling 3; "
                         "coupling 4 needs a separate matrix/RHS correction."));
  check_petsc(PetscOptionsGetBool(nullptr, "fsi_", "-compare_mumps",
                                  &compare_mumps, nullptr));
  check_petsc(PetscOptionsGetBool(nullptr, "fsi_", "-check_jacobian",
                                  &check_jacobian, nullptr));
  check_petsc(PetscOptionsGetBool(nullptr, "fsi_", "-mechanical_diagnostics",
                                  &mechanical_diagnostics, nullptr));
  check_petsc(PetscOptionsGetReal(nullptr, "fsi_", "-comparison_tolerance",
                                  &comparison_tolerance, nullptr));
  check_petsc(PetscOptionsGetReal(nullptr, "fsi_", "-fd_step", &fd_step, nullptr));
  AssertThrow(comparison_tolerance > 0. && fd_step > 0.,
              ExcMessage("FSI diagnostic tolerances must be positive."));
}

template <int dim>
IterativeFSISolver<dim>::~IterativeFSISolver()
{
  // Destructors cannot throw; PETSc objects are released before MPI finalization.
  if (ksp)
    KSPDestroy(&ksp);
  for (auto &is : role_is)
    if (is)
      ISDestroy(&is);
}

template <int dim>
void IterativeFSISolver<dim>::clear_linear_solver()
{
  if (ksp)
    check_petsc(KSPDestroy(&ksp));
  for (auto &is : role_is)
    if (is)
      check_petsc(ISDestroy(&is));
  owned_roles.clear();
  pc_ready = false;
  baseline_iterations = 0;
}

template <int dim>
void IterativeFSISolver<dim>::reset_solver_specific_data()
{
  clear_linear_solver();
  FSISolver<dim>::reset_solver_specific_data();
}

template <int dim>
void IterativeFSISolver<dim>::create_sparsity_pattern()
{
  clear_linear_solver();
  FSISolver<dim>::create_sparsity_pattern();
  create_role_index_sets();
  LA::ParVectorType owned(this->locally_owned_dofs, this->mpi_communicator);
  VectorTools::get_position_vector(*this->fixed_mapping, *this->dof_handler,
                                   owned, this->position_mask);
  owned.compress(VectorOperation::insert);
  reference_positions.reinit(this->locally_owned_dofs,
                               this->locally_relevant_dofs,
                               this->mpi_communicator);
  reference_positions = owned;
}

template <int dim>
void IterativeFSISolver<dim>::create_role_index_sets()
{
  AssertThrow(this->dof_handler->n_dofs() <=
                static_cast<types::global_dof_index>(
                  std::numeric_limits<PetscInt>::max()),
              ExcMessage("FSI indices exceed the PETSc integer width."));
  std::array<std::vector<PetscInt>, n_roles> indices;
  for (const auto dof : this->locally_owned_dofs)
  {
    Role role = constrained;
    const auto component = this->dofs_to_component[
      this->locally_relevant_dofs.index_within_set(dof)];
    if (!this->zero_constraints.is_constrained(dof))
    {
      if (this->ordering->is_velocity(component))
        role = velocity;
      else if (this->ordering->is_pressure(component))
        role = pressure;
      else if (this->ordering->is_position(component))
        role = this->coupled_position_dofs.count(dof) ? mesh_boundary :
                                                       mesh_interior;
      else if (this->ordering->is_lambda(component))
      {
        role = multiplier;
        for (unsigned int d = 0; d < dim; ++d)
        {
          if (this->has_local_lambda_accumulator &&
              dof == this->local_lambda_accumulators[d])
            role = force;
          if (!this->param.fsi.zero_mass_model &&
              this->has_cylinder_velocity_dofs &&
              dof == this->local_cylinder_velocity_dofs[d])
            role = body_velocity;
        }
        if (this->param.fsi.rotation.enable && this->has_rotation_angle &&
            dof == this->rotation_angle_dof)
          role = angle;
      }
      else
        AssertThrow(false, ExcMessage("Unclassified free FSI degree of freedom."));
    }
    owned_roles.push_back(role);
    indices[role].push_back(static_cast<PetscInt>(dof));
  }
  types::global_dof_index count = 0;
  for (unsigned int r = 0; r < n_roles; ++r)
  {
    check_petsc(ISCreateGeneral(this->mpi_communicator,
                                static_cast<PetscInt>(indices[r].size()),
                                indices[r].data(), PETSC_COPY_VALUES, &role_is[r]));
    PetscInt size;
    check_petsc(ISGetSize(role_is[r], &size));
    count += size;
    this->pcout << "FSI role " << role_names[r] << ": " << size << std::endl;
  }
  // Every owned index is visited exactly once; ownership is globally disjoint.
  AssertThrow(count == this->dof_handler->n_dofs(),
              ExcMessage("The FSI role index sets do not partition the system."));
}

template <int dim>
void IterativeFSISolver<dim>::initialize_ksp()
{
  const auto &p = this->param.linear_solver.at(this->solver_type);
  check_petsc(KSPCreate(this->mpi_communicator, &ksp));
  check_petsc(KSPSetOptionsPrefix(ksp, "fsi_"));
  check_petsc(KSPSetType(ksp, KSPFGMRES));
  check_petsc(KSPSetPCSide(ksp, PC_RIGHT));
  check_petsc(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  check_petsc(KSPGMRESSetRestart(ksp, p.max_krylov_vectors));
  check_petsc(KSPSetTolerances(ksp, p.relative_tolerance, p.absolute_tolerance,
                              PETSC_DEFAULT, p.max_iterations));
  PC pc;
  check_petsc(KSPGetPC(ksp, &pc));
  using Prec = Parameters::LinearSolver::PreconditionerType;
  switch (p.preconditioner)
  {
    case Prec::none:
      check_petsc(PCSetType(pc, PCNONE));
      break;
    case Prec::ilu:
      AssertThrow(this->mpi_size == 1, ExcMessage("PETSc ILU is serial only."));
      check_petsc(PCSetType(pc, PCILU));
      check_petsc(PCFactorSetLevels(pc, p.ilu_fill_level));
      break;
    case Prec::block_jacobi:
      check_petsc(PCSetType(pc, PCBJACOBI));
      break;
    case Prec::amg:
      AssertThrow(false, ExcMessage("AMG on the complete FSI saddle system is "
                                    "not an admissible baseline preconditioner."));
  }
  check_petsc(KSPSetFromOptions(ksp));
  PetscBool global_amg;
  check_petsc(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(pc),
                                     PCGAMG, &global_amg));
  AssertThrow(!global_amg, ExcMessage("Do not apply AMG to the entire FSI matrix."));
  check_petsc(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(pc),
                                     PCHYPRE, &global_amg));
  AssertThrow(!global_amg, ExcMessage("Use Hypre only on suitable physical blocks, "
                                     "not the complete FSI matrix."));
  PetscBool flexible;
  check_petsc(PetscObjectTypeCompare(reinterpret_cast<PetscObject>(ksp),
                                     KSPFGMRES, &flexible));
  AssertThrow(flexible, ExcMessage("Keep -fsi_ksp_type fgmres for iterative_fsi."));
  // Invariants of the residual acceptance test, including when options are used.
  check_petsc(KSPSetPCSide(ksp, PC_RIGHT));
  check_petsc(KSPSetNormType(ksp, KSP_NORM_UNPRECONDITIONED));
  check_petsc(KSPSetInitialGuessNonzero(ksp, PETSC_FALSE));
}

template <int dim>
void IterativeFSISolver<dim>::solve_linear_system_iterative()
{
  TimerOutput::Scope timer(this->computing_timer, "Solve FSI FGMRES");
  if (!ksp)
    initialize_ksp();
  const auto &p = this->param.linear_solver.at(this->solver_type);
  const double c0 = this->time_handler.get_bdf_coefficients()[0];
  bool reuse = p.reuse && pc_ready && pc_bdf_c0 == c0;
  LA::ParVectorType solution(this->locally_owned_dofs, this->mpi_communicator);
  LA::ParVectorType residual(this->locally_owned_dofs, this->mpi_communicator);
  PetscReal rtol, atol, dtol;
  PetscInt max_iterations;
  check_petsc(KSPGetTolerances(ksp, &rtol, &atol, &dtol, &max_iterations));
  const double rhs_norm = this->system_rhs.l2_norm();
  const double target = std::max(double(atol), double(rtol) * rhs_norm);
  AssertThrow(target > 0. || rhs_norm == 0.,
              ExcMessage("At least one linear tolerance must be positive."));
  this->preconditioner_rebuilt_at_last_solve = false;
  PetscInt iterations = 0;
  double true_residual = 0.;
  for (unsigned int attempt = 0; attempt < 2; ++attempt)
  {
    solution = 0.;
    check_petsc(KSPSetOperators(ksp, this->system_matrix, this->system_matrix));
    check_petsc(KSPSetReusePreconditioner(ksp, reuse ? PETSC_TRUE : PETSC_FALSE));
    this->preconditioner_rebuilt_at_last_solve |= !reuse;
    check_petsc(KSPSolve(ksp, this->system_rhs, solution));
    KSPConvergedReason reason;
    check_petsc(KSPGetConvergedReason(ksp, &reason));
    check_petsc(KSPGetIterationNumber(ksp, &iterations));
    this->system_matrix.vmult(residual, solution);
    residual.sadd(-1., 1., this->system_rhs);
    true_residual = residual.l2_norm();
    const bool accepted = reason > 0 && std::isfinite(true_residual) &&
                          true_residual <= target * (1. + 1e-8);
    if (accepted)
      break;
    pc_ready = false;
    if (reuse && attempt == 0)
    {
      this->pcout << "FSI: rebuilding the reused PC after linear failure."
                  << std::endl;
      reuse = false;
      continue;
    }
    AssertThrow(false, ExcMessage("FGMRES failed the true-residual test: reason=" +
                                  std::to_string(static_cast<int>(reason)) +
                                  ", ||b-J delta||=" +
                                  Utilities::to_string(true_residual) +
                                  ", target=" + Utilities::to_string(target)));
  }
  if (!reuse)
    baseline_iterations = std::max<PetscInt>(iterations, 1);
  pc_ready = iterations <= std::max<PetscInt>(2 * baseline_iterations,
                                              baseline_iterations + 10);
  pc_bdf_c0 = c0;
  this->newton_update = solution;
  this->zero_constraints.distribute(this->newton_update);
  this->pcout << "FSI FGMRES iterations=" << iterations
              << " true_residual=" << std::scientific << std::setprecision(12)
              << true_residual << " target=" << target << std::endl;
  if (compare_mumps)
  {
    TimerOutput::Scope comparison(this->computing_timer, "FSI MUMPS comparison");
    SolverControl control;
    PETScWrappers::SparseDirectMUMPSReuse direct(control, this->mpi_communicator);
    direct.set_prefix("fsi_reference_");
    direct.solve(this->system_matrix, solution, this->system_rhs);
    this->zero_constraints.distribute(solution);
    const double reference_norm = solution.l2_norm();
    solution -= this->newton_update;
    const double difference = solution.l2_norm();
    this->pcout << "FSI MUMPS correction_difference=" << difference
                << " reference_norm=" << reference_norm << std::endl;
    AssertThrow(difference <= comparison_tolerance * std::max(reference_norm, 1e-30),
                ExcMessage("FGMRES and MUMPS corrections disagree."));
  }
}

template <int dim>
void IterativeFSISolver<dim>::diagnostic_residual(LA::ParVectorType &residual)
{
  // The production RHS is deliberately unchanged. Complete the historical
  // homogeneous increment equations ONLY in this diagnostic residual.
  this->assemble_rhs();
  residual = this->system_rhs;
  residual *= -1.;
  if (this->param.fsi.enable_coupling)
  {
    using Coupling = typename Parameters::FSI<dim>::CouplingStrategy;
    const auto coupling = this->param.fsi.coupling;
    const auto &z = this->evaluation_point;
    std::array<double, dim> weights{};
    for (unsigned int d = 0; d < dim; ++d)
      for (const auto &[j, coefficient] : this->lambda_integral_coeffs[d])
        if (this->locally_relevant_dofs.is_element(j))
          weights[d] += coefficient * z[j];
    if (this->has_local_lambda_accumulator)
      for (unsigned int d = 0; d < dim; ++d)
      {
        double value = z[this->local_lambda_accumulators[d]] - weights[d];
        if (this->has_global_accumulator)
          for (const auto other : this->all_lambda_accumulators[d])
            if (other != this->global_lambda_accumulators[d])
              value -= z[other];
        residual[this->local_lambda_accumulators[d]] = value;
      }
    if (this->param.fsi.zero_mass_model && !this->param.fsi.rotation.enable)
      for (const auto &[i, d] : this->coupled_position_dofs)
        if (this->locally_owned_dofs.is_element(i))
        {
          const double displacement = z[i] - reference_positions[i];
          const double k = this->param.fsi.spring_constant;
          double value = displacement;
          if (coupling == Coupling::all_position_to_all_lambda)
            value -= weights[d] / k;
          else if (i != this->local_position_master_dofs[d])
          {
            const auto master = this->local_position_master_dofs[d];
            value -= z[master] - reference_positions[master];
          }
          else if (coupling == Coupling::local_position_master_to_all_lambda)
            value -= weights[d] / k;
          else if (coupling == Coupling::local_position_master_to_lambda_accumulators)
            for (const auto accumulator : this->all_lambda_accumulators[d])
              value -= z[accumulator] / k;
          else if (!this->has_global_master_position_dofs)
          {
            const auto master = this->global_position_master_dofs[d];
            value -= z[master] - reference_positions[master];
          }
          else if (coupling == Coupling::global_position_master_to_all_lambda)
            value -= weights[d] / k;
          else
            value -= z[this->global_lambda_accumulators[d]] / k;
          residual[i] = value;
        }
  }
  for (const auto i : this->locally_owned_dofs)
    if (this->zero_constraints.is_constrained(i))
      residual[i] = 0.;
  residual.compress(VectorOperation::insert);
  AssertThrow(std::isfinite(residual.l2_norm()),
              ExcMessage("The diagnostic FSI residual is not finite."));
}

template <int dim>
void IterativeFSISolver<dim>::check_global_jacobian()
{
  TimerOutput::Scope timer(this->computing_timer, "FSI global Jv check");
  LA::ParVectorType saved(this->locally_owned_dofs, this->mpi_communicator);
  LA::ParVectorType saved_local(saved), saved_rhs(saved), direction(saved);
  LA::ParVectorType plus(saved), minus(saved), action(saved);
  saved = this->evaluation_point;
  saved_local = this->local_evaluation_point;
  saved_rhs = this->system_rhs;
  for (const auto i : this->locally_owned_dofs)
    direction[i] = std::sin(0.31 * static_cast<double>(i + 1));
  direction.compress(VectorOperation::insert);
  this->zero_constraints.distribute(direction);
  const double norm = direction.linfty_norm();
  if (norm == 0.)
    return;
  direction *= 1. / norm;
  double step = fd_step;
  const auto restore = [&]() {
    this->local_evaluation_point = saved_local;
    this->evaluation_point = saved;
    this->system_rhs = saved_rhs;
  };
  try
  {
    bool valid = false;
    for (unsigned int trial = 0; trial < 12; ++trial, step *= 0.5)
    {
      this->local_evaluation_point = saved;
      this->local_evaluation_point.add(step, direction);
      this->evaluation_point = this->local_evaluation_point;
      const bool valid_plus = this->evaluation_point_is_valid();
      if (valid_plus)
        diagnostic_residual(plus);
      this->local_evaluation_point = saved;
      this->local_evaluation_point.add(-step, direction);
      this->evaluation_point = this->local_evaluation_point;
      const bool valid_minus = this->evaluation_point_is_valid();
      if (valid_plus && valid_minus)
      {
        diagnostic_residual(minus);
        valid = true;
        break;
      }
    }
    AssertThrow(valid, ExcMessage("No valid geometry for the global Jv check."));
    plus -= minus;
    plus *= 0.5 / step;
    this->system_matrix.vmult(action, direction);
    for (const auto i : this->locally_owned_dofs)
      if (this->zero_constraints.is_constrained(i))
        action[i] = 0.;
    action.compress(VectorOperation::insert);
    const double scale = std::max(action.l2_norm(), plus.l2_norm());
    action -= plus;
    const double error = action.l2_norm();
    restore();
    this->pcout << "FSI global Jv error=" << error << " scale=" << scale
                << " h=" << step << std::endl;
    const auto &p = this->param.nonlinear_solver;
    AssertThrow(error <= p.analytical_jacobian_absolute_tolerance +
                          p.analytical_jacobian_relative_tolerance * scale,
                ExcMessage("Global FSI Jv check failed, including manual rows. "
                           "See tests/iterative_fsi/KNOWN_ISSUES.md."));
  }
  catch (...)
  {
    restore();
    throw;
  }
}

template <int dim>
void IterativeFSISolver<dim>::assemble_matrix()
{
  FSISolver<dim>::assemble_matrix();
  if (check_jacobian)
    check_global_jacobian();
}

template <int dim>
void IterativeFSISolver<dim>::solver_specific_post_processing()
{
  if (mechanical_diagnostics)
  {
    LA::ParVectorType saved(this->locally_owned_dofs, this->mpi_communicator);
    LA::ParVectorType saved_local(saved), saved_rhs(saved), residual(saved);
    saved = this->evaluation_point;
    saved_local = this->local_evaluation_point;
    saved_rhs = this->system_rhs;
    this->evaluation_point = *this->present_solution;
    this->local_evaluation_point = *this->present_solution;
    try
    {
      diagnostic_residual(residual);
      for (unsigned int r = mesh_boundary; r < constrained; ++r)
      {
        double maximum = 0.;
        unsigned int local = 0;
        for (const auto i : this->locally_owned_dofs)
        {
          if (owned_roles[local++] == r)
            maximum = std::max(maximum, std::abs(double(residual[i])));
        }
        maximum = Utilities::MPI::max(maximum, this->mpi_communicator);
        this->pcout << "FSI mechanical " << role_names[r]
                    << " residual_linf=" << maximum << std::endl;
      }
    }
    catch (...)
    {
      this->evaluation_point = saved;
      this->local_evaluation_point = saved_local;
      this->system_rhs = saved_rhs;
      throw;
    }
    this->evaluation_point = saved;
    this->local_evaluation_point = saved_local;
    this->system_rhs = saved_rhs;
  }
  FSISolver<dim>::solver_specific_post_processing();
}

template class IterativeFSISolver<2>;
template class IterativeFSISolver<3>;
