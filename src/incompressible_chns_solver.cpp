
#include <compare_matrix.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <incompressible_chns_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <scratch_data.h>
#include <utilities.h>

template <int dim>
CHNSSolver<dim>::CHNSSolver(const ParameterReader<dim> &param)
  : NavierStokesSolver<dim>(param, false)
  , fe(FE_SimplexP<dim>(param.finite_elements.velocity_degree), // Velocity
       dim,
       FE_SimplexP<dim>(param.finite_elements.pressure_degree), // Pressure
       1,
       FE_SimplexP<dim>(param.finite_elements.tracer_degree), // Tracer
       1,
       FE_SimplexP<dim>(param.finite_elements.potential_degree), // Potential
       1)
{
  this->ordering = std::make_shared<ComponentOrderingCHNS<dim>>();

  this->velocity_extractor =
    FEValuesExtractors::Vector(this->ordering->u_lower);
  this->pressure_extractor =
    FEValuesExtractors::Scalar(this->ordering->p_lower);
  tracer_extractor    = FEValuesExtractors::Scalar(this->ordering->phi_lower);
  potential_extractor = FEValuesExtractors::Scalar(this->ordering->mu_lower);

  this->velocity_mask = fe.component_mask(this->velocity_extractor);
  this->pressure_mask = fe.component_mask(this->pressure_extractor);
  tracer_mask         = fe.component_mask(tracer_extractor);
  potential_mask      = fe.component_mask(potential_extractor);

  /**
   * This solver uses a fixed mapping only.
   */
  mapping = this->fixed_mapping.get();

  /**
   * Create the initial condition functions
   */
  this->param.initial_conditions.create_initial_velocity(
    this->ordering->u_lower, this->ordering->n_components);
  this->param.initial_conditions.create_initial_chns_tracer(
    this->ordering->phi_lower, this->ordering->n_components);

  if (param.mms_param.enable)
  {
    // Assign the manufactured solution
    this->exact_solution = std::make_shared<CHNSSolver<dim>::MMSSolution>(
      this->time_handler.current_time, *this->ordering, param.mms);

    // Create the MMS source term function and override source terms
    this->source_terms = std::make_shared<CHNSSolver<dim>::MMSSourceTerm>(
      this->time_handler.current_time, *this->ordering, param);

    // Create entry in error handler for tracer and potential
    for (auto norm : this->param.mms_param.norms_to_compute)
    {
      this->error_handlers[norm]->create_entry("phi");
      this->error_handlers[norm]->create_entry("mu");
    }
  }
  else
  {
    this->source_terms = std::make_shared<CHNSSolver<dim>::SourceTerm>(
      this->time_handler.current_time, *this->ordering, param.source_terms);
    this->exact_solution = std::make_shared<Functions::ZeroFunction<dim>>(
      this->ordering->n_components);
  }
}

template <int dim>
void CHNSSolver<dim>::MMSSourceTerm::vector_value(const Point<dim> &p,
                                                  Vector<double> &values) const
{
  const double phi          = mms.exact_tracer->value(p);
  const double filtered_phi = phi;
  const double rho0         = physical_properties.fluids[0].density;
  const double rho1         = physical_properties.fluids[1].density;
  const double rho  = cahn_hilliard_linear_mixing(filtered_phi, rho0, rho1);
  const double eta0 = rho0 * physical_properties.fluids[0].kinematic_viscosity;
  const double eta1 = rho1 * physical_properties.fluids[1].kinematic_viscosity;
  const double eta  = cahn_hilliard_linear_mixing(filtered_phi, eta0, eta1);
  const double M    = cahn_hilliard_param.mobility;
  const double diff_flux_factor = M * 0.5 * (rho1 - rho0);
  // const double drhodphi =
  //   cahn_hilliard_linear_mixing_derivative(filtered_phi, rho0, rho1);
  const double detadphi =
    cahn_hilliard_linear_mixing_derivative(filtered_phi, eta0, eta1);
  const double epsilon = cahn_hilliard_param.epsilon_interface;
  const double sigma_tilde =
    3. / (2. * sqrt(2.)) * cahn_hilliard_param.surface_tension;
  const double sigma_tilde_over_eps  = sigma_tilde / epsilon;
  const double sigma_tilde_times_eps = sigma_tilde * epsilon;
  const auto &body_force = cahn_hilliard_param.body_force;

  Tensor<1, dim> u, dudt_eulerian;
  for (unsigned int d = 0; d < dim; ++d)
  {
    dudt_eulerian[d] = mms.exact_velocity->time_derivative(p, d);
    u[d]             = mms.exact_velocity->value(p, d);
  }

  // Use convention (grad_u)_ij := dvj/dxi
  Tensor<2, dim> grad_u      = mms.exact_velocity->gradient_vj_xi(p);
  Tensor<1, dim> lap_u       = mms.exact_velocity->vector_laplacian(p);
  Tensor<1, dim> grad_div_u  = mms.exact_velocity->grad_div(p);
  Tensor<1, dim> grad_p      = mms.exact_pressure->gradient(p);
  Tensor<1, dim> uDotGradu   = u * grad_u;
  Tensor<1, dim> grad_mu     = mms.exact_potential->gradient(p);
  Tensor<1, dim> grad_phi    = mms.exact_tracer->gradient(p);
  Tensor<1, dim> J_flux      = diff_flux_factor * grad_mu;
  Tensor<1, dim> div_viscous = (eta * (lap_u + grad_div_u) +
                                2. * detadphi * grad_phi * symmetrize(grad_u));

  // Navier-Stokes momentum (velocity) source term
  Tensor<1, dim> f = -(rho * (dudt_eulerian + uDotGradu + body_force) + J_flux * grad_u +
                       grad_p - div_viscous + phi * grad_mu);
  for (unsigned int d = 0; d < dim; ++d)
    values[u_lower + d] = f[d];

  // Mass conservation (pressure) source term,
  // for - div(u) + f = 0 -> f = div(u_mms).
  values[p_lower] = mms.exact_velocity->divergence(p);

  // Tracer source term
  const double dphidt = mms.exact_tracer->time_derivative(p);
  const double lap_mu = mms.exact_potential->laplacian(p);
  values[phi_lower]   = -(dphidt + u * grad_phi - M * lap_mu);

  // Potential source term
  const double mu      = mms.exact_potential->value(p);
  const double lap_phi = mms.exact_tracer->laplacian(p);
  values[mu_lower]     = -(mu - sigma_tilde_over_eps * phi * (phi * phi - 1.) +
                       sigma_tilde_times_eps * lap_phi);
}

template <int dim>
void CHNSSolver<dim>::create_solver_specific_zero_constraints()
{
  for (const auto &[id, bc] : this->param.cahn_hilliard_bc)
  {
    /**
     * Apply manufactured solution for both tracer and potential
     */
    if (bc.type == BoundaryConditions::Type::dirichlet_mms)
    {
      VectorTools::interpolate_boundary_values(*mapping,
                                               this->dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(
                                                 this->ordering->n_components),
                                               this->zero_constraints,
                                               tracer_mask);
      VectorTools::interpolate_boundary_values(*mapping,
                                               this->dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(
                                                 this->ordering->n_components),
                                               this->zero_constraints,
                                               potential_mask);
    }
  }
}

template <int dim>
void CHNSSolver<dim>::create_solver_specific_nonzero_constraints()
{
  for (const auto &[id, bc] : this->param.cahn_hilliard_bc)
  {
    /**
     * Apply manufactured solution for both tracer and potential
     */
    if (bc.type == BoundaryConditions::Type::dirichlet_mms)
    {
      VectorTools::interpolate_boundary_values(*mapping,
                                               this->dof_handler,
                                               id,
                                               *this->exact_solution,
                                               this->nonzero_constraints,
                                               tracer_mask);
      VectorTools::interpolate_boundary_values(*mapping,
                                               this->dof_handler,
                                               id,
                                               *this->exact_solution,
                                               this->nonzero_constraints,
                                               potential_mask);
    }
  }
}

template <int dim>
void CHNSSolver<dim>::set_solver_specific_initial_conditions()
{
  const Function<dim> *tracer_fun =
    this->param.initial_conditions.set_to_mms ?
      this->exact_solution.get() :
      this->param.initial_conditions.initial_chns_tracer.get();

  // Set tracer only
  VectorTools::interpolate(
    *mapping, this->dof_handler, *tracer_fun, this->newton_update, tracer_mask);
}

template <int dim>
void CHNSSolver<dim>::set_solver_specific_exact_solution()
{
  // Set tracer and potential
  VectorTools::interpolate(*mapping,
                           this->dof_handler,
                           *this->exact_solution,
                           this->local_evaluation_point,
                           tracer_mask);
  VectorTools::interpolate(*mapping,
                           this->dof_handler,
                           *this->exact_solution,
                           this->local_evaluation_point,
                           potential_mask);
}

template <int dim>
void CHNSSolver<dim>::create_sparsity_pattern()
{
  DynamicSparsityPattern dsp(this->locally_relevant_dofs);

  const unsigned int n_components   = this->ordering->n_components;
  auto              &coupling_table = this->coupling_table;
  coupling_table = Table<2, DoFTools::Coupling>(n_components, n_components);
  for (unsigned int i = 0; i < n_components; ++i)
    for (unsigned int j = 0; j < n_components; ++j)
    {
      coupling_table[i][j] = DoFTools::none;

      // u couples to all variables
      if (this->ordering->is_velocity(i))
        coupling_table[i][j] = DoFTools::always;

      // p couples to u only
      if (this->ordering->is_pressure(i) && this->ordering->is_velocity(j))
        coupling_table[i][j] = DoFTools::always;

      // phi couples to u, phi, mu
      if (this->ordering->is_tracer(i))
        if (!this->ordering->is_pressure(j))
          coupling_table[i][j] = DoFTools::always;

      // mu couples to phi, mu
      if (this->ordering->is_potential(i))
        if (this->ordering->is_tracer(j) || this->ordering->is_potential(j))
          coupling_table[i][j] = DoFTools::always;
    }

  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  coupling_table,
                                  dsp,
                                  this->nonzero_constraints,
                                  /* keep_constrained_dofs = */ false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             this->locally_owned_dofs,
                                             this->mpi_communicator,
                                             this->locally_relevant_dofs);
  this->system_matrix.reinit(this->locally_owned_dofs,
                             this->locally_owned_dofs,
                             dsp,
                             this->mpi_communicator);
}

template <int dim>
void CHNSSolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  this->system_matrix = 0;

  ScratchDataCHNS<dim> scratchData(*this->ordering,
                                   fe,
                                   *mapping,
                                   this->quadrature,
                                   this->face_quadrature,
                                   this->time_handler.bdf_coefficients,
                                   this->param);
  CopyData             copyData(fe.n_dofs_per_cell());

#if defined(FEZ_WITH_PETSC)
  AssertThrow(
    MultithreadInfo::n_threads() == 1,
    ExcMessage(
      "Assembly is running with more than 1 thread, but uses PETSc wrappers "
      "for parallel matrix and vectors, which are not thread safe."));
#endif

  // Assemble matrix (multithreaded if supported)
  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  &CHNSSolver::assemble_local_matrix,
                  &CHNSSolver::copy_local_to_global_matrix,
                  scratchData,
                  copyData);

  this->system_matrix.compress(VectorOperation::add);
}

template <int dim>
void CHNSSolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataCHNS<dim>                                 &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      this->previous_solutions,
                      this->source_terms,
                      this->exact_solution);

  auto &local_matrix = copy_data.local_matrix;
  local_matrix       = 0;

  /**
   * Material parameters
   */
  const double mobility = scratch_data.mobility;
  const double sigma_tilde_over_eps =
    scratch_data.sigma_tilde / scratch_data.epsilon;
  const double sigma_tilde_times_eps =
    scratch_data.sigma_tilde * scratch_data.epsilon;
  const double diffusive_flux_factor = scratch_data.diffusive_flux_factor;
  const auto &body_force = scratch_data.body_force;

  const double bdf_c0 = this->time_handler.bdf_coefficients[0];

  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW      = scratch_data.JxW_moving[q];
    const double rho      = scratch_data.density[q];
    const double eta      = scratch_data.dynamic_viscosity[q];
    const double drhodphi = scratch_data.derivative_density_wrt_tracer[q];
    const double detadphi =
      scratch_data.derivative_dynamic_viscosity_wrt_tracer[q];

    const auto &phi_u        = scratch_data.phi_u[q];
    const auto &grad_phi_u   = scratch_data.grad_phi_u[q];
    const auto &div_phi_u    = scratch_data.div_phi_u[q];
    const auto &phi_p        = scratch_data.phi_p[q];
    const auto &phi_phi      = scratch_data.shape_phi[q];
    const auto &grad_phi_phi = scratch_data.grad_shape_phi[q];
    const auto &phi_mu       = scratch_data.shape_mu[q];
    const auto &grad_phi_mu  = scratch_data.grad_shape_mu[q];

    const auto &sym_grad_phi_u = scratch_data.sym_grad_phi_u[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];
    const auto &present_velocity_sym_gradients =
      scratch_data.present_velocity_sym_gradients[q];

    const auto &tracer_value       = scratch_data.tracer_values[q];
    const auto &tracer_gradient    = scratch_data.tracer_gradients[q];
    const auto &potential_gradient = scratch_data.potential_gradients[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const unsigned int comp_i   = scratch_data.components[i];
      const bool         i_is_u   = this->ordering->is_velocity(comp_i);
      const bool         i_is_p   = this->ordering->is_pressure(comp_i);
      const bool         i_is_phi = this->ordering->is_tracer(comp_i);
      const bool         i_is_mu  = this->ordering->is_potential(comp_i);

      for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
      {
        const unsigned int comp_j = scratch_data.components[j];
        bool               assemble =
          this->coupling_table[comp_i][comp_j] == DoFTools::always;
        if (!assemble)
          continue;

        const bool j_is_u   = this->ordering->is_velocity(comp_j);
        const bool j_is_p   = this->ordering->is_pressure(comp_j);
        const bool j_is_phi = this->ordering->is_tracer(comp_j);
        const bool j_is_mu  = this->ordering->is_potential(comp_j);

        double local_matrix_ij = 0.;

        /**
         * Momentum equation
         */
        if (i_is_u)
        {
          if (j_is_u)
          {
            // Time-dependent
            local_matrix_ij += rho * bdf_c0 * phi_u[i] * phi_u[j];

            // Convection
            local_matrix_ij += rho *
                               (grad_phi_u[j] * present_velocity_values +
                                present_velocity_gradients * phi_u[j]) *
                               phi_u[i];

            // Diffusive flux
            local_matrix_ij += phi_u[i] * diffusive_flux_factor *
                               grad_phi_u[j] * potential_gradient;

            // Diffusion
            local_matrix_ij +=
              2. * eta * scalar_product(grad_phi_u[i], sym_grad_phi_u[j]);
          }
          if (j_is_p)
          {
            // Pressure gradient
            local_matrix_ij += -div_phi_u[i] * phi_p[j];
          }
          if (j_is_phi)
          {
            // Convection
            local_matrix_ij +=
              drhodphi * phi_phi[j] *
              (present_velocity_gradients * present_velocity_values) * phi_u[i];
            // Body force
            local_matrix_ij +=
              phi_u[i] * drhodphi * phi_phi[j] * body_force;
            // Diffusion
            local_matrix_ij +=
              2. * detadphi * phi_phi[j] *
              scalar_product(grad_phi_u[i], present_velocity_sym_gradients);
            // Surface tension
            local_matrix_ij += phi_u[i] * phi_phi[j] * potential_gradient;
          }
          if (j_is_mu)
          {
            // Diffusive flux
            local_matrix_ij += phi_u[i] * diffusive_flux_factor *
                               present_velocity_gradients * grad_phi_mu[j];
            // Surface tension
            local_matrix_ij += phi_u[i] * tracer_value * grad_phi_mu[j];
          }
        }

        /**
         * Continuity equation
         */
        if (i_is_p && j_is_u)
        {
          // Continuity : variation w.r.t. u
          local_matrix_ij += -phi_p[i] * div_phi_u[j];
        }

        /**
         * Tracer equation
         */
        if (i_is_phi)
        {
          if (j_is_u)
          {
            // Advection
            local_matrix_ij += phi_phi[i] * phi_u[j] * tracer_gradient;
          }
          if (j_is_phi)
          {
            // Transient
            local_matrix_ij += phi_phi[i] * bdf_c0 * phi_phi[j];
            // Advection
            local_matrix_ij +=
              phi_phi[i] * present_velocity_values * grad_phi_phi[j];
          }
          if (j_is_mu)
          {
            // Diffusion
            local_matrix_ij += mobility * grad_phi_mu[j] * grad_phi_phi[i];
          }
        }

        /**
         * Potential equation
         */
        if (i_is_mu)
        {
          if (j_is_mu)
          {
            // Mass
            local_matrix_ij += phi_mu[i] * phi_mu[j];
          }
          if (j_is_phi)
          {
            // Double well
            local_matrix_ij += -sigma_tilde_over_eps * phi_mu[i] * phi_phi[j] *
                               (3. * tracer_value * tracer_value - 1.);
            // Diffusion
            local_matrix_ij +=
              -sigma_tilde_times_eps * grad_phi_mu[i] * grad_phi_phi[j];
          }
        }

        local_matrix_ij *= JxW;
        local_matrix(i, j) += local_matrix_ij;
      }
    }
  }
  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void CHNSSolver<dim>::copy_local_to_global_matrix(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_matrix,
                                                    copy_data.local_dof_indices,
                                                    this->system_matrix);
}

template <int dim>
void CHNSSolver<dim>::compare_analytical_matrix_with_fd()
{
  ScratchDataCHNS<dim> scratchData(*this->ordering,
                                   fe,
                                   *mapping,
                                   this->quadrature,
                                   this->face_quadrature,
                                   this->time_handler.bdf_coefficients,
                                   this->param);
  CopyData             copyData(fe.n_dofs_per_cell());

  auto errors = Verification::compare_analytical_matrix_with_fd(
    this->dof_handler,
    fe.n_dofs_per_cell(),
    *this,
    &CHNSSolver::assemble_local_matrix,
    &CHNSSolver::assemble_local_rhs,
    scratchData,
    copyData,
    this->present_solution,
    this->evaluation_point,
    this->local_evaluation_point,
    this->mpi_communicator,
    this->param.output.output_prefix,
    false,
    this->param.debug.analytical_jacobian_absolute_tolerance,
    this->param.debug.analytical_jacobian_relative_tolerance);

  this->pcout << "Max absolute error analytical vs fd matrix is "
              << errors.first << std::endl;

  // Only print relative error if absolute is too large
  if (errors.first > this->param.debug.analytical_jacobian_absolute_tolerance)
    this->pcout << "Max relative error analytical vs fd matrix is "
                << errors.second << std::endl;
}

template <int dim>
void CHNSSolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  this->system_rhs = 0;

  ScratchDataCHNS<dim> scratchData(*this->ordering,
                                   fe,
                                   *mapping,
                                   this->quadrature,
                                   this->face_quadrature,
                                   this->time_handler.bdf_coefficients,
                                   this->param);
  CopyData             copyData(fe.n_dofs_per_cell());

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  &CHNSSolver::assemble_local_rhs,
                  &CHNSSolver::copy_local_to_global_rhs,
                  scratchData,
                  copyData);

  this->system_rhs.compress(VectorOperation::add);
}

template <int dim>
void CHNSSolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchDataCHNS<dim>                                 &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      this->previous_solutions,
                      this->source_terms,
                      this->exact_solution);

  auto &local_rhs = copy_data.local_rhs;
  local_rhs       = 0;

  const double mobility = scratch_data.mobility;
  const double sigma_tilde_over_eps =
    scratch_data.sigma_tilde / scratch_data.epsilon;
  const double sigma_tilde_times_eps =
    scratch_data.sigma_tilde * scratch_data.epsilon;
  const auto &body_force = scratch_data.body_force;

  //
  // Volume contributions
  //
  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW = scratch_data.JxW_moving[q];
    const double rho = scratch_data.density[q];
    const double eta = scratch_data.dynamic_viscosity[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];
    const auto &present_velocity_sym_gradients =
      scratch_data.present_velocity_sym_gradients[q];
    const auto &present_pressure_values =
      scratch_data.present_pressure_values[q];
    const auto &source_term_velocity  = scratch_data.source_term_velocity[q];
    const auto &source_term_pressure  = scratch_data.source_term_pressure[q];
    const auto &source_term_tracer    = scratch_data.source_term_tracer[q];
    const auto &source_term_potential = scratch_data.source_term_potential[q];
    const auto &present_velocity_divergence =
      scratch_data.present_velocity_divergence[q];

    const auto &diffusive_flux     = scratch_data.diffusive_flux[q];
    const auto &tracer_value       = scratch_data.tracer_values[q];
    const auto &tracer_gradient    = scratch_data.tracer_gradients[q];
    const auto &potential_value    = scratch_data.potential_values[q];
    const auto &potential_gradient = scratch_data.potential_gradients[q];
    const auto &velocity_dot_tracer_gradient =
      scratch_data.velocity_dot_tracer_gradient[q];
    const double phi_cube_minus_phi =
      tracer_value * (tracer_value * tracer_value - 1.);

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratch_data.previous_velocity_values);
    const double dphidt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, tracer_value, scratch_data.previous_tracer_values);

    const auto &phi_p        = scratch_data.phi_p[q];
    const auto &phi_u        = scratch_data.phi_u[q];
    const auto &grad_phi_u   = scratch_data.grad_phi_u[q];
    const auto &div_phi_u    = scratch_data.div_phi_u[q];
    const auto &phi_phi      = scratch_data.shape_phi[q];
    const auto &grad_phi_phi = scratch_data.grad_shape_phi[q];
    const auto &phi_mu       = scratch_data.shape_mu[q];
    const auto &grad_phi_mu  = scratch_data.grad_shape_mu[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const auto &phi_u_i        = phi_u[i];
      const auto &grad_phi_u_i   = grad_phi_u[i];
      const auto &div_phi_u_i    = div_phi_u[i];
      const auto &phi_p_i        = phi_p[i];
      const auto &phi_phi_i      = phi_phi[i];
      const auto &grad_phi_phi_i = grad_phi_phi[i];
      const auto &phi_mu_i       = phi_mu[i];
      const auto &grad_phi_mu_i  = grad_phi_mu[i];

      double local_rhs_i = -(

        /**
         * Momentum equation
         */

        // Transient
        rho * phi_u_i * dudt

        // Convection
        + rho * (present_velocity_gradients * present_velocity_values) * phi_u_i

        // Body force
        + rho * phi_u_i * body_force

        // Diffusive flux
        + phi_u_i * diffusive_flux

        // Diffusion
        +
        2. * eta * scalar_product(grad_phi_u_i, present_velocity_sym_gradients)

        // Pressure gradient
        - div_phi_u_i * present_pressure_values

        // Surface tension phi * grad(mu)
        + phi_u_i * tracer_value * potential_gradient

        // Source term
        + source_term_velocity * phi_u_i

        /**
         * Continuity equation
         */

        // Continuity
        - present_velocity_divergence * phi_p_i

        // Source term
        + source_term_pressure * phi_p_i

        /**
         * Tracer equation
         */

        // Transient and advection
        + phi_phi_i * (dphidt + velocity_dot_tracer_gradient)

        // Diffusion
        + mobility * potential_gradient * grad_phi_phi_i

        // Source term
        + phi_phi_i * source_term_tracer

        /**
         * Potential equation
         */

        // Mass and double well
        +
        phi_mu_i * (potential_value - sigma_tilde_over_eps * phi_cube_minus_phi)

        // Diffusion
        - sigma_tilde_times_eps * grad_phi_mu_i * tracer_gradient

        // Source term
        + phi_mu_i * source_term_potential);

      local_rhs_i *= JxW;
      local_rhs(i) += local_rhs_i;
    }
  }

  //
  // Face contributions
  //
  if (cell->at_boundary())
    for (const auto i_face : cell->face_indices())
    {
      const auto &face = cell->face(i_face);
      if (face->at_boundary())
      {
        // Open boundary condition with prescribed manufactured solution
        if (this->param.fluid_bc.at(scratch_data.face_boundary_id[i_face])
              .type == BoundaryConditions::Type::open_mms)
        {
          DEAL_II_NOT_IMPLEMENTED();
          // for (unsigned int q = 0; q < scratch_data.n_faces_q_points; ++q)
          // {
          //   const double face_JxW = scratch_data.face_JxW_moving[i_face][q];
          //   const auto  &n        =
          //   scratch_data.face_normals_moving[i_face][q];

          //   const auto &grad_u_exact =
          //     scratch_data.exact_face_velocity_gradients[i_face][q];
          //   const double p_exact =
          //     scratch_data.exact_face_pressure_values[i_face][q];

          //   // This is an open boundary condition, not a traction,
          //   // involving only grad_u_exact and not the symmetric gradient.
          //   const auto sigma_dot_n = -p_exact * n + nu * grad_u_exact * n;

          //   const auto &phi_u_face = scratch_data.phi_u_face[i_face][q];

          //   for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
          //   {
          //     local_rhs(i) -= -phi_u_face[i] * sigma_dot_n * face_JxW;
          //   }
          // }
        }
      }
    }

  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void CHNSSolver<dim>::copy_local_to_global_rhs(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_rhs,
                                                    copy_data.local_dof_indices,
                                                    this->system_rhs);
}

template <int dim>
void CHNSSolver<dim>::compute_solver_specific_errors()
{
  const unsigned int n_active_cells = this->triangulation.n_active_cells();
  Vector<double>     cellwise_errors(n_active_cells);

  const ComponentSelectFunction<dim> tracer_comp_select(
    this->ordering->phi_lower, this->ordering->n_components);
  const ComponentSelectFunction<dim> potential_comp_select(
    this->ordering->mu_lower, this->ordering->n_components);

  this->compute_and_add_errors(*mapping,
                               *this->exact_solution,
                               cellwise_errors,
                               tracer_comp_select,
                               "phi");
  this->compute_and_add_errors(*mapping,
                               *this->exact_solution,
                               cellwise_errors,
                               potential_comp_select,
                               "mu");
}

template <int dim>
void CHNSSolver<dim>::output_results()
{
  TimerOutput::Scope t(this->computing_timer, "Write outputs");

  if (this->param.output.write_results)
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");
    solution_names.push_back("tracer");
    solution_names.push_back("potential");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int i = 0; i < 3; ++i)
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(this->dof_handler);
    data_out.add_data_vector(this->present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    //
    // Partition
    //
    Vector<float> subdomain(this->triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = this->triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(*mapping, 2);

    // Export regular time step
    data_out.write_vtu_with_pvtu_record(
      this->param.output.output_dir,
      this->param.output.output_prefix,
      this->time_handler.current_time_iteration,
      this->mpi_communicator,
      2);
  }
}

// Explicit instantiation
template class CHNSSolver<2>;
template class CHNSSolver<3>;