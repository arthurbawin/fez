
#include <compare_matrix.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
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

template <int dim, bool with_moving_mesh>
CHNSSolver<dim, with_moving_mesh>::CHNSSolver(const ParameterReader<dim> &param)
  : NavierStokesSolver<dim, with_moving_mesh>(param)
{
  if constexpr (with_moving_mesh)
  {
    if (param.finite_elements.use_quads)
      fe = std::make_shared<FESystem<dim>>(
        FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim,
        FE_Q<dim>(param.finite_elements.pressure_degree),
        FE_Q<dim>(param.finite_elements.mesh_position_degree) ^ dim,
        FE_Q<dim>(param.finite_elements.tracer_degree),
        FE_Q<dim>(param.finite_elements.potential_degree));
    else
      fe = std::make_shared<FESystem<dim>>(
        FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^ dim,
        FE_SimplexP<dim>(param.finite_elements.pressure_degree),
        FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^ dim,
        FE_SimplexP<dim>(param.finite_elements.tracer_degree),
        FE_SimplexP<dim>(param.finite_elements.potential_degree));

    this->ordering = std::make_shared<ComponentOrderingCHNS<dim, true>>();
  }
  else
  {
    if (param.finite_elements.use_quads)
      fe = std::make_shared<FESystem<dim>>(
        FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim,
        FE_Q<dim>(param.finite_elements.pressure_degree),
        FE_Q<dim>(param.finite_elements.tracer_degree),
        FE_Q<dim>(param.finite_elements.potential_degree));
    else
      fe = std::make_shared<FESystem<dim>>(
        FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^ dim,
        FE_SimplexP<dim>(param.finite_elements.pressure_degree),
        FE_SimplexP<dim>(param.finite_elements.tracer_degree),
        FE_SimplexP<dim>(param.finite_elements.potential_degree));

    this->ordering = std::make_shared<ComponentOrderingCHNS<dim, false>>();
  }

  this->velocity_extractor =
    FEValuesExtractors::Vector(this->ordering->u_lower);
  this->pressure_extractor =
    FEValuesExtractors::Scalar(this->ordering->p_lower);
  if constexpr (with_moving_mesh)
    this->position_extractor =
      FEValuesExtractors::Vector(this->ordering->x_lower);

  tracer_extractor    = FEValuesExtractors::Scalar(this->ordering->phi_lower);
  potential_extractor = FEValuesExtractors::Scalar(this->ordering->mu_lower);

  this->velocity_mask = fe->component_mask(this->velocity_extractor);
  this->pressure_mask = fe->component_mask(this->pressure_extractor);
  if constexpr (with_moving_mesh)
    this->position_mask = fe->component_mask(this->position_extractor);

  tracer_mask    = fe->component_mask(tracer_extractor);
  potential_mask = fe->component_mask(potential_extractor);

  this->field_names_and_masks["velocity"]  = this->velocity_mask;
  this->field_names_and_masks["pressure"]  = this->pressure_mask;
  this->field_names_and_masks["tracer"]    = this->tracer_mask;
  this->field_names_and_masks["potential"] = this->potential_mask;

  /**
   * Create the initial condition functions
   */
  this->param.initial_conditions.create_initial_velocity(
    this->ordering->u_lower, this->ordering->n_components);
  this->param.initial_conditions.create_initial_chns_tracer(
    this->ordering->phi_lower, this->ordering->n_components);

  // Assign the exact solution
  this->exact_solution =
    std::make_shared<CHNSSolver<dim, with_moving_mesh>::MMSSolution>(
      this->time_handler.current_time, *this->ordering, param.mms);

  if (param.mms_param.enable)
  {
    // Create the MMS source term function and override source terms
    this->source_terms =
      std::make_shared<CHNSSolver<dim, with_moving_mesh>::MMSSourceTerm>(
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
    this->source_terms =
      std::make_shared<CHNSSolver<dim, with_moving_mesh>::SourceTerm>(
        this->time_handler.current_time, *this->ordering, param.source_terms);
  }
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::MMSSourceTerm::vector_value(
  const Point<dim> &p,
  Vector<double>   &values) const
{
  const double phi          = mms.exact_tracer->value(p);
  const double filtered_phi = phi;
  const double rho0         = physical_properties.fluids[0].density;
  const double rho1         = physical_properties.fluids[1].density;
  const double rho  = CahnHilliard::linear_mixing(filtered_phi, rho0, rho1);
  const double eta0 = rho0 * physical_properties.fluids[0].kinematic_viscosity;
  const double eta1 = rho1 * physical_properties.fluids[1].kinematic_viscosity;
  const double eta  = CahnHilliard::linear_mixing(filtered_phi, eta0, eta1);
  const double M    = cahn_hilliard_param.mobility;
  const double diff_flux_factor = M * 0.5 * (rho1 - rho0);
  // const double drhodphi =
  //   CahnHilliard::linear_mixing_derivative(filtered_phi, rho0, rho1);
  const double detadphi =
    CahnHilliard::linear_mixing_derivative(filtered_phi, eta0, eta1);
  const double epsilon = cahn_hilliard_param.epsilon_interface;
  const double sigma_tilde =
    3. / (2. * sqrt(2.)) * cahn_hilliard_param.surface_tension;
  const double sigma_tilde_over_eps  = sigma_tilde / epsilon;
  const double sigma_tilde_times_eps = sigma_tilde * epsilon;
  const auto  &body_force            = cahn_hilliard_param.body_force;

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
  Tensor<1, dim> f = -(rho * (dudt_eulerian + uDotGradu - body_force) +
                       J_flux * grad_u + grad_p - div_viscous + phi * grad_mu);
  for (unsigned int d = 0; d < dim; ++d)
    values[u_lower + d] = f[d];

  // Mass conservation (pressure) source term,
  // for - div(u) + f = 0 -> f = div(u_mms).
  values[p_lower] = mms.exact_velocity->divergence(p);

  if constexpr (with_moving_mesh)
  {
    // Pseudosolid (mesh position) source term
    Tensor<1, dim> f_PS =
      mms.exact_mesh_position
        ->divergence_linear_elastic_stress_variable_coefficients(
          p,
          physical_properties.pseudosolids[0].lame_mu_fun,
          physical_properties.pseudosolids[0].lame_lambda_fun);

    for (unsigned int d = 0; d < dim; ++d)
      values[x_lower + d] = f_PS[d];
  }

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

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim,
                with_moving_mesh>::create_solver_specific_zero_constraints()
{
  for (const auto &[id, bc] : this->param.cahn_hilliard_bc)
  {
    /**
     * Apply manufactured solution for both tracer and potential
     */
    if (bc.type == BoundaryConditions::Type::dirichlet_mms)
    {
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               this->dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(
                                                 this->ordering->n_components),
                                               this->zero_constraints,
                                               tracer_mask);
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               this->dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(
                                                 this->ordering->n_components),
                                               this->zero_constraints,
                                               potential_mask);
    }
  }
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim,
                with_moving_mesh>::create_solver_specific_nonzero_constraints()
{
  for (const auto &[id, bc] : this->param.cahn_hilliard_bc)
  {
    /**
     * Apply manufactured solution for both tracer and potential
     */
    if (bc.type == BoundaryConditions::Type::dirichlet_mms)
    {
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               this->dof_handler,
                                               id,
                                               *this->exact_solution,
                                               this->nonzero_constraints,
                                               tracer_mask);
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               this->dof_handler,
                                               id,
                                               *this->exact_solution,
                                               this->nonzero_constraints,
                                               potential_mask);
    }
  }
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::set_solver_specific_initial_conditions()
{
  const Function<dim> *tracer_fun =
    this->param.initial_conditions.set_to_mms ?
      this->exact_solution.get() :
      this->param.initial_conditions.initial_chns_tracer.get();

  // Set tracer only
  VectorTools::interpolate(*this->moving_mapping,
                           this->dof_handler,
                           *tracer_fun,
                           this->newton_update,
                           tracer_mask);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::set_solver_specific_exact_solution()
{
  // Set tracer and potential
  VectorTools::interpolate(*this->moving_mapping,
                           this->dof_handler,
                           *this->exact_solution,
                           this->local_evaluation_point,
                           tracer_mask);
  VectorTools::interpolate(*this->moving_mapping,
                           this->dof_handler,
                           *this->exact_solution,
                           this->local_evaluation_point,
                           potential_mask);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::create_sparsity_pattern()
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

      // p couples to u , x
      if (this->ordering->is_pressure(i) &&
          (this->ordering->is_velocity(j) || this->ordering->is_position(j)))
        coupling_table[i][j] = DoFTools::always;

      // x couples x,phi,u
      if constexpr (with_moving_mesh)
        if (this->ordering->is_position(i) &&
            (this->ordering->is_position(j) || this->ordering->is_tracer(j) ||
             this->ordering->is_velocity(j)))
          coupling_table[i][j] = DoFTools::always;

      // phi couples to u, phi, mu, x
      if (this->ordering->is_tracer(i))
        if (!this->ordering->is_pressure(j))
          coupling_table[i][j] = DoFTools::always;

      // mu couples to phi, mu, u, x
      if (this->ordering->is_potential(i))
        if (!this->ordering->is_pressure(j))
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

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  this->system_matrix = 0;

  ScratchData scratchData(*this->ordering,
                          *fe,
                          *this->fixed_mapping,
                          *this->moving_mapping,
                          *this->quadrature,
                          *this->face_quadrature,
                          this->time_handler.bdf_coefficients,
                          this->param);
  CopyData    copyData(fe->n_dofs_per_cell());

#if defined(FEZ_WITH_PETSC)
  AssertThrow(
    MultithreadInfo::n_threads() == 1,
    ExcMessage(
      "Assembly is running with more than 1 thread, but uses PETSc wrappers "
      "for parallel matrix and vectors, which are not thread safe."));
#endif
  auto assembly_ptr = this->param.nonlinear_solver.analytic_jacobian ?
                      &CHNSSolver::assemble_local_matrix :
                      &CHNSSolver::assemble_local_matrix_finite_differences;

  // Assemble matrix (multithreaded if supported)
  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  assembly_ptr,
                  &CHNSSolver::copy_local_to_global_matrix,
                  scratchData,
                  copyData);

  this->system_matrix.compress(VectorOperation::add);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::
  assemble_local_matrix_finite_differences(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData                                          &scratch_data,
    CopyData                                             &copy_data)
{
  Verification::compute_local_matrix_finite_differences<dim>(
    cell,
    *this,
    &CHNSSolver::assemble_local_rhs,
    scratch_data,
    copy_data,
    this->evaluation_point,
    this->local_evaluation_point);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
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
  const auto  &body_force            = scratch_data.body_force;

  double                             JxW_fixed, lame_mu = 0., lame_lambda = 0.;
  const double                       alpha = this->param.cahn_hilliard.alpha;
  const double                       beta  = this->param.cahn_hilliard.beta;
  const std::vector<Tensor<1, dim>> *phi_x;
  const std::vector<Tensor<2, dim>> *grad_phi_x, *grad_phi_x_moving;
  const std::vector<double>         *div_phi_x;
  const std::vector<double>         *shape_phi_fixed;
  const std::vector<Tensor<1, dim>> *grad_shape_phi_fixed;
  const Tensor<1, dim>              *source_term_velocity;
  double source_term_pressure, source_term_tracer, source_term_potential;
  const Tensor<1, dim> *phi_x_i, *phi_x_j;
  const Tensor<2, dim> *grad_phi_x_i, *grad_phi_x_j;
  double                div_phi_x_i, div_phi_x_j;
  double                shape_phi_fixed_j;
  const Tensor<1, dim> *grad_shape_phi_fixed_j;

  const double bdf_c0 = this->time_handler.bdf_coefficients[0];

  const unsigned int n_dofs_per_cell = scratch_data.dofs_per_cell;

  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW_moving = scratch_data.JxW_moving[q];
    if constexpr (with_moving_mesh)
      JxW_fixed = scratch_data.JxW_fixed[q];

    const double rho      = scratch_data.density[q];
    const double eta      = scratch_data.dynamic_viscosity[q];
    const double drhodphi = scratch_data.derivative_density_wrt_tracer[q];
    const double detadphi =
      scratch_data.derivative_dynamic_viscosity_wrt_tracer[q];

    const auto &phi_u          = scratch_data.phi_u[q];
    const auto &grad_phi_u     = scratch_data.grad_phi_u[q];
    const auto &sym_grad_phi_u = scratch_data.sym_grad_phi_u[q];
    const auto &div_phi_u      = scratch_data.div_phi_u[q];
    const auto &phi_p          = scratch_data.phi_p[q];
    const auto &phi_phi        = scratch_data.shape_phi[q];
    const auto &grad_phi_phi   = scratch_data.grad_shape_phi[q];
    const auto &phi_mu         = scratch_data.shape_mu[q];
    const auto &grad_phi_mu    = scratch_data.grad_shape_mu[q];

    if constexpr (with_moving_mesh)
    {
      lame_mu               = scratch_data.lame_mu[q];
      lame_lambda           = scratch_data.lame_lambda[q];
      phi_x                 = &scratch_data.phi_x[q];
      grad_phi_x            = &scratch_data.grad_phi_x[q];
      grad_phi_x_moving     = &scratch_data.grad_phi_x_moving[q];
      div_phi_x             = &scratch_data.div_phi_x[q];
      shape_phi_fixed       = &scratch_data.shape_phi_fixed[q];
      grad_shape_phi_fixed  = &scratch_data.grad_shape_phi_fixed[q];
      source_term_velocity  = &scratch_data.source_term_velocity[q];
      source_term_pressure  = scratch_data.source_term_pressure[q];
      source_term_tracer    = scratch_data.source_term_tracer[q];
      source_term_potential = scratch_data.source_term_potential[q];
    }

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];
    const auto &present_velocity_sym_gradients =
      scratch_data.present_velocity_sym_gradients[q];
    const auto &present_velocity_divergence =
      scratch_data.present_velocity_divergence[q];
    const auto &present_pressure_values =
      scratch_data.present_pressure_values[q];

    // Convective velocity (pure eulerian: u, ALE: u - w)
    Tensor<1, dim> u_conv = present_velocity_values;
    if constexpr (with_moving_mesh)
      u_conv -= scratch_data.present_mesh_velocity_values[q];

    const auto u_dot_grad_u = present_velocity_gradients * u_conv;

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratch_data.previous_velocity_values);

    const auto &tracer_value       = scratch_data.tracer_values[q];
    const auto &tracer_gradient    = scratch_data.tracer_gradients[q];
    const auto &potential_value    = scratch_data.potential_values[q];
    const auto &potential_gradient = scratch_data.potential_gradients[q];
    const auto &u_dot_grad_phi = scratch_data.velocity_dot_tracer_gradient[q];

    const double dphidt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, tracer_value, scratch_data.previous_tracer_values);

    // Precomputations of shape functions-independent terms
    const auto to_multiply_by_phi_u_i_phi_phi_j =
      (drhodphi * (dudt + u_dot_grad_u - body_force) + potential_gradient);

    const auto to_multiply_by_phi_u_i_tr_G =
      rho * (dudt - body_force + present_velocity_gradients * u_conv) +
      *source_term_velocity;
    const auto to_multipliy_by_phi_phi_i_tr_G =
      dphidt + u_conv * tracer_gradient + source_term_tracer;

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
    {
      const unsigned int &comp_i = scratch_data.components[i];

      const auto &phi_u_i          = phi_u[i];
      const auto &grad_phi_u_i     = grad_phi_u[i];
      const auto &sym_grad_phi_u_i = sym_grad_phi_u[i];
      const auto &div_phi_u_i      = div_phi_u[i];
      const auto &phi_p_i          = phi_p[i];
      const auto &phi_phi_i        = phi_phi[i];
      const auto &grad_phi_phi_i   = grad_phi_phi[i];
      const auto &phi_mu_i         = phi_mu[i];
      const auto &grad_phi_mu_i    = grad_phi_mu[i];

      if constexpr (with_moving_mesh)
      {
        phi_x_i      = &(*phi_x)[i];
        grad_phi_x_i = &(*grad_phi_x)[i];
        div_phi_x_i  = (*div_phi_x)[i];
      }

      for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
      {
        const unsigned int &comp_j = scratch_data.components[j];
        bool                assemble =
          this->coupling_table[comp_i][comp_j] == DoFTools::always;
        if (!assemble)
          continue;

        const auto &phi_u_j          = phi_u[j];
        const auto &grad_phi_u_j     = grad_phi_u[j];
        const auto &sym_grad_phi_u_j = sym_grad_phi_u[j];
        const auto &div_phi_u_j      = div_phi_u[j];
        const auto &phi_p_j          = phi_p[j];
        const auto &phi_phi_j        = phi_phi[j];
        const auto &grad_phi_phi_j   = grad_phi_phi[j];
        const auto &phi_mu_j         = phi_mu[j];
        const auto &grad_phi_mu_j    = grad_phi_mu[j];

        if constexpr (with_moving_mesh)
        {
          phi_x_j                = &(*phi_x)[j];
          grad_phi_x_j           = &(*grad_phi_x)[j];
          div_phi_x_j            = (*div_phi_x)[j];
          shape_phi_fixed_j      = (*shape_phi_fixed)[j];
          grad_shape_phi_fixed_j = &(*grad_shape_phi_fixed)[j];
        }

        double local_flow_ij = 0.;
        double local_ps_ij   = 0.;

        /**
         * Momentum equation
         */
        if (const_ordering.u_lower <= comp_i && comp_i < const_ordering.u_upper)
        {
          if (const_ordering.u_lower <= comp_j &&
              comp_j < const_ordering.u_upper)
          {
            local_flow_ij +=
              phi_u_i *
              (rho * (bdf_c0 * phi_u_j + grad_phi_u_j * u_conv +
                      present_velocity_gradients * phi_u_j) +
               diffusive_flux_factor * grad_phi_u_j * potential_gradient);

            local_flow_ij +=
              2. * eta * scalar_product(sym_grad_phi_u_i, sym_grad_phi_u_j);
          }
          if (comp_j == const_ordering.p_lower)
          {
            local_flow_ij += -div_phi_u_i * phi_p_j;
          }
          if constexpr (with_moving_mesh)
          {
            if (const_ordering.x_lower <= comp_j &&
                comp_j < const_ordering.x_upper)
            {
              // Variation of momentum equation w.r.t. moving mesh position
              const auto  &G   = (*grad_phi_x_moving)[j];
              const double trG = trace(G);

              local_flow_ij += phi_u_i * to_multiply_by_phi_u_i_tr_G * trG;

              // ALE term
              local_flow_ij += phi_u_i * rho * present_velocity_gradients *
                               (-bdf_c0 * (*phi_x_j) - G * u_conv);

              // Viscous term
              local_flow_ij +=
                2. * eta *
                (scalar_product(-symmetrize(grad_phi_u_i * G),
                                present_velocity_sym_gradients) +
                 scalar_product(sym_grad_phi_u_i,
                                -symmetrize(present_velocity_gradients * G)) +
                 scalar_product(sym_grad_phi_u_i,
                                present_velocity_sym_gradients) *
                   trG);

              // Pressure term
              local_flow_ij +=
                trace(grad_phi_u_i * G) * present_pressure_values -
                div_phi_u_i * present_pressure_values * trG;

              // Diffusive flux term
              local_flow_ij +=
                phi_u_i * diffusive_flux_factor *
                (-(present_velocity_gradients * G) * potential_gradient +
                 present_velocity_gradients *
                   (-transpose(G) * potential_gradient) +
                 (present_velocity_gradients * potential_gradient) * trG);

              // Korteweg term
              local_flow_ij +=
                phi_u_i * (-tracer_value * transpose(G) * potential_gradient +
                           tracer_value * potential_gradient * trG);
            }
          }
          if (comp_j == const_ordering.phi_lower)
          {
            local_flow_ij +=
              phi_u_i * phi_phi_j * to_multiply_by_phi_u_i_phi_phi_j;
            local_flow_ij +=
              2. * detadphi * phi_phi_j *
              scalar_product(sym_grad_phi_u_i, present_velocity_sym_gradients);
          }
          if (comp_j == const_ordering.mu_lower)
          {
            local_flow_ij +=
              phi_u_i * (diffusive_flux_factor * present_velocity_gradients *
                           grad_phi_mu_j +
                         tracer_value * grad_phi_mu_j);
          }
        }

        /**
         * Continuity equation
         */
        if (comp_i == const_ordering.p_lower)
        {
          if (const_ordering.u_lower <= comp_j &&
              comp_j < const_ordering.u_upper)
          {
            // Continuity : variation w.r.t. u
            local_flow_ij += -phi_p_i * div_phi_u_j;
          }

          if constexpr (with_moving_mesh)
          {
            if (const_ordering.x_lower <= comp_j &&
                comp_j < const_ordering.x_upper)
            {
              // Continuity : variation w.r.t. x
              const Tensor<2, dim> &G   = (*grad_phi_x_moving)[j];
              const double          trG = trace(G);

              local_flow_ij +=
                phi_p_i * (source_term_pressure * trG +
                           (trace(present_velocity_gradients * G) -
                            present_velocity_divergence * trG));
            }
          }
        }

        /**
         * Tracer equation
         */
        if (comp_i == const_ordering.phi_lower)
        {
          if (const_ordering.u_lower <= comp_j &&
              comp_j < const_ordering.u_upper)
          {
            // Advection
            local_flow_ij += phi_phi_i * phi_u_j * tracer_gradient;
          }
          if constexpr (with_moving_mesh)
          {
            // Variation of tracer equation w.r.t. x
            if (const_ordering.x_lower <= comp_j &&
                comp_j < const_ordering.x_upper)
            {
              const Tensor<2, dim> &G   = (*grad_phi_x_moving)[j];
              const double          trG = trace(G);

              local_flow_ij += phi_phi_i * to_multipliy_by_phi_phi_i_tr_G * trG;

              // ALE (advection) term
              local_flow_ij +=
                phi_phi_i * ((-bdf_c0) * (*phi_x_j) * tracer_gradient +
                             u_conv * (-(transpose(G)) * tracer_gradient));
              // Laplacian term
              local_flow_ij +=
                mobility *
                ((-(transpose(G)) * grad_phi_phi_i) * potential_gradient +
                 grad_phi_phi_i * (-(transpose(G)) * potential_gradient) +
                 (grad_phi_phi_i * potential_gradient) * trG);
            }
          }
          if (comp_j == const_ordering.phi_lower)
          {
            // Transient
            local_flow_ij += phi_phi_i * bdf_c0 * phi_phi_j;
            // Advection
            local_flow_ij += phi_phi_i * u_conv * grad_phi_phi_j;
          }
          if (comp_j == const_ordering.mu_lower)
          {
            // Diffusion
            local_flow_ij += mobility * grad_phi_mu_j * grad_phi_phi_i;
          }
        }

        /**
         * Potential equation
         */
        if (comp_i == const_ordering.mu_lower)
        {
          if (comp_j == const_ordering.mu_lower)
          {
            // Mass
            local_flow_ij += phi_mu_i * phi_mu_j;
          }
          if (comp_j == const_ordering.phi_lower)
          {
            // Double well
            local_flow_ij += -sigma_tilde_over_eps * phi_mu_i * phi_phi_j *
                             (3. * tracer_value * tracer_value - 1.);
            // Diffusion
            local_flow_ij +=
              -sigma_tilde_times_eps * grad_phi_mu_i * grad_phi_phi_j;
          }
          if constexpr (with_moving_mesh)
          {
            if (const_ordering.x_lower <= comp_j &&
                comp_j < const_ordering.x_upper)
            {
              // Variation of potential equation w.r.t. x
              const Tensor<2, dim> &G   = (*grad_phi_x_moving)[j];
              const double          trG = trace(G);
              local_flow_ij += phi_mu_i * source_term_potential * trG;
              local_flow_ij +=
                phi_mu_i *
                (potential_value - sigma_tilde_over_eps * tracer_value *
                                     (tracer_value * tracer_value - 1.)) *
                trG;
              local_flow_ij +=
                -sigma_tilde_times_eps *
                (scalar_product(grad_phi_mu_i,
                                -transpose(G) * tracer_gradient) +
                 scalar_product(-transpose(G) * grad_phi_mu_i,
                                tracer_gradient) +
                 scalar_product(grad_phi_mu_i, tracer_gradient) * trG);
            }
          }
        }

        /**
         * Pseudo-solid equation
         */
        if constexpr (with_moving_mesh)
        {
          if (const_ordering.x_lower <= comp_i &&
              comp_i < const_ordering.x_upper)
          {
            if (const_ordering.u_lower <= comp_j &&
                comp_j < const_ordering.u_upper)
            {
              // Source term - Velocity part
              local_ps_ij -= (*phi_x_i) * (beta * (phi_u_j * tracer_gradient) *
                                           tracer_gradient);
            }
            if (const_ordering.x_lower <= comp_j &&
                comp_j < const_ordering.x_upper)
            {
              const Tensor<2, dim> &G = (*grad_phi_x_moving)[j];

              // Linear elasticity
              local_ps_ij +=
                lame_lambda * div_phi_x_j * div_phi_x_i +
                lame_mu *
                  scalar_product(*grad_phi_x_j + transpose(*grad_phi_x_j),
                                 *grad_phi_x_i);

              // Variation of source term on current mesh
              local_ps_ij -=
                (*phi_x_i) *
                (beta *
                 ((-bdf_c0) * (*phi_x_j) * tracer_gradient * tracer_gradient +
                  u_conv * ((-transpose(G)) * tracer_gradient) *
                    tracer_gradient +
                  u_dot_grad_phi * ((-transpose(G)) * tracer_gradient)));
            }
            if (comp_j == const_ordering.phi_lower)
            {
              local_ps_ij -=
                (*phi_x_i) *
                (alpha * (shape_phi_fixed_j * (*grad_shape_phi_fixed_j) +
                          shape_phi_fixed_j * (*grad_shape_phi_fixed_j))

                 + beta * ((u_conv * grad_phi_phi_j) * tracer_gradient +
                           u_dot_grad_phi * grad_phi_phi_j));
            }
          }

          local_ps_ij *= JxW_fixed;
        }
        local_flow_ij *= JxW_moving;
        local_matrix(i, j) += local_flow_ij + local_ps_ij;
      }
    }
  }
  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::copy_local_to_global_matrix(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_matrix,
                                                    copy_data.local_dof_indices,
                                                    this->system_matrix);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::compare_analytical_matrix_with_fd()
{
  ScratchData scratchData(*this->ordering,
                          *fe,
                          *this->fixed_mapping,
                          *this->moving_mapping,
                          *this->quadrature,
                          *this->face_quadrature,
                          this->time_handler.bdf_coefficients,
                          this->param);
  CopyData    copyData(fe->n_dofs_per_cell());

  auto errors = Verification::compare_analytical_matrix_with_fd(
    this->dof_handler,
    fe->n_dofs_per_cell(),
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

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  this->system_rhs = 0;

  ScratchData scratchData(*this->ordering,
                          *fe,
                          *this->fixed_mapping,
                          *this->moving_mapping,
                          *this->quadrature,
                          *this->face_quadrature,
                          this->time_handler.bdf_coefficients,
                          this->param);
  CopyData    copyData(fe->n_dofs_per_cell());

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

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
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
    const double JxW_moving = scratch_data.JxW_moving[q];
    const double rho        = scratch_data.density[q];
    const double eta        = scratch_data.dynamic_viscosity[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];
    const auto &present_velocity_sym_gradients =
      scratch_data.present_velocity_sym_gradients[q];

    // Convective velocity (pure eulerian: u, ALE: u - w)
    Tensor<1, dim> u_conv = present_velocity_values;
    if constexpr (with_moving_mesh)
      u_conv -= scratch_data.present_mesh_velocity_values[q];

    const auto u_dot_grad_u = present_velocity_gradients * u_conv;

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

    // Precomputations of shape functions-independent terms
    const auto to_multiply_by_phi_u_i =
      rho * (dudt + u_dot_grad_u - body_force) + diffusive_flux +
      tracer_value * potential_gradient + source_term_velocity;
    const auto to_multiply_by_phi_phi_i =
      dphidt + velocity_dot_tracer_gradient + source_term_tracer;
    const auto to_multiply_by_phi_mu_i =
      potential_value - sigma_tilde_over_eps * phi_cube_minus_phi +
      source_term_potential;

    const auto &phi_p          = scratch_data.phi_p[q];
    const auto &phi_u          = scratch_data.phi_u[q];
    const auto &sym_grad_phi_u = scratch_data.sym_grad_phi_u[q];
    const auto &div_phi_u      = scratch_data.div_phi_u[q];
    const auto &phi_phi        = scratch_data.shape_phi[q];
    const auto &grad_phi_phi   = scratch_data.grad_shape_phi[q];
    const auto &phi_mu         = scratch_data.shape_mu[q];
    const auto &grad_phi_mu    = scratch_data.grad_shape_mu[q];

    //
    // Pseudo-solid related data
    //
    double                             lame_mu = 0., lame_lambda = 0.;
    const double                       alpha = this->param.cahn_hilliard.alpha;
    const double                       beta  = this->param.cahn_hilliard.beta;
    double                             present_displacement_divergence;
    double                             present_trace_strain;
    Tensor<2, dim>                     present_strain;
    const Tensor<2, dim>              *present_position_gradients;
    const Tensor<1, dim>              *source_term_position;
    const std::vector<Tensor<1, dim>> *phi_x;
    const std::vector<Tensor<2, dim>> *grad_phi_x;
    const std::vector<double>         *div_phi_x;
    Tensor<1, dim>                     mesh_forcing;
    double                             tracer_values_fixed;
    const Tensor<1, dim>              *tracer_gradient_fixed;

    if constexpr (with_moving_mesh)
    {
      lame_mu                    = scratch_data.lame_mu[q];
      lame_lambda                = scratch_data.lame_lambda[q];
      phi_x                      = &scratch_data.phi_x[q];
      grad_phi_x                 = &scratch_data.grad_phi_x[q];
      div_phi_x                  = &scratch_data.div_phi_x[q];
      present_position_gradients = &scratch_data.present_position_gradients[q];
      present_displacement_divergence = trace(*present_position_gradients);
      present_strain =
        symmetrize(*present_position_gradients) - unit_symmetric_tensor<dim>();
      present_trace_strain  = present_displacement_divergence - (double)dim;
      source_term_position  = &scratch_data.source_term_position[q];
      tracer_values_fixed   = scratch_data.tracer_values_fixed[q];
      tracer_gradient_fixed = &scratch_data.tracer_gradients_fixed[q];
    }

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const auto &phi_u_i          = phi_u[i];
      const auto &sym_grad_phi_u_i = sym_grad_phi_u[i];
      const auto &div_phi_u_i      = div_phi_u[i];
      const auto &phi_p_i          = phi_p[i];
      const auto &phi_phi_i        = phi_phi[i];
      const auto &grad_phi_phi_i   = grad_phi_phi[i];
      const auto &phi_mu_i         = phi_mu[i];
      const auto &grad_phi_mu_i    = grad_phi_mu[i];

      // Momentum equation
      double local_rhs_flow_i =
        phi_u_i * to_multiply_by_phi_u_i -
        div_phi_u_i * present_pressure_values +
        2. * eta *
          scalar_product(sym_grad_phi_u_i, present_velocity_sym_gradients);

      // Continuity equation
      local_rhs_flow_i +=
        phi_p_i * (-present_velocity_divergence + source_term_pressure);

      // Tracer equation
      local_rhs_flow_i += phi_phi_i * to_multiply_by_phi_phi_i +
                          grad_phi_phi_i * mobility * potential_gradient;

      // Potential equation
      local_rhs_flow_i +=
        phi_mu_i * to_multiply_by_phi_mu_i -
        grad_phi_mu_i * sigma_tilde_times_eps * tracer_gradient;

      // Pseudo-solid
      double local_rhs_ps_i = 0.;
      if constexpr (with_moving_mesh)
      {
        // Body force to attract the mesh towards the tracer interface
        // Still testing for a good model of body force
        mesh_forcing =
          alpha * (tracer_values_fixed * (*tracer_gradient_fixed)) +
          beta * ((u_conv * tracer_gradient) * tracer_gradient);

        // Linear elasticity
        local_rhs_ps_i +=
          lame_lambda * present_trace_strain * (*div_phi_x)[i] +
          2 * lame_mu * scalar_product(present_strain, (*grad_phi_x)[i]) +
          (*phi_x)[i] * (*source_term_position - mesh_forcing);

        local_rhs_ps_i *= scratch_data.JxW_fixed[q];
      }

      local_rhs_flow_i *= JxW_moving;
      local_rhs(i) -= local_rhs_flow_i + local_rhs_ps_i;
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
        }
      }
    }

  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::copy_local_to_global_rhs(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_rhs,
                                                    copy_data.local_dof_indices,
                                                    this->system_rhs);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::compute_solver_specific_errors()
{
  const unsigned int n_active_cells = this->triangulation.n_active_cells();
  Vector<double>     cellwise_errors(n_active_cells);

  const ComponentSelectFunction<dim> tracer_comp_select(
    this->ordering->phi_lower, this->ordering->n_components);
  const ComponentSelectFunction<dim> potential_comp_select(
    this->ordering->mu_lower, this->ordering->n_components);

  this->compute_and_add_errors(*this->moving_mapping,
                               *this->exact_solution,
                               cellwise_errors,
                               tracer_comp_select,
                               "phi");
  this->compute_and_add_errors(*this->moving_mapping,
                               *this->exact_solution,
                               cellwise_errors,
                               potential_comp_select,
                               "mu");
}

// Explicit instantiation
template class CHNSSolver<2, false>;
template class CHNSSolver<3, false>;
template class CHNSSolver<2, true>;
template class CHNSSolver<3, true>;
