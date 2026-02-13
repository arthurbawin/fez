
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

  Tensor<1, dim> u, dudt_eulerian, mesh_velocity;
  for (unsigned int d = 0; d < dim; ++d)
  {
    dudt_eulerian[d] = mms.exact_velocity->time_derivative(p, d);
    u[d]             = mms.exact_velocity->value(p, d);
    // if constexpr (with_moving_mesh)
    // mesh_velocity[d] = mms.exact_mesh_position->value(p, d);
  }
  // if constexpr (with_moving_mesh)
  // u -= mesh_velocity;

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
    // We solve -div(sigma) + f = 0, so no need to put a -1 in front of f
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

  const double bdf_c0 = this->time_handler.bdf_coefficients[0];

  const unsigned int n_dofs_per_cell = scratch_data.dofs_per_cell;

  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW_moving = scratch_data.JxW_moving[q];

    const double rho      = scratch_data.density[q];
    const double eta      = scratch_data.dynamic_viscosity[q];
    const double drhodphi = scratch_data.derivative_density_wrt_tracer[q];
    const double detadphi =
      scratch_data.derivative_dynamic_viscosity_wrt_tracer[q];

    const auto &phi_u      = scratch_data.phi_u[q];
    const auto &grad_phi_u = scratch_data.grad_phi_u[q];
    const auto &div_phi_u  = scratch_data.div_phi_u[q];
    const auto &phi_p      = scratch_data.phi_p[q];

    // x necessary fct
    const std::vector<Tensor<1, dim>> *phi_x      = nullptr;
    const std::vector<Tensor<2, dim>> *grad_phi_x = nullptr;
    const std::vector<double>         *div_phi_x  = nullptr;
    const std::vector<Tensor<2, dim>> *grad_phi_x_moving = nullptr; // ∇_x (δx)

    if constexpr (with_moving_mesh)
    {
      phi_x             = &scratch_data.phi_x[q];
      grad_phi_x        = &scratch_data.grad_phi_x[q];
      grad_phi_x_moving = &scratch_data.grad_phi_x_moving[q];
      div_phi_x         = &scratch_data.div_phi_x[q];
      // // ---- DEBUG: compare grad_phi_x vs grad_phi_x_moving ----
      // if (cell->active_cell_index() == 0 && q == 0)
      // {
      //   double max_diff = 0.0;
      //   unsigned int j_max = numbers::invalid_unsigned_int;

      //   for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
      //     if (this->ordering->is_position(scratch_data.components[j]))
      //     {
      //       const double diff = ((*grad_phi_x)[j] -
      //       (*grad_phi_x_moving)[j]).norm(); if (diff > max_diff)
      //       {
      //         max_diff = diff;
      //         j_max    = j;
      //       }
      //     }

      //   std::cout << "[DBG] cell0 q0: max ||grad_phi_x - grad_phi_x_moving||
      //   = "
      //             << max_diff;

      //   if (j_max != numbers::invalid_unsigned_int)
      //     std::cout << " (at j=" << j_max << ")";

      //   std::cout << std::endl;
      // }
      // // ---- END DEBUG ----
    }

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
    const auto &present_velocity_divergence =
      scratch_data.present_velocity_divergence[q];
    const auto &present_pressure_values =
      scratch_data.present_pressure_values[q];


    Tensor<1, dim> u_conv =
      present_velocity_values; // convecting velocity (Eulerian: u, ALE: u-w)
    if constexpr (with_moving_mesh)
      u_conv -= scratch_data.present_mesh_velocity_values[q]; // u-w

    const auto u_dot_grad_u = present_velocity_gradients * u_conv;

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratch_data.previous_velocity_values);
    // Pseudo Solide lame coeff necessary for ALE
    const double lame_mu = with_moving_mesh ? scratch_data.lame_mu[q] : 0.0;
    const double lame_lambda =
      with_moving_mesh ? scratch_data.lame_lambda[q] : 0.0;


    const auto  &tracer_value       = scratch_data.tracer_values[q];
    const auto  &tracer_gradient    = scratch_data.tracer_gradients[q];
    const auto  &potential_value    = scratch_data.potential_values[q];
    const auto  &potential_gradient = scratch_data.potential_gradients[q];
    const auto   u_dot_grad_phi     = u_conv * tracer_gradient;
    const double dphidt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, tracer_value, scratch_data.previous_tracer_values);
    const auto to_multiply_by_phi_u_i_phi_phi_j =
      (drhodphi * (dudt + u_dot_grad_u - body_force) + potential_gradient);

    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
    {
      const unsigned int &comp_i = scratch_data.components[i];

      const bool i_is_u   = this->ordering->is_velocity(comp_i);
      const bool i_is_p   = this->ordering->is_pressure(comp_i);
      const bool i_is_x   = this->ordering->is_position(comp_i);
      const bool i_is_phi = this->ordering->is_tracer(comp_i);
      const bool i_is_mu  = this->ordering->is_potential(comp_i);

      const auto &phi_u_i          = phi_u[i];
      const auto &grad_phi_u_i     = grad_phi_u[i];
      const auto &sym_grad_phi_u_i = sym_grad_phi_u[i];
      const auto &div_phi_u_i      = div_phi_u[i];
      const auto &phi_p_i          = phi_p[i];
      const auto &phi_phi_i        = phi_phi[i];
      const auto &grad_phi_phi_i   = grad_phi_phi[i];
      const auto &phi_mu_i         = phi_mu[i];
      const auto &grad_phi_mu_i    = grad_phi_mu[i];


      for (unsigned int j = 0; j < n_dofs_per_cell; ++j)
      {
        const unsigned int &comp_j = scratch_data.components[j];
        bool                assemble =
          this->coupling_table[comp_i][comp_j] == DoFTools::always;
        if (!assemble)
          continue;

        // const bool j_is_u   = this->ordering->is_velocity(comp_j);
        // const bool j_is_p   = this->ordering->is_pressure(comp_j);
        // const bool j_is_x   = this->ordering->is_position(comp_j);
        // const bool j_is_phi = this->ordering->is_tracer(comp_j);
        // const bool j_is_mu  = this->ordering->is_potential(comp_j);

        const auto &phi_u_j          = phi_u[j];
        const auto &grad_phi_u_j     = grad_phi_u[j];
        const auto &sym_grad_phi_u_j = sym_grad_phi_u[j];
        const auto &div_phi_u_j      = div_phi_u[j];
        const auto &phi_p_j          = phi_p[j];
        const auto &phi_phi_j        = phi_phi[j];
        const auto &grad_phi_phi_j   = grad_phi_phi[j];
        const auto &phi_mu_j         = phi_mu[j];
        const auto &grad_phi_mu_j    = grad_phi_mu[j];

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
              //  //2. * eta * scalar_product(grad_phi_u_i, sym_grad_phi_u_j);
              2. * eta * scalar_product(sym_grad_phi_u_i, sym_grad_phi_u_j);
          }
          if (comp_j == const_ordering.p_lower)
          {
            local_flow_ij += -div_phi_u_i * phi_p_j;
            // local_flow_ij += phi_p_j
            // *trace(grad_phi_u_i*(*grad_phi_x_moving)[j]);
          }
          if constexpr (with_moving_mesh)
          {
            if (const_ordering.x_lower <= comp_j &&
                comp_j < const_ordering.x_upper)
            {
              const Tensor<2, dim> G   = (*grad_phi_x_moving)[j];
              const double         trG = trace(G);

              // variation volume
              local_flow_ij += phi_u_i *
                               (rho * (dudt - body_force +
                                       present_velocity_gradients * u_conv)) *
                               trG;
              const auto &source_term_velocity =
                scratch_data.source_term_velocity[q];
              local_flow_ij += phi_u_i * source_term_velocity * trG;

              // u–x block (ALE)
              local_flow_ij +=
                phi_u_i *
                (rho * (present_velocity_gradients * (-(bdf_c0 * (*phi_x)[j])) -
                        present_velocity_gradients * G * u_conv));
              // contrib from viscosity
              local_flow_ij +=
                2. * eta *
                (scalar_product(-symmetrize(grad_phi_u_i * G),
                                present_velocity_sym_gradients) +
                 scalar_product(sym_grad_phi_u_i,
                                -symmetrize(present_velocity_gradients * G)) +
                 scalar_product(sym_grad_phi_u_i,
                                present_velocity_sym_gradients) *
                   trG);


              // contrib from pressure
              local_flow_ij +=
                trace(grad_phi_u_i * G) * present_pressure_values -
                div_phi_u_i * present_pressure_values * trG;

              // contrib from diffuse flux
              local_flow_ij +=
                phi_u_i * diffusive_flux_factor *
                (-(present_velocity_gradients * G) * potential_gradient +
                 present_velocity_gradients *
                   (-transpose(G) * potential_gradient) +
                 (present_velocity_gradients * potential_gradient) * trG);

              // contrib from tracer grad_mu
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
              const Tensor<2, dim> G   = (*grad_phi_x_moving)[j];
              const double         trG = trace(G);
              const auto          &source_term_pressure =
                scratch_data.source_term_pressure[q];
              local_flow_ij += phi_p_i * source_term_pressure * trG;

              local_flow_ij +=
                phi_p_i * (trace(present_velocity_gradients * G) -
                           present_velocity_divergence * trG);
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
            if (const_ordering.x_lower <= comp_j &&
                comp_j < const_ordering.x_upper)
            {
              const Tensor<2, dim> G   = (*grad_phi_x_moving)[j];
              const double         trG = trace(G);
              const auto          &source_term_tracer =
                scratch_data.source_term_tracer[q];
              local_flow_ij += phi_phi_i * source_term_tracer * trG;


              local_flow_ij +=
                phi_phi_i * (dphidt + u_conv * tracer_gradient) * trG;
              // phi-x bloc (ALE)
              local_flow_ij +=
                phi_phi_i * ((-bdf_c0) * (*phi_x)[j] * tracer_gradient +
                             u_conv * (-(transpose(G)) * tracer_gradient));

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
              const Tensor<2, dim> G   = (*grad_phi_x_moving)[j];
              const double         trG = trace(G);
              const auto          &source_term_potential =
                scratch_data.source_term_potential[q];
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
          const auto  &mf    = this->param.mesh_forcing;
          const double beta  = (mf.enable ? mf.beta : 0.0);
          const double alpha = (mf.enable ? mf.alpha : 0.0);

          if (const_ordering.x_lower <= comp_i &&
              comp_i < const_ordering.x_upper)
          {
            if (const_ordering.u_lower <= comp_j &&
                comp_j < const_ordering.u_upper)
            {
              if (beta != 0.0)
              {
                local_ps_ij -=
                  (*phi_x)[i] *
                  (beta * (phi_u_j * tracer_gradient) * tracer_gradient);
              }
            }
            if (const_ordering.x_lower <= comp_j &&
                comp_j < const_ordering.x_upper)
            {
              // x-x
              local_ps_ij +=
                lame_lambda * (*div_phi_x)[j] * (*div_phi_x)[i] +
                lame_mu *
                  scalar_product((*grad_phi_x)[j] + transpose((*grad_phi_x)[j]),
                                 (*grad_phi_x)[i]);
              if (alpha != 0.0)
              {
                local_ps_ij -= (*phi_x)[i] * (alpha * tracer_value *
                                              (-transpose((*grad_phi_x)[j])) *
                                              tracer_gradient);
              }
              if (beta != 0.0)
              {
                local_ps_ij -=
                  (*phi_x)[i] *
                  (beta *
                   ((-bdf_c0) * (*phi_x)[j] * tracer_gradient *
                      tracer_gradient +
                    u_conv *
                      ((-transpose((*grad_phi_x)[j])) * tracer_gradient) *
                      tracer_gradient +
                    u_dot_grad_phi *
                      ((-transpose((*grad_phi_x)[j])) * tracer_gradient)));
              }
            }
            if (comp_j == const_ordering.phi_lower)
            {
              if (alpha != 0.0)
              {
                // mesh source term
                local_ps_ij -=
                  ((*phi_x)[i] * (alpha * (phi_phi_j * tracer_gradient +
                                           tracer_value * grad_phi_phi_j)));
              }
              if (beta != 0.0)
              {
                local_ps_ij -=
                  (*phi_x)[i] *
                  (beta * ((u_conv * grad_phi_phi_j) * tracer_gradient +
                           (u_dot_grad_phi)*grad_phi_phi_j));
              }
            }
          }
        }
        local_flow_ij *= JxW_moving;
        if constexpr (with_moving_mesh)
        {
          local_ps_ij *= scratch_data.JxW_fixed[q];
        }
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

    const double rho = scratch_data.density[q];
    const double eta = scratch_data.dynamic_viscosity[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];
    const auto &present_velocity_sym_gradients =
      scratch_data.present_velocity_sym_gradients[q];

    Tensor<1, dim> u_conv =
      present_velocity_values; // convecting velocity (Eulerian: u, ALE: u-w)
    if constexpr (with_moving_mesh)
    {
      const auto &w = scratch_data.present_mesh_velocity_values[q];
      // if (w.norm() > 1e-13)
      // std::cout << "norm(w) = " << w.norm() << std::endl;
      // Assert((w.norm() < 1e-13), ExcMessage("mesh velocity non null"));
      u_conv -= scratch_data.present_mesh_velocity_values[q]; // u-w
    }


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
    const auto &u_conv_dot_tracer_gradient =
      scratch_data.u_conv_dot_tracer_gradient[q];
    const double phi_cube_minus_phi =
      tracer_value * (tracer_value * tracer_value - 1.);

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratch_data.previous_velocity_values);
    const double dphidt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, tracer_value, scratch_data.previous_tracer_values);

    // Terms of the momentum equation multiplied by phi_u_i
    const auto to_multiply_by_phi_u_i =
      rho * (dudt + u_dot_grad_u - body_force) + diffusive_flux +
      tracer_value * potential_gradient + source_term_velocity;
    // #if defined(WITH_SOURCE_TERMS)
    //   to_multiply_by_phi_u_i += source_term_velocity;
    // #endif

    // Terms of the tracer equation multiplied by phi_phi_i
    const auto to_multiply_by_phi_phi_i =
      dphidt + u_conv_dot_tracer_gradient + source_term_tracer;
    // #if defined(WITH_SOURCE_TERMS)
    //   to_multiply_by_phi_phi_i += source_term_tracer;
    // #endif

    // Terms of the potential equation multiplied by phi_mu_i
    const auto to_multiply_by_phi_mu_i =
      potential_value - sigma_tilde_over_eps * phi_cube_minus_phi +
      source_term_potential;
    // #if defined(WITH_SOURCE_TERMS)
    //   to_multiply_by_phi_mu_i +=source_term_potential;
    // #endif

    const auto &phi_p          = scratch_data.phi_p[q];
    const auto &phi_u          = scratch_data.phi_u[q];
    const auto &grad_phi_u     = scratch_data.grad_phi_u[q];
    const auto &sym_grad_phi_u = scratch_data.sym_grad_phi_u[q];
    const auto &div_phi_u      = scratch_data.div_phi_u[q];
    const auto &phi_phi        = scratch_data.shape_phi[q];
    const auto &grad_phi_phi   = scratch_data.grad_shape_phi[q];
    const auto &phi_mu         = scratch_data.shape_mu[q];
    const auto &grad_phi_mu    = scratch_data.grad_shape_mu[q];

    //
    // Pseudo-solid related data
    //
    const double lame_mu = with_moving_mesh ? scratch_data.lame_mu[q] : 0.0;
    const double lame_lambda =
      with_moving_mesh ? scratch_data.lame_lambda[q] : 0.0;

    // double lame_mu, lame_lambda;
    double         present_displacement_divergence;
    double         present_trace_strain;
    Tensor<2, dim> present_strain;


    const Tensor<2, dim>              *present_position_gradients = nullptr;
    const Tensor<1, dim>              *source_term_position       = nullptr;
    const std::vector<Tensor<1, dim>> *phi_x                      = nullptr;
    const std::vector<Tensor<2, dim>> *grad_phi_x                 = nullptr;
    const std::vector<double>         *div_phi_x                  = nullptr;

    if constexpr (with_moving_mesh)
    {
      present_position_gradients = &scratch_data.present_position_gradients[q];
      source_term_position       = &scratch_data.source_term_position[q];

      phi_x                           = &scratch_data.phi_x[q];
      grad_phi_x                      = &scratch_data.grad_phi_x[q];
      div_phi_x                       = &scratch_data.div_phi_x[q];
      present_displacement_divergence = trace((*present_position_gradients));
      present_strain = symmetrize((*present_position_gradients)) -
                       unit_symmetric_tensor<dim>();
      present_trace_strain = present_displacement_divergence - (double)dim;
    }

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const auto &phi_u_i          = phi_u[i];
      const auto &grad_phi_u_i     = grad_phi_u[i];
      const auto &sym_grad_phi_u_i = sym_grad_phi_u[i];
      const auto &div_phi_u_i      = div_phi_u[i];
      const auto &phi_p_i          = phi_p[i];
      const auto &phi_phi_i        = phi_phi[i];
      const auto &grad_phi_phi_i   = grad_phi_phi[i];
      const auto &phi_mu_i         = phi_mu[i];
      const auto &grad_phi_mu_i    = grad_phi_mu[i];

      // /**
      //  * Momentum equation
      //  */
      double local_rhs_flow_i =
        phi_u_i * to_multiply_by_phi_u_i -
        div_phi_u_i * present_pressure_values +
        2. * eta *
          scalar_product(sym_grad_phi_u_i, present_velocity_sym_gradients);


      /**
       * Continuity equation
       */
      local_rhs_flow_i +=
        phi_p_i * (-present_velocity_divergence + source_term_pressure);
      // #if defined(WITH_SOURCE_TERMS)
      // local_rhs_flow_i +=
      //   phi_p_i * source_term_pressure;
      // #endif

      /**
       * Tracer equation
       */
      local_rhs_flow_i += phi_phi_i * to_multiply_by_phi_phi_i +
                          grad_phi_phi_i * mobility * potential_gradient;

      /**
       * Potential equation
       */
      local_rhs_flow_i +=
        phi_mu_i * to_multiply_by_phi_mu_i -
        grad_phi_mu_i * sigma_tilde_times_eps * tracer_gradient;


      double local_rhs_ps_i = 0.;
      if constexpr (with_moving_mesh)
      {
        /**
         * Pseudo-Solid
         */

        local_rhs_ps_i +=
          // Linear elasticity
          lame_lambda * present_trace_strain * (*div_phi_x)[i] +
          2 * lame_mu * scalar_product(present_strain, (*grad_phi_x)[i])
          // Linear elasticity source term
          + (*phi_x)[i] * (*source_term_position);
        // --- mesh forcing near interface (pseudo-solid RHS)
        const auto  &mf    = this->param.mesh_forcing;
        const double beta  = (mf.enable ? mf.beta : 0.0);
        const double alpha = (mf.enable ? mf.alpha : 0.0);
        if (mf.enable && (mf.alpha != 0.0 || mf.beta != 0.0))
        {
          const double phi  = tracer_value;
          const auto  &gphi = tracer_gradient; // ∇φ

          Tensor<1, dim> f_mesh =
            alpha * phi * gphi + beta * (u_conv * gphi) * gphi;


          // Add to pseudo-solid RHS as a body force
          local_rhs_ps_i -= (*phi_x)[i] * f_mesh;
        }
        // #if defined(WITH_SOURCE_TERMS)
        // local_rhs_ps_i += (*phi_x)[i] * (*source_term_position);
        // #endif
      }

      local_rhs_flow_i *= JxW_moving;

      if constexpr (with_moving_mesh)
        local_rhs_ps_i *= scratch_data.JxW_fixed[q];

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

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::output_results()
{
  TimerOutput::Scope t(this->computing_timer, "Write outputs");

  if (this->param.output.write_results)
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");
    if constexpr (with_moving_mesh)
      for (unsigned int d = 0; d < dim; ++d)
        solution_names.push_back("mesh_position");
    solution_names.push_back("tracer");
    solution_names.push_back("potential");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    if constexpr (with_moving_mesh)
      for (unsigned int d = 0; d < dim; ++d)
        data_component_interpretation.push_back(
          DataComponentInterpretation::component_is_part_of_vector);
    for (unsigned int i = 0; i < 2; ++i)
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

    data_out.build_patches(*this->moving_mapping, 2);

    // Export regular time step
    const std::string pvtu_file = data_out.write_vtu_with_pvtu_record(
      this->param.output.output_dir,
      this->param.output.output_prefix,
      this->time_handler.current_time_iteration,
      this->mpi_communicator,
      2);

    this->visualization_times_and_names.emplace_back(
      this->time_handler.current_time, pvtu_file);
  }
}

// Explicit instantiation
template class CHNSSolver<2, false>;
template class CHNSSolver<3, false>;
template class CHNSSolver<2, true>;
template class CHNSSolver<3, true>;