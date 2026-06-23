#include <assembly/ale_geometry.h>
#include <assembly/moving_mesh_forcing_forms.h>
#include <assembly/pseudosolid_forms.h>
#include <assembly/stabilization_forms.h>
#include <cahn_hilliard.h>
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
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <incompressible_chns_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <mesh_and_dof_tools.h>
#include <mesh_forcing_postprocessing.h>
#include <scratch_data.h>
#include <utilities.h>

#include <deal.II/base/quadrature_lib.h>
#include <error_estimation/patches.h>
#include <error_estimation/solution_recovery.h>
#include <metric_field.h>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>

template <int dim, bool with_moving_mesh, bool with_enlarged>
CHNSSolver<dim, with_moving_mesh, with_enlarged>::CHNSSolver(
  const ParameterReader<dim> &param)
  : NavierStokesSolver<dim, with_moving_mesh>(param)
{
  this->pcout << "CHNS model: "
              << CahnHilliard::model_name(param.cahn_hilliard) << std::endl;

  if constexpr (with_moving_mesh && !with_enlarged)
    AssertThrow(
      std::abs(param.cahn_hilliard.mff_enlarged_compression_factor) < 1e-14,
      ExcMessage(
        "mff_enlarged_compression_factor is only available with the enlarged "
        "solver. In non-enlarged runs, set it to 0 and use "
        "mff_physics_compression_factor as the single compression term."));

  if constexpr (with_moving_mesh)
  {
    if (param.finite_elements.use_quads)
      if constexpr (with_enlarged)
        fe = std::make_unique<FESystem<dim>>(
          FESystem<dim>(FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim),
          FE_Q<dim>(param.finite_elements.pressure_degree),
          FESystem<dim>(FE_Q<dim>(param.finite_elements.mesh_position_degree) ^
                        dim),
          FE_Q<dim>(param.finite_elements.tracer_degree),
          FE_Q<dim>(param.finite_elements.potential_degree),
          FE_Q<dim>(param.finite_elements.potential_degree));
      else
        fe = std::make_unique<FESystem<dim>>(
          FESystem<dim>(FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim),
          FE_Q<dim>(param.finite_elements.pressure_degree),
          FESystem<dim>(FE_Q<dim>(param.finite_elements.mesh_position_degree) ^
                        dim),
          FE_Q<dim>(param.finite_elements.tracer_degree),
          FE_Q<dim>(param.finite_elements.potential_degree));
    else
      if constexpr (with_enlarged)
        fe = std::make_unique<FESystem<dim>>(
          FESystem<dim>(FE_SimplexP<dim>(
                          param.finite_elements.velocity_degree) ^
                        dim),
          FE_SimplexP<dim>(param.finite_elements.pressure_degree),
          FESystem<dim>(FE_SimplexP<dim>(
                          param.finite_elements.mesh_position_degree) ^
                        dim),
          FE_SimplexP<dim>(param.finite_elements.tracer_degree),
          FE_SimplexP<dim>(param.finite_elements.potential_degree),
          FE_SimplexP<dim>(param.finite_elements.potential_degree));
      else
        fe = std::make_unique<FESystem<dim>>(
          FESystem<dim>(FE_SimplexP<dim>(
                          param.finite_elements.velocity_degree) ^
                        dim),
          FE_SimplexP<dim>(param.finite_elements.pressure_degree),
          FESystem<dim>(FE_SimplexP<dim>(
                          param.finite_elements.mesh_position_degree) ^
                        dim),
          FE_SimplexP<dim>(param.finite_elements.tracer_degree),
          FE_SimplexP<dim>(param.finite_elements.potential_degree));

    this->ordering =
      std::make_unique<ComponentOrderingCHNS<dim, true, with_enlarged>>();
  }
  else
  {
    if (param.finite_elements.use_quads)
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim),
        FE_Q<dim>(param.finite_elements.pressure_degree),
        FE_Q<dim>(param.finite_elements.tracer_degree),
        FE_Q<dim>(param.finite_elements.potential_degree));
    else
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^
                      dim),
        FE_SimplexP<dim>(param.finite_elements.pressure_degree),
        FE_SimplexP<dim>(param.finite_elements.tracer_degree),
        FE_SimplexP<dim>(param.finite_elements.potential_degree));

    this->ordering =
      std::make_unique<ComponentOrderingCHNS<dim, false, with_enlarged>>();
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
  if constexpr (with_enlarged)
    psi_extractor = FEValuesExtractors::Scalar(this->ordering->psi_lower);

  this->velocity_mask = fe->component_mask(this->velocity_extractor);
  this->pressure_mask = fe->component_mask(this->pressure_extractor);
  if constexpr (with_moving_mesh)
    this->position_mask = fe->component_mask(this->position_extractor);

  tracer_mask    = fe->component_mask(tracer_extractor);
  potential_mask = fe->component_mask(potential_extractor);
  if constexpr (with_enlarged)
    psi_mask = fe->component_mask(psi_extractor);

  this->field_names_and_masks["velocity"]  = this->velocity_mask;
  this->field_names_and_masks["pressure"]  = this->pressure_mask;
  this->field_names_and_masks["tracer"]    = this->tracer_mask;
  this->field_names_and_masks["potential"] = this->potential_mask;
  if constexpr (with_enlarged)
    this->field_names_and_masks["psi"] = this->psi_mask;

  /**
   * Create the initial condition functions
   */
  this->param.initial_conditions.create_initial_velocity(
    this->ordering->u_lower, this->ordering->n_components);
  this->param.initial_conditions.create_initial_chns_tracer(
    this->ordering->phi_lower, this->ordering->n_components);
  if constexpr (with_enlarged)
    this->param.initial_conditions.create_initial_chns_enlarged_psi(
      this->ordering->psi_lower, this->ordering->n_components);

  // Assign the exact solution
  this->exact_solution = std::make_shared<
    CHNSSolver<dim, with_moving_mesh, with_enlarged>::MMSSolution>(
    this->time_handler.current_time, *this->ordering, param.mms);

  if (param.mms_param.enable)
  {
    // Create the MMS source term function and override source terms
    this->source_terms = std::make_shared<
      CHNSSolver<dim, with_moving_mesh, with_enlarged>::MMSSourceTerm>(
      this->time_handler.current_time, *this->ordering, param);

    // Create entry in error handler for tracer and potential
    for (auto &[norm, handler] : this->error_handlers)
    {
      handler.create_entry("phi");
      handler.create_entry("mu");
      if constexpr (with_enlarged)
        handler.create_entry("psi");
    }
  }
  else
  {
    this->source_terms = std::make_shared<
      CHNSSolver<dim, with_moving_mesh, with_enlarged>::SourceTerm>(
      this->time_handler.current_time, *this->ordering, param.source_terms);
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::MMSSourceTerm::
  vector_value(const Point<dim> &p, Vector<double> &values) const
{
  const double phi          = mms.exact_tracer->value(p);
  const double psi          = mms.exact_psi->value(p);
  const double filtered_phi = phi;
  const bool   use_sharp_material_diffuse_capillary =
    CahnHilliard::is_sharp_material_diffuse_capillary_model(
      cahn_hilliard_param);
  const double q_material =
    CahnHilliard::material_phase_marker(cahn_hilliard_param, phi);
  const double dq_dphi =
    CahnHilliard::material_phase_derivative_wrt_tracer(cahn_hilliard_param,
                                                       phi);
  const double rho0         = physical_properties.fluids[0].density;
  const double rho1         = physical_properties.fluids[1].density;
  const double rho  = CahnHilliard::material_property_mixing(cahn_hilliard_param,
                                                            filtered_phi,
                                                            rho0,
                                                            rho1);
  const double eta0 = rho0 * physical_properties.fluids[0].kinematic_viscosity;
  const double eta1 = rho1 * physical_properties.fluids[1].kinematic_viscosity;
  const double eta  = CahnHilliard::material_property_mixing(cahn_hilliard_param,
                                                            filtered_phi,
                                                            eta0,
                                                            eta1);
  const auto   mobility_function =
    CahnHilliard::get_mobility_function(cahn_hilliard_param);
  const auto mobility_derivative_function =
    CahnHilliard::get_mobility_derivative_function(cahn_hilliard_param);
  const double M = mobility_function(cahn_hilliard_param, filtered_phi);
  const double dM_dphi =
    mobility_derivative_function(cahn_hilliard_param, filtered_phi);

  const double diff_flux_factor = M * 0.5 * (rho1 - rho0);
  const double detadphi =
    CahnHilliard::material_property_derivative_wrt_tracer(cahn_hilliard_param,
                                                          filtered_phi,
                                                          eta0,
                                                          eta1);
  const double epsilon = cahn_hilliard_param.epsilon_interface;
  const double sigma_tilde =
    CahnHilliard::sigma_tilde_from_surface_tension(cahn_hilliard_param);
  const double potential_double_well_coefficient =
    CahnHilliard::potential_double_well_coefficient(cahn_hilliard_param,
                                                    sigma_tilde);
  const double potential_gradient_coefficient =
    CahnHilliard::potential_gradient_coefficient(cahn_hilliard_param,
                                                 sigma_tilde);
  const auto  &body_force            = physical_properties.body_force;

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
  const double   mu          = mms.exact_potential->value(p);
  Tensor<1, dim> grad_mu     = mms.exact_potential->gradient(p);
  Tensor<1, dim> grad_phi    = mms.exact_tracer->gradient(p);
  Tensor<1, dim> grad_q      = dq_dphi * grad_phi;
  Tensor<1, dim> J_flux      = diff_flux_factor * grad_mu;
  Tensor<1, dim> div_viscous = (eta * (lap_u + grad_div_u) +
                                2. * detadphi * grad_phi * symmetrize(grad_u));
  const Tensor<1, dim> momentum_diffusive_inertia =
    CahnHilliard::use_abels_diffusive_inertia(cahn_hilliard_param) ?
      J_flux * grad_u :
      Tensor<1, dim>();
  const Tensor<1, dim> momentum_capillary_force =
    CahnHilliard::use_abels_capillary_phi_grad_mu(cahn_hilliard_param) ?
      q_material * grad_mu :
      -CahnHilliard::ding_horriche_capillary_coefficient(
        cahn_hilliard_param) *
        mu * grad_phi;

  // Navier-Stokes momentum (velocity) source term
  Tensor<1, dim> f = -(rho * (dudt_eulerian + uDotGradu - body_force) +
                       momentum_diffusive_inertia + grad_p - div_viscous +
                       momentum_capillary_force);
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
  const double dqdt   = dq_dphi * dphidt;
  const double lap_mu = mms.exact_potential->laplacian(p);
  const double transport_time_derivative =
    use_sharp_material_diffuse_capillary ? dqdt : dphidt;
  const Tensor<1, dim> transport_gradient =
    use_sharp_material_diffuse_capillary ? grad_q : grad_phi;
  values[phi_lower] =
    -(transport_time_derivative + u * transport_gradient -
      (M * lap_mu + dM_dphi * (grad_phi * grad_mu)));

  // Potential source term
  const double lap_phi = mms.exact_tracer->laplacian(p);
  const double potential_mass_factor =
    use_sharp_material_diffuse_capillary ? dq_dphi : 1.;
  values[mu_lower] =
    -(potential_mass_factor * mu -
      potential_double_well_coefficient * phi * (phi * phi - 1.) +
      potential_gradient_coefficient * lap_phi);
  if constexpr (with_enlarged)
  {
    const double lap_psi = mms.exact_psi->laplacian(p);
    const double L =
      cahn_hilliard_param.epsilon_interface_enlarged -
      cahn_hilliard_param.epsilon_interface;
    const double correction_prefactor =
      Assembly::compute_psi_mu_correction_prefactor(
        cahn_hilliard_param, sigma_tilde, epsilon, L * L);
    const double psi_mu_correction =
      correction_prefactor * Assembly::psi_mu_correction_eta(phi) * mu;
    values[psi_lower] = -(psi - phi - psi_mu_correction - L * L * lap_psi);
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim,
                with_moving_mesh,
                with_enlarged>::create_scratch_data()
{
  scratch_data = std::make_unique<ScratchData>(*this->ordering,
                                               *fe,
                                               *this->fixed_mapping,
                                               *this->moving_mapping,
                                               *this->quadrature,
                                               *this->face_quadrature,
                                               this->time_handler,
                                               this->param,
                                               this->param.finite_elements
                                                 .stabilization);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim,
                with_moving_mesh,
                with_enlarged>::create_solver_specific_zero_constraints()
{
  for (const auto &[id, bc] : this->param.cahn_hilliard_bc)
  {
    /**
     * Apply manufactured solution for both tracer and potential
     */
    if (bc.type == BoundaryConditions::Type::dirichlet_mms)
    {
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               *this->dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(
                                                 this->ordering->n_components),
                                               this->zero_constraints,
                                               tracer_mask);
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               *this->dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(
                                                 this->ordering->n_components),
                                               this->zero_constraints,
                                               potential_mask);
      if constexpr (with_enlarged)
        VectorTools::interpolate_boundary_values(
          *this->moving_mapping,
          *this->dof_handler,
          id,
          Functions::ZeroFunction<dim>(this->ordering->n_components),
          this->zero_constraints,
          psi_mask);
    }
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  create_solver_specific_nonzero_constraints()
{
  for (const auto &[id, bc] : this->param.cahn_hilliard_bc)
  {
    /**
     * Apply manufactured solution for both tracer and potential
     */
    if (bc.type == BoundaryConditions::Type::dirichlet_mms)
    {
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               *this->dof_handler,
                                               id,
                                               *this->exact_solution,
                                               this->nonzero_constraints,
                                               tracer_mask);
      VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                               *this->dof_handler,
                                               id,
                                               *this->exact_solution,
                                               this->nonzero_constraints,
                                               potential_mask);
      if constexpr (with_enlarged)
        VectorTools::interpolate_boundary_values(*this->moving_mapping,
                                                 *this->dof_handler,
                                                 id,
                                                 *this->exact_solution,
                                                 this->nonzero_constraints,
                                                 psi_mask);
    }
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  set_solver_specific_initial_conditions()
{
  const Function<dim> *tracer_fun =
    this->param.initial_conditions.set_to_mms ?
      this->exact_solution.get() :
      this->param.initial_conditions.initial_chns_tracer.get();

  // Set tracer only
  VectorTools::interpolate(*this->moving_mapping,
                           *this->dof_handler,
                           *tracer_fun,
                           this->newton_update,
                           tracer_mask);
  if constexpr (with_enlarged)
  {
    const Function<dim> *psi_fun =
      this->param.initial_conditions.set_to_mms ?
        this->exact_solution.get() :
        (this->param.initial_conditions.use_enlarged_psi ?
           static_cast<const Function<dim> *>(
             this->param.initial_conditions.initial_chns_enlarged_psi.get()) :
           tracer_fun);
    VectorTools::interpolate(*this->moving_mapping,
                             *this->dof_handler,
                             *psi_fun,
                             this->newton_update,
                             psi_mask);
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  set_solver_specific_exact_solution()
{
  // Set tracer and potential
  VectorTools::interpolate(*this->moving_mapping,
                           *this->dof_handler,
                           *this->exact_solution,
                           this->local_evaluation_point,
                           tracer_mask);
  VectorTools::interpolate(*this->moving_mapping,
                           *this->dof_handler,
                           *this->exact_solution,
                           this->local_evaluation_point,
                           potential_mask);
  if constexpr (with_enlarged)
    VectorTools::interpolate(*this->moving_mapping,
                             *this->dof_handler,
                             *this->exact_solution,
                             this->local_evaluation_point,
                             psi_mask);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::create_sparsity_pattern()
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

      // PSPG: p couples to p, phi and mu
      if (this->ordering->is_pressure(i) &&
          (this->ordering->is_pressure(j) || this->ordering->is_tracer(j) ||
           this->ordering->is_potential(j)) &&
          this->param.finite_elements.stabilization)
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

  CHNSEnlargedOps<dim, with_moving_mesh, with_enlarged>::extend_coupling_table(
    *this->ordering, this->param.finite_elements.stabilization, coupling_table);

  DoFTools::make_sparsity_pattern(*this->dof_handler,
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

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  this->update_constraints_for_evaluation_point();

  this->system_matrix = 0;

  CopyData copy_data(*fe);

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
  WorkStream::run(this->dof_handler->begin_active(),
                  this->dof_handler->end(),
                  *this,
                  assembly_ptr,
                  &CHNSSolver::copy_local_to_global_matrix,
                  *scratch_data,
                  copy_data);

  this->system_matrix.compress(VectorOperation::add);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  assemble_local_matrix_finite_differences(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData                                          &scratch_data,
    CopyData                                             &copy_data)
{
  Verification::compute_local_matrix_finite_differences<dim>(
    cell, *this, &CHNSSolver::assemble_local_rhs, scratch_data, copy_data);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      *this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_matrix = copy_data.local_matrix();
  local_matrix       = 0;

  /**
   * Material parameters
   */
  const auto &body_force = scratch_data.body_force;
  const auto &cahn_hilliard = this->param.cahn_hilliard;
  const bool  use_abels_diffusive_inertia =
    CahnHilliard::use_abels_diffusive_inertia(cahn_hilliard);
  const bool use_abels_capillary_phi_grad_mu =
    CahnHilliard::use_abels_capillary_phi_grad_mu(cahn_hilliard);
  const double ding_horriche_capillary_coefficient =
    CahnHilliard::ding_horriche_capillary_coefficient(cahn_hilliard);
  const double potential_double_well_coefficient =
    CahnHilliard::potential_double_well_coefficient(
      cahn_hilliard, scratch_data.sigma_tilde);
  const double potential_gradient_coefficient =
    CahnHilliard::potential_gradient_coefficient(cahn_hilliard,
                                                 scratch_data.sigma_tilde);
  const bool use_sharp_material_diffuse_capillary =
    CahnHilliard::is_sharp_material_diffuse_capillary_model(cahn_hilliard);

  const double enlarged_length =
    this->param.cahn_hilliard.epsilon_interface_enlarged -
    this->param.cahn_hilliard.epsilon_interface;
  const double enlarged_length_sq = enlarged_length * enlarged_length;
  const std::vector<Tensor<1, dim>> *phi_x;
  const std::vector<Tensor<2, dim>> *grad_phi_x_moving;
  const Tensor<1, dim>              *source_term_velocity;
  double source_term_pressure, source_term_tracer, source_term_potential;
  const Tensor<1, dim> *phi_x_j;
  Tensor<1, dim>        to_multiply_by_phi_u_i_tr_G;
  double                to_multipliy_by_phi_phi_i_tr_G;

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

    const auto  &phi_u          = scratch_data.phi_u[q];
    const auto  &grad_phi_u     = scratch_data.grad_phi_u[q];
    const auto  &sym_grad_phi_u = scratch_data.sym_grad_phi_u[q];
    const auto  &div_phi_u      = scratch_data.div_phi_u[q];
    const auto  &phi_p          = scratch_data.phi_p[q];
    const auto  &phi_phi        = scratch_data.shape_phi[q];
    const auto  &grad_phi_phi   = scratch_data.grad_shape_phi[q];
    const auto  &phi_mu         = scratch_data.shape_mu[q];
    const auto  &grad_phi_mu    = scratch_data.grad_shape_mu[q];
    const double mobility       = scratch_data.mobility_values[q];
    const double dM_dphi = scratch_data.derivative_mobility_wrt_tracer[q];
    const double diffusive_flux_factor =
      scratch_data.diffusive_flux_factor_values[q];
    const double d_diffusive_flux_factor_dphi =
      dM_dphi * 0.5 * (scratch_data.density1 - scratch_data.density0);

    if constexpr (with_moving_mesh)
    {
      phi_x                 = &scratch_data.phi_x[q];
      grad_phi_x_moving     = &scratch_data.grad_phi_x_moving[q];
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
    const auto &u_conv       = scratch_data.present_convective_velocity[q];
    const auto  u_dot_grad_u = present_velocity_gradients * u_conv;

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratch_data.previous_velocity_values);

    const auto &tracer_value       = scratch_data.tracer_values[q];
    const auto &tracer_gradient    = scratch_data.tracer_gradients[q];
    const auto &material_phase_value =
      scratch_data.material_phase_values[q];
    const auto &material_phase_gradient =
      scratch_data.material_phase_gradients[q];
    const double dq_dphi =
      scratch_data.derivative_material_phase_wrt_tracer[q];
    const double d2q_dphi2 =
      scratch_data.second_derivative_material_phase_wrt_tracer[q];
    const auto &potential_value    = scratch_data.potential_values[q];
    const auto &potential_gradient = scratch_data.potential_gradients[q];

    const double dphidt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, tracer_value, scratch_data.previous_tracer_values);
    const double dqdt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, material_phase_value, scratch_data.previous_material_phase_values);
    const double transport_time_derivative =
      use_sharp_material_diffuse_capillary ? dqdt : dphidt;
    const Tensor<1, dim> &transport_gradient =
      use_sharp_material_diffuse_capillary ? material_phase_gradient :
                                             tracer_gradient;
    const double potential_mass_factor =
      use_sharp_material_diffuse_capillary ? dq_dphi : 1.;
    const double potential_mass_factor_derivative =
      use_sharp_material_diffuse_capillary ? d2q_dphi2 : 0.;

    // Precomputations of shape functions-independent terms
    const auto density_derivative_momentum =
      drhodphi * (dudt + u_dot_grad_u - body_force);

    if constexpr (with_moving_mesh)
    {
      to_multiply_by_phi_u_i_tr_G =
        rho * (dudt - body_force + present_velocity_gradients * u_conv) +
        *source_term_velocity;
      to_multipliy_by_phi_phi_i_tr_G =
        transport_time_derivative + u_conv * transport_gradient +
        source_term_tracer;
    }

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
          phi_x_j = &(*phi_x)[j];
        }

        double local_flow_ij = 0.;

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
                      present_velocity_gradients * phi_u_j));

            if (use_abels_diffusive_inertia)
              local_flow_ij +=
                phi_u_i * diffusive_flux_factor * grad_phi_u_j *
                potential_gradient;

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
              const double trG = Assembly::ALE::jacobian_trace(G);

              local_flow_ij += phi_u_i * to_multiply_by_phi_u_i_tr_G * trG;

              // ALE term
              local_flow_ij += phi_u_i * rho * present_velocity_gradients *
                               Assembly::ALE::convective_direction_variation(
                                 bdf_c0, *phi_x_j, G, u_conv);

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

              if (use_abels_diffusive_inertia)
                {
                  // Diffusive flux term
                  local_flow_ij +=
                    phi_u_i * diffusive_flux_factor *
                    (-(present_velocity_gradients * G) * potential_gradient +
                     present_velocity_gradients *
                       Assembly::ALE::gradient_variation(potential_gradient, G) +
                     (present_velocity_gradients * potential_gradient) * trG);
                }

              if (use_abels_capillary_phi_grad_mu)
                {
                  // Abels Korteweg term: material marker * grad(mu).
                  local_flow_ij +=
                    phi_u_i * (material_phase_value *
                                 Assembly::ALE::gradient_variation(
                                   potential_gradient, G) +
                               material_phase_value * potential_gradient *
                                 trG);
                }
              else
                {
                  // Ding/Horriche final CADYF model:
                  // residual capillarity = -(gamma/eps) * mu_hat * grad(phi).
                  local_flow_ij +=
                    -phi_u_i * ding_horriche_capillary_coefficient *
                    potential_value *
                    (Assembly::ALE::gradient_variation(tracer_gradient, G) +
                     tracer_gradient * trG);
                }
            }
          }
          if (comp_j == const_ordering.phi_lower)
          {
            local_flow_ij +=
              phi_u_i * phi_phi_j * density_derivative_momentum;
            local_flow_ij +=
              2. * detadphi * phi_phi_j *
              scalar_product(sym_grad_phi_u_i, present_velocity_sym_gradients);
            if (use_abels_capillary_phi_grad_mu)
              local_flow_ij +=
                phi_u_i * dq_dphi * phi_phi_j * potential_gradient;
            else
              local_flow_ij +=
                -phi_u_i * ding_horriche_capillary_coefficient *
                potential_value * grad_phi_phi_j;
            if (use_abels_diffusive_inertia)
              local_flow_ij +=
                phi_u_i * phi_phi_j * d_diffusive_flux_factor_dphi *
                (present_velocity_gradients * potential_gradient);
          }
          if (comp_j == const_ordering.mu_lower)
          {
            if (use_abels_capillary_phi_grad_mu)
              local_flow_ij +=
                phi_u_i *
                ((use_abels_diffusive_inertia ?
                    diffusive_flux_factor * present_velocity_gradients *
                      grad_phi_mu_j :
                    Tensor<1, dim>()) +
                 material_phase_value * grad_phi_mu_j);
            else
              local_flow_ij +=
                -phi_u_i * ding_horriche_capillary_coefficient * phi_mu_j *
                tracer_gradient;
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
              const double          trG = Assembly::ALE::jacobian_trace(G);

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
            local_flow_ij += phi_phi_i * phi_u_j * transport_gradient;
          }
          if constexpr (with_moving_mesh)
          {
            // Variation of tracer equation w.r.t. x
            if (const_ordering.x_lower <= comp_j &&
                comp_j < const_ordering.x_upper)
            {
              const Tensor<2, dim> &G   = (*grad_phi_x_moving)[j];
              const double          trG = Assembly::ALE::jacobian_trace(G);

              local_flow_ij += phi_phi_i * to_multipliy_by_phi_phi_i_tr_G * trG;

              // ALE (advection) term
              const Tensor<1, dim> transport_gradient_variation =
                use_sharp_material_diffuse_capillary ?
                  dq_dphi *
                    Assembly::ALE::gradient_variation(tracer_gradient, G) :
                  Assembly::ALE::gradient_variation(tracer_gradient, G);
              local_flow_ij +=
                phi_phi_i *
                (Assembly::ALE::mesh_velocity_variation(bdf_c0, *phi_x_j) *
                   transport_gradient +
                 u_conv * transport_gradient_variation);
              // Laplacian term
              local_flow_ij +=
                mobility *
                (Assembly::ALE::gradient_variation(grad_phi_phi_i, G) *
                   potential_gradient +
                 grad_phi_phi_i *
                   Assembly::ALE::gradient_variation(potential_gradient, G) +
                 (grad_phi_phi_i * potential_gradient) * trG);
            }
          }
          if (comp_j == const_ordering.phi_lower)
          {
            const Tensor<1, dim> transport_gradient_derivative_j =
              use_sharp_material_diffuse_capillary ?
                dq_dphi * grad_phi_phi_j +
                  d2q_dphi2 * phi_phi_j * tracer_gradient :
                grad_phi_phi_j;

            // Transient
            local_flow_ij += phi_phi_i * bdf_c0 *
                             potential_mass_factor * phi_phi_j;
            // Advection
            local_flow_ij +=
              phi_phi_i * u_conv * transport_gradient_derivative_j;
            local_flow_ij +=
              dM_dphi * phi_phi_j * (grad_phi_phi_i * potential_gradient);
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
            local_flow_ij +=
              potential_mass_factor * phi_mu_i * phi_mu_j;
          }
          if (comp_j == const_ordering.phi_lower)
          {
            local_flow_ij += potential_mass_factor_derivative * phi_mu_i *
                             phi_phi_j * potential_value;
            // Double well
            local_flow_ij += -potential_double_well_coefficient * phi_mu_i *
                             phi_phi_j *
                             (3. * tracer_value * tracer_value - 1.);
            // Diffusion
            local_flow_ij +=
              -potential_gradient_coefficient * grad_phi_mu_i *
              grad_phi_phi_j;
          }
          if constexpr (with_moving_mesh)
          {
            if (const_ordering.x_lower <= comp_j &&
                comp_j < const_ordering.x_upper)
            {
              // Variation of potential equation w.r.t. x
              const Tensor<2, dim> &G   = (*grad_phi_x_moving)[j];
              const double          trG = Assembly::ALE::jacobian_trace(G);
              local_flow_ij += phi_mu_i * source_term_potential * trG;
              local_flow_ij +=
                phi_mu_i *
                (potential_mass_factor * potential_value -
                 potential_double_well_coefficient * tracer_value *
                   (tracer_value * tracer_value - 1.)) *
                trG;
              local_flow_ij +=
                -potential_gradient_coefficient *
                Assembly::ALE::gradient_inner_product_jacobian_variation(
                  grad_phi_mu_i, tracer_gradient, G);
            }
          }
        }

        local_flow_ij *= JxW_moving;
        local_matrix(i, j) += local_flow_ij;
      }
    }
  }

  CHNSEnlargedOps<dim, with_moving_mesh, with_enlarged>::assemble_matrix_terms(
    *this->ordering,
    this->coupling_table,
    scratch_data,
    this->param.cahn_hilliard,
    enlarged_length_sq,
    local_matrix);

  Assembly::assemble_chns_matrix_stabilization<dim, with_moving_mesh>(
    *this->ordering,
    this->coupling_table,
    this->param.cahn_hilliard,
    scratch_data,
    bdf_c0,
    this->param.finite_elements.velocity_degree,
    this->param.finite_elements.tracer_degree,
    this->param.finite_elements.stabilization,
    local_matrix);

  if constexpr (with_moving_mesh)
  {
    const auto &pseudosolid = this->param.physical_properties.pseudosolids[0];
    Assembly::Pseudosolid::assemble_chns_matrix<dim>(*this->ordering,
                                                     this->coupling_table,
                                                     pseudosolid,
                                                     scratch_data,
                                                     local_matrix);
    Assembly::MovingMeshForcing::assemble_chns_matrix<dim, with_enlarged>(
      *this->ordering,
      this->coupling_table,
      this->param.cahn_hilliard,
      bdf_c0,
      scratch_data,
      local_matrix);
  }

  for (unsigned int f = 0; f < scratch_data.n_faces; ++f)
  {
    if (!cell->face(f)->at_boundary())
      continue;

    const auto face_id = scratch_data.face_boundary_id[f];
    const auto bc_it   = this->param.cahn_hilliard_bc.find(face_id);
    if (bc_it == this->param.cahn_hilliard_bc.end() ||
        bc_it->second.contact_angle < 0.)
      continue;

    const double contact_angle_surface_coeff =
      CahnHilliard::contact_angle_surface_coefficient(
        this->param.cahn_hilliard, scratch_data.sigma_tilde);
    const double epsilon = this->param.cahn_hilliard.epsilon_interface;
    const double theta   = bc_it->second.contact_angle;

    for (unsigned int qf = 0; qf < scratch_data.n_faces_q_points; ++qf)
    {
      const double phi_val = scratch_data.tracer_values_face[f][qf];
      const double g_phi_prime =
        CahnHilliard::contact_angle_normal_derivative_jacobian(phi_val,
                                                               epsilon,
                                                               theta);
      const double face_weight = scratch_data.face_JxW_moving[f][qf];

      for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
      {
        if (scratch_data.components[i] != const_ordering.mu_lower)
          continue;

        for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
        {
          if (scratch_data.components[j] != const_ordering.phi_lower)
            continue;

          local_matrix(i, j) +=
            contact_angle_surface_coeff * g_phi_prime *
            scratch_data.shape_phi_face[f][qf][j] *
            scratch_data.shape_mu_face[f][qf][i] * face_weight;
        }
      }
    }
  }

  cell->get_dof_indices(copy_data.dof_indices());
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  copy_local_to_global_matrix(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_matrix(),
                                                    copy_data.dof_indices(),
                                                    this->system_matrix);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  compare_analytical_matrix_with_fd()
{
  CopyData copy_data(*fe);
  Verification::compare_analytical_matrix_with_fd<dim>(
    *this,
    &CHNSSolver::assemble_local_matrix,
    &CHNSSolver::assemble_local_rhs,
    *scratch_data,
    copy_data,
    this->param.nonlinear_solver.write_problematic_elements);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  this->update_constraints_for_evaluation_point();

  this->system_rhs = 0;

  CopyData copy_data(*fe);

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(this->dof_handler->begin_active(),
                  this->dof_handler->end(),
                  *this,
                  &CHNSSolver::assemble_local_rhs,
                  &CHNSSolver::copy_local_to_global_rhs,
                  *scratch_data,
                  copy_data);

  this->system_rhs.compress(VectorOperation::add);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      *this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_rhs = copy_data.local_rhs();
  local_rhs       = 0;

  const auto  &body_force = scratch_data.body_force;
  const auto  &cahn_hilliard = this->param.cahn_hilliard;
  const bool   use_abels_diffusive_inertia =
    CahnHilliard::use_abels_diffusive_inertia(cahn_hilliard);
  const bool use_abels_capillary_phi_grad_mu =
    CahnHilliard::use_abels_capillary_phi_grad_mu(cahn_hilliard);
  const double ding_horriche_capillary_coefficient =
    CahnHilliard::ding_horriche_capillary_coefficient(cahn_hilliard);
  const double potential_double_well_coefficient =
    CahnHilliard::potential_double_well_coefficient(
      cahn_hilliard, scratch_data.sigma_tilde);
  const double potential_gradient_coefficient =
    CahnHilliard::potential_gradient_coefficient(cahn_hilliard,
                                                 scratch_data.sigma_tilde);
  const bool use_sharp_material_diffuse_capillary =
    CahnHilliard::is_sharp_material_diffuse_capillary_model(cahn_hilliard);
  const double enlarged_length =
    this->param.cahn_hilliard.epsilon_interface_enlarged -
    this->param.cahn_hilliard.epsilon_interface;
  const double enlarged_length_sq = enlarged_length * enlarged_length;
  //
  // Volume contributions
  //
  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW_moving = scratch_data.JxW_moving[q];
    const double rho        = scratch_data.density[q];
    const double eta        = scratch_data.dynamic_viscosity[q];
    const double mobility   = scratch_data.mobility_values[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];
    const auto &present_velocity_sym_gradients =
      scratch_data.present_velocity_sym_gradients[q];

    // u_conv is set once in reinit_cahn_hilliard_cell (always, not stab-only)
    const auto &u_conv       = scratch_data.present_convective_velocity[q];
    const auto  u_dot_grad_u = present_velocity_gradients * u_conv;

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
    const auto &material_phase_value =
      scratch_data.material_phase_values[q];
    const auto &potential_value    = scratch_data.potential_values[q];
    const auto &potential_gradient = scratch_data.potential_gradients[q];
    const auto &velocity_dot_tracer_gradient =
      scratch_data.velocity_dot_tracer_gradient[q];
    const auto &velocity_dot_material_phase_gradient =
      scratch_data.velocity_dot_material_phase_gradient[q];
    const double dq_dphi =
      scratch_data.derivative_material_phase_wrt_tracer[q];
    const double phi_cube_minus_phi =
      tracer_value * (tracer_value * tracer_value - 1.);
    const Tensor<1, dim> momentum_diffusive_inertia =
      use_abels_diffusive_inertia ? diffusive_flux : Tensor<1, dim>();
    const Tensor<1, dim> momentum_capillary_force =
      use_abels_capillary_phi_grad_mu ?
        material_phase_value * potential_gradient :
        -ding_horriche_capillary_coefficient * potential_value *
          tracer_gradient;

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratch_data.previous_velocity_values);
    const double dphidt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, tracer_value, scratch_data.previous_tracer_values);
    const double dqdt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, material_phase_value, scratch_data.previous_material_phase_values);
    const double transport_time_derivative =
      use_sharp_material_diffuse_capillary ? dqdt : dphidt;
    const double transport_advection =
      use_sharp_material_diffuse_capillary ?
        velocity_dot_material_phase_gradient :
        velocity_dot_tracer_gradient;
    const double potential_mass_factor =
      use_sharp_material_diffuse_capillary ? dq_dphi : 1.;

    // Precomputations of shape functions-independent terms
    const auto to_multiply_by_phi_u_i =
      rho * (dudt + u_dot_grad_u - body_force) +
      momentum_diffusive_inertia + momentum_capillary_force +
      source_term_velocity;
    const auto to_multiply_by_phi_phi_i =
      transport_time_derivative + transport_advection + source_term_tracer;
    const auto to_multiply_by_phi_mu_i =
      potential_mass_factor * potential_value -
      potential_double_well_coefficient * phi_cube_minus_phi +
      source_term_potential;

    const auto &phi_p          = scratch_data.phi_p[q];
    const auto &phi_u          = scratch_data.phi_u[q];
    const auto &sym_grad_phi_u = scratch_data.sym_grad_phi_u[q];
    const auto &div_phi_u      = scratch_data.div_phi_u[q];
    const auto &phi_phi        = scratch_data.shape_phi[q];
    const auto &grad_phi_phi   = scratch_data.grad_shape_phi[q];
    const auto &phi_mu         = scratch_data.shape_mu[q];
    const auto &grad_phi_mu    = scratch_data.grad_shape_mu[q];

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
        grad_phi_mu_i * potential_gradient_coefficient * tracer_gradient;

      local_rhs_flow_i *= JxW_moving;
      local_rhs(i) -= local_rhs_flow_i;
    }
  }

  CHNSEnlargedOps<dim, with_moving_mesh, with_enlarged>::assemble_rhs_terms(
    *this->ordering,
    scratch_data,
    this->param.cahn_hilliard,
    enlarged_length_sq,
    local_rhs);

  Assembly::assemble_chns_rhs_stabilization<dim>(
    *this->ordering,
    scratch_data,
    this->param.finite_elements.stabilization,
    local_rhs);

  if constexpr (with_moving_mesh)
  {
    const auto &pseudosolid = this->param.physical_properties.pseudosolids[0];
    Assembly::Pseudosolid::assemble_chns_rhs<dim>(*this->ordering,
                                                  pseudosolid,
                                                  scratch_data,
                                                  local_rhs);
    Assembly::MovingMeshForcing::assemble_chns_rhs<dim, with_enlarged>(
      *this->ordering, this->param.cahn_hilliard, scratch_data, local_rhs);
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
        const auto face_id = scratch_data.face_boundary_id[i_face];
        const auto ch_bc_it = this->param.cahn_hilliard_bc.find(face_id);
        if (ch_bc_it != this->param.cahn_hilliard_bc.end() &&
            ch_bc_it->second.contact_angle >= 0.)
        {
          const double contact_angle_surface_coeff =
            CahnHilliard::contact_angle_surface_coefficient(
              this->param.cahn_hilliard, scratch_data.sigma_tilde);
          const double epsilon =
            this->param.cahn_hilliard.epsilon_interface;
          const double theta = ch_bc_it->second.contact_angle;

          for (unsigned int qf = 0; qf < scratch_data.n_faces_q_points; ++qf)
          {
            const double phi_val =
              scratch_data.tracer_values_face[i_face][qf];
            const double g_phi =
              CahnHilliard::contact_angle_normal_derivative(phi_val,
                                                            epsilon,
                                                            theta);
            const double face_weight =
              scratch_data.face_JxW_moving[i_face][qf];

            for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
              if (scratch_data.components[i] == const_ordering.mu_lower)
                local_rhs(i) -= contact_angle_surface_coeff * g_phi *
                                scratch_data.shape_mu_face[i_face][qf][i] *
                                face_weight;
          }
        }

        // Open boundary condition with prescribed manufactured solution
        if (this->param.fluid_bc.at(face_id).type ==
            BoundaryConditions::Type::open_mms)
        {
          DEAL_II_NOT_IMPLEMENTED();
        }
      }
    }

  cell->get_dof_indices(copy_data.dof_indices());
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::copy_local_to_global_rhs(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_rhs(),
                                                    copy_data.dof_indices(),
                                                    this->system_rhs);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  add_solver_specific_postprocessing_data()
{
  if (!this->postproc_handler->should_output_volume_fields(this->time_handler))
    return;

  const auto tracer_limiter =
    CahnHilliard::get_limiter_function(this->param.cahn_hilliard);
  const double density0 =
    this->param.physical_properties.fluids[0].density;
  const double density1 =
    this->param.physical_properties.fluids[1].density;

  const bool output_pressure_abels =
    CahnHilliard::is_abels_model(this->param.cahn_hilliard);
  const bool output_material_debug =
    CahnHilliard::is_sharp_material_diffuse_capillary_model(
      this->param.cahn_hilliard);

  std::vector<std::string> component_names{"density"};
  if (output_pressure_abels)
    component_names.push_back("pressure_abels");
  if (output_material_debug)
  {
    component_names.push_back("q_material");
    component_names.push_back("dq_dphi");
    component_names.push_back("chemical_potential_phi");
  }
  const std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(
      component_names.size(),
      DataComponentInterpretation::component_is_scalar);

  // Sample the derived quantities at the support points of a continuous
  // element. This preserves the pointwise CHNS definitions while avoiding the
  // cell-average (DG0) staircase that was previously written to VTU.
  const unsigned int output_degree =
    std::max({1u,
              this->param.finite_elements.pressure_degree,
              this->param.finite_elements.tracer_degree,
              this->param.finite_elements.potential_degree});
  auto output_field =
    std::make_unique<PostProcessingTools::ContinuousDataField<dim>>(
      *this->triangulation,
      fe->reference_cell().is_hyper_cube(),
      output_degree,
      component_names,
      component_interpretation);

  const Quadrature<dim> output_points(output_field->get_unit_support_points());
  FEValues<dim> fe_values(*this->moving_mapping,
                          *fe,
                          output_points,
                          update_values);
  std::vector<double> tracer_values(output_points.size());
  std::vector<double> pressure_values(output_points.size());
  std::vector<double> potential_values(output_points.size());

  for (const auto &cell : this->dof_handler->active_cell_iterators())
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fe_values[tracer_extractor].get_function_values(*this->present_solution,
                                                       tracer_values);
      if (output_pressure_abels)
        fe_values[this->pressure_extractor].get_function_values(
          *this->present_solution, pressure_values);
      if (output_pressure_abels || output_material_debug)
        fe_values[potential_extractor].get_function_values(
          *this->present_solution, potential_values);

      std::vector<std::vector<double>> values(
        output_points.size(), std::vector<double>(component_names.size()));
      for (unsigned int q = 0; q < output_points.size(); ++q)
      {
        const double phi          = tracer_values[q];
        const double property_phi = tracer_limiter(phi);
        const double q_material =
          CahnHilliard::material_phase_marker(this->param.cahn_hilliard, phi);
        const double dq_dphi =
          CahnHilliard::material_phase_derivative_wrt_tracer(
            this->param.cahn_hilliard, phi);
        unsigned int component = 0;
        values[q][component++] =
          CahnHilliard::material_property_mixing(this->param.cahn_hilliard,
                                                 property_phi,
                                                 density0,
                                                 density1);
        if (output_pressure_abels)
          values[q][component++] =
            pressure_values[q] + q_material * potential_values[q];
        if (output_material_debug)
        {
          values[q][component++] = q_material;
          values[q][component++] = dq_dphi;
          values[q][component++] = dq_dphi * potential_values[q];
        }
      }
      output_field->set_cell_values(cell, values);
    }

  this->postproc_handler->add_continuous_data_field(std::move(output_field));

  if constexpr (with_moving_mesh)
  {
    MeshForcingPostProcessing::export_diagnostics<dim, with_enlarged>(
      *this->moving_mapping,
      *this->fixed_mapping,
      this->get_fe_system(),
      *this->quadrature,
      *this->dof_handler,
      this->velocity_extractor,
      this->position_extractor,
      tracer_extractor,
      psi_extractor,
      *this->present_solution,
      *this->previous_solutions,
      this->time_handler,
      this->param.cahn_hilliard,
      *this->postproc_handler);
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim,
                with_moving_mesh,
                with_enlarged>::solver_specific_post_processing()
{
  const auto &line_probe = this->param.postprocessing.line_probe;
  if (line_probe.enable && line_probe.write_results &&
      (this->time_handler.current_time_iteration %
           line_probe.output_frequency ==
         0 ||
       this->time_handler.is_finished()))
  {
    AssertThrow(line_probe.start.size() >= dim,
                ExcMessage("Line probe start point has too few coordinates."));
    AssertThrow(line_probe.end.size() >= dim,
                ExcMessage("Line probe end point has too few coordinates."));
    AssertThrow(line_probe.n_points >= 2,
                ExcMessage("Line probe needs at least two points."));

    std::vector<Point<dim>> probe_points(line_probe.n_points);
    for (unsigned int i = 0; i < line_probe.n_points; ++i)
    {
      const double s =
        static_cast<double>(i) / static_cast<double>(line_probe.n_points - 1);
      for (unsigned int d = 0; d < dim; ++d)
        probe_points[i][d] =
          (1.0 - s) * line_probe.start[d] + s * line_probe.end[d];
    }

    Utilities::MPI::RemotePointEvaluation<dim, dim> cache;
    const auto pressure_values = VectorTools::point_values<1>(
      *this->moving_mapping,
      *this->dof_handler,
      *this->present_solution,
      probe_points,
      cache,
      VectorTools::EvaluationFlags::avg,
      this->ordering->p_lower);
    AssertThrow(cache.all_points_found(),
                ExcMessage("At least one line probe point was not found."));

    const auto tracer_values = VectorTools::point_values<1>(
      cache,
      *this->dof_handler,
      *this->present_solution,
      VectorTools::EvaluationFlags::avg,
      this->ordering->phi_lower);
    const auto potential_values = VectorTools::point_values<1>(
      cache,
      *this->dof_handler,
      *this->present_solution,
      VectorTools::EvaluationFlags::avg,
      this->ordering->mu_lower);
    // Tracer gradients are needed to reconstruct the thermodynamic pressure
    // (free energy density contribution, see 'pressure_physical' below).
    const auto tracer_gradients = VectorTools::point_gradients<1>(
      cache,
      *this->dof_handler,
      *this->present_solution,
      VectorTools::EvaluationFlags::avg,
      this->ordering->phi_lower);

    std::vector<double> psi_values;
    if constexpr (with_enlarged)
      psi_values = VectorTools::point_values<1>(
        cache,
        *this->dof_handler,
        *this->present_solution,
        VectorTools::EvaluationFlags::avg,
        this->ordering->psi_lower);

    if (this->mpi_rank == 0)
    {
      const std::string filename =
        this->param.output.output_dir + line_probe.output_prefix + ".csv";

      // Start a fresh file (overwriting any pre-existing CSV from a previous
      // run) at the very first iteration, then append for the rest of the run.
      const bool is_first_iteration =
        this->time_handler.current_time_iteration == 0;
      const bool write_header =
        is_first_iteration || !std::ifstream(filename).good();
      std::ofstream out(filename,
                         is_first_iteration ? std::ios::trunc : std::ios::app);
      out << std::setprecision(line_probe.precision);

      const auto &cahn_hilliard = this->param.cahn_hilliard;
      const bool  use_abels_capillary_phi_grad_mu =
        CahnHilliard::use_abels_capillary_phi_grad_mu(cahn_hilliard);

      if (write_header)
      {
        out << "time,iteration,point_index";
        for (unsigned int d = 0; d < dim; ++d)
          out << ",x" << d;

        // 'pressure' is the solved pressure unknown, whose meaning depends
        // on the CHNS model:
        //  - Abels-type model: the capillary force is -q*grad(mu), with
        //    q=phi in the legacy model, so the solved pressure absorbs
        //    the gradient term grad(q*mu). When
        //    the chemical potential has relaxed towards a spatial constant
        //    (large mobility), the solved pressure is uniform and the
        //    Young-Laplace jump lives entirely in phi*mu.
        //  - Ding-Horriche model: the solved pressure is the modified
        //    pressure phat = p + Psi, with Psi the free energy density.
        // 'pressure_yl' is the modified/bulk pressure without the localized
        // free-energy density well. This is the pressure to use for
        // plateau-to-plateau Young-Laplace post-processing:
        //  - Abels-type model:  pressure_yl = p + q*mu
        //  - Ding-Horriche:     pressure_yl = phat
        // 'pressure_physical' is the full thermodynamic pressure reconstructed
        // pointwise from the solved unknowns, with
        //   Psi(phi, grad(phi)) = C_w F(phi) + (C_g / 2) |grad(phi)|^2,
        //   F(phi) = (phi^2 - 1)^2 / 4:
        //   pressure_physical = pressure_yl - Psi.
        // In bulk phases (phi = +-1, grad(phi) = 0) Psi vanishes, so both
        // pressures have the same plateau values. Inside the diffuse
        // interface, Psi is O(sigma/epsilon), producing a real localized
        // thermodynamic pressure well at phi ~= 0. Keep this column to inspect
        // the interfacial free-energy contribution, but do not use its
        // pointwise minimum as a Young-Laplace pressure plateau.
        // 'pressure_hat' and 'potential_hat' additionally expose the raw
        // modified unknowns (phat, muhat) solved by the Ding-Horriche model;
        // they are NaN for the Abels-type model.
        out << ",pressure"
            << ",pressure_physical"
            << ",pressure_yl"
            << ",pressure_free_energy"
            << ",pressure_hat"
            << ",tracer"
            << ",potential"
            << ",potential_hat";

        if constexpr (with_enlarged)
          out << ",psi";
        out << '\n';
      }

      const double nan = std::numeric_limits<double>::quiet_NaN();

      const double sigma_tilde =
        CahnHilliard::sigma_tilde_from_surface_tension(cahn_hilliard);
      const double double_well_coefficient =
        CahnHilliard::potential_double_well_coefficient(cahn_hilliard,
                                                        sigma_tilde);
      const double gradient_coefficient =
        CahnHilliard::potential_gradient_coefficient(cahn_hilliard,
                                                     sigma_tilde);
      // For Ding-Horriche, the momentum force is (sigma/eps) * muhat *
      // grad(phi): the free energy entering the pressure reconstruction
      // carries the same prefactor. For Abels the coefficients already
      // include sigma_tilde.
      const double free_energy_prefactor =
        use_abels_capillary_phi_grad_mu ?
          1.0 :
          CahnHilliard::ding_horriche_capillary_coefficient(cahn_hilliard);

      for (unsigned int i = 0; i < line_probe.n_points; ++i)
      {
        const double pressure_raw  = pressure_values[i];
        const double tracer        = tracer_values[i];
        const double q_material =
          CahnHilliard::material_phase_marker(cahn_hilliard, tracer);
        const double potential_raw = potential_values[i];

        const double double_well = 0.25 * (tracer * tracer - 1.0) *
                                   (tracer * tracer - 1.0);
        const double free_energy =
          free_energy_prefactor *
          (double_well_coefficient * double_well +
           0.5 * gradient_coefficient * tracer_gradients[i].norm_square());

        double pressure_solved;
        double pressure_physical;
        double pressure_yl;
        double potential_solved;
        double pressure_hat  = nan;
        double potential_hat = nan;

        if (use_abels_capillary_phi_grad_mu)
        {
          // Capillary force -q*grad(mu): the solved pressure absorbs
          // grad(q*mu) (see header comment).
          pressure_solved   = pressure_raw;
          potential_solved  = potential_raw;
          pressure_yl       = pressure_raw + q_material * potential_raw;
          pressure_physical = pressure_yl - free_energy;
        }
        else
        {
          // Solved unknowns are the Ding-Horriche modified pressure phat
          // and unscaled potential muhat.
          pressure_hat      = pressure_raw;
          potential_hat     = potential_raw;
          pressure_solved   = pressure_hat;
          potential_solved  = potential_hat;
          pressure_yl       = pressure_hat;
          pressure_physical = pressure_yl - free_energy;
        }

        out << this->time_handler.current_time << ','
            << this->time_handler.current_time_iteration << ',' << i;
        for (unsigned int d = 0; d < dim; ++d)
          out << ',' << probe_points[i][d];
        out << ',' << pressure_solved << ',' << pressure_physical << ','
            << pressure_yl << ',' << free_energy << ',' << pressure_hat << ','
            << tracer << ',' << potential_solved << ',' << potential_hat;
        if constexpr (with_enlarged)
          out << ',' << psi_values[i];
        out << '\n';
      }
    }
  }

  if (!should_output_mesh_quality())
    return;

  output_mesh_quality_field();
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
bool CHNSSolver<dim,
                with_moving_mesh,
                with_enlarged>::should_output_mesh_quality() const
{
  if (this->param.bc_data.n_metric_fields == 0)
    return false;

  if (this->param.finite_elements.use_quads)
    return false;

  const unsigned int output_frequency =
    this->param.metric_fields[0].mesh_quality_output_frequency;

  if (output_frequency == 0)
    return false;

  if (this->time_handler.current_time_iteration == 1)
    return true;

  return (this->time_handler.current_time_iteration % output_frequency) == 0;
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
std::vector<Tensor<2, dim>>
CHNSSolver<dim, with_moving_mesh, with_enlarged>::compute_vertexwise_F_inv_T()
  const
{
  std::vector<Tensor<2, dim>> vertex_F_inv_T(
    this->triangulation->n_vertices());
  std::vector<double>         vertex_weights(this->triangulation->n_vertices(),
                                     0.0);
  std::vector<bool>           owned_vertices;

  get_owned_mesh_vertices(*this->triangulation,
                          Utilities::MPI::this_mpi_process(
                            this->triangulation->get_mpi_communicator()),
                          owned_vertices);

  QGaussSimplex<dim> cell_quadrature(2);
  FEValues<dim>      fe_values(*this->fixed_mapping,
                          *fe,
                          cell_quadrature,
                          update_gradients | update_JxW_values);
  std::vector<Tensor<2, dim>> position_gradients(cell_quadrature.size());

  for (const auto &cell : this->dof_handler->active_cell_iterators())
  {
    if (cell->is_artificial())
      continue;

    fe_values.reinit(cell);
    fe_values[this->position_extractor].get_function_gradients(
      *this->present_solution, position_gradients);

    Tensor<2, dim> averaged_F_inv_T;
    double         cell_weight = 0.0;

    for (unsigned int q = 0; q < cell_quadrature.size(); ++q)
    {
      const double JxW = fe_values.JxW(q);
      averaged_F_inv_T += transpose(invert(position_gradients[q])) * JxW;
      cell_weight += JxW;
    }

    if (cell_weight > 0.0)
      averaged_F_inv_T /= cell_weight;

    for (const unsigned int v : cell->vertex_indices())
    {
      const auto vertex_index = cell->vertex_index(v);
      if (!owned_vertices[vertex_index])
        continue;

      vertex_F_inv_T[vertex_index] += averaged_F_inv_T * cell_weight;
      vertex_weights[vertex_index] += cell_weight;
    }
  }

  for (types::global_vertex_index v = 0; v < this->triangulation->n_vertices();
       ++v)
    if (owned_vertices[v] && vertex_weights[v] > 0.0)
      vertex_F_inv_T[v] /= vertex_weights[v];

  return vertex_F_inv_T;
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim,
                with_moving_mesh,
                with_enlarged>::transport_reconstructed_phi_gradient(
  ErrorEstimation::SolutionRecovery::Scalar<dim> &recovery) const
{
  if constexpr (!with_moving_mesh)
    return;

  std::vector<Tensor<1, dim>> transported_gradient =
    recovery.get_reconstructed_gradient();
  const auto                  vertex_F_inv_T = compute_vertexwise_F_inv_T();
  std::vector<bool>           owned_vertices;

  get_owned_mesh_vertices(*this->triangulation,
                          Utilities::MPI::this_mpi_process(
                            this->triangulation->get_mpi_communicator()),
                          owned_vertices);

  // Recover phi on the reference mesh and push its nodal gradient forward to
  // the studied ALE configuration through a vertex-averaged F^{-T}.
  for (types::global_vertex_index v = 0; v < this->triangulation->n_vertices();
       ++v)
    if (owned_vertices[v])
      transported_gradient[v] = vertex_F_inv_T[v] * transported_gradient[v];

  recovery.overwrite_reconstructed_gradient(transported_gradient);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim,
                with_moving_mesh,
                with_enlarged>::output_mesh_quality_field()
{
  TimerOutput::Scope t(this->computing_timer, "Output mesh quality field");

  AssertThrow(this->param.bc_data.n_metric_fields > 0,
              ExcMessage("Mesh quality output requires at least one metric "
                         "field in the parameter file."));

  AssertThrow(!this->param.finite_elements.use_quads,
              ExcMessage("Metric quality output is currently implemented "
                         "only for simplex meshes."));

  ErrorEstimation::PatchHandler<dim> patch_handler(*this->triangulation,
                                                   *this->fixed_mapping,
                                                   *this->dof_handler,
                                                   *this->present_solution,
                                                   this->param
                                                       .finite_elements
                                                       .tracer_degree +
                                                     1,
                                                   tracer_mask);

  this->computing_timer.enter_subsection("Build patches");
  patch_handler.build_patches();
  this->computing_timer.leave_subsection();

  ErrorEstimation::SolutionRecovery::Scalar<dim> recovery(1,
                                                          this->param,
                                                          patch_handler,
                                                          *this->dof_handler,
                                                          *this->present_solution,
                                                          *fe,
                                                          *this->fixed_mapping,
                                                          tracer_mask);

  this->computing_timer.enter_subsection("Reconstruct fields and derivatives");
  recovery.reconstruct_fields(*this->present_solution);
  this->computing_timer.leave_subsection();

  const Mapping<dim> &study_mapping =
    with_moving_mesh ? *this->moving_mapping : *this->fixed_mapping;

  if constexpr (with_moving_mesh)
    transport_reconstructed_phi_gradient(recovery);

  MetricField<dim> field(0, this->param, *this->triangulation);
  field.set_induced_metric_from_graph(recovery);
  field.apply_gradation();

  QGaussSimplex<dim> cell_quadrature(3);
  QGauss<1>          edge_quadrature(3);

  Vector<float> cell_quality =
    field.compute_cell_quality_field(study_mapping,
                                     cell_quadrature,
                                     edge_quadrature);

  DataOut<dim> data_out;
  data_out.attach_triangulation(*this->triangulation);
  data_out.add_data_vector(cell_quality,
                           "cell_quality",
                           DataOut<dim>::type_cell_data);
  // Always export the mesh-quality field on the mesh that is being studied.
  data_out.build_patches(study_mapping);

  const auto &metric_param = this->param.metric_fields[0];
  data_out.write_vtu_with_pvtu_record(this->param.output.output_dir,
                                      metric_param.mesh_quality_output_name,
                                      this->time_handler.current_time_iteration,
                                      this->dof_handler->get_mpi_communicator(),
                                      5);

  this->pcout << "Wrote mesh-quality field '"
              << metric_param.mesh_quality_output_name
              << "' at time step "
              << this->time_handler.current_time_iteration << std::endl;
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim,
                with_moving_mesh,
                with_enlarged>::compute_solver_specific_errors()
{
  const unsigned int n_active_cells = this->triangulation->n_active_cells();
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
  if constexpr (with_enlarged)
  {
    const ComponentSelectFunction<dim> psi_comp_select(
      this->ordering->psi_lower, this->ordering->n_components);
    this->compute_and_add_errors(*this->moving_mapping,
                                 *this->exact_solution,
                                 cellwise_errors,
                                 psi_comp_select,
                                 "psi");
  }
}

// Explicit instantiation
template class CHNSSolver<2, false, false>;
template class CHNSSolver<3, false, false>;
template class CHNSSolver<2, true, false>;
template class CHNSSolver<3, true, false>;
template class CHNSSolver<2, true, true>;
template class CHNSSolver<3, true, true>;
