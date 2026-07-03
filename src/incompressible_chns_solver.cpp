
#include <assembly/elasticity_assemblers.h>
#include <assembly/incompressible_chns_assemblers.h>
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

#include <cmath>
#include <fstream>
#include <limits>

template <int dim, bool with_moving_mesh, bool with_enlarged>
CHNSSolver<dim, with_moving_mesh, with_enlarged>::CHNSSolver(const ParameterReader<dim> &param)
  : NavierStokesSolver<dim, with_moving_mesh>(param)
{
  if constexpr (with_enlarged)
  {
    // Enlarged ALE: same layout as the moving-mesh CHNS, with the extra psi
    // tracer appended after the potential. The psi field reuses the tracer FE
    // degree.
    if (param.finite_elements.use_quads)
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim),
        FE_Q<dim>(param.finite_elements.pressure_degree),
        FESystem<dim>(FE_Q<dim>(param.finite_elements.mesh_position_degree) ^
                      dim),
        FE_Q<dim>(param.finite_elements.tracer_degree),
        FE_Q<dim>(param.finite_elements.potential_degree),
        FE_Q<dim>(param.finite_elements.tracer_degree));
    else
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^
                      dim),
        FE_SimplexP<dim>(param.finite_elements.pressure_degree),
        FESystem<dim>(
          FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^ dim),
        FE_SimplexP<dim>(param.finite_elements.tracer_degree),
        FE_SimplexP<dim>(param.finite_elements.potential_degree),
        FE_SimplexP<dim>(param.finite_elements.tracer_degree));

    this->ordering = std::make_unique<ComponentOrderingCHNS<dim, true, true>>();
  }
  else if constexpr (with_moving_mesh)
  {
    if (param.finite_elements.use_quads)
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim),
        FE_Q<dim>(param.finite_elements.pressure_degree),
        FESystem<dim>(FE_Q<dim>(param.finite_elements.mesh_position_degree) ^
                      dim),
        FE_Q<dim>(param.finite_elements.tracer_degree),
        FE_Q<dim>(param.finite_elements.potential_degree));
    else
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^
                      dim),
        FE_SimplexP<dim>(param.finite_elements.pressure_degree),
        FESystem<dim>(
          FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^ dim),
        FE_SimplexP<dim>(param.finite_elements.tracer_degree),
        FE_SimplexP<dim>(param.finite_elements.potential_degree));

    this->ordering = std::make_unique<ComponentOrderingCHNS<dim, true>>();
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

    this->ordering = std::make_unique<ComponentOrderingCHNS<dim, false>>();
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

  // Assign the exact solution
  this->exact_solution =
    std::make_shared<CHNSSolver<dim, with_moving_mesh, with_enlarged>::MMSSolution>(
      this->time_handler.current_time, *this->ordering, param.mms);

  if (param.mms_param.enable)
  {
    // Create the MMS source term function and override source terms
    this->source_terms =
      std::make_shared<CHNSSolver<dim, with_moving_mesh, with_enlarged>::MMSSourceTerm>(
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
    this->source_terms =
      std::make_shared<CHNSSolver<dim, with_moving_mesh, with_enlarged>::SourceTerm>(
        this->time_handler.current_time, *this->ordering, param.source_terms);
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::MMSSourceTerm::vector_value(
  const Point<dim> &p,
  Vector<double>   &values) const
{
  const double phi          = mms.exact_tracer->value(p);
  const double rho0         = physical_properties.fluids[0].density;
  const double rho1         = physical_properties.fluids[1].density;
  const double eta0 = rho0 * physical_properties.fluids[0].kinematic_viscosity;
  const double eta1 = rho1 * physical_properties.fluids[1].kinematic_viscosity;
  // Material marker m(phi) = q (abels_nlm) or phi (else), and m'(phi): the
  // properties are affine in m, the transported/conserved variable and the
  // capillary marker are m, and the potential mass factor is m'. Identity
  // marker reproduces the Abels/Ding-Horriche source terms.
  const double m_marker =
    CahnHilliard::get_material_phase_function(cahn_hilliard_param)(
      cahn_hilliard_param, phi);
  const double dm_marker =
    CahnHilliard::get_material_phase_derivative_function(cahn_hilliard_param)(
      cahn_hilliard_param, phi);
  const double rho  = CahnHilliard::linear_mixing(m_marker, rho0, rho1);
  const double eta  = CahnHilliard::linear_mixing(m_marker, eta0, eta1);
  // Mobility M(q) with the chain rule dM/dphi = M'(q) q'.
  const double mobility_phi =
    CahnHilliard::get_mobility_limiter_function(cahn_hilliard_param)(phi);
  const double mobility_arg =
    CahnHilliard::get_material_phase_function(cahn_hilliard_param)(
      cahn_hilliard_param, mobility_phi);
  const double mobility_arg_d =
    CahnHilliard::get_material_phase_derivative_function(cahn_hilliard_param)(
      cahn_hilliard_param, mobility_phi);
  const double M = CahnHilliard::get_mobility_function(cahn_hilliard_param)(
    cahn_hilliard_param, mobility_arg);
  const double dM_dphi =
    CahnHilliard::get_mobility_derivative_function(cahn_hilliard_param)(
      cahn_hilliard_param, mobility_arg) *
    mobility_arg_d;
  const double diff_flux_factor = M * 0.5 * (rho1 - rho0);
  // d(eta)/d(phi) = eta_q m' (chain rule through the marker).
  const double detadphi =
    CahnHilliard::linear_mixing_derivative(m_marker, eta0, eta1) * dm_marker;
  const double epsilon = cahn_hilliard_param.epsilon_interface;
  const double sigma_tilde =
    3. / (2. * sqrt(2.)) * cahn_hilliard_param.surface_tension;
  // Model-dependent potential coefficients and Ding-Horriche capillary gamma.
  const double double_well_coeff =
    CahnHilliard::potential_double_well_coefficient(cahn_hilliard_param,
                                                    sigma_tilde);
  const double gradient_coeff =
    CahnHilliard::potential_gradient_coefficient(cahn_hilliard_param,
                                                 sigma_tilde);
  const bool use_ding_horriche =
    CahnHilliard::is_ding_horriche_model(cahn_hilliard_param);
  const double capillary_coeff =
    CahnHilliard::ding_horriche_capillary_coefficient(cahn_hilliard_param);
  const auto &body_force = physical_properties.body_force;

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
  Tensor<1, dim> J_flux      = diff_flux_factor * grad_mu;
  Tensor<1, dim> div_viscous = (eta * (lap_u + grad_div_u) +
                                2. * detadphi * grad_phi * symmetrize(grad_u));

  // Capillary force and diffusive inertia depend on the model (see the
  // assembler): Abels uses phi*grad(mu) with diffusive inertia J.grad(u);
  // Ding-Horriche uses -gamma*mu*grad(phi) and drops diffusive inertia.
  Tensor<1, dim> momentum_capillary;
  Tensor<1, dim> momentum_diffusive_inertia;
  if (use_ding_horriche)
    momentum_capillary = -capillary_coeff * mu * grad_phi;
  else
  {
    momentum_capillary         = m_marker * grad_mu;
    momentum_diffusive_inertia = J_flux * grad_u;
  }

  // Navier-Stokes momentum (velocity) source term
  Tensor<1, dim> f = -(rho * (dudt_eulerian + uDotGradu - body_force) +
                       momentum_diffusive_inertia + grad_p - div_viscous +
                       momentum_capillary);
  for (unsigned int d = 0; d < dim; ++d)
    values[u_lower + d] = f[d];

  // Mass conservation (pressure) source term,
  // for - div(u) + f = 0 -> f = div(u_mms).
  values[p_lower] = mms.exact_velocity->divergence(p);

  if constexpr (with_moving_mesh)
  {
    // Pseudosolid (mesh position) source term
    Tensor<1, dim> f_PS =
      mms.exact_mesh_position->divergence_elastic_stress_tensor(
        physical_properties.pseudosolids[0], p);

    for (unsigned int d = 0; d < dim; ++d)
      values[x_lower + d] = f_PS[d];
  }

  // Transport source term (on the marker m). d(m)/dt = m' d(phi)/dt and
  // grad(m) = m' grad(phi); div(M(q) grad mu) = M lap(mu) + (dM/dphi)
  // grad(phi).grad(mu).
  const double dphidt = mms.exact_tracer->time_derivative(p);
  const double lap_mu = mms.exact_potential->laplacian(p);
  values[phi_lower] =
    -(dm_marker * (dphidt + u * grad_phi) - M * lap_mu -
      dM_dphi * (grad_phi * grad_mu));

  // Potential source term. Mass factor m'(phi) mu; the double-well and gradient
  // terms stay in phi.
  const double lap_phi = mms.exact_tracer->laplacian(p);
  values[mu_lower] = -(dm_marker * mu -
                       double_well_coeff * phi * (phi * phi - 1.) +
                       gradient_coeff * lap_phi);

  // Enlarged (psi) Helmholtz reconstruction source term. The strong form is
  // psi - phi - mu_correction - L^2 lap(psi) = source, so the manufactured
  // source is the negative of that residual evaluated at the exact solution.
  if constexpr (with_enlarged)
  {
    const double psi     = mms.exact_psi->value(p);
    const double lap_psi = mms.exact_psi->laplacian(p);
    const double L =
      cahn_hilliard_param.psi_interface_width_factor * epsilon;
    const double length_scale_sq = L * L;
    const double correction_prefactor =
      Assembly::IncompressibleCHNS::compute_psi_mu_correction_prefactor<dim>(
        cahn_hilliard_param.psi_mu_correction_factor,
        sigma_tilde,
        epsilon,
        length_scale_sq);
    const double psi_mu_correction =
      correction_prefactor *
      Assembly::IncompressibleCHNS::psi_mu_correction_eta(phi) * mu;
    values[psi_lower] =
      -(psi - phi - psi_mu_correction - length_scale_sq * lap_psi);
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::create_scratch_data()
{
  scratch_data = std::make_unique<ScratchData>(*this->ordering,
                                               *fe,
                                               *this->fixed_mapping,
                                               *this->moving_mapping,
                                               *this->quadrature,
                                               *this->face_quadrature,
                                               this->time_handler,
                                               this->param);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::setup_assemblers()
{
  assemblers.clear();

  // CHNS assemblers
  Assembly::IncompressibleCHNS::setup_assemblers<dim,
                                                 ScratchData,
                                                 CopyData,
                                                 with_moving_mesh,
                                                 with_enlarged>(
    this->param, *this->ordering, this->coupling_table, assemblers);

  // Elasticity
  if constexpr (with_moving_mesh)
    Assembly::Elasticity::setup_assemblers<dim, ScratchData, CopyData>(
      this->param, *this->ordering, assemblers);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  create_solver_specific_zero_constraints()
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
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::set_solver_specific_initial_conditions()
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
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::set_solver_specific_exact_solution()
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

      // p couples to u and x. PSPG also couples p to p, phi and mu through
      // the strong momentum residual.
      if (this->ordering->is_pressure(i) &&
          (this->ordering->is_velocity(j) || this->ordering->is_position(j) ||
           this->param.stabilization.enable_supg))
        coupling_table[i][j] = DoFTools::always;

      // x couples x,phi,u
      if constexpr (with_moving_mesh)
        if (this->ordering->is_position(i) &&
            (this->ordering->is_position(j) || this->ordering->is_tracer(j) ||
             this->ordering->is_velocity(j)))
          coupling_table[i][j] = DoFTools::always;

      // x also couples to psi: the enlarged moving-mesh forcing compresses the
      // mesh along the widened marker psi.
      if constexpr (with_enlarged)
        if (this->ordering->is_position(i) && this->ordering->is_psi(j))
          coupling_table[i][j] = DoFTools::always;

      // phi couples to u, phi, mu, x
      if (this->ordering->is_tracer(i))
        if (!this->ordering->is_pressure(j))
          coupling_table[i][j] = DoFTools::always;

      // mu couples to phi, mu, u, x
      if (this->ordering->is_potential(i))
        if (!this->ordering->is_pressure(j))
          coupling_table[i][j] = DoFTools::always;

      // psi (enlarged) Helmholtz reconstruction couples to psi, phi, mu, x
      if constexpr (with_enlarged)
        if (this->ordering->is_psi(i) &&
            (this->ordering->is_psi(j) || this->ordering->is_tracer(j) ||
             this->ordering->is_potential(j) ||
             this->ordering->is_position(j)))
          coupling_table[i][j] = DoFTools::always;
    }

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
  copy_data.cell_is_at_boundary   = cell->at_boundary();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      *this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_matrix      = copy_data.local_matrix();
  auto &local_dof_indices = copy_data.dof_indices();
  local_matrix            = 0;

  for (const auto &assembler : assemblers)
    assembler->assemble_matrix(scratch_data, copy_data);

  cell->get_dof_indices(local_dof_indices);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::copy_local_to_global_matrix(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_matrix(),
                                                    copy_data.dof_indices(),
                                                    this->system_matrix);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::compare_analytical_matrix_with_fd()
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
  copy_data.cell_is_at_boundary   = cell->at_boundary();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      *this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_rhs         = copy_data.local_rhs();
  auto &local_dof_indices = copy_data.dof_indices();
  local_rhs               = 0;

  for (const auto &assembler : assemblers)
    assembler->assemble_rhs(scratch_data, copy_data);

  cell->get_dof_indices(local_dof_indices);
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
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::compute_solver_specific_errors()
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

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  add_solver_specific_postprocessing_data()
{
  if (!this->postproc_handler->should_output_volume_fields(this->time_handler))
    return;

  const auto  &chp     = this->param.cahn_hilliard;
  const auto   marker  = CahnHilliard::get_material_phase_function(chp);
  const auto   marker_derivative =
    CahnHilliard::get_material_phase_derivative_function(chp);
  const auto tracer_limiter = CahnHilliard::get_limiter_function(chp);
  const double density0 = this->param.physical_properties.fluids[0].density;
  const double density1 = this->param.physical_properties.fluids[1].density;

  // Bulk pressure carrying the Young-Laplace jump, and the exposed potential.
  //  * Abels     : capillary phi*grad(mu) -> pressure_abels = p + phi*mu.
  //  * Ding-Horriche : capillary gamma*mu*grad(phi) (gradient of no scalar) ->
  //    the solved pressure already carries the jump; pressure_hat = p.
  //  * abels_nlm : capillary q*grad(mu) -> pressure_sharp = p + q*mu; also
  //    expose the material marker q and the Abels-equivalent potential
  //    mu_phi = m'(phi)*mu (the raw solved potential is mu_q).
  const bool use_ding_horriche = CahnHilliard::is_ding_horriche_model(chp);
  const bool use_abels_nlm     = CahnHilliard::is_abels_nlm_model(chp);
  const std::string pressure_name =
    use_ding_horriche ? "pressure_hat" :
    use_abels_nlm     ? "pressure_sharp" :
                        "pressure_abels";
  std::vector<std::string> component_names{"density", pressure_name};
  if (use_abels_nlm)
  {
    component_names.push_back("q");
    component_names.push_back("potential_phi");
  }
  const std::vector<DataComponentInterpretation::DataComponentInterpretation>
    component_interpretation(component_names.size(),
                             DataComponentInterpretation::component_is_scalar);

  // Sample at the support points of a continuous element so the pointwise CHNS
  // definitions are preserved (no DG0 cell-average staircase).
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
  FEValues<dim>         fe_values(*this->moving_mapping,
                          *fe,
                          output_points,
                          update_values);
  std::vector<double>   tracer_values(output_points.size());
  std::vector<double>   pressure_values(output_points.size());
  std::vector<double>   potential_values(output_points.size());

  for (const auto &cell : this->dof_handler->active_cell_iterators())
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fe_values[tracer_extractor].get_function_values(*this->present_solution,
                                                      tracer_values);
      fe_values[this->pressure_extractor].get_function_values(
        *this->present_solution, pressure_values);
      fe_values[potential_extractor].get_function_values(
        *this->present_solution, potential_values);

      std::vector<std::vector<double>> values(
        output_points.size(), std::vector<double>(component_names.size()));
      for (unsigned int q = 0; q < output_points.size(); ++q)
      {
        const double phi = tracer_values[q];
        // Material marker m = q (abels_nlm) or phi (else); density is affine
        // in m (identity marker -> the original phi mixing).
        const double m = marker(chp, tracer_limiter(phi));
        values[q][0] = CahnHilliard::linear_mixing(m, density0, density1);
        if (use_ding_horriche)
          values[q][1] = pressure_values[q];
        else if (use_abels_nlm)
        {
          const double q_marker = marker(chp, phi);
          values[q][1] = pressure_values[q] + q_marker * potential_values[q];
          values[q][2] = q_marker;
          values[q][3] = marker_derivative(chp, phi) * potential_values[q];
        }
        else
          values[q][1] = pressure_values[q] + phi * potential_values[q];
      }
      output_field->set_cell_values(cell, values);
    }

  this->postproc_handler->add_continuous_data_field(std::move(output_field));
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  solver_specific_post_processing()
{
  const auto &ts = this->param.postprocessing.time_scales;
  if (!ts.enable)
    return;

  const bool do_output =
    ts.write_results &&
    (this->time_handler.current_time_iteration % ts.output_frequency == 0 ||
     this->time_handler.is_finished());
  if (!do_output)
    return;

  // --- Model parameters (Abels-Garcke-Grun, constant mobility) -------------
  const auto  &chp     = this->param.cahn_hilliard;
  const double M       = chp.mobility;
  const double eps     = chp.epsilon_interface;
  const double sigma   = chp.surface_tension;
  const double sigma_t = 3. / (2. * std::sqrt(2.)) * sigma; // CH coefficient

  // Bulk chemical (Cahn-Hilliard) diffusivity, D_phi = M * f''(+/-1) * sigma~/
  // eps, with f(phi) = (phi^2-1)^2/4 so f''(+/-1) = 2.
  const double D_phi = 2. * M * sigma_t / eps;

  // Reference fluid = fluid 0 (the droplet phase, phi = -1).
  const double rho_ref = this->param.physical_properties.fluids[0].density;
  const double eta_ref =
    rho_ref * this->param.physical_properties.fluids[0].kinematic_viscosity;

  // --- Field reductions ----------------------------------------------------
  FEValues<dim> fe_values(*this->moving_mapping,
                          *fe,
                          *this->quadrature,
                          update_values | update_JxW_values);
  const unsigned int        n_q = this->quadrature->size();
  std::vector<Tensor<1, dim>> u_values(n_q);
  std::vector<double>         phi_values(n_q);
  std::vector<double>         mu_values(n_q);
  std::vector<double>         p_values(n_q);

  double max_u      = 0.;
  double vol_phase  = 0.; // droplet volume, integral of (1-phi)/2
  double max_abs_pmu = 0.;
  double max_abs_p    = 0.;

  for (const auto &cell : this->dof_handler->active_cell_iterators())
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fe_values[this->velocity_extractor].get_function_values(
        *this->present_solution, u_values);
      fe_values[tracer_extractor].get_function_values(*this->present_solution,
                                                      phi_values);
      fe_values[potential_extractor].get_function_values(
        *this->present_solution, mu_values);
      fe_values[this->pressure_extractor].get_function_values(
        *this->present_solution, p_values);

      for (unsigned int q = 0; q < n_q; ++q)
      {
        max_u = std::max(max_u, u_values[q].norm());
        vol_phase += 0.5 * (1. - phi_values[q]) * fe_values.JxW(q);
        max_abs_pmu =
          std::max(max_abs_pmu, std::abs(phi_values[q] * mu_values[q]));
        max_abs_p = std::max(max_abs_p, std::abs(p_values[q]));
      }
    }

  const MPI_Comm comm = this->mpi_communicator;
  max_u       = Utilities::MPI::max(max_u, comm);
  vol_phase   = Utilities::MPI::sum(vol_phase, comm);
  max_abs_pmu = Utilities::MPI::max(max_abs_pmu, comm);
  max_abs_p   = Utilities::MPI::max(max_abs_p, comm);

  // Equivalent droplet radius from its measured volume.
  const double R_eq = (dim == 2) ?
                        std::sqrt(vol_phase / numbers::PI) :
                        std::cbrt(3. * vol_phase / (4. * numbers::PI));

  // --- Velocity scales -----------------------------------------------------
  const double U_field = max_u;            // instantaneous, from the field
  const double U_sigma = sigma / eta_ref;  // intrinsic visco-capillary scale

  const double inf = std::numeric_limits<double>::infinity();
  auto safe_ratio  = [&](double num, double den) {
    return (den > 0.) ? num / den : inf;
  };

  // --- Characteristic times ------------------------------------------------
  const double tau_adv_field   = safe_ratio(R_eq, U_field);
  const double tau_adv_sigma   = safe_ratio(R_eq, U_sigma);
  const double tau_phi_macro   = safe_ratio(R_eq * R_eq, D_phi);
  const double tau_phi_int     = safe_ratio(eps * eps, D_phi);
  const double tau_visc        = safe_ratio(rho_ref * R_eq * R_eq, eta_ref);
  const double tau_sigma_visc  = safe_ratio(eta_ref * R_eq, sigma);
  const double tau_sigma_inert = std::sqrt(rho_ref * R_eq * R_eq * R_eq / sigma);

  // --- Dimensionless numbers ----------------------------------------------
  const double Pe_phi_field = safe_ratio(U_field * R_eq, D_phi);
  const double Pe_phi_sigma = safe_ratio(U_sigma * R_eq, D_phi);
  const double Cn           = safe_ratio(eps, R_eq);
  const double Ca_field     = eta_ref * U_field / sigma;
  const double Re_field     = safe_ratio(rho_ref * U_field * R_eq, eta_ref);
  const double S_param      = safe_ratio(std::sqrt(M * eta_ref), eps); // Yue-Feng

  // --- Assemble the row ----------------------------------------------------
  auto &tbl = time_scales_table;
  const auto add = [&](const std::string &key, const double value) {
    tbl.add_value(key, value);
    tbl.set_precision(key, ts.precision);
    tbl.set_scientific(key, true);
  };

  tbl.add_value("iteration", this->time_handler.current_time_iteration);
  add("time", this->time_handler.current_time);
  add("dt", this->time_handler.current_dt);
  add("cfl", this->time_handler.max_cfl_number);
  add("R_eq", R_eq);
  add("vol_phase", vol_phase);
  add("max_u", U_field);
  add("U_sigma", U_sigma);
  add("D_phi", D_phi);
  add("tau_adv_u", tau_adv_field);
  add("tau_adv_sigma", tau_adv_sigma);
  add("tau_phi_macro", tau_phi_macro);
  add("tau_phi_int", tau_phi_int);
  add("tau_visc", tau_visc);
  add("tau_sigma_visc", tau_sigma_visc);
  add("tau_sigma_inert", tau_sigma_inert);
  add("Pe_phi_u", Pe_phi_field);
  add("Pe_phi_sigma", Pe_phi_sigma);
  add("Cn", Cn);
  add("Ca_u", Ca_field);
  add("Re_u", Re_field);
  add("S_yuefeng", S_param);
  add("max_abs_phi_mu", max_abs_pmu);
  add("max_abs_p", max_abs_p);

  if (Utilities::MPI::this_mpi_process(comm) == 0)
  {
    std::ofstream out(this->param.output.output_dir + ts.output_prefix +
                      ".csv");
    // Whitespace-delimited table: a header row followed by one row per output
    // step. Readable with e.g. pandas.read_csv(..., sep=r"\s+").
    tbl.write_text(out, TableHandler::TextOutputFormat::table_with_headers);
  }
}

// Explicit instantiation
template class CHNSSolver<2, false>;
template class CHNSSolver<3, false>;
template class CHNSSolver<2, true>;
template class CHNSSolver<3, true>;
template class CHNSSolver<2, true, true>;
template class CHNSSolver<3, true, true>;
