
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
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <error_estimation/patches.h>
#include <error_estimation/solution_recovery.h>
#include <incompressible_chns_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <mesh_and_dof_tools.h>
#include <mesh_forcing_postprocessing.h>
#include <metric_field.h>
#include <scratch_data.h>
#include <utilities.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <limits>
#include <queue>

namespace
{
  /**
   * Extent of a (possibly quadratic) triangle in direction n.  The supplied
   * points are its three vertices followed by its three edge midpoints.  A
   * scalar quadratic is reconstructed on each edge, so an extremum located
   * between support points is included rather than missed by sampling.
   */
  template <int dim>
  double projected_triangle_extent(
    const std::vector<Point<dim>> &points,
    const Tensor<1, dim>          &direction)
  {
    AssertDimension(points.size(), 6);

    double minimum = std::numeric_limits<double>::max();
    double maximum = std::numeric_limits<double>::lowest();
    const auto update_extrema = [&minimum, &maximum](const double value) {
      minimum = std::min(minimum, value);
      maximum = std::max(maximum, value);
    };

    const ReferenceCell reference_cell = ReferenceCells::get_simplex<dim>();
    for (unsigned int line = 0; line < reference_cell.n_lines(); ++line)
    {
      const unsigned int v0 = reference_cell.line_to_cell_vertices(line, 0);
      const unsigned int v1 = reference_cell.line_to_cell_vertices(line, 1);
      const double p0 = points[v0] * direction;
      const double p1 = points[v1] * direction;
      const double pm = points[dim + 1 + line] * direction;
      update_extrema(p0);
      update_extrema(p1);
      update_extrema(pm);

      // p(t) = a*t^2 + b*t + p0, reconstructed from t=0, 1/2 and 1.
      const double a = 2. * (p0 + p1 - 2. * pm);
      const double b = p1 - p0 - a;
      if (std::abs(a) >
          64. * std::numeric_limits<double>::epsilon() *
            std::max({std::abs(p0), std::abs(p1), std::abs(pm), 1.}))
      {
        const double stationary_point = -b / (2. * a);
        if (stationary_point > 0. && stationary_point < 1.)
          update_extrema(a * stationary_point * stationary_point +
                         b * stationary_point + p0);
      }
    }

    return maximum - minimum;
  }
} // namespace

template <int dim, bool with_moving_mesh, bool with_enlarged>
CHNSSolver<dim, with_moving_mesh, with_enlarged>::CHNSSolver(
  const ParameterReader<dim> &param)
  : NavierStokesSolver<dim, with_moving_mesh>(param)
{
  using Strategy =
    Parameters::TimeIntegration::Adaptation::AdaptationStrategy;
  if (param.time_integration.adaptation.enable &&
      param.time_integration.adaptation.strategy == Strategy::AdaptiveMobility)
    AssertThrow(
      CahnHilliard::is_adaptive_mobility_model(param.cahn_hilliard),
      ExcMessage("The adaptive-mobility time-adaptation strategy requires "
                 "adaptative_mobility, adaptative_mobility_2, or "
                 "adaptative_mobility_3."));

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
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  compute_solver_specific_timestep_adaptation_criterion()
{
  using Strategy =
    Parameters::TimeIntegration::Adaptation::AdaptationStrategy;
  if (this->param.time_integration.adaptation.strategy !=
      Strategy::AdaptiveMobility)
    return;

  TimerOutput::Scope timer(this->computing_timer,
                           "Compute adaptive mobility number");

  const auto &chp = this->param.cahn_hilliard;
  const auto scaling = CahnHilliard::get_adaptive_mobility_scaling(chp);
  const auto mobility_evaluation =
    CahnHilliard::get_mobility_evaluation_function(chp);
  const auto mobility_limiter =
    CahnHilliard::get_mobility_limiter_function(chp);
  const auto material_phase = CahnHilliard::get_material_phase_function(chp);
  const auto material_phase_derivative =
    CahnHilliard::get_material_phase_derivative_function(chp);
  const auto material_phase_second_derivative =
    CahnHilliard::get_material_phase_second_derivative_function(chp);

  FEValues<dim> fe_values(*this->moving_mapping,
                          *fe,
                          *this->quadrature,
                          update_values | update_gradients);
  std::vector<double> tracer_values(this->quadrature->size());
  std::vector<Tensor<1, dim>> velocity_values(this->quadrature->size());
  std::vector<Tensor<1, dim>> tracer_gradients(this->quadrature->size());

  double local_max_mobility = 0.;
  for (const auto &cell : this->dof_handler->active_cell_iterators())
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fe_values[this->velocity_extractor].get_function_values(
        *this->present_solution, velocity_values);
      fe_values[tracer_extractor].get_function_values(*this->present_solution,
                                                      tracer_values);
      fe_values[tracer_extractor].get_function_gradients(
        *this->present_solution, tracer_gradients);

      for (unsigned int q = 0; q < this->quadrature->size(); ++q)
      {
        const double phi = mobility_limiter(tracer_values[q]);
        const auto evaluation = mobility_evaluation(
          chp,
          material_phase(chp, phi),
          material_phase_derivative(chp, phi),
          material_phase_second_derivative(chp, phi),
          velocity_values[q],
          tracer_gradients[q],
          scaling.coefficient,
          scaling.delta);
        AssertThrow(std::isfinite(evaluation.value) && evaluation.value >= 0.,
                    ExcMessage("Adaptive mobility must be finite and "
                               "non-negative at every quadrature point."));
        local_max_mobility =
          std::max(local_max_mobility, evaluation.value);
      }
    }

  const double max_mobility = Utilities::MPI::max(
    local_max_mobility, this->present_solution->get_mpi_communicator());
  const double mobility_number =
    CahnHilliard::compute_adaptive_mobility_number(
      this->time_handler.get_current_timestep(), chp, max_mobility);
  this->time_handler.set_max_adaptive_mobility_number(mobility_number);
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
  // d(eta)/d(phi) = eta_q m' (chain rule through the marker).
  const double detadphi =
    CahnHilliard::linear_mixing_derivative(m_marker, eta0, eta1) * dm_marker;
  const double epsilon = cahn_hilliard_param.epsilon_interface;
  const double sigma_tilde =
    3. / (2. * sqrt(2.)) * cahn_hilliard_param.surface_tension;
  double adaptive_mobility_coefficient = 0.;
  double adaptive_mobility_delta       = 0.;
  if (cahn_hilliard_param.mobility_model ==
      Parameters::CahnHilliard<dim>::MobilityModel::adaptive)
  {
    adaptive_mobility_coefficient =
      cahn_hilliard_param.adaptive_mobility_n * sqrt(2.) * epsilon * epsilon *
      epsilon / sigma_tilde;
    adaptive_mobility_delta = cahn_hilliard_param.adaptive_mobility_delta;
  }
  else if (cahn_hilliard_param.mobility_model ==
           Parameters::CahnHilliard<dim>::MobilityModel::adaptive_mobility_2)
  {
    adaptive_mobility_coefficient =
      cahn_hilliard_param.adaptive_mobility_2_n * 2. * epsilon * epsilon *
      epsilon * epsilon / sigma_tilde;
    adaptive_mobility_delta = cahn_hilliard_param.adaptive_mobility_2_delta;
  }
  else if (cahn_hilliard_param.mobility_model ==
           Parameters::CahnHilliard<dim>::MobilityModel::adaptive_mobility_3)
  {
    adaptive_mobility_coefficient =
      cahn_hilliard_param.adaptive_mobility_3_n * epsilon * epsilon /
      sigma_tilde;
    adaptive_mobility_delta = cahn_hilliard_param.adaptive_mobility_3_delta;
  }
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
  const auto mobility_evaluation =
    CahnHilliard::get_mobility_evaluation_function(cahn_hilliard_param)(
      cahn_hilliard_param,
      mobility_arg,
      mobility_arg_d,
      0.,
      u,
      grad_phi,
      adaptive_mobility_coefficient,
      adaptive_mobility_delta);
  const double M = mobility_evaluation.value;
  const Tensor<1, dim> grad_mobility =
    mobility_evaluation.derivative_wrt_tracer * grad_phi +
    mobility_evaluation.adaptive_sensitivity *
      (grad_u * grad_phi + mms.exact_tracer->hessian(p) * u);
  const double diff_flux_factor = M * 0.5 * (rho1 - rho0);
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
  // grad(phi).grad(mu). For adaptive_mobility, grad(M) is evaluated directly.
  const double dphidt = mms.exact_tracer->time_derivative(p);
  const double lap_mu = mms.exact_potential->laplacian(p);
  values[phi_lower] =
    -(dm_marker * (dphidt + u * grad_phi) - M * lap_mu -
      grad_mobility * grad_mu);

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
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  update_simulation_parameters(const unsigned int fixed_point_iteration)
{
  // Don't update anything at the first iteration.
  if (fixed_point_iteration == 0)
    return;

  const auto &fpu = this->param.mesh.adaptation.metric.fixed_point_updates;

  // Update the interface thickness
  {
    const auto &eps_update_data = fpu.chns_interface_thickness;
    if (eps_update_data.enable &&
        fixed_point_iteration % eps_update_data.update_frequency == 0)
    {
      auto &ch = this->param.cahn_hilliard;

      if (fpu.verbosity == Parameters::Verbosity::verbose)
        this->pcout << "-- Updating interface thickness from "
                    << ch.epsilon_interface << " to "
                    << ch.epsilon_interface * eps_update_data.factor
                    << std::endl;

      // Update Cahn-Hilliard parameters
      ch.epsilon_interface *= eps_update_data.factor;

      // FIXME: Update mobility accordingly?

      // Update initial condition
      {
        std::map<std::string, double> new_constants;
        new_constants[eps_update_data.constant_name] = ch.epsilon_interface;
        this->param.initial_conditions.initial_chns_tracer->update_constants(
          new_constants);
      }
    }
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
  set_solver_specific_time()
{
  for (auto &[id, bc] : this->param.cahn_hilliard_bc)
  {
    (void)id;
    bc.set_time(this->time_handler.current_time);
  }
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

    if (bc.type == BoundaryConditions::Type::input_function)
      VectorTools::interpolate_boundary_values(
        *this->moving_mapping,
        *this->dof_handler,
        id,
        Functions::ZeroFunction<dim>(this->ordering->n_components),
        this->zero_constraints,
        tracer_mask);
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

    if (bc.type == BoundaryConditions::Type::input_function)
      VectorTools::interpolate_boundary_values(
        *this->moving_mapping,
        *this->dof_handler,
        id,
        ScalarFunctionFromComponents<dim>(this->ordering->phi_lower,
                                          this->ordering->n_components,
                                          *bc.tracer),
        this->nonzero_constraints,
        tracer_mask);
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
  const auto marker_second_derivative =
    CahnHilliard::get_material_phase_second_derivative_function(chp);
  const auto tracer_limiter = CahnHilliard::get_limiter_function(chp);
  const auto mobility_tracer_limiter =
    CahnHilliard::get_mobility_limiter_function(chp);
  const auto mobility_evaluation_function =
    CahnHilliard::get_mobility_evaluation_function(chp);
  const double density0 = this->param.physical_properties.fluids[0].density;
  const double density1 = this->param.physical_properties.fluids[1].density;
  double adaptive_mobility_coefficient = 0.;
  double adaptive_mobility_delta       = 0.;
  if (chp.mobility_model == Parameters::CahnHilliard<dim>::MobilityModel::adaptive)
  {
    const double epsilon = chp.epsilon_interface;
    const double sigma_tilde = 3. / (2. * sqrt(2.)) * chp.surface_tension;
    adaptive_mobility_coefficient =
      chp.adaptive_mobility_n * sqrt(2.) * epsilon * epsilon * epsilon /
      sigma_tilde;
    adaptive_mobility_delta = chp.adaptive_mobility_delta;
  }
  else if (chp.mobility_model ==
           Parameters::CahnHilliard<dim>::MobilityModel::adaptive_mobility_2)
  {
    const double epsilon = chp.epsilon_interface;
    const double sigma_tilde = 3. / (2. * sqrt(2.)) * chp.surface_tension;
    adaptive_mobility_coefficient =
      chp.adaptive_mobility_2_n * 2. * epsilon * epsilon * epsilon * epsilon /
      sigma_tilde;
    adaptive_mobility_delta = chp.adaptive_mobility_2_delta;
  }
  else if (chp.mobility_model ==
           Parameters::CahnHilliard<dim>::MobilityModel::adaptive_mobility_3)
  {
    const double epsilon = chp.epsilon_interface;
    const double sigma_tilde = 3. / (2. * sqrt(2.)) * chp.surface_tension;
    adaptive_mobility_coefficient =
      chp.adaptive_mobility_3_n * epsilon * epsilon / sigma_tilde;
    adaptive_mobility_delta = chp.adaptive_mobility_3_delta;
  }

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
  component_names.push_back("mobility");
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
                          update_values | update_gradients);
  std::vector<double>   tracer_values(output_points.size());
  std::vector<double>   pressure_values(output_points.size());
  std::vector<double>   potential_values(output_points.size());
  std::vector<Tensor<1, dim>> velocity_values(output_points.size());
  std::vector<Tensor<1, dim>> tracer_gradients(output_points.size());

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
      fe_values[this->velocity_extractor].get_function_values(
        *this->present_solution, velocity_values);
      fe_values[tracer_extractor].get_function_gradients(*this->present_solution,
                                                          tracer_gradients);

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

        const double mobility_phi = mobility_tracer_limiter(phi);
        const auto mobility_evaluation = mobility_evaluation_function(
          chp,
          marker(chp, mobility_phi),
          marker_derivative(chp, mobility_phi),
          marker_second_derivative(chp, mobility_phi),
          velocity_values[q],
          tracer_gradients[q],
          adaptive_mobility_coefficient,
          adaptive_mobility_delta);
        values[q].back() = mobility_evaluation.value;
      }
      output_field->set_cell_values(cell, values);
    }

  this->postproc_handler->add_continuous_data_field(std::move(output_field));

  if constexpr (with_moving_mesh)
  {
    const unsigned int mesh_forcing_output_degree =
      std::max({1u,
                this->param.finite_elements.velocity_degree,
                this->param.finite_elements.mesh_position_degree,
                this->param.finite_elements.tracer_degree,
                this->param.finite_elements.potential_degree});

    MeshForcingPostProcessing::add_continuous_diagnostics<dim, with_enlarged>(
      *this->moving_mapping,
      *this->fixed_mapping,
      *fe,
      *this->dof_handler,
      this->velocity_extractor,
      this->position_extractor,
      tracer_extractor,
      psi_extractor,
      *this->present_solution,
      *this->previous_solutions,
      this->time_handler,
      this->param.cahn_hilliard,
      mesh_forcing_output_degree,
      *this->postproc_handler);
  }

  if (should_output_mesh_quality())
    add_mesh_quality_postprocessing_data();
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
bool CHNSSolver<dim,
                with_moving_mesh,
                with_enlarged>::should_output_mesh_quality() const
{
  if (this->param.bc_data.n_metric_fields == 0 ||
      this->param.finite_elements.use_quads)
    return false;

  const unsigned int output_frequency =
    this->param.metric_fields[0].mesh_quality_output_frequency;
  if (output_frequency == 0)
    return false;

  return this->time_handler.current_time_iteration == 1 ||
         (this->time_handler.current_time_iteration % output_frequency) == 0;
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
std::vector<Tensor<2, dim>>
CHNSSolver<dim, with_moving_mesh, with_enlarged>::compute_vertexwise_F_inv_T()
  const
{
  std::vector<Tensor<2, dim>> vertex_F_inv_T(
    this->triangulation->n_vertices());
  std::vector<double> vertex_weights(this->triangulation->n_vertices(), 0.);
  std::vector<bool>   owned_vertices;

  get_owned_mesh_vertices(*this->triangulation,
                          Utilities::MPI::this_mpi_process(
                            this->triangulation->get_mpi_communicator()),
                          owned_vertices);

  const QGaussSimplex<dim> cell_quadrature(2);
  FEValues<dim>            fe_values(*this->fixed_mapping,
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
    double         cell_weight = 0.;
    for (unsigned int q = 0; q < cell_quadrature.size(); ++q)
    {
      const double JxW = fe_values.JxW(q);
      averaged_F_inv_T += transpose(invert(position_gradients[q])) * JxW;
      cell_weight += JxW;
    }
    if (cell_weight > 0.)
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

  for (types::global_vertex_index v = 0;
       v < this->triangulation->n_vertices();
       ++v)
    if (owned_vertices[v] && vertex_weights[v] > 0.)
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
  const auto        vertex_F_inv_T = compute_vertexwise_F_inv_T();
  std::vector<bool> owned_vertices;
  get_owned_mesh_vertices(*this->triangulation,
                          Utilities::MPI::this_mpi_process(
                            this->triangulation->get_mpi_communicator()),
                          owned_vertices);

  // The PPR gradient is reconstructed on the fixed mesh, then pushed forward
  // to the studied ALE configuration with the vertex-averaged F^{-T}.
  for (types::global_vertex_index v = 0;
       v < this->triangulation->n_vertices();
       ++v)
    if (owned_vertices[v])
      transported_gradient[v] = vertex_F_inv_T[v] * transported_gradient[v];

  recovery.overwrite_reconstructed_gradient(transported_gradient);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  add_mesh_quality_postprocessing_data()
{
  TimerOutput::Scope timer_scope(this->computing_timer,
                                 "Compute mesh quality audit");

  AssertThrow(this->param.bc_data.n_metric_fields > 0,
              ExcMessage("Mesh quality output requires at least one metric "
                         "field in the parameter file."));
  AssertThrow(!this->param.finite_elements.use_quads,
              ExcMessage("Metric quality output is currently implemented "
                         "only for simplex meshes."));

  const auto &metric_param = this->param.metric_fields[0];
  AssertThrow(metric_param.variable == SolverInfo::VariableType::phase_tracer &&
                metric_param.component == 0,
              ExcMessage("The CHNS mesh-quality audit currently requires "
                         "'variable = phase_tracer' and 'component = 0'."));

  const Mapping<dim> &study_mapping =
    with_moving_mesh ? *this->moving_mapping : *this->fixed_mapping;
  using MeshQualityModel =
    typename Parameters::MetricField<dim>::MeshQualityModel;

  if (metric_param.mesh_quality_model ==
      MeshQualityModel::interface_resolution)
    add_interface_resolution_data(study_mapping);
  else
    add_graph_metric_quality_data(study_mapping);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  add_interface_resolution_data(const Mapping<dim> &study_mapping)
{
  AssertThrow(dim == 2,
              ExcMessage("The interface-resolution mesh audit is currently "
                         "implemented only in 2D."));

  if constexpr (dim == 2)
  {
    // Keep the two poster criteria distinct:
    //
    // 1. Local resolution: the reference computation uses epsilon=0.64*h,
    //    hence a cell is at least as fine when h_n/epsilon <= 1/0.64, or
    //    equivalently epsilon/(0.64*h_n) >= 1.
    //
    // 2. Spatial coverage: for
    //       phi(xi)=tanh(xi/(sqrt(2)*epsilon)),
    //    the target band |xi| <= 2*sqrt(2)*epsilon has total width
    //    4*sqrt(2)*epsilon. The local criterion above must hold throughout
    //    this band. The width is not divided by h_n: doing so would conflate
    //    local cell resolution with resolved-band coverage.
    const double epsilon = this->param.cahn_hilliard.epsilon_interface;
    AssertThrow(epsilon > 0., ExcInternalError());
    const double diffuse_phase_limit = std::tanh(2.);

    const ReferenceCell reference_cell = ReferenceCells::get_simplex<dim>();
    std::vector<Point<dim>> sampling_points(dim + 1);
    for (unsigned int v = 0; v < dim + 1; ++v)
      sampling_points[v] = reference_cell.template vertex<dim>(v);
    for (unsigned int line = 0; line < reference_cell.n_lines(); ++line)
    {
      const unsigned int v0 = reference_cell.line_to_cell_vertices(line, 0);
      const unsigned int v1 = reference_cell.line_to_cell_vertices(line, 1);
      sampling_points.push_back(0.5 *
                                (sampling_points[v0] + sampling_points[v1]));
    }
    Point<dim> barycenter;
    for (unsigned int v = 0; v < dim + 1; ++v)
      barycenter += sampling_points[v];
    barycenter /= static_cast<double>(dim + 1);
    sampling_points.push_back(barycenter);

    const Quadrature<dim> sampling_quadrature(sampling_points);
    FEValues<dim> current_fe_values(study_mapping,
                                    *fe,
                                    sampling_quadrature,
                                    update_quadrature_points | update_values |
                                      update_gradients);
    const QGaussSimplex<dim> measure_quadrature(3);
    FEValues<dim> current_measure_fe_values(study_mapping,
                                            *fe,
                                            measure_quadrature,
                                            update_JxW_values);
    FEValues<dim> reference_measure_fe_values(*this->fixed_mapping,
                                              *fe,
                                              measure_quadrature,
                                              update_JxW_values);

    const unsigned int n_samples = sampling_points.size();
    std::vector<double> tracer_values(n_samples);
    std::vector<Tensor<1, dim>> tracer_gradients(n_samples);
    Vector<float> compression_gain(this->triangulation->n_active_cells());
    Vector<float> local_resolution_ratio(
      this->triangulation->n_active_cells());
    Vector<float> resolved_band(this->triangulation->n_active_cells());

    using GlobalCellIndex = types::global_cell_index;
    // Each record stores {global cell id, resolved flag, interface-seed flag}.
    // Face-adjacency edges are gathered separately so that the connected band
    // can cross MPI subdomain boundaries.
    std::vector<std::array<GlobalCellIndex, 3>> local_cell_records;
    std::vector<std::array<GlobalCellIndex, 2>> local_adjacency_edges;
    local_cell_records.reserve(this->triangulation->n_locally_owned_active_cells());

    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
      {
        const GlobalCellIndex global_cell_index =
          cell->global_active_cell_index();
        std::array<GlobalCellIndex, 3> cell_record = {
          {global_cell_index, 0, 0}};

        for (unsigned int face = 0; face < cell->n_faces(); ++face)
          if (!cell->at_boundary(face))
          {
            // The benchmark meshes are conforming. This also handles the
            // fine-to-coarse side of a hanging face; a refined neighbour on
            // the other side is deliberately rejected below rather than
            // constructing an incomplete connectivity graph.
            const auto neighbour = cell->neighbor(face);
            AssertThrow(
              neighbour->is_active(),
              ExcMessage("The resolved-interface-band output currently "
                         "requires a conforming mesh without hanging faces."));
            if (!neighbour->is_artificial())
              local_adjacency_edges.push_back(
                {{global_cell_index,
                  neighbour->global_active_cell_index()}});
          }

        current_fe_values.reinit(cell);
        current_measure_fe_values.reinit(cell);
        reference_measure_fe_values.reinit(cell);

        double current_measure   = 0.;
        double reference_measure = 0.;
        for (unsigned int q = 0; q < measure_quadrature.size(); ++q)
        {
          current_measure += current_measure_fe_values.JxW(q);
          reference_measure += reference_measure_fe_values.JxW(q);
        }
        AssertThrow(current_measure > 0. && reference_measure > 0.,
                    ExcMessage("The equivalent-size compression gain requires "
                               "strictly positive cell measures."));
        const auto cell_index = cell->active_cell_index();
        compression_gain[cell_index] = static_cast<float>(
          std::pow(reference_measure / current_measure, 1. / dim));

        current_fe_values[tracer_extractor].get_function_values(
          *this->present_solution, tracer_values);
        current_fe_values[tracer_extractor].get_function_gradients(
          *this->present_solution, tracer_gradients);

        unsigned int interface_sample = 0;
        double       minimum_tracer   = tracer_values[0];
        double       maximum_tracer   = tracer_values[0];
        for (unsigned int q = 1; q < n_samples; ++q)
        {
          if (std::abs(tracer_values[q]) <
              std::abs(tracer_values[interface_sample]))
            interface_sample = q;
          minimum_tracer = std::min(minimum_tracer, tracer_values[q]);
          maximum_tracer = std::max(maximum_tracer, tracer_values[q]);
        }

        const double gradient_norm = tracer_gradients[interface_sample].norm();
        if (gradient_norm <= std::numeric_limits<double>::epsilon())
        {
          local_cell_records.push_back(cell_record);
          continue;
        }
        const Tensor<1, dim> normal =
          tracer_gradients[interface_sample] / gradient_norm;

        const auto &current_quadrature_points =
          current_fe_values.get_quadrature_points();
        const std::vector<Point<dim>> current_edge_points(
          current_quadrature_points.begin(),
          current_quadrature_points.begin() + 2 * (dim + 1));

        const double current_normal_size =
          projected_triangle_extent(current_edge_points, normal);
        if (current_normal_size <=
            std::numeric_limits<double>::epsilon())
        {
          local_cell_records.push_back(cell_record);
          continue;
        }

        const double resolution_ratio =
          epsilon / (0.64 * current_normal_size);

        // The local normal-resolution ratio is restricted to the physical
        // target layer. The global compression gain above is not restricted,
        // and the binary field below grows through every face-connected cell
        // satisfying the local normal-size test.
        if (std::abs(tracer_values[interface_sample]) <= diffuse_phase_limit)
          local_resolution_ratio[cell_index] =
            static_cast<float>(resolution_ratio);

        const bool is_locally_resolved = resolution_ratio >= 1.;
        const bool intersects_interface =
          minimum_tracer <= 0. && maximum_tracer >= 0.;
        cell_record[1] = is_locally_resolved ? 1 : 0;
        cell_record[2] =
          (is_locally_resolved && intersects_interface) ? 1 : 0;
        local_cell_records.push_back(cell_record);
      }

    const MPI_Comm communicator =
      this->triangulation->get_mpi_communicator();
    const auto gathered_cell_records =
      Utilities::MPI::all_gather(communicator, local_cell_records);
    const auto gathered_adjacency_edges =
      Utilities::MPI::all_gather(communicator, local_adjacency_edges);

    const std::size_t n_global_cells =
      this->triangulation->n_global_active_cells();
    std::vector<bool> is_resolved(n_global_cells, false);
    std::vector<bool> is_in_resolved_band(n_global_cells, false);
    std::vector<std::vector<GlobalCellIndex>> adjacency(n_global_cells);
    std::queue<GlobalCellIndex> flood_front;

    for (const auto &rank_records : gathered_cell_records)
      for (const auto &record : rank_records)
      {
        const auto id = static_cast<std::size_t>(record[0]);
        AssertIndexRange(id, n_global_cells);
        is_resolved[id] = record[1] != 0;
        if (record[2] != 0)
        {
          is_in_resolved_band[id] = true;
          flood_front.push(record[0]);
        }
      }

    for (const auto &rank_edges : gathered_adjacency_edges)
      for (const auto &edge : rank_edges)
      {
        const auto first  = static_cast<std::size_t>(edge[0]);
        const auto second = static_cast<std::size_t>(edge[1]);
        AssertIndexRange(first, n_global_cells);
        AssertIndexRange(second, n_global_cells);
        adjacency[first].push_back(edge[1]);
        adjacency[second].push_back(edge[0]);
      }

    while (!flood_front.empty())
    {
      const GlobalCellIndex current = flood_front.front();
      flood_front.pop();
      for (const GlobalCellIndex neighbour : adjacency[current])
        if (is_resolved[neighbour] && !is_in_resolved_band[neighbour])
        {
          is_in_resolved_band[neighbour] = true;
          flood_front.push(neighbour);
        }
    }

    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned() &&
          is_in_resolved_band[cell->global_active_cell_index()])
        resolved_band[cell->active_cell_index()] = 1.f;

    const std::string &base =
      this->param.metric_fields[0].mesh_quality_output_name;
    this->postproc_handler->add_cell_data_vector(compression_gain,
                                                 "mesh_compression_gain");
    this->postproc_handler->add_cell_data_vector(
      local_resolution_ratio, base + "_local_resolution_ratio");
    this->postproc_handler->add_cell_data_vector(resolved_band,
                                                 base + "_resolved_band");
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  add_graph_metric_quality_data(const Mapping<dim> &study_mapping)
{
  // This is the historical graph-induced metric M = I + grad(phi) x grad(phi).
  // Its recovery remains on the fixed mesh and the gradient is pushed forward
  // before the quality is evaluated in the current ALE configuration.
  ErrorEstimation::PatchHandler<dim> patch_handler(
    *this->triangulation,
    *this->fixed_mapping,
    *this->dof_handler,
    *this->present_solution,
    this->param.finite_elements.tracer_degree + 1,
    tracer_mask);
  patch_handler.build_patches();

  ErrorEstimation::SolutionRecovery::Scalar<dim> recovery(
    1,
    this->param,
    patch_handler,
    *this->dof_handler,
    *this->present_solution,
    *fe,
    *this->fixed_mapping,
    tracer_mask);
  recovery.reconstruct_fields(*this->present_solution);

  if constexpr (with_moving_mesh)
    transport_reconstructed_phi_gradient(recovery);

  MetricField<dim> metric_field(0, this->param, *this->triangulation);
  metric_field.set_induced_metric_from_graph(recovery);
  metric_field.apply_gradation();

  const std::vector<double> vertex_quality =
    metric_field.compute_vertexwise_cell_quality(study_mapping,
                                                 QGaussSimplex<dim>(3),
                                                 QGauss<1>(3));

  const auto &quality_name =
    this->param.metric_fields[0].mesh_quality_output_name;
  auto output_field =
    std::make_unique<PostProcessingTools::ContinuousDataField<dim>>(
      *this->triangulation,
      false,
      1,
      std::vector<std::string>{quality_name},
      std::vector<DataComponentInterpretation::DataComponentInterpretation>{
        DataComponentInterpretation::component_is_scalar});

  const auto &support_points = output_field->get_unit_support_points();
  std::vector<unsigned int> support_to_vertex(support_points.size(),
                                              numbers::invalid_unsigned_int);
  for (unsigned int q = 0; q < support_points.size(); ++q)
    for (unsigned int v = 0; v < dim + 1; ++v)
      if (support_points[q].distance(
            ReferenceCells::get_simplex<dim>().template vertex<dim>(v)) < 1e-12)
      {
        support_to_vertex[q] = v;
        break;
      }

  for (const auto vertex : support_to_vertex)
    Assert(vertex != numbers::invalid_unsigned_int, ExcInternalError());

  for (const auto &cell : this->dof_handler->active_cell_iterators())
    if (cell->is_locally_owned())
    {
      std::vector<std::vector<double>> values(support_points.size(),
                                              std::vector<double>(1));
      for (unsigned int q = 0; q < support_points.size(); ++q)
        values[q][0] = vertex_quality[cell->vertex_index(support_to_vertex[q])];
      output_field->set_cell_values(cell, values);
    }

  this->postproc_handler->add_continuous_data_field(std::move(output_field));
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::output_line_probe()
{
  const auto &probe = this->param.postprocessing.line_probe;
  const bool  due =
    probe.enable && probe.write_results &&
    (this->time_handler.current_time_iteration % probe.output_frequency == 0 ||
     this->time_handler.is_finished());
  if (!due)
    return;

  TimerOutput::Scope timer(this->computing_timer, "Write CHNS line probe");

  AssertThrow(probe.start.size() >= dim,
              ExcMessage("Line-probe start point has too few coordinates."));
  AssertThrow(probe.end.size() >= dim,
              ExcMessage("Line-probe end point has too few coordinates."));
  AssertThrow(probe.n_points >= 2,
              ExcMessage("Line probe needs at least two points."));

  std::vector<Point<dim>> points(probe.n_points);
  for (unsigned int i = 0; i < probe.n_points; ++i)
  {
    const double s =
      static_cast<double>(i) / static_cast<double>(probe.n_points - 1);
    for (unsigned int d = 0; d < dim; ++d)
      points[i][d] = (1. - s) * probe.start[d] + s * probe.end[d];
  }

  Utilities::MPI::RemotePointEvaluation<dim, dim> cache;
  const auto                                      pressure_values =
    VectorTools::point_values<1>(*this->moving_mapping,
                                 *this->dof_handler,
                                 *this->present_solution,
                                 points,
                                 cache,
                                 VectorTools::EvaluationFlags::avg,
                                 this->ordering->p_lower);
  AssertThrow(cache.all_points_found(),
              ExcMessage("At least one line-probe point was not found."));

  const auto tracer_values =
    VectorTools::point_values<1>(cache,
                                 *this->dof_handler,
                                 *this->present_solution,
                                 VectorTools::EvaluationFlags::avg,
                                 this->ordering->phi_lower);
  const auto potential_values =
    VectorTools::point_values<1>(cache,
                                 *this->dof_handler,
                                 *this->present_solution,
                                 VectorTools::EvaluationFlags::avg,
                                 this->ordering->mu_lower);
  const auto velocity_values =
    VectorTools::point_values<dim>(cache,
                                   *this->dof_handler,
                                   *this->present_solution,
                                   VectorTools::EvaluationFlags::avg,
                                   this->ordering->u_lower);
  const auto tracer_gradients =
    VectorTools::point_gradients<1>(cache,
                                    *this->dof_handler,
                                    *this->present_solution,
                                    VectorTools::EvaluationFlags::avg,
                                    this->ordering->phi_lower);

  std::vector<double> psi_values;
  if constexpr (with_enlarged)
    psi_values = VectorTools::point_values<1>(cache,
                                              *this->dof_handler,
                                              *this->present_solution,
                                              VectorTools::EvaluationFlags::avg,
                                              this->ordering->psi_lower);

  const auto &chp    = this->param.cahn_hilliard;
  const auto  marker = CahnHilliard::get_material_phase_function(chp);
  const auto  marker_d =
    CahnHilliard::get_material_phase_derivative_function(chp);
  const auto marker_dd =
    CahnHilliard::get_material_phase_second_derivative_function(chp);
  const auto mobility_limiter =
    CahnHilliard::get_mobility_limiter_function(chp);
  const auto mobility_evaluator =
    CahnHilliard::get_mobility_evaluation_function(chp);

  const double epsilon     = chp.epsilon_interface;
  const double sigma_tilde = 3. / (2. * std::sqrt(2.)) * chp.surface_tension;
  double       adaptive_coefficient = 0.;
  double       adaptive_delta       = 0.;
  if (chp.mobility_model ==
      Parameters::CahnHilliard<dim>::MobilityModel::adaptive)
  {
    adaptive_coefficient = chp.adaptive_mobility_n * std::sqrt(2.) * epsilon *
                           epsilon * epsilon / sigma_tilde;
    adaptive_delta = chp.adaptive_mobility_delta;
  }
  else if (chp.mobility_model ==
           Parameters::CahnHilliard<dim>::MobilityModel::adaptive_mobility_2)
  {
    adaptive_coefficient = chp.adaptive_mobility_2_n * 2. * epsilon * epsilon *
                           epsilon * epsilon / sigma_tilde;
    adaptive_delta = chp.adaptive_mobility_2_delta;
  }
  else if (chp.mobility_model ==
           Parameters::CahnHilliard<dim>::MobilityModel::adaptive_mobility_3)
  {
    adaptive_coefficient =
      chp.adaptive_mobility_3_n * epsilon * epsilon / sigma_tilde;
    adaptive_delta = chp.adaptive_mobility_3_delta;
  }

  const double double_well_coefficient =
    CahnHilliard::potential_double_well_coefficient(chp, sigma_tilde);
  const double gradient_coefficient =
    CahnHilliard::potential_gradient_coefficient(chp, sigma_tilde);
  const bool   use_ding_horriche = CahnHilliard::is_ding_horriche_model(chp);
  const bool   use_abels_nlm     = CahnHilliard::is_abels_nlm_model(chp);
  const std::string reconstructed_pressure_name =
    use_ding_horriche ? "pressure_hat" :
    use_abels_nlm     ? "pressure_sharp" :
                        "pressure_abels";
  const double free_energy_prefactor =
    use_ding_horriche ? CahnHilliard::ding_horriche_capillary_coefficient(chp) :
                        1.;

  if (this->mpi_rank != 0)
    return;

  const std::string filename =
    this->param.output.output_dir + probe.output_prefix + ".csv";
  const bool first_iteration = this->time_handler.current_time_iteration == 0;
  const bool write_header = first_iteration || !std::ifstream(filename).good();
  std::ofstream out(filename,
                    first_iteration ? std::ios::trunc : std::ios::app);
  AssertThrow(out,
              ExcMessage("Could not open line-probe CSV '" + filename + "'."));
  out << std::setprecision(probe.precision);

  if (write_header)
  {
    out << "time,iteration,point_index";
    for (unsigned int d = 0; d < dim; ++d)
      out << ",x" << d;
    out << ",pressure," << reconstructed_pressure_name
        << ",pressure_yl,pressure_physical"
        << ",pressure_free_energy,tracer,potential,mobility";
    for (unsigned int d = 0; d < dim; ++d)
      out << ",velocity_" << d;
    out << ",velocity_norm";
    if constexpr (with_enlarged)
      out << ",psi";
    out << '\n';
  }

  for (unsigned int i = 0; i < probe.n_points; ++i)
  {
    const double phi             = tracer_values[i];
    const double mu              = potential_values[i];
    const double material_marker = marker(chp, phi);
    const double reconstructed_pressure =
      use_ding_horriche ? pressure_values[i] :
                          pressure_values[i] + material_marker * mu;
    const double pressure_yl = reconstructed_pressure;
    const double double_well = .25 * (phi * phi - 1.) * (phi * phi - 1.);
    const double free_energy =
      free_energy_prefactor *
      (double_well_coefficient * double_well +
       .5 * gradient_coefficient * tracer_gradients[i].norm_square());

    const double mobility_phi = mobility_limiter(phi);
    const auto   mobility     = mobility_evaluator(chp,
                                             marker(chp, mobility_phi),
                                             marker_d(chp, mobility_phi),
                                             marker_dd(chp, mobility_phi),
                                             velocity_values[i],
                                             tracer_gradients[i],
                                             adaptive_coefficient,
                                             adaptive_delta);

    out << this->time_handler.current_time << ','
        << this->time_handler.current_time_iteration << ',' << i;
    for (unsigned int d = 0; d < dim; ++d)
      out << ',' << points[i][d];
    out << ',' << pressure_values[i] << ',' << reconstructed_pressure << ','
        << pressure_yl << ',' << pressure_yl - free_energy << ',' << free_energy
        << ',' << phi << ',' << mu << ',' << mobility.value;
    for (unsigned int d = 0; d < dim; ++d)
      out << ',' << velocity_values[i][d];
    out << ',' << velocity_values[i].norm();
    if constexpr (with_enlarged)
      out << ',' << psi_values[i];
    out << '\n';
  }
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  solver_specific_post_processing()
{
  {
    TimerOutput::Scope t(this->computing_timer,
                         "Compute multiphase indicators");

    this->postproc_handler->compute_multiphase_indicators(
      *this->ordering,
      *this->dof_handler,
      *this->moving_mapping,
      *this->quadrature,
      *this->present_solution,
      this->time_handler);
  }

  output_line_probe();

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
