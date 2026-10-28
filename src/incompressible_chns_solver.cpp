#include <assembly/ale_geometry.h>
#include <assembly/elasticity_assemblers.h>
#include <assembly/incompressible_chns_assemblers.h>
#include <assembly/moving_mesh_forcing_forms.h>
#include <assembly/pseudosolid_forms.h>
#include <assembly/stabilization_forms.h>
#include <cahn_hilliard.h>
#include <compare_matrix.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/quadrature_lib.h>
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
#include <error_estimation/patches.h>
#include <error_estimation/solution_recovery.h>
#include <errors.h>
#include <incompressible_chns_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <mesh_and_dof_tools.h>
#include <mesh_forcing_postprocessing.h>
#include <metric_field.h>
#include <scratch_data.h>
#include <utilities.h>

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>

template <int dim, bool with_moving_mesh, bool with_enlarged>
CHNSSolver<dim, with_moving_mesh, with_enlarged>::CHNSSolver(
  const ParameterReader<dim> &param)
  : NavierStokesSolver<dim, with_moving_mesh>(param)
{
  // The default (Abels) model is not announced, so that runs using it keep the
  // same output as the master branch, whose reference outputs predate this
  // diagnostic line.
  if (!CahnHilliard::is_abels_model(param.cahn_hilliard))
    this->pcout << "CHNS model: "
                << CahnHilliard::model_name(param.cahn_hilliard) << std::endl;

  AssertThrow(!param.postprocessing.chns_conservation.enable ||
                CahnHilliard::is_quasi_incompressible_model(
                  param.cahn_hilliard),
              ExcMessage("CHNS conservation postprocessing currently "
                         "requires CHNS model = quasi_incompressible."));
  AssertThrow(
    !param.postprocessing.chns_energy.enable ||
      CahnHilliard::is_quasi_incompressible_model(param.cahn_hilliard) ||
      CahnHilliard::is_stepien_model(param.cahn_hilliard),
    ExcMessage("CHNS energy postprocessing requires CHNS model = stepien "
               "or quasi_incompressible."));
  AssertThrow(
    !param.postprocessing.chns_phase_energy.enable ||
      CahnHilliard::is_quasi_incompressible_model(param.cahn_hilliard) ||
      CahnHilliard::is_stepien_model(param.cahn_hilliard),
    ExcMessage("CHNS phase energy postprocessing requires CHNS model = stepien "
               "or quasi_incompressible."));

  if (CahnHilliard::is_quasi_incompressible_model(param.cahn_hilliard))
  {
    AssertThrow(!with_moving_mesh && !with_enlarged,
                ExcMessage(
                  "Quasi-incompressible CHNS is implemented on a fixed mesh."));
    AssertThrow(param.time_integration.scheme ==
                    Parameters::TimeIntegration::Scheme::BDF1 ||
                  param.time_integration.scheme ==
                    Parameters::TimeIntegration::Scheme::BDF2,
                ExcMessage(
                  "Quasi-incompressible CHNS requires scheme = BDF1 or BDF2."));
    AssertThrow(param.time_integration.scheme !=
                    Parameters::TimeIntegration::Scheme::BDF2 ||
                  param.time_integration.bdfstart ==
                    Parameters::TimeIntegration::BDFStart::BDF1,
                ExcMessage(
                  "Quasi-incompressible BDF2 requires bdf start method = BDF1 "
                  "to construct a consistent second history state."));
    AssertThrow(
      param.time_integration.scheme !=
          Parameters::TimeIntegration::Scheme::BDF2 ||
        (param.time_integration.bdf_starting_step_ratio > 0. &&
         param.time_integration.bdf_starting_step_ratio < 1.),
      ExcMessage(
        "Quasi-incompressible BDF2 requires 0 < bdf start step ratio < 1."));
    AssertThrow(!param.finite_elements.stabilization &&
                  !param.stabilization.enable_supg &&
                  !param.stabilization.enable_tracer_supg,
                ExcMessage("Disable all stabilization options for the "
                           "Quasi-incompressible CHNS Galerkin scheme."));
    AssertThrow(param.finite_elements.tracer_degree >= 1 &&
                  param.finite_elements.tracer_degree ==
                    param.finite_elements.potential_degree &&
                  param.finite_elements.pressure_degree >= 1 &&
                  param.finite_elements.velocity_degree ==
                    param.finite_elements.pressure_degree + 1,
                ExcMessage(
                  "Quasi-incompressible CHNS requires equal phase/potential "
                  "degrees >= 1 and a Taylor-Hood velocity/pressure "
                  "pair (e.g. 2/1)."));
    AssertThrow(param.cahn_hilliard.epsilon_interface > 0. &&
                  param.cahn_hilliard.surface_tension > 0.,
                ExcMessage("Interface thickness and surface tension must be "
                           "strictly positive."));
    AssertThrow(param.physical_properties.fluids.size() == 2,
                ExcMessage(
                  "Quasi-incompressible CHNS requires exactly two fluids."));
    for (const auto &fluid : param.physical_properties.fluids)
      AssertThrow(fluid.density > 0. && fluid.kinematic_viscosity >= 0.,
                  ExcMessage(
                    "Quasi-incompressible CHNS requires positive densities "
                    "and non-negative viscosities."));
    AssertThrow(!param.with_tree_based_adaptation() &&
                  !param.with_metric_based_adaptation(),
                ExcMessage(
                  "Quasi-incompressible CHNS currently requires a fixed "
                  "mesh; disable mesh adaptation."));
    AssertThrow(param.bc_data.fix_pressure_constant !=
                  param.bc_data.enforce_zero_mean_pressure,
                ExcMessage(
                  "Quasi-incompressible CHNS requires exactly one pressure "
                  "gauge: fix pressure constant or enforce zero "
                  "mean pressure."));
    // No open-boundary momentum or phase flux terms are assembled. Prescribed
    // manufactured traces remain available; their energy includes external
    // work.
    for (const auto &[id, bc] : param.fluid_bc)
      AssertThrow(bc.type == BoundaryConditions::Type::no_slip ||
                    bc.type == BoundaryConditions::Type::slip ||
                    (param.mms_param.enable &&
                     bc.type == BoundaryConditions::Type::velocity_mms),
                  ExcMessage(
                    "Quasi-incompressible CHNS supports no_slip/slip fluid "
                    "boundaries and manufactured velocity traces; "
                    "unsupported boundary id " +
                    std::to_string(id)));
    for (const auto &[id, bc] : param.cahn_hilliard_bc)
      AssertThrow(bc.type == BoundaryConditions::Type::no_flux ||
                    (param.mms_param.enable &&
                     bc.type == BoundaryConditions::Type::dirichlet_mms),
                  ExcMessage(
                    "Quasi-incompressible CHNS supports natural no_flux "
                    "Cahn-Hilliard boundaries and manufactured "
                    "traces; unsupported boundary id " +
                    std::to_string(id)));
    if (param.cahn_hilliard.quasi_incompressible_periodic_box)
    {
      AssertThrow(param.physical_properties.body_force.norm_square() == 0.,
                  ExcMessage(
                    "The Quasi-incompressible CHNS periodic box requires "
                    "zero body force: a nonzero constant gravity "
                    "potential is not periodic."));
      AssertThrow(param.finite_elements.use_quads &&
                    param.mesh.deal_ii_preset_mesh == "rectangle",
                  ExcMessage("The periodic box requires use quads = true "
                             "and dealii preset mesh = rectangle."));
      AssertThrow(param.bc_data.fix_pressure_constant &&
                    !param.bc_data.enforce_zero_mean_pressure,
                  ExcMessage("For a periodic box, pin the pressure at an "
                             "interior support point and disable the zero "
                             "mean pressure constraint."));
      AssertThrow(param.fluid_bc.empty() && param.cahn_hilliard_bc.empty(),
                  ExcMessage("For a periodic box, set number = 0 in both "
                             "fluid and CahnHilliard boundary conditions; "
                             "periodicity applies to every field."));
    }
    // Initial VTU output precedes the diagnostics hook. Prepare the directory
    // here so a fresh run works even before the first diagnostics CSV exists.
    unsigned int output_directory_ready = 1;
    if (this->mpi_rank == 0)
    {
      std::error_code error;
      std::filesystem::create_directories(param.output.output_dir, error);
      output_directory_ready = error ? 0 : 1;
    }
    AssertThrow(
      Utilities::MPI::min(output_directory_ready, this->mpi_communicator) == 1,
      ExcMessage("Cannot create Quasi-incompressible CHNS output directory " +
                 param.output.output_dir));
  }

  if constexpr (with_moving_mesh && !with_enlarged)
    AssertThrow(
      std::abs(param.cahn_hilliard.mff_enlarged_compression_factor) < 1e-14,
      ExcMessage(
        "mff_enlarged_compression_factor is only available with the enlarged "
        "solver. In non-enlarged runs, set it to 0 and use "
        "mff_physics_compression_factor as the single compression term."));

  if (CahnHilliard::is_stepien_model(param.cahn_hilliard))
  {
    AssertThrow(!with_moving_mesh,
                ExcMessage("Stepien ALE is not implemented."));
    AssertThrow(!param.finite_elements.stabilization,
                ExcMessage("Stepien stabilization is not implemented."));
    AssertThrow(!(param.cahn_hilliard.stepien_conservative_transport ||
                  param.cahn_hilliard.stepien_skew_symmetric_inertia) ||
                  (!param.stabilization.enable_supg &&
                   !param.stabilization.enable_tracer_supg),
                ExcMessage(
                  "The experimental Stepien article forms require "
                  "enable supg = false and enable tracer supg = false."));
  }

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
  const double filtered_phi =
    CahnHilliard::get_limiter_function(cahn_hilliard_param)(phi);
  const double property_phi_derivative =
    CahnHilliard::tracer_limiter_derivative(
      phi,
      cahn_hilliard_param.with_tracer_limiter ||
        CahnHilliard::is_quasi_incompressible_model(cahn_hilliard_param));
  const double rho0         = physical_properties.fluids[0].density;
  const double rho1         = physical_properties.fluids[1].density;
  const double rho_product = rho0 * rho1;
  const double drho        = 0.5 * (rho0 - rho1);
  const double rho  = CahnHilliard::linear_mixing(filtered_phi, rho0, rho1);
  const double eta0 = rho0 * physical_properties.fluids[0].kinematic_viscosity;
  const double eta1 = rho1 * physical_properties.fluids[1].kinematic_viscosity;
  const double eta  = CahnHilliard::linear_mixing(filtered_phi, eta0, eta1);
  const double pr0 =
    physical_properties.fluids[0].pressure_ref;
  const double pr1 =
    physical_properties.fluids[1].pressure_ref;
  const double dpr = 0.5 * (pr0 - pr1);
  const auto   evaluated_mobility =
    CahnHilliard::evaluate_mobility(cahn_hilliard_param, phi, rho0, rho1);
  const double M       = evaluated_mobility.value;
  const double dM_dphi = evaluated_mobility.first_derivative;

  const double diff_flux_factor = M * 0.5 * (rho1 - rho0);
  // const double drhodphi =
  //   CahnHilliard::linear_mixing_derivative(filtered_phi, rho0, rho1);
  const double detadphi =
    CahnHilliard::linear_mixing_derivative(filtered_phi, eta0, eta1) *
    property_phi_derivative;
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
  const double   div_u       = mms.exact_velocity->divergence(p);
  const double   lap_p       = mms.exact_pressure->laplacian(p);

  const double   mu          = mms.exact_potential->value(p);
  Tensor<1, dim> grad_mu     = mms.exact_potential->gradient(p);
  const double   lap_mu      = mms.exact_potential->laplacian(p);

  Tensor<1, dim> grad_phi    = mms.exact_tracer->gradient(p);
  const double   lap_phi     = mms.exact_tracer->laplacian(p);
  const double   dphidt      = mms.exact_tracer->time_derivative(p);
  const double   Dphi        = dphidt + u * grad_phi;

  if (CahnHilliard::is_quasi_incompressible_model(cahn_hilliard_param))
  {
    // Continuous strong form (10), with positive extensions only in the
    // kinetic inertia and viscosity. See equations.md. MMS forces the
    // continuous PDE, independently of the selected temporal discretization.
    const double rho_affine = CahnHilliard::linear_mixing(phi, rho0, rho1);
    const double alpha      = (rho1 - rho0) / (rho0 + rho1);
    const auto   grad_w     = grad_mu + alpha * grad_p;
    const double div_flux =
      M * (lap_mu + alpha * lap_p) + dM_dphi * (grad_phi * grad_w);
    const double lambda     = -2. / dim;
    const auto   div_stress = eta * (lap_u + (1. + lambda) * grad_div_u) +
                            2. * detadphi * (grad_phi * symmetrize(grad_u)) +
                            lambda * detadphi * div_u * grad_phi;
    const double div_mass_flux = drho * (u * grad_phi) + rho_affine * div_u;
    const auto   momentum =
      rho * dudt_eulerian + rho_affine * uDotGradu +
      0.5 * u * (drho * property_phi_derivative * dphidt + div_mass_flux) -
      div_stress + grad_p + phi * grad_mu - rho_affine * body_force;
    for (unsigned int d = 0; d < dim; ++d)
      values[u_lower + d] = -momentum[d];
    values[p_lower]   = div_u - alpha * div_flux;
    values[phi_lower] = -(Dphi + phi * div_u - div_flux);
    values[mu_lower] =
      -(mu + potential_gradient_coefficient * lap_phi -
        potential_double_well_coefficient * phi * (phi * phi - 1.));
    return;
  }

  const Tensor<1, dim> grad_q =(rho * grad_mu -
                               drho * grad_p +
                               drho * (mu - dpr) * grad_phi) /
                              rho_product;

  const double lap_q =
    (rho * lap_mu +
     drho * (1. + property_phi_derivative) * (grad_phi * grad_mu) +
     drho * (mu - dpr) * lap_phi - drho * lap_p) /
    rho_product;

  const double div_mobility_grad_q =
    M * lap_q + dM_dphi * (grad_phi * grad_q);

  const bool is_stepien = CahnHilliard::is_stepien_model(cahn_hilliard_param);

  Tensor<1, dim> J_flux      = diff_flux_factor * grad_mu;
  Tensor<1, dim> div_viscous = (eta * (lap_u + grad_div_u) +
                                2. * detadphi * grad_phi * symmetrize(grad_u));
  // Divergence of the second-viscosity stress lambda (div v) I, which the
  // Stepien momentum equation carries because div(v) != 0.
  if (is_stepien)
  {
    const double lambda_visc =
      CahnHilliard::stokes_second_viscosity_ratio * eta;
    const double dlambda_visc_dphi =
      CahnHilliard::stokes_second_viscosity_ratio * detadphi;
    div_viscous +=
      lambda_visc * grad_div_u + dlambda_visc_dphi * div_u * grad_phi;
  }
  const Tensor<1, dim> momentum_diffusive_inertia =
    CahnHilliard::use_abels_diffusive_inertia(cahn_hilliard_param) ?
      J_flux * grad_u :
      Tensor<1, dim>();
  Tensor<1, dim> momentum_capillary_force;
  if (CahnHilliard::use_abels_capillary_phi_grad_mu(cahn_hilliard_param))
    momentum_capillary_force = phi * grad_mu;
  else if (is_stepien)
    momentum_capillary_force = (dpr - mu) * grad_phi;
  else
    momentum_capillary_force =
      -CahnHilliard::ding_horriche_capillary_coefficient(
        cahn_hilliard_param) *
      mu * grad_phi;
  // Navier-Stokes momentum (velocity) source term
  Tensor<1, dim> f = -(rho * (dudt_eulerian + uDotGradu - body_force) +
                       momentum_diffusive_inertia + grad_p - div_viscous +
                       momentum_capillary_force);
  if (cahn_hilliard_param.stepien_skew_symmetric_inertia)
    f -= 0.5 * u * (drho * property_phi_derivative * Dphi + rho * div_u);
  for (unsigned int d = 0; d < dim; ++d)
    values[u_lower + d] = f[d];

  // Mass conservation (pressure) source term,
  // for - div(u) + f = 0 -> f = div(u_mms). The Stepien model adds the
  // quasi-incompressible term -(drho/rho) Dphi/Dt to the residual.
  if (cahn_hilliard_param.stepien_conservative_transport)
  {
    const double alpha           = (rho1 - rho0) / (rho0 + rho1);
    const double flux_conversion = (rho0 + rho1) / (2. * rho_product);
    values[p_lower] = div_u - alpha * flux_conversion * div_mobility_grad_q;
  }
  else if (is_stepien)
    values[p_lower] = div_u + (drho / rho) * Dphi;
  else
    values[p_lower] = div_u;

  if constexpr (with_moving_mesh)
  {
    // Pseudosolid (mesh position) source term
    Tensor<1, dim> f_PS =
      mms.exact_mesh_position->divergence_elastic_stress_tensor(
        physical_properties.pseudosolids[0], p);

    for (unsigned int d = 0; d < dim; ++d)
      values[x_lower + d] = f_PS[d];
  }

  // Tracer source term. The Stepien phase equation is
  //   A Dphi/Dt - div(M grad q) + f = 0, with A = rho1 rho2 / rho.
  if (cahn_hilliard_param.stepien_conservative_transport)
  {
    const double flux_conversion = (rho0 + rho1) / (2. * rho_product);
    values[phi_lower] =
      -(Dphi + phi * div_u - flux_conversion * div_mobility_grad_q);
  }
  else if (is_stepien)
  {
    const double stepien_A = rho_product / rho;
    values[phi_lower]      = -(stepien_A * Dphi - div_mobility_grad_q);
  }
  else
    values[phi_lower] =
      -(dphidt + u * grad_phi - (M * lap_mu + dM_dphi * (grad_phi * grad_mu)));

  // Potential source term
  values[mu_lower] =
    -(mu - potential_double_well_coefficient * phi * (phi * phi - 1.) +
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
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::update_simulation_parameters(
  const unsigned int fixed_point_iteration)
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
                                               this->param);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::setup_assemblers()
{
  assemblers.clear();

  /**
   * Register the energy-stable quasi-incompressible assembler or a stabilized
   * CHNS assembler through the common factory. Other configurations retain
   * the inline volume assembly, including the legacy Stepien and ALE models.
   */
  if (!CahnHilliard::is_quasi_incompressible_model(this->param.cahn_hilliard) &&
      !this->param.stabilization.enable_supg &&
      !this->param.stabilization.enable_tracer_supg)
    return;

  AssertThrow(
    CahnHilliard::is_abels_model(this->param.cahn_hilliard) ||
      CahnHilliard::is_stepien_model(this->param.cahn_hilliard) ||
      CahnHilliard::is_quasi_incompressible_model(this->param.cahn_hilliard),
    ExcMessage("The \"Stabilization\" subsection is only implemented for the "
               "Abels and Stepien models. Use \"set stabilization = true\" "
               "under the finite elements subsection for the other CHNS "
               "models."));

  if constexpr (with_enlarged)
    AssertThrow(false,
                ExcMessage("The \"Stabilization\" subsection is not implemented "
                           "for the enlarged CHNS solver."));

  Assembly::IncompressibleCHNS::
    setup_assemblers<dim, ScratchData, CopyData, with_moving_mesh>(
      this->param, *this->ordering, this->coupling_table, assemblers);
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim,
                with_moving_mesh,
                with_enlarged>::create_solver_specific_zero_constraints()
{
  if (CahnHilliard::is_quasi_incompressible_model(this->param.cahn_hilliard) &&
      !this->param.cahn_hilliard.quasi_incompressible_periodic_box)
  {
    // An omitted velocity boundary would otherwise become an open natural
    // boundary. Validate the actual mesh, including distributed ownership.
    unsigned int closed = 1;
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto face : cell->face_indices())
          if (cell->face(face)->at_boundary() &&
              this->param.fluid_bc.count(cell->face(face)->boundary_id()) == 0)
            closed = 0;
    AssertThrow(Utilities::MPI::min(closed, this->mpi_communicator) == 1,
                ExcMessage("Every non-periodic Quasi-incompressible CHNS mesh "
                           "boundary needs an explicit fluid boundary "
                           "condition; open boundaries are not implemented."));
  }
  if (this->param.cahn_hilliard.quasi_incompressible_periodic_box)
  {
    const auto pressure_mask =
      fe->component_mask(FEValuesExtractors::Scalar(this->ordering->p_lower));
    const auto boundary_dofs =
      DoFTools::extract_boundary_dofs(*this->dof_handler, pressure_mask);
    const unsigned int boundary_pin =
      boundary_dofs.is_element(this->constrained_pressure_dof) ? 1 : 0;
    AssertThrow(Utilities::MPI::max(boundary_pin, this->mpi_communicator) == 0,
                ExcMessage("The periodic pressure reference must select an "
                           "interior pressure support point."));
    for (unsigned int d = 0; d < dim; ++d)
      DoFTools::make_periodicity_constraints(
        *this->dof_handler, 2 * d + 1, 2 * d, d, this->zero_constraints);
  }

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
  if (this->param.cahn_hilliard.quasi_incompressible_periodic_box)
    for (unsigned int d = 0; d < dim; ++d)
      DoFTools::make_periodicity_constraints(
        *this->dof_handler, 2 * d + 1, 2 * d, d, this->nonzero_constraints);

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

      if (CahnHilliard::is_quasi_incompressible_model(
            this->param.cahn_hilliard))
      {
        // Both diffusion rows couple to phi, mu and p; the skew momentum
        // and conservative phase rows couple to all four unknowns.
        if (!this->ordering->is_potential(i) || this->ordering->is_tracer(j) ||
            this->ordering->is_potential(j))
          coupling_table[i][j] = DoFTools::always;
        continue;
      }

      // u couples to all variables
      if (this->ordering->is_velocity(i))
        coupling_table[i][j] = DoFTools::always;

      // p couples to u and x. PSPG also couples p to p, phi and mu through
      // the strong momentum residual.
      if (this->ordering->is_pressure(i) &&
          (this->ordering->is_velocity(j) || this->ordering->is_position(j) ||
           this->param.stabilization.enable_supg))
        coupling_table[i][j] = DoFTools::always;

      // PSPG: p couples to p, phi and mu
      if (this->ordering->is_pressure(i) &&
          (this->ordering->is_pressure(j) || this->ordering->is_tracer(j) ||
           this->ordering->is_potential(j)) &&
          this->param.finite_elements.stabilization)
        coupling_table[i][j] = DoFTools::always;

      // Stepien: the continuity equation has a quasi-incompressible Dphi/Dt
      // term, so pressure couples to the tracer.
      if (this->ordering->is_pressure(i) && this->ordering->is_tracer(j) &&
          CahnHilliard::is_stepien_model(this->param.cahn_hilliard))
        coupling_table[i][j] = DoFTools::always;

      // The conservative divergence constraint contains the same diffusive
      // flux as the phase equation, including its pressure and mu dependence.
      if (this->ordering->is_pressure(i) &&
          this->param.cahn_hilliard.stepien_conservative_transport)
        coupling_table[i][j] = DoFTools::always;

      // x couples x,phi,u
      if constexpr (with_moving_mesh)
        if (this->ordering->is_position(i) &&
            (this->ordering->is_position(j) || this->ordering->is_tracer(j) ||
             this->ordering->is_velocity(j)))
          coupling_table[i][j] = DoFTools::always;

      // phi couples to u, phi, mu, x; and to p for the Stepien model, whose
      // diffusive flux q depends on the pressure.
      if (this->ordering->is_tracer(i))
        if (!this->ordering->is_pressure(j) ||
            CahnHilliard::is_stepien_model(this->param.cahn_hilliard))
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

  /**
   * When the stabilization of the master branch is requested, the whole volume
   * assembly is delegated to its VolumeAssembler, which computes the base terms
   * and the SUPG/PSPG contributions together. setup_assemblers() only fills the
   * list in the cases that assembler supports, so the inline assembly below
   * still handles every other configuration.
   */
  if (!assemblers.empty())
  {
    for (const auto &assembler : assemblers)
      assembler->assemble_matrix(scratch_data, copy_data);
    cell->get_dof_indices(local_dof_indices);
    return;
  }

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

  // Stepien quasi-incompressible model: cell-constant coefficients.
  const bool   is_stepien = CahnHilliard::is_stepien_model(cahn_hilliard);
  const double stepien_cap =
    CahnHilliard::stepien_capillary_coefficient(cahn_hilliard);
  const bool   conservative = cahn_hilliard.stepien_conservative_transport;
  const bool   skew_inertia = cahn_hilliard.stepien_skew_symmetric_inertia;
  const double rho0 = scratch_data.density0;
  const double rho1 = scratch_data.density1;
  const double dpr =
    0.5 * (this->param.physical_properties.fluids[0].pressure_ref -
           this->param.physical_properties.fluids[1].pressure_ref);

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
    // drhodphi is the constant density contrast in the Stepien flux and
    // continuity equation. The actual derivative of rho can differ when
    // the property limiter is active; only Jacobian variations use it.
    const double drhodphi = scratch_data.derivative_density_wrt_tracer[q];
    const double property_phi_derivative =
      CahnHilliard::tracer_limiter_derivative(
        scratch_data.tracer_values[q], cahn_hilliard.with_tracer_limiter);
    const double density_derivative = drhodphi * property_phi_derivative;
    const double detadphi =
      scratch_data.derivative_dynamic_viscosity_wrt_tracer[q] *
      property_phi_derivative;

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
    const auto &potential_value    = scratch_data.potential_values[q];
    const auto &potential_gradient = scratch_data.potential_gradients[q];

    const double dphidt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, tracer_value, scratch_data.previous_tracer_values);

    const double density_bdf = skew_inertia ?
                                 CahnHilliard::bdf_density_derivative(
                                   cahn_hilliard,
                                   rho0,
                                   rho1,
                                   rho,
                                   this->time_handler.bdf_coefficients,
                                   scratch_data.previous_tracer_values,
                                   q) :
                                 0.;
    const double flux_conversion =
      conservative ? (rho0 + rho1) / (2. * rho0 * rho1) : 0.;
    const double alpha = (rho1 - rho0) / (rho0 + rho1);

    // Stepien: per-quadrature coefficients (mirror the residual).
    double         stepien_A = 0.;
    double         lambda_visc = 0., dlambda_visc_dphi = 0.;
    Tensor<1, dim> stepien_grad_q;
    if (is_stepien)
    {
      stepien_A = rho0 * rho1 / rho;
      stepien_grad_q =
        (rho * potential_gradient -
         drhodphi * scratch_data.present_pressure_gradients[q] +
         drhodphi * (potential_value - dpr) * tracer_gradient) /
        (rho0 * rho1);
      // Second viscosity of the Stokes hypothesis, lambda = -2/3 eta. It
      // contributes because the Stepien velocity field is not solenoidal.
      lambda_visc = CahnHilliard::stokes_second_viscosity_ratio * eta;
      dlambda_visc_dphi =
        CahnHilliard::stokes_second_viscosity_ratio * detadphi;
    }

    // Precomputations of shape functions-independent terms
    const auto density_derivative_momentum =
      density_derivative * (dudt + u_dot_grad_u - body_force);

    if constexpr (with_moving_mesh)
    {
      to_multiply_by_phi_u_i_tr_G =
        rho * (dudt - body_force + present_velocity_gradients * u_conv) +
        *source_term_velocity;
      to_multipliy_by_phi_phi_i_tr_G =
        dphidt + u_conv * tracer_gradient + source_term_tracer;
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
        Tensor<1, dim> delta_article_flux;
        if (conservative && (comp_i == const_ordering.p_lower ||
                             comp_i == const_ordering.phi_lower))
        {
          if (comp_j == const_ordering.phi_lower)
            delta_article_flux =
              flux_conversion *
              (dM_dphi * phi_phi_j * stepien_grad_q +
               mobility *
                 (density_derivative * phi_phi_j * potential_gradient +
                  drhodphi * (potential_value - dpr) * grad_phi_phi_j) /
                 (rho0 * rho1));
          else if (comp_j == const_ordering.mu_lower)
            delta_article_flux =
              flux_conversion * mobility *
              (rho * grad_phi_mu_j + drhodphi * phi_mu_j * tracer_gradient) /
              (rho0 * rho1);
          else if (comp_j == const_ordering.p_lower)
            delta_article_flux = -flux_conversion * mobility * drhodphi *
                                 scratch_data.grad_phi_p[q][j] / (rho0 * rho1);
        }


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

            if (skew_inertia)
              local_flow_ij +=
                0.5 * density_bdf * (phi_u_i * phi_u_j) -
                0.5 * rho *
                  (phi_u_i * (grad_phi_u_j * u_conv +
                              present_velocity_gradients * phi_u_j) +
                   present_velocity_values * (grad_phi_u_i * phi_u_j) +
                   phi_u_j * (grad_phi_u_i * u_conv));

            if (use_abels_diffusive_inertia)
              local_flow_ij +=
                phi_u_i * diffusive_flux_factor * grad_phi_u_j *
                potential_gradient;

            local_flow_ij +=
              2. * eta * scalar_product(sym_grad_phi_u_i, sym_grad_phi_u_j);

            if (is_stepien)
              // Second-viscosity stress lambda (div dv)(div w)
              local_flow_ij += lambda_visc * div_phi_u_j * div_phi_u_i;
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
                  // Abels Korteweg term: phi * grad(mu).
                  local_flow_ij +=
                    phi_u_i * (tracer_value *
                                 Assembly::ALE::gradient_variation(
                                   potential_gradient, G) +
                               tracer_value * potential_gradient * trG);
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
            if (skew_inertia)
                local_flow_ij +=
                  0.5 * density_derivative * phi_phi_j *
                  (bdf_c0 * (phi_u_i * present_velocity_values) -
                   phi_u_i * u_dot_grad_u -
                   present_velocity_values * (grad_phi_u_i * u_conv));
            local_flow_ij +=
              2. * detadphi * phi_phi_j *
              scalar_product(sym_grad_phi_u_i, present_velocity_sym_gradients);
            if (use_abels_capillary_phi_grad_mu)
              local_flow_ij +=
                phi_u_i * phi_phi_j * potential_gradient;
            else if (is_stepien)
            {
              // Capillary (-muhat)grad(phi) (coeff 1) and pressure jump
              // +dpr grad(phi): d/dphi -> (dpr - muhat) w.grad(dphi).
              local_flow_ij += (dpr - stepien_cap * potential_value) *
                               (phi_u_i * grad_phi_phi_j);
              // Second viscosity through eta(phi):
              //   dlambda/dphi dphi (div v)(div w).
              local_flow_ij += dlambda_visc_dphi * phi_phi_j *
                               present_velocity_divergence * div_phi_u_i;
            }
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
                 tracer_value * grad_phi_mu_j);
            else if (is_stepien)
              // Capillary -muhat grad(phi): d/dmuhat -> -dmuhat w.grad(phi).
              local_flow_ij +=
                -stepien_cap * phi_mu_j * (phi_u_i * tracer_gradient);
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
          if (conservative)
          {
            local_flow_ij -= phi_p_i * div_phi_u_j;
            local_flow_ij -=
              alpha * (scratch_data.grad_phi_p[q][i] * delta_article_flux);
          }
          else
          {
            if (const_ordering.u_lower <= comp_j &&
                comp_j < const_ordering.u_upper)
            {
              // Continuity : variation w.r.t. u
              local_flow_ij += -phi_p_i * div_phi_u_j;
              if (is_stepien)
                  // d/du of the quasi-incompressible term
                  //   -q (drho/rho) (v.grad phi)
                  local_flow_ij +=
                    -phi_p_i * (drhodphi / rho) * (phi_u_j * tracer_gradient);
            }

            if (is_stepien && comp_j == const_ordering.phi_lower)
            {
              // Continuity vs phi (Stepien): derivative of
              //   -q (drho/rho) (dphi/dt + v.grad phi).
              // d(1/rho)/dphi gives +drho^2/rho^2; d(Dphi/Dt)/dphi gives
              // c0 phi_phi_j + v.grad(phi_phi_j).
              local_flow_ij +=
                phi_p_i * (drhodphi * density_derivative / (rho * rho)) *
                  phi_phi_j * (dphidt + u_conv * tracer_gradient) -
                phi_p_i * (drhodphi / rho) *
                  (bdf_c0 * phi_phi_j + u_conv * grad_phi_phi_j);
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
        }

        /**
         * Tracer equation
         */
        if (comp_i == const_ordering.phi_lower)
        {
          if (conservative)
          {
            local_flow_ij += grad_phi_phi_i * delta_article_flux;
            if (const_ordering.u_lower <= comp_j &&
                comp_j < const_ordering.u_upper)
              local_flow_ij -= tracer_value * (grad_phi_phi_i * phi_u_j);
            if (comp_j == const_ordering.phi_lower)
              local_flow_ij += phi_phi_i * bdf_c0 * phi_phi_j -
                               phi_phi_j * (grad_phi_phi_i * u_conv);
          }
          else
          {
            if (const_ordering.u_lower <= comp_j &&
                comp_j < const_ordering.u_upper)
            {
              // Advection (Stepien scales it by the coefficient A)
              if (is_stepien)
                  local_flow_ij +=
                    stepien_A * phi_phi_i * (phi_u_j * tracer_gradient);
              else
                  local_flow_ij += phi_phi_i * phi_u_j * tracer_gradient;
            }
            if constexpr (with_moving_mesh)
            {
              // Variation of tracer equation w.r.t. x
              if (const_ordering.x_lower <= comp_j &&
                  comp_j < const_ordering.x_upper)
              {
                  const Tensor<2, dim> &G   = (*grad_phi_x_moving)[j];
                  const double          trG = Assembly::ALE::jacobian_trace(G);

                  local_flow_ij +=
                    phi_phi_i * to_multipliy_by_phi_phi_i_tr_G * trG;

                  // ALE (advection) term
                  local_flow_ij +=
                    phi_phi_i *
                    (Assembly::ALE::mesh_velocity_variation(bdf_c0, *phi_x_j) *
                       tracer_gradient +
                     u_conv *
                       Assembly::ALE::gradient_variation(tracer_gradient, G));
                  // Laplacian term
                  local_flow_ij +=
                    mobility *
                    (Assembly::ALE::gradient_variation(grad_phi_phi_i, G) *
                       potential_gradient +
                     grad_phi_phi_i *
                       Assembly::ALE::gradient_variation(potential_gradient,
                                                         G) +
                     (grad_phi_phi_i * potential_gradient) * trG);
              }
            }
            if (comp_j == const_ordering.phi_lower)
            {
              if (is_stepien)
              {
                  // term 1: dA/dphi * (dphi/dt + v.grad phi), dA/dphi = -A
                  // drho/rho
                  local_flow_ij +=
                    phi_phi_i * (-stepien_A * density_derivative / rho) *
                    phi_phi_j * (dphidt + u_conv * tracer_gradient);
                  // term 2: A*(c0 dphi + v.grad dphi)
                  local_flow_ij +=
                    phi_phi_i * stepien_A *
                    (bdf_c0 * phi_phi_j + u_conv * grad_phi_phi_j);
                  // term 3: dM/dphi in the flux M * grad q
                  local_flow_ij +=
                    grad_phi_phi_i * dM_dphi * phi_phi_j * stepien_grad_q;
                  // term 4: phi-part of M * delta(grad q)
                  const Tensor<1, dim> delta_grad_q_phi_j =
                    (density_derivative * phi_phi_j * potential_gradient +
                     drhodphi * (potential_value - dpr) * grad_phi_phi_j) /
                    (rho0 * rho1);
                  local_flow_ij +=
                    grad_phi_phi_i * mobility * delta_grad_q_phi_j;
              }
              else
              {
                  // Transient
                  local_flow_ij += phi_phi_i * bdf_c0 * phi_phi_j;
                  // Advection
                  local_flow_ij += phi_phi_i * u_conv * grad_phi_phi_j;
                  local_flow_ij +=
                    dM_dphi * phi_phi_j * (grad_phi_phi_i * potential_gradient);
              }
            }
            if (comp_j == const_ordering.mu_lower)
            {
              if (is_stepien)
              {
                  // term 4: mu-part of M * delta(grad q)
                  const Tensor<1, dim> delta_grad_q_mu_j =
                    (rho * grad_phi_mu_j +
                     drhodphi * phi_mu_j * tracer_gradient) /
                    (rho0 * rho1);
                  local_flow_ij +=
                    grad_phi_phi_i * mobility * delta_grad_q_mu_j;
              }
              else
                  // Diffusion
                  local_flow_ij += mobility * grad_phi_mu_j * grad_phi_phi_i;
            }
            if (is_stepien && comp_j == const_ordering.p_lower)
            {
              // term 4: pressure-part of M * delta(grad q). This is the new
              // phi-p coupling block enabled in the sparsity pattern.
              const Tensor<1, dim> &grad_phi_p_j =
                scratch_data.grad_phi_p[q][j];
              const Tensor<1, dim> delta_grad_q_p_j =
                -drhodphi * grad_phi_p_j / (rho0 * rho1);
              local_flow_ij += grad_phi_phi_i * mobility * delta_grad_q_p_j;
            }
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
                (potential_value - potential_double_well_coefficient *
                                     tracer_value *
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

  // See assemble_local_matrix(): the master VolumeAssembler takes over the
  // whole volume assembly when it is registered.
  if (!assemblers.empty())
  {
    for (const auto &assembler : assemblers)
      assembler->assemble_rhs(scratch_data, copy_data);
    cell->get_dof_indices(local_dof_indices);
    return;
  }

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
  const double enlarged_length =
    this->param.cahn_hilliard.epsilon_interface_enlarged -
    this->param.cahn_hilliard.epsilon_interface;
  const double enlarged_length_sq = enlarged_length * enlarged_length;

  // Stepien quasi-incompressible model: cell-constant coefficients.
  const bool   is_stepien  = CahnHilliard::is_stepien_model(cahn_hilliard);
  const double stepien_cap =
    CahnHilliard::stepien_capillary_coefficient(cahn_hilliard);
  const bool   conservative = cahn_hilliard.stepien_conservative_transport;
  const bool   skew_inertia = cahn_hilliard.stepien_skew_symmetric_inertia;
  const double rho0     = scratch_data.density0;
  const double rho1     = scratch_data.density1;
  const double drhodphi = 0.5 * (rho0 - rho1); // = (rho1 - rho2)/2
  const double dpr =
    0.5 * (this->param.physical_properties.fluids[0].pressure_ref -
           this->param.physical_properties.fluids[1].pressure_ref);
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
    const auto &potential_value    = scratch_data.potential_values[q];
    const auto &potential_gradient = scratch_data.potential_gradients[q];
    const auto &velocity_dot_tracer_gradient =
      scratch_data.velocity_dot_tracer_gradient[q];
    const double phi_cube_minus_phi =
      tracer_value * (tracer_value * tracer_value - 1.);
    const Tensor<1, dim> momentum_diffusive_inertia =
      use_abels_diffusive_inertia ? diffusive_flux : Tensor<1, dim>();
    Tensor<1, dim> momentum_capillary_force;
    if (use_abels_capillary_phi_grad_mu)
      momentum_capillary_force = tracer_value * potential_gradient;
    else if (is_stepien)
      momentum_capillary_force =
        -stepien_cap * potential_value * tracer_gradient +
        dpr * tracer_gradient;
    else
      momentum_capillary_force =
        -ding_horriche_capillary_coefficient * potential_value *
        tracer_gradient;

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratch_data.previous_velocity_values);
    const double dphidt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, tracer_value, scratch_data.previous_tracer_values);

    // Precomputations of shape functions-independent terms
    const auto to_multiply_by_phi_u_i =
      rho * (dudt + u_dot_grad_u - body_force) + momentum_diffusive_inertia +
      momentum_capillary_force + source_term_velocity;
    const auto to_multiply_by_phi_phi_i =
      dphidt + velocity_dot_tracer_gradient + source_term_tracer;
    const auto to_multiply_by_phi_mu_i =
      potential_value -
      potential_double_well_coefficient * phi_cube_minus_phi +
      source_term_potential;

    const double density_bdf = skew_inertia ?
                                 CahnHilliard::bdf_density_derivative(
                                   cahn_hilliard,
                                   rho0,
                                   rho1,
                                   rho,
                                   this->time_handler.bdf_coefficients,
                                   scratch_data.previous_tracer_values,
                                   q) :
                                 0.;
    const double flux_conversion =
      conservative ? (rho0 + rho1) / (2. * rho0 * rho1) : 0.;
    const double alpha = (rho1 - rho0) / (rho0 + rho1);

    // Stepien quasi-incompressible model: per-quadrature coefficients.
    // Phase eq.:  A * Dphi/Dt - div(M grad q) = 0, with
    //   A = rho1 rho2 / rho,
    //   q = ( rho * mu - (rho1 - rho2)/2 * (p* + p_r) ) / (rho1 rho2).
    // Momentum: second viscosity lambda = -2/3 eta of the Stokes hypothesis.
    double         stepien_A = 0., lambda_visc = 0.;
    Tensor<1, dim> stepien_grad_q;
    if (is_stepien)
    {
      stepien_A   = rho0 * rho1 / rho;
      lambda_visc = CahnHilliard::stokes_second_viscosity_ratio * eta;
      stepien_grad_q =
        (rho * potential_gradient -
         drhodphi * scratch_data.present_pressure_gradients[q] +
         drhodphi * (potential_value - dpr) * tracer_gradient) /
        (rho0 * rho1);
    }

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
      if (is_stepien)
        // Second-viscosity stress lambda (div v) I (div v != 0 for Stepien).
        local_rhs_flow_i += lambda_visc * present_velocity_divergence *
                            div_phi_u_i;

      if (skew_inertia)
        local_rhs_flow_i +=
          0.5 * density_bdf * (phi_u_i * present_velocity_values) -
          0.5 * rho *
            (phi_u_i * u_dot_grad_u +
             present_velocity_values *
               (scratch_data.grad_phi_u[q][i] * u_conv));

      // Continuity equation. J_A = (b/R) M_F grad(xi_F).
      local_rhs_flow_i +=
        phi_p_i * (-present_velocity_divergence + source_term_pressure);
      if (conservative)
        local_rhs_flow_i -= alpha * flux_conversion * mobility *
                            (scratch_data.grad_phi_p[q][i] * stepien_grad_q);
      else if (is_stepien)
        // Quasi-incompressible source: div v = -(rho1-rho2)/(2 rho) Dphi/Dt.
        local_rhs_flow_i +=
          -phi_p_i * (drhodphi / rho) *
          (dphidt + velocity_dot_tracer_gradient);

      // Tracer equation
      if (conservative)
        local_rhs_flow_i +=
          phi_phi_i * (dphidt + source_term_tracer) -
          tracer_value * (grad_phi_phi_i * u_conv) +
          flux_conversion * mobility * (grad_phi_phi_i * stepien_grad_q);
      else if (is_stepien)
        local_rhs_flow_i +=
          phi_phi_i * (stepien_A * (dphidt + velocity_dot_tracer_gradient) +
                       source_term_tracer) +
          grad_phi_phi_i * mobility * stepien_grad_q;
      else
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
        // Open boundary condition with prescribed manufactured solution
        if (this->param.fluid_bc.at(scratch_data.face_boundary_id[i_face])
              .type == BoundaryConditions::Type::open_mms)
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

  auto density_field =
    std::make_unique<PostProcessingTools::DG0DataField<dim>>(
      *this->triangulation,
      fe->reference_cell().is_hyper_cube(),
      std::vector<std::string>{"density"},
      std::vector<DataComponentInterpretation::DataComponentInterpretation>{
        DataComponentInterpretation::component_is_scalar});

  // Keep the existing positive density output and expose the affine density
  // separately: kinetic regularization must not hide physical undershoots.
  std::unique_ptr<PostProcessingTools::DG0DataField<dim>> affine_density_field;
  if (CahnHilliard::is_quasi_incompressible_model(this->param.cahn_hilliard))
    affine_density_field =
      std::make_unique<PostProcessingTools::DG0DataField<dim>>(
        *this->triangulation,
        fe->reference_cell().is_hyper_cube(),
        std::vector<std::string>{"density_affine"},
        std::vector<DataComponentInterpretation::DataComponentInterpretation>{
          DataComponentInterpretation::component_is_scalar});

  FEValues<dim> fe_values(*this->moving_mapping,
                          *fe,
                          *this->quadrature,
                          update_values | update_JxW_values);
  std::vector<double> tracer_values(this->quadrature->size());

  for (const auto &cell : this->dof_handler->active_cell_iterators())
    if (cell->is_locally_owned())
    {
      fe_values.reinit(cell);
      fe_values[tracer_extractor].get_function_values(*this->present_solution,
                                                       tracer_values);

      double density_average = 0.0;
      double affine_density_average = 0.0;
      double cell_measure    = 0.0;
      for (unsigned int q = 0; q < this->quadrature->size(); ++q)
      {
        const double JxW = fe_values.JxW(q);
        const double phi = tracer_limiter(tracer_values[q]);
        density_average +=
          CahnHilliard::linear_mixing(phi, density0, density1) * JxW;
        if (affine_density_field)
          affine_density_average +=
            CahnHilliard::linear_mixing(tracer_values[q], density0, density1) *
            JxW;
        cell_measure += JxW;
      }

      density_field->set_cell_values(
        cell, std::vector<double>{density_average / cell_measure});
      if (affine_density_field)
        affine_density_field->set_cell_values(
          cell, std::vector<double>{affine_density_average / cell_measure});
    }

  this->postproc_handler->add_cell_dg0_data_field(std::move(density_field));
  if (affine_density_field)
    this->postproc_handler->add_cell_dg0_data_field(
      std::move(affine_density_field));

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
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  write_stepien_diagnostics(
    const PostProcessingTools::CHNSIntegralData &data) const
{
  using I       = PostProcessingTools::CHNSIntegralData;
  const auto &v = data.sums;
  if (this->mpi_rank != 0)
    return;
  std::filesystem::create_directories(this->param.output.output_dir);
  const std::string filename =
    this->param.output.output_dir + "stepien_diagnostics.csv";
  const bool    first  = this->time_handler.current_time_iteration == 0;
  const bool    header = first || !std::ifstream(filename).good();
  std::ofstream out(filename, first ? std::ios::trunc : std::ios::app);
  AssertThrow(out.good(), ExcMessage("Cannot write " + filename));
  if (header)
    out << "time,iteration,phase_integral,mass_affine,mass_used,bubble_area,"
           "center_y,rise_velocity,weighted_bubble_area,weighted_center_y,"
           "weighted_rise_velocity,kinetic_energy,interfacial_energy,"
           "gravitational_energy,total_energy,viscous_dissipation,"
           "diffusive_dissipation,phi_min_q,phi_max_q\n";
  const double nan   = std::numeric_limits<double>::quiet_NaN();
  const auto average = [&](const I::Integral moment, const I::Integral volume) {
    return v[volume] > 0. ? v[moment] / v[volume] : nan;
  };
  out << std::setprecision(16) << this->time_handler.current_time << ','
      << this->time_handler.current_time_iteration;
  for (const auto k :
       {I::PhaseIntegral, I::AffineMass, I::PositiveMass, I::BubbleVolume})
    out << ',' << v[k];
  out << ',' << average(I::BubbleHeightMoment, I::BubbleVolume) << ','
      << average(I::BubbleVelocityMoment, I::BubbleVolume) << ','
      << v[I::WeightedBubbleVolume] << ','
      << average(I::WeightedBubbleHeightMoment, I::WeightedBubbleVolume) << ','
      << average(I::WeightedBubbleVelocityMoment, I::WeightedBubbleVolume);
  for (const auto k :
       {I::KineticEnergy, I::InterfacialEnergy, I::GravitationalEnergy})
    out << ',' << v[k];
  out << ',' << data.total_energy() << ',' << v[I::ViscousDissipation] << ','
      << v[I::DiffusiveDissipation] << ',' << data.phi_min << ','
      << data.phi_max << '\n';
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  write_quasi_incompressible_diagnostics(
    const PostProcessingTools::CHNSIntegralData &data) const
{
  using I           = PostProcessingTools::CHNSIntegralData;
  const auto  &pp   = this->param.physical_properties;
  const double rho0 = pp.fluids[0].density, rho1 = pp.fluids[1].density;
  const bool   initial = this->time_handler.current_time_iteration == 0;
  const double dt = initial ? 0. : this->time_handler.get_current_timestep();
  const std::vector<double> coefficients =
    initial ? std::vector<double>{0., 0.} :
              this->time_handler.get_bdf_coefficients();
  const bool bdf2          = coefficients.size() == 3 && coefficients[2] != 0.;
  const bool constant_bdf2 = bdf2 &&
                             std::abs(dt * coefficients[0] - 1.5) < 1.e-12 &&
                             std::abs(dt * coefficients[1] + 2.) < 1.e-12 &&
                             std::abs(dt * coefficients[2] - 0.5) < 1.e-12;
  const auto  &sums    = data.sums;
  const double phi_min = data.phi_min, phi_max = data.phi_max;
  const double density_at_min_phi =
    CahnHilliard::linear_mixing(phi_min, rho0, rho1);
  const double density_at_max_phi =
    CahnHilliard::linear_mixing(phi_max, rho0, rho1);
  if (this->mpi_rank != 0)
    return;
  std::filesystem::create_directories(this->param.output.output_dir);
  const auto filename =
    this->param.output.output_dir + "quasi_incompressible_diagnostics.csv";
  const bool    header = initial || !std::ifstream(filename).good();
  std::ofstream out(filename, initial ? std::ios::trunc : std::ios::app);
  AssertThrow(out.good(), ExcMessage("Cannot write " + filename));
  if (header)
    out
      << "time,iteration,dt,phase_integral,mass_affine,mass_positive,"
         "kinetic_energy,interfacial_energy,gravitational_energy,total_energy,"
         "viscous_dissipation,diffusive_dissipation,numerical_energy_loss,"
         "energy_inequality_residual,energy_balance_residual,phase_change,"
         "mass_change,bubble_area,center_y,rise_velocity,phi_min_q,phi_max_q,"
         "numerical_kinetic_energy_loss,numerical_interfacial_energy_loss,"
         "density_affine_min_q,density_affine_max_q,negative_density_volume_q,"
         "phase_overshoot_volume_q,density_extension_l1,velocity_l2,"
         "velocity_divergence_l2,diffusive_flux_l2,domain_volume,"
         "effective_bdf_order,bdf_c0,bdf_c1,bdf_c2,"
         "physical_energy_increment_plus_dissipation,"
         "phase_time_derivative_integral,mass_time_derivative_integral,"
         "kinetic_temporal_power,bulk_temporal_power,gradient_temporal_power,"
         "gravitational_temporal_power,temporal_energy_balance_residual,"
         "bdf2_constant_step,bdf2_modified_kinetic_energy,"
         "bdf2_modified_gradient_energy,bdf2_kinetic_remainder,"
         "bdf2_gradient_remainder,bdf2_kinetic_identity_residual,"
         "bdf2_gradient_identity_residual\n";
  const double energy = sums[I::KineticEnergy] + sums[I::InterfacialEnergy] +
                        sums[I::GravitationalEnergy];
  const double defect =
    energy - sums[I::OldEnergy] +
    dt * (sums[I::ViscousDissipation] + sums[I::DiffusiveDissipation]);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double temporal_balance =
    sums[I::KineticTemporalPower] + sums[I::BulkTemporalPower] +
    sums[I::GradientTemporalPower] + sums[I::GravitationalTemporalPower] +
    (initial ? 0. :
               sums[I::ViscousDissipation] + sums[I::DiffusiveDissipation]);
  out << std::setprecision(16) << this->time_handler.current_time << ','
      << this->time_handler.current_time_iteration << ',' << dt;
  for (unsigned int k = I::PhaseIntegral; k <= I::GravitationalEnergy; ++k)
    out << ',' << sums[k];
  out << ',' << energy << ',' << sums[I::ViscousDissipation] << ','
      << sums[I::DiffusiveDissipation] << ','
      << (bdf2 ? nan : sums[I::NumericalEnergyLoss]) << ','
      << (bdf2 ? nan : defect) << ','
      << (bdf2 ? nan : defect + sums[I::NumericalEnergyLoss]) << ','
      << sums[I::PhaseIntegral] - sums[I::OldPhaseIntegral] << ','
      << sums[I::AffineMass] - sums[I::OldAffineMass] << ','
      << sums[I::BubbleVolume] << ','
      << (sums[I::BubbleVolume] > 0. ?
            sums[I::BubbleHeightMoment] / sums[I::BubbleVolume] :
            nan)
      << ','
      << (sums[I::BubbleVolume] > 0. ?
            sums[I::BubbleVelocityMoment] / sums[I::BubbleVolume] :
            nan)
      << ',' << phi_min << ',' << phi_max << ','
      << (bdf2 ? nan : sums[I::NumericalKineticLoss]) << ','
      << (bdf2 ? nan : sums[I::NumericalInterfacialLoss]) << ','
      << std::min(density_at_min_phi, density_at_max_phi) << ','
      << std::max(density_at_min_phi, density_at_max_phi) << ','
      << sums[I::NegativeDensityVolume] << ',' << sums[I::PhaseOvershootVolume]
      << ',' << sums[I::DensityExtensionL1] << ','
      << std::sqrt(sums[I::VelocityNormSquared]) << ','
      << std::sqrt(sums[I::DivergenceNormSquared]) << ','
      << std::sqrt(sums[I::DiffusiveFluxNormSquared]) << ',' << sums[I::Volume]
      << ',' << (initial ? 0 : (bdf2 ? 2 : 1)) << ',' << coefficients[0] << ','
      << coefficients[1] << ','
      << (coefficients.size() == 3 ? coefficients[2] : 0.) << ',' << defect
      << ',' << sums[I::PhaseRate] << ',' << sums[I::MassRate] << ','
      << sums[I::KineticTemporalPower] << ',' << sums[I::BulkTemporalPower]
      << ',' << sums[I::GradientTemporalPower] << ','
      << sums[I::GravitationalTemporalPower] << ',' << temporal_balance << ','
      << (constant_bdf2 ? 1 : 0) << ','
      << (bdf2 ? sums[I::ModifiedKineticEnergy] : nan) << ','
      << (bdf2 ? sums[I::ModifiedGradientEnergy] : nan) << ','
      << (constant_bdf2 ? sums[I::BDF2KineticRemainder] : nan) << ','
      << (constant_bdf2 ? sums[I::BDF2GradientRemainder] : nan) << ','
      << (constant_bdf2 ? sums[I::BDF2KineticIdentityResidual] : nan) << ','
      << (constant_bdf2 ? sums[I::BDF2GradientIdentityResidual] : nan) << '\n';
}

template <int dim, bool with_moving_mesh, bool with_enlarged>
void CHNSSolver<dim, with_moving_mesh, with_enlarged>::
  solver_specific_post_processing()
{
  const bool restart_snapshot =
    this->param.checkpoint_restart.restart &&
    this->time_handler.current_time_iteration != 0 &&
    this->time_handler.current_time_iteration ==
      this->time_handler.time_iteration_at_last_restart;
  const bool stepien =
    CahnHilliard::is_stepien_model(this->param.cahn_hilliard);
  const bool detailed =
    stepien ? this->param.cahn_hilliard.write_stepien_diagnostics :
              this->param.cahn_hilliard.write_quasi_incompressible_diagnostics;
  const bool domain_monitoring =
    this->param.postprocessing.chns_conservation.enable ||
    this->param.postprocessing.chns_energy.enable ||
    this->param.postprocessing.chns_phase_energy.enable;
  if ((detailed || domain_monitoring) &&
      !(restart_snapshot && this->postproc_handler->has_chns_reference() &&
        !this->postproc_handler->needs_chns_phase_energy(this->time_handler)))
  {
    TimerOutput::Scope timer(this->computing_timer,
                             "Compute CHNS domain integrals");
    const auto data = stepien ? PostProcessingTools::compute_stepien_integrals(
                                  this->param.cahn_hilliard,
                                  this->param.physical_properties,
                                  *this->ordering,
                                  *this->dof_handler,
                                  *this->moving_mapping,
                                  *this->quadrature,
                                  *this->present_solution) :
                                PostProcessingTools::compute_chns_integrals(
                                  this->param.cahn_hilliard,
                                  this->param.physical_properties,
                                  *this->ordering,
                                  *this->dof_handler,
                                  *this->moving_mapping,
                                  *this->quadrature,
                                  *this->present_solution,
                                  *this->previous_solutions,
                                  this->time_handler);
    // A Stepien CSV alone has no standard table reference to checkpoint.
    // If monitoring is enabled only on restart, start its reference there.
    if (!stepien || domain_monitoring)
      this->postproc_handler->process_chns_integrals(data, this->time_handler);
    if (detailed && !restart_snapshot)
    {
      if (stepien)
        write_stepien_diagnostics(data);
      else
        write_quasi_incompressible_diagnostics(data);
    }
  }

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
    const auto                                      pressure_values =
      VectorTools::point_values<1>(*this->moving_mapping,
                                   *this->dof_handler,
                                   *this->present_solution,
                                   probe_points,
                                   cache,
                                   VectorTools::EvaluationFlags::avg,
                                   this->ordering->p_lower);
    AssertThrow(cache.all_points_found(),
                ExcMessage("At least one line probe point was not found."));

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
    // Tracer gradients are needed to reconstruct the thermodynamic pressure
    // (free energy density contribution, see 'pressure_physical' below).
    const auto tracer_gradients =
      VectorTools::point_gradients<1>(cache,
                                      *this->dof_handler,
                                      *this->present_solution,
                                      VectorTools::EvaluationFlags::avg,
                                      this->ordering->phi_lower);

    std::vector<double> psi_values;
    if constexpr (with_enlarged)
      psi_values =
        VectorTools::point_values<1>(cache,
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
        //  - Abels-type model: the capillary force is -phi*grad(mu), so the
        //    solved pressure absorbs the gradient term grad(phi*mu). When
        //    the chemical potential has relaxed towards a spatial constant
        //    (large mobility), the solved pressure is uniform and the
        //    Young-Laplace jump lives entirely in phi*mu.
        //  - Ding-Horriche model: the solved pressure is the modified
        //    pressure phat = p + Psi, with Psi the free energy density.
        // 'pressure_yl' is the modified/bulk pressure without the localized
        // free-energy density well. This is the pressure to use for
        // plateau-to-plateau Young-Laplace post-processing:
        //  - Abels-type model:  pressure_yl = p + phi*mu
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
      // carries the same prefactor. For Abels and Stepien the coefficients
      // already include sigma_tilde (Stepien capillary force is muhat *
      // grad(phi) with prefactor 1).
      const double free_energy_prefactor =
        use_abels_capillary_phi_grad_mu ||
            CahnHilliard::is_stepien_model(cahn_hilliard) ?
          1.0 :
          CahnHilliard::ding_horriche_capillary_coefficient(cahn_hilliard);

      for (unsigned int i = 0; i < line_probe.n_points; ++i)
      {
        const double pressure_raw  = pressure_values[i];
        const double tracer        = tracer_values[i];
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
          // Capillary force -phi*grad(mu): the solved pressure absorbs
          // grad(phi*mu) (see header comment).
          pressure_solved   = pressure_raw;
          potential_solved  = potential_raw;
          pressure_yl       = pressure_raw + tracer * potential_raw;
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
  if (should_output_mesh_quality())
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
