
#include <assembly/elasticity_assemblers.h>
#include <assembly/incompressible_chns_assemblers.h>
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
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <string>

namespace
{
  /**
   * Check that the parameters are within the scope of the energy-stable
   * quasi-incompressible scheme: fixed mesh, BDF1/BDF2, Taylor-Hood
   * velocity-pressure pair with equal phase/potential degrees, no
   * stabilization, closed (no-slip or slip) or periodic boundaries.
   */
  template <int dim>
  void check_quasi_incompressible_parameters(const ParameterReader<dim> &param,
                                             const bool with_moving_mesh)
  {
    using Scheme     = Parameters::TimeIntegration::Scheme;
    const auto &time = param.time_integration;
    const auto &fe   = param.finite_elements;
    const auto &ch   = param.cahn_hilliard;

    AssertThrow(!with_moving_mesh,
                ExcMessage(
                  "Quasi-incompressible CHNS is implemented on a fixed mesh."));
    AssertThrow(time.scheme == Scheme::BDF1 || time.scheme == Scheme::BDF2,
                ExcMessage(
                  "Quasi-incompressible CHNS requires scheme = BDF1 or BDF2."));
    AssertThrow(time.scheme != Scheme::BDF2 ||
                  time.bdfstart == Parameters::TimeIntegration::BDFStart::BDF1,
                ExcMessage(
                  "Quasi-incompressible BDF2 requires bdf start method = BDF1 "
                  "to construct a consistent second history state."));
    AssertThrow(time.scheme != Scheme::BDF2 ||
                  (time.bdf_starting_step_ratio > 0. &&
                   time.bdf_starting_step_ratio < 1.),
                ExcMessage("Quasi-incompressible BDF2 requires "
                           "0 < bdf start step ratio < 1."));
    AssertThrow(!param.stabilization.enable_supg &&
                  !param.stabilization.enable_tracer_supg,
                ExcMessage("Disable SUPG/PSPG stabilization for the "
                           "quasi-incompressible CHNS Galerkin scheme."));
    AssertThrow(fe.tracer_degree >= 1 &&
                  fe.tracer_degree == fe.potential_degree &&
                  fe.pressure_degree >= 1 &&
                  fe.velocity_degree == fe.pressure_degree + 1,
                ExcMessage("Quasi-incompressible CHNS requires equal "
                           "phase/potential degrees >= 1 and a Taylor-Hood "
                           "velocity/pressure pair (e.g. 2/1)."));
    AssertThrow(ch.epsilon_interface > 0. && ch.surface_tension > 0.,
                ExcMessage("Interface thickness and surface tension must be "
                           "strictly positive."));
    AssertThrow(param.physical_properties.fluids.size() == 2,
                ExcMessage(
                  "Quasi-incompressible CHNS requires exactly two fluids."));
    for (const auto &fluid : param.physical_properties.fluids)
      AssertThrow(fluid.density > 0. && fluid.kinematic_viscosity >= 0.,
                  ExcMessage("Quasi-incompressible CHNS requires positive "
                             "densities and non-negative viscosities."));
    AssertThrow(!param.with_tree_based_adaptation() &&
                  !param.with_metric_based_adaptation(),
                ExcMessage("Quasi-incompressible CHNS requires a fixed mesh; "
                           "disable mesh adaptation."));
    AssertThrow(param.bc_data.fix_pressure_constant !=
                  param.bc_data.enforce_zero_mean_pressure,
                ExcMessage("Quasi-incompressible CHNS requires exactly one "
                           "pressure gauge: fix pressure constant or enforce "
                           "zero mean pressure."));

    // No open-boundary momentum or phase flux terms are assembled.
    // Manufactured traces remain available; their energy includes the work of
    // the boundary data.
    for (const auto &[id, bc] : param.fluid_bc)
      AssertThrow(bc.type == BoundaryConditions::Type::no_slip ||
                    bc.type == BoundaryConditions::Type::slip ||
                    (param.mms_param.enable &&
                     bc.type == BoundaryConditions::Type::velocity_mms),
                  ExcMessage("Quasi-incompressible CHNS supports no_slip/slip "
                             "fluid boundaries and manufactured velocity "
                             "traces; unsupported boundary id " +
                             std::to_string(id)));
    for (const auto &[id, bc] : param.cahn_hilliard_bc)
      AssertThrow(bc.type == BoundaryConditions::Type::no_flux ||
                    (param.mms_param.enable &&
                     bc.type == BoundaryConditions::Type::dirichlet_mms),
                  ExcMessage("Quasi-incompressible CHNS supports natural "
                             "no_flux Cahn-Hilliard boundaries and "
                             "manufactured traces; unsupported boundary id " +
                             std::to_string(id)));

    if (ch.quasi_incompressible_periodic_box)
    {
      AssertThrow(param.physical_properties.body_force.norm_square() == 0.,
                  ExcMessage("The quasi-incompressible periodic box requires "
                             "zero body force: a nonzero constant gravity "
                             "potential is not periodic."));
      AssertThrow(fe.use_quads && param.mesh.deal_ii_preset_mesh == "rectangle",
                  ExcMessage("The periodic box requires use quads = true and "
                             "dealii preset mesh = rectangle."));
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
  }
} // namespace

template <int dim, bool with_moving_mesh>
CHNSSolver<dim, with_moving_mesh>::CHNSSolver(const ParameterReader<dim> &param)
  : NavierStokesSolver<dim, with_moving_mesh>(param)
{
  const bool quasi_incompressible =
    CahnHilliard::is_quasi_incompressible_model(param.cahn_hilliard);
  if (quasi_incompressible)
    check_quasi_incompressible_parameters(param, with_moving_mesh);
  AssertThrow(quasi_incompressible ||
                !(param.postprocessing.chns_conservation.enable ||
                  param.postprocessing.chns_energy.enable ||
                  param.postprocessing.chns_phase_energy.enable),
              ExcMessage("The chns conservation, chns energy and chns phase "
                         "energy postprocessings require CHNS model = "
                         "quasi_incompressible."));

  if constexpr (with_moving_mesh)
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
        this->time_handler.current_time, *this->ordering, this->param);

    // Create entry in error handler for tracer and potential
    for (auto &[norm, handler] : this->error_handlers)
    {
      handler.create_entry("phi");
      handler.create_entry("mu");
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
  if (CahnHilliard::is_quasi_incompressible_model(cahn_hilliard_param))
  {
    quasi_incompressible_vector_value(p, values);
    return;
  }

  const double phi = mms.exact_tracer->value(p);
  // Filter the tracer as in the scratch data, if applicable
  const double filtered_phi = tracer_limiter(phi);
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
      mms.exact_mesh_position->divergence_elastic_stress_tensor(
        physical_properties.pseudosolids[0], p);

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
void CHNSSolver<dim, with_moving_mesh>::MMSSourceTerm::
  quasi_incompressible_vector_value(const Point<dim> &p,
                                    Vector<double>   &values) const
{
  /**
   * Source terms for the continuous strong form of the quasi-incompressible
   * model, independently of the time discretization:
   *
   *   rho~ dv/dt + 1/2 (drho~/dt) v + rho (v.grad) v + 1/2 div(rho v) v
   *     - div(eta~ (grad v + grad v^T - 2/d div v I)) + grad p + phi grad mu
   *     - rho g = f_v,
   *   div v - alpha div J = f_p,
   *   dphi/dt + div(phi v) - div J = f_phi,
   *   mu - sigma~/eps (phi^3 - phi) + sigma~ eps lap(phi) = f_mu,
   *
   * with J = M(phi) (grad mu + alpha grad p). The source terms are stored with
   * the sign convention of the residual in EnergyStable::residual().
   */
  const double rho0  = physical_properties.fluids[0].density;
  const double rho1  = physical_properties.fluids[1].density;
  const double eta0  = rho0 * physical_properties.fluids[0].kinematic_viscosity;
  const double eta1  = rho1 * physical_properties.fluids[1].kinematic_viscosity;
  const double alpha = (rho1 - rho0) / (rho0 + rho1);
  const double sigma_tilde =
    CahnHilliard::sigma_tilde_from_surface_tension(cahn_hilliard_param);
  const double epsilon = cahn_hilliard_param.epsilon_interface;

  const double phi     = mms.exact_tracer->value(p);
  const double dphidt  = mms.exact_tracer->time_derivative(p);
  const double lap_phi = mms.exact_tracer->laplacian(p);
  const double mu      = mms.exact_potential->value(p);
  const double lap_mu  = mms.exact_potential->laplacian(p);
  const double lap_p   = mms.exact_pressure->laplacian(p);
  const double div_u   = mms.exact_velocity->divergence(p);

  // Affine density, and positive extensions of density and viscosity.
  const double clipped = CahnHilliard::tracer_limiter(phi);
  const double clip_derivative =
    CahnHilliard::tracer_limiter_derivative(phi, true);
  const double rho_affine = CahnHilliard::linear_mixing(phi, rho0, rho1);
  const double rho        = CahnHilliard::linear_mixing(clipped, rho0, rho1);
  const double drho       = 0.5 * (rho0 - rho1);
  const double eta        = CahnHilliard::linear_mixing(clipped, eta0, eta1);
  const double detadphi   = 0.5 * (eta0 - eta1) * clip_derivative;
  const auto   mobility =
    CahnHilliard::evaluate_mobility(cahn_hilliard_param, phi);

  Tensor<1, dim> u, dudt_eulerian;
  for (unsigned int d = 0; d < dim; ++d)
  {
    dudt_eulerian[d] = mms.exact_velocity->time_derivative(p, d);
    u[d]             = mms.exact_velocity->value(p, d);
  }

  // Use convention (grad_u)_ij := dvj/dxi
  const Tensor<2, dim> grad_u     = mms.exact_velocity->gradient_vj_xi(p);
  const Tensor<1, dim> lap_u      = mms.exact_velocity->vector_laplacian(p);
  const Tensor<1, dim> grad_div_u = mms.exact_velocity->grad_div(p);
  const Tensor<1, dim> grad_p     = mms.exact_pressure->gradient(p);
  const Tensor<1, dim> grad_mu    = mms.exact_potential->gradient(p);
  const Tensor<1, dim> grad_phi   = mms.exact_tracer->gradient(p);
  const Tensor<1, dim> uDotGradu  = u * grad_u;

  // Diffusive flux divergence div(J)
  const Tensor<1, dim> grad_w   = grad_mu + alpha * grad_p;
  const double         div_flux = mobility.value * (lap_mu + alpha * lap_p) +
                          mobility.first_derivative * (grad_phi * grad_w);

  // div(eta~ (grad u + grad u^T) + lambda eta~ div(u) I), lambda = -2/d
  const double         lambda = -2. / dim;
  const Tensor<1, dim> div_stress =
    eta * (lap_u + (1. + lambda) * grad_div_u) +
    2. * detadphi * (grad_phi * symmetrize(grad_u)) +
    lambda * detadphi * div_u * grad_phi;
  const double div_mass_flux = drho * (u * grad_phi) + rho_affine * div_u;

  const Tensor<1, dim> momentum =
    rho * dudt_eulerian + rho_affine * uDotGradu +
    0.5 * u * (drho * clip_derivative * dphidt + div_mass_flux) - div_stress +
    grad_p + phi * grad_mu - rho_affine * physical_properties.body_force;

  for (unsigned int d = 0; d < dim; ++d)
    values[u_lower + d] = -momentum[d];
  values[p_lower]   = div_u - alpha * div_flux;
  values[phi_lower] = -(dphidt + u * grad_phi + phi * div_u - div_flux);
  values[mu_lower]  = -(mu + sigma_tilde * epsilon * lap_phi -
                       sigma_tilde / epsilon * phi * (phi * phi - 1.));
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::update_simulation_parameters(
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

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::create_scratch_data()
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

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::setup_assemblers()
{
  assemblers.clear();

  // CHNS assemblers
  Assembly::IncompressibleCHNS::
    setup_assemblers<dim, ScratchData, CopyData, with_moving_mesh>(
      this->param, *this->ordering, this->coupling_table, assemblers);

  // Elasticity
  if constexpr (with_moving_mesh)
    Assembly::Elasticity::setup_assemblers<dim, ScratchData, CopyData>(
      this->param, *this->ordering, assemblers);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim,
                with_moving_mesh>::create_solver_specific_zero_constraints()
{
  const auto &ch = this->param.cahn_hilliard;
  if (CahnHilliard::is_quasi_incompressible_model(ch) &&
      !ch.quasi_incompressible_periodic_box)
  {
    // An omitted velocity boundary would otherwise become an open natural
    // boundary, for which no momentum or phase flux terms are assembled.
    unsigned int closed = 1;
    for (const auto &cell : this->dof_handler->active_cell_iterators())
      if (cell->is_locally_owned())
        for (const auto face : cell->face_indices())
          if (cell->face(face)->at_boundary() &&
              this->param.fluid_bc.count(cell->face(face)->boundary_id()) == 0)
            closed = 0;
    AssertThrow(Utilities::MPI::min(closed, this->mpi_communicator) == 1,
                ExcMessage("Every non-periodic quasi-incompressible CHNS mesh "
                           "boundary needs an explicit fluid boundary "
                           "condition; open boundaries are not implemented."));
  }
  if (ch.quasi_incompressible_periodic_box)
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
    }
  }
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim,
                with_moving_mesh>::create_solver_specific_nonzero_constraints()
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
                           *this->dof_handler,
                           *tracer_fun,
                           this->newton_update,
                           tracer_mask);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::set_solver_specific_exact_solution()
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

      if (CahnHilliard::is_quasi_incompressible_model(
            this->param.cahn_hilliard))
      {
        // The momentum, continuity and phase rows couple to all four
        // unknowns (skew convection, phi grad(mu), and the diffusive flux
        // M(phi)(grad mu + alpha grad p)). The potential row only couples to
        // phi and mu.
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

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  CopyData copy_data(*fe);

  this->system_matrix = 0;
  this->do_matrix_assembly(*this,
                           this->dof_handler->begin_active(),
                           this->dof_handler->end(),
                           *scratch_data,
                           copy_data,
                           this->param.nonlinear_solver.analytic_jacobian);
  this->system_matrix.compress(VectorOperation::add);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::assemble_local_matrix(
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

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::copy_local_to_global_matrix(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_matrix(),
                                                    copy_data.dof_indices(),
                                                    this->system_matrix);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::compare_analytical_matrix_with_fd()
{
  CopyData copy_data(*fe);
  Verification::compare_analytical_matrix_with_fd<dim>(
    *this,
    *scratch_data,
    copy_data,
    this->param.nonlinear_solver.write_problematic_elements);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::assemble_rhs()
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

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::assemble_local_rhs(
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

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::copy_local_to_global_rhs(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_rhs(),
                                                    copy_data.dof_indices(),
                                                    this->system_rhs);
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::compute_solver_specific_errors()
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
}

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::write_quasi_incompressible_diagnostics(
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
  const double energy = data.total_energy();
  // E^{n+1} - E^n + dt (viscous + diffusive dissipation). For BDF1, this is
  // minus the numerical dissipation, see equations.md.
  const double defect =
    energy - sums[I::OldEnergy] +
    dt * (sums[I::ViscousDissipation] + sums[I::DiffusiveDissipation]);
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const double temporal_balance =
    sums[I::KineticTemporalPower] + sums[I::BulkTemporalPower] +
    sums[I::GradientTemporalPower] + sums[I::GravitationalTemporalPower] +
    (initial ? 0. :
               sums[I::ViscousDissipation] + sums[I::DiffusiveDissipation]);
  const auto bubble_average = [&](const I::Integral moment) {
    return sums[I::BubbleVolume] > 0. ? sums[moment] / sums[I::BubbleVolume] :
                                        nan;
  };
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
      << sums[I::BubbleVolume] << ',' << bubble_average(I::BubbleHeightMoment)
      << ',' << bubble_average(I::BubbleVelocityMoment) << ',' << phi_min << ','
      << phi_max << ',' << (bdf2 ? nan : sums[I::NumericalKineticLoss]) << ','
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

template <int dim, bool with_moving_mesh>
void CHNSSolver<dim, with_moving_mesh>::solver_specific_post_processing()
{
  const auto &ch                = this->param.cahn_hilliard;
  const auto &pp_param          = this->param.postprocessing;
  const bool  domain_monitoring = pp_param.chns_conservation.enable ||
                                 pp_param.chns_energy.enable ||
                                 pp_param.chns_phase_energy.enable;
  const bool detailed = ch.write_quasi_incompressible_diagnostics;

  // After a restart, the restored state was already monitored before the
  // checkpoint: skip it unless a phase-energy sample is still missing.
  const bool restart_snapshot =
    this->param.checkpoint_restart.restart &&
    this->time_handler.current_time_iteration != 0 &&
    this->time_handler.current_time_iteration ==
      this->time_handler.time_iteration_at_last_restart;

  if (CahnHilliard::is_quasi_incompressible_model(ch) &&
      (detailed || domain_monitoring) &&
      !(restart_snapshot && this->postproc_handler->has_chns_reference() &&
        !this->postproc_handler->needs_chns_phase_energy(this->time_handler)))
  {
    TimerOutput::Scope timer(this->computing_timer,
                             "Compute CHNS domain integrals");
    const auto         data = PostProcessingTools::compute_chns_integrals(
      ch,
      this->param.physical_properties,
      *this->ordering,
      *this->dof_handler,
      *this->moving_mapping,
      *this->quadrature,
      *this->present_solution,
      *this->previous_solutions,
      this->time_handler);
    this->postproc_handler->process_chns_integrals(data, this->time_handler);
    if (detailed && !restart_snapshot)
      write_quasi_incompressible_diagnostics(data);
  }

  TimerOutput::Scope t(this->computing_timer, "Compute multiphase indicators");

  this->postproc_handler->compute_multiphase_indicators(*this->ordering,
                                                        *this->dof_handler,
                                                        *this->moving_mapping,
                                                        *this->quadrature,
                                                        *this->present_solution,
                                                        this->time_handler);
}

// Explicit instantiation
template class CHNSSolver<2, false>;
template class CHNSSolver<3, false>;
template class CHNSSolver<2, true>;
template class CHNSSolver<3, true>;
