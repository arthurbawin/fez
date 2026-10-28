
#include <assembly/incompressible_chns_assemblers.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/grid/tria.h>
#include <post_processing_tools.h>
#include <types.h>

namespace PostProcessingTools
{
  namespace
  {
    void add_phase_energy(CHNSIntegralData           &data,
                          const double                phase,
                          const double                weight,
                          const CHNSEnergyComponents &energy)
    {
      auto &values = data.phase_energy[phase < 0. ? 1 : 0].values;
      for (unsigned int k = 0; k < values.size(); ++k)
        values[k] += weight * energy.values[k];
    }

    void sum_chns_integrals(CHNSIntegralData &data, const MPI_Comm communicator)
    {
      // Pack both phases with the domain integrals for one collective sum.
      std::array<double,
                 CHNSIntegralData::IntegralCount +
                   2 * CHNSEnergyComponents::ComponentCount>
                   integrals{};
      unsigned int index = 0;
      for (const double value : data.sums)
        integrals[index++] = value;
      for (const auto &phase : data.phase_energy)
        for (const double value : phase.values)
          integrals[index++] = value;
      Utilities::MPI::sum<double>(
        ArrayView<const double>(integrals.data(), integrals.size()),
        communicator,
        ArrayView<double>(integrals.data(), integrals.size()));
      index = 0;
      for (auto &value : data.sums)
        value = integrals[index++];
      for (auto &phase : data.phase_energy)
        for (auto &value : phase.values)
          value = integrals[index++];
    }
  } // namespace

  template <int dim>
  DataOutFacesOnBoundary<dim>::DataOutFacesOnBoundary(
    const Triangulation<dim> &triangulation,
    const types::boundary_id  boundary_id)
    : DataOutFaces<dim>(/*surface_only =*/true)
    , triangulation(triangulation)
    , boundary_id(boundary_id)
  {}

  template <int dim>
  typename DataOutFacesOnBoundary<dim>::FaceDescriptor
  DataOutFacesOnBoundary<dim>::first_face()
  {
    // Find first active cell with a face on the prescribed boundary
    for (const auto &cell : this->triangulation.active_cell_iterators())
      if (cell->is_locally_owned())
        for (const unsigned int f : cell->face_indices())
        {
          const auto &face = cell->face(f);
          if (face->at_boundary() && face->boundary_id() == this->boundary_id)
            return FaceDescriptor(cell, f);
        }

    // just return an invalid descriptor if we haven't found a locally
    // owned face. this can happen in parallel where all boundary
    // faces are owned by other processors
    return FaceDescriptor();
  }

  template <int dim>
  typename DataOutFacesOnBoundary<dim>::FaceDescriptor
  DataOutFacesOnBoundary<dim>::next_face(const FaceDescriptor &old_face)
  {
    FaceDescriptor face = old_face;

    // first check whether the present cell has more faces on the boundary.
    // since we started with this face, its cell must clearly be locally owned
    Assert(face.first->is_locally_owned(), ExcInternalError());
    for (unsigned int f = face.second + 1; f < face.first->n_faces(); ++f)
    {
      const auto &face_iter = face.first->face(f);
      if (face_iter->at_boundary() &&
          face_iter->boundary_id() == this->boundary_id)
      {
        // Return this face
        face.second = f;
        return face;
      }
    }

    // otherwise find the next active cell that has a face on the boundary
    typename Triangulation<dim>::active_cell_iterator active_cell = face.first;
    ++active_cell;

    while (active_cell != this->triangulation.end())
    {
      if (active_cell->is_locally_owned())
        for (const unsigned int f : face.first->face_indices())
          if (active_cell->face(f)->at_boundary() &&
              active_cell->face(f)->boundary_id() == this->boundary_id)
          {
            face.first  = active_cell;
            face.second = f;
            return face;
          }
      ++active_cell;
    }

    // If no face was found, return with invalid pointer
    face.first  = this->triangulation.end();
    face.second = 0;
    return face;
  }

  template class DataOutFacesOnBoundary<2>;
  template class DataOutFacesOnBoundary<3>;

  template <int dim, typename VectorType>
  CHNSIntegralData compute_chns_integrals(
    const Parameters::CahnHilliard<dim>       &ch,
    const Parameters::PhysicalProperties<dim> &physical_properties,
    const ComponentOrdering                   &ordering,
    const DoFHandler<dim>                     &dof_handler,
    const Mapping<dim>                        &mapping,
    const Quadrature<dim>                     &quadrature,
    const VectorType                          &solution,
    const std::vector<VectorType>             &previous_solutions,
    const TimeHandler                         &time_handler)
  {
    const FEValuesExtractors::Scalar tracer_extractor(ordering.phi_lower);
    const FEValuesExtractors::Scalar potential_extractor(ordering.mu_lower);
    const FEValuesExtractors::Vector velocity_extractor(ordering.u_lower);
    const auto                      &pp = physical_properties;
    const double rho0 = pp.fluids[0].density, rho1 = pp.fluids[1].density;
    const double eta0  = rho0 * pp.fluids[0].kinematic_viscosity;
    const double eta1  = rho1 * pp.fluids[1].kinematic_viscosity;
    const double sigma = CahnHilliard::sigma_tilde_from_surface_tension(ch);
    const double gamma = sigma * ch.epsilon_interface;
    const double beta_inverse = sigma / ch.epsilon_interface;
    const double alpha        = (rho1 - rho0) / (rho0 + rho1);
    const bool   initial      = time_handler.current_time_iteration == 0 ||
                         time_handler.current_time_iteration ==
                           time_handler.time_iteration_at_last_restart;
    const double dt = initial ? 0. : time_handler.get_current_timestep();
    const std::vector<double> coefficients =
      initial ? std::vector<double>{0., 0.} :
                time_handler.get_bdf_coefficients();
    const bool         bdf2 = coefficients.size() == 3 && coefficients[2] != 0.;
    FEValues<dim>      fv(mapping,
                     dof_handler.get_fe(),
                     quadrature,
                     update_values | update_gradients |
                       update_quadrature_points | update_JxW_values);
    const unsigned int nq = quadrature.size();
    std::vector<double>         phi(nq), old_phi(nq), older_phi(nq);
    std::vector<Tensor<1, dim>> v(nq), old_v(nq), older_v(nq), grad_phi(nq),
      old_grad_phi(nq), older_grad_phi(nq), grad_mu(nq), grad_p(nq);
    std::vector<Tensor<2, dim>>      grad_v(nq);
    const FEValuesExtractors::Scalar pressure(ordering.p_lower);
    CHNSIntegralData                 result;
    result.has_temporal_energy_diagnostics = true;
    auto  &sums                            = result.sums;
    double phi_min                         = std::numeric_limits<double>::max();
    double phi_max = std::numeric_limits<double>::lowest();
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        fv.reinit(cell);
        fv[tracer_extractor].get_function_values(solution, phi);
        fv[tracer_extractor].get_function_gradients(solution, grad_phi);
        fv[velocity_extractor].get_function_values(solution, v);
        fv[velocity_extractor].get_function_gradients(solution, grad_v);
        fv[potential_extractor].get_function_gradients(solution, grad_mu);
        fv[pressure].get_function_gradients(solution, grad_p);
        const auto &old = initial ? solution : previous_solutions.at(0);
        fv[tracer_extractor].get_function_values(old, old_phi);
        fv[tracer_extractor].get_function_gradients(old, old_grad_phi);
        fv[velocity_extractor].get_function_values(old, old_v);
        const auto &older = bdf2 ? previous_solutions.at(1) : old;
        fv[tracer_extractor].get_function_values(older, older_phi);
        fv[tracer_extractor].get_function_gradients(older, older_grad_phi);
        fv[velocity_extractor].get_function_values(older, older_v);
        for (unsigned int q = 0; q < nq; ++q)
        {
          const auto d =
            Assembly::IncompressibleCHNS::EnergyStable::make_point_data(
              ch,
              rho0,
              rho1,
              eta0,
              eta1,
              coefficients,
              phi[q],
              old_phi[q],
              old_v[q],
              pp.body_force,
              older_phi[q],
              older_v[q]);
          const double potential_gravity =
            -(pp.body_force * Tensor<1, dim>(fv.quadrature_point(q)));
          const double old_density =
            CahnHilliard::linear_mixing(old_phi[q], rho0, rho1);
          const auto   strain    = symmetrize(grad_v[q]);
          const auto   deviator  = deviator_tensor<dim>() * strain;
          const double indicator = phi[q] < 0. ? 1. : 0.;
          const double kinetic = 0.5 * d.positive_density * v[q].norm_square();
          const double bulk =
            0.25 * beta_inverse * std::pow(phi[q] * phi[q] - 1., 2);
          const double gradient = 0.5 * gamma * grad_phi[q].norm_square();
          const double interface_energy = bulk + gradient;
          const CHNSEnergyComponents energy{
            {{kinetic, bulk, gradient, d.density * potential_gravity}}};
          add_phase_energy(result, phi[q], fv.JxW(q), energy);
          const double old_energy =
            0.5 * d.old_positive_density * old_v[q].norm_square() +
            0.25 * beta_inverse * std::pow(old_phi[q] * old_phi[q] - 1., 2) +
            0.5 * gamma * old_grad_phi[q].norm_square() +
            old_density * potential_gravity;
          const double numerical_kinetic =
            0.5 * d.old_positive_density * (v[q] - old_v[q]).norm_square();
          const double numerical_interfacial =
            0.5 * gamma * (grad_phi[q] - old_grad_phi[q]).norm_square();
          const auto   grad_w     = grad_mu[q] + alpha * grad_p[q];
          const double phase_rate = d.c0 * phi[q] + d.phase_history;
          const auto   gradient_rate =
            d.c0 * grad_phi[q] + coefficients[1] * old_grad_phi[q] +
            (bdf2 ? coefficients[2] * older_grad_phi[q] : Tensor<1, dim>());
          const double mass_rate = d.density_derivative * phase_rate;
          const double kinetic_power =
            Assembly::IncompressibleCHNS::EnergyStable::inertia(v[q], d) * v[q];
          const double gradient_power = gamma * (grad_phi[q] * gradient_rate);
          const auto   weighted_v     = std::sqrt(d.positive_density) * v[q];
          const auto   old_weighted_v =
            std::sqrt(d.old_positive_density) * old_v[q];
          const double older_positive_density = CahnHilliard::linear_mixing(
            CahnHilliard::tracer_limiter(older_phi[q]), rho0, rho1);
          const auto older_weighted_v =
            std::sqrt(older_positive_density) * older_v[q];
          // These two-level energies satisfy the classical BDF2 identities
          // only when the current and previous time steps are equal.
          const double modified_kinetic =
            0.25 * (weighted_v.norm_square() +
                    (2. * weighted_v - old_weighted_v).norm_square());
          const double old_modified_kinetic =
            0.25 * (old_weighted_v.norm_square() +
                    (2. * old_weighted_v - older_weighted_v).norm_square());
          const double modified_gradient =
            0.25 * gamma *
            (grad_phi[q].norm_square() +
             (2. * grad_phi[q] - old_grad_phi[q]).norm_square());
          const double old_modified_gradient =
            0.25 * gamma *
            (old_grad_phi[q].norm_square() +
             (2. * old_grad_phi[q] - older_grad_phi[q]).norm_square());
          const double bdf2_kinetic_remainder =
            0.25 *
            (weighted_v - 2. * old_weighted_v + older_weighted_v).norm_square();
          const double bdf2_gradient_remainder =
            0.25 * gamma *
            (grad_phi[q] - 2. * old_grad_phi[q] + older_grad_phi[q])
              .norm_square();
          const std::array<double, CHNSIntegralData::IntegralCount> values = {
            {phi[q],
             d.density,
             d.positive_density,
             kinetic,
             interface_energy,
             d.density * potential_gravity,
             2. * d.viscosity * scalar_product(deviator, deviator),
             d.mobility * grad_w.norm_square(),
             numerical_kinetic + numerical_interfacial,
             old_energy,
             old_phi[q],
             old_density,
             indicator,
             indicator * fv.quadrature_point(q)[1],
             indicator * v[q][1],
             1.,
             numerical_kinetic,
             numerical_interfacial,
             d.density < 0. ? 1. : 0.,
             phi[q] < -1. || phi[q] > 1. ? 1. : 0.,
             std::abs(d.positive_density - d.density),
             v[q].norm_square(),
             std::pow(trace(grad_v[q]), 2),
             d.mobility * d.mobility * grad_w.norm_square(),
             phase_rate,
             mass_rate,
             kinetic_power,
             d.well_coefficient *
               Assembly::IncompressibleCHNS::EnergyStable::potential_derivative(
                 phi[q], d) *
               phase_rate,
             gradient_power,
             mass_rate * potential_gravity,
             modified_kinetic,
             modified_gradient,
             bdf2_kinetic_remainder,
             bdf2_gradient_remainder,
             dt * kinetic_power - modified_kinetic + old_modified_kinetic -
               bdf2_kinetic_remainder,
             dt * gradient_power - modified_gradient + old_modified_gradient -
               bdf2_gradient_remainder,
             bulk,
             gradient,
             0.,
             0.,
             0.}};
          for (unsigned int k = 0; k < sums.size(); ++k)
            sums[k] += fv.JxW(q) * values[k];
          phi_min = std::min(phi_min, phi[q]);
          phi_max = std::max(phi_max, phi[q]);
        }
      }
    sum_chns_integrals(result, dof_handler.get_mpi_communicator());
    phi_min = Utilities::MPI::min(phi_min, dof_handler.get_mpi_communicator());
    phi_max = Utilities::MPI::max(phi_max, dof_handler.get_mpi_communicator());
    result.phi_min = phi_min;
    result.phi_max = phi_max;
    return result;
  }

  template <int dim, typename VectorType>
  CHNSIntegralData compute_stepien_integrals(
    const Parameters::CahnHilliard<dim>       &ch,
    const Parameters::PhysicalProperties<dim> &physical_properties,
    const ComponentOrdering                   &ordering,
    const DoFHandler<dim>                     &dof_handler,
    const Mapping<dim>                        &mapping,
    const Quadrature<dim>                     &quadrature,
    const VectorType                          &solution)
  {
    using I           = CHNSIntegralData;
    const auto  &pp   = physical_properties;
    const double rho0 = pp.fluids[0].density, rho1 = pp.fluids[1].density;
    const double eta0 = rho0 * pp.fluids[0].kinematic_viscosity;
    const double eta1 = rho1 * pp.fluids[1].kinematic_viscosity;
    const double a    = 0.5 * (rho0 - rho1);
    const double dpr =
      0.5 * (pp.fluids[0].pressure_ref - pp.fluids[1].pressure_ref);
    const double sigma   = CahnHilliard::sigma_tilde_from_surface_tension(ch);
    const auto   limiter = CahnHilliard::get_limiter_function(ch);
    const FEValuesExtractors::Scalar tracer_extractor(ordering.phi_lower);
    const FEValuesExtractors::Scalar potential_extractor(ordering.mu_lower);
    const FEValuesExtractors::Scalar pressure_extractor(ordering.p_lower);
    const FEValuesExtractors::Vector velocity_extractor(ordering.u_lower);
    FEValues<dim>                    fv(mapping,
                     dof_handler.get_fe(),
                     quadrature,
                     update_values | update_gradients |
                       update_quadrature_points | update_JxW_values);
    const unsigned int               nq = quadrature.size();
    std::vector<double>              phi(nq), mu(nq);
    std::vector<Tensor<1, dim>> u(nq), grad_phi(nq), grad_mu(nq), grad_p(nq);
    std::vector<Tensor<2, dim>> grad_u(nq);
    CHNSIntegralData            result;
    double                      phi_min = std::numeric_limits<double>::max();
    double                      phi_max = std::numeric_limits<double>::lowest();
    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        fv.reinit(cell);
        fv[tracer_extractor].get_function_values(solution, phi);
        fv[tracer_extractor].get_function_gradients(solution, grad_phi);
        fv[potential_extractor].get_function_values(solution, mu);
        fv[potential_extractor].get_function_gradients(solution, grad_mu);
        fv[pressure_extractor].get_function_gradients(solution, grad_p);
        fv[velocity_extractor].get_function_values(solution, u);
        fv[velocity_extractor].get_function_gradients(solution, grad_u);
        for (unsigned int q = 0; q < nq; ++q)
        {
          const double weight = fv.JxW(q);
          const double rho =
            CahnHilliard::linear_mixing(limiter(phi[q]), rho0, rho1);
          const double rho_affine =
            CahnHilliard::linear_mixing(phi[q], rho0, rho1);
          const double eta =
            CahnHilliard::linear_mixing(limiter(phi[q]), eta0, eta1);
          const auto mobility =
            CahnHilliard::evaluate_mobility(ch, phi[q], rho0, rho1);
          const auto grad_xi = (rho * grad_mu[q] - a * grad_p[q] +
                                a * (mu[q] - dpr) * grad_phi[q]) /
                               (rho0 * rho1);
          const double indicator = phi[q] < 0. ? 1. : 0.;
          const double fraction =
            0.5 * (1. - CahnHilliard::tracer_limiter(phi[q]));
          const auto  &x     = fv.quadrature_point(q);
          const auto   Du    = symmetrize(grad_u[q]);
          const double div_u = trace(grad_u[q]);
          const double bulk_unscaled =
            0.25 / ch.epsilon_interface * std::pow(phi[q] * phi[q] - 1., 2);
          const double gradient_unscaled =
            0.5 * ch.epsilon_interface * grad_phi[q].norm_square();
          // Preserve the Stepien diagnostics' property density for inertia and
          // affine density for gravity, including when the limiter is active.
          const CHNSEnergyComponents energy{
            {{0.5 * rho * u[q].norm_square(),
              sigma * bulk_unscaled,
              sigma * gradient_unscaled,
              -rho_affine * (pp.body_force * Tensor<1, dim>(x))}}};
          add_phase_energy(result, phi[q], weight, energy);
          std::array<double, I::IntegralCount> values{};
          values[I::PhaseIntegral] = phi[q];
          values[I::AffineMass]    = rho_affine;
          values[I::PositiveMass]  = rho;
          values[I::KineticEnergy] =
            energy.values[CHNSEnergyComponents::Kinetic];
          values[I::BulkEnergy] = energy.values[CHNSEnergyComponents::Bulk];
          values[I::GradientEnergy] =
            energy.values[CHNSEnergyComponents::Gradient];
          values[I::InterfacialEnergy] =
            sigma * (bulk_unscaled + gradient_unscaled);
          values[I::GravitationalEnergy] =
            energy.values[CHNSEnergyComponents::Gravitational];
          values[I::ViscousDissipation] =
            2. * eta * scalar_product(Du, Du) +
            CahnHilliard::stokes_second_viscosity_ratio * eta * div_u * div_u;
          values[I::DiffusiveDissipation] =
            mobility.value * grad_xi.norm_square();
          values[I::BubbleVolume]                 = indicator;
          values[I::BubbleHeightMoment]           = indicator * x[1];
          values[I::BubbleVelocityMoment]         = indicator * u[q][1];
          values[I::WeightedBubbleVolume]         = fraction;
          values[I::WeightedBubbleHeightMoment]   = fraction * x[1];
          values[I::WeightedBubbleVelocityMoment] = fraction * u[q][1];
          values[I::Volume]                       = 1.;
          values[I::NegativeDensityVolume]        = rho_affine < 0. ? 1. : 0.;
          values[I::PhaseOvershootVolume] =
            phi[q] < -1. || phi[q] > 1. ? 1. : 0.;
          values[I::DensityExtensionL1]    = std::abs(rho - rho_affine);
          values[I::VelocityNormSquared]   = u[q].norm_square();
          values[I::DivergenceNormSquared] = div_u * div_u;
          values[I::DiffusiveFluxNormSquared] =
            mobility.value * mobility.value * grad_xi.norm_square();
          for (unsigned int k = 0; k < values.size(); ++k)
            result.sums[k] += weight * values[k];
          phi_min = std::min(phi_min, phi[q]);
          phi_max = std::max(phi_max, phi[q]);
        }
      }
    const auto communicator = dof_handler.get_mpi_communicator();
    sum_chns_integrals(result, communicator);
    result.phi_min = Utilities::MPI::min(phi_min, communicator);
    result.phi_max = Utilities::MPI::max(phi_max, communicator);
    return result;
  }

  template CHNSIntegralData compute_chns_integrals<2, LA::ParVectorType>(
    const Parameters::CahnHilliard<2> &,
    const Parameters::PhysicalProperties<2> &,
    const ComponentOrdering &,
    const DoFHandler<2> &,
    const Mapping<2> &,
    const Quadrature<2> &,
    const LA::ParVectorType &,
    const std::vector<LA::ParVectorType> &,
    const TimeHandler &);
  template CHNSIntegralData compute_chns_integrals<3, LA::ParVectorType>(
    const Parameters::CahnHilliard<3> &,
    const Parameters::PhysicalProperties<3> &,
    const ComponentOrdering &,
    const DoFHandler<3> &,
    const Mapping<3> &,
    const Quadrature<3> &,
    const LA::ParVectorType &,
    const std::vector<LA::ParVectorType> &,
    const TimeHandler &);

  template CHNSIntegralData compute_stepien_integrals<2, LA::ParVectorType>(
    const Parameters::CahnHilliard<2> &,
    const Parameters::PhysicalProperties<2> &,
    const ComponentOrdering &,
    const DoFHandler<2> &,
    const Mapping<2> &,
    const Quadrature<2> &,
    const LA::ParVectorType &);
  template CHNSIntegralData compute_stepien_integrals<3, LA::ParVectorType>(
    const Parameters::CahnHilliard<3> &,
    const Parameters::PhysicalProperties<3> &,
    const ComponentOrdering &,
    const DoFHandler<3> &,
    const Mapping<3> &,
    const Quadrature<3> &,
    const LA::ParVectorType &);
} // namespace PostProcessingTools
