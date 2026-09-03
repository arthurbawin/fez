#include <cahn_hilliard.h>

#include "../tests.h"

namespace
{
  Parameters::CahnHilliard<2>
  make_profile_parameters(const std::string &correction,
                          const std::string &model = "abels",
                          const std::string &mobility_exponent = "1.0")
  {
    ParameterHandler              prm;
    Parameters::CahnHilliard<2> parameters;
    parameters.declare_parameters(prm);
    prm.enter_subsection("Cahn Hilliard");
    prm.set("CHNS model", model);
    prm.set("interface profile correction", correction);
    prm.set("profile correction strength", "0.3");
    prm.set("profile correction mobility exponent", mobility_exponent);
    prm.set("surface tension", "2.");
    prm.leave_subsection();
    parameters.read_parameters(prm);
    return parameters;
  }

  void test_regularized_profile_correction_2d()
  {
    auto parameters = make_profile_parameters("profile");
    parameters.epsilon_interface = 0.6;

    Tensor<1, 2> zero;
    for (const double phi : {-0.9, -0.99, -0.999, 0.999})
    {
      const double q = (1. - phi * phi) /
                       (std::sqrt(2.) * parameters.epsilon_interface);
      Tensor<1, 2> equilibrium_gradient;
      equilibrium_gradient[0] = q;
      const auto flux = CahnHilliard::phase_diffusion_flux_driver<2>(
        parameters, phi, equilibrium_gradient, zero, 0.8);
      AssertThrow(flux.norm() < 1e-13, ExcInternalError());
    }

    const double phi = -0.999;
    const double q = (1. - phi * phi) /
                     (std::sqrt(2.) * parameters.epsilon_interface);
    Tensor<1, 2> widened_gradient;
    widened_gradient[0] = 0.1 * q;
    const auto widened_flux = CahnHilliard::phase_diffusion_flux_driver<2>(
      parameters, phi, widened_gradient, zero, 0.8);
    const double beta = 0.05;
    const double c = 1. - std::sqrt(1. + beta * beta) /
                           std::sqrt(0.1 * 0.1 + beta * beta);
    const double constant_mobility =
      std::pow(parameters.epsilon_interface,
               parameters.profile_correction_mobility_exponent);
    const double kappa = 0.3 * 2. * constant_mobility *
                         (3. / (2. * std::sqrt(2.)) * 2.) /
                         parameters.epsilon_interface;
    AssertThrow(std::abs(widened_flux[0] - kappa * c * 0.1 * q) < 1e-13,
                ExcInternalError());

    Tensor<1, 2> unresolved_gradient;
    const auto zero_gradient_flux =
      CahnHilliard::phase_diffusion_flux_driver<2>(
        parameters, phi, unresolved_gradient, zero, 0.8);
    AssertThrow(zero_gradient_flux.norm_square() == 0., ExcInternalError());

    deallog << "Regularized profile correction in 2D OK" << std::endl;
  }

  void test_flux_projection_2d()
  {
    Tensor<1, 2> normal;
    normal[0] = 0.6;
    normal[1] = 0.8;
    Tensor<1, 2> grad_mu;
    grad_mu[0] = 5.;
    grad_mu[1] = 0.;

    const auto projected =
      CahnHilliard::project_chemical_potential_gradient<2>(grad_mu, normal);
    AssertThrow(std::abs(projected[0] - 3.2) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(projected[1] + 2.4) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(projected * normal) < 1e-14,
                ExcInternalError());

    deallog << "Flux projection in 2D OK" << std::endl;
  }

  void test_complete_flux_driver_variation_2d()
  {
    Tensor<1, 2> grad_phi_variation;
    grad_phi_variation[0] = 0.2;
    grad_phi_variation[1] = 0.3;
    Tensor<1, 2> grad_mu;
    grad_mu[0] = -0.6;
    grad_mu[1] = 0.9;
    Tensor<1, 2> grad_mu_variation;
    grad_mu_variation[0] = 0.5;
    grad_mu_variation[1] = -0.1;

    const double phi_variation      = -0.25;
    const double mobility           = 0.8;
    const double mobility_variation = 0.17;
    const double step               = 5e-8;

    for (const std::string mode : {"profile", "profile_flux"})
    {
      auto parameters = make_profile_parameters(mode);
      parameters.epsilon_interface = 0.6;
      const double g0 =
        1. / (std::sqrt(2.) * parameters.epsilon_interface);
      const double s_tail = 1. - 0.999 * 0.999;
      const double transition_s = 0.375 * s_tail;
      const double transition_phi = -std::sqrt(1. - transition_s);
      const double tail_phi = -0.999;
      const double tail_q = g0 * (1. - tail_phi * tail_phi);

      struct State
      {
        double phi;
        double grad_x;
        double grad_y;
      };
      const std::vector<State> states = {
        {0.35, 0.7, -0.4},
        {tail_phi, 0.1 * tail_q, 0.},
        {transition_phi, 0.375 * s_tail * g0, 0.},
        {1.01, 0.7, -0.4}};

      for (unsigned int state_index = 0; state_index < states.size();
           ++state_index)
      {
        const auto &state = states[state_index];
        Tensor<1, 2> grad_phi;
        grad_phi[0] = state.grad_x;
        grad_phi[1] = state.grad_y;
        const double gradient_norm = grad_phi.norm();
        const auto normal = CahnHilliard::flux_correction_normal(
          parameters, state.phi, grad_phi);

        const auto analytical =
          CahnHilliard::phase_diffusion_flux_driver_variation<2>(
            parameters,
            state.phi,
            phi_variation,
            grad_phi,
            grad_phi_variation,
            grad_mu,
            grad_mu_variation,
            mobility,
            mobility_variation,
            normal,
            gradient_norm);
        const auto plus = CahnHilliard::phase_diffusion_flux_driver<2>(
          parameters,
          state.phi + step * phi_variation,
          grad_phi + step * grad_phi_variation,
          grad_mu + step * grad_mu_variation,
          mobility + step * mobility_variation);
        const auto minus = CahnHilliard::phase_diffusion_flux_driver<2>(
          parameters,
          state.phi - step * phi_variation,
          grad_phi - step * grad_phi_variation,
          grad_mu - step * grad_mu_variation,
          mobility - step * mobility_variation);
        const auto finite_difference = (plus - minus) / (2. * step);
        const double error = (analytical - finite_difference).norm();
        const double tolerance = 5e-8 * std::max(1., analytical.norm());
        AssertThrow(error < tolerance,
                    ExcMessage("Flux Jacobian mismatch in mode " + mode +
                               ", state " + std::to_string(state_index) +
                               ", relative error times 1e12 " +
                               std::to_string(1e12 * error /
                                              std::max(1., analytical.norm()))));
      }
    }

    deallog << "Complete flux-driver variation in 2D OK" << std::endl;
  }

  void test_profile_correction_parameters_and_guards()
  {
    using Correction =
      Parameters::CahnHilliard<2>::InterfaceProfileCorrection;

    ParameterHandler              default_prm;
    Parameters::CahnHilliard<2> default_parameters;
    default_parameters.declare_parameters(default_prm);
    default_parameters.read_parameters(default_prm);
    AssertThrow(default_parameters.interface_profile_correction ==
                  Correction::none,
                ExcInternalError());
    AssertThrow(std::abs(default_parameters.profile_correction_strength - 0.3) <
                  1e-14,
                ExcInternalError());
    AssertThrow(
      std::abs(default_parameters.profile_correction_mobility_exponent - 1.) <
        1e-14,
      ExcInternalError());

    const auto parameters =
      make_profile_parameters("profile_flux", "abels", "1.7");
    AssertThrow(parameters.interface_profile_correction ==
                  Correction::profile_flux,
                ExcInternalError());
    AssertThrow(std::abs(parameters.profile_correction_strength - 0.3) <
                  1e-14,
                ExcInternalError());
    AssertThrow(
      std::abs(parameters.profile_correction_mobility_exponent - 1.7) < 1e-14,
      ExcInternalError());
    const double sigma_tilde = 3. / (2. * std::sqrt(2.)) * 2.;
    AssertThrow(
      std::abs(CahnHilliard::profile_correction_coefficient(parameters) -
               0.3 * 2. *
                 std::pow(parameters.epsilon_interface, 1.7) * sigma_tilde /
                 parameters.epsilon_interface) <
        1e-14,
      ExcInternalError());
    CahnHilliard::validate_interface_profile_correction(parameters, false);

    bool rejected_non_abels = false;
    try
    {
      const auto ding =
        make_profile_parameters("profile", "ding_horriche");
      CahnHilliard::validate_interface_profile_correction(ding, false);
    }
    catch (const ExceptionBase &)
    {
      rejected_non_abels = true;
    }
    AssertThrow(rejected_non_abels, ExcInternalError());

    bool rejected_tracer_supg = false;
    try
    {
      CahnHilliard::validate_interface_profile_correction(parameters, true);
    }
    catch (const ExceptionBase &)
    {
      rejected_tracer_supg = true;
    }
    AssertThrow(rejected_tracer_supg, ExcInternalError());

    deallog << "Profile correction parameters and guards OK" << std::endl;
  }

  void test_phase_diffusion_flux_driver_modes()
  {
    Tensor<1, 2> grad_phi;
    grad_phi[0] = 3.;
    grad_phi[1] = 4.;
    Tensor<1, 2> grad_mu;
    grad_mu[0] = 5.;
    grad_mu[1] = 0.;

    ParameterHandler              default_prm;
    Parameters::CahnHilliard<2> none;
    none.declare_parameters(default_prm);
    none.read_parameters(default_prm);
    const auto classical = CahnHilliard::phase_diffusion_flux_driver<2>(
      none, 0., grad_phi, grad_mu, 2.);
    AssertThrow(std::abs(classical[0] - 10.) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(classical[1]) < 1e-14, ExcInternalError());

    auto profile = make_profile_parameters("profile");
    profile.epsilon_interface = 1.;
    const auto profile_flux = CahnHilliard::phase_diffusion_flux_driver<2>(
      profile, 0., grad_phi, grad_mu, 2.);
    const double q = 1. / std::sqrt(2.);
    const double beta = 0.05;
    const double profile_denominator =
      std::sqrt(25. + beta * beta * q * q);
    const double profile_factor =
      1. - std::sqrt(1. + beta * beta) * q / profile_denominator;
    const double kappa = 0.3 * 2. *
                         (3. / (2. * std::sqrt(2.)) * 2.);
    AssertThrow(std::abs(profile_flux[0] -
                         (10. + kappa * profile_factor * 3.)) <
                  1e-14,
                ExcInternalError());
    AssertThrow(std::abs(profile_flux[1] -
                         kappa * profile_factor * 4.) <
                  1e-14,
                ExcInternalError());

    auto profile_and_flux = make_profile_parameters("profile_flux");
    profile_and_flux.epsilon_interface = 1.;
    const auto corrected = CahnHilliard::phase_diffusion_flux_driver<2>(
      profile_and_flux, 0., grad_phi, grad_mu, 2.);
    const double normal_x = 3. / 5.;
    const double normal_y = 4. / 5.;
    const double normal_dot_grad_mu = 5. * normal_x;
    AssertThrow(std::abs(corrected[0] -
                         (2. * (5. - normal_x * normal_dot_grad_mu) +
                          kappa * profile_factor * 3.)) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(corrected[1] -
                         (2. * (-normal_y * normal_dot_grad_mu) +
                          kappa * profile_factor * 4.)) < 1e-14,
                ExcInternalError());

    const double tail_phi = -0.999;
    const double tail_q = (1. - tail_phi * tail_phi) / std::sqrt(2.);
    Tensor<1, 2> resolved_tail_gradient;
    resolved_tail_gradient[0] = tail_q;
    Tensor<1, 2> normal_chemical_gradient;
    normal_chemical_gradient[0] = 7.;
    const auto resolved_tail_flux =
      CahnHilliard::phase_diffusion_flux_driver<2>(
        profile_and_flux,
        tail_phi,
        resolved_tail_gradient,
        normal_chemical_gradient,
        2.);
    AssertThrow(resolved_tail_flux.norm() < 1e-13, ExcInternalError());

    Tensor<1, 2> unresolved_tail_gradient;
    unresolved_tail_gradient[0] = 0.1 * tail_q;
    const auto unresolved_profile =
      CahnHilliard::phase_diffusion_flux_driver<2>(
        profile,
        tail_phi,
        unresolved_tail_gradient,
        normal_chemical_gradient,
        2.);
    const auto unresolved_profile_flux =
      CahnHilliard::phase_diffusion_flux_driver<2>(
        profile_and_flux,
        tail_phi,
        unresolved_tail_gradient,
        normal_chemical_gradient,
        2.);
    AssertThrow((unresolved_profile_flux - unresolved_profile).norm() < 1e-13,
                ExcInternalError());

    for (const double bulk_phi : {1., 1.01, -1.01})
    {
      const auto bulk_profile = CahnHilliard::phase_diffusion_flux_driver<2>(
        profile, bulk_phi, grad_phi, grad_mu, 2.);
      const auto bulk_profile_flux =
        CahnHilliard::phase_diffusion_flux_driver<2>(
          profile_and_flux, bulk_phi, grad_phi, grad_mu, 2.);
      AssertThrow((bulk_profile - 2. * grad_mu).norm() < 1e-14,
                  ExcInternalError());
      AssertThrow((bulk_profile_flux - 2. * grad_mu).norm() < 1e-14,
                  ExcInternalError());
    }

    deallog << "Phase diffusion flux-driver modes OK" << std::endl;
  }
} // namespace

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  initlog();

  test_regularized_profile_correction_2d();
  test_flux_projection_2d();
  test_complete_flux_driver_variation_2d();
  test_profile_correction_parameters_and_guards();
  test_phase_diffusion_flux_driver_modes();
}
