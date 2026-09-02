#include <cahn_hilliard.h>

#include "../tests.h"

namespace
{
  Parameters::CahnHilliard<2>
  make_profile_parameters(const std::string &correction,
                          const std::string &model = "abels")
  {
    ParameterHandler              prm;
    Parameters::CahnHilliard<2> parameters;
    parameters.declare_parameters(prm);
    prm.enter_subsection("Cahn Hilliard");
    prm.set("CHNS model", model);
    prm.set("interface profile correction", correction);
    prm.set("profile correction strength", "0.3");
    prm.set("surface tension", "2.");
    prm.leave_subsection();
    parameters.read_parameters(prm);
    return parameters;
  }

  void test_regularized_profile_correction_2d()
  {
    Tensor<1, 2> grad_phi;
    grad_phi[0] = 3.;
    grad_phi[1] = 4.;

    const auto normal =
      CahnHilliard::regularized_interface_normal<2>(grad_phi, 12.);
    AssertThrow(std::abs(normal[0] - 3. / 13.) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(normal[1] - 4. / 13.) < 1e-14,
                ExcInternalError());

    Tensor<1, 2> zero_gradient;
    const auto zero_normal =
      CahnHilliard::regularized_interface_normal<2>(zero_gradient, 2.);
    AssertThrow(zero_normal.norm_square() == 0., ExcInternalError());

    const double phi     = 0.6;
    const double epsilon = 2.;
    Tensor<1, 2> equilibrium_gradient;
    equilibrium_gradient[0] = (1. - phi * phi) /
                              (std::sqrt(2.) * epsilon);
    const auto profile_driver =
      CahnHilliard::profile_correction_flux_driver<2>(
        phi, equilibrium_gradient, epsilon, 0.);
    AssertThrow(profile_driver.norm() < 1e-14, ExcInternalError());

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
    Tensor<1, 2> grad_phi;
    grad_phi[0] = 0.7;
    grad_phi[1] = -0.4;
    Tensor<1, 2> grad_phi_variation;
    grad_phi_variation[0] = 0.2;
    grad_phi_variation[1] = 0.3;
    Tensor<1, 2> grad_mu;
    grad_mu[0] = -0.6;
    grad_mu[1] = 0.9;
    Tensor<1, 2> grad_mu_variation;
    grad_mu_variation[0] = 0.5;
    grad_mu_variation[1] = -0.1;

    const double phi                = 0.35;
    const double phi_variation      = -0.25;
    const double mobility           = 0.8;
    const double mobility_variation = 0.17;
    const double step               = 1e-7;

    for (const std::string mode : {"profile", "profile_flux"})
    {
      auto parameters = make_profile_parameters(mode);
      parameters.epsilon_interface = 0.6;
      const double normal_denominator =
        std::sqrt(grad_phi.norm_square() +
                  std::pow(CahnHilliard::profile_correction_normal_regularization(
                             parameters),
                           2));
      const auto normal = grad_phi / normal_denominator;

      const auto analytical =
        CahnHilliard::phase_diffusion_flux_driver_variation<2>(
          parameters,
          phi,
          phi_variation,
          grad_phi,
          grad_phi_variation,
          grad_mu,
          grad_mu_variation,
          mobility,
          mobility_variation,
          normal,
          normal_denominator);
      const auto plus = CahnHilliard::phase_diffusion_flux_driver<2>(
        parameters,
        phi + step * phi_variation,
        grad_phi + step * grad_phi_variation,
        grad_mu + step * grad_mu_variation,
        mobility + step * mobility_variation);
      const auto minus = CahnHilliard::phase_diffusion_flux_driver<2>(
        parameters,
        phi - step * phi_variation,
        grad_phi - step * grad_phi_variation,
        grad_mu - step * grad_mu_variation,
        mobility - step * mobility_variation);
      const auto finite_difference = (plus - minus) / (2. * step);
      AssertThrow((analytical - finite_difference).norm() < 1e-8,
                  ExcInternalError());
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

    const auto parameters = make_profile_parameters("profile_flux");
    AssertThrow(parameters.interface_profile_correction ==
                  Correction::profile_flux,
                ExcInternalError());
    AssertThrow(std::abs(parameters.profile_correction_strength - 0.3) <
                  1e-14,
                ExcInternalError());
    AssertThrow(
      std::abs(CahnHilliard::profile_correction_normal_regularization(
                 parameters) -
               0.01 / (std::sqrt(2.) * parameters.epsilon_interface)) < 1e-14,
      ExcInternalError());
    const double sigma_tilde = 3. / (2. * std::sqrt(2.)) * 2.;
    AssertThrow(
      std::abs(CahnHilliard::profile_correction_coefficient(parameters, 0.8) -
               0.3 * 2. * 0.8 * sigma_tilde /
                 parameters.epsilon_interface) < 1e-14,
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
    const double delta = 0.01 / std::sqrt(2.);
    const double normal_denominator = std::sqrt(25. + delta * delta);
    const double kappa = 0.3 * 2. * 2. *
                         (3. / (2. * std::sqrt(2.)) * 2.);
    AssertThrow(std::abs(profile_flux[0] -
                         (10. + kappa *
                                  (3. - q * 3. / normal_denominator))) <
                  1e-14,
                ExcInternalError());
    AssertThrow(std::abs(profile_flux[1] -
                         kappa * (4. - q * 4. / normal_denominator)) <
                  1e-14,
                ExcInternalError());

    auto profile_and_flux = make_profile_parameters("profile_flux");
    profile_and_flux.epsilon_interface = 1.;
    const auto corrected = CahnHilliard::phase_diffusion_flux_driver<2>(
      profile_and_flux, 0., grad_phi, grad_mu, 2.);
    const double normal_x = 3. / normal_denominator;
    const double normal_y = 4. / normal_denominator;
    const double normal_dot_grad_mu = 5. * normal_x;
    AssertThrow(std::abs(corrected[0] -
                         (2. * (5. - normal_x * normal_dot_grad_mu) +
                          kappa * (3. - q * normal_x))) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(corrected[1] -
                         (2. * (-normal_y * normal_dot_grad_mu) +
                          kappa * (4. - q * normal_y))) < 1e-14,
                ExcInternalError());

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
