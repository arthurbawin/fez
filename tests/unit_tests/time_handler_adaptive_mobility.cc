#include <cahn_hilliard.h>
#include <parameters.h>
#include <time_handler.h>

#include "../tests.h"

namespace
{
  Parameters::TimeIntegration make_parameters(
    const Parameters::TimeIntegration::Scheme scheme,
    const double                              dt)
  {
    ParameterHandler                    prm;
    Parameters::TimeIntegration         parameters;
    parameters.declare_parameters(prm);

    prm.enter_subsection("Time integration");
    prm.set("dt", std::to_string(dt));
    prm.set("t_initial", "0");
    prm.set("t_end", "1");
    prm.set("scheme", scheme == Parameters::TimeIntegration::Scheme::BDF1 ?
                        "BDF1" :
                        "BDF2");
    prm.set("bdf start method", "BDF1");
    prm.set("bdf start step ratio", "0.25");
    prm.enter_subsection("Adaptation");
    prm.set("enable", "true");
    prm.set("adaptation strategy", "adaptive mobility");
    prm.set("target adaptive mobility number", "1");
    prm.set("reject timestep with large adaptive mobility", "true");
    prm.set("adaptive mobility ratio to reject", "1");
    prm.set("max timestep", "1");
    prm.set("min timestep", "1e-6");
    prm.set("max timestep increase", "10");
    prm.set("max timestep reduction", "0.1");
    prm.leave_subsection();
    prm.leave_subsection();

    parameters.read_parameters(prm);
    return parameters;
  }

  void initialize_vector(LA::ParVectorType &vector, const double value)
  {
    IndexSet owned(1);
    owned.add_index(0);
    owned.compress();

    vector.reinit(owned, MPI_COMM_WORLD);
    vector[0] = value;
    vector.compress(VectorOperation::insert);
  }

  void test_rejection_and_prediction()
  {
    const auto parameters =
      make_parameters(Parameters::TimeIntegration::Scheme::BDF1, 0.2);
    AssertThrow(
      parameters.adaptation.strategy ==
        Parameters::TimeIntegration::Adaptation::AdaptationStrategy::
          AdaptiveMobility,
      ExcInternalError());

    TimeHandler       time_handler(parameters);
    ConditionalOStream quiet(std::cout, false);
    LA::ParVectorType present;
    initialize_vector(present, 42.);
    std::vector<LA::ParVectorType> previous(1);
    initialize_vector(previous[0], 7.);

    time_handler.advance(quiet);
    time_handler.set_max_adaptive_mobility_number(2.);
    const bool accepted =
      time_handler.is_timestep_accepted(present, previous);

    AssertThrow(!accepted, ExcInternalError());
    AssertThrow(time_handler.get_n_rejected_steps() == 1, ExcInternalError());
    AssertThrow(std::abs(time_handler.current_time) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(time_handler.current_dt - 0.09) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(present[0] - 7.) < 1e-14, ExcInternalError());

    time_handler.advance(quiet);
    time_handler.set_max_adaptive_mobility_number(0.9);
    const bool retry_accepted =
      time_handler.is_timestep_accepted(present, previous);

    AssertThrow(retry_accepted, ExcInternalError());
    AssertThrow(std::abs(time_handler.current_time - 0.09) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(time_handler.current_dt - 0.1) < 1e-14,
                ExcInternalError());

    deallog << "BDF1 rejection, rollback, and prediction OK" << std::endl;
  }

  void test_bdf2_startup_is_never_rejected()
  {
    const auto parameters =
      make_parameters(Parameters::TimeIntegration::Scheme::BDF2, 0.2);
    TimeHandler       time_handler(parameters);
    ConditionalOStream quiet(std::cout, false);
    LA::ParVectorType present;
    initialize_vector(present, 3.);
    std::vector<LA::ParVectorType> previous(2);
    initialize_vector(previous[0], 2.);
    initialize_vector(previous[1], 1.);

    time_handler.advance(quiet);
    time_handler.set_max_adaptive_mobility_number(10.);
    const bool first_accepted =
      time_handler.is_timestep_accepted(present, previous);
    AssertThrow(first_accepted, ExcInternalError());
    AssertThrow(time_handler.get_n_rejected_steps() == 0, ExcInternalError());
    AssertThrow(std::abs(time_handler.current_dt - 0.15) < 1e-14,
                ExcInternalError());

    time_handler.advance(quiet);
    time_handler.set_max_adaptive_mobility_number(10.);
    const bool second_accepted =
      time_handler.is_timestep_accepted(present, previous);
    AssertThrow(second_accepted, ExcInternalError());
    AssertThrow(time_handler.get_n_rejected_steps() == 0, ExcInternalError());
    AssertThrow(std::abs(time_handler.current_dt - 0.2) < 1e-14,
                ExcInternalError());

    time_handler.advance(quiet);
    time_handler.set_max_adaptive_mobility_number(2.);
    const bool third_accepted =
      time_handler.is_timestep_accepted(present, previous);
    AssertThrow(!third_accepted, ExcInternalError());
    AssertThrow(time_handler.get_n_rejected_steps() == 1, ExcInternalError());
    AssertThrow(std::abs(time_handler.current_time - 0.2) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(time_handler.current_dt - 0.09) < 1e-14,
                ExcInternalError());

    deallog << "BDF2 startup immunity and post-startup rejection OK"
            << std::endl;
  }

  void test_adaptive_mobility_scaling()
  {
    Parameters::CahnHilliard<2> parameters;
    parameters.surface_tension = 4. * std::sqrt(2.) / 3.;
    parameters.epsilon_interface = 2.;
    parameters.adaptive_mobility_n = 3.;
    parameters.adaptive_mobility_delta = 0.125;
    parameters.adaptive_mobility_2_n = 4.;
    parameters.adaptive_mobility_2_delta = 0.25;
    parameters.adaptive_mobility_3_n = 5.;
    parameters.adaptive_mobility_3_delta = 0.5;

    using MobilityModel = Parameters::CahnHilliard<2>::MobilityModel;

    parameters.mobility_model = MobilityModel::adaptive;
    auto scaling = CahnHilliard::get_adaptive_mobility_scaling(parameters);
    AssertThrow(std::abs(scaling.coefficient - 12. * std::sqrt(2.)) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(scaling.delta - 0.125) < 1e-14,
                ExcInternalError());

    parameters.mobility_model = MobilityModel::adaptive_mobility_2;
    scaling = CahnHilliard::get_adaptive_mobility_scaling(parameters);
    AssertThrow(std::abs(scaling.coefficient - 16. * std::sqrt(2.)) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(scaling.delta - 0.25) < 1e-14,
                ExcInternalError());

    parameters.mobility_model = MobilityModel::adaptive_mobility_3;
    scaling = CahnHilliard::get_adaptive_mobility_scaling(parameters);
    AssertThrow(std::abs(scaling.coefficient - 10.) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(scaling.delta - 0.5) < 1e-14,
                ExcInternalError());

    AssertThrow(std::abs(CahnHilliard::compute_adaptive_mobility_number(
                           0.25, parameters, 8.) -
                         0.5) <
                  1e-14,
                ExcInternalError());

    parameters.mobility_model = MobilityModel::constant;
    bool rejected_constant_mobility = false;
    try
    {
      CahnHilliard::get_adaptive_mobility_scaling(parameters);
    }
    catch (const ExceptionBase &)
    {
      rejected_constant_mobility = true;
    }
    AssertThrow(rejected_constant_mobility, ExcInternalError());

    deallog << "Adaptive mobility scaling and time number OK" << std::endl;
  }

  void test_adaptive_mobility_2_restriction_weight()
  {
    const auto core =
      CahnHilliard::evaluate_adaptive_mobility_2_restriction_weight(0.5);
    AssertThrow(std::abs(core.value - 1.) < 1e-14, ExcInternalError());
    AssertThrow(std::abs(core.first_derivative) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(core.second_derivative) < 1e-14,
                ExcInternalError());

    // At the midpoint of the transition in phi^2, the complementary
    // quintic smoothstep is exactly 1/2. This independently catches the old
    // reciprocal tail extension, for which the weight was greater than one.
    constexpr double phi_core = 0.5;
    constexpr double phi_cutoff = 0.9;
    const double phi_mid = std::sqrt(
      0.5 * (phi_core * phi_core + phi_cutoff * phi_cutoff));
    const auto positive_transition =
      CahnHilliard::evaluate_adaptive_mobility_2_restriction_weight(phi_mid);
    const auto negative_transition =
      CahnHilliard::evaluate_adaptive_mobility_2_restriction_weight(-phi_mid);
    AssertThrow(std::abs(positive_transition.value - 0.5) < 1e-12,
                ExcInternalError());
    AssertThrow(positive_transition.first_derivative < 0.,
                ExcInternalError());
    AssertThrow(std::abs(negative_transition.value -
                         positive_transition.value) < 1e-12,
                ExcInternalError());
    AssertThrow(std::abs(negative_transition.first_derivative +
                         positive_transition.first_derivative) < 1e-9,
                ExcInternalError());
    AssertThrow(std::abs(negative_transition.second_derivative -
                         positive_transition.second_derivative) < 1e-6,
                ExcInternalError());

    for (const double phi : {phi_cutoff, 1.05, -phi_cutoff, -1.05})
    {
      const auto restricted =
        CahnHilliard::evaluate_adaptive_mobility_2_restriction_weight(phi);
      AssertThrow(std::abs(restricted.value) < 1e-12,
                  ExcInternalError());
      AssertThrow(std::abs(restricted.first_derivative) < 1e-12,
                  ExcInternalError());
      AssertThrow(std::abs(restricted.second_derivative) < 1e-9,
                  ExcInternalError());
    }

    // A missing or incorrect analytic derivative must be detected
    // independently from the value formula used by the implementation.
    for (const double phi : {0.6, 0.8})
    {
      constexpr double h_first = 1e-7;
      constexpr double h_second = 2e-6;
      const auto evaluation =
        CahnHilliard::evaluate_adaptive_mobility_2_restriction_weight(phi);
      const double value_plus =
        CahnHilliard::evaluate_adaptive_mobility_2_restriction_weight(
          phi + h_first)
          .value;
      const double value_minus =
        CahnHilliard::evaluate_adaptive_mobility_2_restriction_weight(
          phi - h_first)
          .value;
      const double fd_first = (value_plus - value_minus) / (2. * h_first);
      const double second_plus =
        CahnHilliard::evaluate_adaptive_mobility_2_restriction_weight(
          phi + h_second)
          .value;
      const double second_center = evaluation.value;
      const double second_minus =
        CahnHilliard::evaluate_adaptive_mobility_2_restriction_weight(
          phi - h_second)
          .value;
      const double fd_second =
        (second_plus - 2. * second_center + second_minus) /
        (h_second * h_second);
      AssertThrow(std::abs(evaluation.first_derivative - fd_first) <
                    2e-4 * std::max(1., std::abs(fd_first)),
                  ExcInternalError());
      AssertThrow(std::abs(evaluation.second_derivative - fd_second) <
                    2e-3 * std::max(1., std::abs(fd_second)),
                  ExcInternalError());
    }

    deallog << "Adaptive mobility 2 restriction weight and derivatives OK"
            << std::endl;
  }

  void test_mobility_tracer_argument_selection()
  {
    Parameters::CahnHilliard<2> parameters;
    using MobilityModel = Parameters::CahnHilliard<2>::MobilityModel;

    parameters.mobility_model = MobilityModel::adaptive_mobility_2;
    auto argument = CahnHilliard::select_mobility_tracer_argument(
      parameters, 0.95, 0.999, 0.1, 0.2);
    AssertThrow(std::abs(argument.value - 0.95) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(argument.first_derivative - 1.) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(argument.second_derivative) < 1e-14,
                ExcInternalError());

    parameters.mobility_model = MobilityModel::degenerate;
    argument = CahnHilliard::select_mobility_tracer_argument(
      parameters, 0.95, 0.999, 0.1, 0.2);
    AssertThrow(std::abs(argument.value - 0.999) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(argument.first_derivative - 0.1) < 1e-14,
                ExcInternalError());
    AssertThrow(std::abs(argument.second_derivative - 0.2) < 1e-14,
                ExcInternalError());

    deallog << "Mobility tracer argument selection OK" << std::endl;
  }
} // namespace

int main(int argc, char **argv)
{
  initlog();
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  test_rejection_and_prediction();
  test_bdf2_startup_is_never_rejected();
  test_adaptive_mobility_scaling();
  test_adaptive_mobility_2_restriction_weight();
  test_mobility_tracer_argument_selection();
}
