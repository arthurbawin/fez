#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <time_handler.h>

#include <limits>
#include <sstream>
#include <vector>

#include "../tests.h"

namespace
{
  using Scheme   = Parameters::TimeIntegration::Scheme;
  using BDFStart = Parameters::TimeIntegration::BDFStart;

  Parameters::TimeIntegration make_time_parameters(const Scheme   scheme,
                                                   const BDFStart bdfstart)
  {
    Parameters::TimeIntegration parameters;

    parameters.verbosity               = Parameters::Verbosity::quiet;
    parameters.dt                      = 0.2;
    parameters.t_initial               = 0.;
    parameters.t_end                   = 1.;
    parameters.scheme                  = scheme;
    parameters.bdfstart                = bdfstart;
    parameters.bdf_starting_step_ratio = 0.25;

    parameters.adaptation.verbosity = Parameters::Verbosity::quiet;
    parameters.adaptation.enable    = false;
    parameters.adaptation.strategy  = Parameters::TimeIntegration::Adaptation::
      AdaptationStrategy::BDFTruncationError;
    parameters.adaptation.max_timestep                     = 1.;
    parameters.adaptation.min_timestep                     = 1e-6;
    parameters.adaptation.max_timestep_increase            = 10.;
    parameters.adaptation.max_timestep_reduction           = 0.1;
    parameters.adaptation.reject_timestep_with_large_error = false;
    parameters.adaptation.reject_error_factor              = 2.;
    parameters.adaptation.target_cfl                       = 1.;
    parameters.adaptation.reject_timestep_with_large_cfl   = false;
    parameters.adaptation.reject_cfl_factor                = 2.;
    parameters.adaptation.compute_error_on_estimator       = false;

    parameters.n_time_intervals = 1;

    return parameters;
  }

  TimeHandler reload_steady_checkpoint_as_unsteady(
    const Parameters::TimeIntegration &unsteady_parameters)
  {
    auto steady_parameters   = unsteady_parameters;
    steady_parameters.scheme = Scheme::stationary;
    steady_parameters.t_end  = steady_parameters.t_initial;
    TimeHandler steady_handler(steady_parameters);

    TimeHandler restarted_handler(unsteady_parameters);

    std::stringstream checkpoint;
    {
      boost::archive::text_oarchive archive(checkpoint);
      archive << steady_handler;
    }
    {
      boost::archive::text_iarchive archive(checkpoint);
      archive >> restarted_handler;
    }

    restarted_handler.update_parameters_after_restart(unsteady_parameters);

    return restarted_handler;
  }

  void assert_coefficients_are(const TimeHandler         &time_handler,
                               const std::vector<double> &expected_coefficients)
  {
    const auto &coefficients = time_handler.get_bdf_coefficients();
    AssertThrow(coefficients.size() == expected_coefficients.size(),
                ExcMessage("BDF coefficient vector size mismatch"));

    for (unsigned int i = 0; i < coefficients.size(); ++i)
      AssertThrow(std::abs(coefficients[i] - expected_coefficients[i]) < 1e-14,
                  ExcMessage("BDF coefficient mismatch"));
  }

  void assert_constant_solution_has_zero_time_derivative(
    const TimeHandler &time_handler)
  {
    const std::vector<double>              value(1, 42.);
    const std::vector<std::vector<double>> previous_values(
      time_handler.n_previous_solutions, value);

    const double value_dot =
      time_handler.compute_time_derivative(0, value, previous_values);

    // Allow cancellation roundoff in the variable-step BDF2 sum.
    double derivative_scale = 0.;
    for (const double coefficient : time_handler.get_bdf_coefficients())
      derivative_scale += std::abs(coefficient * value[0]);
    const double tolerance =
      10. * std::numeric_limits<double>::epsilon() * derivative_scale;

    AssertThrow(std::abs(value_dot) <= tolerance,
                ExcMessage("Constant solution should have zero time "
                           "derivative after restart"));
  }

  void advance_once(TimeHandler &time_handler)
  {
    std::ostringstream empty_stream;
    ConditionalOStream pcout(empty_stream, false);
    time_handler.advance(pcout);
  }

  void test_restart_from_steady_to_bdf1()
  {
    const auto parameters =
      make_time_parameters(Scheme::BDF1, BDFStart::initial_condition);
    auto time_handler = reload_steady_checkpoint_as_unsteady(parameters);

    AssertThrow(!time_handler.is_steady(), ExcInternalError());
    AssertThrow(time_handler.n_previous_solutions == 1, ExcInternalError());

    advance_once(time_handler);

    const double dt = parameters.dt;
    assert_coefficients_are(time_handler, {1. / dt, -1. / dt});
    assert_constant_solution_has_zero_time_derivative(time_handler);

    deallog << "steady_to_bdf1 OK" << std::endl;
  }

  void test_restart_from_steady_to_bdf2_initial_condition()
  {
    const auto parameters =
      make_time_parameters(Scheme::BDF2, BDFStart::initial_condition);
    auto time_handler = reload_steady_checkpoint_as_unsteady(parameters);

    AssertThrow(!time_handler.is_steady(), ExcInternalError());
    AssertThrow(time_handler.n_previous_solutions == 2, ExcInternalError());

    advance_once(time_handler);

    const double dt = parameters.dt;
    assert_coefficients_are(time_handler, {1.5 / dt, -2. / dt, 0.5 / dt});
    assert_constant_solution_has_zero_time_derivative(time_handler);

    deallog << "steady_to_bdf2_initial_condition OK" << std::endl;
  }

  void test_restart_from_steady_to_bdf2_bdf1_start()
  {
    const auto parameters = make_time_parameters(Scheme::BDF2, BDFStart::BDF1);
    auto       time_handler = reload_steady_checkpoint_as_unsteady(parameters);

    AssertThrow(!time_handler.is_steady(), ExcInternalError());
    AssertThrow(time_handler.n_previous_solutions == 2, ExcInternalError());

    advance_once(time_handler);

    const double starting_dt =
      parameters.dt * parameters.bdf_starting_step_ratio;
    assert_coefficients_are(time_handler,
                            {1. / starting_dt, -1. / starting_dt, 0.});
    assert_constant_solution_has_zero_time_derivative(time_handler);

    // Accept the reduced BDF1 step through the normal time-step handling path.
    LA::ParVectorType present_solution(MPI_COMM_SELF, 1, 1);
    present_solution = 42.;
    const std::vector<LA::ParVectorType> previous_solutions(
      time_handler.n_previous_solutions, present_solution);
    AssertThrow(time_handler.is_timestep_accepted(present_solution,
                                                  previous_solutions),
                ExcMessage("The initial BDF1 step should be accepted"));

    advance_once(time_handler);

    const double second_dt = parameters.dt - starting_dt;
    AssertThrow(std::abs(time_handler.get_current_timestep() - second_dt) <
                  1e-14,
                ExcMessage("Incorrect time step after BDF1 startup"));

    // Analytical variable-step BDF2 coefficients for steps of 0.05 and 0.15.
    assert_coefficients_are(time_handler, {35. / 3., -80. / 3., 15.});
    assert_constant_solution_has_zero_time_derivative(time_handler);

    deallog << "steady_to_bdf2_bdf1_start OK" << std::endl;
  }
} // namespace

int main(int argc, char *argv[])
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  initlog();

  test_restart_from_steady_to_bdf1();
  test_restart_from_steady_to_bdf2_initial_condition();
  test_restart_from_steady_to_bdf2_bdf1_start();

  deallog << "OK" << std::endl;
}
