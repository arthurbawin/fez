#ifndef TIME_HANDLER_H
#define TIME_HANDLER_H

#include <parameters.h>

/**
 * This class takes care of the time integration-related data:
 *
 * - updating the current time and time step count
 * - update the BDF coefficients if using a BDF method
 */
class TimeHandler
{
public:
  TimeHandler(const Parameters::TimeIntegration &time_parameters);

  /**
   * Update the BDF coefficients given the current and previous
   * time steps.
   */
  void set_bdf_coefficients();

  /**
   * Rotates the computed time step i+1 to position i.
   */
  void rotate();

public:
  double       current_time;
  unsigned int current_time_iteration;
  double       initial_time;
  double       final_time;
  std::vector<double> previous_times;

  double              initial_dt;
  double              current_dt;
  std::vector<double> time_steps;

  Parameters::TimeIntegration::Scheme scheme;

  unsigned int        n_previous_solutions;
  std::vector<double> bdf_coefficients;
};

#endif