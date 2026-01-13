#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <deal.II/base/convergence_table.h>
#include <parameters.h>

using namespace dealii;

/**
 * A thin wrapper to the ConvergenceTable for error computations.
 */
class ErrorHandler
{
public:
  ErrorHandler(const Parameters::MMS             &mms_parameters,
               const Parameters::TimeIntegration &time_parameters)
    : mms_param(mms_parameters)
    , time_param(time_parameters)
    , is_steady(time_parameters.scheme == Parameters::TimeIntegration::Scheme::stationary)
  {}

  /**
   * Create an entry in the error table, that is, an L^p norm in space and/or in
   * time for a given field. For instance, error_handler.create_entry("L2_u").
   */
  void create_entry(const std::string &error_name)
  {
    ordered_keys.push_back(error_name);
    domain_errors.insert({error_name, std::make_unique<double>()});

    // TODO: Reserve vectors with an estimate on the number of time steps?
    // This is easy if only constant time steps are expected for convergence
    // studies, because then the number of time steps is known.
    unsteady_errors[error_name].clear();
  }

  /**
   * Add an integer reference value (number of mesh elements or dof).
   * These values won't be printed in scientific notation.
   */
  void add_reference_data(const std::string &name, const unsigned int value)
  {
    error_table.add_value(name, value);
  }

  /**
   * Add the (constant) time step used for this step of convergence study.
   */
  void add_time_step(double time_step)
  {
    error_table.add_value("dt", time_step);
  }

  /**
   * Add a spatial error entry.
   * 
   * If the simulation is steady, this error is directly added to
   * the underlying error table to compute convergence.
   * 
   * If it is unsteady, this stores the spatial error at time t. The prescribed
   * L^p norm in time is computed at the end of the simulation.
   * The error at all times are kept, e.g. to be plotted in postprocessing.
   */
  void add_error(const std::string &field_name,
                 const double       error_val,
                 const double       time = 0.)
  {
    if(is_steady)
      add_steady_error(field_name, error_val);
    else
      add_unsteady_error(field_name, error_val, time);
  }

private:
  /**
   * Add an error to the underlying error table to compute convergence.
   */
  void add_steady_error(const std::string &error_name, const double error_val)
  {
    AssertThrow(
      domain_errors.count(error_name) == 1,
      ExcMessage(
        "Cannot add steady error value for field " + error_name +
        " because it does not exist in the error handler. Add it first."));
    *(domain_errors.at(error_name)) = error_val;
    error_table.add_value(error_name, error_val);
  }

  /**
   * Store a spatial error at time t for an unsteady simulation.
   */
  void add_unsteady_error(const std::string &error_name,
                          const double       error_val,
                          const double       time)
  {
    AssertThrow(
      domain_errors.count(error_name) == 1,
      ExcMessage(
        "Cannot add unsteady error value for field " + error_name +
        " because it does not exist in the error handler. Add it first."));
    // *(domain_errors.at(error_name)) = error_val;
    // error_table.add_value(error_name, error_val);

    auto &error_vec = unsteady_errors.at(error_name);
    error_vec.push_back({time, error_val});
  }

public:
  // Compute the temporal or spacetime error if needed
  void compute_temporal_error()
  {
    for (const auto &key : ordered_keys)
    {
      auto &error_vec = unsteady_errors.at(key);

      double error = 0.;
      switch (mms_param.time_norm)
      {
        case Parameters::MMS::TimeLpNorm::L1:
        {
          const double dt = std::abs(error_vec[1].first - error_vec[0].first);
          for(const auto &[time, err] : error_vec)
          {
            error += dt * err;
          }
          break;
        }
        case Parameters::MMS::TimeLpNorm::L2:
        {
          DEAL_II_NOT_IMPLEMENTED();
          break;
        }
        case Parameters::MMS::TimeLpNorm::Linfty:
        {
          for (const auto &[time, err] : error_vec)
            error = std::max(error, err);
          break;
        }
      }

      // Add time Lp norm to error table
      error_table.add_value(key, error);
    }
  }

  /**
   * Clear the t - error(t) table, e.g. in between two convergence steps.
   */
  void clear_error_history()
  {
    for (auto &[key, error_vec] : unsteady_errors)
      error_vec.clear();
  }

  template <int dim>
  void compute_rates()
  {
    for (const auto &key : ordered_keys)
    {
      if (mms_param.type == Parameters::MMS::Type::space ||
          mms_param.type == Parameters::MMS::Type::spacetime)
        error_table.evaluate_convergence_rates(
          key, "n_elm", ConvergenceTable::reduction_rate_log2, dim);
      if (mms_param.type == Parameters::MMS::Type::time)
        error_table.evaluate_convergence_rates(
          key, "dt", ConvergenceTable::reduction_rate_log2, 1);
      error_table.set_precision(key, 4);
      error_table.set_scientific(key, true);
    }
  }

  void write_rates()
  {
    // for(const auto &[field, errors]: unsteady_errors)
    // {
    //   std::cout << "Errors for " << field << std::endl;
    //   for(const auto &[t,e] : errors)
    //     std::cout << t << " : " << e << std::endl;
    // }
    error_table.write_text(std::cout);
  }

public:
  const Parameters::MMS             &mms_param;
  const Parameters::TimeIntegration &time_param;
  bool is_steady;

  // For unsteady problems: keep the spatial errors at all time steps
  // For each field, a vector of (t, error(t)) pairs.
  std::map<std::string, std::vector<std::pair<double, double>>> unsteady_errors;

  ConvergenceTable error_table;

  // Use vector of keys to maintain prescribed errors order
  std::vector<std::string>                       ordered_keys;
  std::map<std::string, std::unique_ptr<double>> domain_errors;
};

#endif