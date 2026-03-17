
#include <error_handler.h>
#include <parameters.h>

#include <iomanip>
#include <iostream>

ErrorHandler::ErrorHandler(const Parameters::MMS             &mms_parameters,
                           const Parameters::TimeIntegration &time_parameters)
  : mms_param(mms_parameters)
  , time_param(time_parameters)
  , is_steady(time_parameters.scheme ==
              Parameters::TimeIntegration::Scheme::stationary)
{}

void ErrorHandler::create_entry(const std::string &field_name)
{
  ordered_field_keys.push_back(field_name);
  domain_errors.insert({field_name, std::make_unique<double>()});

  // TODO: Reserve vectors with an estimate on the number of time steps?
  // This is easy if only constant time steps are expected for convergence
  // studies, because then the number of time steps is known.
  unsteady_errors[field_name].clear();
}

void ErrorHandler::add_reference_data(const std::string &name,
                                      const unsigned int value)
{
  error_table.add_value(name, value);
}

void ErrorHandler::add_time_step(double time_step)
{
  error_table.add_value("dt", time_step);
}

void ErrorHandler::add_error(const std::string &field_name,
                             const double       error_val,
                             const double       time)
{
  if (is_steady)
    add_steady_error(field_name, error_val);
  else
    add_unsteady_error(field_name, error_val, time);
}

const std::vector<std::pair<double, double>> &
ErrorHandler::get_unsteady_errors(const std::string &field_name) const
{
  AssertThrow(
    unsteady_errors.count(field_name) > 0,
    ExcMessage(
      "You requested the vector of unsteady errors for the field \"" +
      field_name +
      "\", but this ErrorHandler does not store errors for this field."));
  return unsteady_errors.at(field_name);
}

void ErrorHandler::write_errors(std::ostream      &out,
                                const unsigned int precision) const
{
  std::ios::fmtflags old_flags     = out.flags();
  unsigned int       old_precision = out.precision();

  out << std::scientific << std::setprecision(precision) << std::showpos
      << std::endl;
  if (is_steady)
  {
    out << "Steady-state errors :" << std::endl;
    DEAL_II_NOT_IMPLEMENTED();
  }
  else
  {
    out << "Unsteady errors :" << std::endl;
    out << "\tt\t";
    for (const auto &field_name : ordered_field_keys)
      out << field_name << "\t";
    out << std::endl;

    // This is inefficient, the errors are stored as "columns" and we
    // print as rows.
    const auto        &errors_for_time = unsteady_errors.begin()->second;
    const unsigned int n_steps         = errors_for_time.size();
    for (unsigned int i = 0; i < n_steps; ++i)
    {
      out << errors_for_time[i].first << "\t";
      for (const auto &field_name : ordered_field_keys)
      {
        out << unsteady_errors.at(field_name)[i].second << "\t";
      }
      out << std::endl;
    }
  }
  out.precision(old_precision);
  out.flags(old_flags);
}

void ErrorHandler::add_steady_error(const std::string &error_name,
                                    const double       error_val)
{
  AssertThrow(
    domain_errors.count(error_name) == 1,
    ExcMessage(
      "Cannot add steady error value for field " + error_name +
      " because it does not exist in the error handler. Add it first."));
  *(domain_errors.at(error_name)) = error_val;
  error_table.add_value(error_name, error_val);
}

void ErrorHandler::add_unsteady_error(const std::string &error_name,
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

void ErrorHandler::compute_temporal_error()
{
  for (const auto &key : ordered_field_keys)
  {
    auto &error_vec = unsteady_errors.at(key);

    double error = 0.;
    switch (mms_param.time_norm)
    {
      case Parameters::MMS::TimeLpNorm::L1:
      {
        const double dt = std::abs(error_vec[1].first - error_vec[0].first);
        for (const auto &[time, err] : error_vec)
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

  // When using an adaptive time step, add the total number of time steps,
  // used to compute the convergence rates.
  if (time_param.adaptation.enable)
  {
    const unsigned int n_steps = unsteady_errors.begin()->second.size();
    for (const auto &[field, errors] : unsteady_errors)
      AssertDimension(errors.size(), n_steps);
    error_table.add_value("n_steps", n_steps);
  }
}

void ErrorHandler::clear_error_history()
{
  for (auto &[key, error_vec] : unsteady_errors)
    error_vec.clear();
}

void ErrorHandler::write_rates(std::ostream &out)
{
  error_table.write_text(out);
}
