#ifndef ERROR_HANDLER_H
#define ERROR_HANDLER_H

#include <parameters.h>

/**
 *
 */
class ErrorHandler
{
public:
  ErrorHandler(const Parameters::MMS &mms_parameters)
    : mms_param(mms_parameters)
  {}

  void create_entry(const std::string &error_name)
  {
    ordered_keys.push_back(error_name);
    domain_errors.insert({error_name, std::make_unique<double>()});
  }

  void add_value(const std::string &error_name, const double error_val)
  {
    AssertThrow(
      domain_errors.count(error_name) == 1,
      ExcMessage(
        "Cannot add error value for field " + error_name +
        " because it does not exist in the error handler. Add it first."));
    *(domain_errors.at(error_name)) = error_val;
    error_table.add_value(error_name, error_val);
  }

  void compute_rates()
  {
    for (const auto &key : ordered_keys)
    {
      error_table.evaluate_convergence_rates(
        key, ConvergenceTable::reduction_rate_log2);
      error_table.set_precision(key, 4);
      error_table.set_scientific(key, true);
    }
  }

  void write_rates() { error_table.write_text(std::cout); }

public:
  const Parameters::MMS &mms_param;

  ConvergenceTable error_table;

  // Use vector of keys to maintain prescribed errors order
  std::vector<std::string>                       ordered_keys;
  std::map<std::string, std::unique_ptr<double>> domain_errors;
};

#endif