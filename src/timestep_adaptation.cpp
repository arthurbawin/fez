
#include <components_ordering.h>
#include <parameters.h>
#include <timestep_adaptation.h>

#include <algorithm>

BDFErrorEstimator::BDFErrorEstimator(
  const Parameters::TimeIntegration &time_parameters,
  const TimeHandler                 &time_handler)
  : time_parameters(time_parameters)
  , n_previous_solutions(time_handler.n_previous_solutions + 1)
  , simulation_times(time_handler.simulation_times.size() + 1)
{
  if (time_handler.scheme == Parameters::TimeIntegration::Scheme::BDF1)
    bdf_order = 1;
  else if (time_handler.scheme == Parameters::TimeIntegration::Scheme::BDF2)
    bdf_order = 2;
  else
    DEAL_II_NOT_IMPLEMENTED();
}

void BDFErrorEstimator::advance(const TimeHandler &time_handler)
{
  // Rotate the times
  for (unsigned int i = n_previous_solutions; i >= 1; --i)
  {
    simulation_times[i] = simulation_times[i - 1];
    // time_steps[i]     = time_steps[i - 1];
  }
  simulation_times[0] = time_handler.simulation_times[0];
}

void BDFErrorEstimator::rotate_additional_solution(
  const LA::ParVectorType &solution)
{
  // This reinits additional_solution the first time this function is called,
  // thanks to the crafty operator= for PETSc vectors (petsc_vector.h).
  additional_solution = solution;
}

double BDFErrorEstimator::compute_next_timestep_from_error_estimator(
  const TimeHandler                    &time_handler,
  const ComponentOrdering              &ordering,
  const LA::ParVectorType              &present_solution,
  const std::vector<LA::ParVectorType> &previous_solutions,
  const IndexSet                       &locally_relevant_dofs,
  const std::vector<unsigned char>     &dofs_to_component)
{
  const unsigned int n_starting_steps = bdf_order - 1;
  if (time_handler.current_time_iteration < bdf_order + 1 + n_starting_steps)
    return time_handler.current_dt;

  Assert(additional_solution.size() > 0, ExcInternalError());
  Assert(dofs_to_component.size() > 0,
         ExcMessage("The vector dofs_to_component should be filled before "
                    "entering this function."));

  // Set the handled variables the first time we enter here
  if (handled_variables.empty())
    for (const auto var : SolverInfo::variable_types)
      if (ordering.has_variable(var))
      {
        handled_variables.push_back(var);
        // std::cout << "Solver has variable " + SolverInfo::to_string(var)
        //           << std::endl;
      }

  Assert(!handled_variables.empty(), ExcInternalError());

  std::map<SolverInfo::VariableType, double> max_error;
  for (const auto var : handled_variables)
    max_error[var] = 0.;

  const unsigned int n_sol = bdf_order + 2;

  AssertDimension(simulation_times.size(), n_sol);
  AssertDimension(previous_solutions.size(), bdf_order);

  std::vector<double> times(n_sol), values(n_sol);
  for (unsigned int i = 0; i < n_sol; ++i)
    times[i] = simulation_times[i];

  // Compute the coefficient (-1)^{p+1} * sum_i BDF_coeff_i * (t_{n+1} -
  // t_{n+1-i})^{p+1} multiplying the divided difference
  const auto &alpha = time_handler.bdf_coefficients;
  double      coeff = 0.;
  for (unsigned int i = 0; i <= bdf_order; ++i)
  {
    const double Hi = simulation_times[0] - simulation_times[i];
    coeff += alpha[i] * std::pow(Hi, bdf_order + 1);
  }
  coeff *= (bdf_order % 2 == 0) ? -1. : 1.;

  // Current time step
  const double h = simulation_times[0] - simulation_times[1];

  for (const auto &dof : locally_relevant_dofs)
  {
    // Solution at times t_{n+1} through t_{n+1-(p+1)}.
    values[0] = present_solution[dof];
    for (unsigned int i = 0; i < n_sol - 2; ++i)
      values[i + 1] = previous_solutions[i][dof];
    values[n_sol - 1] = additional_solution[dof];

    // Compute the divided difference of order p + 1
    double dd = 0.;
    if (time_parameters.scheme == Parameters::TimeIntegration::Scheme::BDF1)
      dd = divided_difference<2>(times, values);
    else if (time_parameters.scheme ==
             Parameters::TimeIntegration::Scheme::BDF2)
      dd = divided_difference<3>(times, values);

    const double error = std::abs(h * coeff * dd);

    // Get dof component, then variable associated with this component
    const auto comp =
      dofs_to_component[locally_relevant_dofs.index_within_set(dof)];
    const auto var    = ordering.component_to_variable_type(comp);
    max_error.at(var) = std::max(max_error.at(var), error);
  }

  // Synchronize the max errors across ranks
  for (const auto var : handled_variables)
  {
    const auto comm   = present_solution.get_mpi_communicator();
    max_error.at(var) = Utilities::MPI::max(max_error.at(var), comm);
    if (Utilities::MPI::this_mpi_process(comm) == 0 &&
        time_parameters.adaptation.verbosity == Parameters::Verbosity::verbose)
      std::cout << "Max error for variable " + SolverInfo::to_string(var)
                << " is " << max_error.at(var) << std::endl;
  }

  // Compute the next time step ratios
  const double                               timestep = time_handler.current_dt;
  std::map<SolverInfo::VariableType, double> next_timesteps;
  for (const auto var : handled_variables)
  {
    const double eps = time_parameters.adaptation.target_error.at(var);
    const double ratio =
      std::pow(eps / std::max(1e-15, max_error.at(var)), 1. / (bdf_order + 1));
    next_timesteps[var] = timestep * ratio;
    // std::cout << "Ratio for " + SolverInfo::to_string(var) << " is " << ratio
    //           << " - Next timestep = " << next_timesteps[var] << std::endl;
  }

  // Get key and value of the minimum timestep
  auto it = std::min_element(next_timesteps.begin(),
                             next_timesteps.end(),
                             [](const auto &a, const auto &b) {
                               return a.second < b.second;
                             });


  // std::cout << "Min next timestep is " << it->second
  //           << " and is due to variable " << SolverInfo::to_string(it->first)
  //           << std::endl;

  // Return the smallest prescribed timestep
  return it->second;
}
