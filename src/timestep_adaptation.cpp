
#include <components_ordering.h>
#include <parameters.h>
#include <timestep_adaptation.h>

#include <algorithm>

DeclExceptionMsg(ExcDataNotAttached,
                 "You are trying to use a BDFErrorEstimator which has "
                 "uninitialized data. Call attach_data_to_error_estimator() "
                 "from the TimeHandler before using the error estimator.");

BDFErrorEstimator::BDFErrorEstimator(
  const Parameters::TimeIntegration &time_parameters,
  const TimeHandler                 &time_handler)
  : time_parameters(time_parameters)
  , n_previous_solutions(time_handler.n_previous_solutions + 1)
  , simulation_times(time_handler.simulation_times.size() + 1)
  , save_full_error_estimator(
      time_parameters.adaptation.compute_error_on_estimator)
{
  if (time_handler.scheme == Parameters::TimeIntegration::Scheme::BDF1)
    bdf_order = 1;
  else if (time_handler.scheme == Parameters::TimeIntegration::Scheme::BDF2)
    bdf_order = 2;
  else
    DEAL_II_NOT_IMPLEMENTED();
}

void BDFErrorEstimator::attach_data(
  const ComponentOrdering          &ordering,
  const IndexSet                   &locally_relevant_dofs,
  const std::vector<unsigned char> &dofs_to_component)
{
  this->ordering              = &ordering;
  this->locally_relevant_dofs = &locally_relevant_dofs;
  this->dofs_to_component     = &dofs_to_component;
}

void BDFErrorEstimator::advance(const TimeHandler &time_handler)
{
  // Rotate the times
  if (!time_handler.rolledback_step)
    for (unsigned int i = n_previous_solutions; i >= 1; --i)
      simulation_times[i] = simulation_times[i - 1];
  simulation_times[0] = time_handler.simulation_times[0];
}

void BDFErrorEstimator::rotate_additional_solution(
  const LA::ParVectorType &solution)
{
  // This reinits additional_solution the first time this function is called,
  // thanks to the crafty operator= for PETSc vectors (petsc_vector.h).
  additional_solution = solution;
}

void BDFErrorEstimator::compute_error_estimator(
  const TimeHandler                    &time_handler,
  const LA::ParVectorType              &present_solution,
  const std::vector<LA::ParVectorType> &previous_solutions)
{
  // Check that attach_data() was called
  Assert(ordering, ExcDataNotAttached());
  Assert(locally_relevant_dofs, ExcDataNotAttached());
  Assert(dofs_to_component, ExcDataNotAttached());

  Assert(additional_solution.size() > 0, ExcInternalError());
  Assert(dofs_to_component->size() > 0,
         ExcMessage("The vector dofs_to_component should be filled before "
                    "entering this function."));

  // If the full vector of error estimates for each dof must be stored and
  // it has not yet been initialized, do it now
  if (error_estimator.size() == 0)
    if (save_full_error_estimator)
    {
      locally_owned_elements = present_solution.locally_owned_elements();
      error_estimator.reinit(locally_owned_elements,
                             *locally_relevant_dofs,
                             present_solution.get_mpi_communicator());
      fully_distributed_error_estimator.reinit(
        locally_owned_elements, present_solution.get_mpi_communicator());
    }

  // Set the handled variables the first time we enter here
  if (handled_variables.empty())
    for (const auto var : SolverInfo::variable_types)
      if (ordering->has_variable(var))
      {
        handled_variables.push_back(var);
        max_error[var] = 0.;
      }

  Assert(!handled_variables.empty(), ExcInternalError());

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
  // FIXME: Probalby useless if taking the abs below
  coeff *= (bdf_order % 2 == 0) ? -1. : 1.;

  // Current time step
  const double h = simulation_times[0] - simulation_times[1];

  // Compute trunaction error for each relevant dof
  for (const auto &dof : *locally_relevant_dofs)
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

    if (save_full_error_estimator)
      if (locally_owned_elements.is_element(dof))
        fully_distributed_error_estimator[dof] = error;

    // Get dof component, then variable associated with this component
    const auto comp =
      (*dofs_to_component)[locally_relevant_dofs->index_within_set(dof)];
    const auto var    = ordering->component_to_variable_type(comp);
    max_error.at(var) = std::max(max_error.at(var), error);
  }

  if (save_full_error_estimator)
  {
    fully_distributed_error_estimator.compress(VectorOperation::insert);
    error_estimator = fully_distributed_error_estimator;
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
}

const std::map<SolverInfo::VariableType, double> &
BDFErrorEstimator::get_max_errors() const
{
  Assert(!max_error.empty(), ExcInternalError());
  return max_error;
}

double BDFErrorEstimator::get_next_timestep(const double current_timestep) const
{
  Assert(!handled_variables.empty(), ExcInternalError());
  Assert(!max_error.empty(), ExcInternalError());

  std::map<SolverInfo::VariableType, double> target_steps;

  // Compute next time step ratios
  for (const auto var : handled_variables)
  {
    const double eps = time_parameters.adaptation.target_error.at(var);
    const double ratio =
      std::pow(eps / std::max(1e-15, max_error.at(var)), 1. / (bdf_order + 1));
    target_steps[var] = current_timestep * ratio;
  }

  Assert(!target_steps.empty(), ExcInternalError());

  // Get key and value of minimum timestep
  const auto it = std::min_element(target_steps.begin(),
                                   target_steps.end(),
                                   [](const auto &a, const auto &b) {
                                     return a.second < b.second;
                                   });

  // Return the smallest prescribed timestep
  return it->second;
}

const LA::ParVectorType &
BDFErrorEstimator::get_error_estimator_as_solution() const
{
  Assert(save_full_error_estimator, ExcInternalError());
  return fully_distributed_error_estimator;
}
