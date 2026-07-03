
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/tria.h>
#include <error_handler.h>
#include <parameter_reader.h>
#include <parameters.h>
#include <time_handler.h>

#include <iomanip>
#include <iostream>

ErrorHandler::ErrorHandler(const Parameters::Mesh            &mesh_param,
                           const Parameters::MMS             &mms_parameters,
                           const Parameters::TimeIntegration &time_parameters)
  : mesh_param(mesh_param)
  , mms_param(mms_parameters)
  , time_param(time_parameters)
  , is_steady(time_parameters.is_steady())
  , with_metric_based_adaptation(
      mesh_param.adaptation.with_metric_based_adaptation())
{
  // Declare the reference data so they always appear first
  // See comments in add_reference_data() below.
  if (is_steady)
  {
    if (with_metric_based_adaptation)
    {
      error_table.declare_column("target_nvrt");
      error_table.declare_column("n_vrt");
    }
    error_table.declare_column("n_elm");
    error_table.declare_column("n_dof");
  }
  else
  {
    if (with_metric_based_adaptation)
    {
      error_table.declare_column("n_intervals");
      error_table.declare_column("target_nvrt");
      error_table.declare_column("n_st");
      error_table.declare_column("n_tot_vrt");
      error_table.declare_column("n_tot_elm");
      error_table.declare_column("n_tot_dof");
      error_table.declare_column("dt");
    }
    else
    {
      error_table.declare_column("n_elm");
      error_table.declare_column("n_dof");
      error_table.declare_column("dt");
    }
  }
}

void ErrorHandler::create_entry(const std::string &field_name)
{
  ordered_field_keys.push_back(field_name);
  domain_errors.insert({field_name, std::make_unique<double>()});

  // FIXME: Reserve vectors with an estimate on the number of time steps?
  unsteady_errors[field_name].clear();
}

template <int dim>
void ErrorHandler::add_reference_data(
  const TimeHandler                  &time_handler,
  const TransientFixedPointData<dim> &transient_fixed_point_data,
  const Triangulation<dim>           &triangulation,
  const DoFHandler<dim>              &dof_handler,
  const bool                          set_zero_dofs)
{
  const auto n_global_active_cells = triangulation.n_global_active_cells();
  const auto n_dofs                = set_zero_dofs ? 0 : dof_handler.n_dofs();

  Assert(n_global_active_cells > 0,
         ExcMessage("The provided triangulation is empty"));
  Assert(dof_handler.n_dofs() > 0,
         ExcMessage(
           "The provided dof handler does not store any degree of freedom"));

  if (is_steady)
  {
    // Steady-state computation.

    if (with_metric_based_adaptation)
    {
      // Also add the spatial complexity (number of mesh vertices) and the
      // target spatial complexity.
      add_reference_data("target_nvrt", mms_param.n_target_vertices);
      add_reference_data("n_vrt",
                         transient_fixed_point_data.get_sum_of_vertices());
    }

    add_reference_data("n_elm", n_global_active_cells);
    add_reference_data("n_dof", n_dofs);
  }
  else
  {
    // Unsteady convergence study with constant time step.
    // Add initial time step as reference data.
    // The case with adaptive time step is handled in compute_temporal_error(),
    // by adding the total number of time steps when the simulation is finished.

    // FIXME: For adaptive time stepping, add "n_steps" as reference data here,
    // not in another function

    if (with_metric_based_adaptation)
    {
      // Simulation with multiple time subintervals.
      // Convergence is computed with respect to either the total spatial
      // complexity (= sum of vertices across all meshes), or the total space-
      // time complexity (= total spatial complexity * number of time steps).
      // In the case with constant time step, both complexities are available at
      // the beginning of the simulation, since all meshes are available.
      // With adaptive time step, the total space-time complexity must be
      // computed once the simulation is finished.
      add_reference_data("n_intervals",
                         transient_fixed_point_data.get_n_time_intervals());
      add_reference_data("target_nvrt", mms_param.n_target_vertices);
      add_reference_data("n_st",
                         transient_fixed_point_data
                           .get_effective_space_time_complexity(time_handler));
      add_reference_data("n_tot_vrt",
                         transient_fixed_point_data.get_sum_of_vertices());
      add_reference_data("n_tot_elm",
                         transient_fixed_point_data.get_sum_of_active_cells());
      add_reference_data("n_tot_dof",
                         transient_fixed_point_data.get_sum_of_dofs());
      add_time_step(time_handler.initial_dt);
    }
    else
    {
      add_reference_data("n_elm", n_global_active_cells);
      add_reference_data("n_dof", n_dofs);
      add_time_step(time_handler.initial_dt);
    }
  }
}

template void
ErrorHandler::add_reference_data(const TimeHandler &,
                                 const TransientFixedPointData<2> &,
                                 const Triangulation<2> &,
                                 const DoFHandler<2> &,
                                 const bool);
template void
ErrorHandler::add_reference_data(const TimeHandler &,
                                 const TransientFixedPointData<3> &,
                                 const Triangulation<3> &,
                                 const DoFHandler<3> &,
                                 const bool);

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
  error_vec.emplace_back(time, error_val);
}

void ErrorHandler::compute_temporal_error()
{
  for (const auto &key : ordered_field_keys)
  {
    auto              &error_vec = unsteady_errors.at(key);
    const unsigned int n_steps   = error_vec.size();
    AssertThrow(
      n_steps > 0,
      ExcMessage(
        "Cannot compute unsteady error because no time step was computed!"));

    double error = 0.;
    switch (mms_param.time_norm)
    {
      case Parameters::MMS::TimeLpNorm::L1:
      {
        double t_prev = time_param.t_initial;
        for (unsigned int i = 0; i < n_steps; ++i)
        {
          const double t   = error_vec[i].first;
          const double err = error_vec[i].second;
          const double dt  = t - t_prev;
          // FIXME: correct this if computing error at t = 0
          AssertThrow(
            dt > 0,
            ExcInternalError(
              "Computation of the Lp error norm in time must be adapted if "
              "errors are computed for the initial condition."));
          error += dt * err;
          t_prev = t;
        }
        // }
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

template <int dim>
void ErrorHandler::compute_rates()
{
  for (const auto &key : ordered_field_keys)
  {
    if (mms_param.type == Parameters::MMS::Type::space ||
        mms_param.type == Parameters::MMS::Type::spacetime)
    {
      if (with_metric_based_adaptation)
      {
        if (is_steady)
        {
          // Spatial convergence study with metric based mesh adaptation.
          // Compute convergence rate w.r.t. the final spatial complexity
          // (i.e., the number of mesh vertices in the last adapted mesh).
          error_table.evaluate_convergence_rates(
            key, "n_vrt", ConvergenceTable::reduction_rate_log2, dim);
        }
        else
        {
          // Unsteady simulation with metric based mesh adaptation.
          // Compute convergence w.r.t. the total spatial complexity
          // and w.r.t. the total space-time complexity.

          // FIXME: this is computed only w.r.t. spatial complexity for now
          // Should maybe duplicate the columns for which we want to compute
          // rate w.r.t space-time complexity, or set the reference data as a
          // parameter.
          error_table.evaluate_convergence_rates(
            key, "n_tot_vrt", ConvergenceTable::reduction_rate_log2, dim);
        }
      }
      else
        // Default spatial convergence study without mesh adaptation.
        // Compute convergence rate w.r.t. number of mesh elements.
        error_table.evaluate_convergence_rates(
          key, "n_elm", ConvergenceTable::reduction_rate_log2, dim);
    }

    if (mms_param.type == Parameters::MMS::Type::time)
    {
      if (time_param.adaptation.enable)
      {
        // When using an adaptive time step, compute convergence rates based
        // on the total number of time steps
        error_table.evaluate_convergence_rates(
          key, "n_steps", ConvergenceTable::reduction_rate_log2, 1);
      }
      else
      {
        // Without adaptivity, compute rates based on the constant time step
        error_table.evaluate_convergence_rates(
          key, "dt", ConvergenceTable::reduction_rate_log2, 1);
      }
    }

    error_table.set_precision(key, 4);
    error_table.set_scientific(key, true);
  }
}

template void ErrorHandler::compute_rates<2>();
template void ErrorHandler::compute_rates<3>();

void ErrorHandler::write_rates(std::ostream &out) const
{
  error_table.write_text(out);
}
