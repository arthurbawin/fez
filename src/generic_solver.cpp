
#include <generic_solver.h>
#include <time_handler.h>

template <typename VectorType>
GenericSolver<VectorType>::GenericSolver(
  const Parameters::Output          &output_param,
  const Parameters::NonLinearSolver &nonlinear_solver_param,
  const Parameters::Timer           &timer_param,
  const Parameters::Mesh            &mesh_param,
  const Parameters::TimeIntegration &time_param,
  const Parameters::MMS             &mms_param,
  const SolverInfo::SolverType       solver_type)
  : mpi_communicator(MPI_COMM_WORLD)
  , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
  , mpi_size(Utilities::MPI::n_mpi_processes(mpi_communicator))
  , pcout(std::cout, (mpi_rank == 0))
  , computing_timer(mpi_communicator,
                    pcout,
                    TimerOutput::summary,
                    TimerOutput::wall_times)
  , solver_type(solver_type)
  , output_param(output_param)
  , mesh_param(mesh_param)
  , time_param(time_param)
  , mms_param(mms_param)
{
  // Disable timer if needed
  if (!timer_param.enable_timer)
    computing_timer.disable_output();

  // Create the nonlinear solver (Newton-Raphson solver)
  nonlinear_solver =
    std::make_unique<NewtonSolver<VectorType>>(nonlinear_solver_param, this);

  // Create the error handlers
  for (auto norm : mms_param.norms_to_compute)
    error_handlers.emplace(
      norm, ErrorHandler(this->mesh_param, this->mms_param, this->time_param));
}

template <typename VectorType>
template <int dim>
void GenericSolver<VectorType>::run_convergence_loop()
{
  const std::string initial_mesh_file = mesh_param.filename;
  const bool        metric_based_adaptation =
    mesh_param.adaptation.with_metric_based_adaptation();

  for (unsigned int i_conv = 0; i_conv < mms_param.n_convergence; ++i_conv)
  {
    const bool is_last_step = i_conv == mms_param.n_convergence - 1;

    mms_param.current_step = i_conv;
    for (auto &[norm, handler] : error_handlers)
      handler.clear_error_history();

    // If a manufactured solution test is run, bypass the given mesh file
    // and run the i_conv-th prescribed mms mesh
    // Update mesh file. Keep previous mesh file only if it is a time
    // convergence study.
    bool update_mesh = mms_param.type == Parameters::MMS::Type::space ||
                       mms_param.type == Parameters::MMS::Type::spacetime ||
                       (mms_param.type == Parameters::MMS::Type::time &&
                        mms_param.use_space_convergence_mesh && i_conv == 0);

    if (update_mesh)
    {
      // This change is accounted for in the reset() function of each
      // derived solver
      mms_param.mesh_suffix = i_conv;

      // If this is a time convergence study using an indexed mesh used
      // for space convergence studies, override the mesh_suffix with
      // the specified mesh index
      if (mms_param.type == Parameters::MMS::Type::time &&
          mms_param.use_space_convergence_mesh)
        mms_param.mesh_suffix = mms_param.spatial_mesh_index;

      if (mms_param.run_only_step >= 0)
      {
        mms_param.current_step = mms_param.run_only_step;

        // Set the mesh suffix to the only run step for space studies
        // For time studies, the "run only" only affects the time step.
        if (mms_param.type == Parameters::MMS::Type::space ||
            mms_param.type == Parameters::MMS::Type::spacetime)
          mms_param.mesh_suffix = mms_param.run_only_step;
      }

      if (metric_based_adaptation)
      {
        if (i_conv > 0)
        {
          // Double the target number of mesh vertices
          // FIXME: when GenericSolver is templatized over dim and stores the
          // full parameter structure, double the target number of vertices in
          // each metric field instead.
          mms_param.n_target_vertices *= 2;
        }

        // Restart from the initial mesh. Alternatively we could also restart
        // from the final adapted mesh from the previous convergence step.
        mesh_param.filename = mms_param.mesh_prefix + ".msh";
        pcout << "Mesh file was changed to " << mesh_param.filename
              << std::endl;
      }
      else if (!mms_param.use_deal_ii_cube_mesh)
      {
        mms_param.override_mesh_filename(mesh_param, mms_param.mesh_suffix);
        pcout << "Convergence test with manufactured solution:" << std::endl;
        pcout << "Mesh file was changed to " << mesh_param.filename
              << std::endl;
      }
    }

    // Update time step starting at second iteration
    bool update_time_step =
      (i_conv > 0) && (mms_param.type == Parameters::MMS::Type::time ||
                       mms_param.type == Parameters::MMS::Type::spacetime);

    if (update_time_step)
    {
      // This change is accounted for in the reset() function of each
      // derived solver
      time_param.dt *= mms_param.time_step_reduction_factor;

      if (time_param.adaptation.enable)
        for (auto &[variable, error] : time_param.adaptation.target_error)
          error /= 2.;
    }

    // If mesh adaptation with a Riemannian metric is enabled, perform the
    // required number of fixed-point iterations and compute the error on the
    // last solution.
    const unsigned int n_fixed_point_iterations =
      metric_based_adaptation ? mesh_param.adaptation.metric.n_fixed_point : 1;

    for (unsigned int ifp = 0; ifp < n_fixed_point_iterations; ++ifp)
    {
      mesh_param.adaptation.metric.current_fixed_point_iteration = ifp;

      if (metric_based_adaptation)
        pcout
          << "Convergence test with mesh adaptation - Fixed-point iteration "
          << ifp + 1 << "/" << n_fixed_point_iterations << std::endl;
      if (ifp > 0)
      {
        // Set the updated mesh file for this fixed-point iteration
        mesh_param.filename =
          output_param.output_dir + mesh_param.adaptation.adapt_dir +
          mesh_param.adaptation.adapted_mesh_extension + ".msh";
        pcout << "Mesh file was changed to " << mesh_param.filename
              << std::endl;
      }

      this->run();
    }

    // If unsteady, compute the Lp time norm for this convergence step
    if (time_param.scheme != Parameters::TimeIntegration::Scheme::stationary)
      for (auto &[norm, handler] : error_handlers)
      {
        handler.compute_temporal_error();

        // Print the errors at all timesteps if required
        if (mms_param.print_unsteady_errors_to_console)
          handler.write_errors();
        if (mms_param.print_unsteady_errors_to_file)
        {
          std::ofstream outfile(output_param.output_dir +
                                mms_param.unsteady_errors_file_prefix + ".txt");
          handler.write_errors(outfile);
        }
      }

    // If requested, compute and write convergence rates as soon as available
    if (is_last_step || !mms_param.compute_rates_only_at_end)
    {
      for (auto &[norm, handler] : error_handlers)
        handler.template compute_rates<dim>();
      if (mpi_rank == 0)
      {
        for (auto &[norm, handler] : error_handlers)
        {
          const std::string norm_str =
            Patterns::Tools::Convert<VectorTools::NormType>::to_string(norm);
          std::cout << std::endl;
          std::cout << norm_str << std::endl;
          handler.write_rates();

          if (mms_param.write_convergence_table_to_file)
          {
            std::ofstream outfile(output_param.output_dir +
                                  mms_param.convergence_file_prefix + "_" +
                                  norm_str + ".txt");
            handler.write_rates(outfile);
          }
        }
      }
    }

    if (mms_param.run_only_step >= 0)
      break;
  }
}

template <typename VectorType>
void GenericSolver<VectorType>::run_fixed_point_loop()
{
  Assert(mesh_param.adaptation.enable &&
           mesh_param.adaptation.strategy ==
             Parameters::Mesh::Adaptation::Strategy::RiemannianMetric,
         ExcMessage("This run function is intended for simulations with mesh "
                    "adaptation with a Riemannian metric only."));

  const unsigned int nfp = mesh_param.adaptation.metric.n_fixed_point;

  for (unsigned int ifp = 0; ifp < nfp; ++ifp)
  {
    mesh_param.adaptation.metric.current_fixed_point_iteration = ifp;

    pcout << "Run with metric-based mesh adaptation - Fixed-point iteration "
          << ifp + 1 << "/" << nfp << std::endl;

    // if (ifp > 0)
    // {
    //   // Update the mesh file
    //   mesh_param.filename =
    //     output_param.output_dir + mesh_param.adaptation.adapt_dir +
    //     mesh_param.adaptation.adapted_mesh_extension + ".msh";
    //   pcout << "Mesh file was changed to " << mesh_param.filename <<
    //   std::endl;
    // }

    this->run();
  }
}

// template <typename VectorType>
// void GenericSolver<VectorType>::run_transient_fixed_point_loop()
// {
//   Assert(mesh_param.adaptation.enable &&
//            mesh_param.adaptation.strategy ==
//              Parameters::Mesh::Adaptation::Strategy::RiemannianMetric,
//          ExcMessage("This run function is intended for simulations with mesh
//          "
//                     "adaptation with a Riemannian metric only."));

//   const unsigned int nfp = mesh_param.adaptation.metric.n_fixed_point;

//   for (unsigned int ifp = 0; ifp < nfp; ++ifp)
//   {
//     mesh_param.adaptation.metric.current_fixed_point_iteration = ifp;

//     pcout << "Run with metric-based mesh adaptation - Fixed-point iteration "
//           << ifp + 1 << "/" << nfp << std::endl;

//     // if (ifp > 0)
//     // {
//     //   // Update the mesh file
//     //   mesh_param.filename =
//     //     output_param.output_dir + mesh_param.adaptation.adapt_dir +
//     //     mesh_param.adaptation.adapted_mesh_extension + ".msh";
//     //   pcout << "Mesh file was changed to " << mesh_param.filename <<
//     //   std::endl;
//     // }

//     this->run();
//   }
// }

template <typename VectorType>
void GenericSolver<VectorType>::solve_nonlinear_problem(
  const TimeHandler &time_handler)
{
  nonlinear_solver->solve(time_handler);
}

template <typename VectorType>
void GenericSolver<VectorType>::distribute_nonzero_constraints()
{
  const auto &nonzero_constraints = this->get_nonzero_constraints();
  nonzero_constraints.distribute(local_evaluation_point);
}

template <typename VectorType>
void GenericSolver<VectorType>::adapt_mesh()
{
  AssertThrow(false, ExcPureFunctionCalled());
}

template <typename VectorType>
bool GenericSolver<VectorType>::should_compute_errors(
  const TimeHandler &time_handler) const
{
  if (!mms_param.enable)
    return false;

  // FIXME: decide to compute or not the error on the initial condition
  // That would be better for unsteady simulations...
  // Currently it's not done, so the tests should be updated.
  if (time_handler.current_time_iteration_in_interval == 0)
    return false;

  if (mesh_param.adaptation.with_metric_based_adaptation())
    return mesh_param.adaptation.metric.is_last_fixed_point_iteration();

  return true;
}

template <typename VectorType>
bool GenericSolver<VectorType>::should_add_error_reference_data(
  const TimeHandler &time_handler) const
{
  if (!mms_param.enable)
    return false;

  if (mesh_param.adaptation.with_metric_based_adaptation())
  {
    // if (time_param.is_steady())
    // {
    // Steady-state mesh adaptation with Riemannian metric.
    // Add the spatial complexity of the last fixed-point iteration (last
    // adapted mesh)
    return mesh_param.adaptation.metric.is_last_fixed_point_iteration() &&
           time_handler.current_time_interval ==
             time_handler.n_time_intervals - 1;
    // }
    // else
    // {
    //   // Unsteady adaptation with the transient fixed-point method.
    //   // Spatial complexity is the sum of the complexity of all meshes.
    //   // Add this complexity
    // }
  }

  return true;
}

template <typename VectorType>
bool GenericSolver<VectorType>::should_check_weakly_enforced_velocity(
  const TimeHandler &time_handler) const
{
  // Do not check during the postprocessing step on the initial solution, which
  // typically does not respect the constraint (e.g., no-slip condition).
  if (time_handler.current_time_iteration == 0)
    return false;

  /*
   * When applying the exact solution, the fluid velocity will be exact,
   * but the mesh velocity is only precise up to time integration order.
   * So these velocities differ by some power of the time step, rather
   * than the machine epsilon as checked in this function, thus the
   * no-slip is not checked in this case.
   */
  // if (param.debug.apply_exact_solution)
  // return false;

  // Do not check if using BDF2 and starting with the initial condition, as it
  // will generally not respect the no-slip condition.
  if (time_handler.is_starting_step() &&
      time_param.bdfstart ==
        Parameters::TimeIntegration::BDFStart::initial_condition)
    return false;

  // If the simulation was restarted, the first of the previous solutions is
  // actually the present solution itself, since the checkpoint happened after
  // rotating the solutions. In that case, the initial postprocessing will be
  // computed on that "initial" condition, for which, e.g., mesh velocity is
  // not correct until the next time step. Do not check constraint in that case
  // (it was checked before writing the checkpoint anyway).
  if (time_handler.current_time_iteration ==
      time_handler.time_iteration_at_last_restart)
    return false;

  return true;
}

template <typename VectorType>
template <int dim>
bool GenericSolver<VectorType>::should_compute_reconstructions(
  const ParameterReader<dim> &param,
  const TimeHandler          &time_handler) const
{
  if (should_compute_riemannian_metric(param, time_handler))
  {
    Assert(param.bc_data.n_metric_fields > 0, ExcInternalError());
    if (!param.metric_fields[0].multiscale.use_analytical_derivatives)
      // Metric does not use exact derivatives, needs reconstructed derivatives
      return true;
  }

  return false;
}

template <typename VectorType>
template <int dim>
bool GenericSolver<VectorType>::should_compute_riemannian_metric(
  const ParameterReader<dim> &param,
  const TimeHandler          &time_handler) const
{
  if (param.metrics.always_compute)
    return true;

  if (mesh_param.adaptation.with_metric_based_adaptation())
  {
    // For unsteady simulations, compute recovery and metric even for the
    // initial condition. For steady simulations, compute only on the solution,
    // not the initial condition.
    if (time_handler.is_steady())
      // This is simply to avoid computing the metric from the initial
      // condition.
      return time_handler.current_time_iteration_in_interval > 0;
    else
      // FIXME: for transient adaptation, set the frequency at which the metric
      // should be updated. It's not required to evaluate it at each time step.
      return true;
  }

  return false;
}

template <typename VectorType>
const Parameters::TimeIntegration &
GenericSolver<VectorType>::get_time_parameters() const
{
  return time_param;
}

template <typename VectorType>
const ErrorHandler &GenericSolver<VectorType>::get_error_handler(
  const VectorTools::NormType type) const
{
  const std::string type_str =
    Patterns::Tools::Convert<VectorTools::NormType>::to_string(type);
  AssertThrow(error_handlers.count(type) > 0,
              ExcMessage(
                "This solver does not hold an ErrorHandler for the norm " +
                type_str +
                ". Be sure to specify that this norm should be computed in "
                "the Manufactured solution subsection."));
  return error_handlers.at(type);
}

// Explicit instantiation on selected vector types
template class GenericSolver<LA::ParVectorType>;

template void GenericSolver<LA::ParVectorType>::run_convergence_loop<2>();
template void GenericSolver<LA::ParVectorType>::run_convergence_loop<3>();

template bool GenericSolver<LA::ParVectorType>::should_compute_reconstructions(
  const ParameterReader<2> &,
  const TimeHandler &) const;
template bool GenericSolver<LA::ParVectorType>::should_compute_reconstructions(
  const ParameterReader<3> &,
  const TimeHandler &) const;
template bool
GenericSolver<LA::ParVectorType>::should_compute_riemannian_metric(
  const ParameterReader<2> &,
  const TimeHandler &) const;
template bool
GenericSolver<LA::ParVectorType>::should_compute_riemannian_metric(
  const ParameterReader<3> &,
  const TimeHandler &) const;
