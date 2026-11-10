
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <post_processing_handler.h>
#include <postprocessors_and_evaluators.h>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>

template <int dim>
PostProcessingHandler<dim>::PostProcessingHandler(
  const ComponentOrdering                                 &ordering,
  const ParameterReader<dim>                              &param,
  const Triangulation<dim>                                &triangulation,
  const DoFHandler<dim>                                   &dof_handler,
  const std::vector<std::pair<std::string, unsigned int>> &fields_description)
  : ordering(ordering)
  , post_proc_param(param.postprocessing)
  , output_param(param.output)
  , physical_properties(param.physical_properties)
  , mms_param(param.mms_param)
  , fe_param(param.finite_elements)
  , triangulation(&triangulation, typeid(*this).name())
  , dof_handler(&dof_handler, typeid(*this).name())
  , mpi_communicator(dof_handler.get_mpi_communicator())
  , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
{
  if (output_param.write_results || output_param.skin.write_results)
  {
    solution_names.clear();
    data_component_interpretation.clear();

    for (const auto &[name, n_comp] : fields_description)
      for (unsigned int d = 0; d < n_comp; ++d)
      {
        solution_names.push_back(name);
        data_component_interpretation.push_back(
          n_comp == 1 ?
            DataComponentInterpretation::component_is_scalar :
            DataComponentInterpretation::component_is_part_of_vector);
      }
  }

  this->attach_triangulation_and_dof_handler(triangulation, dof_handler);

  // Create the required DataPostProcessors
  {
    const auto &pp = param.postprocessing;
    using PP       = Parameters::PostProcessing;

    postprocessors.clear();
    if (pp.vorticity.enable and
        pp.vorticity.method == PP::Vorticity::ComputationMethod::discontinuous)
      postprocessors.emplace_back(
        std::make_unique<PostProcessingTools::VorticityPostProcessor<dim>>(
          ordering));
    if (pp.q_criterion.enable and
        pp.q_criterion.method ==
          PP::QCriterion::ComputationMethod::discontinuous)
      postprocessors.emplace_back(
        std::make_unique<PostProcessingTools::QCriterionPostProcessor<dim>>(
          ordering));
  }
}

template <int dim>
void PostProcessingHandler<dim>::attach_triangulation_and_dof_handler(
  const Triangulation<dim> &triangulation,
  const DoFHandler<dim>    &dof_handler)
{
  Assert(&dof_handler.get_triangulation() == &triangulation,
         ExcMessage("The provided triangulation does not match the one "
                    "attached to the provided DoFHandler"));

  // Assign triangulation and dof handler to this object
  this->triangulation = &triangulation;
  this->dof_handler   = &dof_handler;

  // Create new DataOuts
  if (output_param.write_results)
  {
    data_out = std::make_unique<DataOut<dim>>();
    data_out->attach_dof_handler(dof_handler);

    // Write high-order elements if needed
    if (fe_param.mapping_degree > 1)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out->set_flags(flags);
    }
  }

  if (output_param.skin.write_results)
  {
    // build_patches is not (yet) implemented for DataOutFaces in hp context,
    // but at this point the dof_handler might not yet be initialized.
    // The check is done in output_skin_fields instead.
    data_out_skin =
      std::make_unique<PostProcessingTools::DataOutFacesOnBoundary<dim>>(
        triangulation, output_param.skin.boundary_id);
    data_out_skin->attach_dof_handler(dof_handler);

    // Write high-order elements if needed
    if (fe_param.mapping_degree > 1)
    {
      DataOutBase::VtkFlags flags;
      flags.write_higher_order_cells = true;
      data_out_skin->set_flags(flags);
    }
  }

  // Clear the stored cell-based vectors
  subdomains.reinit(0);
  slice_indices.reinit(0);
}

template <int dim>
void PostProcessingHandler<dim>::write_pvd(const PrefixData &prefix_data) const
{
  std::string suffix = "";
  prefix_data.append_to_prefix_or_suffix(output_param, true, suffix);
  suffix += ".pvd";

  if (mpi_rank == 0)
  {
    if (prefix_data.is_prerefinement_step)
    {
      if (!prerefinements_pseudotimes_and_names.empty())
      {
        std::ofstream pvd_output(output_param.output_dir +
                                 output_param.output_prefix + suffix);
        DataOutBase::write_pvd_record(pvd_output,
                                      prerefinements_pseudotimes_and_names);
      }
      if (!prerefinements_pseudotimes_and_names_skin.empty())
      {
        std::ofstream pvd_output(output_param.output_dir +
                                 output_param.skin.output_prefix + suffix);
        DataOutBase::write_pvd_record(
          pvd_output, prerefinements_pseudotimes_and_names_skin);
      }
    }
    else
    {
      if (output_param.write_results)
      {
        std::ofstream pvd_output(output_param.output_dir +
                                 output_param.output_prefix + suffix);
        DataOutBase::write_pvd_record(pvd_output,
                                      visualization_times_and_names);
      }
      if (output_param.skin.write_results)
      {
        std::ofstream pvd_output(output_param.output_dir +
                                 output_param.skin.output_prefix + suffix);
        DataOutBase::write_pvd_record(pvd_output,
                                      visualization_times_and_names_skin);
      }
    }
  }
}

template <int dim>
void PostProcessingHandler<dim>::create_slices()
{
  const std::string &dir = post_proc_param.slices.along_which_axis;

  AssertThrow(dir == "x" || dir == "y" || (dim == 3 && dir == "z"),
              ExcMessage(dim == 2 ?
                           "slicing_direction must be 'x' or 'y' in 2D." :
                           "slicing_direction must be 'x', 'y' or 'z' in 3D."));

  using SliceAxis      = PostProcessingTools::SliceAxis;
  const SliceAxis axis = (dir == "x" ? SliceAxis::x :
                          dir == "y" ? SliceAxis::y :
                                       SliceAxis::z);

  for (const auto id : post_proc_param.slices.boundary_ids)
    PostProcessingTools::set_slice_index_on_boundary<dim>(
      *triangulation, id, post_proc_param.slices.n_slices, axis);

  // Store slice indices as cell-based data.
  // If a face is on the sliced boundary, set its cell slice index to the
  // face user index.
  slice_indices.reinit(triangulation->n_active_cells());
  for (const auto &cell : triangulation->active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() &&
            std::binary_search(post_proc_param.slices.boundary_ids.begin(),
                               post_proc_param.slices.boundary_ids.end(),
                               face->boundary_id()))
        {
          slice_indices[cell->active_cell_index()] = face->user_index();
          break;
        }
}

template <int dim>
void PostProcessingHandler<dim>::clear()
{
  if (data_out)
    data_out->clear_data_vectors();
  if (data_out_skin)
    data_out_skin->clear_data_vectors();
  visualization_times_and_names.clear();
  visualization_times_and_names_skin.clear();
  subdomains.reinit(0);
  slice_indices.reinit(0);
}

template <int dim>
void PostProcessingHandler<dim>::add_force_to_table(
  const Tensor<1, dim>    &forces,
  const TimeHandler       &time_handler,
  TableHandler            &force_table,
  const types::boundary_id boundary_id,
  const unsigned int       i_slice)
{
  // Write forces to table
  std::vector<std::string> dim_str = {"x", "y", "z"};
  force_table.add_value("time", time_handler.current_time);
  force_table.add_value("boundary", boundary_id);
  if (i_slice != numbers::invalid_unsigned_int)
    force_table.add_value("slice", i_slice);
  for (unsigned int d = 0; d < dim; ++d)
  {
    force_table.add_value("F" + dim_str[d], forces[d]);
    force_table.set_precision("F" + dim_str[d],
                              post_proc_param.forces.precision);
    force_table.set_scientific("F" + dim_str[d], true);
  }
}

template <int dim>
void PostProcessingHandler<dim>::add_position_to_table(
  const Tensor<1, dim>    &center_position,
  const TimeHandler       &time_handler,
  TableHandler            &table,
  const types::boundary_id boundary_id)
{
  // Write position to table
  std::vector<std::string> dim_str = {"x", "y", "z"};
  table.add_value("time", time_handler.current_time);
  table.add_value("boundary", boundary_id);
  for (unsigned int d = 0; d < dim; ++d)
  {
    table.add_value(dim_str[d], center_position[d]);
    table.set_precision(dim_str[d],
                        post_proc_param.structure_position.precision);
    table.set_scientific(dim_str[d], true);
  }
}

template <int dim>
template <typename DataType>
void PostProcessingHandler<dim>::add_multiphase_data_to_table(
  const std::array<DataType, 2>                        &data_for_phases,
  const TimeHandler                                    &time_handler,
  TableHandler                                         &table,
  const Parameters::PostProcessing::PostProcessingFile &pp_param)
{
  if constexpr (std::is_same_v<DataType, Tensor<1, dim>>)
  {
    std::vector<std::string> dim_str = {"x", "y", "z"};
    table.add_value("time", time_handler.current_time);
    for (unsigned int i = 0; i < 2; ++i)
      for (unsigned int d = 0; d < dim; ++d)
      {
        std::string key = "phase" + std::to_string(i) + "_" + dim_str[d];
        table.add_value(key, data_for_phases[i][d]);
        table.set_precision(key, pp_param.precision);
        table.set_scientific(key, true);
      }
  }
  else
  {
    table.add_value("time", time_handler.current_time);
    for (unsigned int i = 0; i < 2; ++i)
    {
      std::string key = "phase" + std::to_string(i);
      table.add_value(key, data_for_phases[i]);
      table.set_precision(key, pp_param.precision);
      table.set_scientific(key, true);
    }
  }
}

// Explicit instantiations for dim = 2,3, two phases and double/Tensor<1, dim>
template void PostProcessingHandler<2>::add_multiphase_data_to_table(
  const std::array<Tensor<1, 2>, 2> &,
  const TimeHandler &,
  TableHandler &,
  const Parameters::PostProcessing::PostProcessingFile &);
template void PostProcessingHandler<3>::add_multiphase_data_to_table(
  const std::array<Tensor<1, 3>, 2> &,
  const TimeHandler &,
  TableHandler &,
  const Parameters::PostProcessing::PostProcessingFile &);
template void PostProcessingHandler<2>::add_multiphase_data_to_table(
  const std::array<double, 2> &,
  const TimeHandler &,
  TableHandler &,
  const Parameters::PostProcessing::PostProcessingFile &);
template void PostProcessingHandler<3>::add_multiphase_data_to_table(
  const std::array<double, 2> &,
  const TimeHandler &,
  TableHandler &,
  const Parameters::PostProcessing::PostProcessingFile &);

template <int dim>
void PostProcessingHandler<dim>::write_table(
  std::ostream                                         &out,
  const TableHandler                                   &table,
  const Parameters::PostProcessing::PostProcessingFile &postproc_file) const
{
  if (mpi_rank == 0)
  {
    out << std::scientific << std::setprecision(postproc_file.precision);
    table.write_text(out);
  }
}

template <int dim>
void PostProcessingHandler<dim>::write_forces(std::ostream &out) const
{
  write_table(out, forces_table, post_proc_param.forces);
}

template <int dim>
void PostProcessingHandler<dim>::write_structure_mean_position(
  std::ostream &out) const
{
  write_table(out,
              structure_mean_position_table,
              post_proc_param.structure_position);
}

template <int dim>
std::unique_ptr<PostProcessingTools::PostprocessorAtDofBase<dim>>
PostProcessingHandler<dim>::create_field_postprocessor(
  const PostProcessingTools::PostprocessorAtDofTypes type,
  const ParameterReader<dim>                        &param,
  const Mapping<dim>                                &mapping,
  const Quadrature<dim>                             &cell_quadrature,
  const bool                                         with_moving_mesh)
{
  using namespace PostProcessingTools;
  using Vorticity  = Parameters::PostProcessing::Vorticity;
  using QCriterion = Parameters::PostProcessing::QCriterion;

  switch (type)
  {
    case PostprocessorAtDofTypes::vorticity:
      switch (param.postprocessing.vorticity.method)
      {
        case Vorticity::ComputationMethod::discontinuous:
          // Nothing to do: the DataPostprocessor was created in the
          // constructor.
          return nullptr;
        case Vorticity::ComputationMethod::l2_projection:
          return std::make_unique<
            FieldPostprocessorGenerator<dim, VorticityEvaluator, L2Projection>>(
            param,
            ordering,
            mapping,
            *dof_handler,
            cell_quadrature,
            with_moving_mesh);
        case Vorticity::ComputationMethod::weighted_average:
          return std::make_unique<
            FieldPostprocessorGenerator<dim,
                                        VorticityEvaluator,
                                        WeightedAverage>>(param,
                                                          ordering,
                                                          mapping,
                                                          *dof_handler,
                                                          cell_quadrature,
                                                          with_moving_mesh);
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
    case PostprocessorAtDofTypes::q_criterion:
      switch (param.postprocessing.q_criterion.method)
      {
        case QCriterion::ComputationMethod::discontinuous:
          // Nothing to do: the DataPostprocessor was created in the
          // constructor.
          return nullptr;
        case QCriterion::ComputationMethod::l2_projection:
          return std::make_unique<
            FieldPostprocessorGenerator<dim,
                                        QCriterionEvaluator,
                                        L2Projection>>(param,
                                                       ordering,
                                                       mapping,
                                                       *dof_handler,
                                                       cell_quadrature,
                                                       with_moving_mesh);
        case QCriterion::ComputationMethod::weighted_average:
          return std::make_unique<
            FieldPostprocessorGenerator<dim,
                                        QCriterionEvaluator,
                                        WeightedAverage>>(param,
                                                          ordering,
                                                          mapping,
                                                          *dof_handler,
                                                          cell_quadrature,
                                                          with_moving_mesh);
        default:
          DEAL_II_NOT_IMPLEMENTED();
      }
    case PostprocessorAtDofTypes::mesh_velocity:
      return std::make_unique<MeshVelocityPostprocessor<dim>>(
        ordering, param, mapping, *dof_handler, cell_quadrature);
    default:
      DEAL_II_NOT_IMPLEMENTED();
  }
  DEAL_II_ASSERT_UNREACHABLE();
  return nullptr;
}

template <int dim>
void PostProcessingHandler<dim>::create_field_postprocessors(
  const ParameterReader<dim> &param,
  const Mapping<dim>         &mapping,
  const Quadrature<dim>      &cell_quadrature,
  const bool                  with_moving_mesh)
{
  for (const auto &[type, postprocessor_param_ptr] :
       param.postprocessing.field_postprocessors)
    if (postprocessor_param_ptr->enable)
      field_postprocessors[type] = create_field_postprocessor(
        type, param, mapping, cell_quadrature, with_moving_mesh);
}

template <int dim>
void PostProcessingHandler<dim>::process_chns_integrals(
  const PostProcessingTools::CHNSIntegralData &data,
  const TimeHandler                           &time_handler)
{
  using I       = PostProcessingTools::CHNSIntegralData;
  const auto &v = data.sums;
  if (!chns_reference_initialized)
  {
    chns_reference_initialized = true;
    chns_reference_time        = time_handler.current_time;
    chns_initial_mass          = v[I::AffineMass];
    chns_initial_phase         = v[I::PhaseIntegral];
    chns_initial_volume        = v[I::Volume];
    chns_initial_energy        = data.total_energy();
  }
  const bool process_domain =
    chns_last_iteration != time_handler.current_time_iteration;
  const bool process_phases = needs_chns_phase_energy(time_handler);
  if (!process_domain && !process_phases)
    return;
  if (process_domain)
    chns_last_iteration = time_handler.current_time_iteration;
  if (process_phases)
    chns_phase_energy_last_iteration = time_handler.current_time_iteration;

  const double nan = std::numeric_limits<double>::quiet_NaN();

  // Only rank 0 writes; propagate its errors so that every rank throws.
  std::string error;
  if (mpi_rank == 0)
  {
    try
    {
      const auto add = [](TableHandler      &table,
                          const std::string &key,
                          const double       value,
                          const unsigned int precision) {
        table.add_value(key, value);
        table.set_precision(key, precision);
        table.set_scientific(key, true);
      };
      const auto add_time = [&](TableHandler &table, const auto &param) {
        add(table, "time", time_handler.current_time, param.precision);
        table.add_value("iteration", time_handler.current_time_iteration);
      };
      // Print "<label> at t=<time>: <details>" if verbose, and write the table.
      const auto output = [&](const TableHandler &table,
                              const auto         &param,
                              const std::string  &label,
                              const auto         &write_details) {
        if (param.verbosity == Parameters::Verbosity::verbose)
        {
          std::ostringstream line;
          line << std::scientific << std::setprecision(param.precision) << label
               << " at t=" << time_handler.current_time << ": ";
          write_details(line);
          std::cout << line.str() << '\n';
        }
        if (should_output_postprocessing(time_handler, param))
        {
          std::filesystem::create_directories(output_param.output_dir);
          const std::string filename =
            output_param.output_dir + param.output_prefix + ".txt";
          std::ofstream stream(filename);
          AssertThrow(stream.good(), ExcMessage("Cannot write " + filename));
          write_table(stream, table, param);
        }
      };

      const auto &conservation = post_proc_param.chns_conservation;
      if (conservation.enable && process_domain)
      {
        auto &table = chns_conservation_table;
        add_time(table, conservation);
        add(table,
            "reference_time",
            chns_reference_time,
            conservation.precision);
        const auto entry = [&](const std::string &key, const double value) {
          add(table, key, value, conservation.precision);
        };
        entry("domain_volume", v[I::Volume]);
        entry("mass_affine", v[I::AffineMass]);
        entry("phase_integral", v[I::PhaseIntegral]);
        entry("mass_positive", v[I::PositiveMass]);
        entry("mass_change_from_reference",
              v[I::AffineMass] - chns_initial_mass);
        entry("mass_relative_change",
              chns_initial_mass != 0. ?
                (v[I::AffineMass] - chns_initial_mass) / chns_initial_mass :
                nan);
        entry("phase_change_from_reference",
              v[I::PhaseIntegral] - chns_initial_phase);
        entry("phase_volume_relative_change",
              chns_initial_volume > 0. ?
                (v[I::PhaseIntegral] - chns_initial_phase) /
                  chns_initial_volume :
                nan);
        entry("mass_step_change", v[I::AffineMass] - v[I::OldAffineMass]);
        entry("phase_step_change",
              v[I::PhaseIntegral] - v[I::OldPhaseIntegral]);
        output(table, conservation, "CHNS conservation", [&](auto &line) {
          line << "mass=" << v[I::AffineMass]
               << ", phase=" << v[I::PhaseIntegral];
        });
      }

      const auto &energy = post_proc_param.chns_energy;
      if (energy.enable && process_domain)
      {
        auto &table = chns_energy_table;
        add_time(table, energy);
        add(table, "reference_time", chns_reference_time, energy.precision);
        const auto entry = [&](const std::string &key, const double value) {
          add(table, key, value, energy.precision);
        };
        entry("kinetic_energy", v[I::KineticEnergy]);
        entry("bulk_energy", v[I::BulkEnergy]);
        entry("gradient_energy", v[I::GradientEnergy]);
        entry("interfacial_energy", v[I::InterfacialEnergy]);
        entry("gravitational_energy", v[I::GravitationalEnergy]);
        entry("total_energy", data.total_energy());
        entry("energy_change_from_reference",
              data.total_energy() - chns_initial_energy);
        entry("viscous_dissipation", v[I::ViscousDissipation]);
        entry("diffusive_dissipation", v[I::DiffusiveDissipation]);

        const auto &coefficients = time_handler.get_bdf_coefficients();
        const bool  snapshot     = time_handler.current_time_iteration == 0 ||
                              time_handler.current_time_iteration ==
                                time_handler.time_iteration_at_last_restart;
        const bool bdf2 =
          !snapshot && coefficients.size() == 3 && coefficients[2] != 0.;
        const double dt = snapshot ? 0. : time_handler.get_current_timestep();
        const bool   constant_bdf2 =
          bdf2 && std::abs(dt * coefficients[0] - 1.5) < 1.e-12 &&
          std::abs(dt * coefficients[1] + 2.) < 1.e-12 &&
          std::abs(dt * coefficients[2] - 0.5) < 1.e-12;
        const double dissipation =
          v[I::ViscousDissipation] + v[I::DiffusiveDissipation];
        // E^{n+1} - E^n + dt (viscous + diffusive dissipation). With BDF1,
        // adding the numerical dissipation closes the energy identity.
        const double defect =
          data.total_energy() - v[I::OldEnergy] + dt * dissipation;
        entry("effective_bdf_order", snapshot ? 0. : (bdf2 ? 2. : 1.));
        entry("physical_energy_increment_plus_dissipation", defect);
        entry("energy_balance_residual",
              bdf2 ? nan : defect + v[I::NumericalEnergyLoss]);
        entry("temporal_energy_balance_residual",
              v[I::KineticTemporalPower] + v[I::BulkTemporalPower] +
                v[I::GradientTemporalPower] + v[I::GravitationalTemporalPower] +
                (snapshot ? 0. : dissipation));
        entry("bdf2_modified_kinetic_energy",
              bdf2 ? v[I::ModifiedKineticEnergy] : nan);
        entry("bdf2_modified_gradient_energy",
              bdf2 ? v[I::ModifiedGradientEnergy] : nan);
        entry("bdf2_constant_step", constant_bdf2 ? 1. : 0.);
        entry("bdf2_kinetic_remainder",
              constant_bdf2 ? v[I::BDF2KineticRemainder] : nan);
        entry("bdf2_gradient_remainder",
              constant_bdf2 ? v[I::BDF2GradientRemainder] : nan);
        entry("bdf2_kinetic_identity_residual",
              constant_bdf2 ? v[I::BDF2KineticIdentityResidual] : nan);
        entry("bdf2_gradient_identity_residual",
              constant_bdf2 ? v[I::BDF2GradientIdentityResidual] : nan);
        output(table, energy, "CHNS energy", [&](auto &line) {
          line << "total energy=" << data.total_energy();
        });
      }

      if (process_phases)
      {
        using E                  = PostProcessingTools::CHNSEnergyComponents;
        const auto &phase_energy = post_proc_param.chns_phase_energy;
        auto       &table        = chns_phase_energy_table;
        add_time(table, phase_energy);
        for (unsigned int phase = 0; phase < data.phase_energy.size(); ++phase)
        {
          const auto       &e      = data.phase_energy[phase];
          const std::string suffix = "_phase_" + std::to_string(phase);
          const auto entry = [&](const std::string &key, const double value) {
            add(table, key + suffix, value, phase_energy.precision);
          };
          entry("kinetic_energy", e.values[E::Kinetic]);
          entry("bulk_energy", e.values[E::Bulk]);
          entry("gradient_energy", e.values[E::Gradient]);
          entry("interfacial_energy", e.interfacial_energy());
          entry("gravitational_energy", e.values[E::Gravitational]);
          entry("total_energy", e.total_energy());
        }
        output(table, phase_energy, "CHNS phase energy", [&](auto &line) {
          line << "total energy=" << data.total_energy();
          for (unsigned int phase = 0; phase < data.phase_energy.size();
               ++phase)
            line << ", phase " << phase << '='
                 << data.phase_energy[phase].total_energy();
        });
      }
    }
    catch (const std::exception &exception)
    {
      error = exception.what();
    }
  }
  error = Utilities::MPI::broadcast(mpi_communicator, error, 0);
  AssertThrow(error.empty(), ExcMessage(error));
}

template <int dim>
void PostProcessingHandler<dim>::save_checkpoint_state(
  const std::string &checkpoint_prefix) const
{
  std::string error;
  if (mpi_rank == 0)
    try
    {
      const std::string filename = checkpoint_prefix + ".postprocessing";
      std::ofstream     stream(filename, std::ios::binary);
      AssertThrow(stream.good(), ExcMessage("Cannot write " + filename));
      stream << "FEZ_POSTPROCESSING_V1\n";
      // Binary archives preserve the NaN entries of the BDF diagnostics.
      boost::archive::binary_oarchive archive(stream);
      archive << chns_reference_initialized << chns_reference_time
              << chns_initial_mass << chns_initial_phase << chns_initial_volume
              << chns_initial_energy << chns_last_iteration
              << chns_conservation_table << chns_energy_table
              << chns_phase_energy_last_iteration << chns_phase_energy_table;
    }
    catch (const std::exception &exception)
    {
      error = exception.what();
    }
  error = Utilities::MPI::broadcast(mpi_communicator, error, 0);
  AssertThrow(error.empty(), ExcMessage(error));
}

template <int dim>
void PostProcessingHandler<dim>::load_checkpoint_state(
  const std::string &checkpoint_prefix)
{
  std::string error;
  if (mpi_rank == 0)
    try
    {
      const std::string filename = checkpoint_prefix + ".postprocessing";
      std::ifstream     stream(filename, std::ios::binary);
      // Checkpoints written without a sidecar start a new reference here.
      if (stream)
      {
        std::string version;
        std::getline(stream, version);
        AssertThrow(version == "FEZ_POSTPROCESSING_V1",
                    ExcMessage(
                      "Unsupported postprocessing checkpoint format in " +
                      filename));
        boost::archive::binary_iarchive archive(stream);
        archive >> chns_reference_initialized >> chns_reference_time >>
          chns_initial_mass >> chns_initial_phase >> chns_initial_volume >>
          chns_initial_energy >> chns_last_iteration >>
          chns_conservation_table >> chns_energy_table >>
          chns_phase_energy_last_iteration >> chns_phase_energy_table;
      }
    }
    catch (const std::exception &exception)
    {
      error = exception.what();
    }
  error = Utilities::MPI::broadcast(mpi_communicator, error, 0);
  AssertThrow(error.empty(), ExcMessage(error));
  chns_reference_initialized =
    Utilities::MPI::broadcast(mpi_communicator, chns_reference_initialized, 0);
  chns_last_iteration =
    Utilities::MPI::broadcast(mpi_communicator, chns_last_iteration, 0);
  chns_phase_energy_last_iteration =
    Utilities::MPI::broadcast(mpi_communicator,
                              chns_phase_energy_last_iteration,
                              0);
  std::vector<double> reference = {chns_reference_time,
                                   chns_initial_mass,
                                   chns_initial_phase,
                                   chns_initial_volume,
                                   chns_initial_energy};
  reference = Utilities::MPI::broadcast(mpi_communicator, reference, 0);
  chns_reference_time = reference[0];
  chns_initial_mass   = reference[1];
  chns_initial_phase  = reference[2];
  chns_initial_volume = reference[3];
  chns_initial_energy = reference[4];
}

template class PostProcessingHandler<2>;
template class PostProcessingHandler<3>;
