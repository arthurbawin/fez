
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include <post_processing_handler.h>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <regex>
#include <sstream>

namespace
{
  double
  pvd_time_tolerance(const double time)
  {
    return 1e-12 * std::max(1.0, std::abs(time));
  }

  std::vector<std::pair<double, std::string>>
  read_pvd_entries_until_time(const std::string &filename,
                              const double       max_time)
  {
    std::ifstream input(filename);
    if (!input)
      return {};

    const std::regex dataset_regex("<DataSet[^>]*>");
    const std::regex timestep_regex("timestep=\"([^\"]+)\"");
    const std::regex file_regex("file=\"([^\"]+)\"");
    const double     tolerance = pvd_time_tolerance(max_time);

    std::vector<std::pair<double, std::string>> records;
    std::string                                 line;
    while (std::getline(input, line))
    {
      std::smatch dataset_match;
      if (!std::regex_search(line, dataset_match, dataset_regex))
        continue;

      const std::string dataset = dataset_match.str();
      std::smatch       timestep_match;
      std::smatch       file_match;
      if (!std::regex_search(dataset, timestep_match, timestep_regex) ||
          !std::regex_search(dataset, file_match, file_regex))
        continue;

      try
      {
        const double time = std::stod(timestep_match[1].str());
        if (time <= max_time + tolerance)
          records.emplace_back(time, file_match[1].str());
      }
      catch (const std::exception &)
      {}
    }

    return records;
  }
} // namespace

template <int dim>
PostProcessingHandler<dim>::PostProcessingHandler(
  const ParameterReader<dim>                              &param,
  const Triangulation<dim>                                &triangulation,
  const DoFHandler<dim>                                   &dof_handler,
  const std::vector<std::pair<std::string, unsigned int>> &fields_description)
  : post_proc_param(param.postprocessing)
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
    if (output_param.write_results)
    {
      std::ofstream pvd_output(output_param.output_dir +
                               output_param.output_prefix + suffix);
      DataOutBase::write_pvd_record(pvd_output, visualization_times_and_names);
    }
    if (output_param.skin.write_results)
    {
      std::ofstream pvd_output(output_param.output_dir +
                               output_param.skin.output_prefix + suffix);
      DataOutBase::write_pvd_record(pvd_output,
                                    visualization_times_and_names_skin);
    }
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
      DataOutBase::write_pvd_record(pvd_output,
                                    prerefinements_pseudotimes_and_names_skin);
    }
  }
}

template <int dim>
void PostProcessingHandler<dim>::restore_pvd_entries_until_time(
  const double max_time)
{
  if (mpi_rank != 0)
    return;

  const std::string suffix =
    mms_param.enable ?
      "_convergence_step_" + std::to_string(mms_param.current_step) + ".pvd" :
      ".pvd";

  if (output_param.write_results)
    visualization_times_and_names = read_pvd_entries_until_time(
      output_param.output_dir + output_param.output_prefix + suffix,
      max_time);

  if (output_param.skin.write_results)
    visualization_times_and_names_skin = read_pvd_entries_until_time(
      output_param.output_dir + output_param.skin.output_prefix + suffix,
      max_time);
}

template <int dim>
void PostProcessingHandler<dim>::remove_visualization_record(
  std::vector<std::pair<double, std::string>> &visualization_records,
  const double                                 time,
  const std::string                           &filename) const
{
  const double tolerance = pvd_time_tolerance(time);
  visualization_records.erase(
    std::remove_if(visualization_records.begin(),
                   visualization_records.end(),
                   [&](const auto &record) {
                     return std::abs(record.first - time) <= tolerance ||
                            record.second == filename;
                   }),
    visualization_records.end());
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

  PostProcessingTools::set_slice_index_on_boundary<dim>(
    *triangulation,
    post_proc_param.slices.boundary_id,
    post_proc_param.slices.n_slices,
    axis);

  // Store slice indices as cell-based data.
  // If a face is on the sliced boundary, set its cell slice index to the
  // face user index.
  slice_indices.reinit(triangulation->n_active_cells());
  for (const auto &cell : triangulation->active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto &face : cell->face_iterators())
        if (face->at_boundary() &&
            face->boundary_id() == post_proc_param.slices.boundary_id)
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
  auxiliary_cell_dg0_fields.clear();
  slice_indices.reinit(0);
}

template <int dim>
void PostProcessingHandler<dim>::add_force_to_table(
  const Tensor<1, dim> &forces,
  const TimeHandler    &time_handler,
  TableHandler         &force_table,
  const unsigned int    i_slice)
{
  // Write forces to table
  std::vector<std::string> dim_str = {"x", "y", "z"};
  force_table.add_value("time", time_handler.current_time);
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
  const Tensor<1, dim> &center_position,
  const TimeHandler    &time_handler,
  TableHandler         &table)
{
  // Write position to table
  std::vector<std::string> dim_str = {"x", "y", "z"};
  table.add_value("time", time_handler.current_time);
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
  const Parameters::PostProcessing::PostProcessingBase &pp_param)
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
  const Parameters::PostProcessing::PostProcessingBase &);
template void PostProcessingHandler<3>::add_multiphase_data_to_table(
  const std::array<Tensor<1, 3>, 2> &,
  const TimeHandler &,
  TableHandler &,
  const Parameters::PostProcessing::PostProcessingBase &);
template void PostProcessingHandler<2>::add_multiphase_data_to_table(
  const std::array<double, 2> &,
  const TimeHandler &,
  TableHandler &,
  const Parameters::PostProcessing::PostProcessingBase &);
template void PostProcessingHandler<3>::add_multiphase_data_to_table(
  const std::array<double, 2> &,
  const TimeHandler &,
  TableHandler &,
  const Parameters::PostProcessing::PostProcessingBase &);

template <int dim>
void PostProcessingHandler<dim>::write_table(
  std::ostream                                         &out,
  const TableHandler                                   &table,
  const Parameters::PostProcessing::PostProcessingBase &postproc_base) const
{
  if (mpi_rank == 0)
  {
    out << std::scientific << std::setprecision(postproc_base.precision);
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
void PostProcessingHandler<dim>::process_chns_integrals(
  const PostProcessingTools::CHNSIntegralData &data,
  const TimeHandler                           &time_handler)
{
  using I       = PostProcessingTools::CHNSIntegralData;
  const auto &v = data.sums;
  // Keep the same initial reference even when only the detailed CSV is used.
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

  std::string error;
  if (mpi_rank == 0)
  {
    try
    {
      auto add = [](TableHandler      &table,
                    const std::string &key,
                    const double       value,
                    const unsigned int precision) {
        table.add_value(key, value);
        table.set_precision(key, precision);
        table.set_scientific(key, true);
      };
      auto common = [&](TableHandler &table, const auto &param) {
        add(table, "time", time_handler.current_time, param.precision);
        table.add_value("iteration", time_handler.current_time_iteration);
        add(table, "reference_time", chns_reference_time, param.precision);
      };
      auto add_phase_components = [&](TableHandler      &table,
                                      const unsigned int precision) {
        using E = PostProcessingTools::CHNSEnergyComponents;
        for (unsigned int phase = 0; phase < data.phase_energy.size(); ++phase)
        {
          const auto       &e      = data.phase_energy[phase];
          const std::string suffix = "_phase_" + std::to_string(phase);
          add(table,
              "kinetic_energy" + suffix,
              e.values[E::Kinetic],
              precision);
          add(table, "bulk_energy" + suffix, e.values[E::Bulk], precision);
          add(table,
              "gradient_energy" + suffix,
              e.values[E::Gradient],
              precision);
          add(table,
              "interfacial_energy" + suffix,
              e.interfacial_energy(),
              precision);
          add(table,
              "gravitational_energy" + suffix,
              e.values[E::Gravitational],
              precision);
          add(table, "total_energy" + suffix, e.total_energy(), precision);
        }
      };
      auto output = [&](const TableHandler &table,
                        const auto         &param,
                        const std::string  &message) {
        if (param.verbosity == Parameters::Verbosity::verbose)
        {
          std::ostringstream line;
          line << std::scientific << std::setprecision(param.precision)
               << message << " at t=" << time_handler.current_time;
          if (message == "CHNS conservation")
            line << ": mass=" << v[I::AffineMass]
                 << ", phase=" << v[I::PhaseIntegral];
          else
          {
            line << ": total energy=" << data.total_energy();
            if (!data.has_temporal_energy_diagnostics ||
                message == "CHNS phase energy")
              for (unsigned int phase = 0; phase < data.phase_energy.size();
                   ++phase)
                line << ", phase " << phase << '='
                     << data.phase_energy[phase].total_energy();
          }
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
        common(table, conservation);
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
        output(table, conservation, "CHNS conservation");
      }
      const auto &energy = post_proc_param.chns_energy;
      if (energy.enable && process_domain)
      {
        auto &table = chns_energy_table;
        common(table, energy);
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
        if (!data.has_temporal_energy_diagnostics)
        {
          // Stepien uses the same geometric partition as the phase indicators.
          // Keep the quasi-incompressible table schema unchanged so that its
          // existing checkpoint tables remain usable.
          add_phase_components(table, energy.precision);
        }
        else
        {
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
          const double defect =
            data.total_energy() - v[I::OldEnergy] + dt * dissipation;
          entry("effective_bdf_order", snapshot ? 0. : (bdf2 ? 2. : 1.));
          entry("physical_energy_increment_plus_dissipation", defect);
          entry("energy_balance_residual",
                bdf2 ? nan : defect + v[I::NumericalEnergyLoss]);
          entry("temporal_energy_balance_residual",
                v[I::KineticTemporalPower] + v[I::BulkTemporalPower] +
                  v[I::GradientTemporalPower] +
                  v[I::GravitationalTemporalPower] +
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
        }
        output(table, energy, "CHNS energy");
      }
      if (process_phases)
      {
        const auto &phase_energy = post_proc_param.chns_phase_energy;
        auto       &table        = chns_phase_energy_table;
        add(table, "time", time_handler.current_time, phase_energy.precision);
        table.add_value("iteration", time_handler.current_time_iteration);
        add_phase_components(table, phase_energy.precision);
        output(table, phase_energy, "CHNS phase energy");
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
      stream << "FEZ_POSTPROCESSING_V2\n";
      // Binary archives preserve IEEE NaN for unavailable BDF diagnostics.
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
      // Old checkpoints have no monitoring sidecar. If monitoring is enabled
      // for the first time on restart, its explicit reference_time starts
      // there.
      if (stream)
      {
        std::string version;
        std::getline(stream, version);
        AssertThrow(version == "FEZ_POSTPROCESSING_V1" ||
                      version == "FEZ_POSTPROCESSING_V2",
                    ExcMessage(
                      "Unsupported postprocessing checkpoint format in " +
                      filename));
        boost::archive::binary_iarchive archive(stream);
        archive >> chns_reference_initialized >> chns_reference_time >>
          chns_initial_mass >> chns_initial_phase >> chns_initial_volume >>
          chns_initial_energy >> chns_last_iteration >>
          chns_conservation_table >> chns_energy_table;
        if (version == "FEZ_POSTPROCESSING_V2")
          archive >> chns_phase_energy_last_iteration >>
            chns_phase_energy_table;
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
