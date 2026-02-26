
#include <post_processing_handler.h>

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
  , triangulation(triangulation)
  , mpi_communicator(dof_handler.get_mpi_communicator())
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

  if (output_param.write_results)
  {
    data_out = std::make_unique<DataOut<dim>>();
    data_out->attach_dof_handler(dof_handler);
  }

  if (output_param.skin.write_results)
  {
    // build_patches is not (yet) implemented for DataOutFaces in hp context
    AssertThrow(
      !dof_handler.has_hp_capabilities(),
      ExcMessage(
        "\nYou are using a solver with hp capabilities (i.e., "
        "incompressible_ns_lambda or fsi), and you are also trying "
        "to export results on a boundary (with the \"skin\" "
        "subsection). Unfortunately, this feature is currently not yet "
        "implemented in deal.II when using structures with hp capabilities. "
        "Exportation on a skin is supported with the other non-hp solvers."));
    data_out_skin =
      std::make_unique<PostProcessingTools::DataOutFacesOnBoundary<dim>>(
        triangulation, output_param.skin.boundary_id);
    data_out_skin->attach_dof_handler(dof_handler);
  }
}

template <int dim>
void PostProcessingHandler<dim>::write_pvd() const
{
  const std::string suffix =
    mms_param.enable ?
      "_convergence_step_" + std::to_string(mms_param.current_step) + ".pvd" :
      ".pvd";

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
    triangulation,
    post_proc_param.slices.boundary_id,
    post_proc_param.slices.n_slices,
    axis);

  // Store slice indices as cell-based data.
  // If a face is on the sliced boundary, set its cell slice index to the
  // face user index.
  slice_indices.reinit(triangulation.n_active_cells());
  for (const auto &cell : triangulation.active_cell_iterators())
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
void PostProcessingHandler<dim>::write_table(
  std::ostream                                         &out,
  const TableHandler                                   &table,
  const Parameters::PostProcessing::PostProcessingBase &postproc_base) const
{
  const auto mpi_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
  if (mpi_rank == 0)
  {
    out << std::scientific << std::setprecision(postproc_base.precision);
    table.write_text(out);
  }
}

template class PostProcessingHandler<2>;
template class PostProcessingHandler<3>;
