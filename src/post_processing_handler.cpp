
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
    data_out_skin =
      std::make_unique<PostProcessingTools::DataOutFacesOnBoundary<dim>>(
        triangulation, output_param.skin.boundary_id);
    data_out_skin->attach_dof_handler(dof_handler);
  }

  // if (post_proc_param.slices.enable)
  //   create_slices(dof_handler);
}

template <int dim>
void PostProcessingHandler<dim>::write_pvd() const
{
  if (output_param.write_results)
  {
    std::ofstream pvd_output(output_param.output_dir +
                             output_param.output_prefix + ".pvd");
    DataOutBase::write_pvd_record(pvd_output, visualization_times_and_names);
  }
  if (output_param.skin.write_results)
  {
    std::ofstream pvd_output(output_param.output_dir +
                             output_param.skin.output_prefix + ".pvd");
    DataOutBase::write_pvd_record(pvd_output,
                                  visualization_times_and_names_skin);
  }
}

template <int dim>
void PostProcessingHandler<dim>::create_slices(
  const DoFHandler<dim> &dof_handler)
{
  const unsigned int n_slices = std::max(1u, post_proc_param.slices.n_slices);

  const std::string &dir = post_proc_param.slices.along_which_axis;

  if constexpr (dim == 2)
    AssertThrow(dir == "x" || dir == "y",
                ExcMessage("slicing_direction must be 'x' or 'y' in 2D."));
  else
    AssertThrow(dir == "x" || dir == "y" || dir == "z",
                ExcMessage("slicing_direction must be 'x', 'y' or 'z' in 3D."));

  using SliceAxis = PostProcessingTools::SliceAxis;

  const SliceAxis axis = (dir == "x" ? SliceAxis::x :
                          dir == "y" ? SliceAxis::y :
                                       SliceAxis::z);

  slice_index = PostProcessingTools::compute_slice_index_on_boundary<dim>(
    dof_handler,
    post_proc_param.slices.boundary_id,
    n_slices,
    axis,
    mpi_communicator);
}

template <int dim>
void PostProcessingHandler<dim>::clear()
{
  if (data_out)
    data_out->clear_data_vectors();
  if (data_out_skin)
    data_out_skin->clear_data_vectors();
  subdomains.reinit(0);
  visualization_times_and_names.clear();
  visualization_times_and_names_skin.clear();
  solution_names.clear();
  data_component_interpretation.clear();
  // slice_index.clear();
}

template <int dim>
void PostProcessingHandler<dim>::add_force_to_table_and_write(
  const Tensor<1, dim> &forces,
  const TimeHandler    &time_handler)
{
  const auto &forces_param = post_proc_param.forces;

  // Write forces to table
  std::vector<std::string> dim_str = {"x", "y", "z"};
  forces_table.add_value("time", time_handler.current_time);
  for (unsigned int d = 0; d < dim; ++d)
  {
    forces_table.add_value("F" + dim_str[d], forces[d]);
    forces_table.set_precision("F" + dim_str[d], forces_param.precision);
    forces_table.set_scientific("F" + dim_str[d], true);
  }

  // Write forces to file if time step matches the frequency
  const auto mpi_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
  if (should_output_forces(time_handler) && mpi_rank == 0)
  {
    std::ofstream outfile(output_param.output_dir +
                          post_proc_param.forces.output_prefix + ".txt");
    outfile << std::scientific << std::setprecision(forces_param.precision);
    forces_table.write_text(outfile);
  }
}

template class PostProcessingHandler<2>;
template class PostProcessingHandler<3>;
