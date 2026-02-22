#ifndef POST_PROCESSING_HANDLER_H
#define POST_PROCESSING_HANDLER_H

#include <components_ordering.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/numerics/data_out.h>
#include <parameter_reader.h>
#include <parameters.h>
#include <post_processing_tools.h>
#include <time_handler.h>

using namespace dealii;

/**
 * This class handles post-processing operations, such as exporting the
 * solution for visualization, or computing forces on boundaries.
 *
 * FIXME: these are quite different, and maybe we should split this class
 */
template <int dim>
class PostProcessingHandler
{
public:
  /**
   * Constructor.
   *
   * This function accepts an empty mesh and dof_handler, so it can be called
   * before read_mesh(...) and dof_handler.distribute_dofs(...), but these
   * functions must be called before calling any visualization-related
   * functions, e.g., before adding data to the underlying DataOut* or
   * outputting fields. Since deal.II's add_data_vector(...) functions already
   * check that the dof_handler is non-empty, this is not checked here.
   */
  PostProcessingHandler(const ParameterReader<dim> &param,
                        const Triangulation<dim>   &triangulation,
                        const DoFHandler<dim>      &dof_handler,
                        const std::vector<std::pair<std::string, unsigned int>>
                          &fields_description);

  /**
   * Add a cell-based vector of data associated to a field with name "name" to
   * the underlying DataOut object. The vector data should have a size equal to
   * the number of mesh elements on this partitions, e.g., by reinit'ing the
   * vector with triangulation.n_active_cells().
   */
  template <typename VectorType>
  void add_cell_data_vector(const VectorType &data, const std::string &name);

  /**
   * Similar as the function above, but for a dof-based vector of data.
   * A vector of names is needed, as for the deal.II function add_data_vector().
   */
  template <typename VectorType>
  void add_dof_data_vector(const VectorType               &data,
                           const std::vector<std::string> &names);

  /**
   * Output the fields stored in solution, both in the volume and on the
   * prescribed boundary (skin), if any. Also output the fields that were added
   * to the underlying DataOut and/or DataOutFacesOnBoundary by calling the
   * add_*_data_vector functions above.
   *
   * After the fields have been written, the DataOut and DataOutFacesOnBoundary
   * vectors are cleared rightaway, so that one can start the next
   * postprocessing callback by one or more calls to add_*_vector_data.
   *
   * This function already checks whether the volume and/or skin fields should
   * be written for the current time step, according to the prescribed
   * frequency. Thus, it can simply be called without additional checks.
   */
  template <typename VectorType>
  void output_fields(const Mapping<dim> &mapping,
                     const VectorType   &solution,
                     const TimeHandler  &time_handler);

  /**
   * Write the .pvd files (volume and skin, if applicable).
   * Should be called at the end of the simulation.
   */
  void write_pvd() const;

  /**
   * Compute the hydrodynamic forces on the boundary prescribed in the forces
   * postprocessing parameters at the current time step. Adds these forces
   * to the forces table and write the table to the prescribed file if the time
   * step matches the given output frequency.
   *
   * This function is templated to work with both hp and non-hp solvers, and
   * MappingType can be either a Mapping<dim> or a hp::MappingCollection<dim>,
   * and similarly for the face quadrature.
   */
  template <typename VectorType,
            typename MappingType,
            typename FaceQuadratureType>
  void compute_forces(const ComponentOrdering  &ordering,
                      const DoFHandler<dim>    &dof_handler,
                      const MappingType        &mapping,
                      const FaceQuadratureType &face_quadrature,
                      const VectorType         &solution,
                      const TimeHandler        &time_handler);

  /**
   * Compute the mean position of the structure described by the boundary id
   * in the PostProcessing.StructurePosition parameters, add it to a table
   * and write it to file if required.
   *
   * For a cylinder, for instance, this computes the position of the geometric
   * center of the cylinder.
   *
   * This function is templated to work with both hp and non-hp solvers, and
   * MappingType can be either a Mapping<dim> or a hp::MappingCollection<dim>,
   * and similarly for the face quadrature.
   */
  template <typename VectorType,
            typename MappingType,
            typename FaceQuadratureType>
  void
  compute_structure_mean_position(const ComponentOrdering  &ordering,
                                  const DoFHandler<dim>    &dof_handler,
                                  const MappingType        &mapping,
                                  const FaceQuadratureType &face_quadrature,
                                  const VectorType         &solution,
                                  const TimeHandler        &time_handler);

  /**
   * Reset the underlying data and vectors.
   */
  void clear();

  /**
   * Return true if the volume fields should be output at this time step.
   */
  bool should_output_volume_fields(const TimeHandler &time_handler) const
  {
    return output_param.write_results &&
           (time_handler.current_time_iteration %
                output_param.vtu_output_frequency ==
              0 ||
            time_handler.is_finished());
  }

  /**
   * Return true if the skin fields should be output at this time step.
   */
  bool should_output_skin_fields(const TimeHandler &time_handler) const
  {
    return output_param.skin.write_results &&
           (time_handler.current_time_iteration %
                output_param.skin.output_frequency ==
              0 ||
            time_handler.is_finished());
  }

  /**
   * Return true if the forces should be output at this time step.
   */
  bool should_output_forces(const TimeHandler &time_handler) const
  {
    return should_output_postprocessing(time_handler, post_proc_param.forces);
  }

  /**
   * Return true if the structure's mean position should be output at this time
   * step.
   */
  bool should_output_mean_position(const TimeHandler &time_handler) const
  {
    return should_output_postprocessing(time_handler,
                                        post_proc_param.structure_position);
  }

  /**
   * Return the field name of each solution component.
   */
  const std::vector<std::string> &get_field_names() const
  {
    return solution_names;
  }

  /**
   * Return the data interpretation of each solution component (scalar, part of
   * vector, or part of tensor).
   */
  const std::vector<DataComponentInterpretation::DataComponentInterpretation> &
  get_component_interpretations() const
  {
    return data_component_interpretation;
  }

private:
  /**
   * Return true if the passed postprocessing should be output at this time
   * step.
   */
  bool should_output_postprocessing(
    const TimeHandler                                    &time_handler,
    const Parameters::PostProcessing::PostProcessingBase &postproc_base) const
  {
    return postproc_base.enable && postproc_base.write_results &&
           (time_handler.current_time_iteration %
                postproc_base.output_frequency ==
              0 ||
            time_handler.is_finished());
  }

  /**
   * Output the volume fields for visualization. This includes the fields
   * in the passed @p solution vector, the subdomain (partition) IDs and
   * the additional data that were added with add_cell_data_vector and/or
   * add_dof_data_vector.
   */
  template <typename VectorType>
  void output_volume_fields(const Mapping<dim> &mapping,
                            const VectorType   &solution,
                            const TimeHandler  &time_handler);

  /**
   * Output the fields defined on the skin for visualization. This includes
   * the same fields as in output_volume_fields, with additionally the slice
   * indices, if the boundary associated to the skin was sliced.
   */
  template <typename VectorType>
  void output_skin_fields(const Mapping<dim> &mapping,
                          const VectorType   &solution,
                          const TimeHandler  &time_handler);

  /**
   * Add the computed forces to the passed table with required formatting.
   */
  void add_force_to_table(
    const Tensor<1, dim> &forces,
    const TimeHandler    &time_handler,
    TableHandler         &force_table,
    const unsigned int    i_slice = numbers::invalid_unsigned_int);

  /**
   * Add the computed position of the structure's geometric center to the passed
   * table with required formatting.
   */
  void add_position_to_table(const Tensor<1, dim> &center_position,
                             const TimeHandler    &time_handler,
                             TableHandler         &position_table);

  /**
   * Write the given table to the out stream.
   */
  void write_table(
    std::ostream                                         &out,
    const TableHandler                                   &table,
    const Parameters::PostProcessing::PostProcessingBase &postproc_base) const;

  /**
   * Assign a slice index to the faces on the sliced boundary.
   */
  void create_slices();

private:
  const Parameters::PostProcessing          &post_proc_param;
  const Parameters::Output                  &output_param;
  const Parameters::PhysicalProperties<dim> &physical_properties;

  const Triangulation<dim> &triangulation;
  MPI_Comm                  mpi_communicator;

  std::unique_ptr<DataOut<dim>> data_out;
  std::unique_ptr<PostProcessingTools::DataOutFacesOnBoundary<dim>>
    data_out_skin;

  // Name and component interpretation of the fields to write
  std::vector<std::string> solution_names;
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation;

  // The times and names of the pvtu files in the pvd file
  std::vector<std::pair<double, std::string>> visualization_times_and_names;
  std::vector<std::pair<double, std::string>>
    visualization_times_and_names_skin;

  // Subdomain (partition) IDs
  Vector<float> subdomains;

  // Forces on the prescribed boundary, and on each slice if enabled
  TableHandler  forces_table;
  Vector<float> slice_indices;
  TableHandler  forces_table_per_slice;

  // The position of the geometric center (average) of the structure,
  // if solving a fluid-structure interaction problem
  TableHandler structure_mean_position_table;
};

/* ---------------- Template functions ----------------- */

template <int dim>
template <typename VectorType>
void PostProcessingHandler<dim>::add_cell_data_vector(const VectorType  &data,
                                                      const std::string &name)
{
  data_out->add_data_vector(data, name, DataOut<dim>::type_cell_data);
}

template <int dim>
template <typename VectorType>
void PostProcessingHandler<dim>::add_dof_data_vector(
  const VectorType               &data,
  const std::vector<std::string> &names)
{
  data_out->add_data_vector(data,
                            names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);
}

template <int dim>
template <typename VectorType>
void PostProcessingHandler<dim>::output_fields(const Mapping<dim> &mapping,
                                               const VectorType   &solution,
                                               const TimeHandler  &time_handler)
{
  // Get the partitions only once
  if (subdomains.size() == 0)
  {
    Assert(
      triangulation.n_active_cells() > 0,
      ExcMessage(
        "Cannot create subdomains vector because triangulation is empty."));
    subdomains.reinit(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomains.size(); ++i)
      subdomains(i) = triangulation.locally_owned_subdomain();
  }

  // Compute slices indices once
  if (post_proc_param.slices.enable && slice_indices.size() == 0)
    create_slices();

  // Export fields in volume
  if (should_output_volume_fields(time_handler))
    output_volume_fields(mapping, solution, time_handler);

  // Export fields on prescribed boundary (skin)
  if (should_output_skin_fields(time_handler))
    output_skin_fields(mapping, solution, time_handler);
}

template <int dim>
template <typename VectorType>
void PostProcessingHandler<dim>::output_volume_fields(
  const Mapping<dim> &mapping,
  const VectorType   &solution,
  const TimeHandler  &time_handler)
{
  data_out->add_data_vector(solution,
                            solution_names,
                            DataOut<dim>::type_dof_data,
                            data_component_interpretation);
  data_out->add_data_vector(subdomains, "subdomain");
  data_out->build_patches(mapping, 2);

  const std::string pvtu_file =
    data_out->write_vtu_with_pvtu_record(output_param.output_dir,
                                         output_param.output_prefix,
                                         time_handler.current_time_iteration,
                                         mpi_communicator,
                                         2);
  visualization_times_and_names.emplace_back(time_handler.current_time,
                                             pvtu_file);
  data_out->clear_data_vectors();
}

template <int dim>
template <typename VectorType>
void PostProcessingHandler<dim>::output_skin_fields(
  const Mapping<dim> &mapping,
  const VectorType   &solution,
  const TimeHandler  &time_handler)
{
  data_out_skin->add_data_vector(solution,
                                 solution_names,
                                 DataOutFaces<dim>::type_dof_data,
                                 data_component_interpretation);
  data_out_skin->add_data_vector(subdomains, "subdomain");
  if (post_proc_param.slices.enable)
  {
    data_out_skin->add_data_vector(slice_indices,
                                   "slice index",
                                   DataOutFaces<dim>::type_cell_data);
  }
  data_out_skin->build_patches(mapping, 2);

  const std::string pvtu_file = data_out_skin->write_vtu_with_pvtu_record(
    output_param.output_dir,
    output_param.output_prefix + "_" + output_param.skin.output_prefix,
    time_handler.current_time_iteration,
    mpi_communicator,
    2);
  visualization_times_and_names_skin.emplace_back(time_handler.current_time,
                                                  pvtu_file);
  data_out_skin->clear_data_vectors();
}

template <int dim>
template <typename VectorType,
          typename MappingType,
          typename FaceQuadratureType>
void PostProcessingHandler<dim>::compute_forces(
  const ComponentOrdering  &ordering,
  const DoFHandler<dim>    &dof_handler,
  const MappingType        &mapping,
  const FaceQuadratureType &face_quadrature,
  const VectorType         &solution,
  const TimeHandler        &time_handler)
{
  const auto &forces_param = post_proc_param.forces;
  using Forces             = Parameters::PostProcessing::Forces;

  Tensor<1, dim> forces;
  std::string    method = "";

  std::vector<Tensor<1, dim>> force_per_face(
    dof_handler.get_triangulation().n_faces());

  switch (forces_param.method)
  {
    case Forces::ComputationMethod::stress_vector:
    {
      method = "stress_vector";
      const FEValuesExtractors::Vector velocity_extractor(ordering.u_lower);
      const FEValuesExtractors::Scalar pressure_extractor(ordering.p_lower);

      // FIXME: take viscosity of the mixture in CHNS
      const double mu = physical_properties.fluids[0].dynamic_viscosity;

      forces = PostProcessingTools::compute_forces_on_boundary(
        dof_handler,
        mapping,
        face_quadrature,
        solution,
        forces_param.boundary_id,
        velocity_extractor,
        pressure_extractor,
        mu,
        force_per_face);
      break;
    }
    case Forces::ComputationMethod::lagrange_multiplier:
    {
      method = "lagrange_multiplier";
      AssertThrow(ordering.l_lower != numbers::invalid_unsigned_int,
                  ExcMessage(
                    "Cannot compute forces with a Lagrange multiplier "
                    "because the chosen "
                    "solver does not have a Lagrange multiplier variable."));

      const FEValuesExtractors::Vector lambda_extractor(ordering.l_lower);
      forces = PostProcessingTools::
        compute_forces_on_boundary_with_lagrange_multiplier(
          dof_handler,
          mapping,
          face_quadrature,
          solution,
          forces_param.boundary_id,
          lambda_extractor,
          force_per_face);
      break;
    }
    default:
      DEAL_II_NOT_IMPLEMENTED();
  }

  const auto mpi_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
  if (forces_param.verbosity == Parameters::Verbosity::verbose && mpi_rank == 0)
  {
    std::ios::fmtflags old_flags     = std::cout.flags();
    unsigned int       old_precision = std::cout.precision();

    std::vector<std::string> dim_str = {"x", "y", "z"};
    std::cout << std::scientific << std::setprecision(forces_param.precision)
              << std::showpos << std::endl;
    std::cout << "Forces on boundary with id " << forces_param.boundary_id
              << " computed with method : " << method << std::endl;
    for (unsigned int d = 0; d < dim; ++d)
      std::cout << "F" + dim_str[d] << " = " << forces[d] << std::endl;

    std::cout.precision(old_precision);
    std::cout.flags(old_flags);
  }

  // Add forces to forces table and write if time step matches frequency
  {
    add_force_to_table(forces, time_handler, forces_table);
    std::ofstream outfile(output_param.output_dir +
                          post_proc_param.forces.output_prefix + ".txt");
    if (should_output_forces(time_handler))
      write_table(outfile, forces_table, post_proc_param.forces);
  }

  // Compute forces on each slice of given boundary
  const auto &slices_param = post_proc_param.slices;
  if (slices_param.enable && slices_param.compute_forces_on_slices)
  {
    std::vector<Tensor<1, dim>> forces_per_slice_local(slices_param.n_slices);
    std::vector<Tensor<1, dim>> forces_per_slice = forces_per_slice_local;

    for (const auto &face : triangulation.active_face_iterators())
    {
      if (face->user_index() != numbers::invalid_unsigned_int)
        forces_per_slice_local[face->user_index()] +=
          force_per_face[face->index()];
    }

    for (unsigned int i = 0; i < slices_param.n_slices; ++i)
    {
      forces_per_slice[i] =
        Utilities::MPI::sum(forces_per_slice_local[i],
                            dof_handler.get_mpi_communicator());
      add_force_to_table(forces_per_slice[i],
                         time_handler,
                         forces_table_per_slice,
                         i);
    }

    if (forces_param.verbosity == Parameters::Verbosity::verbose &&
        mpi_rank == 0)
    {
      std::ios::fmtflags old_flags     = std::cout.flags();
      unsigned int       old_precision = std::cout.precision();

      std::vector<std::string> dim_str = {"x", "y", "z"};
      std::cout << std::scientific << std::setprecision(forces_param.precision)
                << std::showpos << std::endl;
      std::cout << "Forces per slice on boundary with id "
                << forces_param.boundary_id << ":" << std::endl;
      for (unsigned int i = 0; i < slices_param.n_slices; ++i)
      {
        std::cout << "Slice " << i << ": ";
        for (unsigned int d = 0; d < dim; ++d)
          std::cout << "F" + dim_str[d] << " = " << forces_per_slice[i][d]
                    << "\t";
        std::cout << std::endl;
      }

      std::cout.precision(old_precision);
      std::cout.flags(old_flags);
    }

    // Write to file
    std::ofstream slices_outfile(output_param.output_dir +
                                 post_proc_param.forces.output_prefix + "_" +
                                 slices_param.output_prefix + ".txt");
    write_table(slices_outfile, forces_table_per_slice, post_proc_param.forces);

    // Check that sum of forces on slices is the force on boundary
    {
      Tensor<1, dim> sum_slices;
      for (const auto &f : forces_per_slice)
        sum_slices += f;
      AssertThrow((forces - sum_slices).norm_square() < 1e-14,
                  ExcMessage("Sum of forces on slices does not match the total "
                             "forces on this boundary"));
    }
  }
}

template <int dim>
template <typename VectorType,
          typename MappingType,
          typename FaceQuadratureType>
void PostProcessingHandler<dim>::compute_structure_mean_position(
  const ComponentOrdering  &ordering,
  const DoFHandler<dim>    &dof_handler,
  const MappingType        &mapping,
  const FaceQuadratureType &face_quadrature,
  const VectorType         &solution,
  const TimeHandler        &time_handler)
{
  AssertThrow(ordering.x_lower != numbers::invalid_unsigned_int,
              ExcMessage("Cannot compute structure position because this "
                         "solver does not have a mesh position variable"));

  const FEValuesExtractors::Vector position_extractor(ordering.x_lower);

  const Tensor<1, dim> mean_position =
    PostProcessingTools::compute_vector_mean_value_on_boundary(
      mapping,
      dof_handler,
      face_quadrature,
      solution,
      post_proc_param.structure_position.boundary_id,
      position_extractor);

  const auto &position_param = post_proc_param.structure_position;
  const auto  mpi_rank = Utilities::MPI::this_mpi_process(mpi_communicator);
  if (position_param.verbosity == Parameters::Verbosity::verbose &&
      mpi_rank == 0)
  {
    std::ios::fmtflags old_flags     = std::cout.flags();
    unsigned int       old_precision = std::cout.precision();

    std::vector<std::string> dim_str = {"x", "y", "z"};
    std::cout << std::scientific << std::setprecision(position_param.precision)
              << std::showpos << std::endl;
    std::cout << "Mean position (geometric center) on boundary with id "
              << position_param.boundary_id << ":" << std::endl;
    for (unsigned int d = 0; d < dim; ++d)
      std::cout << dim_str[d] << " = " << mean_position[d] << std::endl;

    std::cout.precision(old_precision);
    std::cout.flags(old_flags);
  }

  // Add forces to forces table and write if time step matches frequency
  add_position_to_table(mean_position,
                        time_handler,
                        structure_mean_position_table);
  std::ofstream outfile(output_param.output_dir +
                        post_proc_param.structure_position.output_prefix +
                        ".txt");
  if (should_output_mean_position(time_handler))
    write_table(outfile,
                structure_mean_position_table,
                post_proc_param.structure_position);
}

#endif
