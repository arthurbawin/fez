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
   * FEType can be either a  FiniteElement<dim> or a hp::FECollection<dim>,
   * and so on for the mapping and face quadrature.
   */
  template <typename VectorType,
            typename FEType,
            typename MappingType,
            typename FaceQuadratureType>
  void compute_forces(const ComponentOrdering  &ordering,
                      const DoFHandler<dim>    &dof_handler,
                      const FEType             &fe,
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
    return post_proc_param.forces.enable &&
           post_proc_param.forces.write_results &&
           (time_handler.current_time_iteration %
                post_proc_param.forces.output_frequency ==
              0 ||
            time_handler.is_finished());
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

  /**
   * Return the slice index for each dof on the prescribed boundary.
   */
  const Vector<double> &get_slice_index() const { return slice_index; }

private:
  /**
   *
   */
  template <typename VectorType>
  void output_volume_fields(const Mapping<dim> &mapping,
                            const VectorType   &solution,
                            const TimeHandler  &time_handler);

  /**
   *
   */
  template <typename VectorType>
  void output_skin_fields(const Mapping<dim> &mapping,
                          const VectorType   &solution,
                          const TimeHandler  &time_handler);

  /**
   * Add the computed forces to the table with required formatting,
   * and write the forces table to file if needed. Called by compute_forces.
   */
  void add_force_to_table_and_write(const Tensor<1, dim> &forces,
                                    const TimeHandler    &time_handler);

  /**
   *
   */
  void create_slices(const DoFHandler<dim> &dof_handler);

private:
  const Parameters::PostProcessing          &post_proc_param;
  const Parameters::Output                  &output_param;
  const Parameters::PhysicalProperties<dim> &physical_properties;

  const Triangulation<dim> &triangulation;

  MPI_Comm mpi_communicator;

  std::unique_ptr<DataOut<dim>> data_out;
  std::unique_ptr<PostProcessingTools::DataOutFacesOnBoundary<dim>>
    data_out_skin;

  Vector<float> subdomains;

  // The times and names of the pvtu files in the pvd file
  std::vector<std::pair<double, std::string>> visualization_times_and_names;
  std::vector<std::pair<double, std::string>>
    visualization_times_and_names_skin;

  std::vector<std::string> solution_names;
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    data_component_interpretation;

  Vector<double> slice_index;

  TableHandler forces_table;
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
  if (subdomains.size() == 0)
  {
    // Get the partitions only once
    Assert(
      triangulation.n_active_cells() > 0,
      ExcMessage(
        "Cannot create subdomains vector because triangulation is empty."));
    subdomains.reinit(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomains.size(); ++i)
      subdomains(i) = triangulation.locally_owned_subdomain();
  }

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
  data_out_skin->build_patches(mapping, 2);

  const std::string pvtu_file = data_out_skin->write_vtu_with_pvtu_record(
    output_param.output_dir,
    output_param.skin.output_prefix,
    time_handler.current_time_iteration,
    mpi_communicator,
    2);
  visualization_times_and_names_skin.emplace_back(time_handler.current_time,
                                                  pvtu_file);
  data_out_skin->clear_data_vectors();
}

template <int dim>
template <typename VectorType,
          typename FEType,
          typename MappingType,
          typename FaceQuadratureType>
void PostProcessingHandler<dim>::compute_forces(
  const ComponentOrdering  &ordering,
  const DoFHandler<dim>    &dof_handler,
  const FEType             &fe,
  const MappingType        &mapping,
  const FaceQuadratureType &face_quadrature,
  const VectorType         &solution,
  const TimeHandler        &time_handler)
{
  const auto &forces_param = post_proc_param.forces;
  using Forces             = Parameters::PostProcessing::Forces;

  Tensor<1, dim> forces;
  std::string    method = "";

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
        fe,
        mapping,
        face_quadrature,
        solution,
        forces_param.boundary_id,
        velocity_extractor,
        pressure_extractor,
        mu);
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
          fe,
          mapping,
          face_quadrature,
          solution,
          forces_param.boundary_id,
          lambda_extractor);
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
    std::cout << std::endl;

    std::cout.precision(old_precision);
    std::cout.flags(old_flags);
  }

  // Add forces to forces table and write if time step matches frequency
  add_force_to_table_and_write(forces, time_handler);
}

#endif
