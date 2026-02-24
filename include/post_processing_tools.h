#ifndef POST_PROCESSING_TOOLS_H
#define POST_PROCESSING_TOOLS_H

#include <deal.II/base/mpi.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/types.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_values_extractors.h>
#include <deal.II/hp/fe_collection.h>
#include <deal.II/hp/mapping_collection.h>
#include <deal.II/hp/q_collection.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out_faces.h>

using namespace dealii;

/**
 * Collection of utilities for post-processing.
 */
namespace PostProcessingTools
{
  /**
   * Small specialization of deal.II's DataOutFaces to output a single boundary.
   */
  template <int dim>
  class DataOutFacesOnBoundary : public DataOutFaces<dim>
  {
  public:
    /**
     * Constructor
     */
    DataOutFacesOnBoundary(const Triangulation<dim> &triangulation,
                           const types::boundary_id  boundary_id);

    /**
     * A FaceDescriptor is a pair [cell : number of the face].
     */
    using FaceDescriptor = typename DataOutFaces<dim>::FaceDescriptor;

    /**
     * Return the first face which we want output for.
     */
    virtual FaceDescriptor first_face() override;

    /**
     * Return the next face after which we want output for. If there are no more
     * faces, <tt>dofs->end()</tt> is returned as the first component of the
     * return value.
     */
    virtual FaceDescriptor next_face(const FaceDescriptor &face) override;

  protected:
    const Triangulation<dim> &triangulation;
    const types::boundary_id  boundary_id;
  };

  /**
   * Compute the hydrodynamic forces on the given boundary by evaluating the
   * integral of the stress tensor t(n) := sigma \cdot n on the boundary.
   */
  template <int dim, typename VectorType>
  Tensor<1, dim> compute_forces_on_boundary(
    const DoFHandler<dim>            &dof_handler,
    const Mapping<dim>               &mapping,
    const Quadrature<dim - 1>        &face_quadrature,
    const VectorType                 &solution,
    const types::boundary_id          boundary_id,
    const FEValuesExtractors::Vector &velocity_extractor,
    const FEValuesExtractors::Scalar &pressure_extractor,
    const double                      dynamic_viscosity,
    std::vector<Tensor<1, dim>>      &force_per_face);

  /**
   * hp-version of the function above
   */
  template <int dim, typename VectorType>
  Tensor<1, dim> compute_forces_on_boundary(
    const DoFHandler<dim>            &dof_handler,
    const hp::MappingCollection<dim> &mapping_collection,
    const hp::QCollection<dim - 1>   &face_quadrature_collection,
    const VectorType                 &solution,
    const types::boundary_id          boundary_id,
    const FEValuesExtractors::Vector &velocity_extractor,
    const FEValuesExtractors::Scalar &pressure_extractor,
    const double                      dynamic_viscosity,
    std::vector<Tensor<1, dim>>      &force_per_face);

  /**
   * Compute the hydrodynamic forces on the given boundary by evaluating the
   * integral of the given Lagrange multiplier field, through the extractor.
   *
   * It only makes sense to call this function if the Lagrange multiplier field
   * exists to enforce a no-slip condition weakly on the given boundary, as in
   * this case it can be identified to the (opposite of the) stress vector on
   * that boundary.
   *
   * This function effectively returns the opposite of the integral of the
   * Lagrange multiplier on the boundary, so it assumes that the solved weak
   * formulation is compatible with this sign convention. Namely,
   *
   *   + int_boundary \lambda \cdot \phi_u dx
   *
   * appears in the weak formulation, with a positive sign.
   */
  template <int dim, typename VectorType>
  Tensor<1, dim> compute_forces_on_boundary_with_lagrange_multiplier(
    const DoFHandler<dim>            &dof_handler,
    const Mapping<dim>               &mapping,
    const Quadrature<dim - 1>        &face_quadrature,
    const VectorType                 &solution,
    const types::boundary_id          boundary_id,
    const FEValuesExtractors::Vector &lambda_extractor,
    std::vector<Tensor<1, dim>>      &force_per_face);

  /**
   * hp-version of the function above
   */
  template <int dim, typename VectorType>
  Tensor<1, dim> compute_forces_on_boundary_with_lagrange_multiplier(
    const DoFHandler<dim>            &dof_handler,
    const hp::MappingCollection<dim> &mapping_collection,
    const hp::QCollection<dim - 1>   &face_quadrature_collection,
    const VectorType                 &solution,
    const types::boundary_id          boundary_id,
    const FEValuesExtractors::Vector &lambda_extractor,
    std::vector<Tensor<1, dim>>      &force_per_face);

  /**
   * Compute the mean value on the boundary with @p boundary_id of the
   * vector-valued field described by @p field_extractor.
   *
   * This function can be used, e.g., to compute the mean position of a
   * cylinder, that is, the position of its center.
   */
  template <int dim, typename VectorType>
  Tensor<1, dim> compute_vector_mean_value_on_boundary(
    const hp::MappingCollection<dim> &mapping_collection,
    const DoFHandler<dim>            &dof_handler,
    const hp::QCollection<dim - 1>   &face_quadrature_collection,
    const VectorType                 &solution,
    const types::boundary_id          boundary_id,
    const FEValuesExtractors::Vector &field_extractor);

  /**
   * non-hp version of the function above.
   */
  template <int dim, typename VectorType>
  Tensor<1, dim> compute_vector_mean_value_on_boundary(
    const Mapping<dim>               &mapping,
    const DoFHandler<dim>            &dof_handler,
    const Quadrature<dim - 1>        &face_quadrature,
    const VectorType                 &solution,
    const types::boundary_id          boundary_id,
    const FEValuesExtractors::Vector &field_extractor);

  enum class SliceAxis : unsigned int
  {
    x = 0,
    y = 1,
    z = 2
  };

  /**
   * "Slice" the given boundary along the given @p axis, that is, divide that
   * boundary into @p n_slices pieces of equal thickness along axis. The
   * slicing is done based on the coordinates of the faces' barycenter.
   *
   * The slice index to which a face belongs is stored in its user index,
   * accessible with face->user_index(). Since this user index can only be used
   * for one application, a check is done in debug mode to ensure the user index
   * is not in use before overwriting it.
   */
  template <int dim>
  void set_slice_index_on_boundary(const Triangulation<dim> &triangulation,
                                   const types::boundary_id  boundary_id,
                                   const unsigned int        n_slices,
                                   const SliceAxis           axis);

} // namespace PostProcessingTools

/* ---------------- Template functions ----------------- */

template <int dim, typename VectorType>
Tensor<1, dim> PostProcessingTools::compute_forces_on_boundary(
  const DoFHandler<dim>            &dof_handler,
  const Mapping<dim>               &mapping,
  const Quadrature<dim - 1>        &face_quadrature,
  const VectorType                 &solution,
  const types::boundary_id          boundary_id,
  const FEValuesExtractors::Vector &velocity_extractor,
  const FEValuesExtractors::Scalar &pressure_extractor,
  const double                      dynamic_viscosity,
  std::vector<Tensor<1, dim>>      &force_per_face)
{
  Tensor<1, dim> forces, forces_local;
  const double   mu = dynamic_viscosity;

  FEFaceValues<dim> fe_face_values(mapping,
                                   dof_handler.get_fe(),
                                   face_quadrature,
                                   update_values | update_gradients |
                                     update_JxW_values | update_normal_vectors);

  const unsigned int n_faces_q_points = face_quadrature.size();
  std::vector<SymmetricTensor<2, dim>> velocity_sym_gradients(n_faces_q_points);
  std::vector<double>                  pressure_values(n_faces_q_points);

  for (auto cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
      {
        const auto &face = cell->face(i_face);
        if (face->at_boundary() && face->boundary_id() == boundary_id)
        {
          fe_face_values.reinit(cell, i_face);
          const auto &normals = fe_face_values.get_normal_vectors();
          fe_face_values[velocity_extractor].get_function_symmetric_gradients(
            solution, velocity_sym_gradients);
          fe_face_values[pressure_extractor].get_function_values(
            solution, pressure_values);

          auto &f = force_per_face[face->index()];
          f       = 0;
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            const double p          = pressure_values[q];
            const auto  &sym_grad_u = velocity_sym_gradients[q];

            /**
             * If the meshed domain is the fluid domain, the outward normal to
             * the face is the normal going into the solid. To get the forces
             * from the fluid on the solid, take the normal going into the
             * fluid, which is the negative of the returned normals.
             *
             * This way, -p*n = p*normals[q] is oriented towards the solid.
             */
            const auto &n           = -normals[q];
            const auto  sigma_dot_n = -p * n + 2. * mu * sym_grad_u * n;
            f += sigma_dot_n * fe_face_values.JxW(q);
          }

          forces_local += f;
        }
      }
  for (unsigned int d = 0; d < dim; ++d)
    forces[d] =
      Utilities::MPI::sum(forces_local[d], dof_handler.get_mpi_communicator());
  return forces;
}

template <int dim, typename VectorType>
Tensor<1, dim> PostProcessingTools::compute_forces_on_boundary(
  const DoFHandler<dim>            &dof_handler,
  const hp::MappingCollection<dim> &mapping_collection,
  const hp::QCollection<dim - 1>   &face_quadrature_collection,
  const VectorType                 &solution,
  const types::boundary_id          boundary_id,
  const FEValuesExtractors::Vector &velocity_extractor,
  const FEValuesExtractors::Scalar &pressure_extractor,
  const double                      dynamic_viscosity,
  std::vector<Tensor<1, dim>>      &force_per_face)
{
  Tensor<1, dim> forces, forces_local;
  const double   mu = dynamic_viscosity;

  hp::FEFaceValues<dim> hp_fe_face_values(mapping_collection,
                                          dof_handler.get_fe_collection(),
                                          face_quadrature_collection,
                                          update_values | update_gradients |
                                            update_JxW_values |
                                            update_normal_vectors);

  for (auto cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
      {
        const auto &face = cell->face(i_face);
        if (face->at_boundary() && face->boundary_id() == boundary_id)
        {
          const unsigned int fe_index = cell->active_fe_index();

          const unsigned int n_faces_q_points =
            face_quadrature_collection[fe_index].size();
          std::vector<SymmetricTensor<2, dim>> velocity_sym_gradients(
            n_faces_q_points);
          std::vector<double> pressure_values(n_faces_q_points);

          hp_fe_face_values.reinit(cell, i_face);
          const auto &fe_face_values =
            hp_fe_face_values.get_present_fe_values();
          const auto &normals = fe_face_values.get_normal_vectors();
          fe_face_values[velocity_extractor].get_function_symmetric_gradients(
            solution, velocity_sym_gradients);
          fe_face_values[pressure_extractor].get_function_values(
            solution, pressure_values);

          auto &f = force_per_face[face->index()];
          f       = 0;
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            const double p          = pressure_values[q];
            const auto  &sym_grad_u = velocity_sym_gradients[q];

            /**
             * If the meshed domain is the fluid domain, the outward normal to
             * the face is the normal going into the solid. To get the forces
             * from the fluid on the solid, take the normal going into the
             * fluid, which is the negative of the returned normals.
             *
             * This way, -p*n = p*normals[q] is oriented towards the solid.
             */
            const auto &n           = -normals[q];
            const auto  sigma_dot_n = -p * n + 2. * mu * sym_grad_u * n;
            f += sigma_dot_n * fe_face_values.JxW(q);
          }

          forces_local += f;
        }
      }
  for (unsigned int d = 0; d < dim; ++d)
    forces[d] =
      Utilities::MPI::sum(forces_local[d], dof_handler.get_mpi_communicator());
  return forces;
}

template <int dim, typename VectorType>
Tensor<1, dim>
PostProcessingTools::compute_forces_on_boundary_with_lagrange_multiplier(
  const DoFHandler<dim>            &dof_handler,
  const Mapping<dim>               &mapping,
  const Quadrature<dim - 1>        &face_quadrature,
  const VectorType                 &solution,
  const types::boundary_id          boundary_id,
  const FEValuesExtractors::Vector &lambda_extractor,
  std::vector<Tensor<1, dim>>      &force_per_face)
{
  Tensor<1, dim> lambda_integral, lambda_integral_local;

  for (auto &f : force_per_face)
    f = 0;

  FEFaceValues<dim> fe_face_values(mapping,
                                   dof_handler.get_fe(),
                                   face_quadrature,
                                   update_values | update_JxW_values);

  const unsigned int          n_faces_q_points = face_quadrature.size();
  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

  for (auto cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
      {
        const auto &face = cell->face(i_face);
        if (face->at_boundary() && face->boundary_id() == boundary_id)
        {
          fe_face_values.reinit(cell, i_face);
          fe_face_values[lambda_extractor].get_function_values(solution,
                                                               lambda_values);

          auto &f = force_per_face[face->index()];
          f       = 0;
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            const Tensor<1, dim> increment =
              lambda_values[q] * fe_face_values.JxW(q);
            lambda_integral_local += increment;
            f -= increment;
          }
        }
      }
  for (unsigned int d = 0; d < dim; ++d)
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d],
                          dof_handler.get_mpi_communicator());

  // Forces are the opposite of the integral of lambda on the given boundary
  // FIXME: This has to be consistent with the formulation chosen in the
  // solver...
  return -lambda_integral;
}

template <int dim, typename VectorType>
Tensor<1, dim>
PostProcessingTools::compute_forces_on_boundary_with_lagrange_multiplier(
  const DoFHandler<dim>            &dof_handler,
  const hp::MappingCollection<dim> &mapping_collection,
  const hp::QCollection<dim - 1>   &face_quadrature_collection,
  const VectorType                 &solution,
  const types::boundary_id          boundary_id,
  const FEValuesExtractors::Vector &lambda_extractor,
  std::vector<Tensor<1, dim>>      &force_per_face)
{
  Tensor<1, dim> lambda_integral, lambda_integral_local;

  hp::FEFaceValues hp_fe_face_values(mapping_collection,
                                     dof_handler.get_fe_collection(),
                                     face_quadrature_collection,
                                     update_values | update_JxW_values);

  for (auto cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
      {
        const auto &face = cell->face(i_face);
        if (face->at_boundary() && face->boundary_id() == boundary_id)
        {
          const unsigned int fe_index = cell->active_fe_index();
          const unsigned int n_faces_q_points =
            face_quadrature_collection[fe_index].size();
          std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

          hp_fe_face_values.reinit(cell, i_face);
          const auto &fe_face_values =
            hp_fe_face_values.get_present_fe_values();
          fe_face_values[lambda_extractor].get_function_values(solution,
                                                               lambda_values);

          auto &f = force_per_face[face->index()];
          f       = 0;
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            const Tensor<1, dim> increment =
              lambda_values[q] * fe_face_values.JxW(q);
            lambda_integral_local += increment;
            f -= increment;
          }
        }
      }

  for (unsigned int d = 0; d < dim; ++d)
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d],
                          dof_handler.get_mpi_communicator());

  // Forces are the opposite of the integral of lambda on the given boundary
  // FIXME: This has to be consistent with the formulation chosen in the
  // solver...
  return -lambda_integral;
}

template <int dim, typename VectorType>
Tensor<1, dim> PostProcessingTools::compute_vector_mean_value_on_boundary(
  const hp::MappingCollection<dim> &mapping_collection,
  const DoFHandler<dim>            &dof_handler,
  const hp::QCollection<dim - 1>   &face_quadrature_collection,
  const VectorType                 &solution,
  const types::boundary_id          boundary_id,
  const FEValuesExtractors::Vector &field_extractor)
{
  const hp::FECollection<dim> &fe_collection = dof_handler.get_fe_collection();

  AssertDimension(solution.size(), dof_handler.n_dofs());

  hp::FEFaceValues<dim> fe_face_values_collection(
    mapping_collection,
    fe_collection,
    face_quadrature_collection,
    UpdateFlags(update_JxW_values | update_values));

  std::vector<Tensor<1, dim>> values;

  Tensor<1, dim> mean, local_mean;
  double         local_measure = 0.;

  // Compute local_mean value
  for (const auto &cell : dof_handler.active_cell_iterators() |
                            IteratorFilters::LocallyOwnedCell())
  {
    for (const auto &face : cell->face_iterators())
      if (face->at_boundary() && face->boundary_id() == boundary_id)
      {
        fe_face_values_collection.reinit(cell, face);
        const FEFaceValues<dim> &fe_values =
          fe_face_values_collection.get_present_fe_values();

        values.resize(fe_values.n_quadrature_points);
        fe_values[field_extractor].get_function_values(solution, values);
        for (unsigned int k = 0; k < fe_values.n_quadrature_points; ++k)
        {
          local_mean += fe_values.JxW(k) * values[k];
          local_measure += fe_values.JxW(k);
        }
      }
  }

  // FIXME: use MPI_Reduce instead of sum (which uses MPI_Allreduce)
  // if result is only intended for postprocessing and to be written from rank 0
  for (unsigned int d = 0; d < dim; ++d)
    mean[d] =
      Utilities::MPI::sum(local_mean[d], dof_handler.get_mpi_communicator());
  const double measure =
    Utilities::MPI::sum(local_measure, dof_handler.get_mpi_communicator());

  return (mean / measure);
}

template <int dim, typename VectorType>
Tensor<1, dim> PostProcessingTools::compute_vector_mean_value_on_boundary(
  const Mapping<dim>               &mapping,
  const DoFHandler<dim>            &dof_handler,
  const Quadrature<dim - 1>        &face_quadrature,
  const VectorType                 &solution,
  const types::boundary_id          boundary_id,
  const FEValuesExtractors::Vector &field_extractor)
{
  return PostProcessingTools::compute_vector_mean_value_on_boundary(
    hp::MappingCollection<dim>(mapping),
    dof_handler,
    hp::QCollection<dim - 1>(face_quadrature),
    solution,
    boundary_id,
    field_extractor);
}

template <int dim>
void PostProcessingTools::set_slice_index_on_boundary(
  const Triangulation<dim> &triangulation,
  const types::boundary_id  boundary_id,
  const unsigned int        n_slices,
  const SliceAxis           axis)
{
  // Determine the slice thickness "delta" :
  // get the bounding box of the owned vertices on the boundary, then take
  // the max among the bounding boxes and divide the range by n_slices.
  std::vector<Point<dim>> boundary_vertices;
  for (const auto &face : triangulation.active_face_iterators())
    if (face->at_boundary() && face->boundary_id() == boundary_id)
      for (unsigned int v = 0; v < face->n_vertices(); ++v)
        boundary_vertices.push_back(face->vertex(v));

  BoundingBox<dim>   bbox(boundary_vertices);
  MPI_Comm           mpi_comm = triangulation.get_mpi_communicator();
  const unsigned int axis_id  = (unsigned int)axis;
  const double       coord_min =
    Utilities::MPI::min(bbox.lower_bound(axis_id), mpi_comm);
  const double coord_max =
    Utilities::MPI::max(bbox.upper_bound(axis_id), mpi_comm);

  const double delta = (coord_max - coord_min) / n_slices;

  for (const auto &face : triangulation.active_face_iterators())
  {
    if (face->at_boundary() && face->boundary_id() == boundary_id)
    {
      const Point<dim> barry   = face->center();
      unsigned int     i_slice = floor(barry[axis_id] / delta);

      // A point at coord_max will have i_slice = n_slices : decrement it
      if (i_slice == n_slices)
        i_slice--;
      AssertIndexRange(i_slice, n_slices);

      // We are using the face user index to store the slice index, so make
      // sure this index is not already in use
      Assert(face->user_index() == numbers::invalid_unsigned_int,
             ExcMessage("Trying to store the slice index in the face user "
                        "index, but this index is already in use."));

      face->set_user_index(i_slice);
    }
  }
}

#endif
