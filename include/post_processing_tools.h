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

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>

#include <types.h>

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
   * Non-hp version of the function above.
   */
  template <int dim, typename VectorType>
  Tensor<1, dim> compute_vector_mean_value_on_boundary(
    const Mapping<dim>               &mapping,
    const DoFHandler<dim>            &dof_handler,
    const Quadrature<dim - 1>        &face_quadrature,
    const VectorType                 &solution,
    const types::boundary_id          boundary_id,
    const FEValuesExtractors::Vector &field_extractor);

  /**
   * Compute the maximum CFL number over the mesh cells.
   */
  template <int dim, typename VectorType>
  double compute_max_cfl(const double                      timestep,
                         const hp::MappingCollection<dim> &mapping_collection,
                         const DoFHandler<dim>            &dof_handler,
                         const hp::QCollection<dim> &cell_quadrature_collection,
                         const VectorType           &solution,
                         const FEValuesExtractors::Vector &velocity_extractor);

  /**
   * Non-hp version of the function above.
   */
  template <int dim, typename VectorType>
  double compute_max_cfl(const double                      timestep,
                         const Mapping<dim>               &mapping,
                         const DoFHandler<dim>            &dof_handler,
                         const Quadrature<dim>            &cell_quadrature,
                         const VectorType                 &solution,
                         const FEValuesExtractors::Vector &velocity_extractor);

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

  /**
   * Compute vorticity from the velocity gradient.
   *
   * Convention:
   * grad_u[i][j] = d u_i / d x_j.
   *
   * In 2D, the physical vorticity is omega_z. Since Tensor<1,2>
   * has only two components, omega_z is stored in omega[0].
   */
  template <int dim>
  Tensor<1, dim>
  compute_vorticity_from_velocity_gradient(const Tensor<2, dim> &grad_u);

  /**
   * Compute the Q criterion from the velocity gradient.
   *
   * Q = 1/2 (||Omega||^2 - ||S||^2),
   * with S = 1/2 (grad_u + grad_u^T),
   * and Omega = 1/2 (grad_u - grad_u^T).
   */
  template <int dim>
  double
  compute_qcriterion_from_velocity_gradient(const Tensor<2, dim> &grad_u);


  /**
   * Compute a dof vector containing the vorticity.
   *
   * The returned vector has the same layout as the monolithic solution.
   * Only the components selected by @p vorticity_output_extractor are filled.
   *
   * Usual choice:
   *   vorticity_output_extractor = velocity_extractor
   *
   * Then vorticity is stored on the same support as velocity:
   * P2 if velocity is P2, P1 if velocity is P1, etc.
   */
  template <int dim, typename VectorType>
  void compute_vorticity_dof_vector(
    const DoFHandler<dim>            &dof_handler,
    const Mapping<dim>               &mapping,
    const VectorType                 &solution,
    const FiniteElement<dim>         &fe,
    const FEValuesExtractors::Vector &velocity_extractor,
    const FEValuesExtractors::Vector &vorticity_output_extractor,
    LA::ParVectorType                &vorticity_dof_vector);

  /**
   * Compute a dof vector containing the Q criterion.
   *
   * The returned vector has the same layout as the monolithic solution.
   * Only the scalar component selected by @p qcriterion_output_extractor is
   * filled.
   *
   * Examples:
   *   qcriterion_output_extractor = pressure_extractor
   *     -> Qcriterion is stored on the pressure support, usually P1.
   *
   *   qcriterion_output_extractor = FEValuesExtractors::Scalar(u_lower)
   *     -> Qcriterion is stored on the ux support, usually P2.
   */
  template <int dim, typename VectorType>
  void compute_qcriterion_scalar_dof_vector(
    const DoFHandler<dim>            &dof_handler,
    const Mapping<dim>               &mapping,
    const VectorType                 &solution,
    const FiniteElement<dim>         &fe,
    const FEValuesExtractors::Vector &velocity_extractor,
    const FEValuesExtractors::Scalar &qcriterion_output_extractor,
    LA::ParVectorType                &qcriterion_dof_vector);


  template <typename VectorType>
  void add_to_distributed_vector_entry(
    VectorType                         &vector,
    const types::global_dof_index       global_dof,
    const double                        value)
  {
    const std::vector<typename VectorType::size_type> indices = {
      static_cast<typename VectorType::size_type>(global_dof)
    };

    const std::vector<double> values = {value};

    vector.add(indices, values);
  }
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

      // FIXME: With the hp FSI solver, there are already some faces with user
      // index 0 when we enter this function, but it does not seem to affect
      // the results, so the test below is not done.
      //
      // We are using the face user index to store the slice index, so make
      // sure this index is not already in use
      // Assert(face->user_index() == numbers::invalid_unsigned_int,
      //        ExcMessage("Trying to store the slice index in the face user "
      //                   "index, but this index is already in use."));

      face->set_user_index(i_slice);
    }
  }
}

template <int dim, typename VectorType>
double PostProcessingTools::compute_max_cfl(
  const double                      timestep,
  const hp::MappingCollection<dim> &mapping_collection,
  const DoFHandler<dim>            &dof_handler,
  const hp::QCollection<dim>       &cell_quadrature_collection,
  const VectorType                 &solution,
  const FEValuesExtractors::Vector &velocity_extractor)
{
  AssertDimension(solution.size(), dof_handler.n_dofs());

  hp::FEValues<dim> hp_fe_values(mapping_collection,
                                 dof_handler.get_fe_collection(),
                                 cell_quadrature_collection,
                                 UpdateFlags(update_values));

  double                      local_max_cfl = 0.;
  std::vector<Tensor<1, dim>> values;

  for (const auto &cell : dof_handler.active_cell_iterators() |
                            IteratorFilters::LocallyOwnedCell())
  {
    const double h = cell->diameter();

    hp_fe_values.reinit(cell);
    const FEValues<dim> &fe_values = hp_fe_values.get_present_fe_values();

    values.resize(fe_values.n_quadrature_points);
    fe_values[velocity_extractor].get_function_values(solution, values);
    for (unsigned int q = 0; q < fe_values.n_quadrature_points; ++q)
    {
      const double velocity_norm = values[q].norm();
      local_max_cfl              = std::max(local_max_cfl, velocity_norm / h);
    }
  }

  // Synchronize across ranks
  const double max_cfl =
    Utilities::MPI::max(local_max_cfl * timestep,
                        dof_handler.get_mpi_communicator());

  return max_cfl;
}

template <int dim, typename VectorType>
double PostProcessingTools::compute_max_cfl(
  const double                      timestep,
  const Mapping<dim>               &mapping,
  const DoFHandler<dim>            &dof_handler,
  const Quadrature<dim>            &cell_quadrature,
  const VectorType                 &solution,
  const FEValuesExtractors::Vector &velocity_extractor)
{
  return PostProcessingTools::compute_max_cfl(
    timestep,
    hp::MappingCollection<dim>(mapping),
    dof_handler,
    hp::QCollection<dim>(cell_quadrature),
    solution,
    velocity_extractor);
}


template <int dim>
Tensor<1, dim>
PostProcessingTools::compute_vorticity_from_velocity_gradient(
  const Tensor<2, dim> &grad_u)
{
  Tensor<1, dim> omega;

  if constexpr (dim == 2)
  {
    /*
     * grad_u[i][j] = d u_i / d x_j
     *
     * omega_z = d u_y / d x - d u_x / d y.
     *
     * Since Tensor<1,2> has no z component, omega_z is stored in omega[0].
     */
    omega[0] = grad_u[1][0] - grad_u[0][1];
    omega[1] = 0.0;
  }
  else if constexpr (dim == 3)
  {
    omega[0] = grad_u[2][1] - grad_u[1][2]; // d u_z/dy - d u_y/dz
    omega[1] = grad_u[0][2] - grad_u[2][0]; // d u_x/dz - d u_z/dx
    omega[2] = grad_u[1][0] - grad_u[0][1]; // d u_y/dx - d u_x/dy
  }
  else
  {
    DEAL_II_NOT_IMPLEMENTED();
  }

  return omega;
}

template <int dim>
double
PostProcessingTools::compute_qcriterion_from_velocity_gradient(
  const Tensor<2, dim> &grad_u)
{
  double norm_S_squared     = 0.0;
  double norm_Omega_squared = 0.0;

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
    {
      const double S_ij =
        0.5 * (grad_u[i][j] + grad_u[j][i]);

      const double Omega_ij =
        0.5 * (grad_u[i][j] - grad_u[j][i]);

      norm_S_squared += S_ij * S_ij;
      norm_Omega_squared += Omega_ij * Omega_ij;
    }

  return 0.5 * (norm_Omega_squared - norm_S_squared);
}

template <int dim, typename VectorType>
void PostProcessingTools::compute_vorticity_dof_vector(
  const DoFHandler<dim>            &dof_handler,
  const Mapping<dim>               &mapping,
  const VectorType                 &solution,
  const FiniteElement<dim>         &fe,
  const FEValuesExtractors::Vector &velocity_extractor,
  const FEValuesExtractors::Vector &vorticity_output_extractor,
  LA::ParVectorType                &vorticity_dof_vector)
{
  AssertDimension(solution.size(), dof_handler.n_dofs());

  const MPI_Comm mpi_communicator = dof_handler.get_mpi_communicator();

  const IndexSet locally_owned_dofs =
    dof_handler.locally_owned_dofs();

  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  vorticity_dof_vector.reinit(locally_owned_dofs,
                              locally_relevant_dofs,
                              mpi_communicator);
  vorticity_dof_vector = 0.0;

  LA::ParVectorType local_vorticity_sum;
  LA::ParVectorType local_counter;

  local_vorticity_sum.reinit(locally_owned_dofs, mpi_communicator);
  local_counter.reinit(locally_owned_dofs, mpi_communicator);

  local_vorticity_sum = 0.0;
  local_counter       = 0.0;

  /*
   * We evaluate grad(u) at the support points of the first scalar component
   * of the output vector field.
   *
   * If vorticity_output_extractor = velocity_extractor, these points are the
   * velocity support points. Therefore, the exported vorticity has the same
   * polynomial support as velocity.
   */
  const unsigned int first_output_component =
    vorticity_output_extractor.first_vector_component;

  const FEValuesExtractors::Scalar first_output_scalar(
    first_output_component);

  const ComponentMask first_output_mask =
    fe.component_mask(first_output_scalar);

  const FiniteElement<dim> &scalar_output_fe =
    fe.get_sub_fe(first_output_mask);

  AssertThrow(
    scalar_output_fe.has_support_points(),
    ExcMessage("The chosen vorticity output finite element must have "
               "support points."));

  const std::vector<Point<dim>> &unit_support_points =
    scalar_output_fe.get_unit_support_points();

  const Quadrature<dim> quadrature(unit_support_points);

  FEValues<dim> fe_values(mapping,
                          fe,
                          quadrature,
                          update_gradients);

  const unsigned int n_q_points = quadrature.size();

  std::vector<Tensor<2, dim>> velocity_gradients(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(
    fe.n_dofs_per_cell());

  /*
   * Map local output DoFs to quadrature points.
   *
   * For each local DoF belonging to the output vector field, the local shape
   * index gives the corresponding support point index.
   */
  std::vector<std::tuple<unsigned int, unsigned int, unsigned int>>
    local_output_dof_component_qpoint;

  for (unsigned int local_dof = 0; local_dof < fe.n_dofs_per_cell();
       ++local_dof)
  {
    const auto component_shape =
      fe.system_to_component_index(local_dof);

    const unsigned int component = component_shape.first;
    const unsigned int shape     = component_shape.second;

    if (component >= first_output_component &&
        component < first_output_component + dim)
    {
      const unsigned int d = component - first_output_component;

      AssertIndexRange(shape, n_q_points);

      local_output_dof_component_qpoint.emplace_back(local_dof, d, shape);
    }
  }

  /*
   * We loop on owned and ghost cells, but we only write into locally owned
   * DoFs. Ghost cells are useful for owned DoFs on MPI interfaces.
   */
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!(cell->is_locally_owned() || cell->is_ghost()))
      continue;

    fe_values.reinit(cell);

    fe_values[velocity_extractor].get_function_gradients(solution,
                                                         velocity_gradients);

    cell->get_dof_indices(local_dof_indices);

    for (const auto &[local_dof, d, q] : local_output_dof_component_qpoint)
    {
      const types::global_dof_index global_dof =
        local_dof_indices[local_dof];

      if (!locally_owned_dofs.is_element(global_dof))
        continue;

      const Tensor<1, dim> omega =
        compute_vorticity_from_velocity_gradient<dim>(
          velocity_gradients[q]);

      add_to_distributed_vector_entry(local_vorticity_sum,
                                      global_dof,
                                      omega[d]);

      add_to_distributed_vector_entry(local_counter,
                                      global_dof,
                                      1.0);
          }
  }

  local_vorticity_sum.compress(VectorOperation::add);
  local_counter.compress(VectorOperation::add);

  LA::ParVectorType local_vorticity;
  local_vorticity.reinit(locally_owned_dofs, mpi_communicator);
  local_vorticity = 0.0;

  for (const auto global_dof : locally_owned_dofs)
  {
    const double count = local_counter[global_dof];

    if (count > 0.0)
      local_vorticity[global_dof] =
        local_vorticity_sum[global_dof] / count;
  }

  local_vorticity.compress(VectorOperation::insert);

  vorticity_dof_vector = local_vorticity;
}

template <int dim, typename VectorType>
void PostProcessingTools::compute_qcriterion_scalar_dof_vector(
  const DoFHandler<dim>            &dof_handler,
  const Mapping<dim>               &mapping,
  const VectorType                 &solution,
  const FiniteElement<dim>         &fe,
  const FEValuesExtractors::Vector &velocity_extractor,
  const FEValuesExtractors::Scalar &qcriterion_output_extractor,
  LA::ParVectorType                &qcriterion_dof_vector)
{
  AssertDimension(solution.size(), dof_handler.n_dofs());

  const MPI_Comm mpi_communicator = dof_handler.get_mpi_communicator();

  const IndexSet locally_owned_dofs =
    dof_handler.locally_owned_dofs();

  const IndexSet locally_relevant_dofs =
    DoFTools::extract_locally_relevant_dofs(dof_handler);

  qcriterion_dof_vector.reinit(locally_owned_dofs,
                               locally_relevant_dofs,
                               mpi_communicator);
  qcriterion_dof_vector = 0.0;

  LA::ParVectorType local_qcriterion_sum;
  LA::ParVectorType local_counter;

  local_qcriterion_sum.reinit(locally_owned_dofs, mpi_communicator);
  local_counter.reinit(locally_owned_dofs, mpi_communicator);

  local_qcriterion_sum = 0.0;
  local_counter        = 0.0;

  /*
   * This scalar extractor defines the output support.
   *
   * If qcriterion_output_extractor = pressure_extractor:
   *   Qcriterion is stored on pressure DoFs, usually P1.
   *
   * If qcriterion_output_extractor = Scalar(u_lower):
   *   Qcriterion is stored on ux DoFs, usually P2.
   */
  const ComponentMask qcriterion_output_mask =
    fe.component_mask(qcriterion_output_extractor);

  const FiniteElement<dim> &scalar_output_fe =
    fe.get_sub_fe(qcriterion_output_mask);

  AssertThrow(
    scalar_output_fe.has_support_points(),
    ExcMessage("The selected Qcriterion output finite element must have "
               "support points."));

  const std::vector<Point<dim>> &unit_support_points =
    scalar_output_fe.get_unit_support_points();

  const Quadrature<dim> quadrature(unit_support_points);

  FEValues<dim> fe_values(mapping,
                          fe,
                          quadrature,
                          update_gradients);

  const unsigned int n_q_points = quadrature.size();

  std::vector<Tensor<2, dim>> velocity_gradients(n_q_points);

  std::vector<types::global_dof_index> local_dof_indices(
    fe.n_dofs_per_cell());

  const unsigned int qcriterion_component =
    qcriterion_output_extractor.component;

  /*
   * local scalar dof -> support point index.
   */
  std::vector<std::pair<unsigned int, unsigned int>>
    local_scalar_dof_to_qpoint;

  for (unsigned int local_dof = 0; local_dof < fe.n_dofs_per_cell();
       ++local_dof)
  {
    const auto component_shape =
      fe.system_to_component_index(local_dof);

    const unsigned int component = component_shape.first;
    const unsigned int shape     = component_shape.second;

    if (component == qcriterion_component)
    {
      AssertIndexRange(shape, n_q_points);
      local_scalar_dof_to_qpoint.emplace_back(local_dof, shape);
    }
  }

  /*
   * Loop on owned and ghost cells.
   */
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (!(cell->is_locally_owned() || cell->is_ghost()))
      continue;

    fe_values.reinit(cell);

    fe_values[velocity_extractor].get_function_gradients(solution,
                                                         velocity_gradients);

    cell->get_dof_indices(local_dof_indices);

    for (const auto &[local_dof, q] : local_scalar_dof_to_qpoint)
    {
      const types::global_dof_index global_dof =
        local_dof_indices[local_dof];

      if (!locally_owned_dofs.is_element(global_dof))
        continue;

      const double qcriterion =
        compute_qcriterion_from_velocity_gradient<dim>(
          velocity_gradients[q]);

      add_to_distributed_vector_entry(local_qcriterion_sum,
                                      global_dof,
                                      qcriterion);

      add_to_distributed_vector_entry(local_counter,
                                      global_dof,
                                      1.0);
          }
  }

  local_qcriterion_sum.compress(VectorOperation::add);
  local_counter.compress(VectorOperation::add);

  LA::ParVectorType local_qcriterion;
  local_qcriterion.reinit(locally_owned_dofs, mpi_communicator);
  local_qcriterion = 0.0;

  for (const auto global_dof : locally_owned_dofs)
  {
    const double count = local_counter[global_dof];

    if (count > 0.0)
      local_qcriterion[global_dof] =
        local_qcriterion_sum[global_dof] / count;
  }

  local_qcriterion.compress(VectorOperation::insert);

  qcriterion_dof_vector = local_qcriterion;
}

#endif
