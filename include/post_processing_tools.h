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

    FaceDescriptor first_face() override;

    /**
     * Return the next face after which we want output for. If there are no more
     * faces, <tt>dofs->end()</tt> is returned as the first component of the
     * return value.
     */
    FaceDescriptor next_face(const FaceDescriptor &face) override;

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
    const FiniteElement<dim>         &fe,
    const Mapping<dim>               &mapping,
    const Quadrature<dim - 1>        &face_quadrature,
    const VectorType                 &solution,
    const types::boundary_id          boundary_id,
    const FEValuesExtractors::Vector &velocity_extractor,
    const FEValuesExtractors::Scalar &pressure_extractor,
    const double                      dynamic_viscosity);

  /**
   * hp-version of the function above
   */
  template <int dim, typename VectorType>
  Tensor<1, dim> compute_forces_on_boundary(
    const DoFHandler<dim>            &dof_handler,
    const hp::FECollection<dim>      &fe_collection,
    const hp::MappingCollection<dim> &mapping_collection,
    const hp::QCollection<dim - 1>   &face_quadrature_collection,
    const VectorType                 &solution,
    const types::boundary_id          boundary_id,
    const FEValuesExtractors::Vector &velocity_extractor,
    const FEValuesExtractors::Scalar &pressure_extractor,
    const double                      dynamic_viscosity);

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
    const FiniteElement<dim>         &fe,
    const Mapping<dim>               &mapping,
    const Quadrature<dim - 1>        &face_quadrature,
    const VectorType                 &solution,
    const types::boundary_id          boundary_id,
    const FEValuesExtractors::Vector &lambda_extractor);

  /**
   * hp-version of the function above
   */
  template <int dim, typename VectorType>
  Tensor<1, dim> compute_forces_on_boundary_with_lagrange_multiplier(
    const DoFHandler<dim>            &dof_handler,
    const hp::FECollection<dim>      &fe_collection,
    const hp::MappingCollection<dim> &mapping_collection,
    const hp::QCollection<dim - 1>   &face_quadrature_collection,
    const VectorType                 &solution,
    const types::boundary_id          boundary_id,
    const FEValuesExtractors::Vector &lambda_extractor);


  enum class SliceAxis : unsigned int
  {
    x = 0,
    y = 1,
    z = 2
  };


  /**
   * Compute slice indices for degrees of freedom located on a given boundary.
   *
   *  The boundary @p boundary_id is partitioned into @p n_slices along the
   * coordinate direction @p axis. Each degree of freedom on this boundary
   * is assigned to one slice based on its geometric position.
   *
   * The function returns a Vector<double> containing the slice index
   * associated with each degree of freedom. A floating-point vector is used
   * for compatibility with deal.II data structures and post-processing.
   *
   * @return Vector of slice indices for the specified boundary.
   */
  template <int dim>
  Vector<double>
  compute_slice_index_on_boundary(const DoFHandler<dim>   &dof_handler,
                                  const types::boundary_id boundary_id,
                                  const unsigned int       n_slices,
                                  const SliceAxis          axis,
                                  const MPI_Comm           mpi_comm);

} // namespace PostProcessingTools

/* ---------------- Template functions ----------------- */

template <int dim, typename VectorType>
Tensor<1, dim> PostProcessingTools::compute_forces_on_boundary(
  const DoFHandler<dim>            &dof_handler,
  const FiniteElement<dim>         &fe,
  const Mapping<dim>               &mapping,
  const Quadrature<dim - 1>        &face_quadrature,
  const VectorType                 &solution,
  const types::boundary_id          boundary_id,
  const FEValuesExtractors::Vector &velocity_extractor,
  const FEValuesExtractors::Scalar &pressure_extractor,
  const double                      dynamic_viscosity)
{
  Tensor<1, dim> forces, forces_local;
  const double   mu = dynamic_viscosity;

  FEFaceValues<dim> fe_face_values(mapping,
                                   fe,
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
            forces_local += sigma_dot_n * fe_face_values.JxW(q);
          }
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
  const hp::FECollection<dim>      &fe_collection,
  const hp::MappingCollection<dim> &mapping_collection,
  const hp::QCollection<dim - 1>   &face_quadrature_collection,
  const VectorType                 &solution,
  const types::boundary_id          boundary_id,
  const FEValuesExtractors::Vector &velocity_extractor,
  const FEValuesExtractors::Scalar &pressure_extractor,
  const double                      dynamic_viscosity)
{
  Tensor<1, dim> forces, forces_local;
  const double   mu = dynamic_viscosity;

  hp::FEFaceValues<dim> hp_fe_face_values(mapping_collection,
                                          fe_collection,
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
            forces_local += sigma_dot_n * fe_face_values.JxW(q);
          }
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
  const FiniteElement<dim>         &fe,
  const Mapping<dim>               &mapping,
  const Quadrature<dim - 1>        &face_quadrature,
  const VectorType                 &solution,
  const types::boundary_id          boundary_id,
  const FEValuesExtractors::Vector &lambda_extractor)
{
  Tensor<1, dim> lambda_integral, lambda_integral_local;

  FEFaceValues<dim> fe_face_values(mapping,
                                   fe,
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
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
            lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
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
  const hp::FECollection<dim>      &fe_collection,
  const hp::MappingCollection<dim> &mapping_collection,
  const hp::QCollection<dim - 1>   &face_quadrature_collection,
  const VectorType                 &solution,
  const types::boundary_id          boundary_id,
  const FEValuesExtractors::Vector &lambda_extractor)
{
  Tensor<1, dim> lambda_integral, lambda_integral_local;

  hp::FEFaceValues hp_fe_face_values(mapping_collection,
                                     fe_collection,
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
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
            lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
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

#endif
