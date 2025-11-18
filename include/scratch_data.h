#ifndef SCRATCH_DATA_H
#define SCRATCH_DATA_H

#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <types.h>

using namespace dealii;

/**
 * reinit functions can be called with a LA::ParVectorType
 * (when computing an analytic matrix), or with the local std::vector
 * of dof values (when computing matrix with finite differences).
 * This checks at compile time that the given vector is either.
 */
template <typename T>
constexpr bool is_supported_vector_v = std::is_same_v<T, LA::ParVectorType> ||
                                       std::is_same_v<T, std::vector<double>>;

/**
 * Scratch data for the incompressible Navier-Stokes solver.
 */
template <int dim>
class ScratchDataNS
{
private:
  const UpdateFlags required_updates = update_values | update_gradients |
                                       update_quadrature_points |
                                       update_JxW_values | update_jacobians;
  const UpdateFlags required_face_updates =
    update_values | update_gradients | update_quadrature_points |
    update_JxW_values | update_jacobians | update_normal_vectors;

public:
  /**
   * Constructor
   */
  ScratchDataNS(const FESystem<dim>        &fe,
                const Quadrature<dim>      &cell_quadrature,
                const Mapping<dim>         &mapping,
                const Quadrature<dim - 1>  &face_quadrature,
                const unsigned int          dofs_per_cell,
                const std::vector<double>  &bdfCoeffs,
                const ParameterReader<dim> &param)
    : fe_values(mapping, fe, cell_quadrature, required_updates)
    , fe_face_values(mapping, fe, face_quadrature, required_face_updates)
    , n_q_points(cell_quadrature.size())

    // We assume simplicial meshes with all tris or tets
    , n_faces((dim == 3) ? 4 : 3)

    , n_faces_q_points(face_quadrature.size())
    , dofs_per_cell(dofs_per_cell)
    , bdfCoeffs(bdfCoeffs)
  {
    velocity.first_vector_component = u_lower;
    pressure.component              = p_lower;
    this->allocate();

    // Check if weak forms are to be assembled on boundaries
    has_boundary_forms = false;
    for (const auto &[id, bc] : param.fluid_bc)
    {
      if (bc.type == BoundaryConditions::Type::open_mms)
      {
        has_boundary_forms = true;
        break;
      }
    }
  }

  /**
   * Copy constructor. Needed to use WorkStreams: FEValues must be created "by
   * hand" because their copy constructor is deleted to avoid involuntary
   * expensive copies.
   */
  ScratchDataNS(const ScratchDataNS &other)
    : has_boundary_forms(other.has_boundary_forms)
    , fe_values(other.fe_values.get_mapping(),
                other.fe_values.get_fe(),
                other.fe_values.get_quadrature(),
                required_updates)
    , fe_face_values(other.fe_face_values.get_mapping(),
                     other.fe_face_values.get_fe(),
                     other.fe_face_values.get_quadrature(),
                     required_face_updates)
    , n_q_points(other.n_q_points)
    , n_faces(other.n_faces)
    , n_faces_q_points(other.n_faces_q_points)
    , dofs_per_cell(other.dofs_per_cell)
    , bdfCoeffs(other.bdfCoeffs)
  {
    velocity.first_vector_component = u_lower;
    pressure.component              = p_lower;
    this->allocate();
  }

  void allocate()
  {
    JxW.resize(n_q_points);
    components.resize(dofs_per_cell);

    present_velocity_values.resize(n_q_points);
    present_velocity_gradients.resize(n_q_points);
    present_pressure_values.resize(n_q_points);
    // BDF
    previous_velocity_values.resize(bdfCoeffs.size() - 1,
                                    std::vector<Tensor<1, dim>>(n_q_points));

    source_term_full.resize(n_q_points, Vector<double>(n_components));
    source_term_velocity.resize(n_q_points);
    source_term_pressure.resize(n_q_points);

    grad_source_term_full.resize(n_q_points,
                                 std::vector<Tensor<1, dim>>(n_components));
    grad_source_velocity.resize(n_q_points);
    grad_source_pressure.resize(n_q_points);

    phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_u.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_u.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_p.resize(n_q_points, std::vector<double>(dofs_per_cell));

    // Faces
    face_boundary_id.resize(n_faces);
    face_JxW.resize(n_faces, std::vector<double>(n_faces_q_points));
    face_normals.resize(n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

    // present_face_velocity_gradients.resize(
    //   n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
    // present_face_pressure_values.resize(n_faces,
    //                                     std::vector<double>(n_faces_q_points));

    exact_solution_full.resize(n_faces_q_points, Vector<double>(n_components));
    grad_exact_solution_full.resize(n_faces_q_points,
                                    std::vector<Tensor<1, dim>>(n_components));
    exact_face_velocity_gradients.resize(
      n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
    exact_face_pressure_values.resize(n_faces,
                                      std::vector<double>(n_faces_q_points));

    phi_u_face.resize(n_faces,
                      std::vector<std::vector<Tensor<1, dim>>>(
                        n_faces_q_points,
                        std::vector<Tensor<1, dim>>(dofs_per_cell)));
  }

  template <typename VectorType1, typename VectorType2>
  void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
              const VectorType1                    &current_solution,
              const std::vector<VectorType2>       &previous_solutions,
              const std::shared_ptr<Function<dim>> &source_terms,
              const std::shared_ptr<Function<dim>> &exact_solution)
  {
    static_assert(is_supported_vector_v<VectorType1>,
                  "reinit expects the current_solution to be either a deal.II "
                  "wrapper to a PETSc or Trilinos vector, or an std::vector");

    fe_values.reinit(cell);

    for (const unsigned int i : fe_values.dof_indices())
      components[i] = fe_values.get_fe().system_to_component_index(i).first;

    //
    // Volume-related quantities
    //
    if constexpr (std::is_same<VectorType1, std::vector<double>>::value)
    {
      // Evaluate solution with local dof values (when computing finite
      // differences)
      fe_values[velocity].get_function_values_from_local_dof_values(
        current_solution, present_velocity_values);
      fe_values[velocity].get_function_gradients_from_local_dof_values(
        current_solution, present_velocity_gradients);
      fe_values[pressure].get_function_values_from_local_dof_values(
        current_solution, present_pressure_values);
    }
    else
    {
      // Evaluate solution with full solution vector
      fe_values[velocity].get_function_values(current_solution,
                                              present_velocity_values);
      fe_values[velocity].get_function_gradients(current_solution,
                                                 present_velocity_gradients);
      fe_values[pressure].get_function_values(current_solution,
                                              present_pressure_values);
    }

    // Previous solutions
    for (unsigned int i = 0; i < previous_solutions.size(); ++i)
    {
      fe_values[velocity].get_function_values(previous_solutions[i],
                                              previous_velocity_values[i]);
    }

    // Source terms with layout u-v-(w-)p
    source_terms->vector_value_list(fe_values.get_quadrature_points(),
                                    source_term_full);

    // Get jacobian, shape functions and set source terms
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      JxW[q] = fe_values.JxW(q);

      for (int d = 0; d < dim; ++d)
        source_term_velocity[q][d] = source_term_full[q](u_lower + d);
      source_term_pressure[q] = source_term_full[q](p_lower);

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_u[q][k]      = fe_values[velocity].value(k, q);
        grad_phi_u[q][k] = fe_values[velocity].gradient(k, q);
        div_phi_u[q][k]  = fe_values[velocity].divergence(k, q);
        phi_p[q][k]      = fe_values[pressure].value(k, q);
      }
    }

    //
    // Face-related quantities
    //
    if (has_boundary_forms && cell->at_boundary())
      for (const auto i_face : cell->face_indices())
      {
        face_boundary_id.resize(n_faces);
        face_JxW.resize(n_faces, std::vector<double>(n_faces_q_points));
        face_normals.resize(n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

        exact_solution_full.resize(n_faces_q_points, Vector<double>(n_components));
        grad_exact_solution_full.resize(n_faces_q_points,
                                        std::vector<Tensor<1, dim>>(n_components));
        exact_face_velocity_gradients.resize(
          n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
        exact_face_pressure_values.resize(n_faces,
                                          std::vector<double>(n_faces_q_points));

        phi_u_face.resize(n_faces,
                          std::vector<std::vector<Tensor<1, dim>>>(
                            n_faces_q_points,
                            std::vector<Tensor<1, dim>>(dofs_per_cell)));


        const auto &face = cell->face(i_face);
        if (face->at_boundary())
        {
          face_boundary_id[i_face] = face->boundary_id();
          fe_face_values.reinit(cell, face);

          // fe_face_values[velocity].get_function_values(
          //   current_solution, present_face_velocity_values[i_face]);
          // fe_face_values[velocity].get_function_gradients(
          //   current_solution, present_face_velocity_gradients[i_face]);
          // fe_face_values[pressure].get_function_values(
          //   current_solution, present_face_pressure_values[i_face]);

          // Exact solution with layout u-v-(w-)p and its gradient
          exact_solution->vector_value_list(
            fe_face_values.get_quadrature_points(), exact_solution_full);
          exact_solution->vector_gradient_list(
            fe_face_values.get_quadrature_points(), grad_exact_solution_full);

          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            face_JxW[i_face][q]     = fe_face_values.JxW(q);
            face_normals[i_face][q] = fe_face_values.normal_vector(q);

            for (int di = 0; di < dim; ++di)
              for (int dj = 0; dj < dim; ++dj)
                exact_face_velocity_gradients[i_face][q][di][dj] =
                  grad_exact_solution_full[q][u_lower + di][dj];
            exact_face_pressure_values[i_face][q] =
              exact_solution_full[q](p_lower);

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
            {
              phi_u_face[i_face][q][k] = fe_face_values[velocity].value(k, q);
            }
          }
        }
      }
  }

public:
  const unsigned int n_components = dim + 1;
  const unsigned int u_lower      = 0;
  const unsigned int p_lower      = dim;

  FEValuesExtractors::Vector velocity;
  FEValuesExtractors::Scalar pressure;

public:
  bool has_boundary_forms;

  FEValues<dim>     fe_values;
  FEFaceValues<dim> fe_face_values;

  const unsigned int n_q_points;
  const unsigned int n_faces;
  const unsigned int n_faces_q_points;
  const unsigned int dofs_per_cell;

  const std::vector<double> bdfCoeffs;

  std::vector<double>       JxW;
  std::vector<unsigned int> components;

  // Current and previous values and gradients for each quad node
  std::vector<Tensor<1, dim>>              present_velocity_values;
  std::vector<Tensor<2, dim>>              present_velocity_gradients;
  std::vector<double>                      present_pressure_values;
  std::vector<std::vector<Tensor<1, dim>>> previous_velocity_values;

  // Source term on cell
  std::vector<Vector<double>>
    source_term_full; // The source term with n_components
  std::vector<Tensor<1, dim>> source_term_velocity;
  std::vector<double>         source_term_pressure;

  // // Gradient of source term,
  // // at each quad node, for each dof component, result is a Tensor<1, dim>
  std::vector<std::vector<Tensor<1, dim>>> grad_source_term_full;
  std::vector<Tensor<2, dim>>              grad_source_velocity;
  std::vector<Tensor<1, dim>>              grad_source_pressure;

  // Shape functions and gradients for each quad node and each dof
  std::vector<std::vector<Tensor<1, dim>>> phi_u;
  std::vector<std::vector<Tensor<2, dim>>> grad_phi_u;
  std::vector<std::vector<double>>         div_phi_u;
  std::vector<std::vector<double>>         phi_p;

  //
  // Faces
  //
  std::vector<unsigned int>                face_boundary_id;
  std::vector<std::vector<double>>         face_JxW;
  std::vector<std::vector<Tensor<1, dim>>> face_normals;

  // Current and previous values on faces
  // std::vector<std::vector<Tensor<1, dim>>> present_face_velocity_values;
  // std::vector<std::vector<Tensor<2, dim>>> present_face_velocity_gradients;
  // std::vector<std::vector<double>>         present_face_pressure_values;

  std::vector<Vector<double>>              exact_solution_full;
  std::vector<std::vector<Tensor<1, dim>>> grad_exact_solution_full;
  std::vector<std::vector<Tensor<2, dim>>> exact_face_velocity_gradients;
  std::vector<std::vector<double>>         exact_face_pressure_values;

  // Shape functions on faces, for each each quad node and each dof
  std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_u_face;
};

/**
 * Scratch data for the monolithic fluid-structure interaction solver.
 */
template <int dim>
class ScratchDataMonolithicFSI
{
private:
  const UpdateFlags required_updates = update_values | update_gradients |
                                       update_quadrature_points |
                                       update_JxW_values | update_jacobians;
  const UpdateFlags required_face_updates =
    update_values | update_gradients | update_quadrature_points |
    update_JxW_values | update_jacobians | update_normal_vectors;

public:
  ScratchDataMonolithicFSI(const FESystem<dim>       &fe,
                           const Quadrature<dim>     &cell_quadrature,
                           const Mapping<dim>        &fixed_mapping,
                           const Mapping<dim>        &mapping,
                           const Quadrature<dim - 1> &face_quadrature,
                           const unsigned int         dofs_per_cell,
                           const unsigned int         boundary_id,
                           const std::vector<double> &bdfCoeffs)
    : fe_values(mapping,
                fe,
                cell_quadrature,
                required_updates)
                // update_values | update_gradients | update_quadrature_points |
                //   update_JxW_values | update_jacobians |
                //   update_inverse_jacobians)
    , fe_values_fixed(fixed_mapping,
                      fe,
                      cell_quadrature,
                      required_updates)
                      // update_values | update_gradients |
                      //   update_quadrature_points | update_JxW_values |
                      //   update_jacobians | update_inverse_jacobians)
    , fe_face_values(mapping,
                     fe,
                     face_quadrature,
                     required_face_updates)
                     // update_values | update_gradients |
                     //   update_quadrature_points | update_JxW_values |
                     //   update_jacobians | update_inverse_jacobians)
    , fe_face_values_fixed(fixed_mapping,
                           fe,
                           face_quadrature,
                           required_face_updates)
                           // update_values | update_gradients |
                           //   update_quadrature_points | update_JxW_values |
                           //   update_jacobians | update_inverse_jacobians)
    , n_q_points(cell_quadrature.size())

    // We assume that simplicial meshes with all tris or tets
    , n_faces((dim == 3) ? 4 : 3)

    , n_faces_q_points(face_quadrature.size())
    , dofs_per_cell(dofs_per_cell)
    , boundary_id(boundary_id)
    , bdfCoeffs(bdfCoeffs)
  {
    this->allocate();
  }

  /**
   * Copy constructor. Needed to use WorkStreams.
   */
  ScratchDataMonolithicFSI(const ScratchDataMonolithicFSI &other)
    : fe_values(other.fe_values.get_mapping(),
                other.fe_values.get_fe(),
                other.fe_values.get_quadrature(),
                required_updates)
    , fe_values_fixed(other.fe_values_fixed.get_mapping(),
                other.fe_values_fixed.get_fe(),
                other.fe_values_fixed.get_quadrature(),
                required_updates)
    , fe_face_values(other.fe_face_values.get_mapping(),
                     other.fe_face_values.get_fe(),
                     other.fe_face_values.get_quadrature(),
                     required_face_updates)
    , fe_face_values_fixed(other.fe_face_values_fixed.get_mapping(),
                     other.fe_face_values_fixed.get_fe(),
                     other.fe_face_values_fixed.get_quadrature(),
                     required_face_updates)
    , n_q_points(other.n_q_points)
    , n_faces(other.n_faces)
    , n_faces_q_points(other.n_faces_q_points)
    , dofs_per_cell(other.dofs_per_cell)
    , boundary_id(other.boundary_id)
    , bdfCoeffs(other.bdfCoeffs)
  {
    this->allocate();
  }

  void allocate()
  {
    components.resize(dofs_per_cell);

    JxW_moving.resize(n_q_points);
    JxW_fixed.resize(n_q_points);

    present_velocity_values.resize(n_q_points);
    present_velocity_gradients.resize(n_q_points);
    present_pressure_values.resize(n_q_points);
    present_position_values.resize(n_q_points);
    present_position_gradients.resize(n_q_points);
    present_mesh_velocity_values.resize(n_q_points);

    present_face_velocity_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    present_face_position_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    present_face_position_gradient.resize(
      n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
    present_face_lambda_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    present_face_mesh_velocity_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

    source_term_full.resize(n_q_points, Vector<double>(n_components));
    source_term_velocity.resize(n_q_points);
    source_term_pressure.resize(n_q_points);
    source_term_position.resize(n_q_points);

    grad_source_term_full.resize(n_q_points,
                                 std::vector<Tensor<1, dim>>(n_components));
    grad_source_velocity.resize(n_q_points);
    grad_source_pressure.resize(n_q_points);

    // BDF
    previous_velocity_values.resize(bdfCoeffs.size() - 1,
                                    std::vector<Tensor<1, dim>>(n_q_points));
    previous_position_values.resize(bdfCoeffs.size() - 1,
                                    std::vector<Tensor<1, dim>>(n_q_points));
    previous_face_position_values.resize(
      n_faces,
      std::vector<std::vector<Tensor<1, dim>>>(
        bdfCoeffs.size() - 1, std::vector<Tensor<1, dim>>(n_faces_q_points)));

    phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_u.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_u.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_p.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_x.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_x.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_x.resize(n_q_points, std::vector<double>(dofs_per_cell));

    phi_u_face.resize(n_faces,
                      std::vector<std::vector<Tensor<1, dim>>>(
                        n_faces_q_points,
                        std::vector<Tensor<1, dim>>(dofs_per_cell)));

    phi_x_face.resize(n_faces,
                      std::vector<std::vector<Tensor<1, dim>>>(
                        n_faces_q_points,
                        std::vector<Tensor<1, dim>>(dofs_per_cell)));

    grad_phi_x_face.resize(n_faces,
                           std::vector<std::vector<Tensor<2, dim>>>(
                             n_faces_q_points,
                             std::vector<Tensor<2, dim>>(dofs_per_cell)));

    phi_l_face.resize(n_faces,
                      std::vector<std::vector<Tensor<1, dim>>>(
                        n_faces_q_points,
                        std::vector<Tensor<1, dim>>(dofs_per_cell)));

    face_JxW_moving.resize(n_faces, std::vector<double>(n_faces_q_points));
    face_JxW_fixed.resize(n_faces, std::vector<double>(n_faces_q_points));

    face_G.resize(n_faces, std::vector<Tensor<2, dim - 1>>(n_faces_q_points));
    delta_dx.resize(n_faces,
                    std::vector<std::vector<double>>(
                      n_faces_q_points, std::vector<double>(dofs_per_cell)));
  }

  template <typename VectorType1, typename VectorType2>
  void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
              const VectorType1                    &current_solution,
              const std::vector<VectorType2>       &previous_solutions,
              const std::shared_ptr<Function<dim>> &/*source_terms*/,
              const std::shared_ptr<Function<dim>> &/*exact_solution*/)
  {
    static_assert(is_supported_vector_v<VectorType1>,
                  "reinit expects the current_solution to be either a deal.II "
                  "wrapper to a PETSc or Trilinos vector, or an std::vector");

    fe_values.reinit(cell);
    fe_values_fixed.reinit(cell);

    for (const unsigned int i : fe_values.dof_indices())
      components[i] = fe_values.get_fe().system_to_component_index(i).first;

    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Scalar pressure(p_lower);
    const FEValuesExtractors::Vector position(x_lower);
    const FEValuesExtractors::Vector lambda(l_lower);

    //
    // Volume-related quantities
    //
    if constexpr (std::is_same<VectorType1, std::vector<double>>::value)
    {
      //
      // Evaluate velocity and pressure on moving mapping
      //
      fe_values[velocity].get_function_values_from_local_dof_values(
        current_solution, present_velocity_values);
      fe_values[velocity].get_function_gradients_from_local_dof_values(
        current_solution, present_velocity_gradients);
      fe_values[pressure].get_function_values_from_local_dof_values(
        current_solution, present_pressure_values);

      //
      // Evaluate position on fixed mapping
      //
      fe_values_fixed[position].get_function_values_from_local_dof_values(
        current_solution, present_position_values);
      fe_values_fixed[position].get_function_gradients_from_local_dof_values(
        current_solution, present_position_gradients);
    }
    else
    {
      //
      // Evaluate velocity and pressure on moving mapping
      //
      fe_values[velocity].get_function_values(current_solution,
                                              present_velocity_values);
      fe_values[velocity].get_function_gradients(current_solution,
                                                 present_velocity_gradients);
      fe_values[pressure].get_function_values(current_solution,
                                              present_pressure_values);

      //
      // Evaluate position on fixed mapping
      //
      fe_values_fixed[position].get_function_values(current_solution,
                                                    present_position_values);
      fe_values_fixed[position].get_function_gradients(
        current_solution, present_position_gradients);
    }

    // Previous solutions
    for (unsigned int i = 0; i < previous_solutions.size(); ++i)
    {
      fe_values[velocity].get_function_values(previous_solutions[i],
                                              previous_velocity_values[i]);
      fe_values_fixed[position].get_function_values(
        previous_solutions[i], previous_position_values[i]);
    }

    // Current mesh velocity from displacement
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      present_mesh_velocity_values[q] =
        bdfCoeffs[0] * present_position_values[q];
      for (unsigned int iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
      {
        present_mesh_velocity_values[q] +=
          bdfCoeffs[iBDF] * previous_position_values[iBDF - 1][q];
      }
    }

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      JxW_moving[q] = fe_values.JxW(q);
      JxW_fixed[q]  = fe_values_fixed.JxW(q);

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_u[q][k]      = fe_values[velocity].value(k, q);
        grad_phi_u[q][k] = fe_values[velocity].gradient(k, q);
        div_phi_u[q][k]  = fe_values[velocity].divergence(k, q);
        phi_p[q][k]      = fe_values[pressure].value(k, q);

        phi_x[q][k]      = fe_values_fixed[position].value(k, q);
        grad_phi_x[q][k] = fe_values_fixed[position].gradient(k, q);
        div_phi_x[q][k]  = fe_values_fixed[position].divergence(k, q);
      }
    }

    //
    // Face-related values and shape functions,
    // for the faces touching the prescribed boundary_id
    //
    for (const auto i_face : cell->face_indices())
    {
      const auto &face = cell->face(i_face);

      // if (!(face->at_boundary() && face->boundary_id() == boundary_id))
      //   continue;

      fe_face_values.reinit(cell, face);
      fe_face_values_fixed.reinit(cell, face);

      if constexpr (std::is_same<VectorType1, std::vector<double>>::value)
      {
        fe_face_values[velocity].get_function_values_from_local_dof_values(
          current_solution, present_face_velocity_values[i_face]);
        fe_face_values[lambda].get_function_values_from_local_dof_values(
          current_solution, present_face_lambda_values[i_face]);

        fe_face_values_fixed[position]
          .get_function_values_from_local_dof_values(
            current_solution, present_face_position_values[i_face]);
        fe_face_values_fixed[position]
          .get_function_gradients_from_local_dof_values(
            current_solution, present_face_position_gradient[i_face]);
      }
      else
      {
        fe_face_values[velocity].get_function_values(
          current_solution, present_face_velocity_values[i_face]);
        fe_face_values[lambda].get_function_values(
          current_solution, present_face_lambda_values[i_face]);

        fe_face_values_fixed[position].get_function_values(
          current_solution, present_face_position_values[i_face]);
        fe_face_values_fixed[position].get_function_gradients(
          current_solution, present_face_position_gradient[i_face]);
      }

      for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      {
        fe_face_values_fixed[position].get_function_values(
          previous_solutions[i], previous_face_position_values[i_face][i]);
      }

      for (unsigned int q = 0; q < n_faces_q_points; ++q)
      {
        face_JxW_moving[i_face][q] = fe_face_values.JxW(q);
        face_JxW_fixed[i_face][q]  = fe_face_values_fixed.JxW(q);

        //
        // Jacobian of geometric transformation is needed to compute
        // tangent metric for the lambda equation, on moving mapping
        //
        const Tensor<2, dim> J = fe_face_values.jacobian(q);

        if constexpr (dim == 2)
        {
          switch (i_face)
          {
            case 0:
              dxsids_array[0][0] = 1.;
              dxsids_array[0][1] = 0.;
              break;
            case 1:
              dxsids_array[0][0] = -1.;
              dxsids_array[0][1] = 1.;
              break;
            case 2:
              dxsids_array[0][0] = 0.;
              dxsids_array[0][1] = -1.;
              break;
            default:
              DEAL_II_ASSERT_UNREACHABLE();
          }
        }
        else
        {
          switch (i_face)
          {
            // Using dealii's face ordering
            case 3: // Opposite to v0
              dxsids_array[0][0] = -1.;
              dxsids_array[1][0] = -1.;
              dxsids_array[0][1] = 1.;
              dxsids_array[1][1] = 0.;
              dxsids_array[0][2] = 0.;
              dxsids_array[1][2] = 1.;
              break;
            case 2: // Opposite to v1
              dxsids_array[0][0] = 0.;
              dxsids_array[1][0] = 0.;
              dxsids_array[0][1] = 1.;
              dxsids_array[1][1] = 0.;
              dxsids_array[0][2] = 0.;
              dxsids_array[1][2] = 1.;
              break;
            case 1: // Opposite to v2
              dxsids_array[0][0] = 1.;
              dxsids_array[1][0] = 0.;
              dxsids_array[0][1] = 0.;
              dxsids_array[1][1] = 0.;
              dxsids_array[0][2] = 0.;
              dxsids_array[1][2] = 1.;
              break;
            case 0: // Opposite to v3
              dxsids_array[0][0] = 1.;
              dxsids_array[1][0] = 0.;
              dxsids_array[0][1] = 0.;
              dxsids_array[1][1] = 1.;
              dxsids_array[0][2] = 0.;
              dxsids_array[1][2] = 0.;
              break;
            default:
              DEAL_II_ASSERT_UNREACHABLE();
          }
        }

        Tensor<2, dim - 1> G;
        G = 0;
        for (unsigned int di = 0; di < dim - 1; ++di)
          for (unsigned int dj = 0; dj < dim - 1; ++dj)
            for (unsigned int im = 0; im < dim; ++im)
              for (unsigned int in = 0; in < dim; ++in)
                for (unsigned int ip = 0; ip < dim; ++ip)
                  G[di][dj] += dxsids_array[di][im] * J[in][im] * J[in][ip] *
                               dxsids_array[dj][ip];
        face_G[i_face][q]                  = G;
        const Tensor<2, dim - 1> G_inverse = invert(G);

        // Result of G^(-1) * (J * dxsids)^T * grad_phi_x_j * dxsids
        Tensor<2, dim - 1> res;

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_u_face[i_face][q][k] = fe_face_values[velocity].value(k, q);
          phi_l_face[i_face][q][k] = fe_face_values[lambda].value(k, q);

          phi_x_face[i_face][q][k] = fe_face_values_fixed[position].value(k, q);
          grad_phi_x_face[i_face][q][k] =
            fe_face_values_fixed[position].gradient(k, q);

          const auto &grad_phi_x = grad_phi_x_face[i_face][q][k];

          Tensor<2, dim> A =
            transpose(J) *
            (transpose(present_face_position_gradient[i_face][q]) * grad_phi_x +
             transpose(grad_phi_x) *
               present_face_position_gradient[i_face][q]) *
            J;

          res = 0;
          for (unsigned int di = 0; di < dim - 1; ++di)
            for (unsigned int dj = 0; dj < dim - 1; ++dj)
              for (unsigned int im = 0; im < dim - 1; ++im)
                for (unsigned int in = 0; in < dim; ++in)
                  for (unsigned int io = 0; io < dim; ++io)
                    res[di][dj] += G_inverse[di][im] * dxsids_array[im][in] *
                                   A[in][io] * dxsids_array[dj][io];
          delta_dx[i_face][q][k] =
            0.5 * trace(res); // Choose this if multiplying by JxW in the matrix
          // delta_dx[i_face][q][k] = 0.5 * sqrt_det_G * trace(res); // Choose
          // this if multiplying by W
        }

        // Face mesh velocity
        present_face_mesh_velocity_values[i_face][q] =
          bdfCoeffs[0] * present_face_position_values[i_face][q];
        for (unsigned int iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
        {
          present_face_mesh_velocity_values[i_face][q] +=
            bdfCoeffs[iBDF] *
            previous_face_position_values[i_face][iBDF - 1][q];
        }
      }
    }
  }

public:
  const unsigned int n_components = 3 * dim + 1;
  const unsigned int u_lower      = 0;
  const unsigned int p_lower      = dim;
  const unsigned int x_lower      = dim + 1;
  const unsigned int l_lower      = 2 * dim + 1;

public:
  FEValues<dim> fe_values;
  FEValues<dim> fe_values_fixed;

  FEFaceValues<dim> fe_face_values;
  FEFaceValues<dim> fe_face_values_fixed;

  const unsigned int n_q_points;
  const unsigned int n_faces;
  const unsigned int n_faces_q_points;
  const unsigned int dofs_per_cell;

  // The tag of the boundary on which weak Dirichlet BC are
  // applied with Lagrange multiplier. Only 1 for now.
  const unsigned int boundary_id;

  const std::vector<double> bdfCoeffs;

  std::vector<double>              JxW_moving;
  std::vector<double>              JxW_fixed;
  std::vector<std::vector<double>> face_JxW_moving;
  std::vector<std::vector<double>> face_JxW_fixed;

  // The reference jacobians partial xsi_dim/partial xsi_(dim-1)
  std::array<Tensor<1, dim>, dim - 1>          dxsids_array;
  std::vector<std::vector<Tensor<2, dim - 1>>> face_G;

  // At face x quad node x phi_position_j
  std::vector<std::vector<std::vector<double>>> delta_dx;

  std::vector<unsigned int> components;

  // Current and previous values and gradients for each quad node
  std::vector<Tensor<1, dim>>              present_velocity_values;
  std::vector<Tensor<2, dim>>              present_velocity_gradients;
  std::vector<double>                      present_pressure_values;
  std::vector<std::vector<Tensor<1, dim>>> previous_velocity_values;

  std::vector<Tensor<1, dim>>              present_position_values;
  std::vector<Tensor<2, dim>>              present_position_gradients;
  std::vector<std::vector<Tensor<1, dim>>> previous_position_values;
  std::vector<Tensor<1, dim>>              present_mesh_velocity_values;

  // Current and previous values on faces
  std::vector<std::vector<Tensor<1, dim>>> present_face_velocity_values;
  std::vector<std::vector<Tensor<1, dim>>> present_face_position_values;
  std::vector<std::vector<Tensor<2, dim>>> present_face_position_gradient;
  std::vector<std::vector<Tensor<1, dim>>> present_face_mesh_velocity_values;
  std::vector<std::vector<Tensor<1, dim>>> present_face_lambda_values;
  std::vector<std::vector<std::vector<Tensor<1, dim>>>>
    previous_face_position_values;

  // Source term on cell
  std::vector<Vector<double>>
    source_term_full; // The source term with n_components
  std::vector<Tensor<1, dim>> source_term_velocity;
  std::vector<double>         source_term_pressure;
  std::vector<Tensor<1, dim>> source_term_position;

  // // Gradient of source term,
  // // at each quad node, for each dof component, result is a Tensor<1, dim>
  std::vector<std::vector<Tensor<1, dim>>> grad_source_term_full;
  std::vector<Tensor<2, dim>>              grad_source_velocity;
  std::vector<Tensor<1, dim>>              grad_source_pressure;

  // Shape functions and gradients for each quad node and each dof
  std::vector<std::vector<Tensor<1, dim>>> phi_u;
  std::vector<std::vector<Tensor<2, dim>>> grad_phi_u;
  std::vector<std::vector<double>>         div_phi_u;
  std::vector<std::vector<double>>         phi_p;
  std::vector<std::vector<Tensor<1, dim>>> phi_x;
  std::vector<std::vector<Tensor<2, dim>>> grad_phi_x;
  std::vector<std::vector<double>>         div_phi_x;

  // Shape functions on faces for relevant faces, each quad node and each dof
  std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_u_face;
  std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_x_face;
  std::vector<std::vector<std::vector<Tensor<2, dim>>>> grad_phi_x_face;
  std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_l_face;
};

#endif