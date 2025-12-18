#ifndef SCRATCH_DATA_BASE_H
#define SCRATCH_DATA_BASE_H

#include <cahn_hilliard.h>
#include <components_ordering.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <parameter_reader.h>
#include <types.h>

using namespace dealii;

/**
 * The scratch data are helper structures containing all sorts of quantities
 * that must be computed on each elements and faces during assembly, e.g.,
 * the current interpolated FE solution and its gradients, among many others.
 */

/**
 * This scratch data contains all the possible fields, and only allocates and
 * reinits depending on which flags are enabled
 */
template <int dim>
class ScratchData
{
public:
  /**
   * Constructor
   */
  ScratchData(const ComponentOrdering    &ordering,
              const bool                  enable_pseudo_solid,
              const bool                  enable_lagrange_multiplier,
              const bool                  enable_cahn_hilliard,
              const FESystem<dim>        &fe,
              const Mapping<dim>         &fixed_mapping,
              const Mapping<dim>         &moving_mapping,
              const Quadrature<dim>      &cell_quadrature,
              const Quadrature<dim - 1>  &face_quadrature,
              const std::vector<double>  &bdf_coefficients,
              const ParameterReader<dim> &param);

  /**
   * Copy constructor
   */
  ScratchData(const ScratchData &other);

private:
  void allocate();

  void initialize_navier_stokes();
  void initialize_pseudo_solid();
  void initialize_lagrange_multiplier();
  void initialize_cahn_hilliard();

  template <typename VectorType>
  void reinit_navier_stokes_cell(
    const VectorType                     &current_solution,
    const std::vector<VectorType>        &previous_solutions,
    const std::shared_ptr<Function<dim>> &source_terms,
    const std::shared_ptr<Function<dim>> &exact_solution)
  {
    fe_values[velocity].get_function_values(current_solution,
                                            present_velocity_values);
    fe_values[velocity].get_function_gradients(current_solution,
                                               present_velocity_gradients);
    fe_values[pressure].get_function_values(current_solution,
                                            present_pressure_values);

    // Previous solutions
    for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      fe_values[velocity].get_function_values(previous_solutions[i],
                                              previous_velocity_values[i]);

    // Source terms with layout u-v-(w-)p
    source_terms->vector_value_list(fe_values.get_quadrature_points(),
                                    source_term_full_moving);

    // Get jacobian, shape functions and set source terms
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      JxW_moving[q] = fe_values.JxW(q);

      present_velocity_sym_gradients[q] =
        symmetrize(present_velocity_gradients[q]);
      present_velocity_divergence[q] = trace(present_velocity_gradients[q]);

      for (int d = 0; d < dim; ++d)
        source_term_velocity[q][d] = source_term_full_moving[q](u_lower + d);
      source_term_pressure[q] = source_term_full_moving[q](p_lower);

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_u[q][k]          = fe_values[velocity].value(k, q);
        grad_phi_u[q][k]     = fe_values[velocity].gradient(k, q);
        sym_grad_phi_u[q][k] = symmetrize(grad_phi_u[q][k]);
        div_phi_u[q][k]      = fe_values[velocity].divergence(k, q);
        phi_p[q][k]          = fe_values[pressure].value(k, q);
      }
    }
  }

  template <typename VectorType>
  void reinit_navier_stokes_face(
    const unsigned int                    i_face,
    const VectorType                     &current_solution,
    const std::vector<VectorType>        &previous_solutions,
    const std::shared_ptr<Function<dim>> &source_terms,
    const std::shared_ptr<Function<dim>> &exact_solution)
  {
    fe_face_values[velocity].get_function_values(
      current_solution, present_face_velocity_values[i_face]);

    // Exact solution with layout u-v-(w-)p and its gradient
    exact_solution->vector_value_list(fe_face_values.get_quadrature_points(),
                                      exact_solution_full);
    exact_solution->vector_gradient_list(fe_face_values.get_quadrature_points(),
                                         grad_exact_solution_full);

    for (unsigned int q = 0; q < n_faces_q_points; ++q)
    {
      face_JxW_moving[i_face][q]     = fe_face_values.JxW(q);
      face_normals_moving[i_face][q] = fe_face_values.normal_vector(q);

      for (int di = 0; di < dim; ++di)
        for (int dj = 0; dj < dim; ++dj)
          exact_face_velocity_gradients[i_face][q][di][dj] =
            grad_exact_solution_full[q][u_lower + di][dj];
      exact_face_pressure_values[i_face][q] = exact_solution_full[q](p_lower);

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_u_face[i_face][q][k] = fe_face_values[velocity].value(k, q);
      }
    }
  }

  template <typename VectorType>
  void
  reinit_pseudo_solid_cell(const VectorType              &current_solution,
                           const std::vector<VectorType> &previous_solutions,
                           const std::shared_ptr<Function<dim>> &source_terms,
                           const std::shared_ptr<Function<dim>> &exact_solution)
  {
    fe_values_fixed[position].get_function_values(current_solution,
                                                  present_position_values);
    fe_values_fixed[position].get_function_gradients(
      current_solution, present_position_gradients);

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
        bdf_coefficients[0] * present_position_values[q];
      for (unsigned int iBDF = 1; iBDF < bdf_coefficients.size(); ++iBDF)
      {
        present_mesh_velocity_values[q] +=
          bdf_coefficients[iBDF] * previous_position_values[iBDF - 1][q];
      }
    }

    const auto &fixed_quadrature_points =
      fe_values_fixed.get_quadrature_points();

    // Source terms on fixed mapping for x
    source_terms->vector_value_list(fixed_quadrature_points,
                                    source_term_full_fixed);

    // This takes a lot of time, and the Newton solver converges without it
    // // Gradient of source term (for u-p only)
    // source_terms->vector_gradient_list(fe_values.get_quadrature_points(),
    //                                    grad_source_term_full);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      JxW_fixed[q] = fe_values_fixed.JxW(q);

      const Point<dim> &q_point = fixed_quadrature_points[q];
      lame_mu[q] =
        physical_properties.pseudosolids[0].lame_mu_fun->value(q_point);
      lame_lambda[q] =
        physical_properties.pseudosolids[0].lame_lambda_fun->value(q_point);

      AssertThrow(lame_mu[q] >= 0,
                  ExcMessage("Lamé coefficient mu should be positive"));
      AssertThrow(lame_lambda[q] >= 0,
                  ExcMessage("Lamé coefficient lambda should be positive"));

      for (int d = 0; d < dim; ++d)
        source_term_position[q][d] = source_term_full_fixed[q](x_lower + d);

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_x[q][k]      = fe_values_fixed[position].value(k, q);
        grad_phi_x[q][k] = fe_values_fixed[position].gradient(k, q);
        div_phi_x[q][k]  = fe_values_fixed[position].divergence(k, q);
      }
    }
  }

  template <typename VectorType>
  void
  reinit_pseudo_solid_face(const unsigned int             i_face,
                           const VectorType              &current_solution,
                           const std::vector<VectorType> &previous_solutions,
                           const std::shared_ptr<Function<dim>> &source_terms,
                           const std::shared_ptr<Function<dim>> &exact_solution)
  {
    fe_face_values_fixed[position].get_function_values(
      current_solution, present_face_position_values[i_face]);
    fe_face_values_fixed[position].get_function_gradients(
      current_solution, present_face_position_gradient[i_face]);

    for (unsigned int i = 0; i < previous_solutions.size(); ++i)
    {
      fe_face_values_fixed[position].get_function_values(
        previous_solutions[i], previous_face_position_values[i_face][i]);
    }

    for (unsigned int q = 0; q < n_faces_q_points; ++q)
    {
      face_JxW_fixed[i_face][q] = fe_face_values_fixed.JxW(q);

      //
      // Jacobian of geometric transformation is needed to compute
      // tangent matrix for the lambda equation, on moving mapping
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
      const Tensor<2, dim - 1> G_inverse = invert(G);

      // Result of G^(-1) * (J * dxsids)^T * grad_phi_x_j * dxsids
      Tensor<2, dim - 1> res;

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_x_face[i_face][q][k] = fe_face_values_fixed[position].value(k, q);
        grad_phi_x_face[i_face][q][k] =
          fe_face_values_fixed[position].gradient(k, q);

        const auto &grad_phi_x = grad_phi_x_face[i_face][q][k];

        Tensor<2, dim> A =
          transpose(J) *
          (transpose(present_face_position_gradient[i_face][q]) * grad_phi_x +
           transpose(grad_phi_x) * present_face_position_gradient[i_face][q]) *
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
        bdf_coefficients[0] * present_face_position_values[i_face][q];
      for (unsigned int iBDF = 1; iBDF < bdf_coefficients.size(); ++iBDF)
      {
        present_face_mesh_velocity_values[i_face][q] +=
          bdf_coefficients[iBDF] *
          previous_face_position_values[i_face][iBDF - 1][q];
      }
    }
  }

  template <typename VectorType>
  void reinit_lagrange_multiplier_face(
    const unsigned int                    i_face,
    const VectorType                     &current_solution,
    const std::vector<VectorType>        &previous_solutions,
    const std::shared_ptr<Function<dim>> &source_terms,
    const std::shared_ptr<Function<dim>> &exact_solution)
  {
    fe_face_values[lambda].get_function_values(
      current_solution, present_face_lambda_values[i_face]);

    for (unsigned int q = 0; q < n_faces_q_points; ++q)
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
        phi_l_face[i_face][q][k] = fe_face_values[lambda].value(k, q);
  }

  template <typename VectorType>
  void reinit_cahn_hilliard_cell(
    const VectorType                     &current_solution,
    const std::vector<VectorType>        &previous_solutions,
    const std::shared_ptr<Function<dim>> &source_terms,
    const std::shared_ptr<Function<dim>> &exact_solution)
  {
    fe_values[tracer].get_function_values(current_solution, tracer_values);
    fe_values[tracer].get_function_gradients(current_solution,
                                             tracer_gradients);
    fe_values[potential].get_function_values(current_solution,
                                             potential_values);
    fe_values[potential].get_function_gradients(current_solution,
                                                potential_gradients);
    // Previous solutions
    for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      fe_values[tracer].get_function_values(previous_solutions[i],
                                            previous_tracer_values[i]);

    source_terms->vector_value_list(fe_values.get_quadrature_points(),
                                    source_term_full_moving);

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      // Physical properties based on tracer, filter if applicable
      const double filtered_phi = tracer_values[q];
      density[q] =
        cahn_hilliard_linear_mixing(filtered_phi, density0, density1);
      dynamic_viscosity[q] = cahn_hilliard_linear_mixing(filtered_phi,
                                                         dynamic_viscosity0,
                                                         dynamic_viscosity1);
      derivative_density_wrt_tracer[q] =
        cahn_hilliard_linear_mixing_derivative(filtered_phi,
                                               density0,
                                               density1);
      derivative_dynamic_viscosity_wrt_tracer[q] =
        cahn_hilliard_linear_mixing_derivative(filtered_phi,
                                               dynamic_viscosity0,
                                               dynamic_viscosity1);

      source_term_tracer[q]    = source_term_full_moving[q](phi_lower);
      source_term_potential[q] = source_term_full_moving[q](mu_lower);

      diffusive_flux[q] = diffusive_flux_factor *
                          present_velocity_gradients[q] *
                          potential_gradients[q];
      velocity_dot_tracer_gradient[q] =
        present_velocity_values[q] * tracer_gradients[q];

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        shape_phi[q][k]      = fe_values[tracer].value(k, q);
        grad_shape_phi[q][k] = fe_values[tracer].gradient(k, q);
        shape_mu[q][k]       = fe_values[potential].value(k, q);
        grad_shape_mu[q][k]  = fe_values[potential].gradient(k, q);
      }
    }
  }

public:
  template <typename VectorType>
  void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
              const VectorType                     &current_solution,
              const std::vector<VectorType>        &previous_solutions,
              const std::shared_ptr<Function<dim>> &source_terms,
              const std::shared_ptr<Function<dim>> &exact_solution)
  {
    fe_values.reinit(cell);
    if (enable_pseudo_solid)
      fe_values_fixed.reinit(cell);

    for (const unsigned int i : fe_values.dof_indices())
      components[i] = fe_values.get_fe().system_to_component_index(i).first;

    /**
     * Volume contributions
     */
    reinit_navier_stokes_cell(current_solution,
                              previous_solutions,
                              source_terms,
                              exact_solution);
    if (enable_pseudo_solid)
      reinit_pseudo_solid_cell(current_solution,
                               previous_solutions,
                               source_terms,
                               exact_solution);
    if (enable_cahn_hilliard)
      reinit_cahn_hilliard_cell(current_solution,
                                previous_solutions,
                                source_terms,
                                exact_solution);

    /**
     * Face contributions
     */
    if (cell->at_boundary())
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);
        if (face->at_boundary())
        {
          face_boundary_id[i_face] = face->boundary_id();

          fe_face_values.reinit(cell, face);
          if (enable_pseudo_solid)
            fe_face_values_fixed.reinit(cell, face);

          reinit_navier_stokes_face(i_face,
                                    current_solution,
                                    previous_solutions,
                                    source_terms,
                                    exact_solution);
          if (enable_pseudo_solid)
            reinit_pseudo_solid_face(i_face,
                                     current_solution,
                                     previous_solutions,
                                     source_terms,
                                     exact_solution);
          if (enable_lagrange_multiplier)
            reinit_lagrange_multiplier_face(i_face,
                                            current_solution,
                                            previous_solutions,
                                            source_terms,
                                            exact_solution);
        }
      }
  }

private:
  const ComponentOrdering ordering;

  unsigned int n_components;
  unsigned int u_lower;
  unsigned int p_lower;
  unsigned int x_lower;
  unsigned int l_lower;
  unsigned int phi_lower;
  unsigned int mu_lower;

  const bool enable_pseudo_solid;
  const bool enable_lagrange_multiplier;
  const bool enable_cahn_hilliard;

  Parameters::PhysicalProperties<dim> physical_properties;
  Parameters::CahnHilliard            cahn_hilliard_param;

  FEValues<dim> fe_values;
  FEValues<dim> fe_values_fixed;

  FEFaceValues<dim> fe_face_values;
  FEFaceValues<dim> fe_face_values_fixed;

public:
  const unsigned int n_q_points;
  const unsigned int n_faces;
  const unsigned int n_faces_q_points;
  const unsigned int dofs_per_cell;

  const std::vector<double> bdf_coefficients;

  std::vector<unsigned int>                components;
  std::vector<double>                      JxW_moving;
  std::vector<double>                      JxW_fixed;
  std::vector<unsigned int>                face_boundary_id;
  std::vector<std::vector<double>>         face_JxW_moving;
  std::vector<std::vector<double>>         face_JxW_fixed;
  std::vector<std::vector<Tensor<1, dim>>> face_normals_moving;

  /**
   * Navier-Stokes
   */
  FEValuesExtractors::Vector velocity;
  FEValuesExtractors::Scalar pressure;

  // Current and previous values and gradients for each quad node
  std::vector<Tensor<1, dim>>              present_velocity_values;
  std::vector<Tensor<2, dim>>              present_velocity_gradients;
  std::vector<Tensor<2, dim>>              present_velocity_sym_gradients;
  std::vector<double>                      present_velocity_divergence;
  std::vector<double>                      present_pressure_values;
  std::vector<std::vector<Tensor<1, dim>>> previous_velocity_values;

  // Current values on faces (each face, each quad node)
  std::vector<std::vector<Tensor<1, dim>>> present_face_velocity_values;

  // Shape functions in volume (each quad node and each dof)
  std::vector<std::vector<Tensor<1, dim>>> phi_u;
  std::vector<std::vector<Tensor<2, dim>>> grad_phi_u;
  std::vector<std::vector<Tensor<2, dim>>> sym_grad_phi_u;
  std::vector<std::vector<double>>         div_phi_u;
  std::vector<std::vector<double>>         phi_p;

  // Shape functions on faces (each face, quad node and dof)
  std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_u_face;

  // Source term in volume
  std::vector<Vector<double>> source_term_full_moving;
  std::vector<Tensor<1, dim>> source_term_velocity;
  std::vector<double>         source_term_pressure;

  // Exact solution
  std::vector<Vector<double>>              exact_solution_full;
  std::vector<std::vector<Tensor<1, dim>>> grad_exact_solution_full;
  std::vector<std::vector<Tensor<2, dim>>> exact_face_velocity_gradients;
  std::vector<std::vector<double>>         exact_face_pressure_values;

  /**
   * Pseudo-solid and ALE
   */
  FEValuesExtractors::Vector position;

  std::vector<double> lame_mu;
  std::vector<double> lame_lambda;

  std::vector<Tensor<1, dim>>              present_position_values;
  std::vector<Tensor<2, dim>>              present_position_gradients;
  std::vector<Tensor<1, dim>>              present_mesh_velocity_values;
  std::vector<std::vector<Tensor<1, dim>>> previous_position_values;

  // Current and previous values on faces
  std::vector<std::vector<Tensor<1, dim>>> present_face_position_values;
  std::vector<std::vector<Tensor<2, dim>>> present_face_position_gradient;
  std::vector<std::vector<Tensor<1, dim>>> present_face_mesh_velocity_values;
  std::vector<std::vector<std::vector<Tensor<1, dim>>>>
    previous_face_position_values;

  // Shape functions and gradients for each quad node and each dof
  std::vector<std::vector<Tensor<1, dim>>> phi_x;
  std::vector<std::vector<Tensor<2, dim>>> grad_phi_x;
  std::vector<std::vector<double>>         div_phi_x;

  // Shape functions on faces for relevant faces, each quad node and each dof
  std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_x_face;
  std::vector<std::vector<std::vector<Tensor<2, dim>>>> grad_phi_x_face;

  std::vector<Vector<double>> source_term_full_fixed;
  std::vector<Tensor<1, dim>> source_term_position;

  // Gradient of source term, at each quad node, for each dof component, result
  // is a Tensor<1, dim>. Only needed for the Jacobian matrix of NS with moving
  // mesh.
  std::vector<std::vector<Tensor<1, dim>>> grad_source_term_full;
  std::vector<Tensor<2, dim>>              grad_source_velocity;
  std::vector<Tensor<1, dim>>              grad_source_pressure;

  // The reference jacobians partial xsi_dim/partial xsi_(dim-1)
  std::array<Tensor<1, dim>, dim - 1> dxsids_array;

  // At face x quad node x phi_position_j
  std::vector<std::vector<std::vector<double>>> delta_dx;

  /**
   * Lagrange multiplier
   */
  FEValuesExtractors::Vector               lambda;
  std::vector<std::vector<Tensor<1, dim>>> present_face_lambda_values;
  std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_l_face;

  /**
   * Cahn-Hilliard
   */
  FEValuesExtractors::Scalar tracer;
  FEValuesExtractors::Scalar potential;

  double density0;
  double density1;
  double dynamic_viscosity0;
  double dynamic_viscosity1;
  double mobility;
  double epsilon;
  double sigma_tilde;
  double diffusive_flux_factor;

  std::vector<double> density;
  std::vector<double> derivative_density_wrt_tracer;
  std::vector<double> dynamic_viscosity;
  std::vector<double> derivative_dynamic_viscosity_wrt_tracer;

  std::vector<double>              tracer_values;
  std::vector<Tensor<1, dim>>      tracer_gradients;
  std::vector<double>              potential_values;
  std::vector<Tensor<1, dim>>      potential_gradients;
  std::vector<std::vector<double>> previous_tracer_values;

  std::vector<Tensor<1, dim>> diffusive_flux;
  std::vector<double>         velocity_dot_tracer_gradient;

  std::vector<std::vector<double>>         shape_phi;
  std::vector<std::vector<Tensor<1, dim>>> grad_shape_phi;
  std::vector<std::vector<double>>         shape_mu;
  std::vector<std::vector<Tensor<1, dim>>> grad_shape_mu;

  std::vector<double> source_term_tracer;
  std::vector<double> source_term_potential;
};

/**
 * Scratch data for the incompressible NS solver on fixed mesh.
 */
template <int dim>
class ScratchDataIncompressibleNS : public ScratchData<dim>
{
public:
  /**
   * Constructor
   */
  ScratchDataIncompressibleNS(const ComponentOrdering    &ordering,
                              const FESystem<dim>        &fe,
                              const Mapping<dim>         &mapping,
                              const Quadrature<dim>      &cell_quadrature,
                              const Quadrature<dim - 1>  &face_quadrature,
                              const std::vector<double>  &bdf_coefficients,
                              const ParameterReader<dim> &param)
    : ScratchData<dim>(ordering,
                       /*enable_pseudo_solid = */ false,
                       /*enable_lagrange_multiplier = */ false,
                       /*enable_cahn_hilliard = */ false,
                       fe,
                       mapping,
                       mapping,
                       cell_quadrature,
                       face_quadrature,
                       bdf_coefficients,
                       param)
  {}

  /**
   * Copy constructor
   */
  ScratchDataIncompressibleNS(const ScratchDataIncompressibleNS &other)
    : ScratchData<dim>(other)
  {}
};

/**
 * Scratch data for the FSI solver on moving mesh.
 */
template <int dim>
class ScratchDataFSI : public ScratchData<dim>
{
public:
  /**
   * Constructor
   */
  ScratchDataFSI(const ComponentOrdering    &ordering,
                 const FESystem<dim>        &fe,
                 const Mapping<dim>         &fixed_mapping,
                 const Mapping<dim>         &moving_mapping,
                 const Quadrature<dim>      &cell_quadrature,
                 const Quadrature<dim - 1>  &face_quadrature,
                 const std::vector<double>  &bdf_coefficients,
                 const ParameterReader<dim> &param)
    : ScratchData<dim>(ordering,
                       /*enable_pseudo_solid = */ true,
                       /*enable_lagrange_multiplier = */ true,
                       /*enable_cahn_hilliard = */ false,
                       fe,
                       fixed_mapping,
                       moving_mapping,
                       cell_quadrature,
                       face_quadrature,
                       bdf_coefficients,
                       param)
  {}

  /**
   * Copy constructor
   */
  ScratchDataFSI(const ScratchDataFSI &other)
    : ScratchData<dim>(other)
  {}
};

/**
 * Scratch data for the quasi-incompressible Cahn_hilliard Navier-Stokes solver
 * on fixed mesh.
 */
template <int dim>
class ScratchDataCHNS : public ScratchData<dim>
{
public:
  /**
   * Constructor
   */
  ScratchDataCHNS(const ComponentOrdering    &ordering,
                  const FESystem<dim>        &fe,
                  const Mapping<dim>         &mapping,
                  const Quadrature<dim>      &cell_quadrature,
                  const Quadrature<dim - 1>  &face_quadrature,
                  const std::vector<double>  &bdf_coefficients,
                  const ParameterReader<dim> &param)
    : ScratchData<dim>(ordering,
                       /*enable_pseudo_solid = */ false,
                       /*enable_lagrange_multiplier = */ false,
                       /*enable_cahn_hilliard = */ true,
                       fe,
                       mapping,
                       mapping,
                       cell_quadrature,
                       face_quadrature,
                       bdf_coefficients,
                       param)
  {}

  /**
   * Copy constructor
   */
  ScratchDataCHNS(const ScratchDataCHNS &other)
    : ScratchData<dim>(other)
  {}
};

#endif