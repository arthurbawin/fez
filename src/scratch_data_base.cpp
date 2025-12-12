
#include <scratch_data_base.h>

static UpdateFlags get_cell_update_flags(const bool enable_pseudo_solid,
                                         const bool enable_lagrange_multiplier,
                                         const bool enable_cahn_hilliard)
{
  // Flags for Navier-Stokes on fixed mesh only
  UpdateFlags flags = update_values | update_gradients |
                      update_quadrature_points | update_JxW_values;

  if (enable_pseudo_solid)
  {
    // Also update full Jacobian matrix
    flags |= update_jacobians;
  }
  if (enable_lagrange_multiplier)
  {
    // No additional flag
  }
  if (enable_cahn_hilliard)
  {
    // No additional flag
  }
  return flags;
}

static UpdateFlags get_face_update_flags(const bool enable_pseudo_solid,
                                         const bool enable_lagrange_multiplier,
                                         const bool enable_cahn_hilliard)
{
  // Flags for Navier-Stokes on fixed mesh only
  UpdateFlags flags = update_values | update_gradients |
                      update_quadrature_points | update_JxW_values |
                      update_normal_vectors;

  if (enable_pseudo_solid)
  {
    // Also update full Jacobian matrix
    flags |= update_jacobians;
  }
  if (enable_lagrange_multiplier)
  {
    // No additional flag
  }
  if (enable_cahn_hilliard)
  {
    // No additional flag
  }
  return flags;
}

template <int dim>
ScratchData<dim>::ScratchData(const ComponentOrdering &ordering,
                              const bool               enable_pseudo_solid,
                              const bool             enable_lagrange_multiplier,
                              const bool             enable_cahn_hilliard,
                              const FESystem<dim>   &fe,
                              const Mapping<dim>    &fixed_mapping,
                              const Mapping<dim>    &moving_mapping,
                              const Quadrature<dim> &cell_quadrature,
                              const Quadrature<dim - 1>  &face_quadrature,
                              const std::vector<double>  &bdf_coefficients,
                              const ParameterReader<dim> &param)
  : ordering(ordering)
  , n_components(ordering.n_components)
  , enable_pseudo_solid(enable_pseudo_solid)
  , enable_lagrange_multiplier(enable_lagrange_multiplier)
  , enable_cahn_hilliard(enable_cahn_hilliard)
  , physical_properties(param.physical_properties)
  , fe_values(moving_mapping,
              fe,
              cell_quadrature,
              get_cell_update_flags(enable_pseudo_solid,
                                    enable_lagrange_multiplier,
                                    enable_cahn_hilliard))
  , fe_values_fixed(fixed_mapping,
                    fe,
                    cell_quadrature,
                    get_cell_update_flags(enable_pseudo_solid,
                                          enable_lagrange_multiplier,
                                          enable_cahn_hilliard))
  , fe_face_values(moving_mapping,
                   fe,
                   face_quadrature,
                   get_face_update_flags(enable_pseudo_solid,
                                         enable_lagrange_multiplier,
                                         enable_cahn_hilliard))
  , fe_face_values_fixed(fixed_mapping,
                         fe,
                         face_quadrature,
                         get_face_update_flags(enable_pseudo_solid,
                                               enable_lagrange_multiplier,
                                               enable_cahn_hilliard))
  , n_q_points(cell_quadrature.size())
  , n_faces(fe.reference_cell().n_faces())
  , n_faces_q_points(face_quadrature.size())
  , dofs_per_cell(fe.dofs_per_cell)
  , bdf_coefficients(bdf_coefficients)
{
  velocity.first_vector_component = u_lower = ordering.u_lower;
  pressure.component = p_lower = ordering.p_lower;

  // Check if weak forms are to be assembled on boundaries
  for (const auto &[id, bc] : param.fluid_bc)
    if (bc.type == BoundaryConditions::Type::open_mms)
    {
      has_navier_stokes_boundary_forms = true;
      break;
    }

  if (enable_pseudo_solid)
  {
    // Check if the given ordering allows to compute the requested features
    // FIXME: This should be checked at compile-time, but I'm not sure what's
    // the best way...
    AssertThrow(ordering.x_lower != numbers::invalid_unsigned_int,
                ExcMessage(
                  "Cannot create ScratchData with pseudo solid data because "
                  "solver does not have a mesh position variable."));

    position.first_vector_component = x_lower = ordering.x_lower;

    // No boundary integrals for now
    has_pseudo_solid_boundary_forms = false;
  }

  if (enable_lagrange_multiplier)
  {
    AssertThrow(
      ordering.l_lower != numbers::invalid_unsigned_int,
      ExcMessage(
        "Cannot create ScratchData with Lagrange multiplier data because "
        "solver does not have a Lagrange multiplier variable."));

    lambda.first_vector_component = ordering.l_lower;
  }

  if (enable_cahn_hilliard)
  {
    AssertThrow(
      ordering.phi_lower != numbers::invalid_unsigned_int &&
        ordering.mu_lower != numbers::invalid_unsigned_int,
      ExcMessage(
        "Cannot create ScratchData with Cahn Hilliard data because solver does "
        "not have a tracer and/or potential variable(s)."));

    tracer.component = phi_lower = ordering.phi_lower;
    potential.component = mu_lower = ordering.mu_lower;

    // No boundary integrals for now
    has_cahn_hilliard_boundary_forms = false;
  }

  has_boundary_forms =
    enable_lagrange_multiplier || has_navier_stokes_boundary_forms ||
    has_pseudo_solid_boundary_forms || has_cahn_hilliard_boundary_forms;

  allocate();
}

template <int dim>
ScratchData<dim>::ScratchData(const ScratchData &other)
  : ordering(other.ordering)
  , n_components(other.n_components)
  , enable_pseudo_solid(other.enable_pseudo_solid)
  , enable_lagrange_multiplier(other.enable_lagrange_multiplier)
  , enable_cahn_hilliard(other.enable_cahn_hilliard)
  , physical_properties(other.physical_properties)
  , fe_values(other.fe_values.get_mapping(),
              other.fe_values.get_fe(),
              other.fe_values.get_quadrature(),
              other.fe_values.get_update_flags())
  , fe_values_fixed(other.fe_values_fixed.get_mapping(),
                    other.fe_values_fixed.get_fe(),
                    other.fe_values_fixed.get_quadrature(),
                    other.fe_values_fixed.get_update_flags())
  , fe_face_values(other.fe_face_values.get_mapping(),
                   other.fe_face_values.get_fe(),
                   other.fe_face_values.get_quadrature(),
                   other.fe_face_values.get_update_flags())
  , fe_face_values_fixed(other.fe_face_values_fixed.get_mapping(),
                         other.fe_face_values_fixed.get_fe(),
                         other.fe_face_values_fixed.get_quadrature(),
                         other.fe_face_values_fixed.get_update_flags())
  , has_boundary_forms(other.has_boundary_forms)
  , has_navier_stokes_boundary_forms(other.has_navier_stokes_boundary_forms)
  , has_pseudo_solid_boundary_forms(other.has_pseudo_solid_boundary_forms)
  , has_cahn_hilliard_boundary_forms(other.has_cahn_hilliard_boundary_forms)
  , n_q_points(other.n_q_points)
  , n_faces(other.n_faces)
  , n_faces_q_points(other.n_faces_q_points)
  , dofs_per_cell(other.dofs_per_cell)
  , bdf_coefficients(other.bdf_coefficients)
{
  velocity.first_vector_component = u_lower = ordering.u_lower;
  pressure.component = p_lower = ordering.p_lower;
  if (enable_pseudo_solid)
    position.first_vector_component = x_lower = ordering.x_lower;
  if (enable_lagrange_multiplier)
    lambda.first_vector_component = l_lower = ordering.l_lower;
  if (enable_cahn_hilliard)
  {
    tracer.component = phi_lower = ordering.phi_lower;
    potential.component = mu_lower = ordering.mu_lower;
  }

  allocate();
}

template <int dim>
void ScratchData<dim>::allocate()
{
  components.resize(dofs_per_cell);
  JxW_moving.resize(n_q_points);
  JxW_fixed.resize(n_q_points);
  face_boundary_id.resize(n_faces);
  face_JxW_moving.resize(n_faces, std::vector<double>(n_faces_q_points));
  face_JxW_fixed.resize(n_faces, std::vector<double>(n_faces_q_points));
  face_normals_moving.resize(n_faces,
                             std::vector<Tensor<1, dim>>(n_faces_q_points));

  /**
   * Navier-Stokes
   */
  present_velocity_values.resize(n_q_points);
  present_velocity_gradients.resize(n_q_points);
  present_pressure_values.resize(n_q_points);
  previous_velocity_values.resize(bdf_coefficients.size() - 1,
                                  std::vector<Tensor<1, dim>>(n_q_points));

  present_face_velocity_values.resize(
    n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

  phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
  grad_phi_u.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
  div_phi_u.resize(n_q_points, std::vector<double>(dofs_per_cell));
  phi_p.resize(n_q_points, std::vector<double>(dofs_per_cell));

  phi_u_face.resize(n_faces,
                    std::vector<std::vector<Tensor<1, dim>>>(
                      n_faces_q_points,
                      std::vector<Tensor<1, dim>>(dofs_per_cell)));

  source_term_full_moving.resize(n_q_points, Vector<double>(n_components));
  source_term_velocity.resize(n_q_points);
  source_term_pressure.resize(n_q_points);

  exact_solution_full.resize(n_faces_q_points, Vector<double>(n_components));
  grad_exact_solution_full.resize(n_faces_q_points,
                                  std::vector<Tensor<1, dim>>(n_components));
  exact_face_velocity_gradients.resize(
    n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
  exact_face_pressure_values.resize(n_faces,
                                    std::vector<double>(n_faces_q_points));

  if (enable_pseudo_solid)
  {
    lame_mu.resize(n_q_points);
    lame_lambda.resize(n_q_points);

    present_position_values.resize(n_q_points);
    present_position_gradients.resize(n_q_points);
    present_mesh_velocity_values.resize(n_q_points);
    previous_position_values.resize(bdf_coefficients.size() - 1,
                                    std::vector<Tensor<1, dim>>(n_q_points));

    present_face_position_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    present_face_position_gradient.resize(
      n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
    present_face_mesh_velocity_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    previous_face_position_values.resize(
      n_faces,
      std::vector<std::vector<Tensor<1, dim>>>(bdf_coefficients.size() - 1,
                                               std::vector<Tensor<1, dim>>(
                                                 n_faces_q_points)));

    phi_x.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_x.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_x.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_x_face.resize(n_faces,
                      std::vector<std::vector<Tensor<1, dim>>>(
                        n_faces_q_points,
                        std::vector<Tensor<1, dim>>(dofs_per_cell)));
    grad_phi_x_face.resize(n_faces,
                           std::vector<std::vector<Tensor<2, dim>>>(
                             n_faces_q_points,
                             std::vector<Tensor<2, dim>>(dofs_per_cell)));

    source_term_full_fixed.resize(n_q_points, Vector<double>(n_components));
    source_term_position.resize(n_q_points);

    grad_source_term_full.resize(n_q_points,
                                 std::vector<Tensor<1, dim>>(n_components));
    grad_source_velocity.resize(n_q_points);
    grad_source_pressure.resize(n_q_points);

    delta_dx.resize(n_faces,
                    std::vector<std::vector<double>>(
                      n_faces_q_points, std::vector<double>(dofs_per_cell)));
  }

  if (enable_lagrange_multiplier)
  {
    present_face_lambda_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    phi_l_face.resize(n_faces,
                      std::vector<std::vector<Tensor<1, dim>>>(
                        n_faces_q_points,
                        std::vector<Tensor<1, dim>>(dofs_per_cell)));
  }

  if (enable_cahn_hilliard) {}
}

// Explicit instantiations
template class ScratchData<2>;
template class ScratchData<3>;