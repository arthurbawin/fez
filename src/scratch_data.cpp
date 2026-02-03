
#include <scratch_data.h>

/**
 * Get the update flags for the FEValues, depending on the enabled features.
 */
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

/**
 * Get the update flags for the FEFaceValues.
 */
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

template <int dim, bool has_hp_capabilities>
ScratchData<dim, has_hp_capabilities>::ScratchData(
  const ComponentOrdering    &ordering,
  const bool                  enable_pseudo_solid,
  const bool                  enable_lagrange_multiplier,
  const bool                  enable_cahn_hilliard,
  const FESystem<dim>        &fe,
  const Mapping<dim>         &fixed_mapping,
  const Mapping<dim>         &moving_mapping,
  const Quadrature<dim>      &cell_quadrature,
  const Quadrature<dim - 1>  &face_quadrature,
  const std::vector<double>  &bdf_coefficients,
  const ParameterReader<dim> &param)
  : use_quads(param.finite_elements.use_quads)
  , ordering(ordering)
  , n_components(ordering.n_components)
  , enable_pseudo_solid(enable_pseudo_solid)
  , enable_lagrange_multiplier(enable_lagrange_multiplier)
  , enable_cahn_hilliard(enable_cahn_hilliard)
  , physical_properties(param.physical_properties)
  , cahn_hilliard_param(param.cahn_hilliard)
  , mesh_forcing_param(param.mesh_forcing)
  , fe_values(std::make_unique<FEValues<dim>>(
      moving_mapping,
      fe,
      cell_quadrature,
      get_cell_update_flags(enable_pseudo_solid,
                            enable_lagrange_multiplier,
                            enable_cahn_hilliard)))
  , fe_values_fixed(std::make_unique<FEValues<dim>>(
      fixed_mapping,
      fe,
      cell_quadrature,
      get_cell_update_flags(enable_pseudo_solid,
                            enable_lagrange_multiplier,
                            enable_cahn_hilliard)))
  , fe_face_values(std::make_unique<FEFaceValues<dim>>(
      moving_mapping,
      fe,
      face_quadrature,
      get_face_update_flags(enable_pseudo_solid,
                            enable_lagrange_multiplier,
                            enable_cahn_hilliard)))
  , fe_face_values_fixed(std::make_unique<FEFaceValues<dim>>(
      fixed_mapping,
      fe,
      face_quadrature,
      get_face_update_flags(enable_pseudo_solid,
                            enable_lagrange_multiplier,
                            enable_cahn_hilliard)))
  , n_q_points(cell_quadrature.size())
  , n_faces(fe.reference_cell().n_faces())
  , n_faces_q_points(face_quadrature.size())
  , dofs_per_cell(fe.dofs_per_cell)
  , bdf_coefficients(bdf_coefficients)
{
  if constexpr (has_hp_capabilities)
    AssertThrow(
      false,
      ExcMessage(
        "Trying to use ScratchData constructor without hp capabilities, "
        "but this object was created with hp support."));

  initialize_navier_stokes();

  if (enable_pseudo_solid)
    initialize_pseudo_solid();

  if (enable_lagrange_multiplier)
    initialize_lagrange_multiplier();

  if (enable_cahn_hilliard)
    initialize_cahn_hilliard();

  allocate();
}

template <int dim, bool has_hp_capabilities>
ScratchData<dim, has_hp_capabilities>::ScratchData(
  const ComponentOrdering          &ordering,
  const bool                        enable_pseudo_solid,
  const bool                        enable_lagrange_multiplier,
  const bool                        enable_cahn_hilliard,
  const hp::FECollection<dim>      &fe_collection,
  const hp::MappingCollection<dim> &fixed_mapping_collection,
  const hp::MappingCollection<dim> &moving_mapping_collection,
  const hp::QCollection<dim>       &cell_quadrature_collection,
  const hp::QCollection<dim - 1>   &face_quadrature_collection,
  const std::vector<double>        &bdf_coefficients,
  const ParameterReader<dim>       &param)
  : use_quads(param.finite_elements.use_quads)
  , ordering(ordering)
  , n_components(ordering.n_components)
  , enable_pseudo_solid(enable_pseudo_solid)
  , enable_lagrange_multiplier(enable_lagrange_multiplier)
  , enable_cahn_hilliard(enable_cahn_hilliard)
  , physical_properties(param.physical_properties)
  , cahn_hilliard_param(param.cahn_hilliard)
  , mesh_forcing_param(param.mesh_forcing)
  , hp_fe_values(std::make_unique<hp::FEValues<dim>>(
      moving_mapping_collection,
      fe_collection,
      cell_quadrature_collection,
      get_cell_update_flags(enable_pseudo_solid,
                            enable_lagrange_multiplier,
                            enable_cahn_hilliard)))
  , hp_fe_values_fixed(std::make_unique<hp::FEValues<dim>>(
      fixed_mapping_collection,
      fe_collection,
      cell_quadrature_collection,
      get_cell_update_flags(enable_pseudo_solid,
                            enable_lagrange_multiplier,
                            enable_cahn_hilliard)))
  , hp_fe_face_values(std::make_unique<hp::FEFaceValues<dim>>(
      moving_mapping_collection,
      fe_collection,
      face_quadrature_collection,
      get_face_update_flags(enable_pseudo_solid,
                            enable_lagrange_multiplier,
                            enable_cahn_hilliard)))
  , hp_fe_face_values_fixed(std::make_unique<hp::FEFaceValues<dim>>(
      fixed_mapping_collection,
      fe_collection,
      face_quadrature_collection,
      get_face_update_flags(enable_pseudo_solid,
                            enable_lagrange_multiplier,
                            enable_cahn_hilliard)))
  , bdf_coefficients(bdf_coefficients)
{
  if constexpr (!has_hp_capabilities)
    AssertThrow(false,
                ExcMessage(
                  "Trying to use ScratchData constructor with hp capabilities, "
                  "but this object was not created with hp support."));

  /**
   * Set the number of faces and quadrature points.
   * This ScratchData is for now limited to applications with a Lagrange
   * multiplier in mind, where the mapping and quadratures are the same on all
   * cells.
   */
  n_faces          = fe_collection[0].reference_cell().n_faces();
  dofs_per_cell    = fe_collection.max_dofs_per_cell();
  n_q_points       = cell_quadrature_collection[0].size();
  n_faces_q_points = face_quadrature_collection[0].size();
  for (const auto &fe : fe_collection)
  {
    AssertThrow(n_faces == fe.reference_cell().n_faces(),
                ExcMessage(
                  "When initializing the hp ScratchData, data  "
                  "differs among the FESystems of the given FECollection."
                  "The ScratchData with hp capabilities does not yet support "
                  "FESystems associated with different reference cells or "
                  "different dofs per cell."));
  }
  for (const auto &q : cell_quadrature_collection)
  {
    AssertThrow(n_q_points == q.size(),
                ExcMessage("Data mismatch among cell quadratures"));
  }
  for (const auto &q : face_quadrature_collection)
  {
    AssertThrow(n_faces_q_points == q.size(),
                ExcMessage("Data mismatch among face quadratures"));
  }

  initialize_navier_stokes();

  if (enable_pseudo_solid)
    initialize_pseudo_solid();

  if (enable_lagrange_multiplier)
    initialize_lagrange_multiplier();

  if (enable_cahn_hilliard)
    initialize_cahn_hilliard();

  allocate();
}

template <int dim, bool has_hp_capabilities>
ScratchData<dim, has_hp_capabilities>::ScratchData(const ScratchData &other)
  : use_quads(other.use_quads)
  , ordering(other.ordering)
  , n_components(other.n_components)
  , enable_pseudo_solid(other.enable_pseudo_solid)
  , enable_lagrange_multiplier(other.enable_lagrange_multiplier)
  , enable_cahn_hilliard(other.enable_cahn_hilliard)
  , physical_properties(other.physical_properties)
  , cahn_hilliard_param(other.cahn_hilliard_param)
  , mesh_forcing_param(other.mesh_forcing_param)
  , n_q_points(other.n_q_points)
  , n_faces(other.n_faces)
  , n_faces_q_points(other.n_faces_q_points)
  , dofs_per_cell(other.dofs_per_cell)
  , bdf_coefficients(other.bdf_coefficients)
{
  if constexpr (has_hp_capabilities)
  {
    hp_fe_values = std::make_unique<hp::FEValues<dim>>(
      other.hp_fe_values->get_mapping_collection(),
      other.hp_fe_values->get_fe_collection(),
      other.hp_fe_values->get_quadrature_collection(),
      other.hp_fe_values->get_update_flags());
    hp_fe_values_fixed = std::make_unique<hp::FEValues<dim>>(
      other.hp_fe_values_fixed->get_mapping_collection(),
      other.hp_fe_values_fixed->get_fe_collection(),
      other.hp_fe_values_fixed->get_quadrature_collection(),
      other.hp_fe_values_fixed->get_update_flags());
    hp_fe_face_values = std::make_unique<hp::FEFaceValues<dim>>(
      other.hp_fe_face_values->get_mapping_collection(),
      other.hp_fe_face_values->get_fe_collection(),
      other.hp_fe_face_values->get_quadrature_collection(),
      other.hp_fe_face_values->get_update_flags());
    hp_fe_face_values_fixed = std::make_unique<hp::FEFaceValues<dim>>(
      other.hp_fe_face_values_fixed->get_mapping_collection(),
      other.hp_fe_face_values_fixed->get_fe_collection(),
      other.hp_fe_face_values_fixed->get_quadrature_collection(),
      other.hp_fe_face_values_fixed->get_update_flags());
  }
  else
  {
    fe_values =
      std::make_unique<FEValues<dim>>(other.fe_values->get_mapping(),
                                      other.fe_values->get_fe(),
                                      other.fe_values->get_quadrature(),
                                      other.fe_values->get_update_flags());
    fe_values_fixed = std::make_unique<FEValues<dim>>(
      other.fe_values_fixed->get_mapping(),
      other.fe_values_fixed->get_fe(),
      other.fe_values_fixed->get_quadrature(),
      other.fe_values_fixed->get_update_flags());
    fe_face_values = std::make_unique<FEFaceValues<dim>>(
      other.fe_face_values->get_mapping(),
      other.fe_face_values->get_fe(),
      other.fe_face_values->get_quadrature(),
      other.fe_face_values->get_update_flags());
    fe_face_values_fixed = std::make_unique<FEFaceValues<dim>>(
      other.fe_face_values_fixed->get_mapping(),
      other.fe_face_values_fixed->get_fe(),
      other.fe_face_values_fixed->get_quadrature(),
      other.fe_face_values_fixed->get_update_flags());
  }

  initialize_navier_stokes();

  if (enable_pseudo_solid)
    initialize_pseudo_solid();

  if (enable_lagrange_multiplier)
    initialize_lagrange_multiplier();

  if (enable_cahn_hilliard)
    initialize_cahn_hilliard();

  allocate();
}

template <int dim, bool has_hp_capabilities>
const FEValues<dim> *ScratchData<dim, has_hp_capabilities>::reinit(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  const bool                                            fixed_mapping)
{
  if constexpr (!has_hp_capabilities)
  {
    auto &active_fe_values =
      fixed_mapping ? this->fe_values_fixed : this->fe_values;
    active_fe_values->reinit(cell);
    return active_fe_values.get();
  }
  else
  {
    auto &active_hp_fe_values =
      fixed_mapping ? this->hp_fe_values_fixed : this->hp_fe_values;
    active_hp_fe_values->reinit(cell);
    const auto &fe_values = active_hp_fe_values->get_present_fe_values();
    AssertDimension(hp_fe_values->get_fe_collection()[cell->active_fe_index()]
                      .n_dofs_per_cell(),
                    fe_values.dofs_per_cell);
    return &fe_values;
  }
}

template <int dim, bool has_hp_capabilities>
const FEFaceValues<dim> *ScratchData<dim, has_hp_capabilities>::reinit(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  const unsigned int                                    face_no,
  const bool                                            fixed_mapping)
{
  if constexpr (!has_hp_capabilities)
  {
    auto &active_fe_face_values =
      fixed_mapping ? this->fe_face_values_fixed : this->fe_face_values;
    active_fe_face_values->reinit(cell, face_no);
    return active_fe_face_values.get();
  }
  else
  {
    auto &active_hp_fe_face_values =
      fixed_mapping ? this->hp_fe_face_values_fixed : this->hp_fe_face_values;
    active_hp_fe_face_values->reinit(cell, face_no);
    const auto &fe_face_values =
      active_hp_fe_face_values->get_present_fe_values();
    return &fe_face_values;
  }
}

template <int dim, bool has_hp_capabilities>
void ScratchData<dim, has_hp_capabilities>::initialize_navier_stokes()
{
  velocity.first_vector_component = u_lower = ordering.u_lower;
  pressure.component = p_lower = ordering.p_lower;
}

template <int dim, bool has_hp_capabilities>
void ScratchData<dim, has_hp_capabilities>::initialize_pseudo_solid()
{
  // Check if the given ordering allows to compute the requested features
  // FIXME: This should be checked at compile-time, but I'm not sure what's
  // the best way...
  AssertThrow(ordering.x_lower != numbers::invalid_unsigned_int,
              ExcMessage(
                "Cannot create ScratchData with pseudo solid data because "
                "solver does not have a mesh position variable."));

  position.first_vector_component = x_lower = ordering.x_lower;
}

template <int dim, bool has_hp_capabilities>
void ScratchData<dim, has_hp_capabilities>::initialize_lagrange_multiplier()
{
  AssertThrow(
    ordering.l_lower != numbers::invalid_unsigned_int,
    ExcMessage(
      "Cannot create ScratchData with Lagrange multiplier data because "
      "solver does not have a Lagrange multiplier variable."));

  lambda.first_vector_component = ordering.l_lower;
}

template <int dim, bool has_hp_capabilities>
void ScratchData<dim, has_hp_capabilities>::initialize_cahn_hilliard()
{
  AssertThrow(
    ordering.phi_lower != numbers::invalid_unsigned_int &&
      ordering.mu_lower != numbers::invalid_unsigned_int,
    ExcMessage(
      "Cannot create ScratchData with Cahn Hilliard data because solver does "
      "not have a tracer and/or potential variable(s)."));

  tracer.component = phi_lower = ordering.phi_lower;
  potential.component = mu_lower = ordering.mu_lower;

  density0           = physical_properties.fluids[0].density;
  density1           = physical_properties.fluids[1].density;
  const double nu0   = physical_properties.fluids[0].kinematic_viscosity;
  const double nu1   = physical_properties.fluids[1].kinematic_viscosity;
  dynamic_viscosity0 = density0 * nu0;
  dynamic_viscosity1 = density1 * nu1;
  mobility           = cahn_hilliard_param.mobility;
  epsilon            = cahn_hilliard_param.epsilon_interface;
  sigma_tilde = 3. / (2. * sqrt(2.)) * cahn_hilliard_param.surface_tension;
  diffusive_flux_factor = mobility * 0.5 * (density1 - density0);
  body_force            = cahn_hilliard_param.body_force;
}

template <int dim, bool has_hp_capabilities>
void ScratchData<dim, has_hp_capabilities>::allocate()
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
  present_velocity_sym_gradients.resize(n_q_points);
  present_velocity_divergence.resize(n_q_points);
  present_pressure_values.resize(n_q_points);
  previous_velocity_values.resize(bdf_coefficients.size() - 1,
                                  std::vector<Tensor<1, dim>>(n_q_points));

  present_face_velocity_values.resize(
    n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

  phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
  grad_phi_u.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
  sym_grad_phi_u.resize(n_q_points,
                        std::vector<SymmetricTensor<2, dim>>(dofs_per_cell));
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
    grad_phi_x_moving.resize(n_q_points,
                             std::vector<Tensor<2, dim>>(dofs_per_cell));
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
    if (enable_cahn_hilliard && mesh_forcing_param.enable)
      f_mesh_values.resize(n_q_points);
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

  if (enable_cahn_hilliard)
  {
    density.resize(n_q_points);
    derivative_density_wrt_tracer.resize(n_q_points);
    dynamic_viscosity.resize(n_q_points);
    derivative_dynamic_viscosity_wrt_tracer.resize(n_q_points);

    tracer_values.resize(n_q_points);
    tracer_gradients.resize(n_q_points);
    potential_values.resize(n_q_points);
    potential_gradients.resize(n_q_points);
    previous_tracer_values.resize(bdf_coefficients.size() - 1,
                                  std::vector<double>(n_q_points));
    previous_tracer_gradients.resize(bdf_coefficients.size() - 1,
                                 std::vector<Tensor<1, dim>>(n_q_points));


    diffusive_flux.resize(n_q_points);
    velocity_dot_tracer_gradient.resize(n_q_points);
    u_conv_dot_tracer_gradient.resize(n_q_points);
    shape_phi.resize(n_q_points, std::vector<double>(dofs_per_cell));
    grad_shape_phi.resize(n_q_points,
                          std::vector<Tensor<1, dim>>(dofs_per_cell));
    shape_mu.resize(n_q_points, std::vector<double>(dofs_per_cell));
    grad_shape_mu.resize(n_q_points,
                         std::vector<Tensor<1, dim>>(dofs_per_cell));

    source_term_tracer.resize(n_q_points);
    source_term_potential.resize(n_q_points);
  }
}

// Explicit instantiations
template class ScratchData<2, false>;
template class ScratchData<2, true>;
template class ScratchData<3, false>;
template class ScratchData<3, true>;