
#include <deal.II/fe/fe_update_flags.h>
#include <scratch_data.h>

namespace NavierStokesScratch
{
  /**
   * Get the update flags for the FEValues, depending on the enabled features.
   */
  template <unsigned int update_flags>
  UpdateFlags get_cell_update_flags(const bool enable_stabilization)
  {
    // Flags for Navier-Stokes on fixed mesh only
    UpdateFlags flags = update_values | update_gradients |
                        update_quadrature_points | update_JxW_values;

    // Pseudo-solid: also update full Jacobian matrix
    if constexpr ((update_flags & pseudo_solid) != 0)
      flags |= update_jacobians;

    // SUPG/PSPG stabilization: also update hessians to compute strong residuals
    if (enable_stabilization)
      flags |= update_hessians; // | update_inverse_jacobians;

    return flags;
  }

  /**
   * Get the update flags for the FEFaceValues.
   */
  template <unsigned int update_flags>
  UpdateFlags get_face_update_flags(const bool enable_stabilization)
  {
    // Flags for Navier-Stokes on fixed mesh only
    UpdateFlags flags = update_values | update_gradients |
                        update_quadrature_points | update_JxW_values |
                        update_normal_vectors;

    // Pseudo-solid
    if constexpr ((update_flags & pseudo_solid) != 0)
      flags |= update_jacobians;

    // SUPG/PSPG stabilization
    if (enable_stabilization)
      flags |= update_hessians;

    return flags;
  }

  template <int dim, unsigned int update_flags>
  ScratchData<dim, update_flags>::ScratchData(
    const ComponentOrdering    &ordering,
    const FESystem<dim>        &fe,
    const Mapping<dim>         &fixed_mapping,
    const Mapping<dim>         &moving_mapping,
    const Quadrature<dim>      &cell_quadrature,
    const Quadrature<dim - 1>  &face_quadrature,
    const TimeHandler          &time_handler,
    const ParameterReader<dim> &param,
    const bool                  enable_stabilization)
    : param(param)
    , use_quads(param.finite_elements.use_quads)
    , ordering(ordering)
    , n_components(ordering.n_components)
    , enable_stabilization(enable_stabilization)
    , enable_tracer_stabilization(param.stabilization.enable_tracer_supg)
    , physical_properties(param.physical_properties)
    , cahn_hilliard_param(param.cahn_hilliard)
    , fe_values(std::make_unique<FEValues<dim>>(
        moving_mapping,
        fe,
        cell_quadrature,
        get_cell_update_flags<update_flags>(enable_stabilization ||
                                            enable_tracer_stabilization)))
    , fe_values_fixed(std::make_unique<FEValues<dim>>(
        fixed_mapping,
        fe,
        cell_quadrature,
        get_cell_update_flags<update_flags>(enable_stabilization ||
                                            enable_tracer_stabilization)))
    , fe_face_values(std::make_unique<FEFaceValues<dim>>(
        moving_mapping,
        fe,
        face_quadrature,
        get_face_update_flags<update_flags>(enable_stabilization ||
                                            enable_tracer_stabilization)))
    , fe_face_values_fixed(std::make_unique<FEFaceValues<dim>>(
        fixed_mapping,
        fe,
        face_quadrature,
        get_face_update_flags<update_flags>(enable_stabilization ||
                                            enable_tracer_stabilization)))
    , n_q_points(cell_quadrature.size())
    , n_faces(fe.reference_cell().n_faces())
    , n_faces_q_points(face_quadrature.size())
    , dofs_per_cell(fe.dofs_per_cell)
    , max_dofs_per_cell(fe.dofs_per_cell)
    , time_handler(time_handler)
  {
    if constexpr (has_hp_capabilities)
      AssertThrow(
        false,
        ExcMessage(
          "Trying to use ScratchData constructor without hp capabilities, "
          "but this object was created with hp support."));

    initialize_navier_stokes();

    if constexpr (enable_pseudo_solid)
      initialize_pseudo_solid();

    if constexpr (enable_lagrange_multiplier)
      initialize_lagrange_multiplier();

    if constexpr (enable_cahn_hilliard)
      initialize_cahn_hilliard();

    if constexpr (enable_compressible)
      initialize_compressible();

    allocate();
  }

  template <int dim, unsigned int update_flags>
  ScratchData<dim, update_flags>::ScratchData(
    const ComponentOrdering          &ordering,
    const hp::FECollection<dim>      &fe_collection,
    const hp::MappingCollection<dim> &fixed_mapping_collection,
    const hp::MappingCollection<dim> &moving_mapping_collection,
    const hp::QCollection<dim>       &cell_quadrature_collection,
    const hp::QCollection<dim - 1>   &face_quadrature_collection,
    const TimeHandler                &time_handler,
    const ParameterReader<dim>       &param,
    const bool                        enable_stabilization)
    : param(param)
    , use_quads(param.finite_elements.use_quads)
    , ordering(ordering)
    , n_components(ordering.n_components)
    , enable_stabilization(enable_stabilization)
    , enable_tracer_stabilization(param.stabilization.enable_tracer_supg)
    , physical_properties(param.physical_properties)
    , cahn_hilliard_param(param.cahn_hilliard)
    , hp_fe_values(std::make_unique<hp::FEValues<dim>>(
        moving_mapping_collection,
        fe_collection,
        cell_quadrature_collection,
        get_cell_update_flags<update_flags>(enable_stabilization ||
                                            enable_tracer_stabilization)))
    , hp_fe_values_fixed(std::make_unique<hp::FEValues<dim>>(
        fixed_mapping_collection,
        fe_collection,
        cell_quadrature_collection,
        get_cell_update_flags<update_flags>(enable_stabilization ||
                                            enable_tracer_stabilization)))
    , hp_fe_face_values(std::make_unique<hp::FEFaceValues<dim>>(
        moving_mapping_collection,
        fe_collection,
        face_quadrature_collection,
        get_face_update_flags<update_flags>(enable_stabilization ||
                                            enable_tracer_stabilization)))
    , hp_fe_face_values_fixed(std::make_unique<hp::FEFaceValues<dim>>(
        fixed_mapping_collection,
        fe_collection,
        face_quadrature_collection,
        get_face_update_flags<update_flags>(enable_stabilization ||
                                            enable_tracer_stabilization)))
    , n_q_points(cell_quadrature_collection[0].size())
    , n_faces(fe_collection[0].reference_cell().n_faces())
    , n_faces_q_points(face_quadrature_collection[0].size())
    , dofs_per_cell(fe_collection.max_dofs_per_cell())
    , max_dofs_per_cell(fe_collection.max_dofs_per_cell())
    , time_handler(time_handler)
  {
    if constexpr (!has_hp_capabilities)
      AssertThrow(
        false,
        ExcMessage(
          "Trying to use ScratchData constructor with hp capabilities, "
          "but this object was not created with hp support."));

    /**
     * This ScratchData is for now limited to applications with a Lagrange
     * multiplier in mind, where the mapping and quadratures are the same on all
     * cells.
     */
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
                  ExcMessage("This ScratchData expects the same number of "
                             "quadrature nodes in all partitions."));
    }
    for (const auto &q : face_quadrature_collection)
    {
      AssertThrow(n_faces_q_points == q.size(),
                  ExcMessage("This ScratchData expects the same number of face "
                             "quadrature nodes in all partitions."));
    }

    initialize_navier_stokes();

    if constexpr (enable_pseudo_solid)
      initialize_pseudo_solid();

    if constexpr (enable_lagrange_multiplier)
      initialize_lagrange_multiplier();

    if constexpr (enable_cahn_hilliard)
      initialize_cahn_hilliard();

    if constexpr (enable_compressible)
      initialize_compressible();

    allocate();
  }

  template <int dim, unsigned int update_flags>
  ScratchData<dim, update_flags>::ScratchData(const ScratchData &other)
    : param(other.param)
    , use_quads(other.use_quads)
    , ordering(other.ordering)
    , n_components(other.n_components)
    , enable_stabilization(other.enable_stabilization)
    , enable_tracer_stabilization(other.enable_tracer_stabilization)
    , physical_properties(other.physical_properties)
    , cahn_hilliard_param(other.cahn_hilliard_param)
    , n_q_points(other.n_q_points)
    , n_faces(other.n_faces)
    , n_faces_q_points(other.n_faces_q_points)
    , dofs_per_cell(other.dofs_per_cell)
    , max_dofs_per_cell(other.max_dofs_per_cell)
    , time_handler(other.time_handler)
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

    if constexpr (enable_pseudo_solid)
      initialize_pseudo_solid();

    if constexpr (enable_lagrange_multiplier)
      initialize_lagrange_multiplier();

    if constexpr (enable_cahn_hilliard)
      initialize_cahn_hilliard();

    if constexpr (enable_compressible)
      initialize_compressible();

    allocate();
  }

  template <int dim, unsigned int update_flags>
  const FEValues<dim> *ScratchData<dim, update_flags>::reinit(
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

  template <int dim, unsigned int update_flags>
  const FEFaceValues<dim> *ScratchData<dim, update_flags>::reinit(
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

  template <int dim, unsigned int update_flags>
  void ScratchData<dim, update_flags>::initialize_navier_stokes()
  {
    velocity.first_vector_component = u_lower = ordering.u_lower;
    pressure.component = p_lower = ordering.p_lower;

    kinematic_viscosity = physical_properties.fluids[0].kinematic_viscosity;
  }

  template <int dim, unsigned int update_flags>
  void ScratchData<dim, update_flags>::initialize_pseudo_solid()
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

  template <int dim, unsigned int update_flags>
  void ScratchData<dim, update_flags>::initialize_lagrange_multiplier()
  {
    AssertThrow(
      ordering.l_lower != numbers::invalid_unsigned_int,
      ExcMessage(
        "Cannot create ScratchData with Lagrange multiplier data because "
        "solver does not have a Lagrange multiplier variable."));

    lambda.first_vector_component = l_lower = ordering.l_lower;
  }

  template <int dim, unsigned int update_flags>
  void ScratchData<dim, update_flags>::initialize_cahn_hilliard()
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
    body_force            = physical_properties.body_force;
    tracer_limiter = CahnHilliard::get_limiter_function(cahn_hilliard_param);
  }

  template <int dim, unsigned int update_flags>
  void ScratchData<dim, update_flags>::initialize_compressible()
  {
    AssertThrow(
      ordering.t_lower != numbers::invalid_unsigned_int,
      ExcMessage(
        "Cannot create ScratchData with compressible data because solver does "
        "not have a temperature variable(s)."));

    temperature.component = t_lower = ordering.t_lower;

    density_ref     = physical_properties.fluids[0].density;
    pressure_ref    = physical_properties.fluids[0].pressure_ref;
    temperature_ref = physical_properties.fluids[0].temperature_ref;
    alpha_r         = 1.0 / pressure_ref;
    beta_r          = 1.0 / temperature_ref;
  }

  template <int dim, unsigned int update_flags>
  void ScratchData<dim, update_flags>::allocate()
  {
    components.resize(max_dofs_per_cell);
    JxW_moving.resize(n_q_points);
    JxW_fixed.resize(n_q_points);
    face_at_boundary.resize(n_faces);
    face_boundary_id.resize(n_faces, numbers::invalid_unsigned_int);
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
    previous_velocity_values.resize(time_handler.n_previous_solutions,
                                    std::vector<Tensor<1, dim>>(n_q_points));
    present_velocity_time_derivatives.resize(n_q_points);
    present_velocity_laplacians.resize(n_q_points);
    present_velocity_hessians.resize(n_q_points);
    present_velocity_grad_div.resize(n_q_points);
    present_pressure_gradients.resize(n_q_points);

    tau_supg_velocity.resize(n_q_points);
    grad_phi_u_first_component.resize(max_dofs_per_cell);

    tau_supg_velocity.resize(n_q_points);
    grad_phi_u_first_component.resize(max_dofs_per_cell);

    present_face_velocity_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    present_face_velocity_gradients.resize(
      n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
    present_face_velocity_sym_gradients.resize(
      n_faces, std::vector<SymmetricTensor<2, dim>>(n_faces_q_points));
    present_face_velocity_divergence.resize(
      n_faces, std::vector<double>(n_faces_q_points));
    present_face_pressure_values.resize(n_faces,
                                        std::vector<double>(n_faces_q_points));

#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
    face_velocity_source_term.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
#endif

    phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(max_dofs_per_cell));
    grad_phi_u.resize(n_q_points,
                      std::vector<Tensor<2, dim>>(max_dofs_per_cell));
    sym_grad_phi_u.resize(
      n_q_points, std::vector<SymmetricTensor<2, dim>>(max_dofs_per_cell));
    div_phi_u.resize(n_q_points, std::vector<double>(max_dofs_per_cell));
    laplacian_phi_u.resize(n_q_points,
                           std::vector<Tensor<1, dim>>(max_dofs_per_cell));
    grad_div_phi_u.resize(n_q_points,
                          std::vector<Tensor<1, dim>>(max_dofs_per_cell));
    phi_p.resize(n_q_points, std::vector<double>(max_dofs_per_cell));
    grad_phi_p.resize(n_q_points,
                      std::vector<Tensor<1, dim>>(max_dofs_per_cell));

    phi_u_face.resize(n_faces,
                      std::vector<std::vector<Tensor<1, dim>>>(
                        n_faces_q_points,
                        std::vector<Tensor<1, dim>>(max_dofs_per_cell)));
    phi_p_face.resize(n_faces,
                      std::vector<std::vector<double>>(n_faces_q_points,
                                                       std::vector<double>(
                                                         max_dofs_per_cell)));
    grad_phi_u_face.resize(n_faces,
                           std::vector<std::vector<Tensor<2, dim>>>(
                             n_faces_q_points,
                             std::vector<Tensor<2, dim>>(max_dofs_per_cell)));
    sym_grad_phi_u_face.resize(
      n_faces,
      std::vector<std::vector<SymmetricTensor<2, dim>>>(
        n_faces_q_points,
        std::vector<SymmetricTensor<2, dim>>(max_dofs_per_cell)));

    div_phi_u_face.resize(
      n_faces,
      std::vector<std::vector<double>>(n_faces_q_points,
                                       std::vector<double>(max_dofs_per_cell)));

    source_term_full_moving.resize(n_q_points, Vector<double>(n_components));
    source_term_velocity.resize(n_q_points);
    source_term_pressure.resize(n_q_points);

    exact_solution_full_cell.resize(n_q_points, Vector<double>(n_components));
    exact_velocity_values_cell.resize(n_q_points);
    exact_pressure_values_cell.resize(n_q_points);
    exact_temperature_values_cell.resize(n_q_points);

    exact_solution_full.resize(n_faces_q_points, Vector<double>(n_components));
    grad_exact_solution_full.resize(n_faces_q_points,
                                    std::vector<Tensor<1, dim>>(n_components));
    exact_face_velocity_gradients.resize(
      n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
    exact_face_velocity_divergences.resize(
      n_faces, std::vector<double>(n_faces_q_points));
    exact_face_pressure_values.resize(n_faces,
                                      std::vector<double>(n_faces_q_points));

#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
    exact_face_velocity_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    exact_face_lambda_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    exact_face_mesh_velocity_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
#endif

    if constexpr (enable_pseudo_solid)
    {
      lame_mu.resize(n_q_points);
      lame_lambda.resize(n_q_points);

      present_position_values.resize(n_q_points);
      present_position_gradients.resize(n_q_points);
      present_mesh_velocity_values.resize(n_q_points);
      previous_position_values.resize(time_handler.n_previous_solutions,
                                      std::vector<Tensor<1, dim>>(n_q_points));

      present_position_J.resize(n_q_points);
      present_position_inverse_gradients.resize(n_q_points);
      present_position_inverse_gradients_T.resize(n_q_points);

      present_face_position_values.resize(
        n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
      present_face_position_gradient.resize(
        n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));
      present_face_mesh_velocity_values.resize(
        n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
      previous_face_position_values.resize(
        n_faces,
        std::vector<std::vector<Tensor<1, dim>>>(
          time_handler.n_previous_solutions,
          std::vector<Tensor<1, dim>>(n_faces_q_points)));

      phi_x.resize(n_q_points, std::vector<Tensor<1, dim>>(max_dofs_per_cell));
      grad_phi_x.resize(n_q_points,
                        std::vector<Tensor<2, dim>>(max_dofs_per_cell));
      sym_grad_phi_x.resize(
        n_q_points, std::vector<SymmetricTensor<2, dim>>(max_dofs_per_cell));
      grad_phi_x_moving.resize(n_q_points,
                               std::vector<Tensor<2, dim>>(max_dofs_per_cell));
      if (enable_stabilization)
        hessian_phi_x_moving.resize(
          n_q_points, std::vector<Tensor<3, dim>>(max_dofs_per_cell));
      div_phi_x.resize(n_q_points, std::vector<double>(max_dofs_per_cell));
      trace_grad_phi_x.resize(n_q_points,
                              std::vector<double>(max_dofs_per_cell));
      phi_x_face.resize(n_faces,
                        std::vector<std::vector<Tensor<1, dim>>>(
                          n_faces_q_points,
                          std::vector<Tensor<1, dim>>(max_dofs_per_cell)));
      grad_phi_x_face.resize(n_faces,
                             std::vector<std::vector<Tensor<2, dim>>>(
                               n_faces_q_points,
                               std::vector<Tensor<2, dim>>(max_dofs_per_cell)));

      source_term_full_fixed.resize(n_q_points, Vector<double>(n_components));
      source_term_position.resize(n_q_points);

      grad_source_term_full.resize(n_q_points,
                                   std::vector<Tensor<1, dim>>(n_components));
      grad_source_velocity.resize(n_q_points);
      grad_source_pressure.resize(n_q_points);
      grad_source_tracer.resize(n_q_points);
      grad_source_potential.resize(n_q_points);
      grad_source_term_position_current_mesh.resize(n_q_points);

      delta_dx.resize(n_faces,
                      std::vector<std::vector<double>>(n_faces_q_points,
                                                       std::vector<double>(
                                                         max_dofs_per_cell)));
    }

    if constexpr (enable_lagrange_multiplier)
    {
      present_face_lambda_values.resize(
        n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
      phi_l_face.resize(n_faces,
                        std::vector<std::vector<Tensor<1, dim>>>(
                          n_faces_q_points,
                          std::vector<Tensor<1, dim>>(max_dofs_per_cell)));

      input_face_rigid_body_rotation_velocity.resize(
        n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    }

    if constexpr (enable_cahn_hilliard)
    {
      density.resize(n_q_points);
      derivative_density_wrt_tracer.resize(n_q_points);
      dynamic_viscosity.resize(n_q_points);
      derivative_dynamic_viscosity_wrt_tracer.resize(n_q_points);

      tracer_values.resize(n_q_points);
      tracer_time_derivatives.resize(n_q_points);
      tracer_gradients.resize(n_q_points);
      tracer_values_fixed.resize(n_q_points);
      tracer_gradients_fixed.resize(n_q_points);
      potential_values.resize(n_q_points);
      potential_gradients.resize(n_q_points);
      previous_tracer_values.resize(time_handler.n_previous_solutions,
                                    std::vector<double>(n_q_points));

      diffusive_flux.resize(n_q_points);
      shape_phi.resize(n_q_points, std::vector<double>(max_dofs_per_cell));
      grad_shape_phi.resize(n_q_points,
                            std::vector<Tensor<1, dim>>(max_dofs_per_cell));
      shape_phi_fixed.resize(n_q_points,
                             std::vector<double>(max_dofs_per_cell));
      grad_shape_phi_fixed.resize(
        n_q_points, std::vector<Tensor<1, dim>>(max_dofs_per_cell));
      shape_mu.resize(n_q_points, std::vector<double>(max_dofs_per_cell));
      grad_shape_mu.resize(n_q_points,
                           std::vector<Tensor<1, dim>>(max_dofs_per_cell));
      laplacian_shape_mu.resize(n_q_points,
                                std::vector<double>(max_dofs_per_cell));

      if (enable_tracer_stabilization)
      {
        potential_hessians.resize(n_q_points);
        potential_laplacians.resize(n_q_points);
        tau_supg_tracer.resize(n_q_points);
      }

      source_term_tracer.resize(n_q_points);
      source_term_potential.resize(n_q_points);
    }

    if constexpr (enable_compressible)
    {
      present_pressure_gradients.resize(n_q_points);
      present_pressure_absolute_values.resize(n_q_points);
      previous_pressure_values.resize(time_handler.n_previous_solutions,
                                      std::vector<double>(n_q_points));

      present_temperature_values.resize(n_q_points);
      present_temperature_absolute_values.resize(n_q_points);
      present_temperature_gradients.resize(n_q_points);
      previous_temperature_values.resize(time_handler.n_previous_solutions,
                                         std::vector<double>(n_q_points));

      phi_T.resize(n_q_points, std::vector<double>(max_dofs_per_cell));
      grad_phi_T.resize(n_q_points,
                        std::vector<Tensor<1, dim>>(max_dofs_per_cell));

      source_term_temperature.resize(n_q_points);

      present_face_temperature_values.resize(
        n_faces, std::vector<double>(n_faces_q_points));
      present_face_temperature_gradients.resize(
        n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
      present_face_temperature_absolute_values.resize(
        n_faces, std::vector<double>(n_faces_q_points));
      exact_face_temperature_gradients.resize(
        n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

      face_input_pressure_values.resize(n_faces,
                                        std::vector<double>(n_faces_q_points));

      face_input_heat_flux_values.resize(n_faces,
                                         std::vector<double>(n_faces_q_points));

      phi_T_face.resize(n_faces,
                        std::vector<std::vector<double>>(n_faces_q_points,
                                                         std::vector<double>(
                                                           max_dofs_per_cell)));

      grad_phi_T_face.resize(n_faces,
                             std::vector<std::vector<Tensor<1, dim>>>(
                               n_faces_q_points,
                               std::vector<Tensor<1, dim>>(max_dofs_per_cell)));

      density.resize(n_q_points);

      a_p.resize(n_q_points);
      b_T.resize(n_q_points);
    }
  }

  // Explicit instantiations for the used combinations only
  // Incompressible NS
  template class ScratchData<2, ns_only>;
  template class ScratchData<3, ns_only>;

  // Compressible NS
  template class ScratchData<2, compressible>;
  template class ScratchData<3, compressible>;

  // Incompressible NS with Lagrange multiplier
  template class ScratchData<2, lagrange_multiplier | with_hp_capabilities>;
  template class ScratchData<3, lagrange_multiplier | with_hp_capabilities>;

  // FSI solver with Lagrange multiplier
  template class ScratchData<2, pseudo_solid | lagrange_multiplier>;
  template class ScratchData<3, pseudo_solid | lagrange_multiplier>;

  // hp-FSI solver with Lagrange multiplier
  template class ScratchData<2,
                             pseudo_solid | lagrange_multiplier |
                               with_hp_capabilities>;
  template class ScratchData<3,
                             pseudo_solid | lagrange_multiplier |
                               with_hp_capabilities>;

  // Incompressible CHNS
  template class ScratchData<2, cahn_hilliard>;
  template class ScratchData<3, cahn_hilliard>;

  // Incompressible CHNS with mesh movement
  template class ScratchData<2, cahn_hilliard | pseudo_solid>;
  template class ScratchData<3, cahn_hilliard | pseudo_solid>;
} // namespace NavierStokesScratch
