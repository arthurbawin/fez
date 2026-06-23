#include <assembly/sponge_layer.h>
#include <boost/archive/text_iarchive.hpp>
#include <compare_matrix.h>
#include <compressible_ns_solver.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/multithread_info.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/trilinos_solver.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/fe_field_function.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <linear_solver.h>
#include <mesh.h>
#include <scratch_data.h>
#include <utilities.h>

#include <fstream>

template <int dim>
CompressibleNSSolver<dim>::CompressibleNSSolver(
  const ParameterReader<dim> &param)
  : NavierStokesSolver<dim>(param)
{
  if (param.finite_elements.use_quads)
    fe = std::make_unique<FESystem<dim>>(
      FESystem<dim>(FE_Q<dim>(param.finite_elements.velocity_degree) ^
                    dim),                                   // Velocity
      FE_Q<dim>(param.finite_elements.pressure_degree),     // Pressure
      FE_Q<dim>(param.finite_elements.temperature_degree)); // Temperature
  else
    fe = std::make_unique<FESystem<dim>>(
      FESystem<dim>(FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^
                    dim),                                      // Velocity
      FE_SimplexP<dim>(param.finite_elements.pressure_degree), // Pressure
      FE_SimplexP<dim>(
        param.finite_elements.temperature_degree)); // Temperature

  this->ordering = std::make_unique<ComponentOrderingCompressibleNS<dim>>();

  this->velocity_extractor =
    FEValuesExtractors::Vector(this->ordering->u_lower);
  this->pressure_extractor =
    FEValuesExtractors::Scalar(this->ordering->p_lower);
  temperature_extractor = FEValuesExtractors::Scalar(this->ordering->t_lower);

  this->velocity_mask = fe->component_mask(this->velocity_extractor);
  this->pressure_mask = fe->component_mask(this->pressure_extractor);
  temperature_mask    = fe->component_mask(temperature_extractor);

  this->field_names_and_masks["velocity"]    = this->velocity_mask;
  this->field_names_and_masks["pressure"]    = this->pressure_mask;
  this->field_names_and_masks["temperature"] = temperature_mask;

  /**
   * This solver uses a fixed mapping only.
   */
  mapping = this->fixed_mapping.get();

  // Create the initial condition functions for this problem, once the layout of
  // the variables is known (and in particular, the number of components).
  // FIXME: Is there a better way to create the functions?
  this->param.initial_conditions.create_initial_velocity(
    this->ordering->u_lower, this->ordering->n_components);
  this->param.initial_conditions.create_initial_pressure(
    this->ordering->p_lower, this->ordering->n_components);
  this->param.initial_conditions.create_initial_temperature(
    this->ordering->t_lower, this->ordering->n_components);

  if (param.mms_param.enable)
  {
    // Assign the manufactured solution
    this->exact_solution =
      std::make_shared<CompressibleNSSolver<dim>::MMSSolution>(
        this->time_handler.current_time, *this->ordering, param.mms);

    if (param.mms_param.force_source_term)
    {
      // Use the provided source term instead of the source term computed from
      // symbolic differentiation.
      this->source_terms = param.source_terms.fluid_source;
    }
    else
    {
      // Create the source term function for the given MMS and override source
      // terms
      this->source_terms =
        std::make_shared<CompressibleNSSolver<dim>::MMSSourceTerm>(
          this->time_handler.current_time,
          *this->ordering,
          param.physical_properties,
          param.mms);
    }

    // Create entry in error handler for temperature
    for (auto &[norm, handler] : this->error_handlers)
      handler.create_entry("T");
  }
  else
  {
    this->source_terms =
      std::make_shared<CompressibleNSSolver<dim>::SourceTerm>(
        this->time_handler.current_time, *this->ordering, param.source_terms);
    this->exact_solution = std::make_shared<Functions::ZeroFunction<dim>>(
      this->ordering->n_components);
  }
}

template <int dim>
void CompressibleNSSolver<dim>::MMSSourceTerm::vector_value(
  const Point<dim> &p,
  Vector<double>   &values) const
{
  const double mu      = physical_properties.fluids[0].dynamic_viscosity;
  const double k       = physical_properties.fluids[0].thermal_conductivity;
  const double rho_ref = physical_properties.fluids[0].density;
  const double cp =
    physical_properties.fluids[0].heat_capacity_at_constant_pressure;
  const double p_ref      = physical_properties.fluids[0].pressure_ref;
  const double T_ref      = physical_properties.fluids[0].temperature_ref;
  const auto   body_force = physical_properties.body_force;

  Tensor<1, dim> u, dudt_eulerian;
  for (unsigned int d = 0; d < dim; ++d)
  {
    dudt_eulerian[d] = mms.exact_velocity->time_derivative(p, d);
    u[d]             = mms.exact_velocity->value(p, d);
  }

  const double p_ex    = mms.exact_pressure->value(p);
  const double dpdt_ex = mms.exact_pressure->time_derivative(p);

  const double T_ex    = mms.exact_temperature->value(p);
  const double dTdt_ex = mms.exact_temperature->time_derivative(p);

  Tensor<2, dim> grad_u       = mms.exact_velocity->gradient_vi_xj(p);
  Tensor<1, dim> lap_u        = mms.exact_velocity->vector_laplacian(p);
  double         div_u        = mms.exact_velocity->divergence(p);
  Tensor<1, dim> grad_p       = mms.exact_pressure->gradient(p);
  Tensor<1, dim> grad_T       = mms.exact_temperature->gradient(p);
  double         lap_T        = mms.exact_temperature->laplacian(p);
  Tensor<1, dim> grad_u_dot_u = grad_u * u;
  double         u_dot_grad_p = u * grad_p;
  double         u_dot_grad_T = u * grad_T;
  Tensor<1, dim> grad_div_u   = mms.exact_velocity->grad_div(p);

  double alpha_r = 1.0 / p_ref;
  double beta_r  = 1.0 / T_ref;

  const double a_p = alpha_r / (alpha_r * p_ex + 1.0);
  const double b_T = beta_r / (beta_r * T_ex + 1.0);

  {
    double rho = rho_ref * (alpha_r * p_ex + 1.0) / (beta_r * T_ex + 1.0);

    // Navier-Stokes momentum (velocity) source term
    Tensor<1, dim> f =
      -(rho * (dudt_eulerian + grad_u_dot_u) + grad_p -
        mu * (lap_u + 1.0 / 3.0 * grad_div_u) - rho * body_force);

    for (unsigned int d = 0; d < dim; ++d)
      values[u_lower + d] = f[d];

    // Mass conservation (pressure) source term,
    // for div(u) + alpha_r/(alpha_r p^* + 1)[dp^*dt + u dot gradp^*] -
    // beta_r/(beta_r T^* + 1)[dT^*dt + u dot gradT^*] - f = 0
    // -> f = div(u_mms) + alpha_r/(alpha_r p^*_mms + 1)[dp^*_mmsdt + u_mms dot
    // gradp^_mms*] - beta_r/(beta_r T^*_mms + 1)[dT^_mms*dt + u_mms dot
    // gradT^*_mms]
    values[p_lower] = -(div_u + a_p * (dpdt_ex + u_dot_grad_p) -
                        b_T * (dTdt_ex + u_dot_grad_T));

    // Energy equation (temperature) source term
    Tensor<2, dim> D        = symmetrize(grad_u);
    const double   d_ddot_d = scalar_product(D, D);

    double source_energy =
      rho * cp * (dTdt_ex + u_dot_grad_T) - (dpdt_ex + u_dot_grad_p) -
      k * lap_T - 2.0 * mu * d_ddot_d + (2.0 / 3.0) * mu * div_u * div_u;

    values[t_lower] = -source_energy;
  }
}

template <int dim>
void CompressibleNSSolver<dim>::create_scratch_data()
{
  scratch_data = std::make_unique<ScratchData>(*this->ordering,
                                               *fe,
                                               *mapping,
                                               *mapping,
                                               *this->quadrature,
                                               *this->face_quadrature,
                                               this->time_handler,
                                               this->param);
}

template <int dim>
void CompressibleNSSolver<dim>::create_solver_specific_zero_constraints()
{
  for (const auto &[id, bc] : this->param.heat_bc)
  {
    if (bc.type == BoundaryConditions::Type::dirichlet_mms ||
        bc.type == BoundaryConditions::Type::input_function)
    {
      VectorTools::interpolate_boundary_values(*mapping,
                                               *this->dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(
                                                 this->ordering->n_components),
                                               this->zero_constraints,
                                               temperature_mask);
    }
  }
}

template <int dim>
void CompressibleNSSolver<dim>::create_solver_specific_nonzero_constraints()
{
  for (const auto &[id, bc] : this->param.heat_bc)
  {
    /**
     * Apply manufactured solution for temperature
     */
    if (bc.type == BoundaryConditions::Type::dirichlet_mms)
    {
      VectorTools::interpolate_boundary_values(*mapping,
                                               *this->dof_handler,
                                               id,
                                               *this->exact_solution,
                                               this->nonzero_constraints,
                                               temperature_mask);
    }

    else if (bc.type == BoundaryConditions::Type::input_function)
    {
      // Dirichlet temperature given by user
      VectorTools::interpolate_boundary_values(
        *mapping,
        *this->dof_handler,
        id,
        ScalarFunctionFromComponents<dim>(this->ordering->t_lower,
                                          this->ordering->n_components,
                                          *bc.temperature),
        this->nonzero_constraints,
        temperature_mask);
    }
  }
}

template <int dim>
void CompressibleNSSolver<dim>::set_solver_specific_initial_conditions()
{
  const Function<dim> *temperature_fun =
    this->param.initial_conditions.set_to_mms ?
      this->exact_solution.get() :
      this->param.initial_conditions.initial_temperature.get();

  // Set temperature
  VectorTools::interpolate(*mapping,
                           *this->dof_handler,
                           *temperature_fun,
                           this->newton_update,
                           temperature_mask);

  // Set pressure
  const Function<dim> *pressure_fun =
    this->param.initial_conditions.set_to_mms ?
      this->exact_solution.get() :
      this->param.initial_conditions.initial_pressure.get();

  VectorTools::interpolate(*mapping,
                           *this->dof_handler,
                           *pressure_fun,
                           this->newton_update,
                           this->pressure_mask);
}

template <int dim>
void CompressibleNSSolver<dim>::set_solver_specific_exact_solution()
{
  // Set temperature
  VectorTools::interpolate(*mapping,
                           *this->dof_handler,
                           *this->exact_solution,
                           this->local_evaluation_point,
                           temperature_mask);

  // Set pressure
  VectorTools::interpolate(*mapping,
                           *this->dof_handler,
                           *this->exact_solution,
                           this->local_evaluation_point,
                           this->pressure_mask);
}

template <int dim>
void CompressibleNSSolver<dim>::create_sparsity_pattern()
{
  //
  // Sparsity pattern and allocate matrix after the constraints have been
  // defined
  //
  DynamicSparsityPattern dsp(this->locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(*this->dof_handler,
                                  dsp,
                                  this->nonzero_constraints,
                                  /* keep_constrained_dofs = */ false);
  SparsityTools::distribute_sparsity_pattern(dsp,
                                             this->locally_owned_dofs,
                                             this->mpi_communicator,
                                             this->locally_relevant_dofs);
  this->system_matrix.reinit(this->locally_owned_dofs,
                             this->locally_owned_dofs,
                             dsp,
                             this->mpi_communicator);
}

template <int dim>
void CompressibleNSSolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  this->system_matrix = 0;

  CopyData copy_data(*fe);

#if defined(FEZ_WITH_PETSC)
  AssertThrow(
    MultithreadInfo::n_threads() == 1,
    ExcMessage(
      "Assembly is running with more than 1 thread, but uses PETSc wrappers "
      "for parallel matrix and vectors, which are not thread safe."));
#endif

  auto assembly_ptr =
    this->param.nonlinear_solver.analytic_jacobian ?
      &CompressibleNSSolver::assemble_local_matrix :
      &CompressibleNSSolver::assemble_local_matrix_finite_differences;

  // Assemble matrix (multithreaded if supported)
  WorkStream::run(this->dof_handler->begin_active(),
                  this->dof_handler->end(),
                  *this,
                  assembly_ptr,
                  &CompressibleNSSolver::copy_local_to_global_matrix,
                  *scratch_data,
                  copy_data);

  this->system_matrix.compress(VectorOperation::add);
}

template <int dim>
void CompressibleNSSolver<dim>::assemble_local_matrix_finite_differences(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  Verification::compute_local_matrix_finite_differences<dim>(
    cell,
    *this,
    &CompressibleNSSolver::assemble_local_rhs,
    scratch_data,
    copy_data);
}

template <int dim>
void CompressibleNSSolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      *this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_matrix      = copy_data.local_matrix();
  auto &local_dof_indices = copy_data.dof_indices();
  local_matrix            = 0;

  const double mu = this->param.physical_properties.fluids[0].dynamic_viscosity;
  const double k =
    this->param.physical_properties.fluids[0].thermal_conductivity;
  const double rho_ref = this->param.physical_properties.fluids[0].density;
  const double cp      = this->param.physical_properties.fluids[0]
                      .heat_capacity_at_constant_pressure;
  const double p_ref = this->param.physical_properties.fluids[0].pressure_ref;
  const double T_ref =
    this->param.physical_properties.fluids[0].temperature_ref;

  const double alpha_r = 1.0 / p_ref;
  const double beta_r  = 1.0 / T_ref;

  const auto body_force = this->param.physical_properties.body_force;

  const double bdf_c0 = this->time_handler.bdf_coefficients[0];

  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW = scratch_data.JxW_moving[q];

    const double rho = scratch_data.density[q];
    const double a_p = scratch_data.a_p[q];
    const double b_T = scratch_data.b_T[q];

    const auto &phi_u          = scratch_data.phi_u[q];
    const auto &grad_phi_u     = scratch_data.grad_phi_u[q];
    const auto &sym_grad_phi_u = scratch_data.sym_grad_phi_u[q];
    const auto &div_phi_u      = scratch_data.div_phi_u[q];
    const auto &phi_p          = scratch_data.phi_p[q];
    const auto &grad_phi_p     = scratch_data.grad_phi_p[q];
    const auto &phi_T          = scratch_data.phi_T[q];
    const auto &grad_phi_T     = scratch_data.grad_phi_T[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];
    const auto &present_velocity_divergence =
      scratch_data.present_velocity_divergence[q];
    const auto &present_velocity_sym_gradients =
      scratch_data.present_velocity_sym_gradients[q];
    const auto &present_pressure_values =
      scratch_data.present_pressure_values[q];
    const auto &present_pressure_gradients =
      scratch_data.present_pressure_gradients[q];
    const auto &present_temperature_values =
      scratch_data.present_temperature_values[q];
    const auto &present_temperature_gradients =
      scratch_data.present_temperature_gradients[q];

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratch_data.previous_velocity_values);

    const double dpdt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_pressure_values, scratch_data.previous_pressure_values);

    const double dTdt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q,
        present_temperature_values,
        scratch_data.previous_temperature_values);

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const unsigned int component_i = scratch_data.components[i];
      const bool         i_is_u      = this->ordering->is_velocity(component_i);
      const bool         i_is_p      = this->ordering->is_pressure(component_i);
      const bool         i_is_T = this->ordering->is_temperature(component_i);

      for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
      {
        const unsigned int component_j = scratch_data.components[j];
        const bool         j_is_u = this->ordering->is_velocity(component_j);
        const bool         j_is_p = this->ordering->is_pressure(component_j);
        const bool         j_is_T = this->ordering->is_temperature(component_j);

        bool   assemble        = false;
        double local_matrix_ij = 0.;

        if (i_is_u && j_is_u)
        {
          assemble = true;

          local_matrix_ij += phi_u[i] * rho * bdf_c0 * phi_u[j];
          local_matrix_ij += phi_u[i] * rho *
                             (grad_phi_u[j] * present_velocity_values +
                              present_velocity_gradients * phi_u[j]);
          local_matrix_ij +=
            2.0 * mu * scalar_product(sym_grad_phi_u[j], grad_phi_u[i]);
          local_matrix_ij += -2.0 / 3.0 * mu * div_phi_u[j] * div_phi_u[i];
        }

        if (i_is_u && j_is_p)
        {
          assemble = true;

          local_matrix_ij +=
            rho_ref * phi_u[i] *
            (alpha_r / (beta_r * present_temperature_values + 1.0)) * dudt *
            phi_p[j];
          local_matrix_ij +=
            rho_ref * phi_u[i] *
            (alpha_r / (beta_r * present_temperature_values + 1.0)) * phi_p[j] *
            (present_velocity_gradients * present_velocity_values);
          local_matrix_ij += -phi_p[j] * div_phi_u[i];

          local_matrix_ij += -rho_ref * body_force * phi_u[i] *
                             (alpha_r * phi_p[j]) /
                             (beta_r * present_temperature_values + 1);
        }

        if (i_is_u && j_is_T)
        {
          assemble = true;

          local_matrix_ij += -rho_ref * phi_u[i] * beta_r *
                             ((alpha_r * present_pressure_values + 1.0) /
                              ((beta_r * present_temperature_values + 1.0) *
                               (beta_r * present_temperature_values + 1.0))) *
                             dudt * phi_T[j];
          local_matrix_ij +=
            -rho_ref * phi_u[i] * beta_r *
            ((alpha_r * present_pressure_values + 1.0) /
             ((beta_r * present_temperature_values + 1.0) *
              (beta_r * present_temperature_values + 1.0))) *
            phi_T[j] * (present_velocity_gradients * present_velocity_values);

          local_matrix_ij +=
            rho_ref * body_force * phi_u[i] *
            (beta_r * phi_T[j] * (alpha_r * present_pressure_values + 1)) /
            ((beta_r * present_temperature_values + 1) *
             (beta_r * present_temperature_values + 1));
        }

        if (i_is_p && j_is_u)
        {
          assemble = true;

          local_matrix_ij += phi_p[i] * div_phi_u[j];
          local_matrix_ij +=
            phi_p[i] * a_p * phi_u[j] * present_pressure_gradients;
          local_matrix_ij +=
            -phi_p[i] * b_T * phi_u[j] * present_temperature_gradients;
        }

        if (i_is_p && j_is_p)
        {
          assemble = true;

          local_matrix_ij += phi_p[i] * (-alpha_r * alpha_r) /
                             ((alpha_r * present_pressure_values + 1.0) *
                              (alpha_r * present_pressure_values + 1.0)) *
                             dpdt * phi_p[j];
          local_matrix_ij += phi_p[i] * a_p * bdf_c0 * phi_p[j];
          local_matrix_ij += phi_p[i] * (-alpha_r * alpha_r) /
                             ((alpha_r * present_pressure_values + 1.0) *
                              (alpha_r * present_pressure_values + 1.0)) *
                             phi_p[j] * present_velocity_values *
                             present_pressure_gradients;
          local_matrix_ij +=
            phi_p[i] * a_p * present_velocity_values * grad_phi_p[j];
        }

        if (i_is_p && j_is_T)
        {
          assemble = true;

          local_matrix_ij += phi_p[i] * (beta_r * beta_r) /
                             ((beta_r * present_temperature_values + 1.0) *
                              (beta_r * present_temperature_values + 1.0)) *
                             dTdt * phi_T[j];
          local_matrix_ij += -phi_p[i] * b_T * bdf_c0 * phi_T[j];
          local_matrix_ij += phi_p[i] * (beta_r * beta_r) /
                             ((beta_r * present_temperature_values + 1.0) *
                              (beta_r * present_temperature_values + 1.0)) *
                             present_velocity_values *
                             present_temperature_gradients * phi_T[j];
          local_matrix_ij +=
            -phi_p[i] * b_T * present_velocity_values * grad_phi_T[j];
        }

        if (i_is_T && j_is_u)
        {
          assemble = true;

          local_matrix_ij +=
            phi_T[i] * rho * cp * phi_u[j] * present_temperature_gradients;
          local_matrix_ij += -phi_T[i] * phi_u[j] * present_pressure_gradients;
          local_matrix_ij +=
            -phi_T[i] * 4.0 * mu *
            scalar_product(present_velocity_sym_gradients, sym_grad_phi_u[j]);
          local_matrix_ij += phi_T[i] * 4.0 / 3.0 * mu *
                             present_velocity_divergence * div_phi_u[j];
        }

        if (i_is_T && j_is_p)
        {
          assemble = true;

          local_matrix_ij += phi_T[i] * rho_ref * cp * alpha_r /
                             (beta_r * present_temperature_values + 1.0) *
                             dTdt * phi_p[j];
          local_matrix_ij += phi_T[i] * rho_ref * cp * alpha_r /
                             (beta_r * present_temperature_values + 1.0) *
                             phi_p[j] * present_velocity_values *
                             present_temperature_gradients;
          local_matrix_ij += -phi_T[i] * bdf_c0 * phi_p[j];
          local_matrix_ij +=
            -phi_T[i] * present_velocity_values * grad_phi_p[j];
        }

        if (i_is_T && j_is_T)
        {
          assemble = true;

          local_matrix_ij += -phi_T[i] * rho_ref * cp *
                             (alpha_r * present_pressure_values + 1.0) /
                             ((beta_r * present_temperature_values + 1.0) *
                              (beta_r * present_temperature_values + 1.0)) *
                             beta_r * dTdt * phi_T[j];
          local_matrix_ij += phi_T[i] * rho * cp * bdf_c0 * phi_T[j];
          local_matrix_ij += -phi_T[i] * rho_ref * cp * beta_r *
                             (alpha_r * present_pressure_values + 1.0) /
                             ((beta_r * present_temperature_values + 1.0) *
                              (beta_r * present_temperature_values + 1.0)) *
                             phi_T[j] * present_velocity_values *
                             present_temperature_gradients;
          local_matrix_ij +=
            phi_T[i] * rho * cp * present_velocity_values * grad_phi_T[j];
          local_matrix_ij += k * grad_phi_T[i] * grad_phi_T[j];
        }

        if (assemble)
        {
          local_matrix_ij *= JxW;
          local_matrix(i, j) += local_matrix_ij;
        }
      }
    }
  }

  // Sponge layer relaxation: non-physical numerical absorber, kept
  // separate from the NS contributions for clarity.
  {
    const Assembly::SpongeMaterialConstants material{rho_ref, p_ref, T_ref, cp};
    Assembly::sponge_layer_matrix<dim>(*this->ordering,
                                       this->param.sponge_layer,
                                       material,
                                       scratch_data,
                                       local_matrix);
  }

  //
  // Face contributions
  //
  if (cell->at_boundary())
    for (const auto i_face : cell->face_indices())
    {
      const auto &face = cell->face(i_face);
      if (face->at_boundary())
      {
        const auto boundary_id = scratch_data.face_boundary_id[i_face];
        if (this->param.fluid_bc.at(boundary_id).type ==
            BoundaryConditions::Type::open_mms)
        {
          DEAL_II_NOT_IMPLEMENTED();
        }

        const auto &bc_fluid = this->param.fluid_bc.at(boundary_id);
        if (bc_fluid.type == BoundaryConditions::Type::weak_pressure ||
            bc_fluid.type ==
              BoundaryConditions::Type::no_tangential_flow_with_weak_pressure)
        {
          for (unsigned int q = 0; q < scratch_data.n_faces_q_points; ++q)
          {
            const double face_JxW = scratch_data.face_JxW_moving[i_face][q];
            const auto  &n        = scratch_data.face_normals_moving[i_face][q];

            const auto &phi_u_face     = scratch_data.phi_u_face[i_face][q];
            const auto &div_phi_u_face = scratch_data.div_phi_u_face[i_face][q];
            const auto &sym_grad_phi_u_face =
              scratch_data.sym_grad_phi_u_face[i_face][q];
            // const auto &phi_p_face = scratch_data.phi_p_face[i_face][q];

            for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
            {
              const unsigned int component_i = scratch_data.components[i];
              const bool i_is_u = this->ordering->is_velocity(component_i);

              for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
              {
                const unsigned int component_j = scratch_data.components[j];
                const bool j_is_u = this->ordering->is_velocity(component_j);
                // const bool j_is_p = this->ordering->is_pressure(component_j);

                double local_matrix_ij = 0.0;
                bool   assemble        = false;

                if (i_is_u && j_is_u)
                {
                  assemble = true;

                  local_matrix_ij +=
                    (-mu * 2.0 * sym_grad_phi_u_face[j] * n * phi_u_face[i] +
                     2.0 / 3.0 * mu * div_phi_u_face[j] * n * phi_u_face[i]);
                }

                // if (i_is_u && j_is_p)
                // {
                //   assemble = true;

                //   local_matrix_ij += phi_p_face[j] * (n * phi_u_face[i]);
                // }

                if (assemble)
                {
                  local_matrix_ij *= face_JxW;
                  local_matrix(i, j) += local_matrix_ij;
                }
              }
            }
          }
        }
      }
    }

  cell->get_dof_indices(local_dof_indices);
}

template <int dim>
void CompressibleNSSolver<dim>::copy_local_to_global_matrix(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_matrix(),
                                                    copy_data.dof_indices(),
                                                    this->system_matrix);
}

template <int dim>
void CompressibleNSSolver<dim>::compare_analytical_matrix_with_fd()
{
  CopyData copy_data(*fe);
  Verification::compare_analytical_matrix_with_fd<dim>(
    *this,
    &CompressibleNSSolver::assemble_local_matrix,
    &CompressibleNSSolver::assemble_local_rhs,
    *scratch_data,
    copy_data,
    this->param.nonlinear_solver.write_problematic_elements);
}

template <int dim>
void CompressibleNSSolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  this->system_rhs = 0;

  CopyData copy_data(*fe);

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(this->dof_handler->begin_active(),
                  this->dof_handler->end(),
                  *this,
                  &CompressibleNSSolver::assemble_local_rhs,
                  &CompressibleNSSolver::copy_local_to_global_rhs,
                  *scratch_data,
                  copy_data);

  this->system_rhs.compress(VectorOperation::add);
}

template <int dim>
void CompressibleNSSolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      *this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_rhs         = copy_data.local_rhs();
  auto &local_dof_indices = copy_data.dof_indices();
  local_rhs               = 0;

  const double mu = this->param.physical_properties.fluids[0].dynamic_viscosity;
  const double k =
    this->param.physical_properties.fluids[0].thermal_conductivity;
  const double cp = this->param.physical_properties.fluids[0]
                      .heat_capacity_at_constant_pressure;
  const double rho_ref = this->param.physical_properties.fluids[0].density;
  const double p_ref   = this->param.physical_properties.fluids[0].pressure_ref;
  const double T_ref =
    this->param.physical_properties.fluids[0].temperature_ref;

  const double alpha_r = 1.0 / p_ref;
  const double beta_r  = 1.0 / T_ref;

  const auto body_force = this->param.physical_properties.body_force;

  //
  // Volume contributions
  //
  const SymmetricTensor<2, dim> identity_tensor = unit_symmetric_tensor<dim>();
  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW = scratch_data.JxW_moving[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];
    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];
    const auto &present_pressure_values =
      scratch_data.present_pressure_values[q];
    const auto &present_pressure_gradients =
      scratch_data.present_pressure_gradients[q];
    const auto &source_term_velocity = scratch_data.source_term_velocity[q];
    const auto &source_term_pressure = scratch_data.source_term_pressure[q];
    const auto &source_term_temperature =
      scratch_data.source_term_temperature[q];
    const double present_velocity_divergence =
      trace(present_velocity_gradients);
    const auto &D = scratch_data.present_velocity_sym_gradients[q];
    const auto &present_temperature_values =
      scratch_data.present_temperature_values[q];
    const auto &present_temperature_gradients =
      scratch_data.present_temperature_gradients[q];

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_velocity_values, scratch_data.previous_velocity_values);

    const double dpdt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q, present_pressure_values, scratch_data.previous_pressure_values);

    const double dTdt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q,
        present_temperature_values,
        scratch_data.previous_temperature_values);

    const double rho = rho_ref * ((alpha_r * present_pressure_values + 1.0) /
                                  (beta_r * present_temperature_values + 1.0));

    const double a_p = alpha_r / (alpha_r * present_pressure_values + 1.0);
    const double b_T = beta_r / (beta_r * present_temperature_values + 1.0);

    const auto &phi_p      = scratch_data.phi_p[q];
    const auto &phi_u      = scratch_data.phi_u[q];
    const auto &phi_T      = scratch_data.phi_T[q];
    const auto &grad_phi_u = scratch_data.grad_phi_u[q];
    const auto &div_phi_u  = scratch_data.div_phi_u[q];
    const auto &grad_phi_T = scratch_data.grad_phi_T[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      double local_rhs_i = -(
        // Continuity
        present_velocity_divergence * phi_p[i] +
        a_p * (dpdt + present_velocity_values * present_pressure_gradients) *
          phi_p[i] -
        b_T * (dTdt + present_velocity_values * present_temperature_gradients) *
          phi_p[i] +
        source_term_pressure * phi_p[i]

        // Momentum
        + rho * (dudt * phi_u[i] + present_velocity_gradients *
                                     present_velocity_values * phi_u[i]) -
        present_pressure_values * div_phi_u[i] +
        scalar_product(2 * mu * D - 2.0 / 3.0 * mu *
                                      present_velocity_divergence *
                                      identity_tensor,
                       grad_phi_u[i]) +
        source_term_velocity * phi_u[i] -
        rho * body_force * phi_u[i]

        // Energy
        + rho * cp *
            (dTdt + present_velocity_values * present_temperature_gradients) *
            phi_T[i] -
        (dpdt + present_velocity_values * present_pressure_gradients) *
          phi_T[i] +
        k * grad_phi_T[i] * present_temperature_gradients -
        2.0 * mu * scalar_product(D, D) * phi_T[i] +
        2.0 / 3.0 * mu * present_velocity_divergence *
          present_velocity_divergence * phi_T[i] +
        source_term_temperature * phi_T[i]);

      local_rhs_i *= JxW;
      local_rhs(i) += local_rhs_i;
    }
  }

  // Sponge layer relaxation: non-physical numerical absorber, kept
  // separate from the NS contributions for clarity.
  {
    const Assembly::SpongeMaterialConstants material{rho_ref, p_ref, T_ref, cp};
    Assembly::sponge_layer_rhs<dim>(*this->ordering,
                                    this->param.sponge_layer,
                                    material,
                                    scratch_data,
                                    local_rhs);
  }

  //
  // Face contributions
  //
  if (cell->at_boundary())
    for (const auto i_face : cell->face_indices())
    {
      const auto &face = cell->face(i_face);
      if (face->at_boundary())
      {
        // Open boundary condition with prescribed manufactured solution
        if (this->param.fluid_bc.at(scratch_data.face_boundary_id[i_face])
              .type == BoundaryConditions::Type::pressure_mms)
        {
          // DEAL_II_NOT_IMPLEMENTED();
          for (unsigned int q = 0; q < scratch_data.n_faces_q_points; ++q)
          {
            const double face_JxW = scratch_data.face_JxW_moving[i_face][q];
            const auto  &n        = scratch_data.face_normals_moving[i_face][q];

            const auto &grad_u_exact =
              scratch_data.exact_face_velocity_gradients[i_face][q];
            const auto div_u_exact =
              scratch_data.exact_face_velocity_divergences[i_face][q];
            const double p_exact =
              scratch_data.exact_face_pressure_values[i_face][q];

            const SymmetricTensor<2, dim> D = symmetrize(grad_u_exact);
            const SymmetricTensor<2, dim> tau =
              2.0 * mu * D - 2.0 / 3.0 * mu * div_u_exact * identity_tensor;

            const auto &phi_u_face = scratch_data.phi_u_face[i_face][q];

            for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
            {
              local_rhs(i) -=
                ((p_exact * identity_tensor - tau) * n * phi_u_face[i]) *
                face_JxW;
            }
          }
        }

        // Open boundary condition with prescribed manufactured solution (not
        // implemented for compressible flow)
        if (this->param.fluid_bc.at(scratch_data.face_boundary_id[i_face])
              .type == BoundaryConditions::Type::open_mms)
          DEAL_II_NOT_IMPLEMENTED();

        // Pressure condition on a face (traction)
        const auto  boundary_id = scratch_data.face_boundary_id[i_face];
        const auto &bc_fluid    = this->param.fluid_bc.at(boundary_id);
        if (bc_fluid.type == BoundaryConditions::Type::weak_pressure ||
            bc_fluid.type ==
              BoundaryConditions::Type::no_tangential_flow_with_weak_pressure)
        {
          for (unsigned int q = 0; q < scratch_data.n_faces_q_points; ++q)
          {
            const double face_JxW = scratch_data.face_JxW_moving[i_face][q];
            const auto  &n        = scratch_data.face_normals_moving[i_face][q];

            const auto &present_face_velocity_gradients =
              scratch_data.present_face_velocity_gradients[i_face][q];
            const auto &present_face_velocity_divergence =
              scratch_data.present_face_velocity_divergence[i_face][q];
            const double &pressure_bc =
              scratch_data.face_input_pressure_values[i_face][q];

            const SymmetricTensor<2, dim> D =
              symmetrize(present_face_velocity_gradients);
            const SymmetricTensor<2, dim> tau =
              2.0 * mu * D - 2.0 / 3.0 * mu * present_face_velocity_divergence *
                               identity_tensor;

            const auto &phi_u_face = scratch_data.phi_u_face[i_face][q];

            for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
            {
              local_rhs(i) -=
                ((pressure_bc * identity_tensor - tau) * n * phi_u_face[i]) *
                face_JxW;
            }
          }
        }

        const auto  boundary_id_heat = scratch_data.face_boundary_id[i_face];
        const auto &bc_heat          = this->param.heat_bc.at(boundary_id_heat);

        if (bc_heat.type == BoundaryConditions::Type::heat_flux)
        {
          for (unsigned int q = 0; q < scratch_data.n_faces_q_points; ++q)
          {
            const double face_JxW   = scratch_data.face_JxW_moving[i_face][q];
            const auto  &phi_T_face = scratch_data.phi_T_face[i_face][q];

            const double q_n =
              scratch_data.face_input_heat_flux_values[i_face][q];

            for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
            {
              const unsigned int component_i = scratch_data.components[i];
              const bool i_is_T = this->ordering->is_temperature(component_i);

              if (i_is_T)
                local_rhs(i) -= q_n * phi_T_face[i] * face_JxW;
            }
          }
        }

        if (bc_heat.type == BoundaryConditions::Type::no_flux)
        {
          // Nothing to add
        }
      }
    }

  cell->get_dof_indices(local_dof_indices);
}

template <int dim>
void CompressibleNSSolver<dim>::copy_local_to_global_rhs(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_rhs(),
                                                    copy_data.dof_indices(),
                                                    this->system_rhs);
}

template <int dim>
void CompressibleNSSolver<dim>::compute_solver_specific_errors()
{
  const unsigned int n_active_cells = this->triangulation->n_active_cells();
  Vector<double>     cellwise_errors(n_active_cells);

  const ComponentSelectFunction<dim> temperature_comp_select(
    this->ordering->t_lower, this->ordering->n_components);

  this->compute_and_add_errors(*mapping,
                               *this->exact_solution,
                               cellwise_errors,
                               temperature_comp_select,
                               "T");
}

//
// Restart from incompressible checkpoint
//

namespace
{
  types::global_dof_index read_archive_index_token(std::istream &in)
  {
    std::string token;
    in >> token;
    AssertThrow(in, ExcMessage("Unexpected end of checkpoint archive."));

    std::size_t parsed = 0;
    const auto  value =
      static_cast<types::global_dof_index>(std::stoull(token, &parsed));
    AssertThrow(parsed == token.size(),
                ExcMessage("Expected an integer token in checkpoint archive."));

    return value;
  }

  void scan_to_archive_index(std::istream                 &in,
                             const types::global_dof_index expected_value)
  {
    std::string token;
    while (in >> token)
    {
      try
      {
        std::size_t parsed = 0;
        const auto  value =
          static_cast<types::global_dof_index>(std::stoull(token, &parsed));
        if (parsed == token.size() && value == expected_value)
          return;
      }
      catch (...)
      {}
    }

    AssertThrow(false,
                ExcMessage("Could not locate the expected vector size (" +
                           std::to_string(expected_value) +
                           ") in checkpoint archive."));
  }

  void read_archived_vector(std::istream                 &in,
                            LA::ParVectorType            &vector,
                            const types::global_dof_index expected_global_size)
  {
    scan_to_archive_index(in, expected_global_size);

    types::global_dof_index range_first  = read_archive_index_token(in);
    types::global_dof_index range_second = read_archive_index_token(in);

    const types::global_dof_index local_size = vector.locally_owned_size();
    for (unsigned int i = 0; i < 16 && range_second - range_first != local_size;
         ++i)
    {
      range_first  = range_second;
      range_second = read_archive_index_token(in);
    }

    AssertThrow(range_second >= range_first &&
                  range_second - range_first == local_size,
                ExcMessage("Checkpoint local range [" +
                           std::to_string(range_first) + ", " +
                           std::to_string(range_second) +
                           ") does not match locally owned size (" +
                           std::to_string(local_size) + ")."));

    PetscScalar *array = nullptr;
    int          ierr  = VecGetArray(vector.petsc_vector(), &array);
    AssertThrow(ierr == 0, ExcPETScError(ierr));

    for (types::global_dof_index i = 0; i < local_size; ++i)
      in >> array[i];

    ierr = VecRestoreArray(vector.petsc_vector(), &array);
    AssertThrow(ierr == 0, ExcPETScError(ierr));
    AssertThrow(in,
                ExcMessage("Could not read vector values from checkpoint."));

    vector.update_ghost_values();
  }

  void read_archived_double_vector(std::istream        &in,
                                   std::vector<double> &values)
  {
    const unsigned int size = read_archive_index_token(in);
    (void)read_archive_index_token(in); // item version

    values.resize(size);
    for (auto &value : values)
      in >> value;

    AssertThrow(in, ExcMessage("Could not read time vector from checkpoint."));
  }

  void read_archived_time_handler(std::istream &in, TimeHandler &time_handler)
  {
    (void)read_archive_index_token(in); // TimeHandler class id
    (void)read_archive_index_token(in); // TimeHandler class version

    in >> time_handler.initial_time;
    in >> time_handler.final_time;
    in >> time_handler.current_time;
    in >> time_handler.current_time_iteration;
    read_archived_double_vector(in, time_handler.simulation_times);
    in >> time_handler.current_dt;
    read_archived_double_vector(in, time_handler.time_steps);
    read_archived_double_vector(in, time_handler.bdf_coefficients);

    unsigned int with_adaptive_timestep = 0;
    unsigned int steady_scheme          = 0;
    in >> with_adaptive_timestep;
    in >> steady_scheme;
    AssertThrow(in, ExcMessage("Could not read time handler from checkpoint."));

    time_handler.with_adaptive_timestep = with_adaptive_timestep;
    time_handler.steady_scheme          = steady_scheme;
  }

  /**
   * Maps an incompressible NS solution (dim+1 components: velocity + kinematic
   * pressure p_k = p/rho) to the compressible NS layout (dim+2 components:
   * velocity + pressure perturbation p* + temperature perturbation T*) via
   * the linearized ideal-gas EOS evaluated at rho = rho_ref:
   *
   *   velocity  : direct transfer
   *   p*        = rho_ref * p_k
   *   T*        = (rho_ref * T_ref / p_ref) * p_k
   */
  template <int dim>
  class IncompressibleRestartAdapter : public Function<dim>
  {
  public:
    IncompressibleRestartAdapter(
      Functions::FEFieldFunction<dim, LA::ParVectorType> &fe_field,
      const unsigned int                                  n_comp_components,
      const unsigned int                                  p_lower,
      const unsigned int                                  t_lower,
      const double                                        rho_ref,
      const double                                        p_ref,
      const double                                        T_ref)
      : Function<dim>(n_comp_components)
      , fe_field_(fe_field)
      , p_lower_(p_lower)
      , t_lower_(t_lower)
      , rho_ref_(rho_ref)
      , p_ref_(p_ref)
      , T_ref_(T_ref)
    {}

    virtual double value(const Point<dim>  &p,
                         const unsigned int c) const override
    {
      if (c < p_lower_) // velocity components
        return fe_field_.value(p, c);
      const double p_k = fe_field_.value(p, p_lower_); // kinematic pressure
      if (c == p_lower_)
        return rho_ref_ * p_k;
      if (c == t_lower_)
        return (rho_ref_ * T_ref_ / p_ref_) * p_k;
      return 0.0;
    }

  private:
    Functions::FEFieldFunction<dim, LA::ParVectorType> &fe_field_;
    const unsigned int                                  p_lower_;
    const unsigned int                                  t_lower_;
    const double                                        rho_ref_;
    const double                                        p_ref_;
    const double                                        T_ref_;
  };
} // namespace

template <int dim>
void CompressibleNSSolver<dim>::restart()
{
  this->pcout << std::endl;
  this->pcout << "--- Reading checkpoint... ---" << std::endl << std::endl;

  const std::string prefix =
    this->param.output.output_dir + this->param.checkpoint_restart.filename;

  this->triangulation->load(prefix);

  // Build the FE for an incompressible solver. This is used both to
  // detect the checkpoint type (by comparing the global DOF count of the
  // first archived vector against the incompressible/compressible counts)
  // and, if needed, to read the incompressible-shaped vectors during the
  // restart mapping.
  const auto                    &fp = this->param.finite_elements;
  std::shared_ptr<FESystem<dim>> fe_incomp;
  if (fp.use_quads)
    fe_incomp =
      std::make_shared<FESystem<dim>>(FE_Q<dim>(fp.velocity_degree) ^ dim,
                                      FE_Q<dim>(fp.pressure_degree));
  else
    fe_incomp =
      std::make_shared<FESystem<dim>>(FE_SimplexP<dim>(fp.velocity_degree) ^
                                        dim,
                                      FE_SimplexP<dim>(fp.pressure_degree));

  types::global_dof_index n_incomp_dofs = 0;
  types::global_dof_index n_comp_dofs   = 0;
  {
    DoFHandler<dim> probe_incomp(*this->triangulation);
    probe_incomp.distribute_dofs(*fe_incomp);
    n_incomp_dofs = probe_incomp.n_dofs();

    DoFHandler<dim> probe_comp(*this->triangulation);
    probe_comp.distribute_dofs(*this->fe);
    n_comp_dofs = probe_comp.n_dofs();
  }

  // The Boost text archive starts with "22 serialization::archive VERSION"
  // followed by class-tracking metadata, then the global DOF count of the
  // first serialized vector. Scan ahead a few tokens to match either the
  // incompressible or the compressible global count.
  bool from_incompressible = false;
  {
    std::ifstream file(prefix + "_rank" + std::to_string(this->mpi_rank));
    AssertThrow(file.is_open(),
                ExcMessage("Could not open checkpoint file for restart."));

    std::string tok;
    file >> tok >> tok >> tok; // skip "22 serialization::archive VERSION"

    bool detected = false;
    for (int i = 0; i < 30 && (file >> tok); ++i)
    {
      try
      {
        const auto value =
          static_cast<types::global_dof_index>(std::stoull(tok));
        if (value == n_incomp_dofs)
        {
          from_incompressible = true;
          detected            = true;
          break;
        }
        if (value == n_comp_dofs)
        {
          detected = true;
          break;
        }
      }
      catch (...)
      {}
    }
    AssertThrow(detected,
                ExcMessage(
                  "Could not detect checkpoint type. Expected the "
                  "first archived vector size to match either the "
                  "incompressible (" +
                  std::to_string(n_incomp_dofs) + ") or the compressible (" +
                  std::to_string(n_comp_dofs) + ") global DOF count."));
  }

  if (from_incompressible)
  {
    this->pcout << "--- Incompressible checkpoint detected: restarting with "
                   "EOS-mapped velocity, pressure, and temperature ---"
                << std::endl
                << std::endl;
    restart_from_incompressible_checkpoint();
  }
  else
  {
    // Replicate NavierStokesSolver<dim>::restart() without reloading the
    // triangulation, which has already been loaded above.
    this->setup_dofs();

    std::ifstream checkpoint_file(prefix + "_rank" +
                                  std::to_string(this->mpi_rank));
    AssertThrow(checkpoint_file,
                ExcMessage("Could not read from the checkpoint file."));
    boost::archive::text_iarchive archive(checkpoint_file);

    archive >> *this;
    archive >> this->time_handler;

    this->time_handler.update_parameters_after_restart(
      this->param.time_integration, this->pcout);
  }
}

template <int dim>
void CompressibleNSSolver<dim>::restart_from_incompressible_checkpoint()
{
  const std::string prefix =
    this->param.output.output_dir + this->param.checkpoint_restart.filename;

  const auto                    &fp = this->param.finite_elements;
  std::shared_ptr<FESystem<dim>> fe_incomp;
  if (fp.use_quads)
    fe_incomp =
      std::make_shared<FESystem<dim>>(FE_Q<dim>(fp.velocity_degree) ^ dim,
                                      FE_Q<dim>(fp.pressure_degree));
  else
    fe_incomp =
      std::make_shared<FESystem<dim>>(FE_SimplexP<dim>(fp.velocity_degree) ^
                                        dim,
                                      FE_SimplexP<dim>(fp.pressure_degree));

  DoFHandler<dim> dof_handler_incomp(*this->triangulation);
  dof_handler_incomp.distribute_dofs(*fe_incomp);

  const IndexSet owned_incomp = dof_handler_incomp.locally_owned_dofs();
  const IndexSet relevant_incomp =
    DoFTools::extract_locally_relevant_dofs(dof_handler_incomp);

  LA::ParVectorType              solution_incomp(owned_incomp,
                                    relevant_incomp,
                                    this->mpi_communicator);
  std::vector<LA::ParVectorType> previous_solutions_incomp;
  {
    std::ifstream checkpoint_file(prefix + "_rank" +
                                  std::to_string(this->mpi_rank));
    AssertThrow(checkpoint_file.is_open(),
                ExcMessage("Could not open checkpoint file for restart."));

    std::string token;
    checkpoint_file >> token >> token >> token; // archive header

    const types::global_dof_index n_incomp_dofs = dof_handler_incomp.n_dofs();
    read_archived_vector(checkpoint_file, solution_incomp, n_incomp_dofs);

    const unsigned int n_previous_solutions =
      read_archive_index_token(checkpoint_file);
    previous_solutions_incomp.resize(n_previous_solutions);
    for (auto &previous_solution_incomp : previous_solutions_incomp)
    {
      previous_solution_incomp.reinit(owned_incomp,
                                      relevant_incomp,
                                      this->mpi_communicator);
      read_archived_vector(checkpoint_file,
                           previous_solution_incomp,
                           n_incomp_dofs);
    }

    read_archived_time_handler(checkpoint_file, this->time_handler);
  }

  this->setup_dofs();

  const auto  &fluid   = this->param.physical_properties.fluids[0];
  const double rho_ref = fluid.density;
  const double p_ref   = fluid.pressure_ref;
  const double T_ref   = fluid.temperature_ref;

  const auto map_incompressible_solution =
    [&](const LA::ParVectorType &incompressible_solution,
        LA::ParVectorType       &compressible_solution) {
      Functions::FEFieldFunction<dim, LA::ParVectorType> incomp_field(
        dof_handler_incomp, incompressible_solution, *mapping);
      IncompressibleRestartAdapter<dim> adapter(incomp_field,
                                                this->ordering->n_components,
                                                this->ordering->p_lower,
                                                this->ordering->t_lower,
                                                rho_ref,
                                                p_ref,
                                                T_ref);
      VectorTools::interpolate(*mapping,
                               *this->dof_handler,
                               adapter,
                               compressible_solution);
      compressible_solution.compress(VectorOperation::insert);
    };

  map_incompressible_solution(solution_incomp, this->newton_update);

  *this->present_solution      = this->newton_update;
  this->local_evaluation_point = this->newton_update;
  this->evaluation_point       = *this->present_solution;
  this->present_solution->update_ghost_values();
  this->evaluation_point.update_ghost_values();

  const bool target_simulation_is_steady =
    this->param.time_integration.scheme ==
    Parameters::TimeIntegration::Scheme::stationary;
  if (previous_solutions_incomp.empty() && !target_simulation_is_steady)
  {
    for (auto &prev : *this->previous_solutions)
    {
      prev = *this->present_solution;
      prev.update_ghost_values();
    }
  }
  else
  {
    AssertThrow(previous_solutions_incomp.size() ==
                  this->previous_solutions->size(),
                ExcMessage(
                  "The number of previous solutions in the incompressible "
                  "checkpoint does not match the number of previous solutions "
                  "used by the compressible simulation. Restarting with a "
                  "different unsteady time integration scheme is not "
                  "supported."));

    for (unsigned int i = 0; i < this->previous_solutions->size(); ++i)
    {
      LA::ParVectorType mapped_previous(this->locally_owned_dofs,
                                        this->mpi_communicator);
      map_incompressible_solution(previous_solutions_incomp[i],
                                  mapped_previous);

      (*this->previous_solutions)[i] = mapped_previous;
      (*this->previous_solutions)[i].update_ghost_values();
    }
  }

  this->time_handler.update_parameters_after_restart(
    this->param.time_integration, this->pcout);
}

// Explicit instantiation
template class CompressibleNSSolver<2>;
template class CompressibleNSSolver<3>;
