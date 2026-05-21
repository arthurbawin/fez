
#include <assembly/boundary_forms.h>
#include <assembly/lagrange_multiplier.h>
#include <compare_matrix.h>
#include <components_ordering.h>
#include <copy_data.h>
#include <deal.II/base/mpi.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <lagrange_multiplier_tools.h>
#include <linear_solver.h>
#include <mesh.h>
#include <monolithic_fsi_solver.h>
#include <post_processing_tools.h>
#include <scratch_data.h>
#include <utilities.h>
#include <algorithm>
#include <deal.II/base/quadrature_lib.h>
#include <fstream>
#include <limits>
#include <array>
#include <cmath>

namespace MeshConcentrationTools
{


  template <int dim>
  Tensor<1, dim>
  cartesian_direction(const unsigned int d)
  {
    AssertIndexRange(d, dim);

    Tensor<1, dim> e;
    e[d] = 1.0;

    return e;
  }


  template <int dim>
  Tensor<2, dim>
  identity_tensor()
  {
    Tensor<2, dim> I;

    for (unsigned int d = 0; d < dim; ++d)
      I[d][d] = 1.0;

    return I;
  }


  inline double
  clamp_value(const double x,
              const double xmin,
              const double xmax)
  {
    return std::max(xmin, std::min(x, xmax));
  }


  inline double
  smooth_step_tanh(const double x,
                   const double x0,
                   const double delta)
  {
    const double delta_safe =
      std::max(delta, 1e-14);

    return 0.5 * (1.0 + std::tanh((x - x0) / delta_safe));
  }


  inline double
  smooth_step_tanh_derivative(const double x,
                             const double x0,
                             const double delta)
  {
    const double delta_safe = std::max(delta, 1e-14);
    const double z = (x - x0) / delta_safe;
    const double th = std::tanh(z);
    const double sech2 = 1.0 - th * th;
    return 0.5 * sech2 / delta_safe;
  }


  template <int dim>
  double
  average_value(const std::array<double, dim> &values)
  {
    double value = 0.0;

    for (unsigned int d = 0; d < dim; ++d)
      value += values[d];

    return value / static_cast<double>(dim);
  }


  template <int dim>
  double
  cell_extent_in_direction(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const Tensor<1, dim>                                &a,
    const double                                         eps)
  {
    double s_min = std::numeric_limits<double>::max();
    double s_max = -std::numeric_limits<double>::max();

    for (unsigned int v = 0; v < cell->n_vertices(); ++v)
    {
      const Point<dim> Xv = cell->vertex(v);

      double s = 0.0;
      for (unsigned int d = 0; d < dim; ++d)
        s += Xv[d] * a[d];

      s_min = std::min(s_min, s);
      s_max = std::max(s_max, s);
    }

    return std::max(s_max - s_min, eps);
  }


  template <int dim>
  double
  isotropic_mesh_size_from_jacobian(const Tensor<2, dim> &F,
                                    const double          h_ref_iso,
                                    const double          eps)
  {
    const double J =
      std::max(determinant(F), eps);

    return h_ref_iso * std::pow(J, 1.0 / static_cast<double>(dim));
  }


  template <int dim>
  double
  isotropic_mesh_size_derivative_from_jacobian(
    const Tensor<2, dim> &F,
    const Tensor<2, dim> &dF,
    const double          h_ref_iso,
    const double          eps)
  {
    const double h_current =
      isotropic_mesh_size_from_jacobian<dim>(F, h_ref_iso, eps);

    const Tensor<2, dim> F_inv =
      invert(F);

    const double dlnJ =
      trace(F_inv * dF);

    return h_current * dlnJ / static_cast<double>(dim);
  }


  template <int dim>
  Tensor<2, dim>
  neo_hookean_first_piola(const Tensor<2, dim> &F,
                          const double          mu,
                          const double          lambda,
                          const double          eps)
  {
    const double J =
      std::max(determinant(F), eps);

    const Tensor<2, dim> F_inv_T =
      transpose(invert(F));

    return mu * (F - F_inv_T)
           + lambda * std::log(J) * F_inv_T;
  }


  template <int dim>
  Tensor<2, dim>
  neo_hookean_first_piola_derivative(const Tensor<2, dim> &F,
                                     const Tensor<2, dim> &dF,
                                     const double          mu,
                                     const double          lambda,
                                     const double          eps)
  {
    const double J =
      std::max(determinant(F), eps);

    const Tensor<2, dim> F_inv =
      invert(F);

    const Tensor<2, dim> F_inv_T =
      transpose(F_inv);

    const Tensor<2, dim> F_inv_T_dF_T_F_inv_T =
      F_inv_T * transpose(dF) * F_inv_T;

    const double dlnJ =
      trace(F_inv * dF);

    return mu * dF
           + lambda * dlnJ * F_inv_T
           + (mu - lambda * std::log(J)) * F_inv_T_dF_T_F_inv_T;
  }


  template <int dim>
  Tensor<1, dim>
  gradient_abs_velocity_from_recovered_gradient(
    const Tensor<1, dim> &u,
    const Tensor<2, dim> &recovered_grad_u,
    const double          eps)
  {
    const double u_norm =
      std::sqrt(u * u + eps * eps);

    Tensor<1, dim> grad_abs_u;

    /*
     * grad(|u|)_a = sum_c u_c / |u| * d u_c / d x_a
     *
     * Attention : on conserve ta convention existante.
     */
    for (unsigned int a = 0; a < dim; ++a)
      for (unsigned int c = 0; c < dim; ++c)
        grad_abs_u[a] += u[c] / u_norm * recovered_grad_u[c][a];

    return grad_abs_u;
  }


  template <int dim>
  Tensor<1, dim>
  continuous_gradient_abs_velocity_value(
    const LA::ParVectorType                    &grad_abs_velocity_field,
    const std::vector<types::global_dof_index> &local_dof_indices,
    const std::vector<Tensor<1, dim>>          &phi_u_q,
    const FiniteElement<dim>                   &fe,
    const unsigned int                          u_lower)
  {
    Tensor<1, dim> grad_abs_u;

    for (unsigned int local_dof = 0; local_dof < fe.n_dofs_per_cell();
         ++local_dof)
    {
      const auto component_shape =
        fe.system_to_component_index(local_dof);

      const unsigned int component =
        component_shape.first;

      if (component < u_lower || component >= u_lower + dim)
        continue;

      const unsigned int a =
        component - u_lower;

      grad_abs_u[a] +=
        grad_abs_velocity_field[local_dof_indices[local_dof]]
        * phi_u_q[local_dof][a];
    }

    return grad_abs_u;
  }


  template <int dim>
  double
  gradient_abs_velocity_weight(const Tensor<1, dim> &grad_abs_u,
                              const double          grad_min,
                              const double          /*grad_ref*/,
                              const double          grad_max,
                              const double          /*exponent*/,
                              const double          eps)
  {
    const double g =
      grad_abs_u.norm();

    const double g_min_safe =
      grad_min;

    const double g_max_safe =
      std::max(grad_max, grad_min + eps);

    const double g0 =
      0.5 * (g_min_safe + g_max_safe);

    const double atanh_09 =
      std::atanh(0.9);

    const double delta_g =
      (g_max_safe - g_min_safe) / (2.0 * atanh_09);

    return smooth_step_tanh(g, g0, delta_g);
  }


  inline double
  target_size_from_weight(const double w_raw,
                          const double h_background,
                          const double h_min)
  {
    if (h_background <= h_min)
      return h_background;

    const double w =
      clamp_value(w_raw, 0.0, 1.0);

    const double h_target =
      h_background + w * (h_min - h_background);

    return clamp_value(h_target, h_min, h_background);
  }


  template <int dim>
  double
  target_size_from_gradient_abs_velocity(
    const Tensor<1, dim> &grad_abs_u,
    const double          h_background,
    const double          h_min,
    const double          grad_min,
    const double          grad_ref,
    const double          grad_max,
    const double          exponent,
    const double          eps)
  {
    if (h_background <= h_min)
      return h_background;

    const double w =
      gradient_abs_velocity_weight<dim>(grad_abs_u,
                                        grad_min,
                                        grad_ref,
                                        grad_max,
                                        exponent,
                                        eps);

    return target_size_from_weight(w,
                                   h_background,
                                   h_min);
  }


  template <int dim>
  Tensor<1, dim>
  target_size_from_gradient_abs_velocity_derivative(
    const Tensor<1, dim> &grad_abs_u,
    const double          h_background,
    const double          h_min,
    const double          grad_min,
    const double          grad_ref,
    const double          grad_max,
    const double          exponent,
    const double          eps)
  {
    Tensor<1, dim> dh_dg_vec;

    if (h_background <= h_min)
      return dh_dg_vec; // zeros

    const double g = grad_abs_u.norm();
    const double g_safe = std::max(g, eps);

    const double g_min_safe = grad_min;
    const double g_max_safe = std::max(grad_max, grad_min + eps);
    const double g0 = 0.5 * (g_min_safe + g_max_safe);
    const double atanh_09 = std::atanh(0.9);
    const double delta_g = (g_max_safe - g_min_safe) / (2.0 * atanh_09);

    const double dw_dg = smooth_step_tanh_derivative(g, g0, delta_g);

    // dh/dw = h_min - h_background inside (0,1), else 0 after clamping
    const double dh_dw = (h_min - h_background);

    // dh/dg = dh/dw * dw/dg * grad_abs_u / g
    const double pref = (g_safe > 0.0) ? (dh_dw * dw_dg / g_safe) : 0.0;
    for (unsigned int a = 0; a < dim; ++a)
      dh_dg_vec[a] = pref * grad_abs_u[a];

    return dh_dg_vec;
  }


  // New smooth isotropic maintained + correction pressure law
  inline double
  maintained_size_pressure_law(const double h_current,
                               const double h_background,
                               const double h_target,
                               const double stiffness,
                               const double alpha,
                               const double /*release_ratio*/,
                               const double /*transition_width*/,
                               const double eps)
  {
    const double eta = 0.1; // small correction coefficient

    const double h_background_safe = std::max(h_background, eps);
    const double h_target_safe     = std::max(h_target, eps);
    const double h_current_safe    = std::max(h_current, eps);

    const double L_maintain = std::log(h_background_safe / h_target_safe);
    const double L_correct  = std::log(h_current_safe / h_target_safe);

    return alpha * stiffness * (L_maintain + eta * L_correct);
  }


  inline double
  maintained_size_pressure_derivative_wrt_h_current(
    const double h_current,
    const double /*h_background*/,
    const double h_target,
    const double stiffness,
    const double alpha,
    const double /*release_ratio*/,
    const double /*transition_width*/,
    const double eps)
  {
    const double eta = 0.1;

    const double h_current_safe = std::max(h_current, eps);

    return alpha * stiffness * eta / h_current_safe;
  }


  inline double
  maintained_size_pressure_derivative_wrt_h_target(
    const double h_current,
    const double /*h_background*/,
    const double h_target,
    const double stiffness,
    const double alpha,
    const double /*release_ratio*/,
    const double /*transition_width*/,
    const double eps)
  {
    const double eta = 0.1;

    const double h_target_safe = std::max(h_target, eps);

    // derivative of - (1+eta) * log(h_target)
    return -alpha * stiffness * (1.0 + eta) / h_target_safe;
  }


  inline double
  maintained_size_pressure_derivative(
    const double h_current,
    const double dh_current,
    const double h_background,
    const double h_target,
    const double dh_target,
    const double stiffness,
    const double alpha,
    const double /*release_ratio*/,
    const double /*transition_width*/,
    const double eps)
  {
    const double dp_dh_current =
      maintained_size_pressure_derivative_wrt_h_current(
        h_current,
        h_background,
        h_target,
        stiffness,
        alpha,
        0.0,
        0.0,
        eps);

    const double dp_dh_target =
      maintained_size_pressure_derivative_wrt_h_target(
        h_current,
        h_background,
        h_target,
        stiffness,
        alpha,
        0.0,
        0.0,
        eps);

    return dp_dh_current * dh_current + dp_dh_target * dh_target;
  }


  inline double
  maintained_size_pressure_derivative_wrt_alpha(
    const double h_current,
    const double h_background,
    const double h_target,
    const double stiffness,
    const double /*release_ratio*/,
    const double /*transition_width*/,
    const double eps)
  {
    const double eta = 0.1;

    const double h_background_safe = std::max(h_background, eps);
    const double h_target_safe     = std::max(h_target, eps);
    const double h_current_safe    = std::max(h_current, eps);

    const double L_maintain = std::log(h_background_safe / h_target_safe);
    const double L_correct  = std::log(h_current_safe / h_target_safe);

    return stiffness * (L_maintain + eta * L_correct);
  }


  template <int dim>
  Tensor<2, dim>
  isotropic_mesh_concentration_piola(
    const Tensor<2, dim>          &F,
    const Tensor<2, dim>          &F_inv_T,
    const double                   J,
    const Tensor<1, dim>          &grad_abs_u,
    const std::array<double, dim> &h_ref_dir,
    const std::array<double, dim> &h_target_background_dir,
    const std::array<double, dim> &h_min_dir,
    const double                   size_stiffness,
    const double                   alpha,
    const double                   ramp_factor,
    const double                   velocity_gradient_min,
    const double                   velocity_gradient_ref,
    const double                   velocity_gradient_max,
    const double                   velocity_gradient_exponent,
    const double                   release_ratio,
    const double                   transition_width,
    const double                   max_pressure,
    const double                   eps)
  {
    const Tensor<2, dim> I =
      identity_tensor<dim>();

    const double h_ref_iso =
      average_value<dim>(h_ref_dir);

    const double h_background_iso =
      average_value<dim>(h_target_background_dir);

    const double h_min_iso =
      average_value<dim>(h_min_dir);

    const double h_current_iso =
      isotropic_mesh_size_from_jacobian<dim>(
        F,
        h_ref_iso,
        eps);

    const double h_target_iso =
      target_size_from_gradient_abs_velocity<dim>(
        grad_abs_u,
        h_background_iso,
        h_min_iso,
        velocity_gradient_min,
        velocity_gradient_ref,
        velocity_gradient_max,
        velocity_gradient_exponent,
        eps);

    (void)release_ratio;
    (void)transition_width;
    (void)max_pressure;

    const double p_raw =
      maintained_size_pressure_law(
        h_current_iso,
        h_background_iso,
        h_target_iso,
        size_stiffness,
        alpha,
        release_ratio,
        transition_width,
        eps);

    const double p_iso = ramp_factor * p_raw;

    const Tensor<2, dim> sigma_size_spatial =
      p_iso * I;

    return J * sigma_size_spatial * F_inv_T;
  }


  template <int dim>
  Tensor<2, dim>
  isotropic_mesh_concentration_piola_derivative(
    const Tensor<2, dim>          &F,
    const Tensor<2, dim>          &F_inv_T,
    const Tensor<2, dim>          &dF,
    const Tensor<2, dim>          &dF_inv_T,
    const double                   J,
    const double                   dJ,
    const Tensor<1, dim>          &grad_abs_u,
    const std::array<double, dim> &h_ref_dir,
    const std::array<double, dim> &h_target_background_dir,
    const std::array<double, dim> &h_min_dir,
    const double                   size_stiffness,
    const double                   alpha,
    const double                   d_alpha,
    const double                   ramp_factor,
    const double                   velocity_gradient_min,
    const double                   velocity_gradient_ref,
    const double                   velocity_gradient_max,
    const double                   velocity_gradient_exponent,
    const double                   release_ratio,
    const double                   transition_width,
    const double                   max_pressure,
    const double                   eps,
    const bool                     derivative_wrt_mesh)
  {
    const Tensor<2, dim> I =
      identity_tensor<dim>();

    const double h_ref_iso =
      average_value<dim>(h_ref_dir);

    const double h_background_iso =
      average_value<dim>(h_target_background_dir);

    const double h_min_iso =
      average_value<dim>(h_min_dir);

    const double h_current_iso =
      isotropic_mesh_size_from_jacobian<dim>(
        F,
        h_ref_iso,
        eps);

    const double h_target_iso =
      target_size_from_gradient_abs_velocity<dim>(
        grad_abs_u,
        h_background_iso,
        h_min_iso,
        velocity_gradient_min,
        velocity_gradient_ref,
        velocity_gradient_max,
        velocity_gradient_exponent,
        eps);

    (void)release_ratio;
    (void)transition_width;
    (void)max_pressure;

    const double p_raw =
      maintained_size_pressure_law(
        h_current_iso,
        h_background_iso,
        h_target_iso,
        size_stiffness,
        alpha,
        release_ratio,
        transition_width,
        eps);

    const double p_iso = ramp_factor * p_raw;

    const Tensor<2, dim> sigma_size_spatial =
      p_iso * I;


    double dh_current = 0.0;

    if (derivative_wrt_mesh)
      dh_current =
        isotropic_mesh_size_derivative_from_jacobian<dim>(
          F,
          dF,
          h_ref_iso,
          eps);

    /*
     * h_target is frozen in the Newton tangent (dh_target = 0)
     */
    const double dh_target = 0.0;

    const double dp_size =
      derivative_wrt_mesh
        ? maintained_size_pressure_derivative(
            h_current_iso,
            dh_current,
            h_background_iso,
            h_target_iso,
            dh_target,
            size_stiffness,
            alpha,
            release_ratio,
            transition_width,
            eps)
        : 0.0;

    const double dp_alpha =
      maintained_size_pressure_derivative_wrt_alpha(
        h_current_iso,
        h_background_iso,
        h_target_iso,
        size_stiffness,
        release_ratio,
        transition_width,
        eps)
      * d_alpha;

    const double dp = ramp_factor * (dp_size + dp_alpha);

    const Tensor<2, dim> dsigma_size_spatial = dp * I;

    if (!derivative_wrt_mesh)
      return Tensor<2, dim>();

    return dJ * sigma_size_spatial * F_inv_T
           + J * dsigma_size_spatial * F_inv_T
           + J * sigma_size_spatial * dF_inv_T;
  }


  template <int dim>
  double
  isotropic_h_current_cell_value(
    const Tensor<2, dim>          &F,
    const std::array<double, dim> &h_ref_dir,
    const double                   eps)
  {
    const double h_ref_iso =
      average_value<dim>(h_ref_dir);

    return isotropic_mesh_size_from_jacobian<dim>(F, h_ref_iso, eps);
  }


  template <int dim>
  double
  isotropic_h_target_cell_value(
    const Tensor<1, dim>          &grad_abs_u,
    const std::array<double, dim> &h_target_background_dir,
    const std::array<double, dim> &h_min_dir,
    const double                   velocity_gradient_min,
    const double                   velocity_gradient_ref,
    const double                   velocity_gradient_max,
    const double                   velocity_gradient_exponent,
    const double                   eps)
  {
    const double h_background_iso =
      average_value<dim>(h_target_background_dir);

    const double h_min_iso =
      average_value<dim>(h_min_dir);

    return target_size_from_gradient_abs_velocity<dim>(
      grad_abs_u,
      h_background_iso,
      h_min_iso,
      velocity_gradient_min,
      velocity_gradient_ref,
      velocity_gradient_max,
      velocity_gradient_exponent,
      eps);
  }


  template <int dim>
  double
  isotropic_pressure_cell_value(
    const Tensor<2, dim>          &F,
    const Tensor<1, dim>          &grad_abs_u,
    const std::array<double, dim> &h_ref_dir,
    const std::array<double, dim> &h_target_background_dir,
    const std::array<double, dim> &h_min_dir,
    const double                   size_stiffness,
    const double                   alpha,
    const double                   ramp_factor,
    const double                   velocity_gradient_min,
    const double                   velocity_gradient_ref,
    const double                   velocity_gradient_max,
    const double                   velocity_gradient_exponent,
    const double                   release_ratio,
    const double                   transition_width,
    const double                   max_pressure,
    const double                   eps)
  {
    const double h_background_iso =
      average_value<dim>(h_target_background_dir);

    const double h_current_iso =
      isotropic_h_current_cell_value<dim>(
        F,
        h_ref_dir,
        eps);

    const double h_target_iso =
      isotropic_h_target_cell_value<dim>(
        grad_abs_u,
        h_target_background_dir,
        h_min_dir,
        velocity_gradient_min,
        velocity_gradient_ref,
        velocity_gradient_max,
        velocity_gradient_exponent,
        eps);

    (void) release_ratio;
    (void) transition_width;
    (void) max_pressure;

    const double p_raw =
      maintained_size_pressure_law(
        h_current_iso,
        h_background_iso,
        h_target_iso,
        size_stiffness,
        alpha,
        release_ratio,
        transition_width,
        eps);

    return p_raw;
  }
}

template <int dim>
FSISolver<dim>::FSISolver(const ParameterReader<dim> &param)
  : NavierStokesSolver<dim, true>(param)
  , all_lambda_accumulators(dim)
{
  if (param.finite_elements.use_quads)
    fe = std::make_unique<FESystem<dim>>(
      FE_Q<dim>(param.finite_elements.velocity_degree) ^ dim,      // Velocity
      FE_Q<dim>(param.finite_elements.pressure_degree),            // Pressure
      FE_Q<dim>(param.finite_elements.mesh_position_degree) ^ dim, // Position
      FE_Q<dim>(param.finite_elements.no_slip_lagrange_mult_degree) ^
        dim); // Lagrange multiplier
  else
    fe = std::make_unique<FESystem<dim>>(
      FE_SimplexP<dim>(param.finite_elements.velocity_degree) ^ dim, // Velocity
      FE_SimplexP<dim>(param.finite_elements.pressure_degree),       // Pressure
      FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^
        dim, // Position
      FE_SimplexP<dim>(param.finite_elements.no_slip_lagrange_mult_degree) ^
        dim); // Lagrange multiplier

  this->ordering = std::make_unique<ComponentOrderingFSI<dim>>();

  this->velocity_extractor =
    FEValuesExtractors::Vector(this->ordering->u_lower);
  this->pressure_extractor =
    FEValuesExtractors::Scalar(this->ordering->p_lower);
  this->position_extractor =
    FEValuesExtractors::Vector(this->ordering->x_lower);
  this->lambda_extractor = FEValuesExtractors::Vector(this->ordering->l_lower);

  this->velocity_mask = fe->component_mask(this->velocity_extractor);
  this->pressure_mask = fe->component_mask(this->pressure_extractor);
  this->position_mask = fe->component_mask(this->position_extractor);
  this->lambda_mask   = fe->component_mask(this->lambda_extractor);

  this->field_names_and_masks["velocity"]      = this->velocity_mask;
  this->field_names_and_masks["pressure"]      = this->pressure_mask;
  this->field_names_and_masks["mesh position"] = this->position_mask;

  // Set the boundary id on which a weak no slip boundary condition is applied.
  // It is allowed *not* to prescribe a weak no slip on any boundary, to verify
  // that the solver produces the expected flow in the decoupled case.
  unsigned int n_weak_bc = 0;
  for (const auto &[id, bc] : param.fluid_bc)
    if (bc.type == BoundaryConditions::Type::weak_no_slip)
    {
      weak_no_slip_boundary_id = bc.id;
      n_weak_bc++;

      for (const auto &[id, bc] : param.pseudosolid_bc)
        if (bc.type == BoundaryConditions::Type::coupled_to_fluid)
          AssertThrow(
            bc.id == weak_no_slip_boundary_id,
            ExcMessage(
              "A pseudosolid boundary condition was set to "
              "\"coupled_to_fluid\" on boundary \"" +
              bc.gmsh_name +
              "\", but the fluid boundary condition on this boundary was not "
              "set to \"weak_no_slip\". For now, fluid-structure coupling can "
              "only be done through a Lagrange multiplier, which requires "
              "weakly enforced no-slip condition on the coupled boundary."));
    }
  AssertThrow(n_weak_bc <= 1,
              ExcMessage(
                "A weakly enforced no-slip boundary condition is enforced on "
                "more than 1 boundary, which is currently not supported."));

  /**
   * Enforcing zero-mean pressure on moving mesh is not trivial, since
   * the constraint weights depend on the mesh position.
   */
  AssertThrow(!param.bc_data.enforce_zero_mean_pressure,
              ExcMessage("Enforcing zero mean pressure on moving mesh is "
                         "currently not implemented."));

  /**
   * While different coupling schemes are still being tested, keep the debug
   * flag to change the scheme at runtime, but do not allow using the first,
   * inefficient coupling.
   */
  if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
    this->pcout << "Using coupling scheme : "
                << static_cast<unsigned int>(this->param.fsi.coupling)
                << std::endl;
  // AssertThrow(this->param.debug.fsi_coupling_option != 0,
  //             ExcMessage(
  //               "This parameter file still uses the inefficient coupling "
  //               "scheme used for prototyping. Use a better coupling by
  //               setting " "fsi_coupling_option = 1 in the Debug
  //               subsection."));

  // Create the initial condition functions for this problem, once the layout of
  // the variables is known (and in particular, the number of components).
  // FIXME: Is there a better way to create the functions?
  this->param.initial_conditions.create_initial_velocity(
    this->ordering->u_lower, this->ordering->n_components);

  if (param.mms_param.enable)
  {
    // Assign the manufactured solution
    this->exact_solution = std::make_shared<FSISolver<dim>::MMSSolution>(
      this->time_handler.current_time, *this->ordering, param.mms);

    // Create the source term function for the given MMS and override source
    // terms
    this->source_terms = std::make_shared<FSISolver<dim>::MMSSourceTerm>(
      this->time_handler.current_time,
      *this->ordering,
      param.physical_properties,
      param.mms);

    // Create entry in error handler for Lagrange multiplier
    for (auto &[norm, handler] : this->error_handlers)
    {
      handler.create_entry("l");
      if (this->param.fsi.compute_error_on_forces)
        for (unsigned int d = 0; d < dim; ++d)
          handler.create_entry("F_comp" + std::to_string(d));
    }
  }
  else
  {
    this->source_terms = std::make_shared<FSISolver<dim>::SourceTerm>(
      this->time_handler.current_time, *this->ordering, param.source_terms);
    this->exact_solution = std::make_shared<Functions::ZeroFunction<dim>>(
      this->ordering->n_components);
  }
}

template <int dim>
FSISolver<dim>::~FSISolver()
{}

template <int dim>
void FSISolver<dim>::create_scratch_data()
{
  scratch_data = std::make_unique<ScratchData>(*this->ordering,
                                               *fe,
                                               *this->fixed_mapping,
                                               *this->moving_mapping,
                                               *this->quadrature,
                                               *this->face_quadrature,
                                               this->time_handler,
                                               this->param);
}

template <int dim>
void FSISolver<dim>::MMSSourceTerm::vector_value(const Point<dim> &p,
                                                 Vector<double>   &values) const
{
  const double nu = physical_properties.fluids[0].kinematic_viscosity;

  Tensor<1, dim> u, dudt_eulerian;
  for (unsigned int d = 0; d < dim; ++d)
  {
    dudt_eulerian[d] = mms.exact_velocity->time_derivative(p, d);
    u[d]             = mms.exact_velocity->value(p, d);
  }

  // Use convention (grad_u)_ij := dvj/dxi
  Tensor<2, dim> grad_u     = mms.exact_velocity->gradient_vj_xi(p);
  Tensor<1, dim> lap_u      = mms.exact_velocity->vector_laplacian(p);
  Tensor<1, dim> grad_div_u = mms.exact_velocity->grad_div(p);
  Tensor<1, dim> grad_p     = mms.exact_pressure->gradient(p);
  Tensor<1, dim> uDotGradu  = u * grad_u;

  // Velocity source term
  Tensor<1, dim> f =
    -(dudt_eulerian + uDotGradu + grad_p - nu * (lap_u + grad_div_u));
  for (unsigned int d = 0; d < dim; ++d)
    values[ordering.u_lower + d] = f[d];

  // Mass conservation (pressure) source term
  values[ordering.p_lower] = mms.exact_velocity->divergence(p);

  // Pseudosolid (mesh position) source term
  // We solve -div(sigma) + f = 0, so no need to put a -1 in front of f
  Tensor<1, dim> f_PS =
    mms.exact_mesh_position
      ->divergence_linear_elastic_stress_variable_coefficients(
        p,
        physical_properties.pseudosolids[0].lame_mu_fun,
        physical_properties.pseudosolids[0].lame_lambda_fun);

  for (unsigned int d = 0; d < dim; ++d)
    values[ordering.x_lower + d] = f_PS[d];

  // Lagrange multiplier source term (none)
  for (unsigned int d = 0; d < dim; ++d)
    values[ordering.l_lower + d] = 0.;
}

template <int dim>
void FSISolver<dim>::reset_solver_specific_data()
{
  // Position - lambda constraints
  for (auto &vec : lambda_integral_coeffs)
    vec.clear();
  lambda_integral_coeffs.clear();
  coupled_position_dofs.clear();
  has_local_position_master       = false;
  has_local_lambda_accumulator    = false;
  has_global_master_position_dofs = false;
  has_global_accumulator          = false;
  for (unsigned int d = 0; d < dim; ++d)
  {
    local_position_master_dofs[d]  = numbers::invalid_unsigned_int;
    global_position_master_dofs[d] = numbers::invalid_unsigned_int;
    local_lambda_accumulators[d]   = numbers::invalid_unsigned_int;
    global_lambda_accumulators[d]  = numbers::invalid_unsigned_int;
    all_lambda_accumulators[d].clear();
  }

  mesh_concentration_grad_abs_velocity.clear();
}


template <int dim>
void FSISolver<dim>::initialize_mesh_concentration()
{
  if (!this->param.mesh_concentration.enable)
    return;

  mesh_concentration_grad_abs_velocity.reinit(this->locally_owned_dofs,
                                              this->locally_relevant_dofs,
                                              this->mpi_communicator);
}


template <int dim>
void FSISolver<dim>::update_mesh_concentration_field()
{
  if (!this->param.mesh_concentration.enable)
    return;

  initialize_mesh_concentration();

  this->evaluation_point.update_ghost_values();

  LA::ParVectorType local_grad_abs_u;
  local_grad_abs_u.reinit(this->locally_owned_dofs,
                          this->mpi_communicator);
  local_grad_abs_u = 0.0;

  LA::ParVectorType local_weights;
  local_weights.reinit(this->locally_owned_dofs,
                       this->mpi_communicator);
  local_weights = 0.0;

  const auto velocity_local_dof =
    PostProcessingTools::build_local_component_shape_to_dof_table<dim>(
      this->dof_handler.get_fe(),
      this->velocity_extractor);

  std::vector<types::global_dof_index> local_dof_indices(
    this->dof_handler.get_fe().n_dofs_per_cell());

  FEValues<dim> fe_values(*this->moving_mapping,
                          this->dof_handler.get_fe(),
                          *this->quadrature,
                          update_values | update_gradients |
                            update_JxW_values);

  std::vector<Tensor<1, dim>> velocity_values(this->quadrature->size());
  std::vector<Tensor<2, dim>> velocity_gradients(this->quadrature->size());

  const double eps =
    this->param.mesh_concentration.eps;

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    cell->get_dof_indices(local_dof_indices);

    fe_values.reinit(cell);
    fe_values[this->velocity_extractor].get_function_values(
      this->evaluation_point,
      velocity_values);
    fe_values[this->velocity_extractor].get_function_gradients(
      this->evaluation_point,
      velocity_gradients);

    Tensor<1, dim> cell_grad_abs_u;
    double         cell_measure = 0.0;

    for (unsigned int q = 0; q < this->quadrature->size(); ++q)
    {
      cell_grad_abs_u +=
        MeshConcentrationTools::gradient_abs_velocity_from_recovered_gradient<dim>(
          velocity_values[q],
          velocity_gradients[q],
          eps)
        * fe_values.JxW(q);

      cell_measure += fe_values.JxW(q);
    }

    if (cell_measure > eps)
      cell_grad_abs_u /= cell_measure;

    const unsigned int n_scalar_dofs =
      velocity_local_dof[0].size();

    for (unsigned int shape = 0; shape < n_scalar_dofs; ++shape)
    {
      for (unsigned int a = 0; a < dim; ++a)
      {
        const unsigned int local_output_dof =
          velocity_local_dof[a][shape];

        const types::global_dof_index output_dof =
          local_dof_indices[local_output_dof];

        if (this->locally_owned_dofs.is_element(output_dof))
        {
          local_grad_abs_u[output_dof] += cell_grad_abs_u[a];
          local_weights[output_dof] += 1.0;
        }
      }
    }
  }

  local_grad_abs_u.compress(VectorOperation::add);
  local_weights.compress(VectorOperation::add);

  for (const auto &dof : this->locally_owned_dofs)
    if (local_weights[dof] > 0.0)
      local_grad_abs_u[dof] /= local_weights[dof];

  local_grad_abs_u.compress(VectorOperation::insert);
  mesh_concentration_grad_abs_velocity = local_grad_abs_u;
  mesh_concentration_grad_abs_velocity.update_ghost_values();
}

template <int dim>
void FSISolver<dim>::create_lagrange_multiplier_constraints()
{
  lambda_constraints.reinit(this->locally_owned_dofs,
                            this->locally_relevant_dofs);

  // If there is no weakly enforced no slip boundary, this set remains empty and
  // all lambda dofs are constrained.
  IndexSet relevant_boundary_dofs;

  if (weak_no_slip_boundary_id != numbers::invalid_unsigned_int)
  {
    relevant_boundary_dofs =
      DoFTools::extract_boundary_dofs(this->dof_handler,
                                      lambda_mask,
                                      {weak_no_slip_boundary_id});
  }

  const bool requires_local_lambda_accumulator =
    (this->param.fsi.coupling ==
     Coupling::local_position_master_to_lambda_accumulators) ||
    (this->param.fsi.coupling ==
     Coupling::global_position_master_to_global_accumulator);

  // There does not seem to be a 2-3 liner way to extract the locally
  // relevant dofs on a boundary for a given component (extract_dofs
  // returns owned dofs).
  std::vector<types::global_dof_index> local_dofs(fe->n_dofs_per_cell());
  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    if (!(cell->is_locally_owned() || cell->is_ghost()))
      continue;
    cell->get_dof_indices(local_dofs);
    for (unsigned int i = 0; i < local_dofs.size(); ++i)
    {
      types::global_dof_index dof = local_dofs[i];

      // If using the coupling method with accumulators and if dof is a local
      // accumulator, do not constrain it
      bool skip_dof = false;
      if (requires_local_lambda_accumulator)
        for (unsigned int d = 0; d < dim; ++d)
          if (local_lambda_accumulators[d] == dof)
          {
            skip_dof = true;
            break;
          }

      if (!skip_dof)
      {
        unsigned int comp = fe->system_to_component_index(i).first;
        if (this->ordering->is_lambda(comp))
          if (this->locally_relevant_dofs.is_element(dof))
            if (!relevant_boundary_dofs.is_element(dof))
              lambda_constraints.constrain_dof_to_zero(dof);
      }
    }
  }
  lambda_constraints.close();

  if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
  {
    // Print number of owned and constrained lambda dofs
    IndexSet lambda_dofs =
      DoFTools::extract_dofs(this->dof_handler, lambda_mask);
    unsigned int constrained_owned_dofs   = 0;
    unsigned int unconstrained_owned_dofs = 0;
    for (const auto &dof : lambda_dofs)
    {
      if (!lambda_constraints.is_constrained(dof))
        unconstrained_owned_dofs++;
      else
        constrained_owned_dofs++;
    }

    const unsigned int total_constrained_owned_dofs =
      Utilities::MPI::sum(constrained_owned_dofs, this->mpi_communicator);
    this->pcout << total_constrained_owned_dofs
                << " constrained owned lambda dofs" << std::endl;
    const unsigned int total_unconstrained_owned_dofs =
      Utilities::MPI::sum(unconstrained_owned_dofs, this->mpi_communicator);
    this->pcout << total_unconstrained_owned_dofs
                << " unconstrained owned lambda dofs" << std::endl;
  }
}

/**
 * On the cylinder, we have
 *
 * x = X - int_Gamma lambda dx,
 *
 * yielding the affine constraints
 *
 * x_i = X_i + sum_j c_ij * lambda_j, with c_ij = - int_Gamma phi_global_j dx.
 *
 * Each position DoF is linked to all lambda DoF on the cylinder, which may
 * not be owned of even ghosts of the current process.
 *
 * This function does the following:
 *
 * - It computes the coefficients c_ij of the coupling x_i = X_i + c_ij *
 * lambda_j, which are the integral of the global shape functions associated to
 * lambda_j.
 *
 * - It creates the DOF pairings (x_i, vector of lambda_j), which specify to
 * which lambda DOFs a position DOF on the cylinder is constrained (all of them
 * actually).
 *
 *   FIXME: THERE IS ONLY ONE VECTOR ACTUALLY
 */
template <int dim>
void FSISolver<dim>::create_position_lagrange_mult_coupling_data()
{
  /**
   * Get the owned position dofs on the cylinder.
   * We might be missing some owned dofs, e.g., on boundary edges for which
   * no cell face touches the cylinder in 3D. Also add them here.
   */
  IndexSet local_position_dofs =
    DoFTools::extract_boundary_dofs(this->dof_handler,
                                    this->position_mask,
                                    {weak_no_slip_boundary_id});
  local_position_dofs = local_position_dofs & this->locally_owned_dofs;
  {
    std::vector<std::vector<types::global_dof_index>> gathered_pos_bdr_dofs =
      Utilities::MPI::all_gather(this->mpi_communicator,
                                 local_position_dofs.get_index_vector());
    for (const auto &vec : gathered_pos_bdr_dofs)
      for (const auto dof : vec)
        if (this->locally_owned_dofs.is_element(dof))
          local_position_dofs.add_index(dof);
  }

  const bool has_owned_position_dofs_on_boundary =
    local_position_dofs.n_elements() > 0;

  /**
   * Set up some flags depending on the coupling strategy.
   * In particular, we need to know whether:
   *
   * - additional ghost lambda dofs should be accounted for. This is the case
   *   if position (all or masters) dofs are coupled to *all* lambda dofs, in
   * which case they need all cylinder lambda dofs as ghosts to evaluate the
   * total force integral.
   *
   * - local and global position master dofs should be set, if the coupling
   * strategy uses position masters.
   *
   * - local and global lambda accumulators should be set, if the coupling
   * strategy uses force accumulators.
   */
  const auto coupling = this->param.fsi.coupling;

  const bool requires_lambda_ghosts =
    (coupling == Coupling::all_position_to_all_lambda) ||
    (coupling == Coupling::local_position_master_to_all_lambda) ||
    (coupling == Coupling::global_position_master_to_all_lambda);

  // All but the all-to-all strategy use a local position master
  const bool requires_local_position_master =
    (coupling != Coupling::all_position_to_all_lambda);

  const bool requires_global_position_master =
    (coupling == Coupling::global_position_master_to_all_lambda) ||
    (coupling == Coupling::global_position_master_to_global_accumulator);

  const bool requires_local_lambda_accumulator =
    (coupling == Coupling::local_position_master_to_lambda_accumulators) ||
    (coupling == Coupling::global_position_master_to_global_accumulator);

  const bool requires_global_lambda_accumulator =
    (coupling == Coupling::global_position_master_to_global_accumulator);

  if (requires_global_position_master)
    Assert(requires_local_position_master, ExcInternalError());
  if (requires_global_lambda_accumulator)
    Assert(requires_local_lambda_accumulator, ExcInternalError());

  if (requires_lambda_ghosts)
  {
    // Collect the (relevant) lambda dofs
    IndexSet boundary_lambda_dofs =
      DoFTools::extract_boundary_dofs(this->dof_handler,
                                      this->lambda_mask,
                                      {weak_no_slip_boundary_id});

    std::vector<std::vector<types::global_dof_index>> gathered =
      Utilities::MPI::all_gather(this->mpi_communicator,
                                 boundary_lambda_dofs.get_index_vector());

    std::vector<types::global_dof_index> all_boundary_lambda_dofs;
    for (const auto &vec : gathered)
      all_boundary_lambda_dofs.insert(all_boundary_lambda_dofs.end(),
                                      vec.begin(),
                                      vec.end());

    if (has_owned_position_dofs_on_boundary)
    {
      this->locally_relevant_dofs.add_indices(all_boundary_lambda_dofs.begin(),
                                              all_boundary_lambda_dofs.end());
      this->locally_relevant_dofs.compress();

      // (Re-)create the dofs_to_component map and specify that
      // the added non-local dofs are lambda dofs
      fill_dofs_to_component(this->dof_handler,
                             this->locally_relevant_dofs,
                             this->dofs_to_component);
      AssertDimension(this->dofs_to_component.size(),
                      this->locally_relevant_dofs.n_elements());
      // FIXME: all the added lambda dofs are added as "l_lower", i.e., the
      // first lambda component. They should be added with their proper
      // component...
      for (const auto dof : all_boundary_lambda_dofs)
        this->dofs_to_component[this->locally_relevant_dofs.index_within_set(
          dof)] = this->ordering->l_lower;
    }

    // Reinitialize the ghosted parallel vectors with the additional ghosts.
    this->reinit_ghosted_vectors();
  }

  /**
   * Set up the local and global position master dofs.
   */
  if (requires_local_position_master)
  {
    // Set the local_position_master_dofs
    // Simply take the first owned position dofs on the cylinder
    // Here it's assumed that local_position_dofs is organized as
    // x_0, y_0, z_0, x_1, y_1, z_1, ...,
    // and we take the first dim.
    const auto pos_index_vector = local_position_dofs.get_index_vector();
    if (pos_index_vector.size() > 0)
    {
      has_local_position_master = true;
      AssertThrow(pos_index_vector.size() >= dim,
                  ExcMessage(
                    "This partition has position dofs on the cylinder, but has "
                    "less than dim position dofs, which should not happen. It "
                    "should have n * dim position dofs on this boundary."));
      for (unsigned int d = 0; d < dim; ++d)
      {
        local_position_master_dofs[d] = pos_index_vector[d];
        AssertThrow(this->locally_owned_dofs.is_element(pos_index_vector[d]),
                    ExcMessage("Local position master dof " +
                               std::to_string(pos_index_vector[d]) +
                               " is not owned. This should not happen!"));
      }
    }

    if constexpr (running_in_debug_mode())
    {
      if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
      {
        n_ranks_with_position_master =
          Utilities::MPI::sum(has_local_position_master ? 1 : 0,
                              this->mpi_communicator);
        this->pcout << "There are " << n_ranks_with_position_master
                    << " ranks with local position master dofs" << std::endl;
      }
    }

    /**
     * Set up the global position master as the local master on the lowest rank
     *  among those with a local position master.
     */
    if (requires_global_position_master)
    {
      const unsigned int candidate_rank =
        has_local_position_master ? this->mpi_rank :
                                    std::numeric_limits<unsigned int>::max();
      const unsigned int owner_rank =
        Utilities::MPI::min(candidate_rank, this->mpi_communicator);
      has_global_master_position_dofs = (this->mpi_rank == owner_rank);

      if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
        this->pcout << "Global position master is on rank " << owner_rank
                    << std::endl;

      // Set the global position dofs and broadcast them to all ranks
      for (unsigned int d = 0; d < dim; ++d)
      {
        global_position_master_dofs[d] = numbers::invalid_unsigned_int;
        if (has_global_master_position_dofs)
          global_position_master_dofs[d] = local_position_master_dofs[d];
      }

      Utilities::MPI::broadcast(global_position_master_dofs.data(),
                                dim,
                                owner_rank,
                                this->mpi_communicator);

      if constexpr (running_in_debug_mode())
      {
        for (unsigned int d = 0; d < dim; ++d)
          Assert(global_position_master_dofs[d] !=
                   numbers::invalid_unsigned_int,
                 ExcMessage(
                   "The global position master is invalid after broadcast"));
      }
    }
  }

  /**
   * Set up local and global lambda accumulators
   */
  if (requires_local_lambda_accumulator)
  {
    // Normally, this coupling would require adding "dim" dofs per
    // partition to store the integral of each component (accumulators).
    // But we can ruse a little bit: since we are already storing more lambda
    // dofs than required (even in hp mode), we can just use "dim" of these
    // useless dofs to store the force on this proc,
    // while being careful not to affect the no-slip constraint.
    // The global dof indices of these dofs are stored in
    // local_lambda_accumulators.

    // Set the accumulator dofs from among the unused lambda dofs:
    // This rank should have a lambda accumulator if it has at least
    // one owned face on the cylinder
    for (const auto &cell : this->dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
        if (cell->at_boundary())
          for (const auto &face : cell->face_iterators())
            if (face->at_boundary() &&
                face->boundary_id() == weak_no_slip_boundary_id)
            {
              has_local_lambda_accumulator = true;
              goto reduce_accumulators;
            }
  reduce_accumulators:
    n_ranks_with_lambda_accumulator =
      Utilities::MPI::sum(has_local_lambda_accumulator ? 1 : 0,
                          this->mpi_communicator);

    if constexpr (running_in_debug_mode())
    {
      if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
        this->pcout << "There are " << n_ranks_with_lambda_accumulator
                    << " ranks with local lambda accumulators" << std::endl;
    }

    for (unsigned int d = 0; d < dim; ++d)
      local_lambda_accumulators[d] = numbers::invalid_unsigned_int;

    // Now actually set the dofs for the accumulators, if possible
    if (has_local_lambda_accumulator)
    {
      // We can take as accumulators the first dim lambda dofs on this
      // partition that would otherwise be constrained to zero

      // Impact on the no-slip enforcement:
      // The lambda equations on the relevant boundaries are assembled by
      // looping over the cell dofs, not only the face dofs, so if we choose
      // lambda dofs from a cell adjacent to these boundaries, this will
      // affect the no-slip enforcement. To avoid that we can take lambda dofs
      // from a cell that touches the boundary by a vertex only, and take dofs
      // which are not shared which a directly adjacent cell to the boundary.

      std::vector<types::global_dof_index> face_dofs(fe->n_dofs_per_face());
      unsigned int                         n_accumulators = 0;
      for (const auto &cell : this->dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        // if (cell_has_lambda(cell))
        {
          bool skip_cell = false;

          // Skip this cell altogether if it touches the target boundary
          // with a face
          for (const auto &face : cell->face_iterators())
            if (face->at_boundary() &&
                face->boundary_id() == weak_no_slip_boundary_id)
            {
              skip_cell = true;
              break;
            };

          if (!skip_cell)
            for (const auto i_face : cell->face_indices())
            {
              const auto &face      = cell->face(i_face);
              bool        skip_face = false;

              // Skip face if neighbouring cell through this face touches
              // the target boundary
              auto neighbor = cell->neighbor(i_face);
              if (neighbor->state() == IteratorState::IteratorStates::valid)
                for (const auto neighbor_i_face : neighbor->face_indices())
                {
                  const auto &neighbor_face = neighbor->face(neighbor_i_face);
                  if (neighbor_face->at_boundary() &&
                      neighbor_face->boundary_id() == weak_no_slip_boundary_id)
                  {
                    skip_face = true;
                    break;
                  }
                }

              if (!skip_face)
              {
                face->get_dof_indices(face_dofs);
                for (unsigned int i = 0; i < face_dofs.size(); ++i)
                {
                  types::global_dof_index dof = face_dofs[i];
                  unsigned int            comp =
                    fe->face_system_to_component_index(i, i_face).first;
                  unsigned int base =
                    fe->face_system_to_component_index(i, i_face).second;

                  // FIXME:
                  // Hardcoded to the first P2 dof of the first face whose
                  // neighbouring cell does not touch the boundary Its shape
                  // functions index (base) is 2 in 2D (P2 dof on a line)
                  // and 3 in 3D (P2 dof on triangle). This is for simplices
                  // only...
                  AssertThrow(
                    !this->param.finite_elements.use_quads &&
                      this->param.finite_elements
                          .no_slip_lagrange_mult_degree == 2,
                    ExcMessage(
                      "This coupling option for the forces-position on the "
                      "cylinder for now assumes a P2 Lagrange multiplier "
                      "on simplices only. If this changes, the lambda dofs "
                      "chosen as accumulators should be generalized "
                      "accordingly."));
                  unsigned int target_base = (dim == 2) ? 2 : 3;
                  if (base == target_base)
                    if (this->ordering->is_lambda(comp))
                      /**
                       * The accumulator must be an owned dof. It might not
                       * be possible to assign an accumulator, based on the
                       * partition used, see the assert below.
                       */
                      if (this->locally_owned_dofs.is_element(dof))
                      {
                        local_lambda_accumulators[n_accumulators++] = dof;
                        if (n_accumulators == dim)
                          goto accumulators_found;
                      }
                }
              }
            }
        }
    accumulators_found:
      if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
      {
        if constexpr (dim == 2)
        {
          std::cout << "Set lambda accumulator at dof "
                    << local_lambda_accumulators[0] << " - "
                    << local_lambda_accumulators[1] << std::endl;
        }
        else
        {
          std::cout << "Set lambda accumulator at dof "
                    << local_lambda_accumulators[0] << " - "
                    << local_lambda_accumulators[1] << " - "
                    << local_lambda_accumulators[2] << std::endl;
        }
      }
      for (unsigned int d = 0; d < dim; ++d)
        /**
         * On some weird partitions (typically with "too many" MPI procs),
         * there are owned cells on the boundary, but no owned lambda dof that
         * can be used to accumulate the local integral.
         *
         * This is technically an issue with the coupling method itself, as
         * accumulators should be defined on their own, without using
         * otherwise unused lambda dofs. Note that allowing accumulators on a
         * non-boundary face of elements touching the boundary is not
         * sufficient, because in some cases the *only* owned lambda dofs are
         * on a boundary face, and there is really no way to define an
         * accumulator without modifying the flow solution.
         */
        AssertThrow(
          local_lambda_accumulators[d] != numbers::invalid_unsigned_int,
          ExcMessage(
            "\n This rank owns at least one cell touching a boundary where "
            "no-slip should be enforced with a Lagrange multiplier (lambda). "
            "But it doesn't own any lambda degree of freedom that can be "
            "used to safely accumulate the force integral on this rank (all "
            "its lambda dofs are either ghosts, or owned but on the "
            "prescribed "
            "boundary)."
            "\n\n This can happen on somewhat pathological mesh partitions "
            "with isolated elements touching the boundary, and it probably "
            "indicates that the mesh has too few elements for the number of "
            "MPI processes used."
            "\n\n To go around this issue, try running with another number "
            "of MPI processes."));

#if defined(DEBUG_PRINTS)
      {
        // Print accumulators
        std::map<types::global_dof_index, Point<dim>> support_points =
          DoFTools::map_dofs_to_support_points(fixed_mapping_collection,
                                               this->dof_handler);

        {
          std::ofstream outfile(this->param.output.output_dir +
                                "accumulators_dofs" +
                                std::to_string(this->mpi_rank) + ".pos");
          outfile << "View \"accumulators_dofs" << this->mpi_rank << "\"{"
                  << std::endl;
          for (const auto dof : local_lambda_accumulators)
          {
            const Point<dim> &pt = support_points.at(dof);
            if constexpr (dim == 2)
              outfile << "SP(" << pt[0] << "," << pt[1] << ", 0.){1};"
                      << std::endl;
            else
              outfile << "SP(" << pt[0] << "," << pt[1] << "," << pt[2]
                      << "){1};" << std::endl;
          }
          outfile << "};" << std::endl;
          outfile.close();
        }
      }
#endif
    }

    /**
     * Set up the global lambda accumulators similarly to the global position
     * master.
     */
    if (requires_global_lambda_accumulator)
    {
      const unsigned int candidate_rank =
        has_local_lambda_accumulator ? this->mpi_rank :
                                       std::numeric_limits<unsigned int>::max();
      const unsigned int owner_rank =
        Utilities::MPI::min(candidate_rank, this->mpi_communicator);
      has_global_accumulator = (this->mpi_rank == owner_rank);

      if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
        this->pcout << "Global accumulators are on rank " << owner_rank
                    << std::endl;

      // Set the global accumulator dofs and broadcast them to all ranks
      for (unsigned int d = 0; d < dim; ++d)
      {
        global_lambda_accumulators[d] = numbers::invalid_unsigned_int;
        if (has_global_accumulator)
          global_lambda_accumulators[d] = local_lambda_accumulators[d];
      }

      Utilities::MPI::broadcast(global_lambda_accumulators.data(),
                                dim,
                                owner_rank,
                                this->mpi_communicator);

      if constexpr (running_in_debug_mode())
      {
        for (unsigned int d = 0; d < dim; ++d)
          Assert(global_lambda_accumulators[d] != numbers::invalid_unsigned_int,
                 ExcMessage(
                   "The global position master is invalid after broadcast"));
      }
    }

    // Lastly, add accumulators as ghosts on all procs who need them,
    // and reinit the parallel vectors with these additional ghosts.
    {
      // Get all the accumulator dofs
      std::vector<std::array<types::global_dof_index, dim>> gathered =
        Utilities::MPI::all_gather(this->mpi_communicator,
                                   local_lambda_accumulators);
      // all_lambda_accumulators.resize(dim);
      for (unsigned int rank = 0; rank < gathered.size(); ++rank)
        for (unsigned int d = 0; d < dim; ++d)
          if (gathered[rank][d] != numbers::invalid_unsigned_int)
            all_lambda_accumulators[d].push_back(gathered[rank][d]);
    }

    if constexpr (running_in_debug_mode())
    {
      // Check that there are indeed n_ranks_with_lambda_accumulator dofs for
      // each dimension
      for (unsigned int d = 0; d < dim; ++d)
        Assert(
          all_lambda_accumulators[d].size() == n_ranks_with_lambda_accumulator,
          ExcMessage("There are " +
                     std::to_string(all_lambda_accumulators[d].size()) +
                     "lambda accumulators in the local vector on this rank, "
                     "but there are " +
                     std::to_string(n_ranks_with_lambda_accumulator) +
                     " ranks with an accumulator."));
    }

    if (coupling == Coupling::local_position_master_to_lambda_accumulators)
    {
      // Each local position master couples to each local accumulator,
      // and thus needs these accumulators as ghosts.
      if (has_local_lambda_accumulator)
      {
        // Each rank with local accumulator
        for (unsigned int d = 0; d < dim; ++d)
          this->locally_relevant_dofs.add_indices(
            all_lambda_accumulators[d].begin(),
            all_lambda_accumulators[d].end());
        this->locally_relevant_dofs.compress();
      }
      this->reinit_ghosted_vectors();
    }
    else if (coupling == Coupling::global_position_master_to_global_accumulator)
    {
      // Global position master couples to global accumulator:
      // - rank with global position master needs global accumulator as ghost
      // - rank with global accumulator needs the local accumulators as ghosts.
      if (has_global_master_position_dofs)
      {
        for (unsigned int d = 0; d < dim; ++d)
          this->locally_relevant_dofs.add_index(global_lambda_accumulators[d]);
        this->locally_relevant_dofs.compress();
      }
      if (has_global_accumulator)
      {
        for (unsigned int d = 0; d < dim; ++d)
          this->locally_relevant_dofs.add_indices(
            all_lambda_accumulators[d].begin(),
            all_lambda_accumulators[d].end());
        this->locally_relevant_dofs.compress();
      }
      this->reinit_ghosted_vectors();
    }
  }

  /**
   * Compute the weights c_ij and identify the constrained position DOFs.
   * Done only once as cylinder is rigid and those weights will not change.
   */
  std::vector<std::map<types::global_dof_index, double>> coeffs(dim);

  FEFaceValues<dim>  fe_face_values_fixed(*this->fixed_mapping,
                                         *fe,
                                         *this->face_quadrature,
                                         update_values | update_JxW_values);
  const unsigned int n_dofs_per_face = fe->n_dofs_per_face();
  std::vector<types::global_dof_index> face_dofs(n_dofs_per_face);

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    /**
     * Loop only on the owned cells for 2 reasons :
     *
     * - Only owned cells contribute to the integral of lambda on this partition
     *
     * - The force-position coupling is done by hand by modifying the linear
     * system directly. Since each rank only stores its *owned* lines in the
     * matrix/rhs, we are only interested in the *owned* position dofs that are
     * coupled to the lambda dofs. Some ghost dofs are added here, but we only
     * care for the owned.
     *
     *   Important (see below) : since we loop over cell *faces*, we can miss
     * owned position dofs which should be coupled. This happens when owned dofs
     * are located on the edges of slanted tets, whose faces do *not* lie on the
     *   boundary of the obstacle. Thus, we never loop over these faces and
     * cannot get these owned dofs. They are added afterwards after gathering
     * the coupled dofs from other ranks.
     */
    if (cell->is_locally_owned())
    {
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);

        if (!(face->at_boundary() &&
              face->boundary_id() == weak_no_slip_boundary_id))
          continue;

        const unsigned int fe_index = cell->active_fe_index();

        fe_face_values_fixed.reinit(cell, face);
        face->get_dof_indices(face_dofs, fe_index);

        for (unsigned int q = 0; q < this->face_quadrature->size(); ++q)
        {
          const double JxW = fe_face_values_fixed.JxW(q);

          for (unsigned int i_dof = 0; i_dof < n_dofs_per_face; ++i_dof)
          {
            const unsigned int comp =
              fe->face_system_to_component_index(i_dof, i_face).first;

            // Here we need to account for ghost DoF (not only owned), which
            // contribute to the integral on this element
            // FIXME: This should never happen, to check and remove
            if (!this->locally_relevant_dofs.is_element(face_dofs[i_dof]))
              continue;

            /**
             * Lambda face dofs contribute to the weights
             */
            if (this->ordering->is_lambda(comp))
            {
              const unsigned int            d = comp - this->ordering->l_lower;
              const types::global_dof_index lambda_dof = face_dofs[i_dof];

              // Very, very, very important:
              // Even though fe_face_values_fixed is a FEFaceValues, the dof
              // index given to shape_value is still a CELL dof index.
              const unsigned int i_cell_dof =
                fe->face_to_cell_index(i_dof, i_face);
              const double phi_i =
                fe_face_values_fixed.shape_value(i_cell_dof, q);
              coeffs[d][lambda_dof] +=
                -phi_i * JxW / this->param.fsi.spring_constant;

              if constexpr (dim == 3)
                if (d == 2 && this->param.fsi.fix_z_component)
                  coeffs[d][lambda_dof] = 0.;
            }

            /**
             * Position face dofs are added to the list of coupled dofs
             */
            if (this->ordering->is_position(comp))
            {
              const unsigned int d = comp - this->ordering->x_lower;
              coupled_position_dofs.insert({face_dofs[i_dof], d});
            }
          }
        }
      }
    }
  }

  /**
   * Once again we might be missing some owned coupled dofs, on boundary edges
   * Add them here.
   * They are added only if they are already relevant (does not add ghosts).
   * FIXME: Can this be done only once instead?
   */
  {
    using MessageType =
      std::vector<std::pair<types::global_dof_index, unsigned int>>;
    MessageType coupled_position_dofs_vec(coupled_position_dofs.begin(),
                                          coupled_position_dofs.end());

    std::vector<MessageType> gathered_coupled_dofs =
      Utilities::MPI::all_gather(this->mpi_communicator,
                                 coupled_position_dofs_vec);

    for (const auto &vec : gathered_coupled_dofs)
      for (const auto &[dof, dimension] : vec)
        if (this->locally_relevant_dofs.is_element(dof))
          coupled_position_dofs.insert({dof, dimension});
  }

  /**
   * Sanity check on the weights
   * Expected sum is -1/k * |Cylinder|
   */
  {
    const double k                    = this->param.fsi.spring_constant;
    const double r                    = this->param.fsi.cylinder_radius;
    double       expected_weights_sum = -1 / k * 2. * M_PI * r;
    if constexpr (dim == 3)
      expected_weights_sum *= this->param.fsi.cylinder_length;

    const double expected_discrete_weights_sum =
      -1. / k *
      compute_boundary_volume(this->dof_handler,
                              *this->moving_mapping,
                              *this->face_quadrature,
                              weak_no_slip_boundary_id);

    for (unsigned int d = 0; d < dim; ++d)
    {
      // Do not compare for dim = 2 if fixed
      if (d == 2 && this->param.fsi.fix_z_component)
        continue;

      double local_weights_sum = 0.;
      for (const auto &[lambda_dof, weight] : coeffs[d])
        local_weights_sum += weight;

      const double weights_sum =
        Utilities::MPI::sum(local_weights_sum, this->mpi_communicator);

      if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
      {
        this->pcout << "Dim " << d << " : Sum of weights = " << weights_sum
                    << " - expected from mesh : "
                    << expected_discrete_weights_sum
                    << " - expected theoretical : " << expected_weights_sum
                    << std::endl;
      }

      AssertThrow(
        std::abs(weights_sum - expected_discrete_weights_sum) < 1e-10,
        ExcMessage(
          "The sum of weights for component " + std::to_string(d) +
          " of lambda coupling should be -1/k * |Cylinder|, but it's not."));
    }
  }

  /**
   * If using force accumulators, simply store the *local* integral coefficients
   * in a vector. Otherwise, *all* the coefficients must be gathered to evaluate
   * the complete force integral.
   */
  lambda_integral_coeffs.resize(dim);
  if (requires_local_lambda_accumulator)
  {
    for (unsigned int d = 0; d < dim; ++d)
      lambda_integral_coeffs[d] =
        std::vector<std::pair<unsigned int, double>>(coeffs[d].begin(),
                                                     coeffs[d].end());
  }
  else
  {
    for (unsigned int d = 0; d < dim; ++d)
    {
      const auto gathered = Utilities::MPI::all_gather(
        this->mpi_communicator,
        std::vector<std::pair<types::global_dof_index, double>>(
          coeffs[d].begin(), coeffs[d].end()));

      std::map<types::global_dof_index, double> coeffs_map;

      // Accumulate contributions
      for (const auto &vec : gathered)
        for (const auto &[lambda_dof, weight] : vec)
          coeffs_map[lambda_dof] += weight;

      lambda_integral_coeffs[d].insert(lambda_integral_coeffs[d].end(),
                                       coeffs_map.begin(),
                                       coeffs_map.end());
    }
  }
}

template <int dim>
void FSISolver<dim>::remove_cylinder_velocity_constraints(
  AffineConstraints<double> &constraints,
  const bool                 remove_velocity_constraints,
  const bool                 remove_position_constraints) const
{
  if (weak_no_slip_boundary_id == numbers::invalid_unsigned_int)
  {
    this->pcout << "No constraint to remove" << std::endl;
    return;
  }

  IndexSet relevant_boundary_velocity_dofs =
    DoFTools::extract_boundary_dofs(this->dof_handler,
                                    this->velocity_mask,
                                    {weak_no_slip_boundary_id});
  IndexSet relevant_boundary_position_dofs =
    DoFTools::extract_boundary_dofs(this->dof_handler,
                                    this->position_mask,
                                    {weak_no_slip_boundary_id});

  /**
   * There is a tricky corner case that happens when a partition has ghost dofs
   * on a boundary edge, but the faces sharing this edge do not belong to this
   * boundary (for instance, tets making an angle, and the tet whose face is on
   * the boundary belongs to another rank). In that case, the ghost dofs on the
   * boundary are not collected with DoFTools::extract_boundary_dofs, since the
   * ghost faces are simply not on the given boundary.
   *
   * We have to exchange the boundary dofs, and add the missing ghost ones from
   * other ranks.
   */
  {
    std::vector<std::vector<types::global_dof_index>> gathered_vel_bdr_dofs =
      Utilities::MPI::all_gather(
        this->mpi_communicator,
        relevant_boundary_velocity_dofs.get_index_vector());
    std::vector<std::vector<types::global_dof_index>> gathered_pos_bdr_dofs =
      Utilities::MPI::all_gather(
        this->mpi_communicator,
        relevant_boundary_position_dofs.get_index_vector());

    for (const auto &vec : gathered_vel_bdr_dofs)
      for (const auto dof : vec)
        if (this->locally_relevant_dofs.is_element(dof))
          relevant_boundary_velocity_dofs.add_index(dof);
    for (const auto &vec : gathered_pos_bdr_dofs)
      for (const auto dof : vec)
        if (this->locally_relevant_dofs.is_element(dof))
          relevant_boundary_position_dofs.add_index(dof);
  }

  // Check consistency of constraints for RELEVANT (not active) dofs before
  // removing
  {
    const bool consistent = constraints.is_consistent_in_parallel(
      Utilities::MPI::all_gather(this->mpi_communicator,
                                 this->locally_owned_dofs),
      // this->locally_relevant_dofs,
      DoFTools::extract_locally_active_dofs(this->dof_handler),
      this->mpi_communicator,
      true);
    AssertThrow(consistent,
                ExcMessage("Constraints are not consistent before removing"));
  }

  /**
   * Now actually remove the constraints
   */
  {
    AffineConstraints<double> filtered;
    filtered.reinit(this->locally_owned_dofs, this->locally_relevant_dofs);

    for (const auto &line : constraints.get_lines())
    {
      if (remove_velocity_constraints &&
          relevant_boundary_velocity_dofs.is_element(line.index))
        continue;
      if (remove_position_constraints &&
          relevant_boundary_position_dofs.is_element(line.index))
        continue;

      filtered.add_constraint(line.index, line.entries, line.inhomogeneity);

      // Check that entries do not involve an absent velocity dof
      // With the get_view() function, this is done automatically
      for (const auto &entry : line.entries)
      {
        if (remove_velocity_constraints)
          AssertThrow(!relevant_boundary_velocity_dofs.is_element(entry.first),
                      ExcMessage(
                        "Constraint involves a cylinder velocity dof"));
        if (remove_position_constraints)
          AssertThrow(!relevant_boundary_position_dofs.is_element(entry.first),
                      ExcMessage(
                        "Constraint involves a cylinder position dof"));
      }
    }

    filtered.close();
    constraints.clear();
    constraints = std::move(filtered);
  }

  // {
  //   // This does not work:

  //   // IndexSet local_lines = zero_constraints.get_local_lines();
  //   // local_lines.compress();
  //   // this->pcout << local_lines.n_intervals() << std::endl;
  //   // this->pcout << local_lines.n_elements() << std::endl;
  //   // this->pcout << local_lines.size() << std::endl;
  //   // this->pcout << weak_velocity_dofs.n_intervals() << std::endl;
  //   // this->pcout << weak_velocity_dofs.n_elements() << std::endl;
  //   // this->pcout << weak_velocity_dofs.size() << std::endl;
  //   // local_lines.get_view(weak_velocity_dofs);

  //   IndexSet velocity_to_keep = this->locally_relevant_dofs;
  //   velocity_to_keep.subtract_set(relevant_boundary_velocity_dofs);
  //   IndexSet position_to_keep = this->locally_relevant_dofs;
  //   position_to_keep.subtract_set(relevant_boundary_position_dofs);
  //   IndexSet to_keep = velocity_to_keep;
  //   to_keep.add_indices(position_to_keep.begin(), position_to_keep.end());

  //   auto tmp_constraints = constraints.get_view(to_keep);
  //   constraints.reinit(this->locally_owned_dofs,
  //   this->locally_relevant_dofs); constraints.close();
  //   constraints.merge(tmp_constraints);
  // }

  // {
  //   // This does not work either: (test for velocity only)
  //   // Keep everything (relevant) but the relevant boundary dofs
  //   IndexSet to_keep = this->locally_relevant_dofs;
  //   to_keep.subtract_set(relevant_boundary_velocity_dofs);
  //   AffineConstraints<double> tmp;
  //   tmp.copy_from(constraints);

  //   constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
  //   constraints.add_selected_constraints(tmp, to_keep);
  //   constraints.close();
  // }

  ///////////////////////////////////////////////////////////////////////////
  // Print the relevant dofs after removing the constraints.
  // No relevant velocity dof on the boundary should be constrained
  // for (unsigned int r = 0; r < this->mpi_size; ++r)
  // {
  //   MPI_Barrier(this->mpi_communicator);
  //   if (r == this->mpi_rank)
  //     for (unsigned int i = 0; i < n_dofs; ++i)
  //     {
  //       // Support points are defined only for relevant dofs
  //       if (!this->locally_relevant_dofs.is_element(i))
  //         continue;

  //       // Support points are not defined for the additional ghost lambda
  //       dofs if (additional_relevant_dofs.is_element(i))
  //         continue;

  //       if (relevant_boundary_velocity_dofs.is_element(i) ||
  //           relevant_boundary_position_dofs.is_element(i))
  //       {
  //         std::cout << "A: Rank " << r << " : dof " << i << " at "
  //                   << support_points.at(i)
  //                   << " is component : " << dof_to_component[i]
  //                   << " is owned : " <<
  //                   this->locally_owned_dofs.is_element(i)
  //                   << " is relevant : "
  //                   << this->locally_relevant_dofs.is_element(i)
  //                   << " is constrained : " << constraints.is_constrained(i)
  //                   << std::endl;
  //         AssertThrow(!constraints.is_constrained(i),
  //                     ExcMessage("Constrained dof remains"));
  //       }
  //       else
  //       {
  //         std::cout << "A: Rank " << r << " : dof " << i << " at "
  //                   << support_points.at(i)
  //                   << " is component : " << dof_to_component[i]
  //                   << " is owned : " <<
  //                   this->locally_owned_dofs.is_element(i)
  //                   << " is relevant : "
  //                   << this->locally_relevant_dofs.is_element(i)
  //                   << " is constrained : " << constraints.is_constrained(i)
  //                   << " (not u/x or not on boundary)" << std::endl;
  //       }
  //     }
  // }
  ///////////////////////////////////////////////////////////////////////////

  // Check consistency of constraints for RELEVANT (not active) dofs after
  // removing
  {
    const bool consistent = constraints.is_consistent_in_parallel(
      Utilities::MPI::all_gather(this->mpi_communicator,
                                 this->locally_owned_dofs),
      // this->locally_relevant_dofs,
      DoFTools::extract_locally_active_dofs(this->dof_handler),
      this->mpi_communicator,
      true);
    AssertThrow(consistent,
                ExcMessage("Constraints are not consistent after removing"));
  }

  // Check that boundary dofs were correctly removed
  if (remove_velocity_constraints)
    for (const auto &dof : relevant_boundary_velocity_dofs)
      AssertThrow(
        !constraints.is_constrained(dof),
        ExcMessage(
          "On rank " + std::to_string(this->mpi_rank) +
          " : "
          "Velocity dof " +
          std::to_string(dof) +
          " on a boundary with weak no-slip remains "
          "constrained by a boundary condition. This can happen if "
          "velocity dofs lying on both the cylinder and a face "
          "boundary have conflicting prescribed boundary conditions."));
  if (remove_position_constraints)
    for (const auto &dof : relevant_boundary_position_dofs)
      AssertThrow(
        !constraints.is_constrained(dof),
        ExcMessage(
          "On rank " + std::to_string(this->mpi_rank) +
          " : "
          "Position dof " +
          std::to_string(dof) +
          " on a boundary with weak no-slip remains "
          "constrained by a boundary condition. This can happen if "
          "position dofs lying on both the cylinder and a face "
          "boundary have conflicting prescribed boundary conditions."));
}

template <int dim>
void FSISolver<dim>::create_solver_specific_zero_constraints()
{
  this->zero_constraints.close();

  // Merge the zero lambda constraints
  this->zero_constraints.merge(
    lambda_constraints,
    AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed);

  if constexpr (dim == 3)
  {
    /** FIXME: Instead of dim = 3, the test should be whether dofs
     * belong to multiple boundaries, but for now this only happens for the
     * 3D fsi test case.
     */
    if (this->param.fsi.enable_coupling)
    {
      /**
       * Remove both position and velocity constraints on the moving boundary:
       *
       * - Position because it is coupled to the Lagrange multiplier.
       *   If the force-position constraints were handled with an
       *   AffineConstraints, this would be checked by the merge() and
       *   specifying "no_conflicts_allowed". But the constraints are enforced
       *   "by hand", so we have to manually check and remove constrained
       *   position dofs from adjacent faces.
       *
       * - Velocity because a Lagrange multiplier enforces no slip.
       *   If velocity is set by another constraint, the lambda will have
       *   garbage values since the constraint cannot be satisfied.
       */
      this->pcout << "Removing zero constraints on cylinder" << std::endl;
      remove_cylinder_velocity_constraints(this->zero_constraints, true, true);
    }
    else if (weak_no_slip_boundary_id != numbers::invalid_unsigned_int)
    {
      // If boundary has a weakly enforced no-slip, remove velocity constraints.
      remove_cylinder_velocity_constraints(this->zero_constraints, true, false);
    }
  }
}

template <int dim>
void FSISolver<dim>::create_solver_specific_nonzero_constraints()
{
  this->nonzero_constraints.close();

  // Merge the zero lambda constraints
  this->nonzero_constraints.merge(
    lambda_constraints,
    AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed);

  if constexpr (dim == 3)
  {
    if (this->param.fsi.enable_coupling)
    {
      this->pcout << "Removing nonzero constraints on cylinder" << std::endl;
      remove_cylinder_velocity_constraints(this->nonzero_constraints,
                                           true,
                                           true);
    }
    else if (weak_no_slip_boundary_id != numbers::invalid_unsigned_int)
    {
      // If boundary has a weakly enforced no-slip, remove velocity constraints.
      remove_cylinder_velocity_constraints(this->nonzero_constraints,
                                           true,
                                           false);
    }
  }
}

template <int dim>
void FSISolver<dim>::create_sparsity_pattern()
{
  //
  // Sparsity pattern and allocate matrix after the constraints are defined
  //
  DynamicSparsityPattern dsp(this->locally_relevant_dofs);

  const unsigned int n_components   = this->ordering->n_components;
  auto              &coupling_table = this->coupling_table;
  coupling_table = Table<2, DoFTools::Coupling>(n_components, n_components);
  for (unsigned int c = 0; c < n_components; ++c)
    for (unsigned int d = 0; d < n_components; ++d)
    {
      coupling_table[c][d] = DoFTools::none;

      // u couples to all variables
      if (this->ordering->is_velocity(c))
        coupling_table[c][d] = DoFTools::always;

      // p couples to u and x
      if (this->ordering->is_pressure(c))
        if (this->ordering->is_velocity(d) || this->ordering->is_position(d))
          coupling_table[c][d] = DoFTools::always;

      // x couples to itself through pseudo-solid elasticity.
      // If mesh concentration is enabled, x also couples to u because
      // h_target = h_target(||u||).
      if (this->ordering->is_position(c))
      {
        if (this->ordering->is_position(d))
          coupling_table[c][d] = DoFTools::always;

        if (this->param.mesh_concentration.enable &&
            this->ordering->is_velocity(d))
          coupling_table[c][d] = DoFTools::always;
      }
    }

  DoFTools::make_sparsity_pattern(this->dof_handler,
                                  coupling_table,
                                  dsp,
                                  this->nonzero_constraints,
                                  /* keep_constrained_dofs = */ false);

  {
    // Manually add the lambda coupling on the relevant boundary faces
    const unsigned int n_dofs_per_cell = fe->n_dofs_per_cell();
    std::vector<types::global_dof_index> cell_dofs(n_dofs_per_cell);
    for (const auto &cell : this->dof_handler.active_cell_iterators())
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);
        if (!(face->at_boundary() &&
              face->boundary_id() == weak_no_slip_boundary_id))
          continue;

        // Add coupling based on cell, rather than based on faces.
        // This is because in the assembly, we loop on the cell dofs
        // even for face terms, as the FEFaceValues functions run from
        // 0 to n_dofs_per_cell even on faces.
        cell->get_dof_indices(cell_dofs);
        // face->get_dof_indices(face_dofs);

        for (unsigned int i_dof = 0; i_dof < n_dofs_per_cell; ++i_dof)
        {
          const unsigned int comp_i =
            fe->system_to_component_index(i_dof).first;

          if (this->ordering->is_lambda(comp_i))
            for (unsigned int j_dof = 0; j_dof < n_dofs_per_cell; ++j_dof)
            {
              const unsigned int comp_j =
                fe->system_to_component_index(j_dof).first;

              // Lambda couples to u and x on faces where no-slip is enforced
              // weakly
              if (this->ordering->is_velocity(comp_j))
              {
                // Lambda couples to u and vice versa
                dsp.add(cell_dofs[i_dof], cell_dofs[j_dof]);
                dsp.add(cell_dofs[j_dof], cell_dofs[i_dof]);
              }
              if (this->ordering->is_position(comp_j))
              {
                // In the PDEs, lambda couples to x, but x does not couple to
                // lambda. The x - lambda boundary coupling is applied
                // directly in the add_algebraic_position_coupling routines.
                dsp.add(cell_dofs[i_dof], cell_dofs[j_dof]);
              }
            }
        }
      }
  }

  // Add the couplings on the cylinder depending on the chosen coupling scheme
  // Regardless of the method, couple position dofs to local master if there is
  // one on this partitions.
  // Local position masters are already coupled to themselves from the coupling
  // table
  if (has_local_position_master)
    for (const auto &[position_dof, d] : coupled_position_dofs)
      dsp.add(position_dof, local_position_master_dofs[d]);

  // Couple local lambda accumulators (one per dimension) to themselves
  // and to local lambdas of same dimension
  if (has_local_lambda_accumulator)
    for (unsigned int d = 0; d < dim; ++d)
    {
      dsp.add(local_lambda_accumulators[d], local_lambda_accumulators[d]);
      for (const auto &[lambda_dof, weight] : lambda_integral_coeffs[d])
        dsp.add(local_lambda_accumulators[d], lambda_dof);
    }

  switch (this->param.fsi.coupling)
  {
    case Coupling::all_position_to_all_lambda:
    {
      // Add the position-lambda couplings explicitly
      // In a first (current) naive approach, each position dof is coupled to
      // all lambda dofs on cylinder
      // Note : this is highly inefficient, and will be removed atfer testing
      // for alternatives.
      for (const auto &[position_dof, d] : coupled_position_dofs)
        for (const auto &[lambda_dof, weight] : lambda_integral_coeffs[d])
          dsp.add(position_dof, lambda_dof);
      break;
    }
    case Coupling::local_position_master_to_all_lambda:
    {
      // Add position-lambda couplings only for local master position dofs
      if (has_local_position_master)
        for (unsigned int d = 0; d < dim; ++d)
          // Couple the local master position dof in dimension d to the lambda
          // of same dimension (one-way coupling)
          for (const auto &[lambda_dof, weight] : lambda_integral_coeffs[d])
            dsp.add(local_position_master_dofs[d], lambda_dof);
      break;
    }
    case Coupling::global_position_master_to_all_lambda:
    {
      if (has_local_position_master)
      {
        if (has_global_master_position_dofs)
          // Add position-lambda couplings *only* for global master pos dofs
          for (unsigned int d = 0; d < dim; ++d)
            // Couple the global master position dof in dimension d to the
            // lambda of same dimension (one-way coupling)
            for (const auto &[lambda_dof, weight] : lambda_integral_coeffs[d])
              dsp.add(global_position_master_dofs[d], lambda_dof);
        else
          // If this rank does not own the global master position dofs,
          // couple its position dofs to it
          for (unsigned int d = 0; d < dim; ++d)
            // Couple the global master position dof in dimension d to the
            // lambda of same dimension (one-way coupling)
            dsp.add(local_position_master_dofs[d],
                    global_position_master_dofs[d]);
      }
      break;
    }
    case Coupling::local_position_master_to_lambda_accumulators:
    {
      if (has_local_position_master)
        // Couple local position master to all lambda accumulators (one way)
        for (unsigned int d = 0; d < dim; ++d)
        {
          // dsp.add(local_position_master_dofs[d],
          // local_position_master_dofs[d]);
          for (const auto &lambda_accumulator : all_lambda_accumulators[d])
            dsp.add(local_position_master_dofs[d], lambda_accumulator);
        }
      break;
    }
    case Coupling::global_position_master_to_global_accumulator:
    {
      if (has_local_position_master)
        for (unsigned int d = 0; d < dim; ++d)
        {
          // Couple local position master to global position master (one way)
          dsp.add(local_position_master_dofs[d],
                  global_position_master_dofs[d]);

          // Couple global position master to global accumulator (one way)
          dsp.add(global_position_master_dofs[d],
                  global_lambda_accumulators[d]);
        }

      if (has_global_accumulator)
        // Couple global lambda accumulator to each local accumulator
        for (unsigned int d = 0; d < dim; ++d)
          for (const auto &lambda_accumulator : all_lambda_accumulators[d])
            dsp.add(global_lambda_accumulators[d], lambda_accumulator);
      break;
    }
    default:
      DEAL_II_ASSERT_UNREACHABLE();
  }

  SparsityTools::distribute_sparsity_pattern(dsp,
                                             this->locally_owned_dofs,
                                             this->mpi_communicator,
                                             this->locally_relevant_dofs);

  this->system_matrix.reinit(this->locally_owned_dofs,
                             this->locally_owned_dofs,
                             dsp,
                             this->mpi_communicator);

  if (this->param.debug.verbosity == Parameters::Verbosity::verbose)
    this->pcout << "Matrix has " << this->system_matrix.n_nonzero_elements()
                << " nnz and size " << this->system_matrix.m() << " x "
                << this->system_matrix.n() << std::endl;
}

template <int dim>
void FSISolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble matrix");

  this->system_matrix = 0;

  update_mesh_concentration_field();

  CopyData copy_data(fe->n_dofs_per_cell());

#if defined(FEZ_WITH_PETSC)
  AssertThrow(
    MultithreadInfo::n_threads() == 1,
    ExcMessage(
      "Assembly is running with more than 1 thread, but uses PETSc wrappers "
      "for parallel matrix and vectors, which are not thread safe."));
#endif

  // Assemble matrix (multithreaded if supported)
  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  &FSISolver::assemble_local_matrix,
                  &FSISolver::copy_local_to_global_matrix,
                  *scratch_data,
                  copy_data);

  this->system_matrix.compress(VectorOperation::add);

  if (this->param.fsi.enable_coupling)
    add_algebraic_position_coupling_to_matrix();
}

template <int dim>
void FSISolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  this->param.mesh_concentration.set_time(this->time_handler.current_time);

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_matrix = copy_data.local_matrix;
  local_matrix       = 0.0;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

  const double bdf_c0 =
    this->time_handler.bdf_coefficients[0];

  const auto u_lower =
    const_ordering.u_lower;

  const double h_min_default =
    this->param.mesh_concentration.h_min;

  const double h_max_default =
    this->param.mesh_concentration.h_max;

  const double velocity_gradient_min =
    this->param.mesh_concentration.velocity_gradient_min;

  const double velocity_gradient_ref =
    this->param.mesh_concentration.velocity_gradient_ref;

  const double velocity_gradient_max =
    this->param.mesh_concentration.velocity_gradient_max;

  const double velocity_gradient_exponent =
    this->param.mesh_concentration.exponent;

  const double eps =
    this->param.mesh_concentration.eps;

  const double max_pressure =
    this->param.mesh_concentration.max_pressure;

  const double ramp_time =
    this->param.mesh_concentration.ramp_time;

  const double release_ratio =
    0.85;

  const double transition_width =
    0.5;

  std::array<Tensor<1, dim>, dim> size_directions;
  std::array<double, dim>         h_ref_dir;
  std::array<double, dim>         h_target_background_dir;
  std::array<double, dim>         h_min_dir;

  for (unsigned int d = 0; d < dim; ++d)
  {
    size_directions[d] =
      MeshConcentrationTools::cartesian_direction<dim>(d);

    h_ref_dir[d] =
      MeshConcentrationTools::cell_extent_in_direction<dim>(
        cell,
        size_directions[d],
        eps);

    h_target_background_dir[d] =
      MeshConcentrationTools::clamp_value(h_ref_dir[d],
                                          h_min_default,
                                          h_max_default);

    h_min_dir[d] =
      h_min_default;
  }

  cell->get_dof_indices(copy_data.local_dof_indices);

  std::vector<Tensor<1, dim>> to_multiply_by_phi_u_i_momentum(
    scratch_data.dofs_per_cell);

  std::vector<Tensor<1, dim>> to_multiply_by_phi_u_i_position(
    scratch_data.dofs_per_cell);

  std::vector<double> trace_gradu_dot_grad_phi_x_j(
    scratch_data.dofs_per_cell);

  std::vector<Tensor<2, dim>> sym_gradu_dot_grad_phi_x_j(
    scratch_data.dofs_per_cell);

  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double lame_mu =
      scratch_data.lame_mu[q];

    const double lame_lambda =
      scratch_data.lame_lambda[q];

    const double JxW_moving =
      scratch_data.JxW_moving[q];

    const double JxW_fixed =
      scratch_data.JxW_fixed[q];

    const auto &phi_u =
      scratch_data.phi_u[q];

    const auto &grad_phi_u =
      scratch_data.grad_phi_u[q];

    const auto &div_phi_u =
      scratch_data.div_phi_u[q];

    const auto &phi_p =
      scratch_data.phi_p[q];

    const auto &phi_x =
      scratch_data.phi_x[q];

    const auto &grad_phi_x =
      scratch_data.grad_phi_x[q];

    const auto &sym_grad_phi_x =
      scratch_data.sym_grad_phi_x[q];

    const auto &trace_grad_phi_x =
      scratch_data.trace_grad_phi_x[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];

    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];

    const auto &present_velocity_sym_gradients =
      scratch_data.present_velocity_sym_gradients[q];

    const double present_velocity_divergence =
      trace(present_velocity_gradients);

    const double present_pressure_values =
      scratch_data.present_pressure_values[q];

    const auto &dxdt =
      scratch_data.present_mesh_velocity_values[q];

    const auto u_ale =
      present_velocity_values - dxdt;

    const auto u_dot_grad_u_ale =
      present_velocity_gradients * u_ale;

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q,
        present_velocity_values,
        scratch_data.previous_velocity_values);

    for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
    {
      const unsigned int comp_j =
        scratch_data.components[j];

      const bool j_is_u =
        const_ordering.u_lower <= comp_j
        && comp_j < const_ordering.u_upper;

      const bool j_is_x =
        const_ordering.x_lower <= comp_j
        && comp_j < const_ordering.x_upper;

      to_multiply_by_phi_u_i_momentum[j] = Tensor<1, dim>();
      to_multiply_by_phi_u_i_position[j] = Tensor<1, dim>();
      trace_gradu_dot_grad_phi_x_j[j]    = 0.0;
      sym_gradu_dot_grad_phi_x_j[j]      = Tensor<2, dim>();

      if (j_is_u)
      {
        to_multiply_by_phi_u_i_momentum[j] =
          bdf_c0 * phi_u[j]
          + present_velocity_gradients * phi_u[j]
          + grad_phi_u[j] * u_ale;
      }

      if (j_is_x)
      {
        const auto &phi_x_j =
          phi_x[j];

        const auto &grad_phi_x_j =
          grad_phi_x[j];

        const auto trace_grad_phi_x_j =
          trace_grad_phi_x[j];

        const auto gradu_dot_grad_phi_x_j =
          present_velocity_gradients * grad_phi_x_j;

        trace_gradu_dot_grad_phi_x_j[j] =
          trace(gradu_dot_grad_phi_x_j);

        sym_gradu_dot_grad_phi_x_j[j] =
          present_velocity_sym_gradients * grad_phi_x_j;

        to_multiply_by_phi_u_i_position[j] =
          (dudt + u_dot_grad_u_ale) * trace_grad_phi_x_j
          - gradu_dot_grad_phi_x_j * u_ale
          - present_velocity_gradients * bdf_c0 * phi_x_j;
      }
    }

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const unsigned int comp_i =
        scratch_data.components[i];

      const bool i_is_u =
        const_ordering.u_lower <= comp_i
        && comp_i < const_ordering.u_upper;

      const bool i_is_p =
        comp_i == const_ordering.p_lower;

      const bool i_is_x =
        const_ordering.x_lower <= comp_i
        && comp_i < const_ordering.x_upper;

      const bool i_is_l =
        const_ordering.l_lower <= comp_i
        && comp_i < const_ordering.l_upper;

      if (i_is_l)
        continue;

      const auto &coupling_row =
        this->coupling_table[comp_i];

      const auto &phi_u_i =
        phi_u[i];

      const auto &grad_phi_u_i =
        grad_phi_u[i];

      const auto &div_phi_u_i =
        div_phi_u[i];

      const auto &phi_p_i =
        phi_p[i];

      const auto &grad_phi_x_i =
        grad_phi_x[i];

      const Tensor<2, dim> sym_grad_u_dot_grad_phi_u_i =
        present_velocity_sym_gradients * grad_phi_u_i;

      const double trace_sym_grad_u_dot_grad_phi_u_i =
        trace(sym_grad_u_dot_grad_phi_u_i);

      for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
      {
        const unsigned int comp_j =
          scratch_data.components[j];

        const bool j_is_l =
          const_ordering.l_lower <= comp_j
          && comp_j < const_ordering.l_upper;

        if (j_is_l)
          continue;

        const bool j_is_u =
          const_ordering.u_lower <= comp_j
          && comp_j < const_ordering.u_upper;

        const bool j_is_p =
          comp_j == const_ordering.p_lower;

        const bool j_is_x =
          const_ordering.x_lower <= comp_j
          && comp_j < const_ordering.x_upper;

        const bool mesh_concentration_extra_coupling =
          this->param.mesh_concentration.enable
          && i_is_x
          && j_is_x;

        if (coupling_row[comp_j] != DoFTools::always
            && !mesh_concentration_extra_coupling)
          continue;

        double local_flow_matrix_ij =
          j_is_p ? -div_phi_u_i * phi_p[j] : 0.0;

        if (i_is_u)
        {
          if (j_is_u)
          {
            const auto &gui =
              grad_phi_u_i[comp_i];

            const auto &guj =
              grad_phi_u[j][comp_j];

            local_flow_matrix_ij +=
              phi_u_i * to_multiply_by_phi_u_i_momentum[j]
              + nu * (gui[comp_j - u_lower] * guj[comp_i - u_lower]);

            if (comp_i == comp_j)
              local_flow_matrix_ij +=
                nu * gui * guj;
          }

          if (j_is_x)
          {
            const auto trace_grad_phi_x_j =
              trace_grad_phi_x[j];

            local_flow_matrix_ij +=
              phi_u_i * to_multiply_by_phi_u_i_position[j]
              + 2.0 * nu
                  * (-2.0 * scalar_product(sym_grad_phi_x[j],
                                            sym_grad_u_dot_grad_phi_u_i)
                     + trace_grad_phi_x_j
                         * trace_sym_grad_u_dot_grad_phi_u_i)
              - present_pressure_values
                  * (trace(-grad_phi_u_i * grad_phi_x[j])
                     + div_phi_u_i * trace_grad_phi_x_j);
          }
        }

        if (i_is_p)
        {
          if (j_is_u)
            local_flow_matrix_ij +=
              -phi_p_i * div_phi_u[j];

          if (j_is_x)
            local_flow_matrix_ij +=
              phi_p_i
              * (trace_gradu_dot_grad_phi_x_j[j]
                 - present_velocity_divergence * trace_grad_phi_x[j]);
        }

        double local_ps_matrix_ij = 0.0;

        if (i_is_x)
        {
          if (j_is_x)
          {
            const Tensor<2, dim> &F =
              scratch_data.present_position_gradients[q];

            const Tensor<2, dim> dF_j =
              grad_phi_x[j];

            const Tensor<2, dim> dP_elastic_j =
              MeshConcentrationTools::
                neo_hookean_first_piola_derivative<dim>(
                  F,
                  dF_j,
                  lame_mu,
                  lame_lambda,
                  eps);

            local_ps_matrix_ij +=
              scalar_product(dP_elastic_j, grad_phi_x_i);
          }

          if (this->param.mesh_concentration.enable && j_is_x)
          {
            const Tensor<2, dim> &F =
              scratch_data.present_position_gradients[q];

            const double J =
              determinant(F);

            const Tensor<2, dim> F_inv =
              invert(F);

            const Tensor<2, dim> F_inv_T =
              transpose(F_inv);

            const Tensor<2, dim> dF_j =
              grad_phi_x[j];

            const double dJ_j =
              J * trace(F_inv * dF_j);

            const Tensor<2, dim> dF_inv_T_j =
              -F_inv_T * transpose(dF_j) * F_inv_T;

            Point<dim> x_current;

            for (unsigned int d = 0; d < dim; ++d)
              x_current[d] =
                scratch_data.present_position_values[q][d];

            const double alpha =
              this->param.mesh_concentration.alpha_fun->value(x_current);

            const Tensor<1, dim> alpha_gradient =
              this->param.mesh_concentration.alpha_fun->gradient(x_current);

            const double d_alpha_j =
              alpha_gradient * phi_x[j];

            const double size_stiffness =
              2.0 * lame_mu + lame_lambda;

            const double ramp_factor =
              (ramp_time > eps)
                ? std::min(this->time_handler.current_time / ramp_time, 1.0)
                : 1.0;

            const Tensor<1, dim> grad_abs_u =
              MeshConcentrationTools::continuous_gradient_abs_velocity_value<dim>(
                mesh_concentration_grad_abs_velocity,
                copy_data.local_dof_indices,
                phi_u,
                *fe,
                u_lower);

            const Tensor<2, dim> dP_size_j =
              MeshConcentrationTools::
                isotropic_mesh_concentration_piola_derivative<dim>(
                  F,
                  F_inv_T,
                  dF_j,
                  dF_inv_T_j,
                  J,
                  dJ_j,
                  grad_abs_u,
                  h_ref_dir,
                  h_target_background_dir,
                  h_min_dir,
                  size_stiffness,
                  alpha,
                  d_alpha_j,
                  ramp_factor,
                  velocity_gradient_min,
                  velocity_gradient_ref,
                  velocity_gradient_max,
                  velocity_gradient_exponent,
                  release_ratio,
                  transition_width,
                  max_pressure,
                  eps,
                  true);

            local_ps_matrix_ij +=
              scalar_product(dP_size_j, grad_phi_x_i);
          }

          // Contribution of mesh-concentration Piola w.r.t. velocity DOFs
          if (this->param.mesh_concentration.enable && j_is_u)
          {
            // Compute derivative of grad_abs_u at this quad point wrt DOF j
            Tensor<1, dim> d_grad_abs_u;
            for (unsigned int a = 0; a < dim; ++a)
              d_grad_abs_u[a] = 0.0;

            // local velocity component index for this DOF
            const unsigned int vel_comp = comp_j - u_lower;

            // present velocity and gradient at quad
            const Tensor<1, dim> &u_q = present_velocity_values;
            const Tensor<2, dim> &grad_u_q = present_velocity_gradients;

            const double u_norm = std::sqrt(u_q * u_q + eps * eps);
            const double u_norm2 = u_norm * u_norm;

            // grad_phi_u[j] provides gradient of the j-th shape (per component)
            for (unsigned int a = 0; a < dim; ++a)
            {
              double accum = 0.0;
              for (unsigned int c = 0; c < dim; ++c)
              {
                // derivative of s_c = u_c / |u| wrt DOF value v (component vel_comp)
                const double delta_cm = (c == vel_comp) ? 1.0 : 0.0;
                const double ds_c_dv = (delta_cm - u_q[c] * u_q[vel_comp] / u_norm2) / u_norm;

                const double dudx_ca = grad_u_q[c][a];

                // derivative of dudx wrt DOF j: grad_phi_u[j][comp_j][a]
                const double d_dudx_dv = grad_phi_u[j][comp_j][a];

                const double s_c = u_q[c] / u_norm;

                accum += ds_c_dv * dudx_ca + s_c * d_dudx_dv;
              }
              d_grad_abs_u[a] = accum;
            }

            // For now, skip velocity derivative contribution to avoid incomplete implementation
            // TODO: implement full chain rule d(Piola_size)/dU = dp/dh * dh/d(grad_abs_u) * d(grad_abs_u)/dU
          }

          local_ps_matrix_ij *= JxW_fixed;
        }

        local_flow_matrix_ij *= JxW_moving;

        local_matrix(i, j) +=
          local_flow_matrix_ij + local_ps_matrix_ij;
      }
    }
  }

  if (cell->at_boundary())
  {
    for (const auto i_face : cell->face_indices())
    {
      const auto &face =
        cell->face(i_face);

      if (!face->at_boundary())
        continue;

      const auto &fluid_bc =
        this->param.fluid_bc.at(face->boundary_id());

      if (fluid_bc.type == BoundaryConditions::Type::weak_no_slip)
      {
        Assembly::weakly_enforced_no_slip_matrix<true, dim>(
          *this->ordering,
          i_face,
          scratch_data,
          this->time_handler,
          local_matrix);
      }
    }
  }

  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void FSISolver<dim>::copy_local_to_global_matrix(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_matrix,
                                                    copy_data.local_dof_indices,
                                                    this->system_matrix);
}

template <int dim>
void FSISolver<dim>::compare_analytical_matrix_with_fd()
{
  CopyData copy_data(fe->n_dofs_per_cell());

  auto errors = Verification::compare_analytical_matrix_with_fd(
    this->dof_handler,
    fe->n_dofs_per_cell(),
    *this,
    &FSISolver::assemble_local_matrix,
    &FSISolver::assemble_local_rhs,
    *scratch_data,
    copy_data,
    this->present_solution,
    this->evaluation_point,
    this->local_evaluation_point,
    this->mpi_communicator,
    this->param.output.output_dir,
    true,
    this->param.debug.analytical_jacobian_absolute_tolerance,
    this->param.debug.analytical_jacobian_relative_tolerance);

  this->pcout << "Max absolute error analytical vs fd matrix is "
              << errors.first << std::endl;

  // Only print relative error if absolute is too large
  if (errors.first > this->param.debug.analytical_jacobian_absolute_tolerance)
    this->pcout << "Max relative error analytical vs fd matrix is "
                << errors.second << std::endl;
}

template <int dim>
void FSISolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(this->computing_timer, "Assemble RHS");

  this->system_rhs = 0;

  update_mesh_concentration_field();

  CopyData copy_data(fe->n_dofs_per_cell());

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(this->dof_handler.begin_active(),
                  this->dof_handler.end(),
                  *this,
                  &FSISolver::assemble_local_rhs,
                  &FSISolver::copy_local_to_global_rhs,
                  *scratch_data,
                  copy_data);

  this->system_rhs.compress(VectorOperation::add);

  if (this->param.fsi.enable_coupling)
    add_algebraic_position_coupling_to_rhs();
}

template <int dim>
void FSISolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  this->param.mesh_concentration.set_time(this->time_handler.current_time);

  scratch_data.reinit(cell,
                      this->evaluation_point,
                      this->previous_solutions,
                      *this->source_terms,
                      *this->exact_solution);

  auto &local_rhs = copy_data.local_rhs;
  local_rhs       = 0.0;

  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;

  const double h_min_default =
    this->param.mesh_concentration.h_min;

  const double h_max_default =
    this->param.mesh_concentration.h_max;

  const double velocity_gradient_min =
    this->param.mesh_concentration.velocity_gradient_min;

  const double velocity_gradient_ref =
    this->param.mesh_concentration.velocity_gradient_ref;

  const double velocity_gradient_max =
    this->param.mesh_concentration.velocity_gradient_max;

  const double velocity_gradient_exponent =
    this->param.mesh_concentration.exponent;

  const double eps =
    this->param.mesh_concentration.eps;

  const double max_pressure =
    this->param.mesh_concentration.max_pressure;

  const double ramp_time =
    this->param.mesh_concentration.ramp_time;

  const double release_ratio =
    0.85;

  const double transition_width =
    0.5;

  std::array<Tensor<1, dim>, dim> size_directions;
  std::array<double, dim>         h_ref_dir;
  std::array<double, dim>         h_target_background_dir;
  std::array<double, dim>         h_min_dir;

  for (unsigned int d = 0; d < dim; ++d)
  {
    size_directions[d] =
      MeshConcentrationTools::cartesian_direction<dim>(d);

    h_ref_dir[d] =
      MeshConcentrationTools::cell_extent_in_direction<dim>(
        cell,
        size_directions[d],
        eps);

    h_target_background_dir[d] =
      MeshConcentrationTools::clamp_value(h_ref_dir[d],
                                          h_min_default,
                                          h_max_default);

    h_min_dir[d] =
      h_min_default;
  }

  cell->get_dof_indices(copy_data.local_dof_indices);

  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW_moving =
      scratch_data.JxW_moving[q];

    const auto &present_velocity_values =
      scratch_data.present_velocity_values[q];

    const auto &present_velocity_gradients =
      scratch_data.present_velocity_gradients[q];

    const auto &present_velocity_sym_gradients =
      scratch_data.present_velocity_sym_gradients[q];

    const auto &present_pressure_values =
      scratch_data.present_pressure_values[q];

    const auto &present_mesh_velocity_values =
      scratch_data.present_mesh_velocity_values[q];

    const auto &source_term_velocity =
      scratch_data.source_term_velocity[q];

    const auto &source_term_pressure =
      scratch_data.source_term_pressure[q];

    const double present_velocity_divergence =
      trace(present_velocity_gradients);

    const Tensor<1, dim> dudt =
      this->time_handler.compute_time_derivative_at_quadrature_node(
        q,
        present_velocity_values,
        scratch_data.previous_velocity_values);

    const auto &phi_p =
      scratch_data.phi_p[q];

    const auto &phi_u =
      scratch_data.phi_u[q];

    const auto &sym_grad_phi_u =
      scratch_data.sym_grad_phi_u[q];

    const auto &div_phi_u =
      scratch_data.div_phi_u[q];

    const double lame_mu =
      scratch_data.lame_mu[q];

    const double lame_lambda =
      scratch_data.lame_lambda[q];

    const double JxW_fixed =
      scratch_data.JxW_fixed[q];

    const auto &present_position_gradients =
      scratch_data.present_position_gradients[q];

    const auto &source_term_position =
      scratch_data.source_term_position[q];

    const Tensor<2, dim> present_pseudo_solid_piola =
      MeshConcentrationTools::neo_hookean_first_piola<dim>(
        present_position_gradients,
        lame_mu,
        lame_lambda,
        eps);

    const auto u_dot_grad_u_ale =
      present_velocity_gradients
      * (present_velocity_values - present_mesh_velocity_values);

    const auto to_multiply_by_phi_u_i =
      dudt + u_dot_grad_u_ale + source_term_velocity;

    const auto &phi_x =
      scratch_data.phi_x[q];

    const auto &grad_phi_x =
      scratch_data.grad_phi_x[q];

    Tensor<2, dim> sigma_size_q;

    if (this->param.mesh_concentration.enable)
    {
      const Tensor<2, dim> &F =
        present_position_gradients;

      const double J =
        determinant(F);

      const Tensor<2, dim> F_inv =
        invert(F);

      const Tensor<2, dim> F_inv_T =
        transpose(F_inv);

      Point<dim> x_current;

      for (unsigned int d = 0; d < dim; ++d)
        x_current[d] =
          scratch_data.present_position_values[q][d];

      const double alpha =
        this->param.mesh_concentration.alpha_fun->value(x_current);

      const double size_stiffness =
        2.0 * lame_mu + lame_lambda;

      const double ramp_factor =
        (ramp_time > eps)
          ? std::min(this->time_handler.current_time / ramp_time, 1.0)
          : 1.0;

      const Tensor<1, dim> grad_abs_u =
        MeshConcentrationTools::continuous_gradient_abs_velocity_value<dim>(
          mesh_concentration_grad_abs_velocity,
          copy_data.local_dof_indices,
          phi_u,
          *fe,
          const_ordering.u_lower);

      sigma_size_q =
        MeshConcentrationTools::isotropic_mesh_concentration_piola<dim>(
          F,
          F_inv_T,
          J,
          grad_abs_u,
          h_ref_dir,
          h_target_background_dir,
          h_min_dir,
          size_stiffness,
          alpha,
          ramp_factor,
          velocity_gradient_min,
          velocity_gradient_ref,
          velocity_gradient_max,
          velocity_gradient_exponent,
          release_ratio,
          transition_width,
          max_pressure,
          eps);
    }

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const unsigned int comp_i =
        scratch_data.components[i];

      const bool i_is_u =
        const_ordering.u_lower <= comp_i
        && comp_i < const_ordering.u_upper;

      const bool i_is_p =
        comp_i == const_ordering.p_lower;

      const bool i_is_x =
        const_ordering.x_lower <= comp_i
        && comp_i < const_ordering.x_upper;

      const bool i_is_l =
        const_ordering.l_lower <= comp_i
        && comp_i < const_ordering.l_upper;

      if (i_is_l)
        continue;

      double local_rhs_flow_i =
        i_is_p
          ? -phi_p[i] * (-present_velocity_divergence + source_term_pressure)
          : 0.0;

      if (i_is_u)
      {
        local_rhs_flow_i -=
          phi_u[i] * to_multiply_by_phi_u_i
          - div_phi_u[i] * present_pressure_values
          + 2.0 * nu
              * scalar_product(present_velocity_sym_gradients,
                               sym_grad_phi_u[i]);
      }

      local_rhs_flow_i *= JxW_moving;

      double local_rhs_ps_i = 0.0;

      if (i_is_x)
      {
        local_rhs_ps_i -=
          scalar_product(present_pseudo_solid_piola, grad_phi_x[i])
          + phi_x[i] * source_term_position
          + scalar_product(sigma_size_q, grad_phi_x[i]);

        local_rhs_ps_i *= JxW_fixed;
      }

      local_rhs(i) +=
        local_rhs_flow_i + local_rhs_ps_i;
    }
  }

  if (cell->at_boundary())
  {
    for (const auto i_face : cell->face_indices())
    {
      const auto &face =
        cell->face(i_face);

      if (!face->at_boundary())
        continue;

      const auto &fluid_bc =
        this->param.fluid_bc.at(face->boundary_id());

      if (fluid_bc.type == BoundaryConditions::Type::weak_no_slip)
        Assembly::weakly_enforced_no_slip_rhs<true>(
          *this->ordering,
          i_face,
          fluid_bc,
          scratch_data,
          local_rhs);

      if (fluid_bc.type == BoundaryConditions::Type::open_mms)
        Assembly::traction_boundary_mms_rhs(
          *this->ordering,
          i_face,
          nu,
          scratch_data,
          local_rhs);
    }
  }

  cell->get_dof_indices(copy_data.local_dof_indices);
}

template <int dim>
void FSISolver<dim>::copy_local_to_global_rhs(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;

  this->zero_constraints.distribute_local_to_global(copy_data.local_rhs,
                                                    copy_data.local_dof_indices,
                                                    this->system_rhs);
}

template <int dim>
void FSISolver<dim>::add_algebraic_position_coupling_to_matrix()
{
  TimerOutput::Scope t(this->computing_timer, "Apply constraints to matrix");

  //
  // Add algebraic constraints position-lambda
  //
  // Get row entries for each pos_dof
  std::map<types::global_dof_index, std::vector<LA::ConstMatrixIterator>>
    position_rows, master_position_rows;
  for (const auto &[pos_dof, d] : coupled_position_dofs)
    if (this->locally_owned_dofs.is_element(pos_dof))
      position_rows[pos_dof] = get_matrix_rows(this->system_matrix, pos_dof);

  if (has_global_master_position_dofs)
  {
    for (unsigned int d = 0; d < dim; ++d)
      master_position_rows[global_position_master_dofs[d]] =
        get_matrix_rows(this->system_matrix, global_position_master_dofs[d]);
  }
  else if (has_local_position_master)
  {
    for (unsigned int d = 0; d < dim; ++d)
      master_position_rows[local_position_master_dofs[d]] =
        get_matrix_rows(this->system_matrix, local_position_master_dofs[d]);
  }

  switch (this->param.fsi.coupling)
  {
    case Coupling::all_position_to_all_lambda:
    {
      // Constrain matrix
      // Constrain each owned coupled position dof to the sum of lambdas
      for (const auto &[pos_dof, d] : coupled_position_dofs)
        if (this->locally_owned_dofs.is_element(pos_dof))
          constrain_matrix_row(this->system_matrix,
                               pos_dof,
                               position_rows.at(pos_dof),
                               lambda_integral_coeffs[d]);
      break;
    }
    case Coupling::local_position_master_to_all_lambda:
    {
      if (has_local_position_master)
      {
        // Constrain matrix
        // - Constrain the local master position dofs to the sum of lambda
        // - Constrain each other coupled position dofs to the local master
        for (unsigned int d = 0; d < dim; ++d)
          constrain_matrix_row(this->system_matrix,
                               local_position_master_dofs[d],
                               master_position_rows.at(
                                 local_position_master_dofs[d]),
                               lambda_integral_coeffs[d]);

        // Set x_i - x_master = 0 for the other coupled position dofs
        for (const auto &[pos_dof, d] : coupled_position_dofs)
          if (this->locally_owned_dofs.is_element(pos_dof) &&
              pos_dof != local_position_master_dofs[d])
            constrain_matrix_row(this->system_matrix,
                                 pos_dof,
                                 position_rows.at(pos_dof),
                                 local_position_master_dofs[d],
                                 -1.);
      }
      break;
    }
    case Coupling::global_position_master_to_all_lambda:
    {
      if (has_local_position_master)
      {
        // Constrain matrix
        // - Constrain the local master position dofs to the sum of lambda
        // - Constrain each other coupled position dofs to the local master

        if (has_global_master_position_dofs)
        {
          for (unsigned int d = 0; d < dim; ++d)
            constrain_matrix_row(this->system_matrix,
                                 global_position_master_dofs[d],
                                 master_position_rows.at(
                                   global_position_master_dofs[d]),
                                 lambda_integral_coeffs[d]);
        }
        else
        {
          // Constrain local to global
          for (unsigned int d = 0; d < dim; ++d)
            constrain_matrix_row(this->system_matrix,
                                 local_position_master_dofs[d],
                                 master_position_rows.at(
                                   local_position_master_dofs[d]),
                                 global_position_master_dofs[d],
                                 -1.);
        }

        // In any case, set remaining pos dofs to local master
        // On the rank with the global master, the local is also the global
        // Set x_i - x_master = 0 for the other coupled position dofs
        for (const auto &[pos_dof, d] : coupled_position_dofs)
          if (this->locally_owned_dofs.is_element(pos_dof) &&
              pos_dof != local_position_master_dofs[d])
            constrain_matrix_row(this->system_matrix,
                                 pos_dof,
                                 position_rows.at(pos_dof),
                                 local_position_master_dofs[d],
                                 -1.);
      }
      break;
    }
    case Coupling::local_position_master_to_lambda_accumulators:
    {
      std::map<types::global_dof_index, std::vector<LA::ConstMatrixIterator>>
        accumulator_rows;
      if (has_local_lambda_accumulator)
      {
        for (unsigned int d = 0; d < dim; ++d)
          if (local_lambda_accumulators[d] != numbers::invalid_unsigned_int)
          {
            AssertThrow(
              this->locally_owned_dofs.is_element(local_lambda_accumulators[d]),
              ExcMessage("Local accumulator is not locally owned " +
                         std::to_string(local_lambda_accumulators[d])));

            accumulator_rows[local_lambda_accumulators[d]] =
              get_matrix_rows(this->system_matrix,
                              local_lambda_accumulators[d]);
          }
      }

      if (has_local_position_master)
      {
        // Set x_i - x_master = 0 for the other coupled position dofs
        for (const auto &[pos_dof, d] : coupled_position_dofs)
          if (this->locally_owned_dofs.is_element(pos_dof) &&
              pos_dof != local_position_master_dofs[d])
            constrain_matrix_row(this->system_matrix,
                                 pos_dof,
                                 position_rows.at(pos_dof),
                                 local_position_master_dofs[d],
                                 -1.);

        // Couple local master to all lambda accumulators:
        // Constrain: x_master - sum_{i_rank} accumulator_{i_rank} = 0
        for (unsigned int d = 0; d < dim; ++d)
        {
          if (this->locally_owned_dofs.is_element(
                local_position_master_dofs[d]))
          {
            std::vector<std::pair<types::global_dof_index, double>>
              accumulator_coeffs;
            for (auto lambda_accumulator : all_lambda_accumulators[d])
              accumulator_coeffs.push_back({lambda_accumulator, 1.});
            constrain_matrix_row(this->system_matrix,
                                 local_position_master_dofs[d],
                                 master_position_rows.at(
                                   local_position_master_dofs[d]),
                                 accumulator_coeffs);
          }
        }
      }

      if (has_local_lambda_accumulator)
      {
        // Couple local accumulator to local lambda dofs
        // Constrain: local_accumulator - sum_j c_j * lambda_j = 0
        for (unsigned int d = 0; d < dim; ++d)
          if (this->locally_owned_dofs.is_element(local_lambda_accumulators[d]))
            constrain_matrix_row(this->system_matrix,
                                 local_lambda_accumulators[d],
                                 accumulator_rows.at(
                                   local_lambda_accumulators[d]),
                                 lambda_integral_coeffs[d]);
      }

      break;
    }
    case Coupling::global_position_master_to_global_accumulator:
    {
      // Get the accumulator rows
      std::map<types::global_dof_index, std::vector<LA::ConstMatrixIterator>>
        accumulator_rows;
      if (has_local_lambda_accumulator)
      {
        for (unsigned int d = 0; d < dim; ++d)
          if (local_lambda_accumulators[d] != numbers::invalid_unsigned_int)
          {
            AssertThrow(
              this->locally_owned_dofs.is_element(local_lambda_accumulators[d]),
              ExcMessage("Local accumulator is not locally owned " +
                         std::to_string(local_lambda_accumulators[d])));

            accumulator_rows[local_lambda_accumulators[d]] =
              get_matrix_rows(this->system_matrix,
                              local_lambda_accumulators[d]);
          }
      }

      if (has_local_position_master)
      {
        // Set x_i - x_local_master = 0 for the other coupled position dofs
        for (const auto &[pos_dof, d] : coupled_position_dofs)
          if (this->locally_owned_dofs.is_element(pos_dof) &&
              pos_dof != local_position_master_dofs[d])
            constrain_matrix_row(this->system_matrix,
                                 pos_dof,
                                 position_rows.at(pos_dof),
                                 local_position_master_dofs[d],
                                 -1.);

        if (has_global_master_position_dofs)
        {
          // Couple global position master to global accumulator
          // Constrain: x_global - c * F_global = 0
          for (unsigned int d = 0; d < dim; ++d)
            constrain_matrix_row(this->system_matrix,
                                 global_position_master_dofs[d],
                                 master_position_rows.at(
                                   global_position_master_dofs[d]),
                                 global_lambda_accumulators[d],
                                 -1.);
        }
        else
        {
          // Couple local master to global master:
          // Constrain: x_local_master - x_global_master = 0
          for (unsigned int d = 0; d < dim; ++d)
            constrain_matrix_row(this->system_matrix,
                                 local_position_master_dofs[d],
                                 master_position_rows.at(
                                   local_position_master_dofs[d]),
                                 global_position_master_dofs[d],
                                 -1.);
        }
      }

      if (has_local_lambda_accumulator)
      {
        if (has_global_accumulator)
        {
          // Couple global accumulator to its local lambda, and to the other
          // local accumulators.
          // Constrain: global_accumulator
          //                      - (sum_j c_j * lambda_j)_{this_rank}
          //                      - sum_{other_rank} local_accumulator_rank = 0
          for (unsigned int d = 0; d < dim; ++d)
          {
            AssertThrow(this->locally_owned_dofs.is_element(
                          global_lambda_accumulators[d]),
                        ExcInternalError());

            // Add the other lambda accumulators to the coupling vector
            std::vector<std::pair<types::global_dof_index, double>>
              accumulator_coeffs(lambda_integral_coeffs[d]);

            for (auto lambda_accumulator : all_lambda_accumulators[d])
              if (lambda_accumulator != global_lambda_accumulators[d])
                accumulator_coeffs.push_back({lambda_accumulator, 1.});

            constrain_matrix_row(this->system_matrix,
                                 global_lambda_accumulators[d],
                                 accumulator_rows.at(
                                   global_lambda_accumulators[d]),
                                 accumulator_coeffs);
          }
        }
        else
        {
          // Couple local accumulator to its local lambda dofs
          // Constrain: local_accumulator - sum_j c_j * lambda_j = 0
          for (unsigned int d = 0; d < dim; ++d)
            if (this->locally_owned_dofs.is_element(
                  local_lambda_accumulators[d]))
              constrain_matrix_row(this->system_matrix,
                                   local_lambda_accumulators[d],
                                   accumulator_rows.at(
                                     local_lambda_accumulators[d]),
                                   lambda_integral_coeffs[d]);
        }
      }

      break;
    }
    default:
      DEAL_II_ASSERT_UNREACHABLE();
  }

  this->system_matrix.compress(VectorOperation::insert);
}

template <int dim>
void FSISolver<dim>::add_algebraic_position_coupling_to_rhs()
{
  // Set RHS to zero for coupled position dofs
  for (const auto &[pos_dof, d] : coupled_position_dofs)
    if (this->locally_owned_dofs.is_element(pos_dof))
      this->system_rhs(pos_dof) = 0.;

  // Set RHS to zero for local lambda accumulator
  if (this->param.fsi.coupling ==
        Coupling::local_position_master_to_lambda_accumulators ||
      this->param.fsi.coupling ==
        Coupling::global_position_master_to_global_accumulator)
    for (const auto accumulator_dof : local_lambda_accumulators)
      if (this->locally_owned_dofs.is_element(accumulator_dof))
        this->system_rhs(accumulator_dof) = 0.;

  this->system_rhs.compress(VectorOperation::insert);
}

/**
 * Compute integral of lambda (fluid force), compare to position dofs
 */
template <int dim>
void FSISolver<dim>::compare_forces_and_position_on_obstacle() const
{
  Tensor<1, dim> lambda_integral, lambda_integral_local;
  lambda_integral_local = 0;

  FEFaceValues<dim> fe_face_values(*this->moving_mapping,
                                   *fe,
                                   *this->face_quadrature,
                                   update_values | update_JxW_values);

  // Compute integral of lambda on owned boundary
  const unsigned int n_faces_q_points = this->face_quadrature->size();
  std::vector<types::global_dof_index> face_dofs(fe->n_dofs_per_face());

  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

  Tensor<1, dim>    cylinder_displacement_local, max_diff_local;
  std::vector<bool> first_computed_displacement(dim, true);

  for (auto cell : this->dof_handler.active_cell_iterators())
    if (cell->is_locally_owned() && cell->at_boundary())
      for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
      {
        const auto &face = cell->face(i_face);
        if (face->at_boundary() &&
            face->boundary_id() == weak_no_slip_boundary_id)
        {
          fe_face_values.reinit(cell, i_face);

          // Increment lambda integral
          fe_face_values[lambda_extractor].get_function_values(
            this->present_solution, lambda_values);
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
            lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);

          /**
           * Cylinder is rigid, so all displacements should be identical for a
           * given component. If first position dof, save displacement,
           * otherwise compare with saved displacement.
           */
          face->get_dof_indices(face_dofs);

          for (unsigned int i_dof = 0; i_dof < fe->n_dofs_per_face(); ++i_dof)
            if (this->locally_owned_dofs.is_element(face_dofs[i_dof]))
            {
              const unsigned int comp =
                fe->face_system_to_component_index(i_dof, i_face).first;
              if (this->ordering->is_position(comp))
              {
                const unsigned int d = comp - this->ordering->x_lower;

                if (first_computed_displacement[d])
                {
                  // Save displacement
                  first_computed_displacement[d] = false;
                  cylinder_displacement_local[d] =
                    this->present_solution[face_dofs[i_dof]] -
                    this->initial_positions.at(face_dofs[i_dof])[d];
                }
                else
                {
                  // Compare with saved displacement
                  const double displ =
                    this->present_solution[face_dofs[i_dof]] -
                    this->initial_positions.at(face_dofs[i_dof])[d];
                  max_diff_local[d] =
                    std::max(max_diff_local[d],
                             cylinder_displacement_local[d] - displ);
                }
              }
            }
        }
      }

  for (unsigned int d = 0; d < dim; ++d)
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d], this->mpi_communicator);

  // To take the max displacement while preserving sign
  struct MaxAbsOp
  {
    static void
    apply(void *invec, void *inoutvec, int *len, MPI_Datatype * /*dtype*/)
    {
      double *in    = static_cast<double *>(invec);
      double *inout = static_cast<double *>(inoutvec);
      for (int i = 0; i < *len; ++i)
      {
        if (std::fabs(in[i]) > std::fabs(inout[i]))
          inout[i] = in[i];
      }
    }
  };
  MPI_Op mpi_maxabs;
  MPI_Op_create(&MaxAbsOp::apply, /*commutative=*/true, &mpi_maxabs);

  Tensor<1, dim> cylinder_displacement, max_diff, ratio;
  for (unsigned int d = 0; d < dim; ++d)
  {
    /**
     * Cylinder displacement is trivially 0 on processes which do not own a
     * part of the boundary, and is nontrivial otherwise.     Taking the max
     * to synchronize does not work because displacement can be negative.
     * Instead, we take the max while preserving the sign.
     */
    MPI_Allreduce(&cylinder_displacement_local[d],
                  &cylinder_displacement[d],
                  1,
                  MPI_DOUBLE,
                  mpi_maxabs,
                  this->mpi_communicator);

    // Take the max between all max differences disp_i - disp_j
    // for x_i and x_j both on the cylinder.
    // Checks that all displacement are identical.
    max_diff[d] =
      Utilities::MPI::max(max_diff_local[d], this->mpi_communicator);

    // Check that the ratio of both terms in the position
    // boundary condition is -spring_constant
    if (std::abs(cylinder_displacement[d]) > 1e-7)
      ratio[d] = lambda_integral[d] / cylinder_displacement[d];
  }

  if (this->param.fsi.verbosity == Parameters::Verbosity::verbose)
  {
    this->pcout << std::endl;
    this->pcout << std::scientific << std::setprecision(8) << std::showpos;
    this->pcout
      << "Checking consistency between lambda integral and position BC:"
      << std::endl;
    this->pcout << "Integral of lambda on cylinder is " << lambda_integral
                << std::endl;
    this->pcout << "Prescribed displacement        is " << cylinder_displacement
                << std::endl;
    this->pcout << "                         Ratio is " << ratio
                << " (expected: " << -this->param.fsi.spring_constant << ")"
                << std::endl;
    this->pcout << "Max diff between displacements is " << max_diff
                << std::endl;
    this->pcout << std::endl;
  }

  AssertThrow(max_diff.norm() <= 1e-8,
              ExcMessage(
                "Displacement values of the cylinder are not all the same."));

  //
  // Check relative error between lambda/disp ratio vs spring constant
  //
  for (unsigned int d = 0; d < dim; ++d)
  {
    if (std::abs(ratio[d]) < 1e-10)
      continue;
    if (std::abs(lambda_integral[d]) < 1e-12)
      continue;

    const double absolute_error =
      std::abs(ratio[d] - (-this->param.fsi.spring_constant));

    if (absolute_error <= 1e-6)
      continue;

    const double relative_error =
      absolute_error / this->param.fsi.spring_constant;

    this->pcout << "Relative error = " << relative_error << std::endl;

    AssertThrow(relative_error <= 1e-2,
                ExcMessage("Ratio integral vs displacement values is not -k"));
  }
}

template <int dim>
void FSISolver<dim>::check_velocity_boundary() const
{
  LagrangeMultiplierTools::check_no_slip_on_boundary<dim>(
    this->param,
    *scratch_data,
    this->dof_handler,
    this->evaluation_point,
    this->previous_solutions,
    *this->source_terms,
    *this->exact_solution,
    weak_no_slip_boundary_id);
}

template <int dim>
void FSISolver<dim>::check_manufactured_solution_boundary()
{
  Tensor<1, dim> lambdaMMS_integral, lambdaMMS_integral_local;
  Tensor<1, dim> lambda_integral, lambda_integral_local;
  Tensor<1, dim> pns_integral, pns_integral_local;
  lambdaMMS_integral_local = 0;
  lambda_integral_local    = 0;
  pns_integral_local       = 0;

  const double mu = this->param.physical_properties.fluids[0].dynamic_viscosity;

  FEFaceValues<dim> fe_face_values(*this->moving_mapping,
                                   *fe,
                                   *this->face_quadrature,
                                   update_values | update_quadrature_points |
                                     update_JxW_values | update_normal_vectors);
  FEFaceValues<dim> fe_face_values_fixed(*this->fixed_mapping,
                                         *fe,
                                         *this->face_quadrature,
                                         update_values |
                                           update_quadrature_points |
                                           update_JxW_values);

  const unsigned int          n_faces_q_points = this->face_quadrature->size();
  Tensor<1, dim>              lambda_MMS;
  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);

  //
  // First compute integral over cylinder of lambda_MMS
  //
  for (auto cell : this->dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;
    for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
    {
      const auto &face = cell->face(i_face);
      if (face->at_boundary() &&
          face->boundary_id() == weak_no_slip_boundary_id)
      {
        fe_face_values.reinit(cell, i_face);

        // Get FE solution values on the face
        fe_face_values[lambda_extractor].get_function_values(
          this->present_solution, lambda_values);

        // Evaluate exact solution at quadrature points
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          const Point<dim> &qpoint = fe_face_values.quadrature_point(q);
          const auto        normal_to_solid = -fe_face_values.normal_vector(q);

          const double p_MMS =
            this->exact_solution->value(qpoint, this->ordering->p_lower);

          std::static_pointer_cast<FSISolver<dim>::MMSSolution>(
            this->exact_solution)
            ->lagrange_multiplier(qpoint, mu, normal_to_solid, lambda_MMS);

          // Increment the integrals of lambda:

          // This is int - sigma(u_MMS, p_MMS) cdot normal_to_solid
          lambdaMMS_integral_local += lambda_MMS * fe_face_values.JxW(q);

          /**
           * This is int lambda := int sigma(u_MMS, p_MMS) cdot normal_to_fluid
           *                                                   -normal_to_solid
           */
          lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);

          // Increment integral of p * n_solid
          pns_integral_local += p_MMS * normal_to_solid * fe_face_values.JxW(q);
        }
      }
    }
  }

  for (unsigned int d = 0; d < dim; ++d)
  {
    lambdaMMS_integral[d] =
      Utilities::MPI::sum(lambdaMMS_integral_local[d], this->mpi_communicator);
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d], this->mpi_communicator);
  }
  pns_integral =
    Utilities::MPI::sum(pns_integral_local, this->mpi_communicator);

  // // Reference solution for int_Gamma p*n_solid dx is - k * d * f(t).
  // Tensor<1, dim> translation;
  // translation[0] = 0.1;
  // translation[1] = 0.05;
  const Tensor<1, dim> ref_pns;
  // const Tensor<1, dim> ref_pns =
  //   -param.fsi.spring_constant * translation *
  //   std::static_pointer_cast<FSISolver<dim>::MMSSolution>(
  //     exact_solution)->mms.exact_mesh_position->time_function->value(this->time_handler.current_time);
  // const double err_pns = (ref_pns - pns_integral).norm();
  const double err_pns = -1.;

  //
  // Check x_MMS
  //
  Tensor<1, dim> x_MMS;
  double         max_x_error = 0.;
  for (auto cell : this->dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;
    for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
    {
      const auto &face = cell->face(i_face);
      if (face->at_boundary() &&
          face->boundary_id() == weak_no_slip_boundary_id)
      {
        fe_face_values_fixed.reinit(cell, i_face);

        // Evaluate exact solution at quadrature points
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          const Point<dim> &qpoint_fixed =
            fe_face_values_fixed.quadrature_point(q);

          for (unsigned int d = 0; d < dim; ++d)
            x_MMS[d] = this->exact_solution->value(qpoint_fixed,
                                                   this->ordering->x_lower + d);

          const Tensor<1, dim> ref =
            -1. / this->param.fsi.spring_constant * lambdaMMS_integral;
          const double err = ((x_MMS - qpoint_fixed) - ref).norm();
          max_x_error      = std::max(max_x_error, err);
        }
      }
    }
  }

  //
  // Check u_MMS
  //
  Tensor<1, dim> u_MMS, w_MMS;
  double         max_u_error = -1;
  // for (auto cell : this->dof_handler.active_cell_iterators())
  // {
  //   if (!cell->is_locally_owned())
  //     continue;
  //   for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
  //   {
  //     const auto &face = cell->face(i_face);
  //     if (face->at_boundary() && face->boundary_id() == boundary_id)
  //     {
  //       fe_face_values.reinit(cell, i_face);
  //       fe_face_values_fixed.reinit(cell, i_face);

  //       for (unsigned int q = 0; q < n_faces_q_points; ++q)
  //       {
  //         const Point<dim> &qpoint = fe_face_values.quadrature_point(q);
  //         const Point<dim> &qpoint_fixed  =
  //         fe_face_values_fixed.quadrature_point(q);

  //         for (unsigned int d = 0; d < dim; ++d)
  //         {
  //           u_MMS[d] = solution_fun.value(qpoint, u_lower + d);
  //           w_MMS[d] = mesh_velocity_fun.value(qpoint_fixed, x_lower + d);
  //         }

  //         const double err = (u_MMS - w_MMS).norm();
  //         // std::cout << "u_MMS & w_MMS at quad node are " << u_MMS << " ,
  //         "
  //         << w_MMS << " - norm diff = " << err << std::endl; max_u_error =
  //         std::max(max_u_error, err);
  //       }
  //     }
  //   }
  // }

  // if(VERBOSE)
  // {
  this->pcout << std::endl;
  this->pcout << "Checking manufactured solution for k = "
              << this->param.fsi.spring_constant << " :" << std::endl;
  this->pcout << "integral lambda         = " << lambda_integral << std::endl;
  this->pcout << "integral lambdaMMS      = " << lambdaMMS_integral
              << std::endl;
  this->pcout << "integral pMMS * n_solid = " << pns_integral << std::endl;
  this->pcout << "reference: -k*d*f(t)    = " << ref_pns
              << " - err = " << err_pns << std::endl;
  this->pcout << "max error on (x_MMS -    X0) vs -1/k * integral lambda = "
              << max_x_error << std::endl;
  this->pcout << "max error on  u_MMS          vs w_MMS                  = "
              << max_u_error << std::endl;
  this->pcout << std::endl;
  // }
}

template <int dim>
void FSISolver<dim>::compute_lambda_error_on_boundary(
  double         &lambda_l2_error,
  double         &lambda_linf_error,
  Tensor<1, dim> &error_on_integral)
{
  double lambda_l2_local   = 0;
  double lambda_linf_local = 0;

  Tensor<1, dim> lambda_integral, exact_integral, lambda_integral_local,
    exact_integral_local;
  lambda_integral_local = 0;
  exact_integral_local  = 0;

#if !defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
  const double rho = this->param.physical_properties.fluids[0].density;
  const double nu =
    this->param.physical_properties.fluids[0].kinematic_viscosity;
  const double mu = nu * rho;
#endif

  FEFaceValues<dim> fe_face_values(*this->moving_mapping,
                                   *fe,
                                   *this->face_quadrature,
                                   update_values | update_quadrature_points |
                                     update_JxW_values | update_normal_vectors);

  const unsigned int          n_faces_q_points = this->face_quadrature->size();
  std::vector<Tensor<1, dim>> lambda_values(n_faces_q_points);
  Tensor<1, dim>              diff, exact;

  for (auto cell : this->dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    for (unsigned int i_face = 0; i_face < cell->n_faces(); ++i_face)
    {
      const auto &face = cell->face(i_face);

      if (face->at_boundary() &&
          face->boundary_id() == weak_no_slip_boundary_id)
      {
        fe_face_values.reinit(cell, i_face);

        // Get FE solution values on the face
        fe_face_values[lambda_extractor].get_function_values(
          this->present_solution, lambda_values);

        // Evaluate exact solution at quadrature points
        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          const Point<dim> &qpoint = fe_face_values.quadrature_point(q);

#if defined(LAGRANGE_MULTIPLIER_WITH_SOURCE_TERM)
          // The lambda_MMS is also prescribed, use this solution
          for (unsigned int d = 0; d < dim; ++d)
            exact[d] =
              this->exact_solution->value(qpoint, this->ordering->l_lower + d);
#else
          const auto normal_to_mesh  = fe_face_values.normal_vector(q);
          const auto normal_to_solid = -normal_to_mesh;

          // Careful:
          // int lambda := int sigma(u_MMS, p_MMS) cdot  normal_to_fluid
          //                                                   =
          //                                             normal_to_mesh
          //                                                   =
          //                                            -normal_to_solid
          //
          // Got to take the consistent normal to compare int lambda_h with
          // solution.
          //
          // Solution<dim> computes lambda_exact = - sigma cdot ns, where n is
          // expected to be the normal to the SOLID.

          // lambda_MMS is not prescribed, the exact lambda is expected to be
          // the traction
          std::static_pointer_cast<FSISolver<dim>::MMSSolution>(
            this->exact_solution)
            ->lagrange_multiplier(qpoint, mu, normal_to_solid, exact);
#endif

          diff = lambda_values[q] - exact;

          lambda_l2_local += diff * diff * fe_face_values.JxW(q);
          lambda_linf_local =
            std::max(lambda_linf_local, std::abs(diff.norm()));

          // Increment the integral of lambda
          lambda_integral_local += lambda_values[q] * fe_face_values.JxW(q);
          exact_integral_local += exact * fe_face_values.JxW(q);
        }
      }
    }
  }

  lambda_l2_error =
    Utilities::MPI::sum(lambda_l2_local, this->mpi_communicator);
  lambda_l2_error = std::sqrt(lambda_l2_error);

  lambda_linf_error =
    Utilities::MPI::max(lambda_linf_local, this->mpi_communicator);

  for (unsigned int d = 0; d < dim; ++d)
  {
    lambda_integral[d] =
      Utilities::MPI::sum(lambda_integral_local[d], this->mpi_communicator);
    exact_integral[d] =
      Utilities::MPI::sum(exact_integral_local[d], this->mpi_communicator);
    error_on_integral[d] = std::abs(lambda_integral[d] - exact_integral[d]);
  }
}

template <int dim>
void FSISolver<dim>::compute_solver_specific_errors()
{
  double         l2_l = 0., li_l = 0.;
  Tensor<1, dim> error_on_integral;
  this->compute_lambda_error_on_boundary(l2_l, li_l, error_on_integral);
  // linf_error_Fx = std::max(linf_error_Fx, error_on_integral[0]);
  // linf_error_Fy = std::max(linf_error_Fy, error_on_integral[1]);

  const double t = this->time_handler.current_time;
  for (auto &[norm, handler] : this->error_handlers)
  {
    if (norm == VectorTools::L2_norm)
      handler.add_error("l", l2_l, t);
    if (norm == VectorTools::Linfty_norm)
      handler.add_error("l", li_l, t);

    if (this->param.fsi.compute_error_on_forces)
    {
      // The error on the forces is |F_h - F_exact|, there is no need to
      // distinguish between L^p norms.
      for (unsigned int d = 0; d < dim; ++d)
        handler.add_error("F_comp" + std::to_string(d),
                          error_on_integral[d],
                          t);
    }
  }
}

template <int dim>
void FSISolver<dim>::add_solver_specific_postprocessing_data()
{
  if (!this->postproc_handler->should_output_volume_fields(this->time_handler))
    return;

  /*
   * Champs pseudo-solide classiques.
   */
  Vector<float> lame_mu_cell(this->triangulation.n_active_cells());
  Vector<float> lame_lambda_cell(this->triangulation.n_active_cells());

  /*
   * Champs de diagnostic pour la concentration isotrope.
   */
  Vector<float> mesh_alpha_cell(this->triangulation.n_active_cells());

  Vector<float> mesh_h_current_iso_cell(this->triangulation.n_active_cells());
  Vector<float> mesh_h_target_iso_cell(this->triangulation.n_active_cells());
  Vector<float> mesh_e_iso_cell(this->triangulation.n_active_cells());
  Vector<float> mesh_p_iso_cell(this->triangulation.n_active_cells());
  Vector<float> mesh_w_iso_cell(this->triangulation.n_active_cells());

  Vector<float> mesh_grad_abs_u_norm_cell(this->triangulation.n_active_cells());
  Vector<float> mesh_grad_abs_u_x_cell(this->triangulation.n_active_cells());
  Vector<float> mesh_grad_abs_u_y_cell(this->triangulation.n_active_cells());
  Vector<float> mesh_grad_abs_u_z_cell(this->triangulation.n_active_cells());

  const auto &mu_fun =
    this->param.physical_properties.pseudosolids[0].lame_mu_fun;

  const auto &lambda_fun =
    this->param.physical_properties.pseudosolids[0].lame_lambda_fun;

  /*
   * Si la concentration est activée, on met à jour le champ continu
   * grad(|u|) utilisé par la loi h_target.
   */
  if (this->param.mesh_concentration.enable)
  {
    this->param.mesh_concentration.set_time(this->time_handler.current_time);
    update_mesh_concentration_field();
  }

  /*
   * On évalue sur le mapping fixe :
   *
   * - x(X), pour retrouver la position actuelle ;
   * - F = grad_X(x), pour calculer h_current_iso ;
   * - les fonctions de forme vitesse, pour reconstruire grad(|u|) continu ;
   * - JxW_fixed, pour moyenner par cellule en configuration de référence.
   */
  FEValues<dim> fe_values(
    *this->fixed_mapping,
    *fe,
    *this->quadrature,
    update_values | update_gradients | update_JxW_values);

  std::vector<Tensor<1, dim>> position_values(
    this->quadrature->size());

  std::vector<Tensor<2, dim>> position_gradients(
    this->quadrature->size());

  std::vector<types::global_dof_index> local_dof_indices(
    fe->n_dofs_per_cell());

  std::vector<Tensor<1, dim>> phi_u_q(
    fe->n_dofs_per_cell());

  /*
   * Paramètres identiques à ceux utilisés dans l'assemblage.
   */
  const double h_min_default =
    this->param.mesh_concentration.h_min;

  const double h_max_default =
    this->param.mesh_concentration.h_max;

  const double velocity_gradient_min =
    this->param.mesh_concentration.velocity_gradient_min;

  const double velocity_gradient_ref =
    this->param.mesh_concentration.velocity_gradient_ref;

  const double velocity_gradient_max =
    this->param.mesh_concentration.velocity_gradient_max;

  const double velocity_gradient_exponent =
    this->param.mesh_concentration.exponent;

  const double eps =
    this->param.mesh_concentration.eps;

  const double max_pressure =
    this->param.mesh_concentration.max_pressure;

  const double ramp_time =
    this->param.mesh_concentration.ramp_time;

  const double release_ratio =
    0.85;

  const double transition_width =
    0.5;

  const double ramp_factor =
    (ramp_time > eps)
      ? std::min(this->time_handler.current_time / ramp_time, 1.0)
      : 1.0;

  for (const auto &cell : this->dof_handler.active_cell_iterators())
  {
    if (!cell->is_locally_owned())
      continue;

    const unsigned int cell_id =
      cell->active_cell_index();

    /*
     * Valeurs classiques au centre de la cellule.
     */
    const double lame_mu_cell_value =
      mu_fun->value(cell->center());

    const double lame_lambda_cell_value =
      lambda_fun->value(cell->center());

    lame_mu_cell[cell_id] =
      static_cast<float>(lame_mu_cell_value);

    lame_lambda_cell[cell_id] =
      static_cast<float>(lame_lambda_cell_value);

    if (!this->param.mesh_concentration.enable)
      continue;

    const double size_stiffness =
      2.0 * lame_mu_cell_value + lame_lambda_cell_value;

    /*
     * Tailles de référence par direction.
     *
     * Même en isotrope, on les garde uniquement pour construire :
     *
     *   h_ref_iso = moyenne(h_ref_dir)
     *
     * et :
     *
     *   h_background_iso = moyenne(h_target_background_dir)
     */
    std::array<Tensor<1, dim>, dim> size_directions;
    std::array<double, dim>         h_ref_dir;
    std::array<double, dim>         h_target_background_dir;
    std::array<double, dim>         h_min_dir;

    for (unsigned int d = 0; d < dim; ++d)
    {
      size_directions[d] =
        MeshConcentrationTools::cartesian_direction<dim>(d);

      h_ref_dir[d] =
        MeshConcentrationTools::cell_extent_in_direction<dim>(
          cell,
          size_directions[d],
          eps);

      h_target_background_dir[d] =
        MeshConcentrationTools::clamp_value(h_ref_dir[d],
                                            h_min_default,
                                            h_max_default);

      h_min_dir[d] =
        h_min_default;
    }

    /*
     * Récupération des champs aux points de quadrature.
     */
    fe_values.reinit(cell);

    fe_values[this->position_extractor].get_function_values(
      this->present_solution,
      position_values);

    fe_values[this->position_extractor].get_function_gradients(
      this->present_solution,
      position_gradients);

    cell->get_dof_indices(local_dof_indices);

    double h_current_iso_cell      = 0.0;
    double h_target_iso_cell       = 0.0;
    double p_iso_cell              = 0.0;
    double w_iso_cell              = 0.0;
    double e_iso_cell              = 0.0;
    double alpha_cell              = 0.0;
    double grad_abs_u_norm_cell    = 0.0;
    double volume_cell             = 0.0;

    std::array<double, dim> grad_abs_u_dir;

    for (unsigned int d = 0; d < dim; ++d)
      grad_abs_u_dir[d] = 0.0;

    for (unsigned int q = 0; q < this->quadrature->size(); ++q)
    {
      const Tensor<2, dim> &F_q =
        position_gradients[q];

      Point<dim> x_current_q;

      for (unsigned int d = 0; d < dim; ++d)
        x_current_q[d] =
          position_values[q][d];

      const double alpha_q =
        this->param.mesh_concentration.alpha_fun->value(x_current_q);

      const double JxW_q =
        fe_values.JxW(q);

      /*
       * Fonctions de forme de vitesse au point q.
       *
       * Elles servent à reconstruire le champ continu
       * mesh_concentration_grad_abs_velocity.
       */
      for (unsigned int k = 0; k < fe->n_dofs_per_cell(); ++k)
        phi_u_q[k] =
          fe_values[this->velocity_extractor].value(k, q);

      const Tensor<1, dim> grad_abs_u_q =
        MeshConcentrationTools::continuous_gradient_abs_velocity_value<dim>(
          mesh_concentration_grad_abs_velocity,
          local_dof_indices,
          phi_u_q,
          *fe,
          const_ordering.u_lower);

      const double grad_abs_u_norm_q =
        grad_abs_u_q.norm();

      const double h_current_iso_q =
        MeshConcentrationTools::isotropic_h_current_cell_value<dim>(
          F_q,
          h_ref_dir,
          eps);

      const double h_target_iso_q =
        MeshConcentrationTools::isotropic_h_target_cell_value<dim>(
          grad_abs_u_q,
          h_target_background_dir,
          h_min_dir,
          velocity_gradient_min,
          velocity_gradient_ref,
          velocity_gradient_max,
          velocity_gradient_exponent,
          eps);

      const double p_iso_q =
        MeshConcentrationTools::isotropic_pressure_cell_value<dim>(
          F_q,
          grad_abs_u_q,
          h_ref_dir,
          h_target_background_dir,
          h_min_dir,
          size_stiffness,
          alpha_q,
          ramp_factor,
          velocity_gradient_min,
          velocity_gradient_ref,
          velocity_gradient_max,
          velocity_gradient_exponent,
          release_ratio,
          transition_width,
          max_pressure,
          eps);

      const double w_iso_q =
        MeshConcentrationTools::gradient_abs_velocity_weight<dim>(
          grad_abs_u_q,
          velocity_gradient_min,
          velocity_gradient_ref,
          velocity_gradient_max,
          velocity_gradient_exponent,
          eps);

      h_current_iso_cell +=
        h_current_iso_q * JxW_q;

      h_target_iso_cell +=
        h_target_iso_q * JxW_q;

      p_iso_cell +=
        p_iso_q * JxW_q;

      w_iso_cell +=
        w_iso_q * JxW_q;

      alpha_cell +=
        alpha_q * JxW_q;

      grad_abs_u_norm_cell +=
        grad_abs_u_norm_q * JxW_q;

      for (unsigned int d = 0; d < dim; ++d)
        grad_abs_u_dir[d] +=
          grad_abs_u_q[d] * JxW_q;

      volume_cell +=
        JxW_q;
    }

    if (volume_cell > eps)
    {
      h_current_iso_cell /=
        volume_cell;

      h_target_iso_cell /=
        volume_cell;

      p_iso_cell /=
        volume_cell;

      w_iso_cell /=
        volume_cell;

      alpha_cell /=
        volume_cell;

      grad_abs_u_norm_cell /=
        volume_cell;

      for (unsigned int d = 0; d < dim; ++d)
        grad_abs_u_dir[d] /=
          volume_cell;
    }

    e_iso_cell =
      (h_current_iso_cell - h_target_iso_cell)
      / std::max(h_target_iso_cell, eps);

    mesh_h_current_iso_cell[cell_id] =
      static_cast<float>(h_current_iso_cell);

    mesh_h_target_iso_cell[cell_id] =
      static_cast<float>(h_target_iso_cell);

    mesh_e_iso_cell[cell_id] =
      static_cast<float>(e_iso_cell);

    mesh_p_iso_cell[cell_id] =
      static_cast<float>(p_iso_cell);

    mesh_w_iso_cell[cell_id] =
      static_cast<float>(w_iso_cell);

    mesh_alpha_cell[cell_id] =
      static_cast<float>(alpha_cell);

    mesh_grad_abs_u_norm_cell[cell_id] =
      static_cast<float>(grad_abs_u_norm_cell);

    mesh_grad_abs_u_x_cell[cell_id] =
      static_cast<float>(grad_abs_u_dir[0]);

    if constexpr (dim >= 2)
      mesh_grad_abs_u_y_cell[cell_id] =
        static_cast<float>(grad_abs_u_dir[1]);

    if constexpr (dim == 3)
      mesh_grad_abs_u_z_cell[cell_id] =
        static_cast<float>(grad_abs_u_dir[2]);
  }

  /*
   * Champs pseudo-solide.
   */
  this->postproc_handler->add_cell_data_vector(lame_mu_cell,
                                               "lame_mu");

  this->postproc_handler->add_cell_data_vector(lame_lambda_cell,
                                               "lame_lambda");

  /*
   * Champs concentration isotrope.
   */
  if (this->param.mesh_concentration.enable)
  {
    this->postproc_handler->add_cell_data_vector(mesh_h_current_iso_cell,
                                                 "mesh_h_current_iso");

    this->postproc_handler->add_cell_data_vector(mesh_h_target_iso_cell,
                                                 "mesh_h_target_iso");

    this->postproc_handler->add_cell_data_vector(mesh_e_iso_cell,
                                                 "mesh_e_iso");

    this->postproc_handler->add_cell_data_vector(mesh_p_iso_cell,
                                                 "mesh_p_iso");

    this->postproc_handler->add_cell_data_vector(mesh_w_iso_cell,
                                                 "mesh_w_iso");

    this->postproc_handler->add_cell_data_vector(mesh_alpha_cell,
                                                 "mesh_alpha");

    this->postproc_handler->add_cell_data_vector(mesh_grad_abs_u_norm_cell,
                                                 "mesh_grad_abs_u_norm");

    this->postproc_handler->add_cell_data_vector(mesh_grad_abs_u_x_cell,
                                                 "mesh_grad_abs_u_x");

    if constexpr (dim >= 2)
      this->postproc_handler->add_cell_data_vector(mesh_grad_abs_u_y_cell,
                                                   "mesh_grad_abs_u_y");

    if constexpr (dim == 3)
      this->postproc_handler->add_cell_data_vector(mesh_grad_abs_u_z_cell,
                                                   "mesh_grad_abs_u_z");
  }
}

template <int dim>
void FSISolver<dim>::solver_specific_post_processing()
{
  if (this->param.mms_param.enable)
    if (this->param.debug.fsi_check_mms_on_boundary)
      check_manufactured_solution_boundary();

  // Check position - lambda coupling if coupled
  if (this->param.fsi.enable_coupling)
    compare_forces_and_position_on_obstacle();

  /**
   * Check that no-slip condition is satisfied.
   *
   * When applying the exact solution, the fluid velocity will be exact,
   * but the mesh velocity is only precise up to time integration order.
   * So these velocities differ by some power of the time step, rather
   * than the machine epsilon as checked in this function, thus the
   * no-slip is not checked in this case.
   *
   * Also, not checking when using BDF2 and starting with the initial
   * condition, as it will generally not respect the no-slip condition.
   */
  if (!this->param.debug.apply_exact_solution)
  {
    if (!(this->time_handler.is_starting_step() &&
          this->param.time_integration.bdfstart ==
            Parameters::TimeIntegration::BDFStart::initial_condition))
      check_velocity_boundary();
  }
}

// Explicit instantiation
template class FSISolver<2>;
template class FSISolver<3>;
