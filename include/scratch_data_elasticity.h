#ifndef SCRATCH_DATA_LINEAR_ELASTICITY_H
#define SCRATCH_DATA_LINEAR_ELASTICITY_H

#include <deal.II/base/quadrature.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <parameter_reader.h>
#include <types.h>

using namespace dealii;

/**
 * Small scratch data for the linear elasticity equation on fixed mesh.
 */
template <int dim>
class ScratchDataElasticity
{
public:
  /**
   * Constructor
   */
  ScratchDataElasticity(const FESystem<dim>        &fe,
                              const Mapping<dim>         &mapping,
                              const Quadrature<dim>      &cell_quadrature,
                              const Quadrature<dim - 1>  &face_quadrature,
                              const ParameterReader<dim> &param,
                              const bool evaluate_chns_forcing = false,
                              const bool with_enlarged_psi     = false)
    : param(param)
    , fe_values(mapping,
                fe,
                cell_quadrature,
                update_values | update_gradients | update_quadrature_points |
                  update_JxW_values)
    , evaluate_chns_forcing(evaluate_chns_forcing || with_enlarged_psi)
    , with_enlarged_psi(with_enlarged_psi)
    , n_q_points(cell_quadrature.size())
    , n_faces(fe.reference_cell().n_faces())
    , n_faces_q_points(face_quadrature.size())
    , dofs_per_cell(fe.dofs_per_cell)
  {
    position.first_vector_component = 0;
    if (with_enlarged_psi)
    {
      // The enlarged marker psi is the extra scalar field appended after the
      // mesh position (components 0..dim-1), so it sits at component dim.
      psi.component  = dim;
      position_mask  = fe.component_mask(position);
      epsilon        = param.cahn_hilliard.epsilon_interface;
      const double L = param.cahn_hilliard.psi_interface_width_factor * epsilon;
      psi_length_scale_sq = L * L;
    }
    allocate();
  }

  /**
   * Copy constructor
   */
  ScratchDataElasticity(const ScratchDataElasticity &other)
    : param(other.param)
    , fe_values(other.fe_values.get_mapping(),
                other.fe_values.get_fe(),
                other.fe_values.get_quadrature(),
                other.fe_values.get_update_flags())
    , evaluate_chns_forcing(other.evaluate_chns_forcing)
    , with_enlarged_psi(other.with_enlarged_psi)
    , n_q_points(other.n_q_points)
    , n_faces(other.n_faces)
    , n_faces_q_points(other.n_faces_q_points)
    , dofs_per_cell(other.dofs_per_cell)
    , source_term_fixed_mesh_multiplier(other.source_term_fixed_mesh_multiplier)
    , source_term_moving_mesh_multiplier(
        other.source_term_moving_mesh_multiplier)
    , chns_compression_multiplier(other.chns_compression_multiplier)
    , psi_length_scale_sq(other.psi_length_scale_sq)
    , epsilon(other.epsilon)
  {
    position.first_vector_component = 0;
    if (with_enlarged_psi)
    {
      psi.component = dim;
      position_mask = other.position_mask;
    }
    allocate();
  }

private:
  void allocate()
  {
    JxW_fixed.resize(n_q_points);
    lame_mu.resize(n_q_points);
    lame_lambda.resize(n_q_points);

    components.resize(dofs_per_cell);

    position_values.resize(n_q_points);
    present_position_gradients.resize(n_q_points);
    position_sym_gradients.resize(n_q_points);

    present_position_J.resize(n_q_points);
    present_position_inverse_gradients.resize(n_q_points);
    present_position_inverse_gradients_T.resize(n_q_points);

    phi_x.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_x.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    sym_grad_phi_x.resize(n_q_points,
                          std::vector<SymmetricTensor<2, dim>>(dofs_per_cell));
    div_phi_x.resize(n_q_points, std::vector<double>(dofs_per_cell));

    source_term_full.resize(n_q_points, Vector<double>(dim));
    source_term_position_fixed_mesh.resize(n_q_points);
    qpoints_current_mesh.resize(n_q_points);
    source_term_full_current_mesh.resize(n_q_points, Vector<double>(dim));
    source_term_position_current_mesh.resize(n_q_points);
    grad_source_term_full_current_mesh.resize(n_q_points,
                                              std::vector<Tensor<1, dim>>(dim));
    grad_source_term_position_current_mesh.resize(n_q_points);
    source_term_position.resize(n_q_points);

    if (evaluate_chns_forcing)
    {
      chns_tracer_values.resize(n_q_points);
      chns_tracer_gradients.resize(n_q_points);
      chns_tracer_hessians.resize(n_q_points);
    }

    if (with_enlarged_psi)
    {
      JxW_moving.resize(n_q_points);
      psi_values.resize(n_q_points);
      psi_gradients.resize(n_q_points);
      shape_psi.resize(n_q_points, std::vector<double>(dofs_per_cell));
      grad_shape_psi.resize(n_q_points,
                            std::vector<Tensor<1, dim>>(dofs_per_cell));
      grad_phi_x_moving.resize(n_q_points,
                               std::vector<Tensor<2, dim>>(dofs_per_cell));
    }
  }

public:
  template <typename VectorType>
  void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
              const VectorType                     &current_solution,
              const std::shared_ptr<Function<dim>> &source_terms,
              const std::shared_ptr<Function<dim>> & /*exact_solution*/)
  {
    fe_values.reinit(cell);

    for (const unsigned int i : fe_values.dof_indices())
      components[i] = fe_values.get_fe().system_to_component_index(i).first;

    /**
     * Volume contributions
     */
    fe_values[position].get_function_values(current_solution, position_values);
    fe_values[position].get_function_gradients(current_solution,
                                               present_position_gradients);
    fe_values[position].get_function_symmetric_gradients(
      current_solution, position_sym_gradients);

    const auto &quadrature_points = fe_values.get_quadrature_points();
    source_terms->vector_value_list(quadrature_points, source_term_full);

    // First map the quadrature point to the current configuration
    // The quadrature points in the current configuration are simply the
    // position field interpolated at the reference space quadrature points
    for (unsigned int q = 0; q < n_q_points; ++q)
      qpoints_current_mesh[q] = position_values[q];
    source_terms->vector_value_list(qpoints_current_mesh,
                                    source_term_full_current_mesh);

    // These are computed with finite differences by deal.II (AutoDerivative)
    source_terms->vector_gradient_list(qpoints_current_mesh,
                                       grad_source_term_full_current_mesh);

    // Cahn-Hilliard moving-mesh forcing: evaluate the prescribed phase and its
    // analytic gradient/Hessian on the current (deformed) mesh. The Hessian is
    // needed to linearize the forcing w.r.t. the mesh position.
    if (evaluate_chns_forcing)
    {
      const auto &tracer =
        *param.initial_conditions.initial_chns_tracer_callback;
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        chns_tracer_values[q]    = tracer.value(qpoints_current_mesh[q]);
        chns_tracer_gradients[q] = tracer.gradient(qpoints_current_mesh[q]);
        chns_tracer_hessians[q]  = tracer.hessian(qpoints_current_mesh[q]);
      }
    }

    // Enlarged presolver: build the psi finite-element field on the moving
    // (deformed) mesh, so the psi Helmholtz reconstruction is assembled on the
    // current configuration exactly as in the full CHNS solver. The physical
    // tracer phi stays analytic (chns_tracer_* above); only psi is an unknown.
    if (with_enlarged_psi)
    {
      const MappingFEField<dim, dim, VectorType> moving_mapping(
        cell->get_dof_handler(), current_solution, position_mask);
      FEValues<dim> fe_values_moving(moving_mapping,
                                     fe_values.get_fe(),
                                     fe_values.get_quadrature(),
                                     update_values | update_gradients |
                                       update_quadrature_points |
                                       update_JxW_values);
      fe_values_moving.reinit(cell);

      fe_values_moving[psi].get_function_values(current_solution, psi_values);
      fe_values_moving[psi].get_function_gradients(current_solution,
                                                   psi_gradients);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        JxW_moving[q] = fe_values_moving.JxW(q);
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          shape_psi[q][k]         = fe_values_moving[psi].value(k, q);
          grad_shape_psi[q][k]    = fe_values_moving[psi].gradient(k, q);
          grad_phi_x_moving[q][k] = fe_values_moving[position].gradient(k, q);
        }
      }
    }

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      JxW_fixed[q] = fe_values.JxW(q);

      for (unsigned int d = 0; d < dim; ++d)
      {
        source_term_position_fixed_mesh[q][d] = source_term_full[q](d);
        source_term_position_current_mesh[q][d] =
          source_term_full_current_mesh[q](d);

        // Premultiply the gradient by the coefficient in front of it
        for (unsigned int dj = 0; dj < dim; ++dj)
          grad_source_term_position_current_mesh[q][d][dj] =
            source_term_moving_mesh_multiplier *
            grad_source_term_full_current_mesh[q][d][dj];
      }

      // Source term effectively used
      // Multipliers cannot both be nonzero
      Assert(!(std::abs(source_term_fixed_mesh_multiplier) > 1e-14 &&
               std::abs(source_term_moving_mesh_multiplier) > 1e-14),
             ExcInternalError());

      source_term_position[q] =
        source_term_fixed_mesh_multiplier * source_term_position_fixed_mesh[q] +
        source_term_moving_mesh_multiplier *
          source_term_position_current_mesh[q];

      const Point<dim> &pt = quadrature_points[q];
      lame_mu[q] =
        param.physical_properties.pseudosolids[0].lame_mu_fun->value(pt);
      lame_lambda[q] =
        param.physical_properties.pseudosolids[0].lame_lambda_fun->value(pt);

      AssertThrow(lame_mu[q] >= 0,
                  ExcMessage("Lamé coefficient mu should be positive"));

      // Data for hyperelastic models
      const Tensor<2, dim> &F               = present_position_gradients[q];
      present_position_J[q]                 = determinant(F);
      present_position_inverse_gradients[q] = invert(F);
      present_position_inverse_gradients_T[q] =
        transpose(present_position_inverse_gradients[q]);

      if constexpr (running_in_debug_mode())
      {
        // Throw if
      }
      // AssertThrow(present_position_J[q] > 0, ExcMessage("Inverted"));

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_x[q][k]          = fe_values[position].value(k, q);
        grad_phi_x[q][k]     = fe_values[position].gradient(k, q);
        sym_grad_phi_x[q][k] = fe_values[position].symmetric_gradient(k, q);
        div_phi_x[q][k]      = fe_values[position].divergence(k, q);
      }
    }
  }

private:
  const ParameterReader<dim> &param;
  // Parameters::PhysicalProperties<dim> physical_properties;

  FEValues<dim> fe_values;

  // Whether to evaluate the Cahn-Hilliard phase (and its derivatives) on the
  // current mesh, for the moving-mesh forcing of the elasticity presolver.
  const bool evaluate_chns_forcing;

  // Component mask of the mesh position, used to build the moving mapping that
  // evaluates psi on the deformed configuration (enlarged presolver only).
  ComponentMask position_mask;

public:
  // Whether the presolver carries the enlarged marker psi as an extra finite-
  // element field (hybrid mode: analytic phi source, psi FE unknown). When set,
  // psi is reconstructed on the moving mesh and drives the enlarged forcing.
  const bool with_enlarged_psi;

  const unsigned int active_fe_index = 0;

  const unsigned int n_q_points;
  const unsigned int n_faces;
  const unsigned int n_faces_q_points;
  const unsigned int dofs_per_cell;

  std::vector<unsigned int> components;

  std::vector<double> JxW_fixed;
  std::vector<double> lame_mu;
  std::vector<double> lame_lambda;

  FEValuesExtractors::Vector position;

  std::vector<Tensor<1, dim>>          position_values;
  std::vector<Tensor<2, dim>>          present_position_gradients;
  std::vector<SymmetricTensor<2, dim>> position_sym_gradients;

  // Data for hyperelastic models
  std::vector<double>         present_position_J;                   // = det(F)
  std::vector<Tensor<2, dim>> present_position_inverse_gradients;   // = F^{-1}
  std::vector<Tensor<2, dim>> present_position_inverse_gradients_T; // = F^{-T}

  std::vector<std::vector<Tensor<1, dim>>>          phi_x;
  std::vector<std::vector<Tensor<2, dim>>>          grad_phi_x;
  std::vector<std::vector<SymmetricTensor<2, dim>>> sym_grad_phi_x;
  std::vector<std::vector<double>>                  div_phi_x;

  // Prescribed Cahn-Hilliard phase and its analytic derivatives, evaluated on
  // the current (deformed) mesh. Filled only when evaluate_chns_forcing.
  std::vector<double>                  chns_tracer_values;
  std::vector<Tensor<1, dim>>          chns_tracer_gradients;
  std::vector<SymmetricTensor<2, dim>> chns_tracer_hessians;

  // Enlarged marker psi as a finite-element field on the moving mesh, and the
  // moving-mesh quantities needed to assemble its Helmholtz reconstruction and
  // the enlarged forcing. Filled only when with_enlarged_psi. The extractor
  // psi sits at component dim (right after the mesh position).
  FEValuesExtractors::Scalar               psi;
  std::vector<double>                      JxW_moving;
  std::vector<double>                      psi_values;
  std::vector<Tensor<1, dim>>              psi_gradients;
  std::vector<std::vector<double>>         shape_psi;
  std::vector<std::vector<Tensor<1, dim>>> grad_shape_psi;
  // Spatial gradient of the mesh-position shape functions on the moving mesh
  // (G_k = grad_x N_k), used for the ALE x-variation of the psi residual.
  std::vector<std::vector<Tensor<2, dim>>> grad_phi_x_moving;

  // Enlarged length scale squared L^2 = (psi_interface_width_factor * eps)^2
  // and the interface thickness eps, mirrored from the Cahn-Hilliard params.
  double psi_length_scale_sq = 0.;
  double epsilon             = 0.;

  /**
   * Evaluation of the given source term on the fixed mesh, that is, of f(X).
   */
  std::vector<Vector<double>> source_term_full;
  std::vector<Tensor<1, dim>> source_term_position_fixed_mesh;

  /**
   * Evaluation of the given source term on the current mesh, that is, of
   * f(x(X)). This is useful when one has access to the source term on the
   * current configuration only, whereas the linear elasticity equation is
   * solved on the reference configuration.
   */
  std::vector<Point<dim>>                  qpoints_current_mesh;
  std::vector<Vector<double>>              source_term_full_current_mesh;
  std::vector<Tensor<1, dim>>              source_term_position_current_mesh;
  std::vector<std::vector<Tensor<1, dim>>> grad_source_term_full_current_mesh;
  std::vector<Tensor<2, dim>> grad_source_term_position_current_mesh;

  /**
   * The source term that is effectively used, defined by
   *
   * a * source_term_position_fixed_mesh + b *
   * source_term_position_current_mesh,
   *
   * with a = source_term_fixed_mesh_multiplier and
   *      b = source_term_moving_mesh_multiplier.
   */
  double                      source_term_fixed_mesh_multiplier;
  double                      source_term_moving_mesh_multiplier;

  // Continuation multiplier ramping the Cahn-Hilliard compression forcing from
  // a small fraction up to its physical value during the presolve.
  double chns_compression_multiplier = 1.;

  std::vector<Tensor<1, dim>> source_term_position;
};

#endif
