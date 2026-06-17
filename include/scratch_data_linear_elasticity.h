#ifndef SCRATCH_DATA_LINEAR_ELASTICITY_H
#define SCRATCH_DATA_LINEAR_ELASTICITY_H

#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <parameter_reader.h>
#include <types.h>

#include <cmath>
#include <sstream>

using namespace dealii;

/**
 * Small scratch data for the linear elasticity equation on fixed mesh.
 */
template <int dim>
class ScratchDataLinearElasticity
{
public:
  /**
   * Constructor
   */
  ScratchDataLinearElasticity(const FESystem<dim>        &fe,
                              const Mapping<dim>         &mapping,
                              const DoFHandler<dim>      &dof_handler,
                              const ComponentMask        &position_mask,
                              const Quadrature<dim>      &cell_quadrature,
                              const Quadrature<dim - 1>  &face_quadrature,
                              const ParameterReader<dim> &param,
                              const bool presolve_chns_marker = false,
                              const bool presolve_enlarged = false)
    : physical_properties(param.physical_properties)
    , param(param)
    , dof_handler(dof_handler)
    , position_mask(position_mask)
    , fe(fe)
    , cell_quadrature(cell_quadrature)
    , fe_values(mapping,
                fe,
                cell_quadrature,
                update_values | update_gradients | update_quadrature_points |
                  update_JxW_values)
    , presolve_chns_marker(presolve_chns_marker)
    , presolve_enlarged(presolve_enlarged)
    , n_q_points(cell_quadrature.size())
    , n_faces(fe.reference_cell().n_faces())
    , n_faces_q_points(face_quadrature.size())
    , dofs_per_cell(fe.dofs_per_cell)
  {
    position.first_vector_component = 0;
    if (presolve_chns_marker)
    {
      tracer.component = dim;
      epsilon = param.cahn_hilliard.epsilon_interface;
      sigma_tilde =
        3. / (2. * std::sqrt(2.)) * param.cahn_hilliard.surface_tension;
    }
    if (presolve_enlarged)
      enlarged.component = dim + 1;
    allocate();
  }

  /**
   * Copy constructor
   */
  ScratchDataLinearElasticity(const ScratchDataLinearElasticity &other)
    : physical_properties(other.physical_properties)
    , param(other.param)
    , dof_handler(other.dof_handler)
    , position_mask(other.position_mask)
    , fe(other.fe)
    , cell_quadrature(other.cell_quadrature)
    , fe_values(other.fe_values.get_mapping(),
                other.fe_values.get_fe(),
                other.fe_values.get_quadrature(),
                other.fe_values.get_update_flags())
    , presolve_chns_marker(other.presolve_chns_marker)
    , presolve_enlarged(other.presolve_enlarged)
    , n_q_points(other.n_q_points)
    , n_faces(other.n_faces)
    , n_faces_q_points(other.n_faces_q_points)
    , dofs_per_cell(other.dofs_per_cell)
  {
    position.first_vector_component = 0;
    if (presolve_chns_marker)
    {
      tracer.component = dim;
      epsilon = other.epsilon;
      sigma_tilde = other.sigma_tilde;
    }
    if (presolve_enlarged)
      enlarged.component = dim + 1;
    allocate();
  }

private:
  void allocate()
  {
    JxW.resize(n_q_points);
    lame_mu.resize(n_q_points);
    lame_lambda.resize(n_q_points);

    position_values.resize(n_q_points);
    position_sym_gradients.resize(n_q_points);
    position_gradients.resize(n_q_points);
    position_strains.resize(n_q_points);
    position_trace_strains.resize(n_q_points);

    // neo hookean
    position_J.resize(n_q_points);
    position_inv_gradients.resize(n_q_points);
    position_inv_gradients_T.resize(n_q_points);

    phi_x.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_x.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_x.resize(n_q_points, std::vector<double>(dofs_per_cell));

    source_term_full.resize(n_q_points, Vector<double>(dim));
    source_term_position.resize(n_q_points);
    qpoints_current_mesh.resize(n_q_points);
    source_term_full_current_mesh.resize(n_q_points, Vector<double>(dim));
    source_term_position_current_mesh.resize(n_q_points);
    grad_source_term_full_current_mesh.resize(n_q_points,
                                              std::vector<Tensor<1, dim>>(dim));
    grad_source_term_position_current_mesh.resize(n_q_points);

    components.resize(dofs_per_cell);
    for (unsigned int k = 0; k < dofs_per_cell; ++k)
      components[k] = fe.system_to_component_index(k).first;

    if (presolve_chns_marker)
    {
      JxW_fixed.resize(n_q_points);
      JxW_moving.resize(n_q_points);
      tracer_values.resize(n_q_points);
      tracer_gradients.resize(n_q_points);
      analytic_tracer_values.resize(n_q_points);
      analytic_tracer_gradients.resize(n_q_points);
      velocity_dot_tracer_gradient.assign(n_q_points, 0.);
      present_convective_velocity.assign(n_q_points, Tensor<1, dim>());
      phi_u.resize(n_q_points,
                   std::vector<Tensor<1, dim>>(dofs_per_cell,
                                               Tensor<1, dim>()));
      shape_phi.resize(n_q_points, std::vector<double>(dofs_per_cell));
      grad_shape_phi.resize(n_q_points,
                            std::vector<Tensor<1, dim>>(dofs_per_cell));
      grad_phi_x_moving.resize(n_q_points,
                               std::vector<Tensor<2, dim>>(dofs_per_cell));
      potential_values.assign(n_q_points, 0.);
      shape_mu.resize(n_q_points, std::vector<double>(dofs_per_cell, 0.));
      source_term_psi.assign(n_q_points, 0.);

      if (presolve_enlarged)
      {
        psi_values.resize(n_q_points);
        psi_gradients.resize(n_q_points);
        shape_psi.resize(n_q_points, std::vector<double>(dofs_per_cell));
        grad_shape_psi.resize(n_q_points,
                              std::vector<Tensor<1, dim>>(dofs_per_cell));
      }
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

    /**
     * Volume contributions
     */
    fe_values[position].get_function_values(current_solution, position_values);
    fe_values[position].get_function_symmetric_gradients(
      current_solution, position_sym_gradients);
    fe_values[position].get_function_gradients(current_solution,
                                               position_gradients);

    const auto &quadrature_points = fe_values.get_quadrature_points();
    source_terms->vector_value_list(quadrature_points, source_term_full);

    const SymmetricTensor<2, dim> identity_tensor =
      unit_symmetric_tensor<dim>();

    // First map the quadrature point to the current configuration
    // The quadrature points in the current configuration are simply the
    // position field interpolated at the reference space quadrature points
    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      qpoints_current_mesh[q] = position_values[q];
    }
    source_terms->vector_value_list(qpoints_current_mesh,
                                    source_term_full_current_mesh);

    // These are computed with finite differences by deal.II (AutoDerivative)
    source_terms->vector_gradient_list(qpoints_current_mesh,
                                       grad_source_term_full_current_mesh);

    std::unique_ptr<FEValues<dim>> fe_values_moving;
    if (presolve_chns_marker)
    {
      const MappingFEField<dim, dim, VectorType> moving_mapping(dof_handler,
                                                                current_solution,
                                                                position_mask);
      fe_values_moving = std::make_unique<FEValues<dim>>(
        moving_mapping,
        fe,
        cell_quadrature,
        update_values | update_gradients | update_quadrature_points |
          update_JxW_values);
      fe_values_moving->reinit(cell);

      (*fe_values_moving)[tracer].get_function_values(current_solution,
                                                       tracer_values);
      (*fe_values_moving)[tracer].get_function_gradients(current_solution,
                                                         tracer_gradients);
      if (presolve_enlarged)
      {
        (*fe_values_moving)[enlarged].get_function_values(current_solution,
                                                          psi_values);
        (*fe_values_moving)[enlarged].get_function_gradients(current_solution,
                                                             psi_gradients);
      }
    }

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      JxW[q]                    = fe_values.JxW(q);
      if (presolve_chns_marker)
      {
        JxW_fixed[q]  = fe_values.JxW(q);
        JxW_moving[q] = fe_values_moving->JxW(q);
        analytic_tracer_values[q] =
          param.initial_conditions.initial_chns_tracer_callback->value(
            qpoints_current_mesh[q]);
        analytic_tracer_gradients[q] =
          param.initial_conditions.initial_chns_tracer_callback->gradient(
            qpoints_current_mesh[q]);
      }
      position_strains[q]       = position_sym_gradients[q] - identity_tensor;
      position_trace_strains[q] = trace(position_strains[q]);

      for (unsigned int d = 0; d < dim; ++d)
      {
        source_term_position[q][d] = source_term_full[q](d);
        source_term_position_current_mesh[q][d] =
          source_term_full_current_mesh[q](d);
        for (unsigned int dj = 0; dj < dim; ++dj)
          grad_source_term_position_current_mesh[q][d][dj] =
            grad_source_term_full_current_mesh[q][d][dj];
      }

      const Point<dim> &pt = quadrature_points[q];
      lame_mu[q] = physical_properties.pseudosolids[0].lame_mu_fun->value(pt);
      lame_lambda[q] =
        physical_properties.pseudosolids[0].lame_lambda_fun->value(pt);

      AssertThrow(lame_mu[q] >= 0,
                  ExcMessage("Lamé coefficient mu should be positive"));

      // neo-hookean
      const Tensor<2, dim> &F = position_gradients[q];
      position_J[q]           = determinant(F);

      position_inv_gradients[q]   = invert(F);
      position_inv_gradients_T[q] = transpose(position_inv_gradients[q]);

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_x[q][k]      = fe_values[position].value(k, q);
        grad_phi_x[q][k] = fe_values[position].gradient(k, q);
        div_phi_x[q][k]  = fe_values[position].divergence(k, q);

        if (presolve_chns_marker)
        {
          shape_phi[q][k]      = (*fe_values_moving)[tracer].value(k, q);
          grad_shape_phi[q][k] = (*fe_values_moving)[tracer].gradient(k, q);
          grad_phi_x_moving[q][k] =
            (*fe_values_moving)[position].gradient(k, q);
          if (presolve_enlarged)
          {
            shape_psi[q][k] = (*fe_values_moving)[enlarged].value(k, q);
            grad_shape_psi[q][k] =
              (*fe_values_moving)[enlarged].gradient(k, q);
          }
        }
      }
    }
  }

private:
  Parameters::PhysicalProperties<dim> physical_properties;
  const ParameterReader<dim>          &param;
  const DoFHandler<dim>               &dof_handler;
  const ComponentMask                 &position_mask;
  const FESystem<dim>                 &fe;
  const Quadrature<dim>               &cell_quadrature;

  FEValues<dim> fe_values;
  bool          presolve_chns_marker;
  bool          presolve_enlarged;

public:
  const unsigned int n_q_points;
  const unsigned int n_faces;
  const unsigned int n_faces_q_points;
  const unsigned int dofs_per_cell;

  std::vector<double> JxW;
  std::vector<double> lame_mu;
  std::vector<double> lame_lambda;

  FEValuesExtractors::Vector position;
  FEValuesExtractors::Scalar tracer;
  FEValuesExtractors::Scalar enlarged;

  std::vector<Tensor<1, dim>>          position_values;
  std::vector<SymmetricTensor<2, dim>> position_sym_gradients;
  std::vector<Tensor<2, dim>>          position_gradients;
  std::vector<SymmetricTensor<2, dim>> position_strains;
  std::vector<double>                  position_trace_strains;

  // neo-Hookean
  std::vector<double>         position_J;
  std::vector<Tensor<2, dim>> position_inv_gradients;
  std::vector<Tensor<2, dim>> position_inv_gradients_T;

  std::vector<std::vector<Tensor<1, dim>>> phi_x;
  std::vector<std::vector<Tensor<2, dim>>> grad_phi_x;
  std::vector<std::vector<double>>         div_phi_x;

  std::vector<Vector<double>> source_term_full;
  std::vector<Tensor<1, dim>> source_term_position;

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

  std::vector<unsigned int> components;
  std::vector<double>       JxW_fixed;
  std::vector<double>       JxW_moving;

  std::vector<double>         tracer_values;
  std::vector<Tensor<1, dim>> tracer_gradients;
  std::vector<double>         analytic_tracer_values;
  std::vector<Tensor<1, dim>> analytic_tracer_gradients;
  std::vector<double>         psi_values;
  std::vector<Tensor<1, dim>> psi_gradients;
  std::vector<double>         potential_values;
  std::vector<double>         source_term_psi;
  std::vector<double>         velocity_dot_tracer_gradient;
  std::vector<Tensor<1, dim>> present_convective_velocity;
  std::vector<std::vector<Tensor<1, dim>>> phi_u;

  std::vector<std::vector<double>>         shape_phi;
  std::vector<std::vector<Tensor<1, dim>>> grad_shape_phi;
  std::vector<std::vector<double>>         shape_mu;
  std::vector<std::vector<double>>         shape_psi;
  std::vector<std::vector<Tensor<1, dim>>> grad_shape_psi;
  std::vector<std::vector<Tensor<2, dim>>> grad_phi_x_moving;

  double epsilon     = 0.;
  double sigma_tilde = 0.;
};

#endif
