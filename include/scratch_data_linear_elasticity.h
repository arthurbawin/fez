#ifndef SCRATCH_DATA_LINEAR_ELASTICITY_H
#define SCRATCH_DATA_LINEAR_ELASTICITY_H

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
                              const Quadrature<dim>      &cell_quadrature,
                              const Quadrature<dim - 1>  &face_quadrature,
                              const ParameterReader<dim> &param)
    : physical_properties(param.physical_properties)
    , fe_values(mapping,
                fe,
                cell_quadrature,
                update_values | update_gradients | update_quadrature_points |
                  update_JxW_values)
    , n_q_points(cell_quadrature.size())
    , n_faces(fe.reference_cell().n_faces())
    , n_faces_q_points(face_quadrature.size())
    , dofs_per_cell(fe.dofs_per_cell)
  {
    position.first_vector_component = 0;
    allocate();
  }

  /**
   * Copy constructor
   */
  ScratchDataLinearElasticity(const ScratchDataLinearElasticity &other)
    : physical_properties(other.physical_properties)
    , fe_values(other.fe_values.get_mapping(),
                other.fe_values.get_fe(),
                other.fe_values.get_quadrature(),
                other.fe_values.get_update_flags())
    , n_q_points(other.n_q_points)
    , n_faces(other.n_faces)
    , n_faces_q_points(other.n_faces_q_points)
    , dofs_per_cell(other.dofs_per_cell)
  {
    position.first_vector_component = 0;
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

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      JxW[q] = fe_values.JxW(q);

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
      AssertThrow(lame_lambda[q] >= 0,
                  ExcMessage("Lamé coefficient lambda should be positive"));

      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_x[q][k]      = fe_values[position].value(k, q);
        grad_phi_x[q][k] = fe_values[position].gradient(k, q);
        div_phi_x[q][k]  = fe_values[position].divergence(k, q);
      }
    }
  }

private:
  Parameters::PhysicalProperties<dim> physical_properties;

  FEValues<dim> fe_values;

public:
  const unsigned int n_q_points;
  const unsigned int n_faces;
  const unsigned int n_faces_q_points;
  const unsigned int dofs_per_cell;

  std::vector<double> JxW;
  std::vector<double> lame_mu;
  std::vector<double> lame_lambda;

  FEValuesExtractors::Vector position;

  std::vector<Tensor<1, dim>>          position_values;
  std::vector<SymmetricTensor<2, dim>> position_sym_gradients;

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
};

#endif
