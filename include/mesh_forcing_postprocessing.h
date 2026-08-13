#ifndef MESH_FORCING_POSTPROCESSING_H
#define MESH_FORCING_POSTPROCESSING_H

#include <assembly/elasticity_assemblers.h>
#include <deal.II/fe/fe_values.h>
#include <post_processing_handler.h>
#include <time_handler.h>

#include <algorithm>

using namespace dealii;

/**
 * Post-processing utilities for the three CHNS-ALE mesh-forcing terms.
 *
 * The pseudosolid forcing assembled in SourceFromCHNSTracerAssembler is
 *
 *   f_mesh = alpha_enlarged + alpha_physics - beta,
 *
 * where alpha_enlarged is zero for the non-enlarged solver.  Keeping beta with
 * a positive mff_transport_factor matches the historical diagnostic convention
 * and makes the sign in the decomposition explicit.
 *
 * Unlike the original implementation on chns-ding-horriche-model-switch,
 * which exported cell averages in DG0, these diagnostics are sampled at the
 * support points of a continuous auxiliary finite-element field.  This makes
 * their VTU representation consistent with the other derived CHNS fields on
 * the current branch.
 */
namespace MeshForcingPostProcessing
{
  template <int dim, bool with_enlarged, typename VectorType>
  void add_continuous_diagnostics(
    const Mapping<dim>                  &moving_mapping,
    const Mapping<dim>                  &fixed_mapping,
    const FESystem<dim>                 &fe_system,
    const DoFHandler<dim>               &dof_handler,
    const FEValuesExtractors::Vector    &velocity_extractor,
    const FEValuesExtractors::Vector    &position_extractor,
    const FEValuesExtractors::Scalar    &tracer_extractor,
    const FEValuesExtractors::Scalar    &psi_extractor,
    const VectorType                    &present_solution,
    const std::vector<VectorType>       &previous_solutions,
    const TimeHandler                   &time_handler,
    const Parameters::CahnHilliard<dim> &cahn_hilliard,
    const unsigned int                   output_degree,
    PostProcessingHandler<dim>          &postproc_handler)
  {
    std::vector<std::string> component_names;
    component_names.reserve(3 * dim);
    for (unsigned int d = 0; d < dim; ++d)
      component_names.emplace_back("mff_enlarged_compression");
    for (unsigned int d = 0; d < dim; ++d)
      component_names.emplace_back("mff_physics_compression");
    for (unsigned int d = 0; d < dim; ++d)
      component_names.emplace_back("mff_transport");

    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      component_interpretation(
        3 * dim, DataComponentInterpretation::component_is_part_of_vector);

    const bool use_quads = fe_system.reference_cell().is_hyper_cube();
    auto       output_field =
      std::make_unique<PostProcessingTools::ContinuousDataField<dim>>(
        dof_handler.get_triangulation(),
        use_quads,
        std::max(1u, output_degree),
        component_names,
        component_interpretation);

    const Quadrature<dim> output_points(
      output_field->get_unit_support_points());
    FEValues<dim> fe_values_moving(moving_mapping,
                                   fe_system,
                                   output_points,
                                   update_values | update_gradients);
    FEValues<dim> fe_values_fixed(fixed_mapping,
                                  fe_system,
                                  output_points,
                                  update_values);

    const unsigned int          n_output_points = output_points.size();
    std::vector<Tensor<1, dim>> velocity_values(n_output_points);
    std::vector<Tensor<1, dim>> position_values(n_output_points);
    std::vector<std::vector<Tensor<1, dim>>> previous_position_values(
      previous_solutions.size(), std::vector<Tensor<1, dim>>(n_output_points));
    std::vector<double>         marker_values(n_output_points);
    std::vector<Tensor<1, dim>> marker_gradients(n_output_points);
    std::vector<double>         tracer_values(n_output_points);
    std::vector<Tensor<1, dim>> tracer_gradients(n_output_points);

    const double epsilon = cahn_hilliard.epsilon_interface;
    const double gamma   = cahn_hilliard.mff_regularization_gamma;
    const double enlarged_normalization =
      Assembly::Elasticity::enlarged_mesh_forcing_normalization(
        gamma, cahn_hilliard.mff_enlarged_lobe_position_exponent);

    for (const auto &cell : dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        fe_values_moving.reinit(cell);
        fe_values_fixed.reinit(cell);

        fe_values_moving[velocity_extractor].get_function_values(
          present_solution, velocity_values);
        fe_values_fixed[position_extractor].get_function_values(
          present_solution, position_values);
        for (unsigned int i = 0; i < previous_solutions.size(); ++i)
          fe_values_fixed[position_extractor].get_function_values(
            previous_solutions[i], previous_position_values[i]);

        if constexpr (with_enlarged)
        {
          fe_values_moving[psi_extractor].get_function_values(present_solution,
                                                              marker_values);
          fe_values_moving[psi_extractor].get_function_gradients(
            present_solution, marker_gradients);
        }
        else
        {
          fe_values_moving[tracer_extractor].get_function_values(
            present_solution, marker_values);
          fe_values_moving[tracer_extractor].get_function_gradients(
            present_solution, marker_gradients);
        }
        fe_values_moving[tracer_extractor].get_function_values(present_solution,
                                                               tracer_values);
        fe_values_moving[tracer_extractor].get_function_gradients(
          present_solution, tracer_gradients);

        std::vector<std::vector<double>> values(n_output_points,
                                                std::vector<double>(3 * dim));
        for (unsigned int q = 0; q < n_output_points; ++q)
        {
          const Tensor<1, dim> mesh_velocity =
            time_handler
              .template compute_time_derivative_at_quadrature_node<dim>(
                q, position_values[q], previous_position_values);
          const Tensor<1, dim> convective_velocity =
            velocity_values[q] - mesh_velocity;

          const auto tracer_factor =
            Assembly::Elasticity::mesh_forcing_factor(tracer_values[q], gamma);
          const Tensor<1, dim> physics_alpha =
            cahn_hilliard.mff_physics_compression_factor * epsilon *
            tracer_factor.value * tracer_gradients[q];

          double         marker_epsilon = epsilon;
          Tensor<1, dim> enlarged_alpha;
          if constexpr (with_enlarged)
          {
            marker_epsilon = cahn_hilliard.psi_interface_width_factor * epsilon;
            const auto marker_factor =
              Assembly::Elasticity::enlarged_mesh_forcing_factor(
                marker_values[q],
                gamma,
                cahn_hilliard.mff_enlarged_lobe_position_exponent,
                enlarged_normalization);
            enlarged_alpha = cahn_hilliard.mff_enlarged_compression_factor *
                             marker_epsilon * marker_factor.value *
                             marker_gradients[q];
          }

          const Tensor<1, dim> beta =
            cahn_hilliard.mff_transport_factor * marker_epsilon *
            marker_epsilon *
            ((convective_velocity * marker_gradients[q]) * marker_gradients[q]);

          for (unsigned int d = 0; d < dim; ++d)
          {
            values[q][d]           = enlarged_alpha[d];
            values[q][dim + d]     = physics_alpha[d];
            values[q][2 * dim + d] = beta[d];
          }
        }

        output_field->set_cell_values(cell, values);
      }

    postproc_handler.add_continuous_data_field(std::move(output_field));
  }
} // namespace MeshForcingPostProcessing

#endif
