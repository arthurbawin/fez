#ifndef MESH_FORCING_POSTPROCESSING_H
#define MESH_FORCING_POSTPROCESSING_H

#include <assembly/moving_mesh_forcing_forms.h>
#include <deal.II/fe/fe_values.h>
#include <post_processing_handler.h>
#include <time_handler.h>

using namespace dealii;

namespace MeshForcingPostProcessing
{
  template <int dim>
  struct CellDiagnostics
  {
    Tensor<1, dim> enlarged_alpha_components;
    Tensor<1, dim> physics_alpha_components;
    Tensor<1, dim> beta_components;
  };

  template <int dim>
  struct OutputFields
  {
    using CellIterator = typename DoFHandler<dim>::active_cell_iterator;

    OutputFields(const Triangulation<dim> &triangulation, const bool use_quads)
      : enlarged_alpha(std::make_unique<PostProcessingTools::DG0DataField<dim>>(
          triangulation,
          use_quads,
          PostProcessingTools::make_vector_component_names<dim>(
            "mesh_force_enlarged_alpha"),
          PostProcessingTools::make_vector_component_interpretation<dim>()))
      , physics_alpha(std::make_unique<PostProcessingTools::DG0DataField<dim>>(
          triangulation,
          use_quads,
          PostProcessingTools::make_vector_component_names<dim>(
            "mesh_force_physics_alpha"),
          PostProcessingTools::make_vector_component_interpretation<dim>()))
      , beta(std::make_unique<PostProcessingTools::DG0DataField<dim>>(
          triangulation,
          use_quads,
          PostProcessingTools::make_vector_component_names<dim>(
            "mesh_force_beta"),
          PostProcessingTools::make_vector_component_interpretation<dim>()))
    {}

    void store(const CellIterator         &cell,
               const CellDiagnostics<dim> &diagnostics)
    {
      enlarged_alpha->set_cell_values(cell,
                                      diagnostics.enlarged_alpha_components);
      physics_alpha->set_cell_values(cell,
                                     diagnostics.physics_alpha_components);
      beta->set_cell_values(cell, diagnostics.beta_components);
    }

    void write(PostProcessingHandler<dim> &postproc_handler)
    {
      postproc_handler.add_cell_dg0_data_field(std::move(enlarged_alpha));
      postproc_handler.add_cell_dg0_data_field(std::move(physics_alpha));
      postproc_handler.add_cell_dg0_data_field(std::move(beta));
    }

    std::unique_ptr<PostProcessingTools::DG0DataField<dim>> enlarged_alpha;
    std::unique_ptr<PostProcessingTools::DG0DataField<dim>> physics_alpha;
    std::unique_ptr<PostProcessingTools::DG0DataField<dim>> beta;
  };

  template <int dim, typename VectorType>
  void fill_marker_fields(const FEValues<dim>              &fe_values_moving,
                          const FEValuesExtractors::Scalar &marker_extractor,
                          const VectorType                 &present_solution,
                          std::vector<double>             &marker_values,
                          std::vector<Tensor<1, dim>>     &marker_gradients)
  {
    fe_values_moving[marker_extractor].get_function_values(present_solution,
                                                           marker_values);
    fe_values_moving[marker_extractor].get_function_gradients(present_solution,
                                                              marker_gradients);
  }

  template <int dim, bool with_enlarged>
  CellDiagnostics<dim> compute_cell_diagnostics(
    const FEValues<dim>                                   &fe_values_fixed,
    const std::vector<Tensor<1, dim>>                     &velocity_values,
    const std::vector<Tensor<1, dim>>                     &position_values,
    const std::vector<std::vector<Tensor<1, dim>>>        &previous_position_values,
    const std::vector<double>                             &enlarged_values,
    const std::vector<Tensor<1, dim>>                     &enlarged_gradients,
    const std::vector<double>                             &tracer_values,
    const std::vector<Tensor<1, dim>>                     &tracer_gradients,
    const TimeHandler                                     &time_handler,
    const Parameters::CahnHilliard<dim>                   &cahn_hilliard_param,
    const double                                           enlarged_forcing_epsilon)
  {
    CellDiagnostics<dim> diagnostics;
    double               cell_measure = 0.;

    for (unsigned int q = 0; q < velocity_values.size(); ++q)
    {
      const Tensor<1, dim> mesh_velocity =
        time_handler.template compute_time_derivative_at_quadrature_node<dim>(
          q, position_values[q], previous_position_values);
      const Tensor<1, dim> u_conv = velocity_values[q] - mesh_velocity;
      double enlarged_factor          = 0.;
      double enlarged_factor_jacobian = 0.;
      Assembly::MovingMeshForcing::mesh_forcing_factor_and_jacobian(
        cahn_hilliard_param,
        enlarged_values[q],
        enlarged_factor,
        enlarged_factor_jacobian);
      double tracer_factor          = 0.;
      double tracer_factor_jacobian = 0.;
      Assembly::MovingMeshForcing::mesh_forcing_factor_and_jacobian(
        cahn_hilliard_param,
        tracer_values[q],
        tracer_factor,
        tracer_factor_jacobian);

      (void)enlarged_factor_jacobian;
      (void)tracer_factor_jacobian;

      Tensor<1, dim> enlarged_alpha;
      if constexpr (with_enlarged)
        enlarged_alpha =
          cahn_hilliard_param.mff_enlarged_compression_factor *
          enlarged_forcing_epsilon * enlarged_factor * enlarged_gradients[q];
      const Tensor<1, dim> enlarged_beta =
        cahn_hilliard_param.mff_transport_factor *
        (enlarged_forcing_epsilon * enlarged_forcing_epsilon) *
        ((u_conv * enlarged_gradients[q]) * enlarged_gradients[q]);
      const Tensor<1, dim> physics_alpha =
        cahn_hilliard_param.mff_physics_compression_factor *
        cahn_hilliard_param.epsilon_interface * tracer_factor *
        tracer_gradients[q];
      const double weight = fe_values_fixed.JxW(q);

      diagnostics.enlarged_alpha_components += enlarged_alpha * weight;
      diagnostics.physics_alpha_components += physics_alpha * weight;
      diagnostics.beta_components += enlarged_beta * weight;
      cell_measure += weight;
    }

    if (cell_measure > 0.)
    {
      diagnostics.enlarged_alpha_components /= cell_measure;
      diagnostics.physics_alpha_components /= cell_measure;
      diagnostics.beta_components /= cell_measure;
    }

    return diagnostics;
  }

  template <int dim, bool with_enlarged, typename VectorType>
  void export_diagnostics(
    const Mapping<dim>                                 &moving_mapping,
    const Mapping<dim>                                 &fixed_mapping,
    const FESystem<dim>                                &fe_system,
    const Quadrature<dim>                              &quadrature,
    const DoFHandler<dim>                              &dof_handler,
    const FEValuesExtractors::Vector                   &velocity_extractor,
    const FEValuesExtractors::Vector                   &position_extractor,
    const FEValuesExtractors::Scalar                   &tracer_extractor,
    const FEValuesExtractors::Scalar                   &psi_extractor,
    const VectorType                                   &present_solution,
    const std::vector<VectorType>                      &previous_solutions,
    const TimeHandler                                  &time_handler,
    const Parameters::CahnHilliard<dim>                &cahn_hilliard_param,
    PostProcessingHandler<dim>                         &postproc_handler)
  {
    const bool use_quads = fe_system.reference_cell().is_hyper_cube();
    OutputFields<dim> output_fields(dof_handler.get_triangulation(), use_quads);

    FEValues<dim> fe_values_moving(moving_mapping,
                                   fe_system,
                                   quadrature,
                                   update_values | update_gradients);
    FEValues<dim> fe_values_fixed(fixed_mapping,
                                  fe_system,
                                  quadrature,
                                  update_values | update_JxW_values);

    const unsigned int n_q_points = quadrature.size();
    std::vector<Tensor<1, dim>> velocity_values(n_q_points);
    std::vector<Tensor<1, dim>> position_values(n_q_points);
    std::vector<std::vector<Tensor<1, dim>>> previous_position_values(
      previous_solutions.size(),
      std::vector<Tensor<1, dim>>(n_q_points));
    std::vector<double>         enlarged_values(n_q_points);
    std::vector<Tensor<1, dim>> enlarged_gradients(n_q_points);
    std::vector<double>         tracer_values(n_q_points);
    std::vector<Tensor<1, dim>> tracer_gradients(n_q_points);

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

        fill_marker_fields(fe_values_moving,
                           with_enlarged ? psi_extractor : tracer_extractor,
                           present_solution,
                           enlarged_values,
                           enlarged_gradients);
        fill_marker_fields(fe_values_moving,
                           tracer_extractor,
                           present_solution,
                           tracer_values,
                           tracer_gradients);

        output_fields.store(cell,
                            compute_cell_diagnostics<dim, with_enlarged>(
                              fe_values_fixed,
                              velocity_values,
                              position_values,
                              previous_position_values,
                              enlarged_values,
                              enlarged_gradients,
                              tracer_values,
                              tracer_gradients,
                              time_handler,
                              cahn_hilliard_param,
                              with_enlarged ?
                                cahn_hilliard_param
                                  .epsilon_interface_enlarged :
                                cahn_hilliard_param.epsilon_interface));
      }

    output_fields.write(postproc_handler);
  }
} // namespace MeshForcingPostProcessing

#endif
