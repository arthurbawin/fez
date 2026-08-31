#ifndef POSTPROCESSORS_AND_EVALUATORS_H
#define POSTPROCESSORS_AND_EVALUATORS_H

#include <assembly/elasticity_assemblers.h>
#include <cahn_hilliard.h>
#include <components_ordering.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <field_postprocessors.h>
#include <parameter_reader.h>
#include <parameters.h>
#include <types.h>
#include <utilities.h>

#include <type_traits>

namespace PostProcessingTools
{
  using namespace dealii;

  /**
   * A DataPostProcessor to compute the vorticity. No particular treatment is
   * applied to the vorticity field, thus if the velocity is continuous, the
   * output vorticity will be discontinuous.
   */
  template <int dim>
  class VorticityPostProcessor : public DataPostprocessorVector<dim>
  {
  public:
    /**
     * Constructor.
     */
    VorticityPostProcessor(const ComponentOrdering &ordering);

    /**
     * Evaluate the vorticity as curl(u).
     */
    virtual void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override;

  private:
    const unsigned int u_lower;
  };

  /**
   * An Evaluator to compute the vorticity at quadrature nodes, store some
   * data needed to perform these computations (e.g., required update flags in
   * the main solver's FEValues, etc.), and specify the types involved.
   *
   * This class thus does a similar job as VorticityPostProcessor above, but
   * with a few more data due to the fact that we're computing and outputting
   * the data ourselves, instead of using the DataPostprocessor facilities.
   */
  template <int dim>
  class VorticityEvaluator
  {
  public:
    /**
     * Some bookkeeping data.
     *
     * The vorticity (curl of velocity) is a scalar field in 2D, and a vector
     * field in 3D. In deal.II, this is represented by a curl_type, which is
     * always a Tensor<1, dim> in 3D, but in 2D, depending on the version of the
     * library, it can be either a Tensor<1, 1> or a double. The former needs to
     * be indexed [0] to return its value. The shape functions, on the other
     * hand, are always double for scalars.
     */
    static constexpr unsigned int n_components = dim == 2 ? 1 : 3;
    using curl_type =
      typename FEValuesViews::Vector<dim>::template solution_curl_type<double>;
    using quantity_type = curl_type;
    using shape_type    = std::conditional_t<dim == 2, double, Tensor<1, dim>>;

    /**
     * Constructor.
     */
    VorticityEvaluator(const ComponentOrdering    &ordering,
                       const ParameterReader<dim> &param);

    /**
     * Reinit member data. This evaluator does not require any, so this does
     * nothing.
     */
    void reinit(const PostprocessorAtDofBase<dim> &) {}

    /**
     * Return the name of each of the postprocessed component.
     */
    std::vector<std::string> get_data_names() const;

    /**
     * Return the interpretation of each postprocessed component.
     */
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_components_interpretation() const;

    /**
     * Return the update flags required by this evaluator to compute the
     * vorticity. These flags are given to the FEValues of the main solver.
     */
    UpdateFlags get_main_solver_update_flags() const;

    /**
     * Evaluate the vorticity.
     */
    template <typename VectorType, typename FEValuesType>
    void evaluate_quantity(
      const VectorType              &solution,
      const std::vector<VectorType> & /* previous_solutions */,
      const TimeHandler             & /* time_handler */,
      const FEValuesType            &fe_values,
      std::vector<quantity_type>    &computed_values) const
    {
      fe_values[velocity_extractor].get_function_curls(solution,
                                                       computed_values);
    }

    /**
     * Return the @p index-th entry of @p values as a shape_type object.
     * This function basically exists to return a Tensor<1, 1> as a double,
     * as it's the type used for the 2D curl until deal.II version 9.7.
     * It's likely this function can be removed at some point.
     */
    shape_type get_quantity(const std::vector<quantity_type> &values,
                            const unsigned int                index) const
    {
      if constexpr (dim == 2)
        {
          if constexpr (std::is_arithmetic_v<curl_type>)
            return values[index];
          else
            return values[index][0];
        }
      else
        return values[index];
    }

  public:
    /**
     * The parameters associated with the vorticity.
     */
    const Parameters::PostProcessing::PostProcessingField &param;

  private:
    /**
     * Extractor for the velocity in the main solver's FEValues.
     */
    FEValuesExtractors::Vector velocity_extractor;
  };

  /**
   * A DataPostProcessor to compute the Q criterion.
   */
  template <int dim>
  class QCriterionPostProcessor : public DataPostprocessorScalar<dim>
  {
  public:
    /**
     * Constructor.
     */
    QCriterionPostProcessor(const ComponentOrdering &ordering);

    /**
     * Evaluate the Q-criterion from the velocity gradient.
     */
    virtual void evaluate_vector_field(
      const DataPostprocessorInputs::Vector<dim> &input_data,
      std::vector<Vector<double>> &computed_quantities) const override;

  private:
    const unsigned int u_lower;
  };

  /**
   * An Evaluator to compute the Q-criterion at quadrature nodes, store some
   * data needed to perform these computations (e.g., required update flags in
   * the main solver's FEValues, etc.), and specify the types involved.
   */
  template <int dim>
  class QCriterionEvaluator
  {
  public:
    static constexpr unsigned int n_components = 1;
    using quantity_type                        = double;
    using shape_type                           = double;

    /**
     * Constructor.
     */
    QCriterionEvaluator(const ComponentOrdering    &ordering,
                        const ParameterReader<dim> &param);

    /**
     * Resize velocity_gradients to the number of quadrature points.
     */
    void reinit(const PostprocessorAtDofBase<dim> &projection_base);

    /**
     * Return the name of the postprocessed component.
     */
    std::vector<std::string> get_data_names() const;

    /**
     * Return the interpretation of the postprocessed component.
     */
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_components_interpretation() const;

    /**
     * Return the update flags required by this evaluator to compute the
     * Q-criterion. These flags are given to the FEValues of the main solver.
     */
    UpdateFlags get_main_solver_update_flags() const;

    /**
     * Evaluate the Q-criterion.
     */
    template <typename VectorType, typename FEValuesType>
    void evaluate_quantity(
      const VectorType              &solution,
      const std::vector<VectorType> & /* previous_solutions */,
      const TimeHandler             & /* time_handler */,
      const FEValuesType            &fe_values,
      std::vector<quantity_type>    &computed_values)
    {
      AssertDimension(computed_values.size(), velocity_gradients.size());
      fe_values[velocity_extractor].get_function_gradients(solution,
                                                           velocity_gradients);
      const unsigned int size = computed_values.size();
      for (unsigned int i = 0; i < size; ++i)
        computed_values[i] = compute_q_criterion<dim>(velocity_gradients[i]);
    }

    /**
     * Return the @p index-th entry of @p values as a shape_type object.
     * This function is trivial here, and it's likely that it can be removed at
     * some point.
     */
    shape_type get_quantity(const std::vector<quantity_type> &values,
                            const unsigned int                index) const
    {
      return values[index];
    }

  public:
    /**
     * The parameters associated with the Q-criterion.
     */
    const Parameters::PostProcessing::PostProcessingField &param;

  private:
    /**
     * Extractor for the velocity in the main solver's FEValues.
     */
    FEValuesExtractors::Vector velocity_extractor;

    /**
     * Used to get the velocity gradients at the quadrature nodes, then compute
     * the criterion from them.
     */
    std::vector<Tensor<2, dim>> velocity_gradients;
  };

  /** Reconstruct the CHNS mixture density at quadrature points. */
  template <int dim>
  class DensityEvaluator
  {
  public:
    static constexpr unsigned int n_components = 1;
    using quantity_type                        = double;
    using shape_type                           = double;

    DensityEvaluator(const ComponentOrdering    &ordering,
                     const ParameterReader<dim> &parameters)
      : param(parameters.postprocessing.density)
      , cahn_hilliard(parameters.cahn_hilliard)
      , tracer_extractor(ordering.phi_lower)
      , density_0(parameters.physical_properties.fluids.at(0).density)
      , density_1(parameters.physical_properties.fluids.at(1).density)
      , tracer_limiter(CahnHilliard::get_limiter_function(cahn_hilliard))
      , material_phase(
          CahnHilliard::get_material_phase_function(cahn_hilliard))
    {
      AssertThrow(ordering.phi_lower != numbers::invalid_unsigned_int,
                  ExcMessage("Density postprocessing requires a CHNS tracer."));
      AssertThrow(parameters.physical_properties.fluids.size() >= 2,
                  ExcMessage("Density postprocessing requires two fluids."));
    }

    void reinit(const PostprocessorAtDofBase<dim> &projection_base)
    {
      tracer_values.resize(projection_base.get_n_q_points());
    }

    std::vector<std::string> get_data_names() const
    {
      return {"density"};
    }

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_components_interpretation() const
    {
      return {DataComponentInterpretation::component_is_scalar};
    }

    UpdateFlags get_main_solver_update_flags() const
    {
      return update_values;
    }

    template <typename VectorType, typename FEValuesType>
    void evaluate_quantity(
      const VectorType              &solution,
      const std::vector<VectorType> & /* previous_solutions */,
      const TimeHandler             & /* time_handler */,
      const FEValuesType            &fe_values,
      std::vector<quantity_type>    &computed_values)
    {
      AssertDimension(computed_values.size(), tracer_values.size());
      fe_values[tracer_extractor].get_function_values(solution, tracer_values);
      for (unsigned int q = 0; q < computed_values.size(); ++q)
      {
        const double marker =
          material_phase(cahn_hilliard, tracer_limiter(tracer_values[q]));
        computed_values[q] =
          CahnHilliard::linear_mixing(marker, density_0, density_1);
      }
    }

    shape_type get_quantity(const std::vector<quantity_type> &values,
                            const unsigned int                index) const
    {
      return values[index];
    }

    const Parameters::PostProcessing::PostProcessingField &param;

  private:
    const Parameters::CahnHilliard<dim> &cahn_hilliard;
    FEValuesExtractors::Scalar           tracer_extractor;
    double                               density_0;
    double                               density_1;
    CahnHilliard::TracerLimiterFunction  tracer_limiter;
    CahnHilliard::MaterialPhaseFunction<dim> material_phase;
    std::vector<double>                       tracer_values;
  };

  /** Evaluate the selected Cahn-Hilliard mobility model. */
  template <int dim>
  class MobilityEvaluator
  {
  public:
    static constexpr unsigned int n_components = 1;
    using quantity_type                        = double;
    using shape_type                           = double;

    MobilityEvaluator(const ComponentOrdering    &ordering,
                      const ParameterReader<dim> &parameters)
      : param(parameters.postprocessing.mobility)
      , cahn_hilliard(parameters.cahn_hilliard)
      , velocity_extractor(ordering.u_lower)
      , tracer_extractor(ordering.phi_lower)
      , tracer_limiter(
          CahnHilliard::get_mobility_limiter_function(cahn_hilliard))
      , material_phase(
          CahnHilliard::get_material_phase_function(cahn_hilliard))
      , material_phase_derivative(
          CahnHilliard::get_material_phase_derivative_function(cahn_hilliard))
      , material_phase_second_derivative(
          CahnHilliard::get_material_phase_second_derivative_function(
            cahn_hilliard))
      , mobility_function(
          CahnHilliard::get_mobility_evaluation_function(cahn_hilliard))
      , adaptive_coefficient(0.)
      , adaptive_delta(0.)
    {
      AssertThrow(ordering.u_lower != numbers::invalid_unsigned_int &&
                    ordering.phi_lower != numbers::invalid_unsigned_int,
                  ExcMessage(
                    "Mobility postprocessing requires CHNS velocity and tracer."));
      if (CahnHilliard::is_adaptive_mobility_model(cahn_hilliard))
      {
        const auto scaling =
          CahnHilliard::get_adaptive_mobility_scaling(cahn_hilliard);
        adaptive_coefficient = scaling.coefficient;
        adaptive_delta       = scaling.delta;
      }
    }

    void reinit(const PostprocessorAtDofBase<dim> &projection_base)
    {
      const auto n = projection_base.get_n_q_points();
      tracer_values.resize(n);
      velocity_values.resize(n);
      tracer_gradients.resize(n);
    }

    std::vector<std::string> get_data_names() const
    {
      return {"mobility"};
    }

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_components_interpretation() const
    {
      return {DataComponentInterpretation::component_is_scalar};
    }

    UpdateFlags get_main_solver_update_flags() const
    {
      return update_values | update_gradients;
    }

    template <typename VectorType, typename FEValuesType>
    void evaluate_quantity(
      const VectorType              &solution,
      const std::vector<VectorType> & /* previous_solutions */,
      const TimeHandler             & /* time_handler */,
      const FEValuesType            &fe_values,
      std::vector<quantity_type>    &computed_values)
    {
      AssertDimension(computed_values.size(), tracer_values.size());
      fe_values[tracer_extractor].get_function_values(solution, tracer_values);
      fe_values[tracer_extractor].get_function_gradients(solution,
                                                         tracer_gradients);
      fe_values[velocity_extractor].get_function_values(solution,
                                                        velocity_values);

      for (unsigned int q = 0; q < computed_values.size(); ++q)
      {
        const double phi = tracer_limiter(tracer_values[q]);
        computed_values[q] =
          mobility_function(cahn_hilliard,
                            material_phase(cahn_hilliard, phi),
                            material_phase_derivative(cahn_hilliard, phi),
                            material_phase_second_derivative(cahn_hilliard, phi),
                            velocity_values[q],
                            tracer_gradients[q],
                            adaptive_coefficient,
                            adaptive_delta)
            .value;
      }
    }

    shape_type get_quantity(const std::vector<quantity_type> &values,
                            const unsigned int                index) const
    {
      return values[index];
    }

    const Parameters::PostProcessing::PostProcessingField &param;

  private:
    const Parameters::CahnHilliard<dim> &cahn_hilliard;
    FEValuesExtractors::Vector           velocity_extractor;
    FEValuesExtractors::Scalar           tracer_extractor;
    CahnHilliard::MobilityTracerLimiterFunction tracer_limiter;
    CahnHilliard::MaterialPhaseFunction<dim>     material_phase;
    CahnHilliard::MaterialPhaseFunction<dim>     material_phase_derivative;
    CahnHilliard::MaterialPhaseFunction<dim>
      material_phase_second_derivative;
    CahnHilliard::MobilityEvaluationFunction<dim> mobility_function;
    double                                        adaptive_coefficient;
    double                                        adaptive_delta;
    std::vector<double>                           tracer_values;
    std::vector<Tensor<1, dim>>                   velocity_values;
    std::vector<Tensor<1, dim>>                   tracer_gradients;
  };

  enum class MeshForcingField
  {
    physics_compression,
    enlarged_compression,
    transport
  };

  /** Evaluate one contribution to the CHNS-ALE mesh forcing. */
  template <int dim, MeshForcingField field>
  class MeshForcingEvaluator
  {
  public:
    static constexpr unsigned int n_components = dim;
    using quantity_type                        = Tensor<1, dim>;
    using shape_type                           = Tensor<1, dim>;

    MeshForcingEvaluator(const ComponentOrdering    &ordering,
                         const ParameterReader<dim> &parameters)
      : param(select_parameters(parameters))
      , cahn_hilliard(parameters.cahn_hilliard)
      , velocity_extractor(ordering.u_lower)
      , position_extractor(ordering.x_lower)
      , tracer_extractor(ordering.phi_lower)
      , psi_extractor(ordering.psi_lower)
      , with_enlarged(ordering.psi_lower != numbers::invalid_unsigned_int)
      , enlarged_normalization(
          Assembly::Elasticity::enlarged_mesh_forcing_normalization(
            cahn_hilliard.mff_regularization_gamma,
            cahn_hilliard.mff_enlarged_lobe_position_exponent))
    {
      AssertThrow(ordering.u_lower != numbers::invalid_unsigned_int &&
                    ordering.x_lower != numbers::invalid_unsigned_int &&
                    ordering.phi_lower != numbers::invalid_unsigned_int,
                  ExcMessage("Mesh-forcing postprocessing requires a moving "
                             "mesh CHNS solver."));
      if constexpr (field == MeshForcingField::enlarged_compression)
        AssertThrow(with_enlarged,
                    ExcMessage("Enlarged mesh-forcing compression requires "
                               "the enlarged CHNS solver."));
    }

    void reinit(const PostprocessorAtDofBase<dim> &projection_base)
    {
      const auto n = projection_base.get_n_q_points();
      marker_values.resize(n);
      marker_gradients.resize(n);
      tracer_values.resize(n);
      tracer_gradients.resize(n);
      velocity_values.resize(n);
      position_values.resize(n);
    }

    std::vector<std::string> get_data_names() const
    {
      return std::vector<std::string>(dim, field_name());
    }

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
    get_components_interpretation() const
    {
      return std::vector<
        DataComponentInterpretation::DataComponentInterpretation>(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    }

    UpdateFlags get_main_solver_update_flags() const
    {
      return update_values | update_gradients;
    }

    template <typename VectorType, typename FEValuesType>
    void evaluate_quantity(
      const VectorType              &solution,
      const std::vector<VectorType> &previous_solutions,
      const TimeHandler             &time_handler,
      const FEValuesType            &fe_values,
      std::vector<quantity_type>    &computed_values)
    {
      AssertDimension(computed_values.size(), marker_values.size());
      const double epsilon = cahn_hilliard.epsilon_interface;
      const double gamma   = cahn_hilliard.mff_regularization_gamma;

      if constexpr (field == MeshForcingField::physics_compression)
      {
        fe_values[tracer_extractor].get_function_values(solution, tracer_values);
        fe_values[tracer_extractor].get_function_gradients(solution,
                                                           tracer_gradients);
        for (unsigned int q = 0; q < computed_values.size(); ++q)
          computed_values[q] =
            cahn_hilliard.mff_physics_compression_factor * epsilon *
            Assembly::Elasticity::mesh_forcing_factor(tracer_values[q], gamma)
              .value *
            tracer_gradients[q];
        return;
      }

      const auto &marker_extractor = with_enlarged ? psi_extractor :
                                                     tracer_extractor;
      fe_values[marker_extractor].get_function_values(solution, marker_values);
      fe_values[marker_extractor].get_function_gradients(solution,
                                                         marker_gradients);

      const double marker_epsilon =
        with_enlarged ? cahn_hilliard.psi_interface_width_factor * epsilon :
                        epsilon;

      if constexpr (field == MeshForcingField::enlarged_compression)
      {
        for (unsigned int q = 0; q < computed_values.size(); ++q)
          computed_values[q] =
            cahn_hilliard.mff_enlarged_compression_factor * marker_epsilon *
            Assembly::Elasticity::enlarged_mesh_forcing_factor(
              marker_values[q],
              gamma,
              cahn_hilliard.mff_enlarged_lobe_position_exponent,
              enlarged_normalization)
              .value *
            marker_gradients[q];
        return;
      }

      fe_values[velocity_extractor].get_function_values(solution,
                                                        velocity_values);
      fe_values[position_extractor].get_function_values(solution,
                                                        position_values);
      previous_position_values.resize(previous_solutions.size());
      for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      {
        previous_position_values[i].resize(computed_values.size());
        fe_values[position_extractor].get_function_values(
          previous_solutions[i], previous_position_values[i]);
      }

      for (unsigned int q = 0; q < computed_values.size(); ++q)
      {
        const Tensor<1, dim> mesh_velocity =
          time_handler.template compute_time_derivative_at_quadrature_node<dim>(
            q, position_values[q], previous_position_values);
        const Tensor<1, dim> convective_velocity =
          velocity_values[q] - mesh_velocity;
        computed_values[q] =
          cahn_hilliard.mff_transport_factor * marker_epsilon * marker_epsilon *
          ((convective_velocity * marker_gradients[q]) * marker_gradients[q]);
      }
    }

    shape_type get_quantity(const std::vector<quantity_type> &values,
                            const unsigned int                index) const
    {
      return values[index];
    }

    const Parameters::PostProcessing::PostProcessingField &param;

  private:
    static const Parameters::PostProcessing::PostProcessingField &
    select_parameters(const ParameterReader<dim> &parameters)
    {
      if constexpr (field == MeshForcingField::physics_compression)
        return parameters.postprocessing.mff_physics_compression;
      else if constexpr (field == MeshForcingField::enlarged_compression)
        return parameters.postprocessing.mff_enlarged_compression;
      else
        return parameters.postprocessing.mff_transport;
    }

    static std::string field_name()
    {
      if constexpr (field == MeshForcingField::physics_compression)
        return "mff_physics_compression";
      else if constexpr (field == MeshForcingField::enlarged_compression)
        return "mff_enlarged_compression";
      else
        return "mff_transport";
    }

    const Parameters::CahnHilliard<dim> &cahn_hilliard;
    FEValuesExtractors::Vector           velocity_extractor;
    FEValuesExtractors::Vector           position_extractor;
    FEValuesExtractors::Scalar           tracer_extractor;
    FEValuesExtractors::Scalar           psi_extractor;
    bool                                 with_enlarged;
    double                               enlarged_normalization;
    std::vector<double>                  marker_values;
    std::vector<Tensor<1, dim>>          marker_gradients;
    std::vector<double>                  tracer_values;
    std::vector<Tensor<1, dim>>          tracer_gradients;
    std::vector<Tensor<1, dim>>          velocity_values;
    std::vector<Tensor<1, dim>>          position_values;
    std::vector<std::vector<Tensor<1, dim>>> previous_position_values;
  };

  template <int dim>
  using MFFPhysicsCompressionEvaluator =
    MeshForcingEvaluator<dim, MeshForcingField::physics_compression>;

  template <int dim>
  using MFFEnlargedCompressionEvaluator =
    MeshForcingEvaluator<dim, MeshForcingField::enlarged_compression>;

  template <int dim>
  using MFFTransportEvaluator =
    MeshForcingEvaluator<dim, MeshForcingField::transport>;

  /**
   * Compute mesh velocity, for solvers solving for the mesh position.
   * The resulting field uses the same finite element representation as the mesh
   * position field in the main solver, so that we can simply take the time
   * derivative of the nodal values. Consequently, the "degree" parameter is
   * ignored for this class.
   */
  template <int dim>
  class MeshVelocityPostprocessor : public PostprocessorAtDofBase<dim>
  {
  public:
    /**
     * Constructor.
     *
     * FIXME: the mapping and cell_quadrature are unused, since this
     * postprocessor does not need an FEValues. The PostprocessorAtDofBase could
     * be simplified to create an FEValues only if needed.
     */
    MeshVelocityPostprocessor(const ComponentOrdering    &ordering,
                              const ParameterReader<dim> &param,
                              const Mapping<dim>         &mapping,
                              const DoFHandler<dim>      &solver_dof_handler,
                              const Quadrature<dim>      &cell_quadrature);

  protected:
    /**
     * Take the time derivative at dofs of the mesh position solution.
     */
    void
    do_postprocess(const LA::ParVectorType              &present_solution,
                   const std::vector<LA::ParVectorType> &previous_solutions,
                   const TimeHandler                    &time_handler) override;

  private:
    /**
     * Owned mesh position degrees of freedom in the main solver's dof_handler.
     */
    IndexSet solver_owned_position_dofs;
  };
} // namespace PostProcessingTools

#endif
