#ifndef POSTPROCESSORS_AND_EVALUATORS_H
#define POSTPROCESSORS_AND_EVALUATORS_H

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
    void evaluate_quantity(const VectorType           &solution,
                           const FEValuesType         &fe_values,
                           std::vector<quantity_type> &computed_values) const
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
      // deal.II 9.7 uses Tensor<1, 1> for the 2D curl; newer versions use
      // a scalar. Detect the actual type, including development versions.
      if constexpr (std::is_same_v<curl_type, shape_type>)
        return values[index];
      else
      {
        static_assert(dim == 2);
        return values[index][0];
      }
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
    void evaluate_quantity(const VectorType           &solution,
                           const FEValuesType         &fe_values,
                           std::vector<quantity_type> &computed_values)
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
