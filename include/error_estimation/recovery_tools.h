#ifndef RECOVERY_TOOLS_H
#define RECOVERY_TOOLS_H

#include <components_ordering.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/grid/tria.h>
#include <error_estimation/patches.h>
#include <error_estimation/recovery_tools.h>
#include <error_estimation/solution_recovery.h>
#include <metric_field.h>
#include <parameter_reader.h>
#include <solver_info.h>

namespace ErrorEstimation
{
  using namespace dealii;

  /**
   * Initialize a PatchHandler and SolutionRecovery from the parameters
   * relative to the metric fields.
   *
   * Essentially:
   *
   * - create the unique_ptr data for PatchHandler and solutionRecovery
   * for the given variable
   */
  template <int dim, typename VectorType>
  void initialize_reconstruction_data(
    const ParameterReader<dim>                        &param,
    const parallel::DistributedTriangulationBase<dim> &triangulation,
    const Mapping<dim>                                &mapping,
    const DoFHandler<dim>                             &dof_handler,
    const VectorType                                  &present_solution,
    const ComponentOrdering                           &ordering,
    std::vector<std::unique_ptr<MetricField<dim>>>    &metrics,
    std::vector<std::unique_ptr<PatchHandler<dim>>>   &patch_handlers,
    std::vector<std::unique_ptr<SolutionRecovery::Scalar<dim>>> &recoveries);

} // namespace ErrorEstimation

/*------------------------ template functions -------------------------------*/

namespace ErrorEstimation
{
  template <int dim, typename VectorType>
  void initialize_reconstruction_data(
    const ParameterReader<dim>                        &param,
    const parallel::DistributedTriangulationBase<dim> &triangulation,
    const Mapping<dim>                                &mapping,
    const DoFHandler<dim>                             &dof_handler,
    const VectorType                                  &present_solution,
    const ComponentOrdering                           &ordering,
    std::vector<std::unique_ptr<MetricField<dim>>>    &metrics,
    std::vector<std::unique_ptr<PatchHandler<dim>>>   &patch_handlers,
    std::vector<std::unique_ptr<SolutionRecovery::Scalar<dim>>> &recoveries)
  {
    if (param.metric_fields.size() == 0)
      return;

    metrics.clear();
    patch_handlers.clear();
    recoveries.clear();

    const auto &fe = dof_handler.get_fe();

    for (const auto &metric_param : param.metric_fields)
    {
      const SolverInfo::VariableType var  = metric_param.variable;
      const unsigned int             comp = metric_param.component;

      // Check that the solver stores this variable
      AssertThrow(
        ordering.has_variable(var),
        ExcMessage(
          "Trying to initialize reconstruction data with respect to variable " +
          SolverInfo::to_string(var) +
          " with a solver which does not store this variable."));

      // Create metric field
      metrics.emplace_back(std::make_unique<MetricField<dim>>(metric_param.id,
                                                              param,
                                                              triangulation));

      const unsigned int first_comp = ordering.variable_to_first_component(var);
      const FEValuesExtractors::Scalar extractor(first_comp + comp);
      const ComponentMask              mask = fe.component_mask(extractor);

      // Create patch handler
      patch_handlers.emplace_back(
        std::make_unique<ErrorEstimation::PatchHandler<dim>>(
          triangulation,
          mapping,
          dof_handler,
          present_solution,
          param.finite_elements.get_variable_degree(var) + 1,
          mask));

      auto &patch_handler = patch_handlers.back();

      // Build the patches
      // computing_timer.enter_subsection("Build patches");
      patch_handler->build_patches();
      // computing_timer.leave_subsection();

      // Create recovery operator for this time subinterval
      recoveries.emplace_back(
        std::make_unique<ErrorEstimation::SolutionRecovery::Scalar<dim>>(
          std::min(param.finite_elements.get_variable_degree(var) + 1, 2u),
          param,
          *patch_handler,
          dof_handler,
          present_solution,
          fe,
          mapping,
          mask,
          /* isoparametric = */ true,
          /* single_reconstruction = */ true));
    }
  }
} // namespace ErrorEstimation

#endif
