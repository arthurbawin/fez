
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/sparsity_tools.h>
#include <post_processing_handler.h>
#include <postprocessors_and_evaluators.h>
#include <utilities.h>

namespace PostProcessingTools
{
  template <int dim>
  VorticityPostProcessor<dim>::VorticityPostProcessor(
    const ComponentOrdering &ordering)
    : DataPostprocessorVector<dim>("vorticity", update_gradients)
    , u_lower(ordering.u_lower)
  {
    Assert(u_lower != numbers::invalid_unsigned_int,
           ExcMessage("Cannot postprocess vorticity field because this "
                      "solver does not solve for the velocity variable!"));
  }

  template <int dim>
  void VorticityPostProcessor<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &input_data,
    std::vector<Vector<double>>                &computed_quantities) const
  {
    AssertDimension(input_data.solution_values.size(),
                    computed_quantities.size());

    for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
    {
      AssertDimension(computed_quantities[p].size(), dim);

      auto       &curl     = computed_quantities[p];
      const auto &grad_sol = input_data.solution_gradients[p];

      if constexpr (dim == 2)
        curl[0] = grad_sol[u_lower + 1][0] - grad_sol[u_lower + 0][1];
      if constexpr (dim == 3)
      {
        curl[0] = grad_sol[u_lower + 2][1] - grad_sol[u_lower + 1][2];
        curl[1] = grad_sol[u_lower + 0][2] - grad_sol[u_lower + 2][0];
        curl[2] = grad_sol[u_lower + 1][0] - grad_sol[u_lower + 0][1];
      }
    }
  }

  template class VorticityPostProcessor<2>;
  template class VorticityPostProcessor<3>;

  template <int dim>
  VorticityEvaluator<dim>::VorticityEvaluator(const ComponentOrdering &ordering,
                                              const ParameterReader<dim> &param)
    : param(param.postprocessing.vorticity)
    , velocity_extractor(ordering.u_lower)
  {}

  template <int dim>
  std::vector<std::string> VorticityEvaluator<dim>::get_data_names() const
  {
    return std::vector<std::string>(n_components, "vorticity");
  }

  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  VorticityEvaluator<dim>::get_components_interpretation() const
  {
    if (n_components == 1)
      return {DataComponentInterpretation::component_is_scalar};
    else
      return std::vector<
        DataComponentInterpretation::DataComponentInterpretation>(
        n_components,
        DataComponentInterpretation::DataComponentInterpretation::
          component_is_part_of_vector);
  }

  template <int dim>
  UpdateFlags VorticityEvaluator<dim>::get_main_solver_update_flags() const
  {
    return update_gradients;
  }

  template class VorticityEvaluator<2>;
  template class VorticityEvaluator<3>;

  template <int dim>
  QCriterionPostProcessor<dim>::QCriterionPostProcessor(
    const ComponentOrdering &ordering)
    : DataPostprocessorScalar<dim>("q_criterion", update_gradients)
    , u_lower(ordering.u_lower)
  {
    Assert(u_lower != numbers::invalid_unsigned_int,
           ExcMessage("Cannot postprocess Q criterion field because this "
                      "solver does not solve for the velocity variable!"));
  }

  template <int dim>
  void QCriterionPostProcessor<dim>::evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &input_data,
    std::vector<Vector<double>>                &computed_quantities) const
  {
    AssertDimension(input_data.solution_values.size(),
                    computed_quantities.size());

    for (unsigned int p = 0; p < input_data.solution_gradients.size(); ++p)
    {
      AssertDimension(computed_quantities[p].size(), 1);
      const auto &grad_sol   = input_data.solution_gradients[p];
      computed_quantities[p] = compute_q_criterion<dim>(grad_sol, u_lower);
    }
  }

  template class QCriterionPostProcessor<2>;
  template class QCriterionPostProcessor<3>;

  template <int dim>
  QCriterionEvaluator<dim>::QCriterionEvaluator(
    const ComponentOrdering    &ordering,
    const ParameterReader<dim> &param)
    : param(param.postprocessing.q_criterion)
    , velocity_extractor(ordering.u_lower)
  {}

  template <int dim>
  void QCriterionEvaluator<dim>::reinit(
    const PostprocessorAtDofBase<dim> &projection_base)
  {
    velocity_gradients.resize(projection_base.get_n_q_points());
  }

  template <int dim>
  std::vector<std::string> QCriterionEvaluator<dim>::get_data_names() const
  {
    return std::vector<std::string>(n_components, "qcriterion");
  }

  template <int dim>
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
  QCriterionEvaluator<dim>::get_components_interpretation() const
  {
    return {DataComponentInterpretation::component_is_scalar};
  }

  template <int dim>
  UpdateFlags QCriterionEvaluator<dim>::get_main_solver_update_flags() const
  {
    return update_gradients;
  }

  template class QCriterionEvaluator<2>;
  template class QCriterionEvaluator<3>;

  template <int dim>
  MeshVelocityPostprocessor<dim>::MeshVelocityPostprocessor(
    const ComponentOrdering    &ordering,
    const ParameterReader<dim> &param,
    const Mapping<dim>         &mapping,
    const DoFHandler<dim>      &solver_dof_handler,
    const Quadrature<dim>      &cell_quadrature)
    : PostprocessorAtDofBase<dim>(
        ordering,
        param,
        mapping,
        solver_dof_handler,
        cell_quadrature,
        update_values,
        std::vector<std::string>(dim, "mesh_velocity"),
        std::vector<DataComponentInterpretation::DataComponentInterpretation>(
          dim,
          DataComponentInterpretation::DataComponentInterpretation::
            component_is_part_of_vector))
  {
    AssertThrow(ordering.x_lower != numbers::invalid_unsigned_int,
                ExcMessage(
                  "Cannot postprocess mesh velocity field because this "
                  "solver does not solve for the mesh position variable!"));

    // Get the base element associated with mesh position in the main solver's
    // FESystem, then create an FESystem with dim copies.
    const auto        &solver_fe = solver_dof_handler.get_fe();
    const unsigned int base_index =
      solver_fe.component_to_base_index(ordering.x_lower).first;
    const auto &position_fe_base = solver_fe.base_element(base_index);

    // Depending on the main solver's FESystem, the base FE for mesh position
    // could be a single copy, or a FESystem itself.
    const unsigned int n_components = position_fe_base.n_components();
    if (n_components == dim)
      this->fe = position_fe_base.clone();
    else
    {
      // If base FE has neither 1 nor dim components, then something weird is
      // happening
      AssertThrow(
        n_components == 1,
        ExcMessage(
          "Could not determine the base finite element space "
          "associated with the mesh position. Expected a "
          "FiniteElement with either 1 or dim components, and instead got " +
          std::to_string(n_components)));
      this->fe = std::make_unique<FESystem<dim>>(position_fe_base, dim);
    }

    // Distribute dofs and allocate solution vector
    this->dof_handler.distribute_dofs(*this->fe);
    this->solution.reinit(this->dof_handler.locally_owned_dofs(),
                          this->mpi_communicator);

    /**
     * Then we need to map the dof indices associated with the mesh position in
     * the main dof_handler (dh), to their indices in the smaller dh.
     *
     * By keeping the same partitions and finite element space for x,
     * we here assume that the owned position dofs are identically numbered
     * between the main and smaller dh. Thus, we assume that
     *
     * - on each partition, there is the same number of owned position dofs
     * between both dof handlers, and
     * - their support points and components are the same.

     * This way, we can simply compress the set of owned position dofs from
     * the main solution vector, i.e., we can 1:1 map the range [0,
     n_owned_dofs)
     * associated with the smaller dh with the IndexSet obtained with
     * DoFTools::extract_dofs(solver_dof_handler, position_mask).
     *
     * In debug, we actually check these assumptions.
     */

    const auto position_extractor =
      FEValuesExtractors::Vector(ordering.x_lower);
    const auto position_mask = solver_fe.component_mask(position_extractor);

    solver_owned_position_dofs =
      DoFTools::extract_dofs(solver_dof_handler, position_mask);

    if (running_in_debug_mode())
    {
      // Check that there is the same number of owned dofs
      const auto owned_position_dofs = this->dof_handler.locally_owned_dofs();

      Assert(
        solver_owned_position_dofs.n_elements() ==
          owned_position_dofs.n_elements(),
        ExcMessage(
          "This postprocessor assumes that there is the same number of owned "
          "mesh position degrees of freedom between the dof handler of the  "
          "main solver and this one's.  But on this partition, there are " +
          std::to_string(solver_owned_position_dofs.n_elements()) +
          " owned dofs in the main dof handler and " +
          std::to_string(owned_position_dofs.n_elements()) +
          " in the postprocessor's dof handler."));

      // Then check that simply compressing the indices yields the same set as
      // the one obtained by after identifying the support points
      const auto solver_support_points =
        DoFTools::map_dofs_to_support_points(mapping, solver_dof_handler);
      const auto support_points =
        DoFTools::map_dofs_to_support_points(mapping, this->dof_handler);

      std::vector<types::global_dof_index> identified_dofs(
        owned_position_dofs.n_elements(), numbers::invalid_unsigned_int);
      std::vector<types::global_dof_index> compressed_dofs(
        owned_position_dofs.n_elements(), numbers::invalid_unsigned_int);

      const auto local_range = this->solution.local_range();

      unsigned int cnt = 0;
      for (const auto &[solver_dof, solver_pt] : solver_support_points)
        if (solver_owned_position_dofs.is_element(solver_dof))
        {
          compressed_dofs[cnt++] = solver_dof;
          // Also check if this dof support point is found in the smaller map
          bool found = false;
          for (const auto &[dof, pt] : support_points)
            if (owned_position_dofs.is_element(dof))
              if (solver_pt.distance(pt) < 1e-12)
              {
                found = true;
                // Add dofs at same support point in component order
                if (identified_dofs[dof - local_range.first] ==
                    numbers::invalid_unsigned_int)
                {
                  identified_dofs[dof - local_range.first] = solver_dof;
                  break;
                }
              }
          Assert(found, ExcInternalError());
        }
      for (unsigned int i = 0; i < identified_dofs.size(); ++i)
        Assert(
          identified_dofs[i] == compressed_dofs[i],
          ExcMessage(
            "This postprocessor assumes that the owned mesh position degrees "
            "of freedom can simply be compressed to obtain their dof indices "
            "in the postprocessor's dof handler. But on this partition, there "
            "is is a mismatch between the compressed indices and the indices "
            "obtained by identifying the support points between both dof "
            "handlers. This essentially means that there is the same number of "
            "owned dofs, but they are numbered differently between the main "
            "solver and the postprocessor. In that case, the set of indices to "
            "use in the postprocessor should be set to the \"identified_dofs\" "
            "set used to do this verification."));
    }
  }

  template <int dim>
  void MeshVelocityPostprocessor<dim>::do_postprocess(
    const LA::ParVectorType              &present_solution,
    const std::vector<LA::ParVectorType> &previous_solutions,
    const TimeHandler                    &time_handler)
  {
    const auto first = this->solution.local_range().first;

    unsigned int cnt = 0;
    for (const auto dof : solver_owned_position_dofs)
      this->solution[first + cnt++] =
        time_handler.compute_time_derivative(dof,
                                             present_solution,
                                             previous_solutions);
    this->solution.compress(VectorOperation::insert);
  }

  template class MeshVelocityPostprocessor<2>;
  template class MeshVelocityPostprocessor<3>;
} // namespace PostProcessingTools
