
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/sparsity_tools.h>
#include <post_processing_handler.h>
#include <utilities.h>
#include <vorticity_postprocessors.h>

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
  VorticityAtDofBase<dim>::VorticityAtDofBase(
    const ParameterReader<dim> &param,
    const ComponentOrdering    &ordering,
    const Mapping<dim>         &mapping,
    const DoFHandler<dim>      &dof_handler,
    const Quadrature<dim>      &cell_quadrature)
    : PostprocessorAtDofBase<dim>(ordering,
                                  param,
                                  mapping,
                                  dof_handler,
                                  cell_quadrature,
                                  update_gradients)
  {
    using DCI = DataComponentInterpretation::DataComponentInterpretation;

    this->data_names = std::vector<std::string>(n_components, "vorticity");

    const unsigned int degree = param.postprocessing.vorticity.degree;

    if constexpr (n_components == 1)
    {
      if (param.finite_elements.use_quads)
        this->fe = std::make_unique<FE_Q<dim>>(degree);
      else
        this->fe = std::make_unique<FE_SimplexP<dim>>(degree);

      this->data_interpretation = {DCI::component_is_scalar};
    }
    else
    {
      if (param.finite_elements.use_quads)
        this->fe =
          std::make_unique<FESystem<dim>>(FE_Q<dim>(degree) ^ n_components);
      else
        this->fe = std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(degree) ^
                                                   n_components);


      this->data_interpretation =
        std::vector<DCI>(n_components, DCI::component_is_part_of_vector);
    }
    this->dof_handler.distribute_dofs(*this->fe);

    fe_values_vorticity = std::make_unique<FEValues<dim>>(
      mapping, *this->fe, cell_quadrature, update_values | update_JxW_values);
  }

  template class VorticityAtDofBase<2>;
  template class VorticityAtDofBase<3>;

  template <int dim>
  VorticityL2Projection<dim>::VorticityL2Projection(
    const ParameterReader<dim> &param,
    const ComponentOrdering    &ordering,
    const Mapping<dim>         &mapping,
    const DoFHandler<dim>      &dof_handler,
    const Quadrature<dim>      &cell_quadrature,
    const bool                  with_moving_mesh)
    : VorticityAtDofBase<dim>(param,
                              ordering,
                              mapping,
                              dof_handler,
                              cell_quadrature)
    , with_moving_mesh(with_moving_mesh)
    , matrix_is_assembled(false)
  {
    reinit();
  }

  template <int dim>
  void VorticityL2Projection<dim>::reinit()
  {
    const auto locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    const auto locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(this->dof_handler);

    system_rhs.reinit(locally_owned_dofs, this->mpi_communicator);
    this->solution.reinit(locally_owned_dofs, this->mpi_communicator);

    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(this->dof_handler, dsp);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs,
                                               this->mpi_communicator,
                                               locally_relevant_dofs);
    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         this->mpi_communicator);
  }

  template <int dim>
  template <typename VectorType>
  void VorticityL2Projection<dim>::assemble_system(
    const VectorType &present_solution)
  {
    using shape_type     = std::conditional_t<dim == 2, double, Tensor<1, dim>>;
    using extractor_type = std::conditional_t<dim == 2,
                                              FEValuesExtractors::Scalar,
                                              FEValuesExtractors::Vector>;

    const bool assemble_matrix = with_moving_mesh or !matrix_is_assembled;

    system_rhs = 0;
    if (assemble_matrix)
      system_matrix = 0;

    const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
    const unsigned int n_q_points =
      this->solver_fe_values.get_quadrature().size();

    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<shape_type> phi_u(dofs_per_cell);

    std::vector<typename VorticityAtDofBase<dim>::curl_type> velocity_curls(
      n_q_points);

    const FEValuesExtractors::Vector velocity_extractor(this->ordering.u_lower);
    const extractor_type             vorticity_extractor(0);

    if (assemble_matrix)
    {
      for (const auto &cell : this->dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          this->fe_values_vorticity->reinit(cell);

          this->solver_fe_values.reinit(
            cell->as_dof_handler_iterator(this->solver_dof_handler));

          local_matrix = 0;
          local_rhs    = 0;

          // Current curls at cell quadrature nodes
          this->solver_fe_values[velocity_extractor].get_function_curls(
            present_solution, velocity_curls);

          for (unsigned int q = 0; q < n_q_points; ++q)
          {
            shape_type curl;
            if constexpr (dim == 2)
              curl = velocity_curls[q][0];
            else
              curl = velocity_curls[q];

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              phi_u[k] =
                (*this->fe_values_vorticity)[vorticity_extractor].value(k, q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              // RHS
              local_rhs(i) +=
                (phi_u[i] * curl) * this->fe_values_vorticity->JxW(q);

              // Mass matrix
              for (unsigned int j = 0; j < dofs_per_cell; ++j)
                local_matrix(i, j) +=
                  phi_u[j] * phi_u[i] * this->fe_values_vorticity->JxW(q);
            }
          }
          cell->distribute_local_to_global(local_matrix,
                                           local_rhs,
                                           system_matrix,
                                           system_rhs);
        }
      system_matrix.compress(VectorOperation::add);
      system_rhs.compress(VectorOperation::add);

      matrix_is_assembled = true;
    }
    else
    {
      // Assemble only RHS
      for (const auto &cell : this->dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          local_rhs = 0;
          this->fe_values_vorticity->reinit(cell);
          this->solver_fe_values.reinit(
            cell->as_dof_handler_iterator(this->solver_dof_handler));
          this->solver_fe_values[velocity_extractor].get_function_curls(
            present_solution, velocity_curls);

          for (unsigned int q = 0; q < n_q_points; ++q)
          {
            shape_type curl;
            if constexpr (dim == 2)
              curl = velocity_curls[q][0];
            else
              curl = velocity_curls[q];

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
              local_rhs(i) +=
                (*this->fe_values_vorticity)[vorticity_extractor].value(i, q) *
                curl * this->fe_values_vorticity->JxW(q);
          }
          cell->distribute_local_to_global(local_rhs, system_rhs);
        }
      system_rhs.compress(VectorOperation::add);
    }
  }

  template void
  VorticityL2Projection<2>::assemble_system(const LA::ParVectorType &);
  template void
  VorticityL2Projection<3>::assemble_system(const LA::ParVectorType &);

  template <int dim>
  void VorticityL2Projection<dim>::solve()
  {
    // Solve with CG
    SolverControl                          solver_control(1e5, 1e-7);
    LA::SolverCG                           cg_solver(solver_control);
    PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
    cg_solver.solve(system_matrix, this->solution, system_rhs, preconditioner);

    if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0 &&
        this->param.postprocessing.vorticity.verbosity ==
          Parameters::Verbosity::verbose)
      std::cout << "Vorticity L2 projection: " << solver_control.last_step()
                << " CG iterations needed to obtain convergence." << std::endl;
  }

  template <int dim>
  void VorticityL2Projection<dim>::do_postprocess(
    const LA::ParVectorType &present_solution)
  {
    this->assemble_system(present_solution);
    this->solve();
  }

  template class VorticityL2Projection<2>;
  template class VorticityL2Projection<3>;

  template <int dim>
  VorticityWeightedAverage<dim>::VorticityWeightedAverage(
    const ParameterReader<dim> &param,
    const ComponentOrdering    &ordering,
    const Mapping<dim>         &mapping,
    const DoFHandler<dim>      &dof_handler,
    const Quadrature<dim>      &cell_quadrature)
    : VorticityAtDofBase<dim>(param,
                              ordering,
                              mapping,
                              dof_handler,
                              cell_quadrature)
  {
    reinit();
  }

  template <int dim>
  void VorticityWeightedAverage<dim>::reinit()
  {
    const auto locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    const auto locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(this->dof_handler);

    this->solution.reinit(locally_owned_dofs, this->mpi_communicator);
    weights.reinit(locally_owned_dofs, this->mpi_communicator);
  }

  template <int dim>
  void VorticityWeightedAverage<dim>::do_postprocess(
    const LA::ParVectorType &present_solution)
  {
    this->solution = 0;
    weights        = 0;

    const FEValuesExtractors::Vector velocity_extractor(this->ordering.u_lower);

    const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
    Vector<double>     local_rhs(dofs_per_cell), local_weights(dofs_per_cell);

    const unsigned int n_q_points =
      this->solver_fe_values.get_quadrature().size();

    AssertDimension(n_q_points,
                    this->fe_values_vorticity->get_quadrature().size());

    std::vector<typename VorticityAtDofBase<dim>::curl_type> velocity_curls(
      n_q_points);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        local_rhs     = 0;
        local_weights = 0;

        this->fe_values_vorticity->reinit(cell);
        this->solver_fe_values.reinit(
          cell->as_dof_handler_iterator(this->solver_dof_handler));

        this->solver_fe_values[velocity_extractor].get_function_curls(
          present_solution, velocity_curls);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const double JxW  = this->fe_values_vorticity->JxW(q);
          const double curl = velocity_curls[q][0];

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            // const double shape = this->fe_values_vorticity->shape_value(i,
            // q); local_rhs(i) += curl * shape * JxW; local_weights(i) += shape
            // * JxW;
            local_rhs(i) += curl * JxW;
            local_weights(i) += JxW;
          }
        }
        cell->distribute_local_to_global(local_rhs, this->solution);
        cell->distribute_local_to_global(local_weights, weights);
      }
    this->solution.compress(VectorOperation::add);
    weights.compress(VectorOperation::add);

    // Solve the diagonal system
    const unsigned int start = (this->solution.local_range().first),
                       end   = (this->solution.local_range().second);
    for (unsigned int i = start; i < end; ++i)
    {
      // Weight is element's volume, which should be positive
      Assert(std::abs(weights(i)) > 1e-14, ExcInternalError());
      this->solution(i) /= weights(i);
    }
    this->solution.compress(VectorOperation::insert);
  }

  template class VorticityWeightedAverage<2>;
  template class VorticityWeightedAverage<3>;

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
  QCriterionWeightedAverage<dim>::QCriterionWeightedAverage(
    const ParameterReader<dim> &param,
    const ComponentOrdering    &ordering,
    const Mapping<dim>         &mapping,
    const DoFHandler<dim>      &dof_handler,
    const Quadrature<dim>      &cell_quadrature)
    : PostprocessorAtDofBase<dim>(ordering,
                                  param,
                                  mapping,
                                  dof_handler,
                                  cell_quadrature,
                                  update_gradients)
  {
    this->data_names          = {"q_criterion_dofs"};
    this->data_interpretation = {
      DataComponentInterpretation::DataComponentInterpretation::
        component_is_scalar};

    const unsigned int degree = param.postprocessing.q_criterion.degree;

    if (param.finite_elements.use_quads)
      this->fe = std::make_unique<FE_Q<dim>>(degree);
    else
      this->fe = std::make_unique<FE_SimplexP<dim>>(degree);

    this->dof_handler.distribute_dofs(*this->fe);

    fe_values = std::make_unique<FEValues<dim>>(
      mapping, *this->fe, cell_quadrature, update_values | update_JxW_values);

    reinit();
  }

  template <int dim>
  void QCriterionWeightedAverage<dim>::reinit()
  {
    const auto locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    const auto locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(this->dof_handler);

    this->solution.reinit(locally_owned_dofs, this->mpi_communicator);
    weights.reinit(locally_owned_dofs, this->mpi_communicator);
  }

  template <int dim>
  void QCriterionWeightedAverage<dim>::do_postprocess(
    const LA::ParVectorType &present_solution)
  {
    this->solution = 0;
    weights        = 0;

    const FEValuesExtractors::Vector velocity_extractor(this->ordering.u_lower);

    const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
    Vector<double>     local_rhs(dofs_per_cell), local_weights(dofs_per_cell);

    const unsigned int n_q_points =
      this->solver_fe_values.get_quadrature().size();
    AssertDimension(n_q_points, fe_values->get_quadrature().size());
    std::vector<Tensor<2, dim>> velocity_gradients(n_q_points);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        local_rhs     = 0;
        local_weights = 0;

        fe_values->reinit(cell);
        this->solver_fe_values.reinit(
          cell->as_dof_handler_iterator(this->solver_dof_handler));

        this->solver_fe_values[velocity_extractor].get_function_gradients(
          present_solution, velocity_gradients);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const double          JxW         = fe_values->JxW(q);
          const Tensor<2, dim> &grad_u      = velocity_gradients[q];
          const double          q_criterion = compute_q_criterion<dim>(grad_u);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            // const double shape = this->fe_values_vorticity->shape_value(i,
            // q); local_rhs(i) += curl * shape * JxW; local_weights(i) += shape
            // * JxW;
            local_rhs(i) += q_criterion * JxW;
            local_weights(i) += JxW;
          }
        }
        cell->distribute_local_to_global(local_rhs, this->solution);
        cell->distribute_local_to_global(local_weights, weights);
      }
    this->solution.compress(VectorOperation::add);
    weights.compress(VectorOperation::add);

    // Solve the diagonal system
    const unsigned int start = (this->solution.local_range().first),
                       end   = (this->solution.local_range().second);
    for (unsigned int i = start; i < end; ++i)
    {
      // Weight is element's volume, which should be positive
      Assert(std::abs(weights(i)) > 1e-14, ExcInternalError());
      this->solution(i) /= weights(i);
    }
    this->solution.compress(VectorOperation::insert);
  }

  template class QCriterionWeightedAverage<2>;
  template class QCriterionWeightedAverage<3>;
} // namespace PostProcessingTools
