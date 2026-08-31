#ifndef FIELD_POSTPROCESSORS_H
#define FIELD_POSTPROCESSORS_H

#include <components_ordering.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_update_flags.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_values_views.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_component_interpretation.h>
#include <parameter_reader.h>
#include <post_processing_tools.h>
#include <time_handler.h>

namespace PostProcessingTools
{
  /**
   * List of the available classes derived from deal.II's DataPostprocessor.
   */
  enum class PostprocessorTypes
  {
    vorticity,
    q_criterion
  };

  /**
   * List of the available dof-based postprocessors (see PostprocessorAtDofBase)
   */
  enum class PostprocessorAtDofTypes
  {
    vorticity,
    q_criterion,
    mesh_velocity,
    density,
    mobility,
    mff_physics_compression,
    mff_enlarged_compression,
    mff_transport
  };

  /**
   * Convert a PostprocessorAtDofTypes to a string.
   */
  inline std::string to_string(const PostprocessorAtDofTypes type)
  {
    switch (type)
    {
      case PostprocessorAtDofTypes::vorticity:
        return "vorticity";
      case PostprocessorAtDofTypes::q_criterion:
        return "Q criterion";
      case PostprocessorAtDofTypes::mesh_velocity:
        return "mesh velocity";
      case PostprocessorAtDofTypes::density:
        return "density";
      case PostprocessorAtDofTypes::mobility:
        return "mobility";
      case PostprocessorAtDofTypes::mff_physics_compression:
        return "MFF physics compression";
      case PostprocessorAtDofTypes::mff_enlarged_compression:
        return "MFF enlarged compression";
      case PostprocessorAtDofTypes::mff_transport:
        return "MFF transport";
    }
    // Cannot reach here
    DEAL_II_ASSERT_UNREACHABLE();
    return "unknown";
  }

  /**
   * A base class enabling the computation of a postprocessed field with a
   * dof-based representation, with its own dof handler.
   *
   * This class exists to allow postprocessing only a subset of the main
   * solver's components. For instance, if the full solution vector contains
   * dofs for the velocity, pressure, and potentially other fields,
   * postprocessing, say, the vorticity only, is a bit awkward because naively
   * outputting a dof-based field requires providing data for *all* dofs in the
   * main solver's dof handler. This class instead stores its own smaller dof
   * handler, and adds its data to a PostProcessingHandler.
   *
   * Note that DataPostprocessor do not have this issue,
   * since they evaluate the postprocessed data directly at the visualization
   * points, and not at the degrees of freedom.
   */
  template <int dim>
  class PostprocessorAtDofBase
  {
  public:
    /**
     * Constructor.
     */
    PostprocessorAtDofBase(
      const ComponentOrdering        &ordering,
      const ParameterReader<dim>     &param,
      const Mapping<dim>             &mapping,
      const DoFHandler<dim>          &dof_handler,
      const Quadrature<dim>          &cell_quadrature,
      const UpdateFlags               flags,
      const std::vector<std::string> &names,
      const std::vector<
        DataComponentInterpretation::DataComponentInterpretation>
        &component_interpretation);

    /**
     * Postprocess the @p present_solution vector. This function fills the
     * underlying solution vector.
     *
     * Ideally, this function would be a pure virtual function template, to
     * allow each derived class to implement its own postprocess function
     * templatized on VectorType. But, this is not allowed in C++, so instead
     * there is a do_postprocess function with the required concrete types for
     * @p present_solution. The derived classes can implement a templatized
     * do_postprocess, and simply forward the calls of the do_postprocess'es
     * with concrete types to this template.
     */
    template <typename VectorType>
    void postprocess(const VectorType              &present_solution,
                     const std::vector<VectorType> &previous_solutions,
                     const TimeHandler             &time_handler)
    {
      do_postprocess(present_solution, previous_solutions, time_handler);
    }

    /**
     * Add the data stored in "solution" to the DataOut of the passed
     * @p postproc_handler.
     */
    void add_data(PostProcessingHandler<dim> &postproc_handler);

    /**
     * Return the number of quadrature points in the stored FEValues.
     */
    unsigned int get_n_q_points() const
    {
      AssertDimension(solver_fe_values.get_quadrature().size(),
                      fe_values->get_quadrature().size());
      return solver_fe_values.get_quadrature().size();
    }

  protected:
    /**
     * Pure virtual function that actually computes the vorticity.
     * Must be specialized for the different vector types.
     */
    virtual void
    do_postprocess(const LA::ParVectorType              &present_solution,
                   const std::vector<LA::ParVectorType> &previous_solutions,
                   const TimeHandler                    &time_handler) = 0;

  protected:
    /**
     * Variable ordering
     */
    const ComponentOrdering &ordering;

    /**
     * Parameters
     */
    const ParameterReader<dim> &param;

    /**
     * A copy of the MPI communicator
     */
    MPI_Comm mpi_communicator;

    /**
     * Const reference to the solver's dof_handler
     */
    const DoFHandler<dim> &solver_dof_handler;

    /**
     * Dof handler for this postprocessed field only
     */
    DoFHandler<dim> dof_handler;

    /**
     * Finite element representing the postprocessed field
     */
    std::unique_ptr<FiniteElement<dim>> fe;

    /**
     * The (completely distributed) postprocessed field at owned dofs
     */
    LA::ParVectorType solution;

    /**
     * FEValues used to evaluate quantities from the main solver.
     */
    FEValues<dim> solver_fe_values;

    /**
     * FEValues associated with the postprocessed field and its FE
     */
    std::unique_ptr<FEValues<dim>> fe_values;

    /**
     * Names of each component of the data stored in "solution".
     */
    std::vector<std::string> data_names;

    /**
     * Interpretation of each component of the data stored in "solution".
     */
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_interpretation;
  };

  /**
   * This class computes the L2 projection of a generic field, whose evaluation
   * is described through the Evaluator template parameter. The do_postprocess()
   * function solves for u_h the problem:
   *
   *   (u_h, v_h)_\Omega = (f, v_h)_\Omega, for all v_h,
   *
   * where f is a quantity to project onto a finite element space. This quantity
   * is of type QuantityType, and can be either scalar- or vector-valued,
   * whereas the shape functions are of type ShapeType. The Evaluator object is
   * responsible for evaluating the quantity of interest, through a function
   * evaluate_quantity(...) (see e.g. the VorticityEvaluator class template for
   * a list of the functions that an Evaluator must provide).
   *
   * When using quads/hexes, the mass matrix on the left-hand side is made
   * diagonal by using Lagrange shape functions defined from
   * Gauss-Lobatto-Legendre (GLL) quadrature nodes. In that case, solving the
   * above system is trivial.
   *
   * With simplices, the mass matrix is not diagonal in general, but it is SPD,
   * so we can use a conjugate gradient solver to solve the system efficiently.
   */
  template <int dim,
            int n_components,
            typename Evaluator,
            typename QuantityType,
            typename ShapeType>
  class L2Projection : public PostprocessorAtDofBase<dim>
  {
  public:
    /**
     * Constructor.
     */
    L2Projection(const Evaluator            &evaluator,
                 const ComponentOrdering    &ordering,
                 const ParameterReader<dim> &param,
                 const Mapping<dim>         &mapping,
                 const DoFHandler<dim>      &dof_handler,
                 const Quadrature<dim>      &cell_quadrature,
                 const bool                  with_moving_mesh);

    /**
     * Compute the projection: assemble and solve the system.
     */
    virtual void
    do_postprocess(const LA::ParVectorType              &present_solution,
                   const std::vector<LA::ParVectorType> &previous_solutions,
                   const TimeHandler                    &time_handler) override;

  protected:
    /**
     * Allocate the system matrix, RHS and solution vector.
     */
    void reinit();

    /**
     * Assemble the mass matrix and RHS.
     */
    template <typename VectorType>
    void assemble_system(const VectorType              &present_solution,
                         const std::vector<VectorType> &previous_solutions,
                         const TimeHandler             &time_handler);

    /**
     * Solve the linear system.
     */
    void solve();

  protected:
    /**
     * An object responsible of evaluating the quantity of interest at
     * quadrature points, as well as providing some data relative to the
     * projected field (i.e., the degree of the finite element representation,
     * etc.).
     */
    Evaluator evaluator;

    /**
     * True if using a moving mesh, in which case the mass matrix is recomputed
     * at each projection.
     */
    const bool with_moving_mesh;

    /**
     * This flag specified if the mass matrix has been assembled at least once.
     * On fixed grids, the matrix is not reassembled afterwards.
     */
    bool matrix_is_assembled;

    /**
     * Matrix and RHS of the L2 projection system.
     */
    LA::ParMatrixType system_matrix;
    LA::ParVectorType system_rhs;
  };

  /**
   * This class computes the weighted average of a generic field, whose
   * evaluation is described through the Evaluator template parameter. The
   * do_postprocess() function approximates an L2 projection through row-sum
   * mass lumping, i.e., the diagonal entry of the mass matrix is replaced by
   * the sum of the elements of its line. This amounts to solve:
   *
   *          (int_\Omega phi_i dx) * u_i = int_\Omega f * phi_i dx
   *
   * for a scalar-valued projected field f, and
   *
   *  (int_\Omega phi_i \cdot [1, ..., 1]^T dx) * u_i = int_\Omega f \cdot phi_i
   * dx
   *
   * for vector-valued fields, assuming Lagrange vector-valued shape functions.
   * These amount to the weighted averages:
   *
   *                             int_\Omega f * phi_i dx
   *                    u_i =   --------------------------
   *                               int_\Omega phi_i dx
   *
   * for scalar fields, and for vector-valued fields:
   *
   *                             int_\Omega f \cdot phi_i dx
   *             u_i =   -------------------------------------------- .
   *                        int_\Omega phi_i \cdot [1, ..., 1]^T dx
   *
   * For higher-order elements, the weights may be zero (e.g., for P2
   * interpolation at vertex nodes), in which case the system is singular, and
   * we instead compute a simple average weighted by the elements volume:
   *
   *                             int_{K including i} f dx
   *                    u_i =   ------------------------------
   *                             int_{K including i} 1 dx
   *
   * for scalar fields, and for vector fields:
   *
   *                            int_{K including i} f_comp dx
   *                   u_i =   -------------------------------- .
   *                              int_{K including i} 1 dx
   */
  template <int dim,
            int n_components,
            typename Evaluator,
            typename QuantityType,
            typename ShapeType>
  class WeightedAverage : public PostprocessorAtDofBase<dim>
  {
  public:
    /**
     * Constructor.
     */
    WeightedAverage(const Evaluator            &evaluator,
                    const ComponentOrdering    &ordering,
                    const ParameterReader<dim> &param,
                    const Mapping<dim>         &mapping,
                    const DoFHandler<dim>      &dof_handler,
                    const Quadrature<dim>      &cell_quadrature,
                    const bool                  with_moving_mesh);

    /**
     * Compute the weighted average.
     */
    virtual void
    do_postprocess(const LA::ParVectorType              &present_solution,
                   const std::vector<LA::ParVectorType> &previous_solutions,
                   const TimeHandler                    &time_handler) override;

  protected:
    /**
     * Allocate vectors.
     */
    void reinit();

  protected:
    /**
     * An object responsible of evaluating the quantity of interest at
     * quadrature points, as well as providing some data relative to the
     * projected field (i.e., the degree of the finite element representation,
     * etc.).
     */
    Evaluator evaluator;

    /**
     * True if using a moving mesh, in which case the vector of weights is
     * recomputed at each assembly.
     */
    const bool with_moving_mesh;

    /**
     * Vector of weights.
     */
    LA::ParVectorType weights;
  };

  /**
   * Given an Evaluator, which describes the quantity to postprocess, and a
   * ProjectionBase, which describes what kind of projection to use, this class
   * produces a postprocessor.
   *
   * An example call to generate a postprocessor for the vorticity using an L2
   * projection is for example:
   *
   * FieldPostprocessorGenerator<dim, VorticityEvaluator, L2Projection>
   *     my_postprocessor(..);
   * my_postprocessor.postprocess(solution);
   * my_postprocessor.add_data(postproc_handler);
   *
   */
  template <int dim,
            template <int>
            class EvaluatorType,
            template <int, unsigned int, typename, typename, typename>
            class ProjectionBaseType>
  class FieldPostprocessorGenerator
    : public ProjectionBaseType<dim,
                                EvaluatorType<dim>::n_components,
                                EvaluatorType<dim>,
                                typename EvaluatorType<dim>::quantity_type,
                                typename EvaluatorType<dim>::shape_type>
  {
  public:
    /**
     * Constructor.
     */
    FieldPostprocessorGenerator(const ParameterReader<dim> &param,
                                const ComponentOrdering    &ordering,
                                const Mapping<dim>         &mapping,
                                const DoFHandler<dim>      &dof_handler,
                                const Quadrature<dim>      &cell_quadrature,
                                const bool                  with_moving_mesh);
  };

  /* ---------------- Template functions ----------------- */

  template <int dim>
  PostprocessorAtDofBase<dim>::PostprocessorAtDofBase(
    const ComponentOrdering        &ordering,
    const ParameterReader<dim>     &param,
    const Mapping<dim>             &mapping,
    const DoFHandler<dim>          &solver_dof_handler,
    const Quadrature<dim>          &cell_quadrature,
    const UpdateFlags               flags,
    const std::vector<std::string> &names,
    const std::vector<DataComponentInterpretation::DataComponentInterpretation>
      &component_interpretation)
    : ordering(ordering)
    , param(param)
    , mpi_communicator(solver_dof_handler.get_mpi_communicator())
    , solver_dof_handler(solver_dof_handler)
    , dof_handler(solver_dof_handler.get_triangulation())
    , solver_fe_values(mapping,
                       solver_dof_handler.get_fe(),
                       cell_quadrature,
                       flags)
    , data_names(names)
    , data_interpretation(component_interpretation)
  {}

  template <int dim>
  void PostprocessorAtDofBase<dim>::add_data(
    PostProcessingHandler<dim> &postproc_handler)
  {
    postproc_handler.add_data_vector(dof_handler,
                                     solution,
                                     data_names,
                                     data_interpretation);
  }

  template <int dim,
            int n_components,
            typename Evaluator,
            typename QuantityType,
            typename ShapeType>
  L2Projection<dim, n_components, Evaluator, QuantityType, ShapeType>::
    L2Projection(const Evaluator            &evaluator,
                 const ComponentOrdering    &ordering,
                 const ParameterReader<dim> &param,
                 const Mapping<dim>         &mapping,
                 const DoFHandler<dim>      &dof_handler,
                 const Quadrature<dim>      &cell_quadrature,
                 const bool                  with_moving_mesh)
    : PostprocessorAtDofBase<dim>(ordering,
                                  param,
                                  mapping,
                                  dof_handler,
                                  param.finite_elements.use_quads ?
                                    QGaussLobatto<dim>(evaluator.param.degree +
                                                       1) :
                                    cell_quadrature,
                                  evaluator.get_main_solver_update_flags(),
                                  evaluator.get_data_names(),
                                  evaluator.get_components_interpretation())
    , evaluator(evaluator)
    , with_moving_mesh(with_moving_mesh)
    , matrix_is_assembled(false)
  {
    const unsigned int degree = evaluator.param.degree;

    if constexpr (n_components == 1)
    {
      if (param.finite_elements.use_quads)
        this->fe = std::make_unique<FE_Q<dim>>(QGaussLobatto<1>(degree + 1));
      else
        this->fe = std::make_unique<FE_SimplexP<dim>>(degree);
    }
    else
    {
      if (param.finite_elements.use_quads)
        this->fe = std::make_unique<FESystem<dim>>(
          FE_Q<dim>(QGaussLobatto<1>(degree + 1)) ^ n_components);
      else
        this->fe = std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(degree) ^
                                                   n_components);
    }
    this->dof_handler.distribute_dofs(*this->fe);

    if (param.finite_elements.use_quads)
      // Use a dim-dimensional GLL quadrature matching the one used to create
      // the finite element space, yielding a diagonal mass matrix.
      //
      // Important: this quadrature rule has to match the one used in the
      // initializer list, as it is also used to create the FEValues
      // associated with the main solver.
      this->fe_values =
        std::make_unique<FEValues<dim>>(mapping,
                                        *this->fe,
                                        QGaussLobatto<dim>(degree + 1),
                                        update_values | update_JxW_values);
    else
      this->fe_values = std::make_unique<FEValues<dim>>(
        mapping, *this->fe, cell_quadrature, update_values | update_JxW_values);

    // Allocate matrix and vectors
    reinit();
    this->evaluator.reinit(*this);
  }

  template <int dim,
            int n_components,
            typename Evaluator,
            typename QuantityType,
            typename ShapeType>
  void
  L2Projection<dim, n_components, Evaluator, QuantityType, ShapeType>::reinit()
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

  template <int dim,
            int n_components,
            typename Evaluator,
            typename QuantityType,
            typename ShapeType>
  template <typename VectorType>
  void L2Projection<dim, n_components, Evaluator, QuantityType, ShapeType>::
    assemble_system(const VectorType              &present_solution,
                    const std::vector<VectorType> &previous_solutions,
                    const TimeHandler             &time_handler)
  {
    using extractor_type = std::conditional_t<n_components == 1,
                                              FEValuesExtractors::Scalar,
                                              FEValuesExtractors::Vector>;

    const bool assemble_matrix = with_moving_mesh or !matrix_is_assembled;

    system_rhs = 0;
    if (assemble_matrix)
      system_matrix = 0;

    const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    Vector<double>     local_rhs(dofs_per_cell);

    std::vector<ShapeType> phi_u(dofs_per_cell);

    const unsigned int n_q_points = this->fe_values->get_quadrature().size();

    AssertDimension(n_q_points, this->solver_fe_values.get_quadrature().size());

    std::vector<QuantityType> quantity_to_project(n_q_points);

    const extractor_type projection_extractor(0);

    if (assemble_matrix)
    {
      for (const auto &cell : this->dof_handler.active_cell_iterators())
        if (cell->is_locally_owned())
        {
          this->fe_values->reinit(cell);
          this->solver_fe_values.reinit(
            cell->as_dof_handler_iterator(this->solver_dof_handler));

          local_matrix = 0;
          local_rhs    = 0;

          // Quantity to project evaluated at quadrature nodes
          evaluator.evaluate_quantity(present_solution,
                                      previous_solutions,
                                      time_handler,
                                      this->solver_fe_values,
                                      quantity_to_project);

          for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double    JxW = this->fe_values->JxW(q);
            const ShapeType quantity =
              evaluator.get_quantity(quantity_to_project, q);

            for (unsigned int k = 0; k < dofs_per_cell; ++k)
              phi_u[k] = (*this->fe_values)[projection_extractor].value(k, q);

            if (this->param.finite_elements.use_quads)
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                // RHS
                local_rhs(i) += (phi_u[i] * quantity) * JxW;
                // Diagonal mass matrix
                local_matrix(i, i) += phi_u[i] * phi_u[i] * JxW;
              }
            }
            else
            {
              for (unsigned int i = 0; i < dofs_per_cell; ++i)
              {
                // RHS
                local_rhs(i) += (phi_u[i] * quantity) * JxW;
                // Mass matrix
                for (unsigned int j = 0; j < dofs_per_cell; ++j)
                  local_matrix(i, j) += phi_u[j] * phi_u[i] * JxW;
              }
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
          this->fe_values->reinit(cell);
          this->solver_fe_values.reinit(
            cell->as_dof_handler_iterator(this->solver_dof_handler));
          evaluator.evaluate_quantity(present_solution,
                                      previous_solutions,
                                      time_handler,
                                      this->solver_fe_values,
                                      quantity_to_project);

          for (unsigned int q = 0; q < n_q_points; ++q)
          {
            const double    JxW = this->fe_values->JxW(q);
            const ShapeType quantity =
              evaluator.get_quantity(quantity_to_project, q);

            for (unsigned int i = 0; i < dofs_per_cell; ++i)
            {
              const ShapeType phi_u =
                (*this->fe_values)[projection_extractor].value(i, q);
              local_rhs(i) += (phi_u * quantity) * JxW;
            }
          }
          cell->distribute_local_to_global(local_rhs, system_rhs);
        }
      system_rhs.compress(VectorOperation::add);
    }
  }

  template <int dim,
            int n_components,
            typename Evaluator,
            typename QuantityType,
            typename ShapeType>
  void
  L2Projection<dim, n_components, Evaluator, QuantityType, ShapeType>::solve()
  {
    if (this->param.finite_elements.use_quads)
    {
      // When using quads, the solve is trivial as the mass matrix is diagonal
      // when using interpolation nodes at the GLL nodes.
      const unsigned int start = (this->solution.local_range().first),
                         end   = (this->solution.local_range().second);
      for (unsigned int i = start; i < end; ++i)
        this->solution(i) = system_rhs(i) / system_matrix.diag_element(i);
      this->solution.compress(VectorOperation::insert);
    }
    else
    {
      // Solve with CG
      SolverControl                          solver_control(1e5, 1e-7);
      LA::SolverCG                           cg_solver(solver_control);
      PETScWrappers::PreconditionBlockJacobi preconditioner(system_matrix);
      cg_solver.solve(system_matrix,
                      this->solution,
                      system_rhs,
                      preconditioner);

      if (Utilities::MPI::this_mpi_process(this->mpi_communicator) == 0 &&
          evaluator.param.verbosity == Parameters::Verbosity::verbose)
        std::cout << "L2 projection: " << solver_control.last_step()
                  << " CG iterations needed to obtain convergence."
                  << std::endl;
    }
  }

  template <int dim,
            int n_components,
            typename Evaluator,
            typename QuantityType,
            typename ShapeType>
  void L2Projection<dim, n_components, Evaluator, QuantityType, ShapeType>::
    do_postprocess(
      const LA::ParVectorType &present_solution,
      const std::vector<LA::ParVectorType> &previous_solutions,
      const TimeHandler                    &time_handler)
  {
    this->assemble_system(present_solution, previous_solutions, time_handler);
    this->solve();
  }

  template <int dim,
            int n_components,
            typename Evaluator,
            typename QuantityType,
            typename ShapeType>
  WeightedAverage<dim, n_components, Evaluator, QuantityType, ShapeType>::
    WeightedAverage(const Evaluator            &evaluator,
                    const ComponentOrdering    &ordering,
                    const ParameterReader<dim> &param,
                    const Mapping<dim>         &mapping,
                    const DoFHandler<dim>      &dof_handler,
                    const Quadrature<dim>      &cell_quadrature,
                    const bool                  with_moving_mesh)
    : PostprocessorAtDofBase<dim>(ordering,
                                  param,
                                  mapping,
                                  dof_handler,
                                  cell_quadrature,
                                  evaluator.get_main_solver_update_flags(),
                                  evaluator.get_data_names(),
                                  evaluator.get_components_interpretation())
    , evaluator(evaluator)
    , with_moving_mesh(with_moving_mesh)
  {
    const unsigned int degree = evaluator.param.degree;

    if constexpr (n_components == 1)
    {
      if (param.finite_elements.use_quads)
        this->fe = std::make_unique<FE_Q<dim>>(degree);
      else
        this->fe = std::make_unique<FE_SimplexP<dim>>(degree);
    }
    else
    {
      if (param.finite_elements.use_quads)
        this->fe =
          std::make_unique<FESystem<dim>>(FE_Q<dim>(degree) ^ n_components);
      else
        this->fe = std::make_unique<FESystem<dim>>(FE_SimplexP<dim>(degree) ^
                                                   n_components);
    }
    this->dof_handler.distribute_dofs(*this->fe);
    this->fe_values = std::make_unique<FEValues<dim>>(
      mapping, *this->fe, cell_quadrature, update_values | update_JxW_values);

    // Allocate vectors
    reinit();
    this->evaluator.reinit(*this);
  }

  template <int dim,
            int n_components,
            typename Evaluator,
            typename QuantityType,
            typename ShapeType>
  void WeightedAverage<dim, n_components, Evaluator, QuantityType, ShapeType>::
    reinit()
  {
    const auto locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    const auto locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(this->dof_handler);
    this->solution.reinit(locally_owned_dofs, this->mpi_communicator);
    weights.reinit(locally_owned_dofs, this->mpi_communicator);
  }

  template <int dim,
            int n_components,
            typename Evaluator,
            typename QuantityType,
            typename ShapeType>
  void WeightedAverage<dim, n_components, Evaluator, QuantityType, ShapeType>::
    do_postprocess(
      const LA::ParVectorType &present_solution,
      const std::vector<LA::ParVectorType> &previous_solutions,
      const TimeHandler                    &time_handler)
  {
    using extractor_type = std::conditional_t<n_components == 1,
                                              FEValuesExtractors::Scalar,
                                              FEValuesExtractors::Vector>;
    const extractor_type projection_extractor(0);

    this->solution = 0;
    weights        = 0;

    const unsigned int dofs_per_cell = this->fe->dofs_per_cell;
    Vector<double>     local_rhs(dofs_per_cell), local_weights(dofs_per_cell);

    std::vector<ShapeType> phi_u(dofs_per_cell);

    Tensor<1, dim> ones;
    for (unsigned int d = 0; d < dim; ++d)
      ones[d] = 1.;

    const unsigned int n_q_points = this->fe_values->get_quadrature().size();
    AssertDimension(n_q_points, this->solver_fe_values.get_quadrature().size());
    std::vector<QuantityType> quantity_to_average(n_q_points);

    for (const auto &cell : this->dof_handler.active_cell_iterators())
      if (cell->is_locally_owned())
      {
        local_rhs     = 0;
        local_weights = 0;

        this->fe_values->reinit(cell);
        this->solver_fe_values.reinit(
          cell->as_dof_handler_iterator(this->solver_dof_handler));

        // Quantity to project evaluated at quadrature nodes
        evaluator.evaluate_quantity(present_solution,
                                    previous_solutions,
                                    time_handler,
                                    this->solver_fe_values,
                                    quantity_to_average);

        for (unsigned int q = 0; q < n_q_points; ++q)
        {
          const double    JxW = this->fe_values->JxW(q);
          const ShapeType quantity =
            evaluator.get_quantity(quantity_to_average, q);

          for (unsigned int i = 0; i < dofs_per_cell; ++i)
          {
            if (evaluator.param.degree == 1)
            {
              const ShapeType shape =
                (*this->fe_values)[projection_extractor].value(i, q);
              local_rhs(i) += (quantity * shape) * JxW;

              if constexpr (std::is_same_v<ShapeType, double>)
                local_weights(i) += shape * JxW;
              else if constexpr (std::is_same_v<ShapeType, Tensor<1, dim>>)
                local_weights(i) += shape * ones * JxW;
              else
                DEAL_II_NOT_IMPLEMENTED();
            }
            else
            {
              // Otherwise, average the scalar field or the component of the
              // vector field.
              if constexpr (std::is_same_v<ShapeType, double>)
                local_rhs(i) += quantity * JxW;
              else if constexpr (std::is_same_v<ShapeType, Tensor<1, dim>>)
              {
                const unsigned int comp =
                  this->fe->system_to_component_index(i).first;
                local_rhs(i) += quantity[comp] * JxW;
              }
              else
                DEAL_II_NOT_IMPLEMENTED();

              local_weights(i) += JxW;
            }
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
      Assert(std::abs(weights(i)) > 1e-14, ExcInternalError());
      this->solution(i) /= weights(i);
    }
    this->solution.compress(VectorOperation::insert);
  }

  template <int dim,
            template <int>
            class EvaluatorType,
            template <int, unsigned int, typename, typename, typename>
            class ProjectionBaseType>
  FieldPostprocessorGenerator<dim, EvaluatorType, ProjectionBaseType>::
    FieldPostprocessorGenerator(const ParameterReader<dim> &param,
                                const ComponentOrdering    &ordering,
                                const Mapping<dim>         &mapping,
                                const DoFHandler<dim>      &dof_handler,
                                const Quadrature<dim>      &cell_quadrature,
                                const bool                  with_moving_mesh)
    : ProjectionBaseType<dim,
                         EvaluatorType<dim>::n_components,
                         EvaluatorType<dim>,
                         typename EvaluatorType<dim>::quantity_type,
                         typename EvaluatorType<dim>::shape_type>(
        EvaluatorType<dim>(ordering, param),
        ordering,
        param,
        mapping,
        dof_handler,
        cell_quadrature,
        with_moving_mesh)
  {}
} // namespace PostProcessingTools

#endif
