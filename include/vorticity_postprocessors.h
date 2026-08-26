#ifndef VORTICITY_POSTPROCESSORS
#define VORTICITY_POSTPROCESSORS

#include <components_ordering.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <parameter_reader.h>
#include <parameters.h>
#include <types.h>

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
   * A base class for postprocessors that compute a dof-based vorticity field.
   */
  template <int dim>
  class VorticityAtDofBase : public PostprocessorAtDofBase<dim>
  {
  public:
    using curl_type =
      typename FEValuesViews::Vector<dim>::template solution_curl_type<double>;
    static constexpr unsigned int n_components = dim == 2 ? 1 : 3;

    /**
     * Constructor.
     */
    VorticityAtDofBase(const ParameterReader<dim> &param,
                       const ComponentOrdering    &ordering,
                       const Mapping<dim>         &mapping,
                       const DoFHandler<dim>      &dof_handler,
                       const Quadrature<dim>      &cell_quadrature);

  protected:
    /**
     * FEValues used to evaluate the vorticity shape functions.
     */
    std::unique_ptr<FEValues<dim>> fe_values_vorticity;
  };

  /**
   * Compute the vorticity field with an L2 projection.
   * The function do_postprocess() solves for u_h the problem:
   *
   *   (u_h, v_h)_\Omega = (curl(u), v_h)_\Omega, for all v_h.
   *
   * The projected vorticity u_h is scalar in 2D, and vector-valued in 3D.
   *
   * When using quads/hexes, the mass matrix is made diagonal by using
   * Lagrange shape functions defined from Gauss-Lobatto-Legendre (GLL)
   * quadrature nodes, in which case solving the above system is trivial.
   *
   * With simplices, the mass matrix is not diagonal in general, but it is SPD,
   * so we can use a conjugate gradient solver to solve the system efficiently.
   */
  template <int dim>
  class VorticityL2Projection : public VorticityAtDofBase<dim>
  {
  public:
    /**
     * Constructor.
     */
    VorticityL2Projection(const ParameterReader<dim> &param,
                          const ComponentOrdering    &ordering,
                          const Mapping<dim>         &mapping,
                          const DoFHandler<dim>      &dof_handler,
                          const Quadrature<dim>      &cell_quadrature,
                          const bool                  with_moving_mesh);

  protected:
    /**
     * Compute the vorticity field: assemble and solve the L2 projection system.
     */
    virtual void
    do_postprocess(const LA::ParVectorType &present_solution) override;

    /**
     * Resize the vectors and matrix.
     */
    void reinit();

    /**
     * Assemble the mass matrix and RHS.
     * On fixed grids, the matrix is assembled only once as it will not change.
     * On moving gris, the mass matrix is assembled every time this function is
     * called.
     * TODO: heuristic to limit assembly frequency.
     */
    template <typename VectorType>
    void assemble_system(const VectorType &present_solution);

    /**
     * Solve the system (diagonal solve or CG).
     */
    void solve();

  protected:
    const bool        with_moving_mesh;
    bool              matrix_is_assembled;
    LA::ParMatrixType system_matrix;
    LA::ParVectorType system_rhs;
  };

  /**
   * Compute the vorticity field using a weighted average.
   * The function do_postprocess approximates an L2 projection and solves in 2D:
   *
   *   (int_\Omega phi_i dx) * u_i = int_\Omega curl(u) * phi_i dx,
   *
   * which amounts to the weighted average:
   *
   *                int_\Omega curl(u) * phi_i dx
   *       u_i =   ------------------------------- .
   *                    int_\Omega (phi_i) dx
   *
   * This amounts to lumping the mass matrix by summing all row entries into the
   * diagonal. For higher-order shape functions, the lumped diagonal can be zero
   * (e.g., for P2 interpolation at vertex nodes), in which case we simply
   * weight the average by the elements volume:
   *
   *                int_int_{K including i} curl(u) dx
   *       u_i =   ------------------------------------ .
   *                  int_int_{K including i} 1 dx
   *
   * In 3D, the same simple average is used for for each curl component:
   *
   *                int_int_{K including i} curl(u)_comp dx
   *       u_i =   ----------------------------------------- .
   *                    int_int_{K including i} 1 dx
   */
  template <int dim>
  class VorticityWeightedAverage : public VorticityAtDofBase<dim>
  {
  public:
    /**
     * Constructor.
     * FIXME: use with_moving_mesh to avoid recomputing the weights on fixed
     * grids.
     */
    VorticityWeightedAverage(const ParameterReader<dim> &param,
                             const ComponentOrdering    &ordering,
                             const Mapping<dim>         &mapping,
                             const DoFHandler<dim>      &dof_handler,
                             const Quadrature<dim>      &cell_quadrature);

  protected:
    /**
     * Compute the vorticity field: sum the contributions and weights at the
     * nodes.
     */
    virtual void
    do_postprocess(const LA::ParVectorType &present_solution) override;

    /**
     * Resize the vectors.
     */
    void reinit();

  protected:
    /**
     * Weights associated with each node.
     */
    LA::ParVectorType weights;
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
   * Compute the Q-criterion field using a weighted average.
   * Similarly to the vorticity, do_postprocess() computes the simple average
   * weighted by the elements volume:
   *
   *                   int_{K including i} Q dx
   *          q_i =   -------------------------- .
   *                   int_{K including i} 1 dx
   */
  template <int dim>
  class QCriterionWeightedAverage : public PostprocessorAtDofBase<dim>
  {
  public:
    /**
     * Constructor.
     */
    QCriterionWeightedAverage(const ParameterReader<dim> &param,
                              const ComponentOrdering    &ordering,
                              const Mapping<dim>         &mapping,
                              const DoFHandler<dim>      &dof_handler,
                              const Quadrature<dim>      &cell_quadrature);

  protected:
    /**
     * Compute the Q-criterion field by averaging the evaluation on the cells.
     */
    virtual void
    do_postprocess(const LA::ParVectorType &present_solution) override;

    /**
     * Resize the vectors.
     */
    void reinit();

  protected:
    /**
     * FEValues used to evaluate the Q-criterion shape functions.
     */
    std::unique_ptr<FEValues<dim>> fe_values;

    /**
     * Weights associated with each node.
     */
    LA::ParVectorType weights;
  };
} // namespace PostProcessingTools

#endif
