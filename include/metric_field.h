#ifndef METRIC_FIELD_h
#define METRIC_FIELD_h

#include <deal.II/lac/vector.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <error_estimation/solution_recovery.h>
#include <metric_tensor.h>
#include <parameter_reader.h>
#include <parameters.h>
#include <time_handler.h>
#include <types.h>

#if defined(FEZ_WITH_MMG)
#  include <mmg/libmmg.h>
#endif

using namespace dealii;

// Forward declaration
template <int dim>
class TransientFixedPointData;

/**
 * A metric field is a discrete field of MetricTensors, stored at the owned mesh
 * vertices.
 *
 * The metric field is stored in two ways:
 * - as an std::vector of MetricTensor<dim>, implicitly tied to the mesh
 * vertices,
 * - as an isoparametric FE field with dim*(dim+1)/2 vector components.
 *
 * The former allows manipulating metrics as MetricTensor<dim> and use all the
 * functions from the interface in metric_tensor.h, and the latter allows using
 * parallel vectors and their handling of ghost data.
 *
 * FIXME/ONGOING: A priori, we cannot use the interpolation functions from the
 * FE space though, because they do not guarantee that the interpolated metric
 * remains SPD, and we use the log-euclidean interpolation instead.
 * ===> but then we can store the log-metrics and interpolate them.
 */
template <int dim>
class MetricField
{
  static constexpr unsigned int n_components =
    MetricTensor<dim>::n_independent_components;

public:
  /**
   * Standard constructor, not initializing any data. After constructing an
   * object with this constructor, use reinit() to get a valid MetricField.
   */
  MetricField();

  /**
   * Constructor. The parameter @p index specifies that this metric field is
   * created from the index-th metric field parameters in param.metric_fields.
   * A reference to the whole set of parameters is still kept for convenience.
   */
  explicit MetricField(const unsigned int          index,
                       const ParameterReader<dim> &param,
                       const Triangulation<dim>   &triangulation);

  /**
   * Copy constructor. Deleted to avoid copying by mistake.
   */
  MetricField(const MetricField<dim> &other) = delete;

  /**
   * Copy operator is deleted as well.
   */
  MetricField &operator=(const MetricField<dim> &other) = delete;

  /**
   * Initialize this object from a valid (initialized) triangulation.
   */
  void reinit(const unsigned int          index,
              const ParameterReader<dim> &param,
              const Triangulation<dim>   &triangulation);

  /**
   * Clear the dof handler and index sets, and reset the vectors (both vertex
   * and FE representation of the metric field, maps from vertex data to dof
   * data, edges for gradation).
   */
  void clear();

  /**
   * Copy the metric tensors stored in @p other into the current metric field.
   * This function simply copies the values into both underlying metrics
   * representations, then updates the ghost values. Thus, it expects @p other
   * to be a MetricField initialized on the same triangulation as this field.
   */
  void copy_metrics_from(const MetricField<dim> &other);

  /**
   * Set the metrics of this field to the analytical field described by @p function.
   */
  void set_metrics_from_function(const TensorFunction<2, dim> &function);

  /**
   * Set this field to the Riemannian metric induced by the graph (x, f(x)),
   * where f(x) is a scalar-valued field. This metric is given by
   *
   * [M] = I + grad(f) \otimes grad(f), with I the identity tensor.
   *
   * Since the metrics are stored at the mesh vertices, this requires knowing
   * the gradient of f(x) at these locations, which is generally not readily
   * available in a classic finite element setting. Here, the gradient is
   * assumed to have been obtained by smoothing the solution, using the
   * ErrorEstimation::SolutionRecovery interface.
   */
  void set_induced_metric_from_graph(
    const ErrorEstimation::SolutionRecovery::Scalar<dim>
      &reconstructed_gradient);

  /**
   * Compute the unscaled metric Q in the formulation of the optimal metric,
   * then perform either of the following:
   *
   * - if the simulation is steady, set M = Q, i.e., set this metric field to
   *   the optimal field before scaling.
   *
   * - if the simulation is unsteady, increment the time integral of the
   * anisotropic measure Q for this time step. This integral, given by:
   *
   *                             /t_{i+1}
   *                     M =     |        Q(t) dt,
   *                             /t_i
   *
   *  is the optimal metric before scaling when using the transient fixed-point
   *  adaptation method.
   *
   * This function does not require a recovery operator, and is thus intended
   * only to compute the anisotropic measure from given exact derivatives. To
   * compute the metric from the smoothed derivatives of a numerical solution,
   * use the function below.
   */
  void increment_anisotropic_measure(const TimeHandler &time_handler);

  /**
   * Same as the function above, but compute the anisotropic measure from the
   * smoothed derivatives stored in recovery, which is expected to store the
   * derivatives of order up to p + 1, where p is the degree of the solution.
   */
  void increment_anisotropic_measure(
    const ErrorEstimation::SolutionRecovery::Base<dim> &recovery,
    const TimeHandler                                  &time_handler,
    const unsigned int                                  component = 0);

  /**
   * Compute integral on mesh of metric determinant.
   */
  double compute_integral_determinant(const double exponent) const;

  /**
   * Compute the measure of the given @p cell with respect to this Riemannian
   * metric field.
   *
   * The metric is interpolated inside the current cell from its nodal values
   * through the FE representation of this MetricField.
   *
   * The input @p fe_values is expected to have already been reinitialized on
   * the target cell and to provide both metric values and JxW values.
   */
  double
  compute_cell_measure(const FEValues<dim> &fe_values) const;

  /**
   * Compute the measure of the given cell edge with respect to this
   * Riemannian metric field.
   *
   * The metric is interpolated linearly along the edge from the nodal values
   * stored at the edge endpoints. This implementation is restricted to simplex
   * cells and assumes a 1D quadrature on the reference interval.
   */
  double
  compute_cell_edge_measure(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const unsigned int                                    edge_no,
    const Quadrature<1>                                  &edge_quadrature) const;

  /**
   * Compute the quality of the given cell with respect to this Riemannian
   * metric field.
   *
   * The input @p fe_values is expected to be reinitialized on @p cell.
   */
  double
  compute_cell_quality(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const FEValues<dim>                                  &fe_values,
    const Quadrature<1>                                  &edge_quadrature) const;

  /**
   * Compute a cell-wise field of metric qualities on all active cells.
   *
   * The FE representation of the metric field is synchronized from the nodal
   * metric storage before evaluating the qualities.
   *
   * Locally owned cells are assigned their metric quality, while non-owned
   * cells are filled with zero so that the returned vector can be written as
   * cell data with DataOut in parallel.
   */
  Vector<float>
  compute_cell_quality_field(const Quadrature<dim> &cell_quadrature,
                             const Quadrature<1>   &edge_quadrature);

  Vector<float>
  compute_cell_quality_field(const Mapping<dim>    &geometry_mapping,
                             const Quadrature<dim> &cell_quadrature,
                             const Quadrature<1>   &edge_quadrature);

  void
  write_cell_quality_pvtu(const std::string     &filename,
                          const Quadrature<dim> &cell_quadrature,
                          const Quadrature<1>   &edge_quadrature);

  /**
   * Compute the Riemannian metric minimizing an interpolation error estimate in
   * W^{s,p} norm, as described by F. Alauzet & A. Loseille [ref] and J.-M.
   * Mirebeau [ref]. This metric writes:
   *
   * M = c * (det Q) ^ (tau / 2 * p) * Q,
   *
   * where the metric Q is a measure of the anisotropy of the error estimate,
   * tau := p * dim / (p * ((k + 1) - s) + dim) is a parameter originating from
   * Hölder's inequality, and c is a global scaling factor.
   *
   * The metric Q is computed from the derivatives of order k+1 of the FE
   * solution, which can be either recovered numerically or given through
   * analytical derivatives callbacks.
   */
  void apply_optimal_steady_multiscale_scaling();

  /**
   * Return the vector of MetricTensors constituting this field.
   */
  const std::vector<MetricTensor<dim>> &get_metrics() const { return metrics; }

  /**
   * Apply gradation (smoothing) to the metric field.
   *
   * Metric intersection is not a commutative operation, so if the metric field
   * is graded in parallel, the result depends on the mesh partitioning.
   * The MetricField<dim> parameter "deterministic" specifies whether gradation
   * should be done so as to yield a unique result for all partitions (e.g., for
   * testing) or not. If deterministic is true, then all metrics are simply
   * gathered to the root process, which then smoothes the whole field and
   * scatters back the graded metrics. If false, each rank applies the
   * prescribed gradation to its metrics, then the ghosts are exchanged and the
   * metrics are smoothed again, until convergence.
   *
   * FIXME: the term "deterministic" is a bit of a misnomer, because the result
   * is always reproducible and deterministic, it should probably be changed to
   * "partition independent".
   */
  void apply_gradation();

  /**
   * Intersect all the metrics in this field with the metrics in @p other.
   * The number of metric tensors in both fields must match, but aside from
   * this, no other safety check is performed.
   */
  void intersect_with(const MetricField<dim> &other);

  /**
   * Gather the metrics and their associated mesh vertex to the root process.
   */
  std::vector<std::pair<Point<dim>, MetricTensor<dim>>> gather_metrics() const;

#if defined(FEZ_WITH_MMG)
  /**
   * Fill the MMG5_pSol pointer with the metric field, for mesh adaptation with
   * the MMG library. Specify that the solution is tensor-valued and assign the
   * metric at each mesh vertex.
   *
   * The input @p gathered_metrics should be the result of the function
   * gather_metrics() above.
   *
   * This operation should for now be called from the root process only, since
   * MMG is serial.
   */
  void
  set_mmg_solution(const std::vector<std::pair<Point<dim>, MetricTensor<dim>>>
                             &gathered_metrics,
                   MMG5_pMesh pointer_to_mesh,
                   MMG5_pSol  pointer_to_sol) const;
#endif

  /**
   * Return the number of owned mesh vertices on this rank.
   */
  unsigned int get_n_owned_vertices() const;

  /**
   * Return the number of owned mesh vertices on all ranks, i.e., the total
   * number of mesh vertices in the triangulation.
   */
  unsigned int get_n_total_owned_vertices() const;

  /**
   * Return the MPI communicator.
   */
  MPI_Comm get_mpi_communicator() const;

  /**
   * Return true if this field's vector of owned vertices matches
   * @p owned_vertices.
   */
  bool has_same_owned_vertices(const std::vector<bool> owned_vertices) const;

  /**
   * Write the metrics and their associated mesh vertex to the given stream,
   * in lexicographic order with respect to the associated mesh vertex.
   */
  void write_metrics(std::ostream &out = std::cout) const;

  /**
   * Write metric field to .vtu file.
   *
   * Note: In ParaView, the Tensor Glyph filter scales the tensor field
   * with the eigenvalues. If we want to visualize the unit ellipsoids,
   * then we should write M^(-1) instead of M.
   */
  void write_pvtu(const std::string &filename,
                  bool               write_inverse_metrics = true);

  void write_pvtu(const Mapping<dim> &geometry_mapping,
                  const std::string  &filename,
                  bool                write_inverse_metrics = true);

  /**
   * Multiply all the metrics in this field by @p factor, which should be
   * a positive number (this is checked in debug).
   */
  MetricField<dim> &operator*=(const double &factor);

private:
  /**
   * Create the array of edges used to apply the gradation.
   */
  void create_edges_for_gradation();

  /**
   * Apply gradation from the root process.
   */
  void apply_gradation_deterministic();

  /**
   * Apply gradation in parallel, with ghosts updates.
   */
  void apply_gradation_non_deterministic();

  /**
   * Compute the unscaled metric Q in the formulation of the optimal metric,
   * then perform either M = factor * Q or M += factor * Q, depending on @p add.
   *
   * A recovery operator is not needed if the anisotropic measure is computed
   * from the exact derivatives of a known field.
   */
  void set_to_or_add_anisotropic_measure(
    const double                                        factor,
    const bool                                          add,
    const ErrorEstimation::SolutionRecovery::Base<dim> *recovery  = nullptr,
    const unsigned int                                  component = 0);

  /**
   * Compute the unscaled metric Q assuming a linearly interpolated field.
   * In this case, Q is simply the absolute value of the Hessian matrix.
   */
  void set_to_or_add_anisotropic_measure_P1(
    const std::vector<Tensor<2, dim>> &solution_hessians,
    const double                       factor,
    const bool                         add);

  /**
   * Compute the unscaled metric Q assuming a quadratic FE solution.
   * This function is called in 2D only and uses J.-M. Mirebeau's analytical
   * solution. Not sure there is an exact solution in 3D.
   */
  void set_to_or_add_anisotropic_measure_P2();

  /**
   * Compute the unscaled metric Q for the general case of a FE solution of
   * arbitrary polynomial degree, in 2D or 3D. Uses the log-simplex method
   * from Coulaud & Loseille [ref].
   */
  void set_to_or_add_anisotropic_measure_Pn();

  /**
   * Transfer the MetricTensor<dim> stored in the metrics std::vector to their
   * components stored as dofs in local_metrics_fe, then update the ghosted
   * metrics.
   */
  void metrics_to_tensor_solution();

  double
  compute_cell_edge_measure(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const unsigned int                                    edge_no,
    const Quadrature<1>                                  &edge_quadrature,
    const Mapping<dim>                                   &geometry_mapping) const;

  double
  compute_cell_quality(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    const FEValues<dim>                                  &fe_values,
    const Quadrature<1>                                  &edge_quadrature,
    const Mapping<dim>                                   &geometry_mapping) const;

  /**
   * Transfer the metrics from their components represented as dofs to the
   * vector of MetricTensor<dim>.
   */
  void tensor_solution_to_metrics();

private:
  ObserverPointer<const ParameterReader<dim>, MetricField<dim>> param;
  ObserverPointer<const Triangulation<dim>, MetricField<dim>>   triangulation;

  DoFHandler<dim> dof_handler;

  MPI_Comm     mpi_communicator;
  unsigned int mpi_rank;

  // The FE space used for dummy computations
  std::shared_ptr<FiniteElement<dim>> fe;
  std::shared_ptr<Mapping<dim>>       mapping;

  unsigned int index;

  // The polynomial degree of the field used to compute this metric field
  unsigned int solution_polynomial_degree;

  IndexSet locally_owned_dofs;
  IndexSet locally_relevant_dofs;

  // The metrics components, stored as finite element dofs.
  // This allows to easily exchange dofs and update ghosted metrics.
  LA::ParVectorType metrics_fe;
  LA::ParVectorType local_metrics_fe;

  // Number of mesh vertices on this partition
  unsigned int n_vertices;

  // A vector of flags indicating if the mesh vertex in [0, n_vertices) is owned
  std::vector<bool> owned_vertices;
  unsigned int      n_owned_vertices;
  unsigned int      n_total_owned_vertices;

  // The stored metric tensors
  std::vector<MetricTensor<dim>> metrics;

  /**
   * Maps to go from a representation of the metric tensors to the other (dofs
   * vs full MetricTensor<dim>).
   *
   * Each entry of vertex_to_metric_dofs is an array of n_components global dof
   * indices, so that the c-th component dof can be updated from the full metric
   * at mesh vertex v as:
   *
   * local_metrics_fe[vertex_to_metric_dofs[v][c]] = metric.access_raw_entry(c)
   *
   * Each entry of metric_dofs_to_vertex is a pair {vertex index, component}, so
   * that the full metric is updated as:
   *
   * const auto &pair =
   *   metric_dofs_to_vertex[locally_relevant_dofs.index_within_set(dof)];
   * const auto v                   = pair.first;
   * const auto c                   = pair.second;
   * metrics[v].access_raw_entry(c) = metrics_fe[dof];
   */
  std::vector<std::array<types::global_dof_index, n_components>>
    vertex_to_metric_dofs;
  std::vector<std::pair<types::global_vertex_index, unsigned int>>
    metric_dofs_to_vertex;

  // If true, the metrics are gathered to the root process, then graded.
  // If false, gradation is computed in parallel, but the result depends on the
  // partitioning.
  bool deterministic_gradation;

  /**
   * Edges for metric gradation. A straightforward way to store the edge is to
   * store the global vertex indices, but these are local to each partition.
   * If gradation is done on the root process (partition independent gradation),
   * we store the full Point<dim> instead.
   *
   * FIXME: Maybe we could use lines and/or line iterators directly
   */
  std::vector<std::pair<Point<dim>, Point<dim>>>
    edges_for_deterministic_gradation;
  std::vector<std::pair<types::global_vertex_index, types::global_vertex_index>>
    edges_for_nondeterministic_gradation;

  // The class TransientFixedPointData handles the scaling of a collection of
  // metrics, and is thus granted access to the vector of metrics, to avoid
  // repeated calls to metric_to_tensor_solution to update the ghost values.
  friend class TransientFixedPointData<dim>;
};

/**
 * Postprocessor to export the n_components groups of dofs as a tensor field.
 *
 * The computed quantities are the inverse metrics, so that visualization with
 * Paraview shows the correct unit ellipsoids.
 */
template <int dim>
class MetricPostprocessor : public DataPostprocessorTensor<dim>
{
public:
  MetricPostprocessor()
    : DataPostprocessorTensor<dim>("metric", update_values)
  {}

  virtual void evaluate_vector_field(
    const DataPostprocessorInputs::Vector<dim> &input_data,
    std::vector<Vector<double>> &computed_quantities) const override
  {
    AssertDimension(input_data.solution_values.size(),
                    computed_quantities.size());

    Tensor<2, dim> res;

    for (unsigned int p = 0; p < input_data.solution_values.size(); ++p)
    {
      AssertDimension(computed_quantities[p].size(),
                      (Tensor<2, dim>::n_independent_components));
      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int e = 0; e < dim; ++e)
        {
          res[d][e] =
            input_data.solution_values
              [p][SymmetricTensor<2, dim>::component_to_unrolled_index(
                TableIndices<2>(d, e))];
        }

      // Invert the metric for visualization
      res = invert(res);

      for (unsigned int d = 0; d < dim; ++d)
        for (unsigned int e = 0; e < dim; ++e)
        {
          computed_quantities[p][Tensor<2, dim>::component_to_unrolled_index(
            TableIndices<2>(d, e))] = res[d][e];
        }
    }
  }
};

/* ---------------- template and inline functions ----------------- */

template <int dim>
inline unsigned int MetricField<dim>::get_n_owned_vertices() const
{
  return n_owned_vertices;
}

template <int dim>
inline unsigned int MetricField<dim>::get_n_total_owned_vertices() const
{
  return n_total_owned_vertices;
}

template <int dim>
inline MPI_Comm MetricField<dim>::get_mpi_communicator() const
{
  return mpi_communicator;
}

template <int dim>
inline bool MetricField<dim>::has_same_owned_vertices(
  const std::vector<bool> other_owned_vertices) const
{
  if (owned_vertices.size() != other_owned_vertices.size())
    return false;
  for (unsigned int i = 0; i < n_vertices; ++i)
    if (owned_vertices[i] != other_owned_vertices[i])
      return false;
  return true;
}

template <int dim>
inline MetricField<dim> &MetricField<dim>::operator*=(const double &d)
{
  Assert(d > 0, ExcMultiplyByNonPositive(d));
  // Multiply each metric
  for (auto &m : metrics)
    m *= d;
  // Synchronize the FE representation of the metrics
  metrics_to_tensor_solution();
  return *this;
}

#endif
