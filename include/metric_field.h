#ifndef METRIC_FIELD_h
#define METRIC_FIELD_h

#include <deal.II/base/tensor_function.h>
#include <deal.II/numerics/data_postprocessor.h>
#include <error_estimation/solution_recovery.h>
#include <metric_tensor.h>
#include <parameter_reader.h>
#include <parameters.h>
#include <types.h>

#if defined(FEZ_WITH_MMG)
#  include <mmg/libmmg.h>
#endif

using namespace dealii;

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
   * Constructor. The parameter @p index specifies that this metric field is
   * created from the index-th metric field parameters in param.metric_fields.
   * A reference to the whole set of parameters is still kept for convenience.
   */
  MetricField(const unsigned int          index,
              const ParameterReader<dim> &param,
              const Triangulation<dim>   &triangulation);

  /**
   * Copy constructor. Deleted to avoid copying by mistake.
   */
  MetricField(const MetricField<dim> &other) = delete;

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
   * Compute integral on mesh of metric determinant.
   */
  double compute_integral_determinant(const double exponent) const;

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
  void compute_optimal_multiscale_metric();

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
   * Return the number of owned mesh vertices on all rank, i.e., the total
   * number of mesh vertices in the triangulation.
   */
  unsigned int get_n_total_owned_vertices() const;

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
   * Compute the unscaled metric Q in the formulation of the optimal metric.
   */
  void compute_anisotropic_measure();

  /**
   * Compute the unscaled metric Q assuming a linearly interpolated field.
   * In this case, Q is simply the absolute value of the Hessian matrix.
   */
  void compute_anisotropic_measure_P1();

  /**
   * Compute the unscaled metric Q assuming a quadratic FE solution.
   * This function is called in 2D only and uses J.-M. Mirebeau's analytical
   * solution. Not sure there is an exact solution in 3D.
   */
  void compute_anisotropic_measure_P2();

  /**
   * Compute the unscaled metric Q for the general case of a FE solution of
   * arbitrary polynomial degree, in 2D or 3D. Uses the log-simplex method
   * from Coulaud & Loseille [ref].
   */
  void compute_anisotropic_measure_Pn();

  /**
   * Transfer the MetricTensor<dim> stored in the metrics std::vector to their
   * components stored as dofs in local_metrics_fe, then update the ghosted
   * metrics.
   */
  void metrics_to_tensor_solution();

  /**
   * Transfer the metrics from their components represented as dofs to the
   * vector of MetricTensor<dim>.
   */
  void tensor_solution_to_metrics();

private:
  const unsigned int          index;
  const ParameterReader<dim> &param;
  const Triangulation<dim>   &triangulation;
  DoFHandler<dim>             dof_handler;

  MPI_Comm     mpi_communicator;
  unsigned int mpi_rank;

  // The FE space used for dummy computations
  std::shared_ptr<FiniteElement<dim>> fe;
  std::shared_ptr<Mapping<dim>>       mapping;

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
inline unsigned int MetricField<dim>::get_n_total_owned_vertices() const
{
  return n_total_owned_vertices;
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
