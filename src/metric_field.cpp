
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/data_out.h>
#include <metric_field.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <random>

template <int dim>
MetricField<dim>::MetricField(const Triangulation<dim> &mesh)
  : triangulation(mesh)
{
  const unsigned int n_vertices = triangulation.n_vertices();
  _metrics.resize(n_vertices);

  for (auto &tensor : _metrics)
  {
    tensor       = unit_symmetric_tensor<dim>();
    tensor[0][0] = 0.1;
  }
}

template <int dim>
void MetricField<dim>::computeMetrics()
{
  const double hMin = 1e-10;
  const double hMax = 1.;

  const double lMin = 1. / (hMax * hMax);
  const double lMax = 1. / (hMin * hMin);

  // Compute raw metrics
  this->computeMetricsP1();

  // Bound eigenvalues
  for (auto &metric : _metrics)
    metric.boundEigenvalues(lMin, lMax);
}

template <int dim>
double MetricField<dim>::computeIntegralDeterminant() const
{
  QGauss<dim>     quadrature_formula(2);
  FE_Q<dim>       fe(1);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  FEValues<dim> fe_values(fe,
                          quadrature_formula,
                          update_quadrature_points | update_JxW_values);

  double integral = 0.0;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    fe_values.reinit(cell);
    const auto &q_points = fe_values.get_quadrature_points();
    const auto &JxW      = fe_values.get_JxW_values();

    for (unsigned int q = 0; q < q_points.size(); ++q)
    {
      const Point<dim> &qp       = q_points[q];
      double            min_dist = std::numeric_limits<double>::max();
      unsigned int      closest_vertex_index = 0;

      for (unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        const Point<dim> &vertex = cell->vertex(v);
        double            dist   = vertex.distance(qp);
        if (dist < min_dist)
        {
          min_dist             = dist;
          closest_vertex_index = cell->vertex_index(v);
        }
      }

      const Tensor<2, dim> &tensor = _metrics[closest_vertex_index];
      const double          det    = determinant(tensor);
      integral += det * JxW[q];
    }
  }

  return integral;
}

template <int dim>
bool gradationOnEdge(const Point<dim>  &p,
                     const Point<dim>  &q,
                     const double       gradation,
                     const double       relativeTolerance,
                     MetricTensor<dim> &Mp,
                     MetricTensor<dim> &Mq)
{
  bool metricChanged = false;

  // Span Mp to q, intersect and check if Mq needs to be reduced
  MetricTensor<dim> MpAtq = Mp.spanMetric(gradation, q - p);
  MpAtq                   = Mq.intersection(MpAtq);

  const double relative_norm_q = (MpAtq - Mq).norm() / Mq.norm();

  if (relative_norm_q > relativeTolerance)
  {
    Mq            = MpAtq;
    metricChanged = true;
  };

  // Idem for Mq at p
  MetricTensor<dim> MqAtp = Mq.spanMetric(gradation, p - q);
  MqAtp                   = Mp.intersection(MqAtp);

  const double relative_norm_p = (MqAtp - Mp).norm() / Mp.norm();

  if (relative_norm_p > relativeTolerance)
  {
    Mp            = MqAtp;
    metricChanged = true;
  };

  return metricChanged;
}

template <int dim>
void MetricField<dim>::metricGradation(const double       gradation,
                                       const unsigned int maxIteration,
                                       const double       tolerance)
{
  DoFHandler<dim>                dof_handler(triangulation);
  const std::vector<Point<dim>> &vertices = triangulation.get_vertices();

  // Create and shuffle set of edges
  std::set<std::pair<unsigned int, unsigned int>> edges_set;

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    unsigned int n_edges = cell->n_lines();

    for (unsigned int e = 0; e < n_edges; ++e)
    {
      const auto  &edge = cell->line(e);
      unsigned int v0   = edge->vertex_index(0);
      unsigned int v1   = edge->vertex_index(1);

      if (v0 > v1)
        std::swap(v0, v1);

      edges_set.insert(std::make_pair(v0, v1));
    }
  }

  // Convert to vector and shuffle in a reproducible way
  std::vector<std::pair<unsigned int, unsigned int>> edges(edges_set.begin(),
                                                           edges_set.end());

  const unsigned int seed = 42;
  std::mt19937       rng(seed);
  std::shuffle(edges.begin(), edges.end(), rng);

  unsigned int iter       = 0, numCorrected;
  bool         correction = true;

  while (correction && iter < maxIteration)
  {
    numCorrected = 0;
    correction   = false;
    iter++;
    for (const auto &edge : edges)
    {
      const Point<dim> &p = vertices[edge.first];
      const Point<dim> &q = vertices[edge.second];

      MetricTensor<dim> &Mp = _metrics[edge.first];
      MetricTensor<dim> &Mq = _metrics[edge.second];

      if (gradationOnEdge(p, q, gradation, tolerance, Mp, Mq))
      {
        numCorrected++;
        correction = true;
      };
    }
    std::cout << "Metric gradation: Sweep " << iter << " - Corrected "
              << numCorrected << " edges" << std::endl;
  }
}

template <int dim>
void MetricField<dim>::intersectWith(const MetricField<dim> &otherField)
{
  const unsigned int nMetrics = this->_metrics.size();
  AssertThrow(otherField._metrics.size() == nMetrics,
              ExcDimensionMismatch(otherField._metrics.size(), nMetrics));
  for (unsigned int i = 0; i < nMetrics; ++i)
  {
    MetricTensor<dim> &metric = _metrics[i];
    metric                    = metric.intersection(otherField._metrics[i]);
  }
}

/**
 * For now the metrics are written using a FESystem with dim x dim
 * components. When I know more about deal.ii, I'll get back to this
 * with a better solution (-:
 */
template <int dim>
void MetricField<dim>::writeToVTU(const std::string &filename,
                                  bool               exportInverseMetrics) const
{
  // Step 1: Set up FE system with dim*dim scalar components
  FESystem<dim>   fe(FE_SimplexP<dim>(1), dim * dim);
  DoFHandler<dim> dof_handler(triangulation);
  dof_handler.distribute_dofs(fe);

  AssertDimension(_metrics.size(), triangulation.n_vertices());

  // Step 2: Create the output vector with size = dof_handler.n_dofs()
  Vector<double> tensor_data(dof_handler.n_dofs());

  // Step 3: Assign tensor components at each vertex to appropriate component
  // DoFs
  std::vector<types::global_dof_index> local_dof_indices(fe.dofs_per_cell);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    cell->get_dof_indices(local_dof_indices);

    const unsigned int n_vertices_per_cell = cell->n_vertices();

    for (unsigned int v = 0; v < n_vertices_per_cell; ++v)
    {
      const unsigned int global_vertex_index = cell->vertex_index(v);

      const SymmetricTensor<2, dim> &tensor     = _metrics[global_vertex_index];
      SymmetricTensor<2, dim>        inv_tensor = invert(tensor);

      for (unsigned int c = 0; c < dim; ++c)
      {
        for (unsigned int d = 0; d < dim; ++d)
        {
          const unsigned int tensor_comp = c * dim + d;
          const unsigned int fe_comp =
            fe.component_to_system_index(tensor_comp, v);

          if (exportInverseMetrics)
          {
            tensor_data[local_dof_indices[fe_comp]] = inv_tensor[c][d];
          }
          else
          {
            tensor_data[local_dof_indices[fe_comp]] = tensor[c][d];
          }
        }
      }
    }
  }

  // Step 4: Define component names and interpretation
  std::vector<std::string> names;
  std::vector<DataComponentInterpretation::DataComponentInterpretation>
    interpretation;

  for (unsigned int i = 0; i < dim; ++i)
    for (unsigned int j = 0; j < dim; ++j)
    {
      names.emplace_back("m" + std::to_string(i) + std::to_string(j));
      interpretation.emplace_back(
        DataComponentInterpretation::component_is_part_of_tensor);
    }

  // Step 5: Output using DataOut
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(tensor_data,
                           names,
                           DataOut<dim>::type_dof_data,
                           interpretation);
  data_out.build_patches();

  std::ofstream output(filename);
  data_out.write_vtu(output);
}

template class MetricField<2>;
template class MetricField<3>;
