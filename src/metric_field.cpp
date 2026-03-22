
#include <deal.II/base/function_lib.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/tensor.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/numerics/data_out.h>
#include <mesh_and_dof_tools.h>
#include <metric_field.h>
#include <metric_tensor.h>
#include <metric_tensor_tools.h>
#include <parameter_reader.h>
#include <parameters.h>

#include <algorithm>
#include <fstream>
#include <limits>
#include <random>

template <int dim>
MetricField<dim>::MetricField(const ParameterReader<dim> &param,
                              const Triangulation<dim>   &triangulation)
  : param(param)
  , triangulation(triangulation)
  , dof_handler(triangulation)
  , mpi_communicator(dof_handler.get_mpi_communicator())
  , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
  , n_vertices(triangulation.n_vertices())
  , metrics(n_vertices)
  , deterministic_gradation(param.metric_fields[0].gradation.deterministic)
{
  constexpr unsigned int mapping_degree = 1;

  // Isoparametric representation to associate a metric to each mesh vertex
  if (param.finite_elements.use_quads)
  {
    fe =
      std::make_shared<FESystem<dim>>(FE_Q<dim>(mapping_degree) ^ n_components);
    mapping = std::make_shared<MappingQ<dim>>(mapping_degree);
  }
  else
  {
    fe = std::make_shared<FESystem<dim>>(FE_SimplexP<dim>(mapping_degree) ^
                                         n_components);
    mapping =
      std::make_shared<MappingFE<dim>>(FE_SimplexP<dim>(mapping_degree));
  }

  // Distribute metrics dofs and allocate vectors
  dof_handler.distribute_dofs(*fe);

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);
  metrics_fe.reinit(locally_owned_dofs,
                    locally_relevant_dofs,
                    mpi_communicator);
  local_metrics_fe.reinit(locally_owned_dofs, mpi_communicator);

  get_owned_mesh_vertices(triangulation, mpi_rank, owned_vertices);

  // Create the maps from the vector of metrics to their FE representation
  vertex_to_metric_dofs.resize(n_vertices);
  metric_dofs_to_vertex.resize(locally_relevant_dofs.n_elements(),
                               {numbers::invalid_unsigned_int,
                                numbers::invalid_unsigned_int});

  const unsigned int n_dofs_per_cell = dof_handler.get_fe().dofs_per_cell;
  std::vector<types::global_dof_index> local_dof_indices(n_dofs_per_cell);

  // Loop over owned and ghost cells
  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    cell->get_dof_indices(local_dof_indices);
    for (unsigned int v = 0; v < cell->n_vertices(); ++v)
    {
      const auto vertex_index = cell->vertex_index(v);
      for (unsigned int c = 0; c < n_components; ++c)
      {
        const unsigned int local_index =
          cell->get_fe().component_to_system_index(c, v);

        // Data to go from metrics to local_metrics_fe
        vertex_to_metric_dofs[vertex_index][c] = local_dof_indices[local_index];

        // Data to go from local_metrics_fe to metrics
        metric_dofs_to_vertex[locally_relevant_dofs.index_within_set(
          local_dof_indices[local_index])] = {vertex_index, c};
      }
    }
  }

  if constexpr (running_in_debug_mode())
  {
    /**
     * It seems that all components are either or ghosted at a support point,
     * that is, there are no metrics which own only a part of their components.
     * If it was not the case, it might be tricky to ensure that ghosted metrics
     * are updated correctly.
     *
     * For now, check here that all metrics own either all or none of their
     * components.
     */
    for (const auto &dof_array : vertex_to_metric_dofs)
    {
      unsigned int n_owned = 0;
      for (const auto dof : dof_array)
        if (locally_owned_dofs.is_element(dof))
          n_owned++;
      Assert(n_owned == 0 || n_owned == n_components,
             ExcMessage(
               "There are metrics which do not own all of their components"));
    }
  }

  // Metrics are initialized to the identity
  // If enabled, initialize to the given callback
  const auto metric_param = param.metric_fields[0];

  if (metric_param.analytical_metric.enable)
  {
    set_metrics_from_function(MetricFunctionFromComponents<dim>(
      *metric_param.analytical_metric.callback));

    // Evaluate on the dof-based metrics
    VectorTools::interpolate(*mapping,
                             dof_handler,
                             *metric_param.analytical_metric.callback,
                             local_metrics_fe);
    metrics_fe = local_metrics_fe;
  }

  // Create the edges to apply gradation
  create_edges_for_gradation();
}

template <int dim>
void MetricField<dim>::create_edges_for_gradation()
{
  if (deterministic_gradation)
  {
    using PointEdge = std::pair<Point<dim>, Point<dim>>;

    // Use Point<dim> directly instead, because the vertex indices have no
    // meaning when they're all gathered on rank 0
    std::set<PointEdge, PointEdgeComparator<dim>> local_edges_set;
    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
      for (const unsigned int l : cell->line_indices())
      {
        Point<dim> v0 = cell->line(l)->vertex(0);
        Point<dim> v1 = cell->line(l)->vertex(1);
        local_edges_set.insert({v0, v1});
      }

    // Gather to rank 0
    auto &e = edges_for_deterministic_gradation;
    e.clear();
    const auto gathered_edges =
      Utilities::MPI::gather(mpi_communicator,
                             std::vector<PointEdge>(local_edges_set.begin(),
                                                    local_edges_set.end()),
                             0);
    for (const auto &vec : gathered_edges)
      e.insert(e.end(), vec.begin(), vec.end());

    // Shuffle in a reproducible way (or sort for debug)
    std::sort(e.begin(), e.end(), PointEdgeComparator<dim>());
    e.erase(std::unique(e.begin(), e.end(), PointEdgeEquality<dim>()), e.end());

    // const unsigned int seed = 42;
    // std::mt19937       rng(seed);
    // std::shuffle(e.begin(), e.end(), rng);

    // if (mpi_rank == 0)
    // {
    //   std::cout << "Number of edges for gradation: " << e.size() <<
    //   std::endl; for (const auto &[p, q] : e)
    //   {
    //     std::cout << "edge " << p << " - " << q << std::endl;
    //   }
    // }
  }
  else
  {
    using PointEdge =
      std::pair<types::global_vertex_index, types::global_vertex_index>;

    // Create the set of edges used to apply the target gradation
    std::set<PointEdge> local_edges_set;
    for (const auto &cell : dof_handler.active_cell_iterators())
      // if (cell->is_locally_owned() || cell->is_ghost())
      for (const unsigned int l : cell->line_indices())
      {
        unsigned int v0 = cell->line(l)->vertex_index(0);
        unsigned int v1 = cell->line(l)->vertex_index(1);
        if (v0 > v1)
          std::swap(v0, v1);
        local_edges_set.insert({v0, v1});
      }

    // Convert to vector and shuffle in a reproducible way
    auto &e = edges_for_nondeterministic_gradation;
    e = std::vector<PointEdge>(local_edges_set.begin(), local_edges_set.end());

    // std::sort(e.begin(), e.end(),
    // PointEdgeCoordComparator<dim>(triangulation.get_vertices()));

    const unsigned int seed = 42;
    std::mt19937       rng(seed);
    std::shuffle(e.begin(), e.end(), rng);

    // if (mpi_rank == 0)
    // {
    //   std::cout << "Number of edges for gradation: " << e.size() <<
    //   std::endl; for (const auto &[vp, vq] : e)
    //   {
    //     std::cout << "edge " << triangulation.get_vertices()[vp] << " - "
    //               << triangulation.get_vertices()[vq] << std::endl;
    //   }
    // }
  }
}

template <int dim>
void MetricField<dim>::metrics_to_tensor_solution()
{
  AssertDimension(vertex_to_metric_dofs.size(), n_vertices);

  for (types::global_vertex_index v = 0; v < n_vertices; ++v)
  {
    const auto &m    = metrics[v];
    const auto &dofs = vertex_to_metric_dofs[v];
    for (unsigned int c = 0; c < n_components; ++c)
      if (locally_owned_dofs.is_element(dofs[c]))
        local_metrics_fe[dofs[c]] = m.access_raw_entry(c);
  }
  local_metrics_fe.compress(VectorOperation::insert);
  metrics_fe = local_metrics_fe;
}

template <int dim>
void MetricField<dim>::tensor_solution_to_metrics()
{
  AssertDimension(metric_dofs_to_vertex.size(),
                  locally_relevant_dofs.n_elements());

  for (const auto dof : locally_relevant_dofs)
  {
    const auto &pair =
      metric_dofs_to_vertex[locally_relevant_dofs.index_within_set(dof)];
    const auto v                   = pair.first;  // Mesh vertex index
    const auto c                   = pair.second; // Tensor component
    metrics[v].access_raw_entry(c) = metrics_fe[dof];
  }
}

template <int dim>
void MetricField<dim>::set_metrics_from_function(
  const TensorFunction<2, dim> &function)
{
  const std::vector<Point<dim>> &vertices = triangulation.get_vertices();
  const std::vector<bool> &used_vertices  = triangulation.get_used_vertices();

  AssertDimension(vertices.size(), n_vertices);
  AssertDimension(used_vertices.size(), n_vertices);

  for (unsigned int i = 0; i < n_vertices; ++i)
  {
    if (used_vertices[i])
    {
      metrics[i] = symmetrize(function.value(vertices[i]));
    }
  }
}

template <int dim>
void MetricField<dim>::apply_gradation()
{
  if (deterministic_gradation)
    apply_gradation_deterministic();
  else
    apply_gradation_non_deterministic();
}

template <int dim>
void MetricField<dim>::apply_gradation_deterministic()
{
  // Gather all metrics to the root process
  using MessageType = std::pair<Point<dim>, MetricTensor<dim>>;

  const std::vector<Point<dim>> &vertices = triangulation.get_vertices();

  std::vector<MessageType>                local_metrics;
  std::vector<types::global_vertex_index> indices(
    n_vertices, numbers::invalid_unsigned_int);
  unsigned int count = 0;
  for (unsigned int i = 0; i < n_vertices; ++i)
    if (owned_vertices[i])
    {
      local_metrics.push_back({vertices[i], metrics[i]});
      indices[i] = count++;
    }

  std::vector<std::vector<MessageType>> gathered_metrics =
    Utilities::MPI::gather(mpi_communicator, local_metrics, 0);

  if (mpi_rank == 0)
  {
    // Store pointers to the metrics
    std::map<Point<dim>, MetricTensor<dim> *, PointComparator<dim>> all_metrics;
    for (auto &vec : gathered_metrics)
      for (auto &[pt, metric] : vec)
        all_metrics[pt] = &metric;

    const auto &metric_param   = param.metric_fields[0];
    const auto  spanning_space = metric_param.gradation.spanning_space;
    const auto  gradation      = metric_param.gradation.gradation;
    const auto  max_iterations = metric_param.gradation.max_iterations;
    const auto  tolerance      = metric_param.gradation.tolerance;

    std::cout << "Applying metric gradation with :" << std::endl;
    std::cout << "deterministic  = " << deterministic_gradation << std::endl;
    std::cout << "gradation      = " << gradation << std::endl;
    std::cout << "max_iterations = " << max_iterations << std::endl;
    std::cout << "tolerance      = " << tolerance << std::endl;

    unsigned int iter = 0, n_corrected = 0;
    bool         correction = true;

    // Apply gradation
    while (correction && iter < max_iterations)
    {
      n_corrected = 0;
      correction  = false;
      iter++;
      for (const auto &[p, q] : edges_for_deterministic_gradation)
      {
        Assert(all_metrics.count(p) > 0, ExcMessage("point not in map"));
        Assert(all_metrics.count(q) > 0, ExcMessage("point not in map"));

        MetricTensor<dim> *Mp = all_metrics.at(p);
        MetricTensor<dim> *Mq = all_metrics.at(q);

        if (MetricTensorTools::gradation_on_edge(
              p, q, spanning_space, gradation, tolerance, *Mp, *Mq))
        {
          n_corrected++;
          correction = true;
        };
      }
      std::cout << "Metric gradation: Sweep " << iter
                << " - Number of modified edges: " << n_corrected << std::endl;
    }
  }

  // Send back the graded metrics
  local_metrics =
    Utilities::MPI::scatter(mpi_communicator, gathered_metrics, 0);
  for (unsigned int i = 0; i < n_vertices; ++i)
    if (owned_vertices[i])
      metrics[i] = local_metrics[indices[i]].second;
}

template <int dim>
void MetricField<dim>::apply_gradation_non_deterministic()
{
  const auto &metric_param   = param.metric_fields[0];
  const auto  spanning_space = metric_param.gradation.spanning_space;
  const auto  gradation      = metric_param.gradation.gradation;
  const auto  max_iterations = metric_param.gradation.max_iterations;
  const auto  tolerance      = metric_param.gradation.tolerance;

  if (mpi_rank == 0)
  {
    std::cout << "Applying metric gradation with :" << std::endl;
    std::cout << "deterministic  = " << deterministic_gradation << std::endl;
    std::cout << "gradation      = " << gradation << std::endl;
    std::cout << "max_iterations = " << max_iterations << std::endl;
    std::cout << "tolerance      = " << tolerance << std::endl;
  }

  const std::vector<Point<dim>> &vertices = triangulation.get_vertices();

  unsigned int iter = 0;

  for (unsigned int i_ghost_updates = 0; i_ghost_updates < 5; ++i_ghost_updates)
  {
    unsigned int n_corrected = 0, local_n_corrected_with_ghosts = 0;
    bool         correction = true;

    // Apply gradation to the metrics on this partition
    while (correction && iter < max_iterations)
    {
      n_corrected = 0;
      correction  = false;
      iter++;
      for (const auto &[index_p, index_q] :
           edges_for_nondeterministic_gradation)
      {
        AssertIndexRange(index_p, metrics.size());
        AssertIndexRange(index_p, vertices.size());
        AssertIndexRange(index_q, metrics.size());
        AssertIndexRange(index_q, vertices.size());

        const Point<dim>  &p  = vertices[index_p];
        const Point<dim>  &q  = vertices[index_q];
        MetricTensor<dim> &Mp = metrics[index_p];
        MetricTensor<dim> &Mq = metrics[index_q];

        if (MetricTensorTools::gradation_on_edge(
              p, q, spanning_space, gradation, tolerance, Mp, Mq))
        {
          n_corrected++;
          correction = true;
        };
      }
      local_n_corrected_with_ghosts += n_corrected;
      std::cout << "Rank " << mpi_rank << " - Metric gradation: Sweep " << iter
                << " - Number of modified edges: " << n_corrected << std::endl;
    }

    const auto n_corrected_with_ghosts =
      Utilities::MPI::max(local_n_corrected_with_ghosts, mpi_communicator);
    if (n_corrected_with_ghosts == 0)
      break;

    // Update ghosts
    if (mpi_rank == 0)
      std::cout << "Updating ghosts" << std::endl;
    metrics_to_tensor_solution();
    tensor_solution_to_metrics();
  }
}

template <int dim>
void MetricField<dim>::intersect_with(const MetricField<dim> &other)
{
  AssertDimension(n_vertices, other.n_vertices);
  AssertDimension(metrics.size(), other.metrics.size());
  for (unsigned int i = 0; i < n_vertices; ++i)
  {
    MetricTensor<dim> &metric = metrics[i];
    metric                    = metric.intersection(other.metrics[i]);
  }
}

template <int dim>
void MetricField<dim>::compute_metrics()
{
  // Compute raw metrics
  compute_metrics_P1();

  // // Bound eigenvalues
  // for (auto &metric : metrics)
  //   metric.bound_eigenvalues(param.metric_fields[0].min_eigenvalue,
  // param.metric_fields[0].max_eigenvalue);
}

template <int dim>
double MetricField<dim>::compute_integral_determinant() const
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

      const Tensor<2, dim> &tensor = metrics[closest_vertex_index];
      const double          det    = determinant(tensor);
      integral += det * JxW[q];
    }
  }

  return integral;
}

#if defined(FEZ_WITH_MMG)
template <int dim>
void MetricField<dim>::set_mmg_solution(MMG5_pMesh pointer_to_mesh,
                                        MMG5_pSol  pointer_to_sol) const
{
  // Get the total number of mesh vertices.
  const auto n_owned_vertices =
    std::count(owned_vertices.begin(), owned_vertices.end(), true);
  const auto n_total_vertices =
    Utilities::MPI::sum(n_owned_vertices, mpi_communicator);

  // Gather the metrics to the root process
  const auto all_metrics = gather_metrics();

  // Do the mesh adaptation work from the root process
  // Maybe look into using ParMMG, but it seems to be no longer in development
  if (mpi_rank == 0)
  {
    int ier;

    if constexpr (dim == 2)
    {
      /** a) give info for the sol structure: sol applied on vertex entities,
          number of vertices=4, the sol is scalar*/
      ier = MMG2D_Set_solSize(pointer_to_mesh,
                              pointer_to_sol,
                              MMG5_Vertex,
                              n_total_vertices,
                              MMG5_Tensor);
      AssertThrow(ier == 1, ExcMessage("Error in MMG2D_Set_solSize"));

      /** b) give solutions values and positions */
      for (MMG5_int i = 1; i <= n_total_vertices; ++i)
      {
        // Default indexing for mesh vertices seems to match with MMG's ordering
        const auto &m = all_metrics[i - 1].second;
        ier = MMG2D_Set_tensorSol(pointer_to_sol, m[0][0], m[0][1], m[1][1], i);
        AssertThrow(ier == 1, ExcMessage("Error in MMG2D_Set_tensorSol"));
      }
    }
    else
    {
      DEAL_II_NOT_IMPLEMENTED();
    }

    /** 4) (not mandatory): check if the number of given entities match with
     * mesh size */
    ier = MMG2D_Chk_meshData(pointer_to_mesh, pointer_to_sol);
    AssertThrow(ier == 1, ExcMessage("Error in MMG2D_Chk_meshData"));

    std::cout << "Successfully wrote sol structure" << std::endl;
  }
}
#endif

template <int dim>
std::vector<std::pair<Point<dim>, MetricTensor<dim>>>
MetricField<dim>::gather_metrics() const
{
  using MessageType = std::pair<Point<dim>, MetricTensor<dim>>;

  std::vector<MessageType> global_metrics;

  const std::vector<Point<dim>> &vertices = triangulation.get_vertices();

  // Gather the mesh vertices and metrics
  std::vector<MessageType> local_metrics;
  for (unsigned int i = 0; i < n_vertices; ++i)
  {
    if (owned_vertices[i])
      local_metrics.push_back({vertices[i], metrics[i]});
  }

  std::vector<std::vector<MessageType>> gathered_metrics =
    Utilities::MPI::gather(mpi_communicator, local_metrics, 0);
  for (const auto &vec : gathered_metrics)
    global_metrics.insert(global_metrics.end(), vec.begin(), vec.end());

  return global_metrics;
}

template <int dim>
void MetricField<dim>::write_metrics(std::ostream &out) const
{
  auto all_metrics = gather_metrics();

  if (mpi_rank == 0)
  {
    // Sort based on lexicographic order of the mesh vertices
    std::sort(
      all_metrics.begin(),
      all_metrics.end(),
      PointMetricComparator<dim, std::pair<Point<dim>, MetricTensor<dim>>>());

    out << std::showpos;
    for (const auto &[pt, metric] : all_metrics)
      out << "Mesh vertex : " << pt << " - Metric : " << metric
          << " - det : " << determinant(metric) << std::endl;
  }
}

template <int dim>
void MetricField<dim>::write_pvtu(const std::string &filename,
                                  bool               write_inverse_metrics)
{
  (void)write_inverse_metrics;

  metrics_to_tensor_solution();

  MetricPostprocessor<dim> postprocessor;

  // FIXME: add this to the postprocessing handler
  DataOut<dim> data_out;
  data_out.attach_dof_handler(dof_handler);
  data_out.add_data_vector(metrics_fe, postprocessor);
  data_out.build_patches(1);
  data_out.write_vtu_with_pvtu_record(
    "./", filename, 0, dof_handler.get_mpi_communicator(), 2);
}

template class MetricField<2>;
template class MetricField<3>;
