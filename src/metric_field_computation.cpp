
#include <metric_field.h>
#include <metric_tensor_tools.h>

template <int dim>
void MetricField<dim>::compute_metrics_P1()
{
  std::vector<SymmetricTensor<2, dim>> hessians(n_vertices);
  const std::vector<Point<dim>>       &vertices = triangulation.get_vertices();

  // Get the hessians at all mesh vertices
  param.metric_fields[0].analytical_field->hessian_list(vertices, hessians);

  // TODO: this is "embarrassingly parallel" and can be multithreaded
  for (unsigned int v = 0; v < n_vertices; ++v)
  {
    if (owned_vertices[v])
    {
      // Get hessian at p, from exact solution or recovered derivatives
      // if (useExactDerivatives)
      // {
      std::cout << "Hessian at " << vertices[v] << " = " << hessians[v]
                << std::endl;
      metrics[v] = MetricTensorTools::absolute_value(
        hessians[v],
        param.metric_fields[0].min_eigenvalue,
        param.metric_fields[0].max_eigenvalue);
      // }
      // else
      // {
      //   // Use recovered derivatives
      // }

      // // TODO: Target Lp norm for now, add W1,p
    }
  }
}

template void MetricField<2>::compute_metrics_P1();
template void MetricField<3>::compute_metrics_P1();
