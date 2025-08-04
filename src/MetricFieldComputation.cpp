
#include <MetricField.h>

#include <deal.II/base/function.h>
#include <deal.II/lac/vector.h>

template <int dim>
class ExactTestHessian : public Function<dim>
{
public:
  // Set number of components based on dim
  ExactTestHessian() : Function<dim>((dim == 2) ? 3 : 6) {}

  // Main function call to evaluate the function at a point
  virtual void vector_value(const Point<dim> &p, Vector<double> &values) const override
  {
    // Resize vector if needed
    AssertDimension(values.size(), this->n_components);

    const double x = p[0];
    const double y = p[1];

    const double d = 1.;
    const double valx = (2.0 * x - sin(y * 5.0));

    const double uxx = 1.0 / (d * d) * tanh(valx / d) * (pow(tanh(valx / d), 2.0) - 1.0) * 8.0;
    const double uxy = -1.0 / (d * d) * cos(y * 5.0) * tanh(valx / d) *
                   (pow(tanh(valx / d), 2.0) - 1.0) * 2.0E+1;
    const double uyy =
      -(sin(y * 5.0) * (pow(tanh(valx / d), 2.0) - 1.0) * 2.5E+1) / d +
      1.0 / (d * d) * pow(cos(y * 5.0), 2.0) * tanh(valx / d) *
        (pow(tanh(valx / d), 2.0) - 1.0) * 5.0E+1;
    const double uzz = 1.;
    const double uxz = 0.;
    const double uyz = 0.;

    if constexpr (dim == 2)
    {
      // 2D case: 3 components for symmetric matrix
      // Same ordering as dealii SymmetricTensor
      values[0] = uxx;
      values[1] = uyy;
      values[2] = uxy;     
    }
    else if constexpr (dim == 3)
    {
      // 3D case: 6 components
      const double z = p[2];

      // TODO: Check ordering for SymmetricTensor
      values[0] = uxx; 
      values[1] = uyy;
      values[2] = uzz;
      values[3] = uxy; // Check ordering 
      values[4] = uxz; // Check ordering
      values[5] = uyz; // Check ordering
    }
    else
    {
      AssertThrow(false, ExcNotImplemented());
    }
  }

  // Optional: override all vector values at once (e.g., for quadrature points)
  virtual void
  vector_value_list(const std::vector<Point<dim>> &points,
                    std::vector<Vector<double>>   &value_list) const override
  {
    AssertDimension(points.size(), value_list.size());
    for (unsigned int i = 0; i < points.size(); ++i)
        this->vector_value(points[i], value_list[i]);
  }
};

template <int dim>
void MetricField<dim>::computeMetricsP1()
{
  bool useExactDerivatives = true;

  ExactTestHessian<dim> hessian;
  Vector<double> hessianAtP((dim == 2) ? 3 : 6);

  // Check that derivatives were given if using analytical derivatives
  // AssertThrow(, ExcMessage("Using analytical derivatives but none were provided"));

  const std::vector<Point<dim>> &vertices = triangulation.get_vertices();
  const unsigned int n_vertices = triangulation.n_vertices();

  // TODO: Check if we can safely use OpenMP in postprocessing routines
  for (unsigned int v = 0; v < n_vertices; ++v)
  {
    const Point<dim> &p = vertices[v];
    // MetricTensor<dim> &metric = _metrics[v];
    auto &metric = _metrics[v];

    // Compute derivatives at p
    if(useExactDerivatives)
    {
      hessian.vector_value(p, hessianAtP);
    }
    else
    {
      // Use recovered derivatives
    }

    // // TODO: Target Lp norm for now, add W1,p

    std::cout << "Hessian is" << std::endl;
    std::cout << hessianAtP << std::endl;

    metric = absoluteValue<dim>(hessianAtP);
    std::cout << "Metric  is" << std::endl;
    std::cout << metric << std::endl;
    std::cout << "Bounded Metric  is" << std::endl;
    metric.boundEigenvalues(1./(100.*100.), 1./(1e-10 * 1e-10));
    std::cout << metric << std::endl;
  }
}

#include <MetricField_inst.h>