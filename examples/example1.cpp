
#include <GenericSolver.h>
#include <MetricTensor.h>
#include <MetricField.h>
#include <SolutionRecovery.h>

#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/utilities.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/tria.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>

#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>

#include <deal.II/numerics/vector_tools.h>

#include <fstream>
#include <iostream>

template <int dim>
class MyFun : public Function<dim>
{
public:
  virtual double value(const Point<dim>  &p,
                       const unsigned int component = 0) const override;
};
 
template <int dim>
double MyFun<dim>::value(const Point<dim> &p,
                                 const unsigned int /*component*/) const
{
  const double x = p[0];
  const double y = p[1];
  // const double z = p[2];

  const double d = 0.1;

  return tanh((2. * x - sin(5. * y)) / d);
}


template <int dim>
class XFunction : public Function<dim>
{
public:
  XFunction() : Function<dim>(1) {}

  virtual double value(const Point<dim> &p,
                       const unsigned int component = 0) const override
  {
    return 1.23 * p[0] * p[0]; // x-coordinate
  }
};

void write_dof_locations(const DoFHandler<2> &dof_handler,
                         const std::string   &filename)
{
  MappingFE<2> mapping(FE_SimplexP<2>(1));

  const std::map<types::global_dof_index, Point<2>> dof_location_map =
    DoFTools::map_dofs_to_support_points(mapping, dof_handler);
 
  std::ofstream dof_location_file(filename);
  DoFTools::write_gnuplot_dof_support_point_info(dof_location_file,
                                                 dof_location_map);
}

int main()
{
  try
    {
      const unsigned int dim = 2;

      Triangulation<dim> mesh = makeMesh<dim>("../data/meshes/start.msh");

      // // Some tests with metrics
      MetricTensor<dim> m1, m2;
      m1[0][0] = 1.;
      m1[0][1] = 2.;
      m1[1][0] = 2.;
      m1[1][1] = 3.;

      m2[0][0] = 2.;
      m2[0][1] = 0.;
      m2[1][0] = 0.;
      m2[1][1] = 3.;


      Vector<double> foo({1., 3., 2.});
      MetricTensor<2> m = absoluteValue<2>(foo);

      std::cout << "absoluteValue" << std::endl;
      std::cout << m << std::endl;

      std::cout << "bounding" << std::endl;
      m.boundEigenvalues(1., 2.);
      std::cout << m << std::endl;

      // // std::cout << m1.log() << std::endl;
      // std::cout << m1.intersection(m2) << std::endl;

      std::cout << "dotprod" << std::endl;
      Point<2> p(1., 1.);

      const double res = p * m1 * p;

      std::cout << res << std::endl;

      std::cout << "spanned metric" << std::endl;
      const double gradation = 2.;
      Point<2> pq(1., 1.);

      std::cout << m.spanMetric(gradation, pq) << std::endl;

      MetricField<dim> metricField(mesh);
      // metricField.computeMetrics();
      for(auto &metric : metricField._metrics)
      {
        metric = unit_symmetric_tensor<2>();
        metric[0][0] = 10.;
      }

      MetricField<dim> metricField2(mesh);
      for(auto &metric : metricField2._metrics)
      {
        metric = unit_symmetric_tensor<2>();
        metric[1][1] = 10.;
      }

      // metricField._metrics[42][0][0] = 10.;
      // metricField._metrics[120][1][1] = 10.;

      // std::cout << "Final metric field" << std::endl;
      // for(auto &metric : metricField._metrics)
      // {
      //   std::cout << metric << std::endl;
      // }

      metricField.writeToVTU("metrics.vtu");
      metricField2.writeToVTU("metrics2.vtu");

      // const double gradation = 2.;
      const unsigned int maxIteration = 20.;
      const double tolerance = 0.1;

      // metricField.metricGradation(gradation, maxIteration, tolerance);
      metricField.intersectWith(metricField2);

      metricField.writeToVTU("metricsIntersection.vtu");

      DoFHandler<2> dof_handler(mesh);
      FE_SimplexP<dim> fe(2);
      dof_handler.distribute_dofs(fe);

      write_dof_locations(dof_handler, "dof_location.gnuplot");

      SolutionRecoveryNamespace::Patches<2> patches(mesh, dof_handler, fe.degree);
      // patches.write_patches_to_pos(50);

      Vector<double> solution;
      solution.reinit(dof_handler.n_dofs());
      XFunction<dim> x_function;
      MyFun<dim> myFun;
      VectorTools::interpolate(dof_handler, myFun, solution);

      // Output solution
      DataOut<2> data_out;
      data_out.attach_dof_handler(dof_handler);
      data_out.add_data_vector(solution, "solution");
      data_out.build_patches();
      std::ofstream     output("solution.vtk");
      data_out.write_vtk(output);

      SolutionRecoveryNamespace::SolutionRecovery rec(patches, solution, fe);

      rec.write_derivatives_to_vtu(1);
      rec.write_derivatives_to_vtu(2);
      rec.write_derivatives_to_vtu(3);

      // std::cout << metricField.compute_determinant_integral() << std::endl;

      // // Assigning a function to a grid
      // FE_SimplexP<dim>       fe(1); // Linear elements
      // DoFHandler<dim> dof_handler(mesh);

      // dof_handler.distribute_dofs(fe);

      // // 4. Interpolate function onto the DoFs
      // Vector<double> solution(dof_handler.n_dofs());
      // MyFun<dim>      my_function;
      // VectorTools::interpolate(dof_handler, my_function, solution);

      // // 5. Output to VTU for Paraview
      // {
      //   DataOut<dim> data_out;
      //   data_out.attach_dof_handler(dof_handler);
      //   data_out.add_data_vector(solution, "MyFunction");

      //   data_out.build_patches();
      //   std::ofstream output("solution.vtu");
      //   data_out.write_vtu(output);
      // }

    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl
                << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

	return 0;
}