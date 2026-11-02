#include <parsed_function_symengine.h>

#include "../tests.h"

void near(double actual, double expected, double tolerance)
{
  AssertThrow(std::abs(actual - expected) < tolerance,
              ExcMessage("Incorrect parsed value or derivative"));
}
void check_function()
{
  ManufacturedSolutions::ParsedFunctionSDBase<2> f(1);
  ParameterHandler                               prm;
  f.declare_parameters(prm, 1);
  prm.set("Function expression", "if(x>0,a*x*x+x*y,2*x*x+x*y)");
  prm.set("Function constants", "a=3");
  f.parse_parameters(prm);
  const Point<2> p(.25, .4);
  near(f.value(p), .2875, 1e-12);
  near(f.gradient(p)[0], 1.9, 1e-6);
  near(f.gradient(p)[1], .25, 1e-6);
  near(f.hessian(p)[0][0], 6., 1e-6);
  near(f.hessian(p)[0][1], 1., 1e-6);
  near(f.hessian(p)[1][1], 0., 1e-6);
  near(f.gradient(Point<2>(-.25, .4))[0], -.6, 1e-6);
  f.update_constants({{"a", 4.}});
  near(f.gradient(p)[0], 2.4, 1e-6);
  near(f.hessian(p)[0][0], 8., 1e-6);
  // Some nonsmooth derivatives remain symbolic instead of raising an error.
  prm.set("Function expression", "abs(x)+x*y");
  f.parse_parameters(prm);
  near(f.gradient(p)[0], 1.4, 1e-6);
  near(f.gradient(Point<2>(-.25, .4))[0], -.6, 1e-6);
  near(f.hessian(p)[0][0], 0., 1e-6);
  near(f.hessian(p)[0][1], 1., 1e-6);
  // Smooth input must return to the symbolic path, including time derivatives.
  prm.set("Function expression", "a*x*x+x*y+t*t*t");
  f.parse_parameters(prm);
  f.set_time(.5);
  near(f.gradient(p)[0], 1.9, 1e-12);
  near(f.hessian(p)[0][0], 6., 1e-12);
  near(f.time_derivative(p), .75, 1e-12);
  bool rejected = false;
  try
  {
    prm.set("Function expression", "if(x>0,x*t,2*x*t)");
    f.parse_parameters(prm);
  }
  catch (const std::exception &)
  {
    rejected = true;
  }
  AssertThrow(rejected,
              ExcMessage("Unsupported time derivative was silently accepted"));
  rejected = false;
  try
  {
    prm.set("Function expression", "x+)");
    f.parse_parameters(prm);
    // muParser validates syntax when the expression is first evaluated.
    (void)f.value(p);
  }
  catch (const std::exception &)
  {
    rejected = true;
  }
  AssertThrow(rejected, ExcMessage("Malformed input was silently accepted"));
  deallog << "Piecewise derivatives, constants and symbolic derivatives OK"
          << std::endl;
}
int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  deal_II_exceptions::disable_abort_on_exception();
  initlog();
  try
  {
    check_function();
  }
  catch (const std::exception &e)
  {
    std::cerr << e.what() << std::endl;
    return 1;
  }
}
