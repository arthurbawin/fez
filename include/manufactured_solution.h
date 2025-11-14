#ifndef MANUFACTURED_SOLUTION_H
#define MANUFACTURED_SOLUTION_H

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <parsed_function_symengine.h>

namespace ManufacturedSolution
{
  using namespace dealii;

  /**
   *
   */
  template <int dim>
  class ManufacturedSolution
  {
  public:
    ManufacturedSolution()
      : exact_velocity(std::make_shared<VectorSDParsedFunction<dim>>())
      , exact_pressure(std::make_shared<ScalarSDParsedFunction<dim>>())
    {}

    void set_time(const double new_time)
    {
      exact_velocity->set_time(new_time);
      exact_pressure->set_time(new_time);
    }

  public:
    std::shared_ptr<VectorSDParsedFunction<dim>> exact_velocity;
    std::shared_ptr<ScalarSDParsedFunction<dim>> exact_pressure;

    void declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Manufactured solution");
      {
        prm.enter_subsection("exact velocity");
        exact_velocity->declare_parameters(prm, dim);
        prm.leave_subsection();
        prm.enter_subsection("exact pressure");
        exact_pressure->declare_parameters(prm, 1);
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
    void read_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Manufactured solution");
      {
        prm.enter_subsection("exact velocity");
        exact_velocity->parse_parameters(prm);
        prm.leave_subsection();
        prm.enter_subsection("exact pressure");
        exact_pressure->parse_parameters(prm);
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  };
} // namespace ManufacturedSolution

#endif