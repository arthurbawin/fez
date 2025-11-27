#ifndef MANUFACTURED_SOLUTION_H
#define MANUFACTURED_SOLUTION_H

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>
#include <parsed_function_symengine.h>
#include <preset_mms.h>

namespace ManufacturedSolutions
{
  using namespace dealii;

  enum class PresetMeshDisplacement
  {
    none,
    rigid_motion_kernel
  };

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
      , exact_mesh_displacement(std::make_shared<VectorSDParsedFunction<dim>>())
      , exact_vector_lagrange_mult(std::make_shared<VectorSDParsedFunction<dim>>())
      , mesh_displacement_time_function(std::make_shared<ScalarSDParsedFunction<dim>>())
    {}

    void set_time(const double new_time)
    {
      exact_velocity->set_time(new_time);
      exact_pressure->set_time(new_time);
      exact_mesh_displacement->set_time(new_time);
      exact_vector_lagrange_mult->set_time(new_time);
      mesh_displacement_time_function->set_time(new_time);
    }

  public:
    std::shared_ptr<VectorSDParsedFunction<dim>> exact_velocity;
    std::shared_ptr<ScalarSDParsedFunction<dim>> exact_pressure;
    std::shared_ptr<VectorSDParsedFunction<dim>> exact_mesh_displacement;
    std::shared_ptr<VectorSDParsedFunction<dim>> exact_vector_lagrange_mult;

    std::shared_ptr<MMSFunction<dim>> exact_preset_mesh_displacement;
    PresetMeshDisplacement preset_mesh_space_function;
    std::shared_ptr<ScalarSDParsedFunction<dim>> mesh_displacement_time_function;

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);
  };
} // namespace ManufacturedSolution

#endif