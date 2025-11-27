
#include <manufactured_solution.h>
#include <preset_mms.h>
#include <utilities.h>

namespace ManufacturedSolutions
{
  template <int dim>
  void ManufacturedSolution<dim>::declare_parameters(ParameterHandler &prm)
  {
    const std::string default_point = (dim == 2) ? "0, 0" : "0, 0, 0";

    prm.enter_subsection("Manufactured solution");
    {
      prm.enter_subsection("exact velocity");
      exact_velocity->declare_parameters(prm, dim);
      prm.leave_subsection();
      prm.enter_subsection("exact pressure");
      exact_pressure->declare_parameters(prm, 1);
      prm.leave_subsection();
      prm.enter_subsection("exact mesh displacement");
      exact_mesh_displacement->declare_parameters(prm, dim);
      prm.enter_subsection("space component");
      {
        prm.declare_entry("preset",
                          "none",
                          Patterns::Selection("none|rigid_motion_kernel"),
                          "");
        prm.enter_subsection("rigid_motion_kernel");
        {
          prm.declare_entry("translation",
                            default_point,
                            Patterns::List(Patterns::Double(), dim, dim, ","),
                            "Translation vector");
          prm.declare_entry("r0", "0.1", Patterns::Double(), "R0");
          prm.declare_entry("r1", "0.2", Patterns::Double(), "R1");
          prm.declare_entry("center",
                            default_point,
                            Patterns::List(Patterns::Double(), dim, dim, ","),
                            "Initial center of the kernel");
        }
        prm.leave_subsection();
      }
      prm.leave_subsection();
      prm.enter_subsection("time component");
      {
        mesh_displacement_time_function->declare_parameters(prm, 1);
      }
      prm.leave_subsection();
      prm.leave_subsection();
      prm.enter_subsection("exact vector lagrange multiplier");
      exact_vector_lagrange_mult->declare_parameters(prm, dim);
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <int dim>
  void ManufacturedSolution<dim>::read_parameters(ParameterHandler &prm)
  {
    prm.enter_subsection("Manufactured solution");
    {
      prm.enter_subsection("exact velocity");
      exact_velocity->parse_parameters(prm);
      prm.leave_subsection();
      prm.enter_subsection("exact pressure");
      exact_pressure->parse_parameters(prm);
      prm.leave_subsection();
      prm.enter_subsection("exact mesh displacement");
      exact_mesh_displacement->parse_parameters(prm);
      // prm.enter_subsection("space component");
      // {
      //   const std::string parsed_preset = prm.get("preset");
      //   if (parsed_preset == "none")
      //     preset_mesh_space_function = PresetMeshDisplacement::none;
      //   else if (parsed_preset == "rigid_motion_kernel")
      //   {
      //     preset_mesh_space_function = PresetMeshDisplacement::rigid_motion_kernel;
      //   }
      //   prm.enter_subsection("rigid_motion_kernel");
      //   {
      //     const Tensor<1, dim> translation =
      //       parse_rank_1_tensor<dim>(prm.get("translation"));
      //     const double     r0     = prm.get_double("r0");
      //     const double     r1     = prm.get_double("r1");
      //     const Point<dim> center(parse_rank_1_tensor<dim>(prm.get("center")));
      //     if (preset_mesh_space_function ==
      //         PresetMeshDisplacement::rigid_motion_kernel)
      //       exact_preset_mesh_displacement = std::make_shared<RigidMeshPosition2<dim>>(
      //         *mesh_displacement_time_function,
      //         center,
      //         r0,
      //         r1,
      //         translation,
      //         1.);
      //   }
      //   prm.leave_subsection();
      // }
      // prm.leave_subsection();
      // prm.enter_subsection("time component");
      // {
      //   mesh_displacement_time_function->parse_parameters(prm);
      // }
      prm.leave_subsection();
      prm.enter_subsection("exact vector lagrange multiplier");
      exact_vector_lagrange_mult->parse_parameters(prm);
      prm.leave_subsection();
    }
    prm.leave_subsection();

  }

  // Explicit instantiation
  template class ManufacturedSolution<2>;
  template class ManufacturedSolution<3>;
} // namespace ManufacturedSolutions