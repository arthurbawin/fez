#include "chns_adaptive_mobility_jacobian.h"

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  const std::string                mobility = "adaptative_mobility";

  // Check the additive gradient term before entering tracer SUPG, so an
  // unsupported-stabilization assertion cannot hide a weak-form derivative
  // error. This also exercises the evaluator's velocity derivative for m > 0.
  run_case<false>("abels", "none", false, false, mobility, 0.15);
  AssertThrow(failed_directions == 0,
              ExcMessage("Positive-m adaptive mobility weak Jacobian failed"));

  for (const double gradient_coefficient : {0., 0.15})
  {
    // Momentum SUPG also checks the phase-gradient dependence of the
    // diffusive inertia, independently of tracer stabilization.
    for (const std::string correction : {"none", "profile", "profile_flux"})
    {
      run_case<false>(
        "abels", correction, true, false, mobility, gradient_coefficient);
      run_case<true>(
        "abels", correction, true, false, mobility, gradient_coefficient);
      run_case<true, true>(
        "abels", correction, true, false, mobility, gradient_coefficient);
    }

    // Quadratic tracer and curved mesh directions in the shared harness
    // exercise tracer and reconstructed-profile Hessians, including their
    // non-affine ALE variations in the complete phase-flux divergence.
    for (const bool momentum_supg : {false, true})
      for (const std::string correction : {"none", "profile", "profile_flux"})
      {
        run_case<false>("abels",
                        correction,
                        momentum_supg,
                        true,
                        mobility,
                        gradient_coefficient);
        run_case<true>("abels",
                       correction,
                       momentum_supg,
                       true,
                       mobility,
                       gradient_coefficient);
        run_case<true, true>("abels",
                             correction,
                             momentum_supg,
                             true,
                             mobility,
                             gradient_coefficient);
      }
    run_case<false>(
      "ding_horriche", "none", true, true, mobility, gradient_coefficient);
  }

  AssertThrow(failed_directions == 0,
              ExcMessage(std::to_string(failed_directions) +
                         " adaptive mobility component directions failed"));
  initlog();
  deallog << "Adaptive mobility component Jacobians: m=0 and m>0, fixed, ALE, "
             "enlarged, profile flux, Ding and frozen-tau stabilization OK"
          << std::endl;
}
