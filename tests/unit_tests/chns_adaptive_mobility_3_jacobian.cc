#include "chns_adaptive_mobility_jacobian.h"

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  for (const bool stabilized : {false, true})
    for (const std::string correction : {"none", "profile", "profile_flux"})
      run_case<false>("abels", correction, stabilized, false);
  run_case<false>("abels", "none", true, true);
  run_case<false>("ding_horriche", "none", true, true);
  run_case<true>("abels", "none", true, true);
  run_case<true>("abels", "profile_flux", true, false);
  run_case<true, true>("abels", "none", true, true);
  run_case<true, true>("abels", "profile_flux", true, false);
  for (const std::string correction : {"profile", "profile_flux"})
  {
    run_case<false>("abels", correction, true, true);
    run_case<true>("abels", correction, true, true);
    run_case<true, true>("abels", correction, true, true);
  }
  AssertThrow(failed_directions == 0,
              ExcMessage(std::to_string(failed_directions) +
                         " mobility 3 component directions failed"));
  initlog();
  deallog << "Adaptive mobility 3 component Jacobians: fixed, ALE, enlarged, "
             "profile flux, Ding and frozen-tau stabilization OK"
          << std::endl;
}
