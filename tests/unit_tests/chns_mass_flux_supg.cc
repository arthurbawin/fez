#include "chns_adaptive_mobility_jacobian.h"

int main(int argc, char **argv)
{
  Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);
  for (const std::string model : {"abels", "abels_nlm", "ding_horriche"})
    for (const std::string mobility : {"constant",
                                       "degenerate",
                                       "adaptative_mobility",
                                       "adaptative_mobility_2",
                                       "adaptative_mobility_3"})
      for (const std::string correction : {"none", "profile_flux"})
      {
        // Profile corrections are only supported by the linear Abels model.
        if (model != "abels" && correction != "none")
          continue;
        run_case<false>(model, correction, true, false, mobility, 0.15, true);
        run_case<true>(model, correction, true, false, mobility, 0.15, true);
        run_case<true, true>(
          model, correction, true, false, mobility, 0.15, true);
      }
  AssertThrow(failed_directions == 0,
              ExcMessage("Mass-flux SUPG Jacobian failed"));
  initlog();
  deallog << "Mass-flux SUPG: tau, residual and frozen-tau Jacobians OK"
          << std::endl;
}
