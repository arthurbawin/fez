#ifndef SOURCE_TERMS_H
#define SOURCE_TERMS_H

#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>

using namespace dealii;

namespace Parameters
{
  /**
   * This class contains all the possible source terms.
   */
  template <int dim>
  class SourceTerms
  {
  public:
    SourceTerms()
      : fluid_source(std::make_shared<Functions::ParsedFunction<dim>>(dim + 1))
      , pseudosolid_source(std::make_shared<Functions::ParsedFunction<dim>>(dim))
    {}

    void set_time(const double new_time)
    {
      fluid_source->set_time(new_time);
      pseudosolid_source->set_time(new_time);
    }

  public:
    // Source term for the Navier-Stokes momentum and mass equations.
    // Layout is u-v-(w-)p.
    std::shared_ptr<Functions::ParsedFunction<dim>> fluid_source;
    std::shared_ptr<Functions::ParsedFunction<dim>> pseudosolid_source;

    void declare_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Source terms");
      {
        prm.enter_subsection("fluid");
        fluid_source->declare_parameters(prm, dim + 1);
        prm.leave_subsection();
        prm.enter_subsection("pseudosolid");
        pseudosolid_source->declare_parameters(prm, dim);
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }

    void read_parameters(ParameterHandler &prm)
    {
      prm.enter_subsection("Source terms");
      {
        prm.enter_subsection("fluid");
        fluid_source->parse_parameters(prm);
        prm.leave_subsection();
        prm.enter_subsection("pseudosolid");
        pseudosolid_source->parse_parameters(prm);
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  };
} // namespace Parameters

#endif