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
      , pseudosolid_source(
          std::make_shared<Functions::ParsedFunction<dim>>(dim))
      , linear_elasticity_source(
          std::make_shared<Functions::ParsedFunction<dim>>(dim))
      , cahnhilliard_source(std::make_shared<Functions::ParsedFunction<dim>>(2))
    {}

    void set_time(const double new_time)
    {
      fluid_source->set_time(new_time);
      pseudosolid_source->set_time(new_time);
      // Shouldn't be required as the LinearElasticitySolver is steady-state
      linear_elasticity_source->set_time(new_time);
      cahnhilliard_source->set_time(new_time);
    }

  public:
    /**
     * Source term for the Navier-Stokes momentum and mass equations.
     * Combined vector + scalar source term, thus dim + 1 components :
     * u-v-(w-)p.
     */
    std::shared_ptr<Functions::ParsedFunction<dim>> fluid_source;

    /**
     * Source term for the pseudosolid (linear elasticity) equation.
     * Vector-valued source term, thus dim components : x-y-(z).
     *
     * This source term is called from all solvers having the pseudosolid
     * equation as an additional, coupled equation to the model, as opposed to
     * linear_elasticity_source below, which is only called from the dedicated
     * LinearElasticitySolver.
     */
    std::shared_ptr<Functions::ParsedFunction<dim>> pseudosolid_source;

    /**
     * Source term for the linear elasticity equation, when solved alone
     * in the dedicated solver. This source term can be used, e.g., to adapt
     * the mesh to an initial source term, before continuing with another
     * solver, which may or may not have a user-defined pseudosolid_source term
     * for the pseudosolid equation.
     *
     * This source term is thus only called from the LinearElasticitySolver.
     */
    std::shared_ptr<Functions::ParsedFunction<dim>> linear_elasticity_source;

    /**
     * Source term for the pseudosolid (linear elasticity) equation.
     * Two scalar-valued source terms, thus 2 components : phi-mu.
     */
    std::shared_ptr<Functions::ParsedFunction<dim>> cahnhilliard_source;

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
        prm.enter_subsection("cahn hilliard");
        cahnhilliard_source->declare_parameters(prm, 2);
        prm.leave_subsection();
      }
      prm.leave_subsection();

      // LinearElasticity source term is in the Linear elasticity section
      prm.enter_subsection("Linear elasticity");
      {
        prm.enter_subsection("source term");
        linear_elasticity_source->declare_parameters(prm, dim);
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
        prm.enter_subsection("cahn hilliard");
        cahnhilliard_source->parse_parameters(prm);
        prm.leave_subsection();
      }
      prm.leave_subsection();

      prm.enter_subsection("Linear elasticity");
      {
        prm.enter_subsection("source term");
        linear_elasticity_source->parse_parameters(prm);
        prm.leave_subsection();
      }
      prm.leave_subsection();
    }
  };
} // namespace Parameters

#endif
