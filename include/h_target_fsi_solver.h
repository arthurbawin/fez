#ifndef H_TARGET_FSI_SOLVER_H
#define H_TARGET_FSI_SOLVER_H

#include <deal.II/fe/fe_values_extractors.h>
#include <monolithic_fsi_solver.h>
#include <scratch_data.h>

using namespace dealii;

template <int dim>
class FSISolverHTarget : public FSISolver<dim>
{
  using ScratchData = ScratchDataFSIHTarget<dim>;

public:
  FSISolverHTarget(const ParameterReader<dim> &param);

  virtual void create_scratch_data() override;

  virtual void set_solver_specific_initial_conditions() override;

  virtual void set_solver_specific_exact_solution() override;

  virtual void add_solver_specific_postprocessing_data() override;

protected:
  virtual std::vector<std::pair<std::string, unsigned int>>
  get_additional_variables_description() const override
  {
    std::vector<std::pair<std::string, unsigned int>> description;
    description.push_back({"eta_h_target", 1});
    description.push_back({"lambda", dim});
    return description;
  }

  FEValuesExtractors::Scalar h_target_extractor;
  ComponentMask              h_target_mask;

  class ConstantHTarget : public Function<dim>
  {
  public:
    ConstantHTarget(const ComponentOrdering &ordering, const double value)
      : Function<dim>(ordering.n_components)
      , ordering(ordering)
      , value(value)
    {}

    virtual void vector_value(const Point<dim> &,
                              Vector<double>   &values) const override
    {
      values = 0.;
      values[ordering.h_lower] = value;
    }

  private:
    const ComponentOrdering &ordering;
    const double             value;
  };
};

#endif
