
#include <deal.II/base/parameter_handler.h>
#include <metric_field_parameters.h>
#include <metric_tensor.h>
#include <parameters.h>

namespace Parameters
{
  template <int dim>
  MetricField<dim>::MetricField()
    : verbosity(Verbosity::verbose)
  {
    const unsigned n_components = MetricTensor<dim>::n_independent_components;
    analytical_metric.callback =
      std::make_shared<Functions::ParsedFunction<dim>>(n_components);
    analytical_field =
      std::make_shared<ManufacturedSolutions::ParsedFunctionSDBase<dim>>(1);
  }

  template <int dim>
  void MetricField<dim>::set_time(const double newtime)
  {
    analytical_metric.callback->set_time(newtime);
    analytical_field->set_time(newtime);
  }

  template <int dim>
  void MetricField<dim>::declare_parameters(ParameterHandler  &prm,
                                            const unsigned int index) const
  {
    prm.enter_subsection("Metric field " + std::to_string(index));
    {
      DECLARE_VERBOSITY_PARAM(prm, "verbose")
      prm.declare_entry(
        "min mesh size",
        "1e-8",
        Patterns::Double(1e-15),
        "Minimum allowed mesh size along a principal direction");
      prm.declare_entry(
        "max mesh size",
        "1e+6",
        Patterns::Double(1e-15),
        "Maximum allowed mesh size along a principal direction");
      prm.enter_subsection("Analytical metric");
      {
        prm.declare_entry("enable", "false", Patterns::Bool(), "");
        analytical_metric.callback->declare_parameters(prm);
      }
      prm.leave_subsection();
      prm.enter_subsection("Analytical scalar field");
      {
        // prm.declare_entry("enable", "false", Patterns::Bool(), "");
        analytical_field->declare_parameters(prm);
      }
      prm.leave_subsection();
      prm.enter_subsection("Gradation");
      {
        prm.declare_entry("enable", "false", Patterns::Bool(), "");
        prm.declare_entry("deterministic", "false", Patterns::Bool(), "");
        prm.declare_entry("gradation", "1.5", Patterns::Double(1.01), "");
        prm.declare_entry("max iteration", "100", Patterns::Integer(1), "");
        prm.declare_entry("tolerance", "1e-2", Patterns::Double(0), "");
        prm.declare_entry(
          "spanning space",
          "metric",
          Patterns::Selection("euclidean|metric|exp_metric"),
          "Space in which a single metric spans a full metric field");
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template <int dim>
  void MetricField<dim>::read_parameters(ParameterHandler  &prm,
                                         const unsigned int index)
  {
    prm.enter_subsection("Metric field " + std::to_string(index));
    {
      READ_VERBOSITY_PARAM(prm, verbosity)
      min_meshsize   = prm.get_double("min mesh size");
      max_meshsize   = prm.get_double("max mesh size");
      min_eigenvalue = 1. / (max_meshsize * max_meshsize);
      max_eigenvalue = 1. / (min_meshsize * min_meshsize);
      AssertThrow(min_eigenvalue > 1e-13,
                  ExcMessage(
                    "The specified maximum mesh size yields metric eigenvalues "
                    "very close to the machine epsilon, which might lead to "
                    "unstable metric computations."));
      prm.enter_subsection("Analytical metric");
      {
        analytical_metric.enable = prm.get_bool("enable");
        analytical_metric.callback->parse_parameters(prm);
      }
      prm.leave_subsection();
      prm.enter_subsection("Analytical scalar field");
      {
        // prm.declare_entry("enable", "false", Patterns::Bool(), "");
        analytical_field->parse_parameters(prm);
      }
      prm.leave_subsection();
      prm.enter_subsection("Gradation");
      {
        gradation.enable               = prm.get_bool("enable");
        gradation.deterministic        = prm.get_bool("deterministic");
        gradation.gradation            = prm.get_double("gradation");
        gradation.max_iterations       = prm.get_integer("max iteration");
        gradation.tolerance            = prm.get_double("tolerance");
        const std::string parsed_space = prm.get("spanning space");
        if (parsed_space == "euclidean")
          gradation.spanning_space =
            MetricTensor<dim>::SpanningSpace::euclidean;
        else if (parsed_space == "metric")
          gradation.spanning_space = MetricTensor<dim>::SpanningSpace::metric;
        else if (parsed_space == "exp_metric")
          gradation.spanning_space =
            MetricTensor<dim>::SpanningSpace::exp_metric;
        else
          throw std::runtime_error("Unknown gradation spanning space : " +
                                   parsed_space);
      }
      prm.leave_subsection();
    }
    prm.leave_subsection();
  }

  template struct MetricField<2>;
  template struct MetricField<3>;
} // namespace Parameters
