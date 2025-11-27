#ifndef PARSED_FUNCTION_SYMENGINE_H
#define PARSED_FUNCTION_SYMENGINE_H

#include <deal.II/base/exception_macros.h>
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/differentiation/sd/symengine_number_types.h>
#include <manufactured_solution.h>

namespace ManufacturedSolutions
{
  using namespace dealii;

  /**
   * This class is an alternative to deal.II's ParsedFunction, where the
   * function is parsed from the parameter file, then its derivatives are
   * computed using SymEngine through deal.II. The symbolic derivatives are then
   * re-converted to a MuParser function to (hopefully?) limit overhead from
   * symbolic substitutions.  This class overrides deal.II's value, gradient and
   * hessian functions using the derivatives computed with SymEngine.
   *
   * It is also an MMSFunction, allowing to use it to easily construct source
   * terms for convergence studies with manufactured solutions. As an MMS
   * function, it is limited to scalar- or vector-valued functions, so it is
   * only possible to create such a function with n_components = 1 or dim.
   *
   * Because it is easily available, this function also provides second
   * time derivatives for each component (unlike MMSFunctions).
   */
  template <int dim>
  class ParsedFunctionSDBase : public MMSFunction<dim>
  {
  public:
    /**
     * Constructor
     */
    ParsedFunctionSDBase(const unsigned int n_components);

    static void declare_parameters(ParameterHandler  &prm,
                                   const unsigned int n_components = 1,
                                   const std::string &input_expr   = "");

    void parse_parameters(ParameterHandler &prm);

    /**
     * Return true if all components of this functions are functions of
     * time only (not of x, y or z).
     */
    inline bool is_function_of_time_only() const
    {
      bool res = true;
      for (unsigned int i = 0; i < n_components; ++i)
        res &= function_of_time_only[i];
      return res;
    }

    /**
     * Identical to ParsedFunction's value.
     * Simply calls value from the underlying FunctionParser.
     */
    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      return function_object.value(p, component);
    }

    /**
     * Identical to ParsedFunction's vector_value.
     * Simply calls vector_value from the underlying FunctionParser.
     */
    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      function_object.vector_value(p, values);
    }

    virtual void set_time(const double newtime) override
    {
      this->function_object.set_time(newtime);
      dfdt.set_time(newtime);
      d2fdt2.set_time(newtime);
      for (unsigned int i_comp = 0; i_comp < n_components; ++i_comp)
      {
        grad_function_object[i_comp]->set_time(newtime);
        hess_function_object[i_comp]->set_time(newtime);
      }
    }

    /**
     * Overload of the deal.II and MMSFunction functions
     */
    virtual double
    time_derivative(const Point<dim>  &p,
                    const unsigned int component = 0) const override
    {
      return dfdt.value(p, component);
    }

    double time_second_derivative(const Point<dim>  &p,
                                  const unsigned int component = 0) const
    {
      return d2fdt2.value(p, component);
    }

    virtual Tensor<1, dim>
    gradient(const Point<dim>  &p,
             const unsigned int component = 0) const override
    {
      Tensor<1, dim> grad;
      for (unsigned int d = 0; d < dim; ++d)
        grad[d] = grad_function_object[component]->value(p, d);
      return grad;
    }

    virtual SymmetricTensor<2, dim>
    hessian(const Point<dim>  &p,
            const unsigned int component = 0) const override
    {
      SymmetricTensor<2, dim> hess;
      for (unsigned int di = 0; di < dim; ++di)
        for (unsigned int dj = di; dj < dim; ++dj)
          hess[di][dj] =
            hess_function_object[component]->value(p, di * dim + dj);
      return hess;
    }

    std::string get_function_expression(const unsigned int component = 0) const
    {
      return function_object.get_expressions()[component];
    }

  private:
    /**
     * Create the callbacks for the spatial and time derivatives.
     */
    virtual void
    create_symbolic_derivatives(const std::string                   variables,
                                const std::map<std::string, double> constants,
                                const bool time_dependent);

  protected:
    const unsigned int  n_components;
    FunctionParser<dim> function_object;
    FunctionParser<dim> dfdt;
    FunctionParser<dim> d2fdt2;
    // FunctionParser<dim> are not copyable : using smart pointers instead
    std::vector<std::shared_ptr<FunctionParser<dim>>> grad_function_object;
    std::vector<std::shared_ptr<FunctionParser<dim>>> hess_function_object;
    std::vector<bool>                                 function_of_time_only;
  };

} // namespace ManufacturedSolutions

#endif