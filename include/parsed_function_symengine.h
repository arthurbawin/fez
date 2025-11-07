#ifndef PARSED_FUNCTION_SYMENGINE_H
#define PARSED_FUNCTION_SYMENGINE_H

#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/differentiation/sd/symengine_number_types.h>

namespace ManufacturedSolution
{
  using namespace dealii;
  using namespace Differentiation::SD;

  /**
   * This class is an alternative to deal.II's ParsedFunction, where
   * the function is parsed from the parameter file, then its derivatives
   * are computed using SymEngine through deal.II.
   *
   * It has the same interface, with additional functions to compute
   * derivatives.
   */
  template <int dim>
  class ParsedFunctionSymEngineBase : public Function<dim>
  {
  public:
    /**
     * Constructor
     */
    ParsedFunctionSymEngineBase(const unsigned int n_components = 1);

    static void declare_parameters(ParameterHandler  &prm,
                                   const unsigned int n_components = 1,
                                   const std::string &input_expr   = "");

    void parse_parameters(ParameterHandler &prm);

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      function_object.vector_value(p, values);
    }

    virtual void set_time(const double newtime) override = 0;

  private:
    /**
     *
     */
    virtual void
    create_symbolic_derivatives(const std::string                   variables,
                                const std::map<std::string, double> constants,
                                const bool time_dependent) = 0;

  protected:
    const unsigned int  n_components;
    FunctionParser<dim> function_object;
  };

  /**
   * A scalar parsed function with symbolic differentiation.
   * For instance, the gradient is a Tensor<1, dim, Expression>.
   */
  template <int dim>
  class ScalarSDParsedFunction : public ParsedFunctionSymEngineBase<dim>
  {
  public:
    /**
     * Constructor
     */
    ScalarSDParsedFunction()
      : ParsedFunctionSymEngineBase<dim>(1)
      , dfdt(1)
      , grad_function_object(dim)
      , hess_function_object(dim * dim)
    {}

    virtual void set_time(const double newtime) override
    {
      this->function_object.set_time(newtime);
      dfdt.set_time(newtime);
      grad_function_object.set_time(newtime);
      hess_function_object.set_time(newtime);
    }

    double time_derivative(const Point<dim> &p) const { return dfdt.value(p); }

    void gradient(const Point<dim> &p, Tensor<1, dim> &grad) const
    {
      for (unsigned int d = 0; d < dim; ++d)
        grad[d] = grad_function_object.value(p, d);
    }

    double laplacian(const Point<dim> &p) const
    {
      double res =
        hess_function_object.value(p, 0) + hess_function_object.value(p, 3);
      if constexpr (dim == 3)
        res += hess_function_object.value(p, 8);
      return res;
    }

  private:
    virtual void
    create_symbolic_derivatives(const std::string                   variables,
                                const std::map<std::string, double> constants,
                                const bool time_dependent) override;

  private:
    FunctionParser<dim> dfdt;
    FunctionParser<dim> grad_function_object;
    FunctionParser<dim> hess_function_object;
  };

  /**
   * A vector parsed function with symbolic differentiation.
   * For instance, the gradient is a Tensor<2, dim, Expression>.
   */
  template <int dim>
  class VectorSDParsedFunction : public ParsedFunctionSymEngineBase<dim>
  {
  public:
    /**
     * Constructor
     */
    VectorSDParsedFunction()
      : ParsedFunctionSymEngineBase<dim>(dim)
      , dfdt(dim)
      , grad_function_object(dim, std::make_shared<FunctionParser<dim>>(dim))
      , hess_function_object(dim,
                             std::make_shared<FunctionParser<dim>>(dim * dim))
    {}

    virtual void set_time(const double newtime) override
    {
      this->function_object.set_time(newtime);
      for (unsigned int d = 0; d < dim; ++d)
      {
        dfdt.set_time(newtime);
        grad_function_object[d]->set_time(newtime);
        hess_function_object[d]->set_time(newtime);
      }
    }

    void time_derivative(const Point<dim> &p, Tensor<1, dim> &res) const
    {
      for (unsigned int d = 0; d < dim; ++d)
        res[d] = dfdt.value(p);
    }

    /**
     * Convention : grad_ij = \partial v_i/\partial x_j
     */
    void gradient(const Point<dim> &p, Tensor<2, dim> &grad) const
    {
      for (unsigned int di = 0; di < dim; ++di)
        for (unsigned int dj = 0; dj < dim; ++dj)
          grad[di][dj] = grad_function_object[di]->value(p, dj);
    }

    void laplacian(const Point<dim> &p, Tensor<1, dim> &res) const
    {
      for (unsigned int d = 0; d < dim; ++d)
      {
        res[d] = hess_function_object[d]->value(p, 0) +
                 hess_function_object[d]->value(p, 3);
        if constexpr (dim == 3)
          res[d] += hess_function_object[d]->value(p, 8);
      }
    }

  private:
    virtual void
    create_symbolic_derivatives(const std::string                   variables,
                                const std::map<std::string, double> constants,
                                const bool time_dependent) override;

  private:
    FunctionParser<dim> dfdt;
    // FunctionParser<dim> are not copyable : using smart pointers instead
    std::vector<std::shared_ptr<FunctionParser<dim>>> grad_function_object;
    std::vector<std::shared_ptr<FunctionParser<dim>>> hess_function_object;
  };

} // namespace ManufacturedSolution

#endif