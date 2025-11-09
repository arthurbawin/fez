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
   *
   * FIXME: The class functions for value, gradients, etc, are a bit
   * different from deal.II's Function interface because here we expect purely
   * scalar or vector-valued functions with 1 or dim components, instead of an
   * arbitrary number of components.
   */
  template <int dim>
  class ParsedFunctionSDBase : public Function<dim>
  {
  public:
    /**
     * Constructor
     */
    ParsedFunctionSDBase(const unsigned int n_components = 1);

    static void declare_parameters(ParameterHandler  &prm,
                                   const unsigned int n_components = 1,
                                   const std::string &input_expr   = "");

    void parse_parameters(ParameterHandler &prm);

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

    virtual void set_time(const double newtime) override = 0;

  private:
    /**
     * Create the callbacks for the spatial and time derivatives.
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
  class ScalarSDParsedFunction : public ParsedFunctionSDBase<dim>
  {
  public:
    /**
     * Constructor
     */
    ScalarSDParsedFunction()
      : ParsedFunctionSDBase<dim>(1)
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

    void print_function(std::ostream &out) const
    {
      out << "f = " << this->function_object.get_expressions()[0] << std::endl;
    }

    void print_time_derivative(std::ostream &out) const
    {
      out << "dfdt = " << dfdt.get_expressions()[0] << std::endl;
    }

    void print_gradient(std::ostream &out) const
    {
      out << "grad f = " << std::endl;
      const auto expr = grad_function_object.get_expressions();
      for(unsigned int d = 0; d < dim; ++d)
        out << "\t" << expr[d] << std::endl;
    }

    // void print_hessian(std::ostream &out) const
    // {
    //   out << "grad f = " << std::endl;
    //   const auto expr = grad_function_object.get_expressions();
    //   for(unsigned int d = 0; d < dim; ++d)
    //     out << "\t" << expr[d] << std::endl;
    // }

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
  class VectorSDParsedFunction : public ParsedFunctionSDBase<dim>
  {
  public:
    /**
     * Constructor
     */
    VectorSDParsedFunction()
      : ParsedFunctionSDBase<dim>(dim)
      , dfdt(dim)
      , grad_function_object(dim)
      , hess_function_object(dim)
    {
      for(unsigned int d = 0; d < dim; ++d)
      {
        grad_function_object[d] = std::make_shared<FunctionParser<dim>>(dim);
        hess_function_object[d] = std::make_shared<FunctionParser<dim>>(dim * dim);
      }
    }

    virtual void set_time(const double newtime) override
    {
      this->function_object.set_time(newtime);
      dfdt.set_time(newtime);
      for (unsigned int d = 0; d < dim; ++d)
      {
        grad_function_object[d]->set_time(newtime);
        hess_function_object[d]->set_time(newtime);
      }
    }

    // The value function below has different arguments but hides the function from the base class
    using ParsedFunctionSDBase<dim>::value;

    /**
     * Fills the vector res with the dim components of this vector-valued
     * function.
     */
    void value(const Point<dim> &p, Tensor<1, dim> &res) const
    {
      for (unsigned int d = 0; d < dim; ++d)
        res[d] = this->function_object.value(p, d);
    }

    void time_derivative(const Point<dim> &p, Tensor<1, dim> &res) const
    {
      for (unsigned int d = 0; d < dim; ++d)
        res[d] = dfdt.value(p, d);
    }

    /**
     * Convention : grad_ij = \partial v_i/\partial x_j
     */
    void gradient_vi_xj(const Point<dim> &p, Tensor<2, dim> &grad) const
    {
      for (unsigned int di = 0; di < dim; ++di)
        for (unsigned int dj = 0; dj < dim; ++dj)
          grad[di][dj] = grad_function_object[di]->value(p, dj);
    }

    /**
     * Opposite convention
     * Note: tranpose(grad) creates a new tensor, so this is done in place
     * instead
     */
    void gradient_vj_xi(const Point<dim> &p, Tensor<2, dim> &grad) const
    {
      for (unsigned int di = 0; di < dim; ++di)
        for (unsigned int dj = 0; dj < dim; ++dj)
          grad[di][dj] = grad_function_object[dj]->value(p, di);
    }

    double divergence(const Point<dim> &p) const
    {
      double res = 0.;
      for (unsigned int d = 0; d < dim; ++d)
        res += grad_function_object[d]->value(p, d); // partial v_d/partial x_d
      return res;
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

    void print_function(std::ostream &out) const
    {
      out << "f = " << std::endl;
      const auto expr = this->function_object.get_expressions();
      for(unsigned int d = 0; d < dim; ++d)
        out << "\t" << expr[d] << std::endl;
    }

    void print_time_derivative(std::ostream &out) const
    {
      out << "dfdt = " << std::endl;
      const auto expr = dfdt.get_expressions();
      for(unsigned int d = 0; d < dim; ++d)
        out << "\t" << expr[d] << std::endl;
    }

    void print_gradient(std::ostream &out) const
    {
      out << "grad f = " << std::endl;
      for(unsigned int di = 0; di < dim; ++di)
      {
        out << "dv_" << di << "/dx_j = ";
        const auto expr = grad_function_object[di]->get_expressions();
        for(unsigned int dj = 0; dj < dim; ++dj)
          out << "\t" << expr[dj];
        out << std::endl;
      }

      out << "grad f check = " << std::endl;
      for(const auto &grad_comp : grad_function_object)
      {
        out << "dv_comp/dx_j = ";
        const auto expr = grad_comp->get_expressions();
        for(auto str : expr)
          out << "\t" << str;
        out << std::endl;
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