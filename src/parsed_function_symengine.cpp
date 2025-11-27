
#include <deal.II/base/function_parser.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/differentiation/sd/symengine_tensor_operations.h>
#include <parsed_function_symengine.h>

namespace ManufacturedSolutions
{
  using namespace Differentiation::SD;

  template <int dim>
  ParsedFunctionSDBase<dim>::ParsedFunctionSDBase(
    const unsigned int n_components)
    : MMSFunction<dim>(n_components, 0.)
    , n_components(n_components)
    , function_object(n_components)
  {}

  /**
   * This is exactly the deal.II function (parsed_function.cc)
   */
  template <int dim>
  void ParsedFunctionSDBase<dim>::declare_parameters(
    ParameterHandler  &prm,
    const unsigned int n_components,
    const std::string &input_expr)
  {
    Assert(n_components > 0, ExcZero());

    std::string vnames;
    switch (dim)
    {
      case 1:
        vnames = "x,t";
        break;
      case 2:
        vnames = "x,y,t";
        break;
      case 3:
        vnames = "x,y,z,t";
        break;
      default:
        AssertThrow(false, ExcNotImplemented());
        break;
    }
    prm.declare_entry(
      "Variable names",
      vnames,
      Patterns::Anything(),
      "The names of the variables as they will be used in the "
      "function, separated by commas. By default, the names of variables "
      "at which the function will be evaluated are `x' (in 1d), `x,y' (in 2d) "
      "or "
      "`x,y,z' (in 3d) for spatial coordinates and `t' for time. You can then "
      "use these variable names in your function expression and they will be "
      "replaced by the values of these variables at which the function is "
      "currently evaluated. However, you can also choose a different set "
      "of names for the independent variables at which to evaluate your "
      "function "
      "expression. For example, if you work in spherical coordinates, you may "
      "wish to set this input parameter to `r,phi,theta,t' and then use these "
      "variable names in your function expression.");

    // The expression of the function
    // If the string is an empty string, 0 is set for each components.
    std::string expr = input_expr;
    if (expr == "")
    {
      expr = "0";
      for (unsigned int i = 1; i < n_components; ++i)
        expr += "; 0";
    }
    else
    {
      // If the user specified an input expr, the number of component
      // specified need to match n_components.
      AssertDimension((std::count(expr.begin(), expr.end(), ';') + 1),
                      n_components);
    }


    prm.declare_entry(
      "Function expression",
      expr,
      Patterns::Anything(),
      "The formula that denotes the function you want to evaluate for "
      "particular values of the independent variables. This expression "
      "may contain any of the usual operations such as addition or "
      "multiplication, as well as all of the common functions such as "
      "`sin' or `cos'. In addition, it may contain expressions like "
      "`if(x>0, 1, -1)' where the expression evaluates to the second "
      "argument if the first argument is true, and to the third argument "
      "otherwise. For a full overview of possible expressions accepted "
      "see the documentation of the muparser library at "
      "http://muparser.beltoforion.de/."
      "\n\n"
      "If the function you are describing represents a vector-valued "
      "function with multiple components, then separate the expressions "
      "for individual components by a semicolon.");
    prm.declare_entry(
      "Function constants",
      "",
      Patterns::Anything(),
      "Sometimes it is convenient to use symbolic constants in the "
      "expression that describes the function, rather than having to "
      "use its numeric value everywhere the constant appears. These "
      "values can be defined using this parameter, in the form "
      "`var1=value1, var2=value2, ...'."
      "\n\n"
      "A typical example would be to set this runtime parameter to "
      "`pi=3.1415926536' and then use `pi' in the expression of the "
      "actual formula. (That said, for convenience this class actually "
      "defines both `pi' and `Pi' by default, but you get the idea.)");
  }

  template <int dim>
  void ParsedFunctionSDBase<dim>::parse_parameters(ParameterHandler &prm)
  {
    std::string vnames         = prm.get("Variable names");
    std::string expression     = prm.get("Function expression");
    std::string constants_list = prm.get("Function constants");

    std::vector<std::string> const_list =
      Utilities::split_string_list(constants_list, ',');
    std::map<std::string, double> constants;
    for (const auto &constant : const_list)
    {
      std::vector<std::string> this_c =
        Utilities::split_string_list(constant, '=');
      AssertThrow(this_c.size() == 2,
                  ExcMessage("The list of constants, <" + constants_list +
                             ">, is not a comma-separated list of "
                             "entries of the form 'name=value'."));
      constants[this_c[0]] = Utilities::string_to_double(this_c[1]);
    }

    // set pi and Pi as synonyms for the corresponding value. note that
    // this overrides any value a user may have given
    constants["pi"] = numbers::PI;
    constants["Pi"] = numbers::PI;

    bool time_dependent = false;

    const unsigned int nn = (Utilities::split_string_list(vnames)).size();
    switch (nn)
    {
      case dim:
        // Time independent function
        function_object.initialize(vnames, expression, constants);
        break;
      case dim + 1:
        // Time dependent function
        time_dependent = true;
        function_object.initialize(vnames, expression, constants, true);
        break;
      default:
        AssertThrow(false,
                    ExcMessage("The list of variables specified is <" + vnames +
                               "> which is a list of length " +
                               Utilities::int_to_string(nn) +
                               " but it has to be a list of length equal to" +
                               " either dim (for a time-independent function)" +
                               " or dim+1 (for a time-dependent function)."));
    }

    /**
     * This is the change from deal.II's function.
     */
    this->create_symbolic_derivatives(vnames, constants, time_dependent);
  }

  namespace {

    // SymEngine parses exponents as "**", whereas muParser expects "^".
    // This function replaces the all **'s in a string by ^'s.
    std::string replace_all_exponents(std::string s)
    {
      size_t pos = 0;
      const std::string from = "**";
      const std::string to   = "^";
      while ((pos = s.find(from, pos)) != std::string::npos) {
        s.replace(pos, from.length(), to);
        pos += to.length();
      }
      return s;
    }
  }

  template <int dim>
  void ScalarSDParsedFunction<dim>::create_symbolic_derivatives(
    const std::string                   variables,
    const std::map<std::string, double> constants,
    const bool                          time_dependent)
  {
    // Get the parsed expression
    const std::string expr = this->function_object.get_expressions()[0];

    Tensor<1, dim, Expression> independent_variables;
    independent_variables[0] = Expression("x");
    independent_variables[1] = Expression("y");
    if constexpr (dim == 3)
      independent_variables[2] = Expression("z");
    const Expression time("t");

    // Set the function itself as a symbolic expression
    const Expression f(expr, true);

    //
    // Get symbolic gradient
    //
    const Tensor<1, dim, Expression> grad_f =
      differentiate(f, independent_variables);

    // Get the string expressions of the spatial derivatives
    std::vector<std::string> grad_expressions;
    for (unsigned int d = 0; d < dim; ++d)
    {
      std::stringstream sstream;
      sstream << grad_f[d];
      grad_expressions.push_back(replace_all_exponents(sstream.str()));
    }
    grad_function_object.initialize(variables,
                                    grad_expressions,
                                    constants,
                                    time_dependent);

    //
    // Get symbolic hessian
    //
    // Get the string expression of the 2nd spatial derivatives
    std::vector<std::string> hess_expressions;
    for (unsigned int di = 0; di < dim; ++di)
    {
      // Get symbolic gradient of gradient component
      const Tensor<1, dim, Expression> hess_i =
        differentiate(grad_f[di], independent_variables);

      for (unsigned int dj = 0; dj < dim; ++dj)
      {
        std::stringstream sstream;
        sstream << hess_i[dj];
        hess_expressions.push_back(replace_all_exponents(sstream.str()));
      }
    }
    hess_function_object.initialize(variables,
                                    hess_expressions,
                                    constants,
                                    time_dependent);

    //
    // Get time derivative
    //
    const Expression fdot = f.differentiate(time);
    {
      std::stringstream sstream;
      sstream << fdot;
      dfdt.initialize(variables, replace_all_exponents(sstream.str()), constants, time_dependent);
    }
  }

  template <int dim>
  void VectorSDParsedFunction<dim>::create_symbolic_derivatives(
    const std::string                   variables,
    const std::map<std::string, double> constants,
    const bool                          time_dependent)
  {
    // Semicolon separated list of time derivatives (one per vector component)
    std::string time_derivatives;

    for (unsigned int i_comp = 0; i_comp < dim; ++i_comp)
    {
      // Get the parsed expression
      const std::string expr = this->function_object.get_expressions()[i_comp];

      Tensor<1, dim, Expression> independent_variables;
      independent_variables[0] = Expression("x");
      independent_variables[1] = Expression("y");
      if constexpr (dim == 3)
        independent_variables[2] = Expression("z");
      const Expression time("t");

      // Set the vector component as a symbolic expression
      const Expression f(expr, true);

      //
      // Get symbolic gradient of component
      //
      const Tensor<1, dim, Expression> grad_f =
        differentiate(f, independent_variables);

      // Get the string expressions of the spatial derivatives
      std::vector<std::string> grad_expressions;
      for (unsigned int d = 0; d < dim; ++d)
      {
        std::stringstream sstream;
        sstream << grad_f[d];
        grad_expressions.push_back(replace_all_exponents(sstream.str()));
      }
      grad_function_object[i_comp]->initialize(variables,
                                               grad_expressions,
                                               constants,
                                               time_dependent);
      //
      // Get symbolic hessian of component
      //
      // Get the string expression of the 2nd spatial derivatives
      std::vector<std::string> hess_expressions;
      for (unsigned int di = 0; di < dim; ++di)
      {
        // Get symbolic gradient of gradient component
        const Tensor<1, dim, Expression> hess_i =
          differentiate(grad_f[di], independent_variables);

        for (unsigned int dj = 0; dj < dim; ++dj)
        {
          std::stringstream sstream;
          sstream << hess_i[dj];
          hess_expressions.push_back(replace_all_exponents(sstream.str()));
        }
      }
      hess_function_object[i_comp]->initialize(variables,
                                               hess_expressions,
                                               constants,
                                               time_dependent);

      //
      // Get time derivative
      //
      const Expression fdot = f.differentiate(time);
      {
        std::stringstream sstream;
        sstream << fdot;
        time_derivatives += replace_all_exponents(sstream.str()) + ";";
      }
    }
    dfdt.initialize(variables, time_derivatives, constants, time_dependent);
  }

  // Explicit instantiations
  template class ParsedFunctionSDBase<2>;
  template class ParsedFunctionSDBase<3>;
  template class ScalarSDParsedFunction<2>;
  template class ScalarSDParsedFunction<3>;
  template class VectorSDParsedFunction<2>;
  template class VectorSDParsedFunction<3>;
} // namespace ManufacturedSolution