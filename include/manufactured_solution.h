#ifndef MANUFACTURED_SOLUTION_H
#define MANUFACTURED_SOLUTION_H

#include <deal.II/base/function.h>
#include <deal.II/base/parameter_handler.h>

DeclExceptionMsg(TimeDerivativeIsIgnored,
                 "You are trying to use a function requiring time derivatives, "
                 "but they are not implemented for this manufactured field "
                 "(ignore_time_derivative = true).");
DeclExceptionMsg(HessianIsIgnored,
                 "You are trying to use a function requiring second-order "
                 "derivatives, but the hessian for this manufactured field was "
                 "not implemented (ignore_hessian = true).");

namespace ManufacturedSolutions
{
  using namespace dealii;

  enum class PresetMMS
  {
    none,
    time_dependent_vector,
    rigid_motion_kernel,
    moving_radial_kernel,
    normal_radial_kernel
  };

  template <int dim>
  class MMSFunction;
  template <int dim>
  class ParsedFunctionSDBase;

  /**
   * This class parses and handles the exact solution for the various fields.
   * These solutions are typically used to run convergence studies through the
   * method of manufactured solutions (MMS), but an exact solution for one or
   * more fields can also be imposed while solving for the remaining variables.
   *
   * In the parameter file, exact solutions are set in the Exact solution
   * subsection.
   */
  template <int dim>
  class ManufacturedSolution
  {
  public:
    ManufacturedSolution() {}

    void set_time(const double new_time)
    {
      exact_velocity->set_time(new_time);
      exact_pressure->set_time(new_time);
      exact_mesh_position->set_time(new_time);
      exact_tracer->set_time(new_time);
      exact_potential->set_time(new_time);
      exact_temperature->set_time(new_time);
    }

    void declare_parameters(ParameterHandler &prm);
    void read_parameters(ParameterHandler &prm);

  public:
    std::shared_ptr<MMSFunction<dim>> exact_velocity;
    std::shared_ptr<MMSFunction<dim>> exact_pressure;
    std::shared_ptr<MMSFunction<dim>> exact_mesh_position;
    std::shared_ptr<MMSFunction<dim>> exact_tracer;
    std::shared_ptr<MMSFunction<dim>> exact_potential;
    std::shared_ptr<MMSFunction<dim>> exact_temperature;

    std::map<std::string, std::shared_ptr<MMSFunction<dim>>> exact_solution;

    // If true, the field is specified instead of being solved for
    std::map<std::string, bool> set_field_as_solution;

  private:
    PresetMMS preset_velocity_type      = PresetMMS::none;
    PresetMMS preset_pressure_type      = PresetMMS::none;
    PresetMMS preset_mesh_position_type = PresetMMS::none;
    PresetMMS preset_tracer_type        = PresetMMS::none;
    PresetMMS preset_potential_type     = PresetMMS::none;
    PresetMMS preset_temperature_type   = PresetMMS::none;
  };

  /**
   * A Function<dim> which offers some additional quality-of-life functions
   * that are useful when evaluating source terms for convergence studies with
   * manufactured solutions, such as:
   *
   * - time_derivative
   * - vector gradient and laplacian
   * - divergence, only defined for vector-valued functions
   * - gradient of divergence
   * - divergence of linear elastic stress tensor
   *
   * (Computing the newtonian stress tensor for fluids requires both velocity
   * and pressure, so this needs to be assembled from two MMSFunctions.)
   *
   * An MMS function is meant to represent scalar- or vector-valued functions,
   * so it is only possible to create such a function with n_components = 1 or
   * dim.
   *
   * To make sure that the derived functions from Function<dim> are implemented,
   * tests should be run in debug at least once, since deal.II functions throw
   * if the "pure" functions are called (they are default implemented to return
   * 0). In particular, these functions are *not* re-declared here as pure
   * virtual to avoid cluttering. Unless specific functions are ignored by
   * passing the appropriate flag, each function deriving from MMSFunction
   * should provide an implementation for:
   *
   * - value
   * - time derivative
   * - gradient
   * - hessian
   *
   */
  template <int dim>
  class MMSFunction : public Function<dim>
  {
  public:
    MMSFunction(const unsigned int n_components,
                const bool         ignore_time_derivative = false,
                const bool         ignore_hessian         = false,
                const double       initial_time           = 0.)
      : Function<dim>(n_components, initial_time)
      , ignore_time_derivative(ignore_time_derivative)
      , ignore_hessian(ignore_hessian)
    {
      AssertThrow(n_components == 1 || n_components == dim,
                  ExcMessage(
                    "You are trying to create an MMSFunction with " +
                    std::to_string(n_components) +
                    ", but this kind of function represents a scalar-valued "
                    "(with 1 component) or a vector-valued (with dim "
                    "components) field."));
    }

  public:
    /**
     * If true, then the time derivatives and/or Hessian are not implemented.
     * This is used for e.g. pressure fields, where only the gradient is
     * needed for now.
     */
    const bool ignore_time_derivative;
    const bool ignore_hessian;

  public:
    /**
     * Time derivative
     */
    virtual double time_derivative(const Point<dim>  &p,
                                   const unsigned int component = 0) const = 0;
    /**
     * Laplacian (deal.II override)
     */
    virtual double
    laplacian(const Point<dim>  &p,
              const unsigned int component = 0) const override final
    {
      Assert(!ignore_hessian, HessianIsIgnored());
      double                        res  = 0.;
      const SymmetricTensor<2, dim> hess = this->hessian(p, component);
      for (unsigned int d = 0; d < dim; ++d)
        res += hess[d][d];
      return res;
    }

    /**
     * Gradient of vector field with convention:
     *              grad_ij = \partial v_i/\partial x_j
     */
    virtual Tensor<2, dim> gradient_vi_xj(const Point<dim> &p) const final
    {
      Assert(this->n_components == dim,
             ExcMessage(
               "Vector gradient() is defined only if n_components == dim"));
      Tensor<2, dim> grad;
      for (unsigned int di = 0; di < dim; ++di)
      {
        Tensor<1, dim> grad_comp = this->gradient(p, di);
        for (unsigned int dj = 0; dj < dim; ++dj)
          grad[di][dj] = grad_comp[dj];
      }
      return grad;
    }

    /**
     * Gradient of vector field with opposite convention.
     */
    virtual Tensor<2, dim> gradient_vj_xi(const Point<dim> &p) const final
    {
      Assert(this->n_components == dim,
             ExcMessage(
               "Vector gradient() is defined only if n_components == dim"));
      return transpose(this->gradient_vi_xj(p));
    }

    /**
     * Vector laplacian.
     */
    virtual Tensor<1, dim> vector_laplacian(const Point<dim> &p) const final
    {
      Assert(this->n_components == dim,
             ExcMessage(
               "Vector laplacian() is defined only if n_components == dim"));
      Tensor<1, dim> res;
      for (unsigned int d = 0; d < dim; ++d)
        res[d] = this->laplacian(p, d);
      return res;
    }

    /**
     * Divergence of vector-valued field.
     */
    virtual double divergence(const Point<dim> &p) const final
    {
      Assert(this->n_components == dim,
             ExcMessage("divergence() is defined only if n_components == dim"));
      double res = 0.;
      for (unsigned int d = 0; d < dim; ++d)
        res += this->gradient(p, d)[d];
      return res;
    }

    /**
     * Grad-div of vector-valued field. This is res_i = d^2 u_k/dx_k dx_i.
     */
    virtual Tensor<1, dim> grad_div(const Point<dim> &p) const final
    {
      Assert(!ignore_hessian, HessianIsIgnored());
      Assert(this->n_components == dim,
             ExcMessage("grad_div() is defined only if n_components == dim"));
      Tensor<1, dim> res;
      for (unsigned int dk = 0; dk < dim; ++dk)
      {
        // Get hess of u_k
        const SymmetricTensor<2, dim> hess = this->hessian(p, dk);
        for (unsigned int di = 0; di < dim; ++di)
          res[di] += hess[dk][di];
      }
      return res;
    }

    /**
     * Divergence of the isothermal linear elastic stress tensor.
     *
     * The stress tensor is sigma = 2 * mu * eps(u) + lambda * trace(eps(u)) *
     * I, where eps(u) = (grad(u) + grad(u)^T)/2 is the infinitesimal strain
     * tensor.
     *
     * Assuming constant Lamé coefficients, div(sigma) is given by:
     *         mu * lap(u) + (lambda + mu) * grad(div(u)).
     */
    virtual Tensor<1, dim>
    divergence_linear_elastic_stress(const Point<dim> &p,
                                     const double      lame_mu,
                                     const double      lame_lambda) const final
    {
      return lame_mu * this->vector_laplacian(p) +
             (lame_mu + lame_lambda) * this->grad_div(p);
    }

    /**
     * Attention: here we consider eps = (grad(u) + grad(u)^T)/2
     *                                 = (grad(x) + grad(x)^T)/2 - I
     */
    virtual Tensor<1, dim>
    divergence_linear_elastic_stress_variable_coefficients(
      const Point<dim>                          &p,
      std::shared_ptr<ParsedFunctionSDBase<dim>> lame_mu,
      std::shared_ptr<ParsedFunctionSDBase<dim>> lame_lambda) const final;

    /**
     * Check that the provided time and spatial derivatives match
     * with their finite differences approximations.
     * This needs to be called from the constructor of each derived class.
     */
    void check_derivatives(const double tol_order_1 = 1e-6,
                           const double tol_order_2 = 1e-4);
  };
} // namespace ManufacturedSolutions

#endif