#ifndef STABILIZATION_TOOLS_H
#define STABILIZATION_TOOLS_H

#include <deal.II/base/tensor.h>
#include <parameter_reader.h>
#include <solver_info.h>
#include <time_handler.h>

#include <vector>

/**
 * This namespace gathers routines related to the stabilization of convection-
 * diffusion equations, notably for the computation of the SUPG/PSPG terms to
 * stabilize the Navier-Stokes equations.
 */
namespace StabilizationTools
{
  using namespace dealii;

  /**
   * Compute the *square* of the cell length used to define the stabilization
   * parameter tau_SUPG, since this length only intervenes through its square in
   * the definition of tau_SUPG.
   *
   * In two dimensions, we use Tezduyar's definition which estimates the length
   * of the cell
   * in the direction of the @p convective velocity u:
   *
   * h := 2 * ||u|| / (sum_i |u \cdot grad phi_i|),
   *
   * where the phi_i are the isoparametric, scalar-valued shape functions of the
   * field to stabilize.
   *
   * FIXME/ONGOING: This definition seems not well suited for 3D anisotropic
   * elements. For instance, consider a velocity field u = (1, 0, 1) and the
   * tetrahedra parameterized by the constant "a" and described by:
   *
   * t1 = (0 0 0), (1 0 0), (1 a 0), (1 a 1)
   * t2 = (0 0 0), (1 0 0), (1 a 1), (1 0 1).
   *
   * This definition yields h_1 = 1, but h_2 = sqrt(2), and these values are
   * unchanged even though both triangles become the same as a goes to zero.
   *
   * In three dimensions, the cell length is defined as its diameter, and is
   * computed using deal.II's diameter() function (longest diagonal for
   * quads/hexes and longest side for simplices).
   *
   * FIXME: This does not handle anisotropic elements well, as it simply returns
   * the longest side.
   */
  template <int dim>
  inline double
  compute_h_tau_squared(const unsigned int    n_dofs_per_cell,
                        const double          cell_diameter,
                        const Tensor<1, dim> &convective_velocity,
                        const double          velocity_norm_squared,
                        const std::vector<Tensor<1, dim>> &grad_scalar_shape);

  /**
   * Compute the *square* of the cell length used to define the stabilization
   * parameter according to Tezduyar:
   *
   * h_squared := (2 * ||u|| / (sum_i |u \cdot grad phi_i|))^2.
   */
  template <int dim>
  double compute_squared_cell_length_along_velocity_field(
    const unsigned int                 n_dofs_per_cell,
    const double                       cell_diameter,
    const Tensor<1, dim>              &convective_velocity,
    const double                       velocity_norm_squared,
    const std::vector<Tensor<1, dim>> &grad_scalar_shape);

  /**
   * Compute the stabilization parameter tau_SUPG according to Tezduyar's
   * definition, including the modification for higher-order elements from
   * Saavedra, L. P., Munch, P., & Blais, B. (2025). A matrix-free stabilized
   * solver for the incompressible Navier-Stokes equations. This writes:
   *
   *             /      2                 2                     2 \ -1/2
   *            | / 1  \    / 2 ||u|| p  \        /  4 diff p^2 \ |
   * tau_SUPG = | |----|  + | ----------- |   + 9 | ----------- | | ,
   *            | \ dt /     \  h_conv   /        \  h_diff^2   / |
   *             \                                               /
   *
   * where dt is the current time step, u is the @p convective_velocity, p is
   * the polynomial @p degree of the field to stabilize, h_conv = h_diff are
   * taken to be identical and given by the result of compute_h_tau_squared(),
   * and diff is the characteristic @p diffusivity coefficient of the convection-
   * diffusion problem of interest.
   *
   * For steady-state computations, the first term is ignored.
   *
   * This definition differs from Garon & Fortin's definition in the coefficient
   * in the unsteady term.
   *
   */
  template <int dim>
  inline double
  compute_tau_supg(const TimeHandler                 &time_handler,
                   const unsigned int                 n_dofs_per_cell,
                   const double                       cell_diameter,
                   const unsigned int                 degree,
                   const double                       diffusivity,
                   const Tensor<1, dim>              &convective_velocity,
                   const std::vector<Tensor<1, dim>> &grad_scalar_shape);
} // namespace StabilizationTools

/* ---------------- Template functions ----------------- */

namespace StabilizationTools
{
  template <int dim>
  inline double
  compute_h_tau_squared(const unsigned int    n_dofs_per_cell,
                        const double          cell_diameter,
                        const Tensor<1, dim> &convective_velocity,
                        const double          velocity_norm_squared,
                        const std::vector<Tensor<1, dim>> &grad_scalar_shape)
  {
    if constexpr (dim == 2)
    {
      // In 2D, use Verdier's formula to compute the squared cell length.
      return compute_squared_cell_length_along_velocity_field(
        n_dofs_per_cell,
        cell_diameter,
        convective_velocity,
        velocity_norm_squared,
        grad_scalar_shape);
    }
    else
    {
      // In 3D, use the cell diameter.
      return cell_diameter * cell_diameter;
    }
    DEAL_II_ASSERT_UNREACHABLE();
  }

  template <int dim>
  inline double compute_squared_cell_length_along_velocity_field(
    const unsigned int                 n_dofs_per_cell,
    const double                       cell_diameter,
    const Tensor<1, dim>              &convective_velocity,
    const double                       velocity_norm_squared,
    const std::vector<Tensor<1, dim>> &grad_scalar_shape)
  {
    if (velocity_norm_squared < 1e-12)
      // Return *square* of cell_diameter
      return cell_diameter * cell_diameter;

    double sum = 0.;
    for (unsigned int i = 0; i < n_dofs_per_cell; ++i)
      sum += std::abs(convective_velocity * grad_scalar_shape[i]);
    return 4. * velocity_norm_squared / (sum * sum);
  }

  template <int dim>
  inline double
  compute_tau_supg(const TimeHandler                 &time_handler,
                   const unsigned int                 n_dofs_per_cell,
                   const double                       cell_diameter,
                   const unsigned int                 degree,
                   const double                       diffusivity,
                   const Tensor<1, dim>              &convective_velocity,
                   const std::vector<Tensor<1, dim>> &grad_scalar_shape)
  {
    const double velocity_norm_squared = convective_velocity.norm_square();
    const double one_over_h_tau_squared =
      1. / compute_h_tau_squared(n_dofs_per_cell,
                                 cell_diameter,
                                 convective_velocity,
                                 velocity_norm_squared,
                                 grad_scalar_shape);

    Assert(one_over_h_tau_squared > 0, ExcInternalError());

    const double conv =
      4. * velocity_norm_squared * degree * degree * one_over_h_tau_squared;
    double diff = 4. * diffusivity * degree * degree * one_over_h_tau_squared;
    diff *= 9. * diff;

    double tau = conv + diff;
    if (!time_handler.is_steady())
    {
      const double unsteady = 1. / time_handler.get_current_timestep();
      tau += unsteady * unsteady;
    }

    Assert(tau > 0, ExcInternalError());

    tau = pow(tau, -0.5);
    return tau;
  }
} // namespace StabilizationTools

#endif
