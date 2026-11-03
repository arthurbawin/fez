#ifndef CAHN_HILLIARD_H
#define CAHN_HILLIARD_H

#include <parameters.h>
#include <deal.II/base/tensor.h>
#include <deal.II/lac/vector_type_traits.h>

#include <cmath>
#include <vector>

namespace CahnHilliard
{
  constexpr double profile_correction_tail = 1. - 0.999 * 0.999;

  struct ProfileInverse
  {
    double value;
    double derivative;
  };

  /** Inverse tanh at phase nodes, with an odd C2 continuation near +/-1.
   * The continuation has bounded positive slope, so pure phases and nodal
   * overshoots remain finite without clipping the transported phase.
   */
  inline ProfileInverse inverse_profile(const double phi)
  {
    const double cutoff = std::sqrt(1. - 0.25 * profile_correction_tail);
    if (std::abs(phi) <= cutoff)
      return {std::atanh(phi), 1. / (1. - phi * phi)};
    const double slope  = 1. / (1. - cutoff * cutoff);
    const double length = (1. - cutoff * cutoff) / cutoff;
    const double offset = std::abs(phi) - cutoff;
    return {std::copysign(std::atanh(cutoff) +
                            slope *
                              (3. * offset +
                               2. * length * std::expm1(-offset / length)),
                          phi),
            slope * (3. - 2. * std::exp(-offset / length))};
  }

  template <int dim>
  struct ProfileVariation
  {
    double                 value;
    dealii::Tensor<1, dim> gradient;
    dealii::Tensor<2, dim> hessian;
  };

  /** The correction uses phi_r=tanh(s_h), where s_h interpolates the inverse
   * phase values at the nodes. The CH equation and mobility still use phi_h.
   * Store derivatives of s_h to differentiate the reconstruction without
   * allocating another set of shape-function tables.
   */
  template <int dim>
  struct ReconstructedProfile
  {
    double                 value = 0.;
    dealii::Tensor<1, dim> inverse_gradient;
    dealii::Tensor<2, dim> inverse_hessian;

    dealii::Tensor<1, dim> gradient() const
    {
      return (1. - value * value) * inverse_gradient;
    }

    dealii::Tensor<2, dim> hessian() const
    {
      return (1. - value * value) *
             (inverse_hessian -
              2. * value * outer_product(inverse_gradient, inverse_gradient));
    }

    ProfileVariation<dim> variation(const double                  ds,
                                    const dealii::Tensor<1, dim> &dg,
                                    const dealii::Tensor<2, dim> &dh,
                                    const bool with_hessian) const
    {
      const double          first  = 1. - value * value;
      const double          second = -2. * value * first;
      ProfileVariation<dim> result{first * ds,
                                   first * dg + second * ds * inverse_gradient,
                                   {}};
      if (with_hessian)
      {
        const double third = -2. * (1. - 3. * value * value) * first;
        result.hessian =
          first * dh + second * ds * inverse_hessian +
          third * ds * outer_product(inverse_gradient, inverse_gradient) +
          second * (outer_product(dg, inverse_gradient) +
                    outer_product(inverse_gradient, dg));
      }
      return result;
    }
  };

  /** Read raw constrained phase coefficients from the current ghosted state. */
  template <int dim, typename CellIterator, typename VectorType>
  std::vector<ProfileInverse>
  profile_inverse_values(const CellIterator &cell,
                         const VectorType   &solution,
                         const unsigned int  phase_component)
  {
    const auto &fe = cell->get_fe();
    std::vector<dealii::types::global_dof_index> indices(fe.n_dofs_per_cell());
    cell->get_dof_indices(indices);
    std::vector<ProfileInverse> nodal(indices.size());
    for (unsigned int j = 0; j < indices.size(); ++j)
      if (fe.system_to_component_index(j).first == phase_component)
      {
        if constexpr (!dealii::is_serial_vector<VectorType>::value)
        {
          AssertThrow(solution.in_local_range(indices[j]) ||
                        solution.has_ghost_elements(),
                      dealii::ExcMessage("Profile reconstruction requires "
                                         "locally relevant ghost values."));
#if defined(FEZ_WITH_PETSC)
          AssertThrow(solution.in_local_range(indices[j]) ||
                        solution.ghost_elements().is_element(indices[j]),
                      dealii::ExcMessage("Missing phase ghost value in "
                                         "profile reconstruction."));
#endif
        }
        nodal[j] = inverse_profile(solution[indices[j]]);
      }
    return nodal;
  }

  /** Shared cell-local reconstruction for assembly and diagnostics. */
  template <int dim>
  ReconstructedProfile<dim> reconstruct_profile(
    const std::vector<ProfileInverse>        &nodal,
    const std::vector<double>                &shape,
    const std::vector<dealii::Tensor<1, dim>> &gradients)
  {
    AssertDimension(nodal.size(), shape.size());
    AssertDimension(nodal.size(), gradients.size());
    ReconstructedProfile<dim> profile;
    double inverse_value = 0.;
    for (unsigned int j = 0; j < nodal.size(); ++j)
    {
      inverse_value += nodal[j].value * shape[j];
      profile.inverse_gradient += nodal[j].value * gradients[j];
    }
    profile.value = std::tanh(inverse_value);
    return profile;
  }

  template <int dim>
  struct ProfileCorrection
  {
    dealii::Tensor<1, dim> flux;
    double                 divergence = 0.;
  };

  /** Derivatives of the profile flux and its divergence at one quadrature
   * point. The global mobility amplitude is held constant during linearization.
   */
  template <int dim>
  struct ProfileCorrectionLinearization
  {
    dealii::Tensor<1, dim> flux_phi;
    dealii::Tensor<2, dim> flux_gradient;
    double                 divergence_phi = 0.;
    dealii::Tensor<1, dim> divergence_gradient;

    ProfileCorrection<dim>
    variation(const double                  dphi,
              const dealii::Tensor<1, dim> &dgradient,
              const dealii::Tensor<2, dim> &dhessian) const
    {
      return {dphi * flux_phi + flux_gradient * dgradient,
              divergence_phi * dphi + divergence_gradient * dgradient +
                scalar_product(flux_gradient, dhessian)};
    }
  };

  /**
   * Regularized profile flux F = a(phi)*(1-c*q/R)*grad(phi), where
   * q=(1-phi^2)/(sqrt(2)*epsilon), R^2=|grad(phi)|^2+beta^2*q^2 and
   * c=sqrt(1+beta^2). Thus F vanishes for the equilibrium tanh profile.
   * The C2 activation a turns the correction off near the pure phases;
   * beta regularizes the normal where the phase gradient vanishes.
   * The caller supplies the global mobility amplitude separately.
   * When requested, compute the derivatives once for all trial functions.
   */
  template <int dim>
  ProfileCorrection<dim> evaluate_profile_correction(
    const double                         phi,
    const dealii::Tensor<1, dim>        &gradient,
    const dealii::Tensor<2, dim>        &hessian,
    const double                         epsilon,
    const bool                           compute_divergence = true,
    ProfileCorrectionLinearization<dim> *linearization      = nullptr)
  {
    if (linearization)
      *linearization = {};
    constexpr double beta         = 0.05;
    constexpr double tail         = profile_correction_tail;
    const double     phase_factor = 1. - phi * phi;
    if (phase_factor <= 0.25 * tail)
      return {};

    double a = 1., da = 0., dda = 0.;
    if (phase_factor < 0.5 * tail)
    {
      const double width = 0.25 * tail;
      const double t     = (phase_factor - width) / width;
      const double dt    = -2. * phi / width;
      const double first = 30. * t * t * (t - 1.) * (t - 1.);
      a                  = t * t * t * (10. + t * (-15. + 6. * t));
      da                 = first * dt;
      if (linearization && compute_divergence)
        dda =
          60. * t * (2. * t * t - 3. * t + 1.) * dt * dt - 2. * first / width;
    }
    const double           q = phase_factor / (std::sqrt(2.) * epsilon);
    const double           s = gradient.norm_square();
    const double           R = std::sqrt(s + beta * beta * q * q);
    const double           c = std::sqrt(1. + beta * beta);
    const double           b = 1. - c * q / R;
    const double           f = a * b;
    ProfileCorrection<dim> result;
    result.flux = f * gradient;
    if (!compute_divergence && !linearization)
      return result;

    // With F = f(phi,s)*grad(phi), s=|grad(phi)|^2, its divergence is
    // f*tr(H) + f_phi*s + 2*f_s*grad(phi).H.grad(phi).
    const double dq     = -2. * phi / (std::sqrt(2.) * epsilon);
    const double inv_R3 = 1. / (R * R * R);
    const double b_phi  = -c * dq * s * inv_R3;
    const double b_s    = 0.5 * c * q * inv_R3;
    const double f_phi  = da * b + a * b_phi;
    const double f_s    = a * b_s;
    const auto   Hg     = hessian * gradient;
    const double gHg    = gradient * Hg;
    if (compute_divergence)
      result.divergence = f * trace(hessian) + f_phi * s + 2. * f_s * gHg;
    if (linearization)
    {
      linearization->flux_phi = f_phi * gradient;
      linearization->flux_gradient =
        2. * f_s * outer_product(gradient, gradient);
      for (unsigned int d = 0; d < dim; ++d)
        linearization->flux_gradient[d][d] += f;
      if (compute_divergence)
      {
        const double ddq    = -2. / (std::sqrt(2.) * epsilon);
        const double inv_R5 = inv_R3 / (R * R);
        const double b_pp =
          -c * s * (ddq * inv_R3 - 3. * beta * beta * q * dq * dq * inv_R5);
        const double b_ps = -c * dq * (inv_R3 - 1.5 * s * inv_R5);
        const double b_ss = -0.75 * c * q * inv_R5;
        const double f_pp = dda * b + 2. * da * b_phi + a * b_pp;
        const double f_ps = da * b_s + a * b_ps;
        const double f_ss = a * b_ss;
        linearization->divergence_phi =
          f_phi * trace(hessian) + f_pp * s + 2. * f_ps * gHg;
        linearization->divergence_gradient =
          (2. * f_s * trace(hessian) + 2. * f_ps * s + 2. * f_phi +
           4. * f_ss * gHg) *
            gradient +
          4. * f_s * Hg;
      }
    }
    return result;
  }


  template <int dim>
  inline bool
  is_quasi_incompressible_model(const Parameters::CahnHilliard<dim> &param)
  {
    return param.chns_model ==
           Parameters::CahnHilliard<dim>::CHNSModel::QuasiIncompressible;
  }

  template <int dim>
  inline bool
  is_abels_model(const Parameters::CahnHilliard<dim> &param)
  {
    return param.chns_model == Parameters::CahnHilliard<dim>::CHNSModel::Abels;
  }

  template <int dim>
  inline bool
  is_ding_horriche_model(const Parameters::CahnHilliard<dim> &param)
  {
    return param.chns_model ==
           Parameters::CahnHilliard<dim>::CHNSModel::DingHorriche;
  }

  template <int dim>
  inline bool
  is_stepien_model(const Parameters::CahnHilliard<dim> &param)
  {
    return param.chns_model ==
           Parameters::CahnHilliard<dim>::CHNSModel::Stepien;
  }

  template <int dim>
  inline bool
  use_abels_diffusive_inertia(
    const Parameters::CahnHilliard<dim> &param)
  {
    return is_abels_model(param);
  }

  /**
   * Set to true to make the Stepien momentum equation carry the Abels
   * capillary force phi grad(mu) instead of its own (dpr - mu) grad(phi).
   * Used to reduce the Stepien model to Abels one term at a time: the two
   * forms differ by the pure gradient grad(phi mu) once dpr = 0, so they are
   * equivalent up to the pressure shift p -> p - phi mu.
   *
   * Note that the dpr grad(phi) contribution is dropped along with the
   * Stepien form, so this is only meaningful when p_ref,0 = p_ref,1.
   */
  inline constexpr bool use_abels_capillary_for_stepien = false;

  template <int dim>
  inline bool
  use_abels_capillary_phi_grad_mu(
    const Parameters::CahnHilliard<dim> &param)
  {
    return is_abels_model(param) || is_quasi_incompressible_model(param) ||
           (use_abels_capillary_for_stepien && is_stepien_model(param));
  }

  template <int dim>
  inline bool
  use_ding_horriche_capillary_mu_grad_phi(
    const Parameters::CahnHilliard<dim> &param)
  {
    return is_ding_horriche_model(param);
  }

  template <int dim>
  inline bool
  use_stepien_capillary_form(
    const Parameters::CahnHilliard<dim> &param)
  {
    return is_stepien_model(param);
  }

  /**
   * Ratio lambda / eta of the second viscosity to the dynamic viscosity in the
   * Stepien viscous stress
   *
   *   tau = eta(phi) (grad v + grad v^T) + lambda(phi) (div v) I.
   *
   * Unlike the Abels model, the Stepien velocity field is not solenoidal, so
   * the second term does contribute to the momentum equation. Stokes'
   * hypothesis (vanishing bulk viscosity kappa = lambda + 2/3 eta) gives
   * lambda = -2/3 eta; lambda therefore inherits the linear tracer dependence
   * of the dynamic viscosity, and dlambda/dphi = -2/3 deta/dphi.
   */
  inline constexpr double stokes_second_viscosity_ratio = -2. / 3.;

  template <int dim>
  inline const char *
  model_name(const Parameters::CahnHilliard<dim> &param)
  {
    if (is_abels_model(param))
      return "Abels";
    if (is_ding_horriche_model(param))
      return "Ding-Horriche";
    if (is_stepien_model(param))
      return "Stepien";
    if (is_quasi_incompressible_model(param))
      return "Quasi-incompressible CHNS";

    DEAL_II_ASSERT_UNREACHABLE();
  }

  template <int dim>
  inline double
  sigma_tilde_from_surface_tension(
    const Parameters::CahnHilliard<dim> &param)
  {
    return 3. / (2. * std::sqrt(2.)) * param.surface_tension;
  }

  /** Constant amplitude: no quadrature maximum or state-dependent scaling. */
  template <int dim>
  inline double
  profile_correction_amplitude(const Parameters::CahnHilliard<dim> &param)
  {
    if (!param.enable_profile_correction)
      return 0.;
    AssertThrow(is_quasi_incompressible_model(param),
                dealii::ExcMessage("Profile correction is only implemented for "
                                   "CHNS model = quasi_incompressible."));
    AssertThrow(!param.enable_density_mobility_compensation,
                dealii::ExcMessage("Profile correction is not implemented with "
                                   "density mobility compensation."));
    AssertThrow(param.epsilon_interface > 0. && param.surface_tension > 0.,
                dealii::ExcMessage("Profile correction requires positive "
                                   "interface thickness and surface tension."));
    using Model = typename Parameters::CahnHilliard<dim>::MobilityModel;
    double reference = param.mobility;
    if (param.mobility_model == Model::degenerate_absolute)
      reference = param.degenerate_absolute_mobility_prefactor *
                  param.epsilon_interface * param.epsilon_interface;
    else if (param.mobility_model == Model::degenerate)
    {
      reference = param.profile_correction_reference_mobility;
      AssertThrow(reference > 0.,
                  dealii::ExcMessage("Profile correction with mobility model = "
                                     "degenerate requires a positive profile "
                                     "correction reference mobility."));
    }
    // Both constant and degenerate_quadratic use the mobility prefactor.
    return 2. * param.profile_correction_strength *
           (sigma_tilde_from_surface_tension(param) / param.epsilon_interface) *
           reference;
  }

  template <int dim>
  inline double
  potential_double_well_coefficient(
    const Parameters::CahnHilliard<dim> &param,
    const double                         sigma_tilde)
  {
    if (is_ding_horriche_model(param))
      return 1.;

    if (is_stepien_model(param))
      return sigma_tilde / param.epsilon_interface;

    return sigma_tilde / param.epsilon_interface;
  }

  template <int dim>
  inline double
  potential_gradient_coefficient(
    const Parameters::CahnHilliard<dim> &param,
    const double                         sigma_tilde)
  {
    if (is_ding_horriche_model(param))
      return param.epsilon_interface * param.epsilon_interface;

    if (is_stepien_model(param))
      return param.epsilon_interface * sigma_tilde;

    return sigma_tilde * param.epsilon_interface;
  }

  template <int dim>
  inline double
  ding_horriche_capillary_coefficient(
    const Parameters::CahnHilliard<dim> &param)
  {
    /*
     * Horriche's final CADYF system (3.19)-(3.22) uses the unscaled potential
     *
     *   mu_hat = phi^3 - phi - eps^2 Delta phi
     *
     * and the physical momentum force (gamma / eps) mu_hat grad(phi). For the
     * tanh profile used by FEZ, gamma must be the normalized coefficient
     *
     *   gamma = 3 / (2 sqrt(2)) sigma
     *
     * so that the diffuse-interface energy integrates to the physical surface
     * tension sigma. Using sigma / eps directly would give an effective
     * surface tension (2 sqrt(2) / 3) sigma and under-predict the
     * Young-Laplace jump.
     */
    const double sigma_tilde =
      3. / (2. * std::sqrt(2.)) * param.surface_tension;
    return sigma_tilde / param.epsilon_interface;
  }

  template <int dim>
  inline double
  stepien_capillary_coefficient(
    const Parameters::CahnHilliard<dim> & /*param*/)
  {
    // The Stepien momentum capillary force is muhat * grad(phi). Since the
    // potential unknown mu already stores muhat (same scaling as the potential
    // equation), the force prefactor is exactly 1.
    return 1.;
  }

  /**
   * Simply return the passed phase marker
   */
  inline double tracer_identity(const double phase_marker)
  {
    return phase_marker;
  }

  /**
   * Apply a limiter to the phase tracer : res = max(-1, min(1, phi))
   */
  inline double tracer_limiter(const double phase_marker)
  {
    return std::max(-1., std::min(1., phase_marker));
  }

  /**
   * Derivative of the optional clipping to [-1,1]. At the two corners,
   * choose the derivative from inside the interval for Newton's method.
   */
  inline double tracer_limiter_derivative(const double phase_marker,
                                          const bool   enabled)
  {
    return enabled && (phase_marker < -1. || phase_marker > 1.) ? 0. : 1.;
  }

  /**
   * Return a pointer to the limiter function used for the phase field tracer
   */
  using TracerLimiterFunction = double (*)(double);

  template <int dim>
  TracerLimiterFunction
  get_limiter_function(const Parameters::CahnHilliard<dim> param)
  {
    if (param.with_tracer_limiter || is_quasi_incompressible_model(param))
      return &tracer_limiter;
    else
      return &tracer_identity;
  }

  /**
   * Return a pointer to the tracer limiter function used for the degenerate
   * mobility calcul
   */
  using MobilityTracerLimiterFunction = double (*)(double);

  template <int dim>
  MobilityTracerLimiterFunction
  get_mobility_limiter_function(const Parameters::CahnHilliard<dim> param)
  {
    if (param.mobility_tracer_limiter)
      return &tracer_limiter;
    else
      return &tracer_identity;
  }

  /**
   * Apply linear mixing from value A (when phase marker = 1) to value B (phase
   * marker = -1).
   */
  inline double linear_mixing(const double phase_marker,
                              const double val_a,
                              const double val_b)
  {
    return 0.5 * ((val_a - val_b) * phase_marker + (val_a + val_b));
  }

  /**
   * Derivative w.r.t. the tracer of the linear mixing function
   */
  inline double linear_mixing_derivative(const double /*phase_marker*/,
                                         const double val_a,
                                         const double val_b)
  {
    return 0.5 * (val_a - val_b);
  }

  /**
   * Constant mobility
   */
  template <int dim>
  inline double constant_mobility(const Parameters::CahnHilliard<dim> &param,
                                  const double /*phase_marker*/)
  {
    return param.mobility;
  }

  /**
   * Degenerate mobility, according to the formula in the .param
   */
  template <int dim>
  inline double degenerate_mobility(const Parameters::CahnHilliard<dim> &param,
                                    const double phase_marker)
  {
    dealii::Point<dim> p;
    p[0] = phase_marker;
    return param.degenerate_mobility->value(p);
  }

  /**
   * Choice between constant and degenerate mobility
   */
  template <int dim>
  using MobilityFunction = double (*)(const Parameters::CahnHilliard<dim> &,
                                      double);

  template <int dim>
  MobilityFunction<dim>
  get_mobility_function(const Parameters::CahnHilliard<dim> &param)
  {
    if (param.mobility_model ==
        Parameters::CahnHilliard<dim>::MobilityModel::constant)
      return &constant_mobility<dim>;
    else
      return &degenerate_mobility<dim>;
  }

  template <int dim>
  inline double
  constant_mobility_derivative(const Parameters::CahnHilliard<dim> & /*param*/,
                               const double /*phase_marker*/)
  {
    return 0.;
  }

  template <int dim>
  inline double
  degenerate_mobility_derivative(const Parameters::CahnHilliard<dim> &param,
                                 const double phase_marker)
  {
    dealii::Point<dim> p;
    p[0] = phase_marker;
    return param.degenerate_mobility->gradient(p)[0];
  }

  template <int dim>
  MobilityFunction<dim>
  get_mobility_derivative_function(const Parameters::CahnHilliard<dim> &param)
  {
    if (param.mobility_model ==
        Parameters::CahnHilliard<dim>::MobilityModel::constant)
      return &constant_mobility_derivative<dim>;
    else
      return &degenerate_mobility_derivative<dim>;
  }

  template <int dim>
  inline double constant_mobility_second_derivative(
    const Parameters::CahnHilliard<dim> & /*param*/,
    const double /*phase_marker*/)
  {
    return 0.;
  }

  template <int dim>
  inline double degenerate_mobility_second_derivative(
    const Parameters::CahnHilliard<dim> &param,
    const double                         phase_marker)
  {
    dealii::Point<dim> p;
    p[0] = phase_marker;
    return param.degenerate_mobility->hessian(p)[0][0];
  }

  template <int dim>
  MobilityFunction<dim> get_mobility_second_derivative_function(
    const Parameters::CahnHilliard<dim> &param)
  {
    if (param.mobility_model ==
        Parameters::CahnHilliard<dim>::MobilityModel::constant)
      return &constant_mobility_second_derivative<dim>;
    else
      return &degenerate_mobility_second_derivative<dim>;
  }

  struct MobilityEvaluation
  {
    double value;
    double first_derivative;
    double second_derivative;
  };

  template <int dim>
  inline double
  bdf_density_derivative(const Parameters::CahnHilliard<dim>    &param,
                         const double                            rho0,
                         const double                            rho1,
                         const double                            rho,
                         const std::vector<double>              &coefficients,
                         const std::vector<std::vector<double>> &previous_phi,
                         const unsigned int                      q)
  {
    double     result  = coefficients[0] * rho;
    const auto limiter = get_limiter_function(param);
    for (unsigned int i = 1; i < coefficients.size(); ++i)
      result += coefficients[i] *
                linear_mixing(limiter(previous_phi[i - 1][q]), rho0, rho1);
    return result;
  }

  /**
   * Evaluate the base mobility and its derivatives with respect to the raw
   * phase field, then optionally apply Stepien's density compensation.
   * The density uses the same property limiter as the phase equation; the
   * base mobility uses its own limiter. Both chain rules are included.
   */
  template <int dim>
  inline MobilityEvaluation
  evaluate_mobility(const Parameters::CahnHilliard<dim> &param,
                    const double                         phase_marker,
                    const double                         density0,
                    const double                         density1)
  {
    const double mobility_phi =
      get_mobility_limiter_function(param)(phase_marker);
    const double mobility_phi_derivative =
      tracer_limiter_derivative(phase_marker, param.mobility_tracer_limiter);
    MobilityEvaluation result{};

    if (param.mobility_model ==
        Parameters::CahnHilliard<dim>::MobilityModel::degenerate_absolute)
    {
      // Rising-bubble law of Brunk & ten Eikelder, section 4.3:
      // m_A = c eps^2 |1-phi^2| (c=0.1 by default). FEZ diffuses xi_F rather
      // than w_A = k xi_F, hence M_F = k^2 m_A, k = 2 rho0 rho1/(rho0+rho1).
      AssertThrow(
        density0 > 0. && density1 > 0.,
        dealii::ExcMessage(
          "degenerate_absolute mobility requires positive densities."));
      const double k = 2. * density0 * density1 / (density0 + density1);
      const double conversion =
        is_quasi_incompressible_model(param) ? 1. : k * k;
      const double amplitude =
        conversion * param.degenerate_absolute_mobility_prefactor *
        param.epsilon_interface * param.epsilon_interface;
      const double well = 1. - mobility_phi * mobility_phi;
      // At the two kinks choose zero from the Clarke generalized derivative.
      const double sign = well > 0. ? 1. : (well < 0. ? -1. : 0.);
      result            = {amplitude * std::abs(well),
                           -2. * amplitude * mobility_phi * sign * mobility_phi_derivative,
                           -2. * amplitude * sign * mobility_phi_derivative *
                             mobility_phi_derivative};
    }

    else
      result = {get_mobility_function(param)(param, mobility_phi),
                get_mobility_derivative_function(param)(param, mobility_phi) *
                  mobility_phi_derivative,
                get_mobility_second_derivative_function(param)(param,
                                                               mobility_phi) *
                  mobility_phi_derivative * mobility_phi_derivative};

    if (!param.enable_density_mobility_compensation)
      return result;

    const double density_phi = get_limiter_function(param)(phase_marker);
    const double rho         = linear_mixing(density_phi, density0, density1);
    AssertThrow(density0 > 0. && density1 > 0. && rho > 0.,
                dealii::ExcMessage(
                  "Density mobility compensation requires strictly positive "
                  "fluid and mixture densities. Check the phase field or "
                  "enable the tracer limiter."));
    const double rho_derivative =
      linear_mixing_derivative(density_phi, density0, density1) *
      tracer_limiter_derivative(phase_marker, param.with_tracer_limiter);
    const double density_ratio                  = density0 * density1 / rho;
    const double factor                         = density_ratio * density_ratio;
    const double logarithmic_density_derivative = rho_derivative / rho;

    // For F=(rho0*rho1/rho)^2: F'=-2*(rho'/rho)*F and
    // F''=6*(rho'/rho)^2*F away from limiter corners.
    const double base_value      = result.value;
    const double base_derivative = result.first_derivative;
    result.value *= factor;
    result.first_derivative =
      factor *
      (base_derivative - 2. * logarithmic_density_derivative * base_value);
    result.second_derivative =
      factor * (result.second_derivative -
                4. * logarithmic_density_derivative * base_derivative +
                6. * logarithmic_density_derivative *
                  logarithmic_density_derivative * base_value);
    return result;
  }

} // namespace CahnHilliard

#endif
