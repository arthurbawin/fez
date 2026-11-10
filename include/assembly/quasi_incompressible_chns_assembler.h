#ifndef QUASI_INCOMPRESSIBLE_CHNS_ASSEMBLER_H
#define QUASI_INCOMPRESSIBLE_CHNS_ASSEMBLER_H

#include <assembly/assembler.h>
#include <cahn_hilliard.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/tensor.h>
#include <parameter_reader.h>

#include <cmath>
#include <vector>

namespace Assembly
{
  namespace IncompressibleCHNS
  {
    /**
     * Energy-stable Galerkin scheme for the quasi-incompressible
     * Cahn-Hilliard Navier-Stokes model with mass-averaged velocity v.
     *
     * Notation: phase 0 (phi = 1) has density rho_1 and viscosity eta_1,
     * phase 1 (phi = -1) has rho_2 and eta_2, and
     *
     *   rho(phi)  = (1+phi)/2 rho_1 + (1-phi)/2 rho_2   (affine density),
     *   rho~, eta~: the same laws evaluated at clip(phi, -1, 1),
     *   alpha     = (rho_2 - rho_1) / (rho_1 + rho_2),
     *   sigma~    = 3 sigma / (2 sqrt(2)),
     *   J         = M(phi) (grad mu + alpha grad p)     (diffusive flux).
     *
     * With BDF1, the unknowns (v, p, phi, mu)^{n+1} satisfy, for all test
     * functions (z, q, psi, chi):
     *
     *   momentum:
     *     (rho~(phi^n) (v - v^n)/dt + (rho~(phi) - rho~(phi^n))/(2 dt) v, z)
     *     + 1/2 (rho(phi) (v.grad) v, z) - 1/2 (rho(phi) (v.grad) z, v)
     *     + (eta~(phi) (grad v + grad v^T), grad z)
     *     - 2/d (eta~(phi) div v, div z)
     *     - (p, div z) + (phi grad mu, z) - (rho(phi) g, z) = 0,
     *
     *   continuity:
     *     -(div v, q) - alpha (J, grad q) = 0,
     *
     *   phase:
     *     ((phi - phi^n)/dt, psi) - (phi v, grad psi) + (J, grad psi) = 0,
     *
     *   potential:
     *     (mu, chi) - sigma~/eps (F'(phi, phi^n), chi)
     *     - sigma~ eps (grad phi, grad chi) = 0,
     *
     * where every coefficient without superscript is evaluated at t^{n+1} and
     * F'(a, b) = (a + b)(a^2 + b^2 - 2)/4 is the exact divided difference of
     * F(s) = (s^2 - 1)^2/4. Testing with (v, p, mu, phi - phi^n) gives the
     * discrete energy identity documented in
     * examples/incompressible_chns/quasi_incompressible/equations.md.
     *
     * The convective terms cancel for z = v, and the continuity row is the
     * opposite of the strong equation, to keep the usual symmetric Stokes
     * pressure-velocity block.
     *
     * BDF2 variant (no proved energy inequality): the inertia becomes
     * c0 rho~ v + sqrt(rho~) sum_{i>=1} c_i sqrt(rho~^{n+1-i}) v^{n+1-i}, i.e.
     * the BDF2 derivative of sqrt(rho~) v multiplied by sqrt(rho~), and the
     * potential uses the implicit F'(phi) = phi^3 - phi.
     */
    namespace EnergyStable
    {
      using namespace dealii;

      /**
       * A state, test function, or Newton direction at one quadrature point.
       * Velocity gradients follow deal.II: (grad v)_ij = d_j v_i.
       */
      template <int dim>
      struct Field
      {
        Tensor<1, dim> velocity;
        Tensor<2, dim> velocity_gradient;
        double         pressure = 0.;
        Tensor<1, dim> pressure_gradient;
        double         phase = 0.;
        Tensor<1, dim> phase_gradient;
        double         potential = 0.;
        Tensor<1, dim> potential_gradient;
      };

      /**
       * Exact divided difference (F(a) - F(b)) / (a - b) of
       * F(s) = (s^2 - 1)^2 / 4, extended continuously to a = b. The polynomial
       * form also avoids cancellation when a ~ b.
       */
      inline double averaged_potential_derivative(const double a,
                                                  const double b)
      {
        return 0.25 * (a + b) * (a * a + b * b - 2.);
      }

      /**
       * Derivative of averaged_potential_derivative() with respect to a.
       */
      inline double averaged_potential_second_derivative(const double a,
                                                         const double b)
      {
        return 0.25 * (3. * a * a + 2. * a * b + b * b - 2.);
      }

      /**
       * Material coefficients and frozen history at one quadrature point.
       */
      template <int dim>
      struct PointData
      {
        // Affine density rho(phi) and its derivative.
        double density, density_derivative;
        // Positive extension rho~(phi), its derivative, and rho~(phi^n).
        double positive_density, positive_density_derivative,
          old_positive_density;
        // Positive viscosity extension eta~(phi) and its derivative.
        double viscosity, viscosity_derivative;
        double mobility, mobility_derivative;
        double alpha;
        // Leading BDF coefficient (1/dt for BDF1).
        double c0;
        double well_coefficient, gradient_coefficient;
        double old_phase;
        // sum_{i>=1} c_i phi^{n+1-i}.
        double phase_history;
        bool   second_order = false;
        // BDF2 only: sum_{i>=1} c_i sqrt(rho~^{n+1-i}) v^{n+1-i}.
        Tensor<1, dim> weighted_velocity_history;
        Tensor<1, dim> old_velocity, body_force, source_velocity;
        double source_pressure = 0., source_phase = 0., source_potential = 0.;
      };

      template <int dim>
      PointData<dim> make_point_data(const Parameters::CahnHilliard<dim> &param,
                                     const double                         rho0,
                                     const double                         rho1,
                                     const double                         eta0,
                                     const double                         eta1,
                                     const std::vector<double> &coefficients,
                                     const double               phase,
                                     const double               old_phase,
                                     const Tensor<1, dim>      &old_velocity,
                                     const Tensor<1, dim>      &body_force,
                                     const double          older_phase    = 0.,
                                     const Tensor<1, dim> &older_velocity = {})
      {
        AssertThrow(
          coefficients.size() == 2 || coefficients.size() == 3,
          ExcMessage(
            "Quasi-incompressible CHNS expects BDF1/BDF2 coefficients."));
        PointData<dim> d;
        d.density            = CahnHilliard::linear_mixing(phase, rho0, rho1);
        d.density_derivative = 0.5 * (rho0 - rho1);
        const double clipped = CahnHilliard::tracer_limiter(phase);
        const double clip_derivative =
          CahnHilliard::tracer_limiter_derivative(phase, true);
        d.positive_density = CahnHilliard::linear_mixing(clipped, rho0, rho1);
        d.positive_density_derivative = d.density_derivative * clip_derivative;
        d.old_positive_density =
          CahnHilliard::linear_mixing(CahnHilliard::tracer_limiter(old_phase),
                                      rho0,
                                      rho1);
        d.viscosity = CahnHilliard::linear_mixing(clipped, eta0, eta1);
        d.viscosity_derivative = 0.5 * (eta0 - eta1) * clip_derivative;
        const auto mobility    = CahnHilliard::evaluate_mobility(param, phase);
        AssertThrow(
          mobility.value >= 0.,
          ExcMessage("Quasi-incompressible CHNS requires non-negative mobility "
                     "at every quadrature point, including overshoots."));
        d.mobility            = mobility.value;
        d.mobility_derivative = mobility.first_derivative;
        d.alpha               = (rho1 - rho0) / (rho0 + rho1);
        d.c0                  = coefficients[0];
        // The third coefficient is zero during the BDF1 startup of a BDF2
        // run, even though two history vectors are allocated.
        d.second_order  = coefficients.size() == 3 && coefficients[2] != 0.;
        d.phase_history = coefficients[1] * old_phase;
        if (d.second_order)
        {
          const double older_density = CahnHilliard::linear_mixing(
            CahnHilliard::tracer_limiter(older_phase), rho0, rho1);
          d.phase_history += coefficients[2] * older_phase;
          d.weighted_velocity_history =
            coefficients[1] * std::sqrt(d.old_positive_density) * old_velocity +
            coefficients[2] * std::sqrt(older_density) * older_velocity;
        }
        const double sigma =
          CahnHilliard::sigma_tilde_from_surface_tension(param);
        d.well_coefficient     = sigma / param.epsilon_interface;
        d.gradient_coefficient = sigma * param.epsilon_interface;
        d.old_phase            = old_phase;
        d.old_velocity         = old_velocity;
        d.body_force           = body_force;
        return d;
      }

      /**
       * Discrete inertia. BDF1:
       *   rho~^n (v - v^n)/dt + (rho~ - rho~^n)/(2 dt) v.
       */
      template <int dim>
      Tensor<1, dim> inertia(const Tensor<1, dim> &velocity,
                             const PointData<dim> &d)
      {
        if (d.second_order)
          return d.c0 * d.positive_density * velocity +
                 std::sqrt(d.positive_density) * d.weighted_velocity_history;
        return d.c0 *
               (0.5 * (d.positive_density - d.old_positive_density) * velocity +
                d.old_positive_density * (velocity - d.old_velocity));
      }

      template <int dim>
      double potential_derivative(const double phase, const PointData<dim> &d)
      {
        return d.second_order ?
                 phase * (phase * phase - 1.) :
                 averaged_potential_derivative(phase, d.old_phase);
      }

      template <int dim>
      double potential_second_derivative(const double          phase,
                                         const PointData<dim> &d)
      {
        return d.second_order ?
                 3. * phase * phase - 1. :
                 averaged_potential_second_derivative(phase, d.old_phase);
      }

      /**
       * Residual R(s; t) of the four equations at state @p s, tested with
       * @p t. The assembler stores -R in the right-hand side. The test fields
       * and history are held fixed during Newton differentiation.
       */
      template <int dim>
      double residual(const Field<dim>     &s,
                      const Field<dim>     &t,
                      const PointData<dim> &d)
      {
        const auto  &v     = s.velocity;
        const double div_v = trace(s.velocity_gradient);
        const double div_w = trace(t.velocity_gradient);
        const auto   flux =
          d.mobility * (s.potential_gradient + d.alpha * s.pressure_gradient);

        // Momentum
        double r = (inertia(v, d) - d.density * d.body_force +
                    s.phase * s.potential_gradient + d.source_velocity) *
                   t.velocity;
        r += 0.5 * d.density *
             ((s.velocity_gradient * v) * t.velocity -
              (t.velocity_gradient * v) * v);
        r += 2. * d.viscosity *
               scalar_product(symmetrize(s.velocity_gradient),
                              symmetrize(t.velocity_gradient)) -
             (2. / dim) * d.viscosity * div_v * div_w;
        r -= s.pressure * div_w;

        // Continuity
        r += (-div_v + d.source_pressure) * t.pressure -
             d.alpha * (flux * t.pressure_gradient);

        // Phase
        r += (d.c0 * s.phase + d.phase_history + d.source_phase) * t.phase -
             s.phase * (v * t.phase_gradient) + flux * t.phase_gradient;

        // Potential
        r +=
          (s.potential - d.well_coefficient * potential_derivative(s.phase, d) +
           d.source_potential) *
            t.potential -
          d.gradient_coefficient * (s.phase_gradient * t.potential_gradient);
        return r;
      }

      /**
       * Exact Gateaux derivative D R(s; t)[z], including both appearances of
       * v in the convective mass flux rho(phi) v, and M'(phi) in both flux
       * rows. At the clipping and absolute-value corners, the constitutive
       * helpers specify a generalized derivative; finite differences match
       * away from them.
       */
      template <int dim>
      double jacobian(const Field<dim>     &s,
                      const Field<dim>     &t,
                      const Field<dim>     &z,
                      const PointData<dim> &d)
      {
        const auto  &v      = s.velocity;
        const auto  &dv     = z.velocity;
        const double div_v  = trace(s.velocity_gradient);
        const double div_w  = trace(t.velocity_gradient);
        const double div_dv = trace(z.velocity_gradient);
        const double drho   = d.density_derivative * z.phase;
        const double deta   = d.viscosity_derivative * z.phase;
        const auto   dflux =
          d.mobility_derivative * z.phase *
            (s.potential_gradient + d.alpha * s.pressure_gradient) +
          d.mobility * (z.potential_gradient + d.alpha * z.pressure_gradient);
        const auto dinertia =
          d.second_order ?
            d.c0 * d.positive_density * dv +
              d.positive_density_derivative * z.phase *
                (d.c0 * v + d.weighted_velocity_history /
                              (2. * std::sqrt(d.positive_density))) :
            0.5 * d.c0 *
              ((d.positive_density + d.old_positive_density) * dv +
               d.positive_density_derivative * z.phase * v);

        // Momentum
        double r =
          (dinertia - drho * d.body_force + z.phase * s.potential_gradient +
           s.phase * z.potential_gradient) *
          t.velocity;
        r += 0.5 * drho *
             ((s.velocity_gradient * v) * t.velocity -
              (t.velocity_gradient * v) * v);
        r +=
          0.5 * d.density *
          ((z.velocity_gradient * v + s.velocity_gradient * dv) * t.velocity -
           (t.velocity_gradient * dv) * v - (t.velocity_gradient * v) * dv);
        r += 2. * deta *
               scalar_product(symmetrize(s.velocity_gradient),
                              symmetrize(t.velocity_gradient)) +
             2. * d.viscosity *
               scalar_product(symmetrize(z.velocity_gradient),
                              symmetrize(t.velocity_gradient)) -
             (2. / dim) * (deta * div_v + d.viscosity * div_dv) * div_w;
        r -= z.pressure * div_w;

        // Continuity
        r -= div_dv * t.pressure + d.alpha * (dflux * t.pressure_gradient);

        // Phase
        r += d.c0 * z.phase * t.phase -
             (z.phase * v + s.phase * dv) * t.phase_gradient +
             dflux * t.phase_gradient;

        // Potential
        r +=
          (z.potential - d.well_coefficient *
                           potential_second_derivative(s.phase, d) * z.phase) *
            t.potential -
          d.gradient_coefficient * (z.phase_gradient * t.potential_gradient);
        return r;
      }

      /**
       * Volume assembler of the energy-stable scheme, on a fixed mesh and
       * without stabilization.
       */
      template <int dim, typename ScratchData, typename CopyData>
      class VolumeAssembler : public AssemblerBase<ScratchData, CopyData>
      {
      public:
        explicit VolumeAssembler(const ParameterReader<dim> &param)
          : param(param)
        {}

        void assemble_rhs(const ScratchData &sd, CopyData &copy) const override
        {
          auto &rhs = copy.local_rhs();
          for (unsigned int q = 0; q < sd.n_q_points; ++q)
          {
            const auto s = state(sd, q);
            const auto d = data(sd, q);
            for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
              rhs(i) -= sd.JxW_moving[q] * residual(s, shape(sd, q, i), d);
          }
        }

        void assemble_matrix(const ScratchData &sd,
                             CopyData          &copy) const override
        {
          auto                   &matrix = copy.local_matrix();
          std::vector<Field<dim>> shapes(sd.dofs_per_cell);
          for (unsigned int q = 0; q < sd.n_q_points; ++q)
          {
            const auto s = state(sd, q);
            const auto d = data(sd, q);
            for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
              shapes[i] = shape(sd, q, i);
            for (unsigned int i = 0; i < sd.dofs_per_cell; ++i)
              for (unsigned int j = 0; j < sd.dofs_per_cell; ++j)
                matrix(i, j) +=
                  sd.JxW_moving[q] * jacobian(s, shapes[i], shapes[j], d);
          }
        }

      private:
        static Field<dim> state(const ScratchData &sd, const unsigned int q)
        {
          return {sd.present_velocity_values[q],
                  sd.present_velocity_gradients[q],
                  sd.present_pressure_values[q],
                  sd.present_pressure_gradients[q],
                  sd.tracer_values[q],
                  sd.tracer_gradients[q],
                  sd.potential_values[q],
                  sd.potential_gradients[q]};
        }

        static Field<dim>
        shape(const ScratchData &sd, const unsigned int q, const unsigned int i)
        {
          return {sd.phi_u[q][i],
                  sd.grad_phi_u[q][i],
                  sd.phi_p[q][i],
                  sd.grad_phi_p[q][i],
                  sd.shape_phi[q][i],
                  sd.grad_shape_phi[q][i],
                  sd.shape_mu[q][i],
                  sd.grad_shape_mu[q][i]};
        }

        PointData<dim> data(const ScratchData &sd, const unsigned int q) const
        {
          const auto &coefficients = sd.time_handler.get_bdf_coefficients();
          const bool  bdf2 = coefficients.size() == 3 && coefficients[2] != 0.;
          auto        d =
            make_point_data(param.cahn_hilliard,
                            sd.density0,
                            sd.density1,
                            sd.dynamic_viscosity0,
                            sd.dynamic_viscosity1,
                            coefficients,
                            sd.tracer_values[q],
                            sd.previous_tracer_values.at(0)[q],
                            sd.previous_velocity_values.at(0)[q],
                            sd.body_force,
                            bdf2 ? sd.previous_tracer_values.at(1)[q] : 0.,
                            bdf2 ? sd.previous_velocity_values.at(1)[q] :
                                   Tensor<1, dim>());
          d.source_velocity  = sd.source_term_velocity[q];
          d.source_pressure  = sd.source_term_pressure[q];
          d.source_phase     = sd.source_term_tracer[q];
          d.source_potential = sd.source_term_potential[q];
          return d;
        }

        const ParameterReader<dim> &param;
      };
    } // namespace EnergyStable
  }   // namespace IncompressibleCHNS
} // namespace Assembly

#endif
