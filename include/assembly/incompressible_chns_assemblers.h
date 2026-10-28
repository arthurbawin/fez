#ifndef INCOMPRESSIBLE_CHNS_ASSEMBLERS_H
#define INCOMPRESSIBLE_CHNS_ASSEMBLERS_H

#include <assembly/assembler.h>
#include <boundary_conditions.h>
#include <cahn_hilliard.h>
#include <components_ordering.h>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/table.h>
#include <deal.II/base/tensor.h>
#include <parameter_reader.h>

#include <cmath>
#include <type_traits>
#include <vector>

namespace Assembly
{
  /**
   * A namespace gathering the assembly routines for the (quasi-)incompressible
   * Cahn-Hilliard Navier-Stokes models.
   */
  namespace IncompressibleCHNS
  {
    /**
     * Temporal discretizations of the quasi-incompressible CHNS model.
     * BDF1 retains the one-step energy identity; BDF2 differentiates
     * sqrt(positive_density)*velocity and uses the implicit cubic potential.
     * Only BDF1 has a proved unconditional total-energy inequality here.
     * See examples/incompressible_chns/quasi_incompressible/equations.md.
     */
    namespace EnergyStable
    {
      using namespace dealii;

      /** A state, test function, or Newton direction at one quadrature point.
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

      // Exact divided difference of F(s)=(s*s-1)^2/4, with its continuous
      // extension at a=b. This polynomial also avoids cancellation for a~b.
      inline double averaged_potential_derivative(const double a,
                                                  const double b)
      {
        return 0.25 * (a + b) * (a * a + b * b - 2.);
      }

      inline double averaged_potential_second_derivative(const double a,
                                                         const double b)
      {
        return 0.25 * (3. * a * a + 2. * a * b + b * b - 2.);
      }

      template <int dim>
      struct PointData
      {
        double density, density_derivative;
        double positive_density, positive_density_derivative,
          old_positive_density;
        double         viscosity, viscosity_derivative;
        double         mobility, mobility_derivative;
        double         alpha, c0, well_coefficient, gradient_coefficient;
        double         old_phase, phase_history;
        bool           second_order = false;
        Tensor<1, dim> old_velocity, weighted_velocity_history, body_force,
          source_velocity;
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
        const double derivative =
          CahnHilliard::tracer_limiter_derivative(phase, true);
        d.positive_density = CahnHilliard::linear_mixing(clipped, rho0, rho1);
        d.positive_density_derivative = d.density_derivative * derivative;
        d.old_positive_density =
          CahnHilliard::linear_mixing(CahnHilliard::tracer_limiter(old_phase),
                                      rho0,
                                      rho1);
        d.viscosity = CahnHilliard::linear_mixing(clipped, eta0, eta1);
        d.viscosity_derivative = 0.5 * (eta0 - eta1) * derivative;
        const auto mobility =
          CahnHilliard::evaluate_mobility(param, phase, rho0, rho1);
        AssertThrow(
          mobility.value >= 0.,
          ExcMessage("Quasi-incompressible CHNS requires non-negative mobility "
                     "at every quadrature point, including overshoots."));
        d.mobility            = mobility.value;
        d.mobility_derivative = mobility.first_derivative;
        d.alpha               = (rho1 - rho0) / (rho0 + rho1);
        d.c0                  = coefficients[0];
        // The third coefficient is zero during FEZ's BDF1 startup, even
        // though two history vectors are allocated for a BDF2 run.
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

      /** Residual R (the assembler stores -R in the right-hand side).
       * The pressure equation is minus (41), to match FEZ's Stokes sign.
       * Test fields and history are held fixed during Newton differentiation.
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
        r += (-div_v + d.source_pressure) * t.pressure -
             d.alpha * (flux * t.pressure_gradient);
        r += (d.c0 * s.phase + d.phase_history + d.source_phase) * t.phase -
             s.phase * (v * t.phase_gradient) + flux * t.phase_gradient;
        r +=
          (s.potential - d.well_coefficient * potential_derivative(s.phase, d) +
           d.source_potential) *
            t.potential -
          d.gradient_coefficient * (s.phase_gradient * t.potential_gradient);
        return r;
      }

      /** Exact Gateaux derivative D R(s)[z], including both appearances of v
       * in the convective mass flux rho(phi)*v, and M'(phi) in both flux rows.
       * At clipping/absolute-value corners the constitutive helpers specify a
       * generalized derivative; classical finite differences apply away from
       * them.
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
        r -= div_dv * t.pressure + d.alpha * (dflux * t.pressure_gradient);
        r += d.c0 * z.phase * t.phase -
             (z.phase * v + s.phase * dv) * t.phase_gradient +
             dflux * t.phase_gradient;
        r +=
          (z.potential - d.well_coefficient *
                           potential_second_derivative(s.phase, d) * z.phase) *
            t.potential -
          d.gradient_coefficient * (z.phase_gradient * t.potential_gradient);
        return r;
      }

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
          auto &matrix = copy.local_matrix();
          for (unsigned int q = 0; q < sd.n_q_points; ++q)
          {
            const auto              s = state(sd, q);
            const auto              d = data(sd, q);
            std::vector<Field<dim>> shapes(sd.dofs_per_cell);
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

    /**
     * These flags are used to specify which terms to assemble in addition to
     * the base incompressible CHNS system.
     */
    enum AssemblyFlags : unsigned int
    {
      /**
       * Assemble the incompressible CHNS system without any stabilization.
       */
      chns = 0,

      /**
       * Add SUPG/PSPG stabilization forms for the Navier-Stokes equations.
       */
      stabilization = 1 << 0,

      /**
       * Add SUPG stabilization form for the phase tracer equation.
       */
      tracer_stabilization = 1 << 1,

      /**
       * Account for moving mesh contributions (ALE).
       */
      moving_mesh = 1 << 2,

      /**
       * Assemble the Stepien quasi-incompressible model instead of the default
       * Abels model. The two models share the potential (chemical potential)
       * equation but differ in the momentum, continuity and phase equations;
       * see the volume assembler for the corresponding weak forms.
       */
      stepien = 1 << 3
    };

    /**
     * Create the volume and relevant boundary assemblers, and store them as
     * unique pointers in @p assemblers.
     */
    template <int dim,
              typename ScratchData,
              typename CopyData,
              bool with_moving_mesh>
    void setup_assemblers(
      const ParameterReader<dim>         &param,
      const ComponentOrdering            &ordering,
      const Table<2, DoFTools::Coupling> &coupling_table,
      std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
        &assemblers);

    /**
     * Abstract base class for the incompressible CHNS forms.
     */
    template <typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags>
    class Base : public AssemblerBase<ScratchData, CopyData>
    {
    public:
      /**
       * Constructor
       */
      Base(const ComponentOrdering &ordering)
        : ordering(ordering)
      {}

    public:
      static constexpr bool with_stabilization =
        (assembly_flags & stabilization) != 0;
      static constexpr bool with_tracer_stabilization =
        (assembly_flags & tracer_stabilization) != 0;
      static constexpr bool with_moving_mesh =
        (assembly_flags & moving_mesh) != 0;
      static constexpr bool with_stepien = (assembly_flags & stepien) != 0;

      static_assert(!(with_stepien && with_moving_mesh),
                    "The Stepien forms are only implemented on a fixed mesh.");

      const ComponentOrdering &ordering;
    };

    /**
     * Assembler for the incompressible CHNS system in the volume.
     *
     * TODO: Add expression of the weak form.
     */
    template <int dim,
              typename ScratchData,
              typename CopyData,
              unsigned int assembly_flags = chns>
    class VolumeAssembler : public Base<ScratchData, CopyData, assembly_flags>
    {
      using BaseType = Base<ScratchData, CopyData, assembly_flags>;

    public:
      VolumeAssembler(const ComponentOrdering            &ordering,
                      const Table<2, DoFTools::Coupling> &coupling_table)
        : Base<ScratchData, CopyData, assembly_flags>(ordering)
        , coupling_table(coupling_table)
      {}

      /**
       * Assemble local matrix.
       */
      virtual void assemble_matrix(const ScratchData &scratch_data,
                                   CopyData          &copy_data) const override;

      /**
       * Assemble local right-hand side vector.
       */
      virtual void assemble_rhs(const ScratchData &scratch_data,
                                CopyData          &copy_data) const override;

    public:
      const Table<2, DoFTools::Coupling> &coupling_table;
    };
  } // namespace IncompressibleCHNS
} // namespace Assembly

/* ---------------- Template functions ----------------- */

namespace Assembly
{
  namespace IncompressibleCHNS
  {
    namespace internal
    {
      /**
       * Emplace the volume assembler matching the runtime SUPG switches
       * @p supg and @p tracer_supg, on top of the compile-time @p model_flags
       * which carry the CHNS model and the moving-mesh bit.
       *
       * Keeping the model and mesh bits as a template parameter avoids
       * spelling out one branch per combination of the four flags.
       */
      template <int dim,
                typename ScratchData,
                typename CopyData,
                unsigned int model_flags,
                bool         allow_tracer_stabilization = true>
      void emplace_volume_assembler(
        const ComponentOrdering            &ordering,
        const Table<2, DoFTools::Coupling> &coupling_table,
        const bool                          supg,
        const bool                          tracer_supg,
        std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
          &assemblers)
      {
        const auto emplace = [&](auto flags) {
          assemblers.emplace_back(
            std::make_unique<VolumeAssembler<dim,
                                             ScratchData,
                                             CopyData,
                                             decltype(flags)::value>>(
              ordering, coupling_table));
        };

        if constexpr (allow_tracer_stabilization)
        {
          if (supg && tracer_supg)
            emplace(std::integral_constant<unsigned int,
                                           model_flags | stabilization |
                                             tracer_stabilization>{});
          else if (tracer_supg)
            emplace(
              std::integral_constant<unsigned int,
                                     model_flags | tracer_stabilization>{});
          else if (supg)
            emplace(std::integral_constant<unsigned int,
                                           model_flags | stabilization>{});
          else
            emplace(std::integral_constant<unsigned int, model_flags>{});
        }
        else
        {
          // The caller guarantees that tracer_supg is false here, so the
          // corresponding assemblers are never instantiated.
          if (supg)
            emplace(std::integral_constant<unsigned int,
                                           model_flags | stabilization>{});
          else
            emplace(std::integral_constant<unsigned int, model_flags>{});
        }
      }
    } // namespace internal

    template <int dim,
              typename ScratchData,
              typename CopyData,
              bool with_moving_mesh>
    void setup_assemblers(
      const ParameterReader<dim>         &param,
      const ComponentOrdering            &ordering,
      const Table<2, DoFTools::Coupling> &coupling_table,
      std::vector<std::unique_ptr<AssemblerBase<ScratchData, CopyData>>>
        &assemblers)
    {
      // Perform some static checks where possible:
      static_assert(with_moving_mesh == ScratchData::enable_pseudo_solid,
                    "To enable moving_mesh computations in the CHNS "
                    "assemblers, the provided ScratchData should be "
                    "initialized with a pseudo-solid update flag.");

      if (CahnHilliard::is_quasi_incompressible_model(param.cahn_hilliard))
      {
        AssertThrow(!with_moving_mesh && !param.finite_elements.stabilization &&
                      !param.stabilization.enable_supg &&
                      !param.stabilization.enable_tracer_supg,
                    ExcMessage("The energy-stable quasi-incompressible CHNS "
                               "assembler requires a fixed mesh without "
                               "additional stabilization."));
        assemblers.emplace_back(
          std::make_unique<
            EnergyStable::VolumeAssembler<dim, ScratchData, CopyData>>(param));
        return;
      }

      using namespace BoundaryConditions;

      const bool supg        = param.stabilization.enable_supg;
      const bool tracer_supg = param.stabilization.enable_tracer_supg;
      constexpr unsigned int moving_mesh_flag =
        with_moving_mesh ? moving_mesh : chns;

      if constexpr (with_moving_mesh)
        AssertThrow(
          !(supg || tracer_supg),
          ExcMessage(
            "CHNS stabilization on a moving mesh is not implemented yet."));

      /**
       * The volume assembler reads the mobility as a single scalar
       * (ScratchData::mobility) rather than per quadrature point, so a
       * degenerate mobility would silently be evaluated as the constant one.
       */
      AssertThrow(
        param.cahn_hilliard.mobility_model ==
            Parameters::CahnHilliard<dim>::MobilityModel::constant &&
          !param.cahn_hilliard.enable_density_mobility_compensation,
        ExcMessage("The \"Stabilization\" subsection is only implemented for a "
                   "constant mobility without density mobility compensation. "
                   "Disable enable supg and enable tracer supg to use a "
                   "variable mobility."));

      // Assign the volume assembler
      if (CahnHilliard::is_stepien_model(param.cahn_hilliard))
      {
        // The Stepien forms are written for a fixed mesh only, so the ALE
        // instantiations are never requested.
        if constexpr (with_moving_mesh)
          AssertThrow(false, ExcMessage("Stepien ALE is not implemented."));
        else
          internal::emplace_volume_assembler<dim, ScratchData, CopyData,
                                             stepien>(
            ordering, coupling_table, supg, tracer_supg, assemblers);
      }
      else
        internal::emplace_volume_assembler<dim,
                                           ScratchData,
                                           CopyData,
                                           moving_mesh_flag>(
          ordering, coupling_table, supg, tracer_supg, assemblers);

      // Assign the relevant boundary assemblers
      // ...
    }
  } // namespace IncompressibleCHNS
} // namespace Assembly

#endif
