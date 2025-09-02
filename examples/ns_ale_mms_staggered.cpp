
#include <Mesh.h>
#include <deal.II/base/function.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/timer.h>
#include <deal.II/lac/generic_linear_algebra.h>

namespace LA
{
#if defined(DEAL_II_WITH_PETSC) && !defined(DEAL_II_PETSC_WITH_COMPLEX) && \
  !(defined(DEAL_II_WITH_TRILINOS) && defined(FORCE_USE_OF_TRILINOS))
  using namespace dealii::LinearAlgebraPETSc;
#  define USE_PETSC_LA
#elif defined(DEAL_II_WITH_TRILINOS)
  using namespace dealii::LinearAlgebraTrilinos;
#else
#  error DEAL_II_WITH_PETSC or DEAL_II_WITH_TRILINOS required
#endif
} // namespace LA

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/index_set.h>
#include <deal.II/base/table_handler.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/grid_refinement.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>

#include <cmath>
#include <fstream>
#include <iostream>

// Fluid
#define VISCOSITY 0.1

// Pseudo-solid
#define LAMBDA_PS 1.
#define MU_PS 1.

// Enable dudt
#define WITH_TRANSIENT
// Enable (u dot grad) u
#define WITH_CONVECTION
// Enable ALE
#define WITH_ALE

namespace NS_MMS
{
  using namespace dealii;

  double G_fun(const double t)
  {
#if !defined(WITH_TRANSIENT)
    return 1.;
#else
    // return t;
    // return t*t;
    // return t*t*t;
    return -sin(2. * M_PI * t);
#endif
  }

  double Gdot_fun(const double t)
  {
#if !defined(WITH_TRANSIENT)
    return 0.;
#else
    // return 1.;
    // return 2.*t;
    // return 3.*t*t;
    return -2 * M_PI * cos(2. * M_PI * t);
#endif
  }

  double phi_fun(const double t)
  {
    return (t > 0) ? 4. * (t - tanh(t)) : 0.;
  }

  template <int dim>
  double u_fun(const double G, const double mu, const Point<dim> &p)
  {
    if constexpr (dim == 2)
      return G / (2. * mu) * (p[1] * (p[1] - 1.) + p[0]);
    else
      return G / (2. * mu) * (p[1] * (p[1] - 1.) + 2. * p[0] + p[2]);
  }

  template <int dim>
  double v_fun(const double G, const double mu, const Point<dim> &p)
  {
    if constexpr (dim == 2)
      return G / (2. * mu) * (p[0] * (p[0] - 1.) - p[1]);
    else
      return G / (2. * mu) * (p[2] * (p[2] - 1.) + p[0] - p[1]);
  }

  template <int dim>
  double w_fun(const double G, const double mu, const Point<dim> &p)
  {
    if constexpr (dim == 2)
      return G / (2. * mu) * (p[0] * (p[0] - 1.) - p[1]);
    else
      return G / (2. * mu) * (p[0] * (p[0] - 1.) + p[1] - p[2]);
  }

  template <int dim>
  double p_fun(const double G, const Point<dim> &p)
  {
    if constexpr (dim == 2)
      return G * (p[0] + p[1]);
    else
      return G * (p[0] + p[1] + p[2]);
  }

  template <int dim>
  double
  chi_fun(const double phi, const Point<dim> &p, const unsigned int component)
  {
    if constexpr (dim == 2)
      // return phi / 4. * p[0] * p[1] * (p[component] - 1.);
      return p[0] * p[1] * (p[component] - 1.);
    else
      return phi / 4. * p[0] * p[1] * p[2] * (p[component] - 1.);
  }

  template <int dim>
  class Solution : public Function<dim>
  {
  public:
    Solution(const double time, const unsigned int n_components)
      : Function<dim>(n_components, time)
    {}

    virtual double value(const Point<dim>  &p,
                         const unsigned int component = 0) const override
    {
      const double t = this->get_time();
      const double G = G_fun(t);

      // Used only for the pressure evaluation
      if (component == dim)
        return p_fun(G, p);
      else
      {
        throw std::runtime_error(
          "value is expected to be used for pressure constrain only");
      }

      return 0;
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t   = this->get_time();
      const double G   = G_fun(t);
      const double phi = phi_fun(t);
      const double mu  = VISCOSITY;

      if constexpr (dim == 2)
      {
        // Velocity
        values[0] = u_fun(G, mu, p);
        values[1] = v_fun(G, mu, p);
        // Pressure
        values[2] = p_fun(G, p);
        // Mesh position
        values[3] = chi_fun(phi, p, 0);
        values[4] = chi_fun(phi, p, 1);
      }
      else
      {
        // Velocity
        values[0] = u_fun(G, mu, p);
        values[1] = v_fun(G, mu, p);
        values[2] = w_fun(G, mu, p);
        // Pressure
        values[3] = p_fun(G, p);
        // Mesh position
        values[4] = chi_fun(phi, p, 0);
        values[5] = chi_fun(phi, p, 1);
        values[6] = chi_fun(phi, p, 2);
      }
    }

    virtual void
    vector_gradient(const Point<dim> & /*p*/,
                    std::vector<Tensor<1, dim>> & /*gradients*/) const override
    {}
  };

  template <int dim>
  class SourceTerm : public Function<dim>
  {
  public:
    SourceTerm(const double time, const unsigned int n_components)
      : Function<dim>(n_components, time)
    {}

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t        = this->get_time();
      const double G        = G_fun(t);
      const double Gdot     = Gdot_fun(t);
      const double phi      = phi_fun(t);
      const double x        = p[0];
      const double y        = p[1];
      const double h        = 1.;
      const double mu       = VISCOSITY;
      const double lambda_s = LAMBDA_PS;
      const double mu_s     = MU_PS;

      if constexpr (dim == 2)
      {
#if !defined(WITH_TRANSIENT) && !defined(WITH_CONVECTION)
        // Steady Stokes
        values[0] = 0.;
        values[1] = 0.;
#elif !defined(WITH_TRANSIENT) && defined(WITH_CONVECTION)
        // Steady NS
        values[0] = -(x - y * (h - y)) / (4 * mu * mu) -
                    ((y + x * (h - x)) * (h - 2 * y)) / (4 * mu * mu);
        values[1] = ((x - y * (h - y)) * (h - 2 * x)) / (4 * mu * mu) -
                    (y + x * (h - x)) / (4 * mu * mu);
#elif defined(WITH_TRANSIENT) && !defined(WITH_CONVECTION)
        // Unsteady Stokes
        values[0] = -((x - y * (h - y)) * Gdot) / (2 * mu);
        values[1] = ((y + x * (h - x)) * Gdot) / (2 * mu);
#elif defined(WITH_TRANSIENT) && defined(WITH_CONVECTION)
        // Unsteady NS
        values[0] = -((x - y * (h - y)) * Gdot) / (2 * mu) -
                    (G * G * (x - y * (h - y))) / (4 * mu * mu) -
                    (G * G * (y + x * (h - x)) * (h - 2 * y)) / (4 * mu * mu);
        values[1] = ((y + x * (h - x)) * Gdot) / (2 * mu) -
                    (G * G * (y + x * (h - x))) / (4 * mu * mu) +
                    (G * G * (x - y * (h - y)) * (h - 2 * x)) / (4 * mu * mu);
#endif

        // Pressure source term
        values[2] = 0.;

        // Pseudo-solid source term
        values[3] = - (phi / 4. * (lambda_s * (h - 4 * y) + mu_s * (h - 6 * y)));
        values[4] = - (phi / 4. * (lambda_s * (h - 4 * x) + mu_s * (h - 6 * x)));
      }
      else
      {
        const double z = p[2];

#if !defined(WITH_TRANSIENT) && !defined(WITH_CONVECTION)
        // Steady Stokes
        values[0] = 0.;
        values[1] = 0.;
        values[2] = 0.;
#elif !defined(WITH_TRANSIENT) && defined(WITH_CONVECTION)
        // Steady NS
        // values[0] = - (x - y*(h - y))/(4*mu*mu) - ((y + x*(h - x))*(h -
        // 2*y))/(4*mu*mu); values[1] =  ((x - y*(h - y))*(h - 2*x))/(4*mu*mu) -
        // (y + x*(h - x))/(4*mu*mu);
        DEAL_II_ASSERT_UNREACHABLE();
#elif defined(WITH_TRANSIENT) && !defined(WITH_CONVECTION)
        // Unsteady Stokes
        values[0] = -(Gdot * (2 * x + z - y * (h - y))) / (2 * mu);
        values[1] = (Gdot * (y - x + z * (h - z))) / (2 * mu);
        values[2] = (Gdot * (z - y + x * (h - x))) / (2 * mu);
#elif defined(WITH_TRANSIENT) && defined(WITH_CONVECTION)
        // Unsteady NS
        values[0] =
          (G * G * (z - y + x * (h - x))) / (4 * mu * mu) -
          (G * G * (2 * x + z - y * (h - y))) / (2 * mu * mu) -
          (Gdot * (2 * x + z - y * (h - y))) / (2 * mu) -
          (G * G * (h - 2 * y) * (y - x + z * (h - z))) / (4 * mu * mu);
        values[1] =
          (Gdot * (y - x + z * (h - z))) / (2 * mu) -
          (G * G * (y - x + z * (h - z))) / (4 * mu * mu) -
          (G * G * (2 * x + z - y * (h - y))) / (4 * mu * mu) -
          (G * G * (h - 2 * z) * (z - y + x * (h - x))) / (4 * mu * mu);
        values[2] =
          (G * G * (y - x + z * (h - z))) / (4 * mu * mu) -
          (G * G * (z - y + x * (h - x))) / (4 * mu * mu) +
          (Gdot * (z - y + x * (h - x))) / (2 * mu) +
          (G * G * (h - 2 * x) * (2 * x + z - y * (h - y))) / (4 * mu * mu);
#endif
        // Add source term for linear elasticity
        DEAL_II_ASSERT_UNREACHABLE();
      }
    }
  };

  template <int dim>
  class ScratchData
  {
  public:
    ScratchData(const FESystem<dim>       &fe,
                const Quadrature<dim>     &cell_quadrature,
                const Mapping<dim>        &mapping,
                const Quadrature<dim - 1> &face_quadrature,
                const unsigned int         dofs_per_cell,
                const std::vector<double> &bdfCoeffs)
      : fe_values(mapping,
                  fe,
                  cell_quadrature,
                  update_values | update_gradients | update_quadrature_points |
                    update_JxW_values)
      , fe_face_values(mapping,
                       fe,
                       face_quadrature,
                       update_values | update_gradients |
                         update_quadrature_points | update_JxW_values)
      , n_q_points(cell_quadrature.size())
      , n_faces_q_points(face_quadrature.size())
      , dofs_per_cell(dofs_per_cell)
      , bdfCoeffs(bdfCoeffs)
    {
      this->allocate();
    }

    void allocate();

    template <typename VectorType>
    void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
                const VectorType                   &current_solution,
                const std::vector<LA::MPI::Vector> &previous_solutions,
                const Function<dim>                &source_term_fun)
    {
      fe_values.reinit(cell);

      for (const unsigned int i : fe_values.dof_indices())
        components[i] = fe_values.get_fe().system_to_component_index(i).first;

      const FEValuesExtractors::Vector velocities(0);         // 0 -> dim-1
      const FEValuesExtractors::Scalar pressure(dim);         // dim -> dim
      const FEValuesExtractors::Vector displacement(dim + 1); // dim+1 -> 2*dim
      // const FEValuesExtractors::Vector lambda(2 * dim + 1); // 2*dim+1
      // -> 3*dim

      if constexpr (std::is_same<VectorType, LA::MPI::Vector>::value)
      {
        fe_values[velocities].get_function_values(current_solution,
                                                  present_velocity_values);
        fe_values[velocities].get_function_gradients(
          current_solution, present_velocity_gradients);
        fe_values[pressure].get_function_values(current_solution,
                                                present_pressure_values);
        fe_values[displacement].get_function_values(
          current_solution, present_displacement_values);
        fe_values[displacement].get_function_gradients(
          current_solution, present_displacement_gradients);
      }
      else if constexpr (std::is_same<VectorType, std::vector<double>>::value)
      {
        fe_values[velocities].get_function_values_from_local_dof_values(
          current_solution, present_velocity_values);
        fe_values[velocities].get_function_gradients_from_local_dof_values(
          current_solution, present_velocity_gradients);
        fe_values[pressure].get_function_values_from_local_dof_values(
          current_solution, present_pressure_values);
        fe_values[displacement].get_function_values_from_local_dof_values(
          current_solution, present_displacement_values);
        fe_values[displacement].get_function_gradients_from_local_dof_values(
          current_solution, present_displacement_gradients);
      }
      else
      {
        static_assert(false,
                      "reinit expects LA::MPI::Vector or std::vector<double>");
      }

      // Source term
      const auto &fe = fe_values.get_fe();
      source_term_fun.vector_value_list(fe_values.get_quadrature_points(),
                                        source_term_full);

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        for (int d = 0; d < dim; ++d)
        {
          const unsigned int comp    = fe.system_to_component_index(d).first;
          source_term_velocity[q][d] = source_term_full[q](comp);

          const unsigned int comp_disp =
            fe.system_to_component_index(dim + 1 + d).first;
          source_term_displacement[q][d] = source_term_full[q](comp_disp);
        }
      }

      for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      {
        fe_values[velocities].get_function_values(previous_solutions[i],
                                                  previous_velocity_values[i]);
        fe_values[displacement].get_function_values(
          previous_solutions[i], previous_displacement_values[i]);
      }

      // Current mesh velocity from displacement
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        present_mesh_velocity_values[q] = bdfCoeffs[0] *
        present_displacement_values[q]; for(unsigned int iBDF = 1; iBDF <
        bdfCoeffs.size(); ++iBDF)
        {
          present_mesh_velocity_values[q] += bdfCoeffs[iBDF] *
          previous_displacement_values[iBDF - 1][q];
        }
      }

      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        JxW[q] = fe_values.JxW(q);
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_u[q][k]         = fe_values[velocities].value(k, q);
          grad_phi_u[q][k]    = fe_values[velocities].gradient(k, q);
          div_phi_u[q][k]     = fe_values[velocities].divergence(k, q);
          phi_p[q][k]         = fe_values[pressure].value(k, q);
          phi_disp[q][k]      = fe_values[displacement].value(k, q);
          grad_phi_disp[q][k] = fe_values[displacement].gradient(k, q);
          div_phi_disp[q][k]  = fe_values[displacement].divergence(k, q);
        }
      }

      // // Reinit fe_face_values on face touching the cylinder, if any.
      // for (const auto face_no : cell->face_indices())
      // {
      //   const auto &face = cell->face(face_no);

      //   if (!(face->at_boundary() && face->boundary_id() == boundary_id))
      //     continue;

      //   fe_face_values.reinit(cell, face);

      //   fe_face_values[velocities].get_function_values(current_solution,
      //   present_face_velocity_values);
      //   fe_face_values[displacement].get_function_values(current_solution,
      //   present_face_displacement_values);
      //   fe_face_values[lambda].get_function_values(current_solution,
      //   present_face_lambda_values);

      //   for(unsigned int i = 0; i < previous_solutions.size(); ++i)
      //   {
      //     fe_face_values[displacement].get_function_values(previous_solutions[i],
      //     previous_face_displacement_values[i]);
      //   }

      //   for (unsigned int q = 0; q < n_faces_q_points; ++q)
      //   {
      //     face_JxW[q] = fe_face_values.JxW(q);

      //     for (unsigned int k = 0; k < dofs_per_cell; ++k)
      //     {
      //       phi_u_face[q][k] = fe_face_values[velocities].value(k, q);
      //       phi_x_face[q][k] = fe_face_values[displacement].value(k, q);
      //       phi_l_face[q][k] = fe_face_values[lambda].value(k, q);
      //     }

      //     // Face mesh velocity
      //     present_face_mesh_velocity_values[q] = bdfCoeffs[0] *
      //     present_face_displacement_values[q]; for(unsigned int iBDF = 1;
      //     iBDF < bdfCoeffs.size(); ++iBDF)
      //     {
      //       present_face_mesh_velocity_values[q] += bdfCoeffs[iBDF] *
      //       previous_face_displacement_values[iBDF - 1][q];
      //     }
      //   }
      // }
    }

  public:
    FEValues<dim>     fe_values;
    FEFaceValues<dim> fe_face_values;

    const unsigned int         n_q_points;
    const unsigned int         n_faces_q_points;
    const unsigned int         dofs_per_cell;
    const std::vector<double> &bdfCoeffs;

    std::vector<double> JxW;
    std::vector<double> face_JxW;

    std::vector<unsigned int> components;

    // Current and previous values and gradients for each quad node
    std::vector<Tensor<1, dim>>              present_velocity_values;
    std::vector<Tensor<2, dim>>              present_velocity_gradients;
    std::vector<double>                      present_pressure_values;
    std::vector<std::vector<Tensor<1, dim>>> previous_velocity_values;

    std::vector<Tensor<1, dim>>              present_displacement_values;
    std::vector<Tensor<2, dim>>              present_displacement_gradients;
    std::vector<std::vector<Tensor<1, dim>>> previous_displacement_values;
    std::vector<Tensor<1, dim>>              present_mesh_velocity_values;

    // Current and previous values on faces
    std::vector<Tensor<1, dim>>              present_face_velocity_values;
    std::vector<Tensor<1, dim>>              present_face_displacement_values;
    std::vector<Tensor<1, dim>>              present_face_mesh_velocity_values;
    std::vector<Tensor<1, dim>>              present_face_lambda_values;
    std::vector<std::vector<Tensor<1, dim>>> previous_face_displacement_values;

    // Source term on cell
    std::vector<Vector<double>>
      source_term_full; // The source term with n_components
    std::vector<Tensor<1, dim>>
      source_term_velocity;
    std::vector<Tensor<1, dim>>
      source_term_displacement;

    // Shape functions and gradients for each quad node and each dof
    std::vector<std::vector<double>>         div_phi_u;
    std::vector<std::vector<Tensor<1, dim>>> phi_u;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_u;
    std::vector<std::vector<double>>         phi_p;
    std::vector<std::vector<Tensor<1, dim>>> phi_disp;
    std::vector<std::vector<double>>         div_phi_disp;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_disp;

    // Face shape functions for each quad node and each dof
    // Only on the face matching the cylinder
    std::vector<std::vector<Tensor<1, dim>>> phi_u_face;
    std::vector<std::vector<Tensor<1, dim>>> phi_x_face;
    std::vector<std::vector<Tensor<1, dim>>> phi_l_face;
  };

  template <int dim>
  void ScratchData<dim>::allocate()
  {
    components.resize(dofs_per_cell);

    present_velocity_values.resize(n_q_points);
    present_velocity_gradients.resize(n_q_points);
    present_pressure_values.resize(n_q_points);
    present_displacement_values.resize(n_q_points);
    present_displacement_gradients.resize(n_q_points);
    present_mesh_velocity_values.resize(n_q_points);

    source_term_full.resize(n_q_points, Vector<double>(2 * dim + 1));
    source_term_velocity.resize(n_q_points);
    source_term_displacement.resize(n_q_points);

    present_face_velocity_values.resize(n_faces_q_points);
    present_face_displacement_values.resize(n_faces_q_points);
    present_face_lambda_values.resize(n_faces_q_points);
    present_face_mesh_velocity_values.resize(n_faces_q_points);

    // BDF
    previous_velocity_values.resize(2, std::vector<Tensor<1, dim>>(n_q_points));
    previous_displacement_values.resize(
      2, std::vector<Tensor<1, dim>>(n_q_points));
    previous_face_displacement_values.resize(
      2, std::vector<Tensor<1, dim>>(n_faces_q_points));

    div_phi_u.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_u.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    phi_p.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_disp.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_disp.resize(n_q_points,
                         std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_disp.resize(n_q_points, std::vector<double>(dofs_per_cell));

    phi_u_face.resize(n_faces_q_points,
                      std::vector<Tensor<1, dim>>(dofs_per_cell));
    phi_x_face.resize(n_faces_q_points,
                      std::vector<Tensor<1, dim>>(dofs_per_cell));
    phi_l_face.resize(n_faces_q_points,
                      std::vector<Tensor<1, dim>>(dofs_per_cell));

    JxW.resize(n_q_points);
    face_JxW.resize(n_faces_q_points);
  }

  class SimulationParameters
  {
  public:
    unsigned int velocity_degree;
    unsigned int displacement_degree;
    double       viscosity;
    double       pseudo_solid_mu;
    double       pseudo_solid_lambda;
    unsigned int bdf_order;
    double       t0;
    double       t1;
    double       dt;
    double       prev_dt;
    unsigned int nTimeSteps;
    unsigned int nConvergenceCycles;

  public:
    SimulationParameters(){};
  };

  template <int dim>
  class MMS
  {
  public:
    MMS(const SimulationParameters &param);

    void run();

  private:
    void set_bdf_coefficients(const unsigned int order);
    void make_grid(const unsigned int iMesh);
    void setup_system();
    void create_zero_constraints();
    void create_nonzero_constraints();
    void constrain_pressure_point(AffineConstraints<double> &constraints,
                                  bool                       set_to_zero);
    void create_sparsity_pattern();
    void set_initial_condition();
    void apply_nonzero_constraints();
    void set_exact_solution();
    void assemble_matrix(bool first_step);
    void assemble_local_matrix(
      bool                                                  first_step,
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData<dim>                                     &scratchData,
      LA::MPI::Vector                                      &current_solution,
      std::vector<LA::MPI::Vector>                         &previous_solutions,
      std::vector<types::global_dof_index>                 &local_dof_indices,
      FullMatrix<double>                                   &local_matrix,
      bool                                                  distribute);
    void assemble_local_matrix_fd(
      bool                                                  first_step,
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData<dim>                                     &scratchData,
      LA::MPI::Vector                                      &current_solution,
      std::vector<LA::MPI::Vector>                         &previous_solutions,
      std::vector<types::global_dof_index>                 &local_dof_indices,
      FullMatrix<double>                                   &local_matrix,
      Vector<double>                                       &ref_local_rhs,
      Vector<double>                                       &perturbed_local_rhs,
      std::vector<double>                                  &cell_dof_values);
    void assemble_rhs(bool first_step);
    void assemble_local_rhs(
      bool                                                  first_step,
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData<dim>                                     &scratchData,
      LA::MPI::Vector                                      &current_solution,
      std::vector<LA::MPI::Vector>                         &previous_solutions,
      std::vector<types::global_dof_index>                 &local_dof_indices,
      Vector<double>                                       &local_rhs,
      std::vector<double>                                  &cell_dof_values,
      bool                                                  distribute,
      bool update_cell_dof_values,
      bool use_full_solution);

    void solve_direct(bool first_step);
    void solve_newton();
    void move_mesh();
    void output_results(const unsigned int convergence_index,
                        const unsigned int time_step);
    void compute_errors(bool only_update_Linf);
    void reset();

    SimulationParameters param;

    MPI_Comm           mpi_communicator;
    const unsigned int mpi_rank;

    FESystem<dim> fe;

    // Ordering of the FE system
    // Each field is in the half-open [lower, upper)
    // Check for matching component by doing e.g.:
    // if(u_lower <= comp && comp < u_upper)
    const unsigned int n_components = 2 * dim + 1;
    const unsigned int u_lower      = 0;
    const unsigned int u_upper      = dim;
    const unsigned int p_lower      = dim;
    const unsigned int p_upper      = dim + 1;
    const unsigned int x_lower      = dim + 1;
    const unsigned int x_upper      = 2 * dim + 1;

  public:
    bool is_velocity(const unsigned int component) const
    {
      return u_lower <= component && component < u_upper;
    }
    bool is_pressure(const unsigned int component) const
    {
      return p_lower <= component && component < p_upper;
    }
    bool is_position_or_displacement(const unsigned int component) const
    {
      return x_lower <= component && component < x_upper;
    }

  public:
    parallel::fullydistributed::Triangulation<dim>             triangulation;
    MappingFE<dim>                                             mapping;
    std::unique_ptr<MappingFEField<dim, dim, LA::MPI::Vector>> mapping_fe_field;

    // Description of the .msh mesh entities
    std::map<unsigned int, std::string> mesh_domains_tag2name;
    std::map<std::string, unsigned int> mesh_domains_name2tag;

    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> zero_constraints;
    AffineConstraints<double> nonzero_constraints;

    // The global index of the pressure DoF to constrain to the value
    // of the manufactured solution.
    types::global_dof_index constrained_pressure_dof =
      numbers::invalid_dof_index;
    Point<dim> constrained_pressure_support_point;

    // Constrain non-boundary lambda dofs to 0
    AffineConstraints<double> lambda_constraints;
    // Constrain boundary displacement dofs to F/k
    AffineConstraints<double> displacement_constraints;

    LA::MPI::SparseMatrix system_matrix;

    // With ghosts (read only)
    LA::MPI::Vector present_solution;
    LA::MPI::Vector evaluation_point;

    // Without ghosts (owned)
    LA::MPI::Vector local_evaluation_point;
    LA::MPI::Vector newton_update;
    LA::MPI::Vector system_rhs;

    LA::MPI::Vector exact_solution;
    LA::MPI::Vector exact_solution_with_ghosts;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    std::vector<Point<dim>> initial_mesh_position;

    std::vector<LA::MPI::Vector> previous_solutions;
    std::vector<double>          bdfCoeffs;

    double current_time;

    Solution<dim>   solution_fun;
    SourceTerm<dim> source_term_fun;

    // Contiguous maps from global vertex index to its position dofs,
    // and vice versa.
    std::vector<std::vector<unsigned int>> vertex2position_dof;
    std::vector<unsigned int>              position_dof2vertex;

    // Store the first 2 owned cells on each proc to force
    // FEValues to reinit even if we call it twice on the same cell,
    // when computing finite differences on the residual.
    // Storing 2 so that we always have a different one available.
    // This is not very efficient, have to ask on the forum.
    std::vector<typename DoFHandler<dim>::active_cell_iterator> dummy_cells;

    double              linf_error_u;
    double              linf_error_p;
    double              linf_error_disp;
    ConvergenceTable    convergence_table;
    std::vector<double> l2_err_u;
    std::vector<double> l2_err_p;
    std::vector<double> l2_err_disp;
  };

  template <int dim>
  MMS<dim>::MMS(const SimulationParameters &param)
    : param(param)
    , mpi_communicator(MPI_COMM_WORLD)
    , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
    , fe(FE_SimplexP<dim>(param.velocity_degree), // Velocity
         dim,
         FE_SimplexP<dim>(param.velocity_degree - 1), // Pressure
         1,
         FE_SimplexP<dim>(
           param.displacement_degree), // Displacement or position
         dim)
    , triangulation(mpi_communicator)
    , mapping(FE_SimplexP<dim>(1))
    , dof_handler(triangulation)
    , pcout(std::cout, (mpi_rank == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , current_time(param.t0)
    , solution_fun(Solution<dim>(current_time, n_components))
    , source_term_fun(SourceTerm<dim>(current_time, n_components))
  {}

  template <int dim>
  void MMS<dim>::set_bdf_coefficients(const unsigned int order)
  {
    const double dt      = param.dt;
    const double prev_dt = param.prev_dt;

    switch (order)
    {
      case 0: // Stationary
        bdfCoeffs.resize(1);
        bdfCoeffs[0] = 0.;
        break;
      case 1:
        bdfCoeffs.resize(2);
        bdfCoeffs[0] = 1. / dt;
        bdfCoeffs[1] = -1. / dt;
        break;
      case 2:
        bdfCoeffs.resize(3);
        // bdfCoeffs[0] =  3. / (2. * dt);
        // bdfCoeffs[1] = -2. / dt;
        // bdfCoeffs[2] =  1. / (2. * dt);
        bdfCoeffs[0] = 1.0 / dt + 1.0 / (dt + prev_dt);
        bdfCoeffs[1] = -1.0 / dt - 1.0 / (prev_dt);
        bdfCoeffs[2] = dt / prev_dt * 1. / (dt + prev_dt);
        break;
      default:
        throw std::runtime_error(
          "Can only choose BDF1 or BDF2 time integration method");
    }
  }

  template <int dim>
  void MMS<dim>::make_grid(const unsigned int iMesh)
  {
    pcout << "Making grid..." << std::endl;

    Triangulation<dim> serial_tria;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(serial_tria);

    std::string meshFile = "";

    if constexpr (dim == 2)
    {
      meshFile = "../data/meshes/square" + std::to_string(iMesh) + ".msh";
    }
    else
    {
      meshFile = "../data/meshes/cube" + std::to_string(iMesh) + ".msh";
    }

    std::ifstream input(meshFile);
    AssertThrow(input, ExcMessage("Could not open mesh file: " + meshFile));
    grid_in.read_msh(input);

    // Partition serial triangulation:
    GridTools::partition_triangulation(
      Utilities::MPI::n_mpi_processes(mpi_communicator), serial_tria);

    // Create building blocks:
    const TriangulationDescription::Description<dim> description =
      TriangulationDescription::Utilities::
        create_description_from_triangulation(serial_tria, mpi_communicator);

    // Create a fully distributed triangulation:
    // copy_triangulation does not seems to work, so maybe give reference to the
    // mesh
    triangulation.create_triangulation(description);

    // Save initial position of the mesh vertices
    initial_mesh_position.resize(triangulation.n_vertices());
    for (auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto v : cell->vertex_indices())
      {
        const unsigned int global_vertex_index     = cell->vertex_index(v);
        initial_mesh_position[global_vertex_index] = cell->vertex(v);
      }
    }

    read_gmsh_physical_names(meshFile,
                             mesh_domains_tag2name,
                             mesh_domains_name2tag);

    // Print mesh info
    if (mpi_rank == 0)
    {
      std::cout << "Mesh info:" << std::endl
                << " dimension: " << dim << std::endl
                << " no. of cells: " << serial_tria.n_active_cells()
                << std::endl;

      std::map<types::boundary_id, unsigned int> boundary_count;
      for (const auto &face : serial_tria.active_face_iterators())
        if (face->at_boundary())
          boundary_count[face->boundary_id()]++;

      std::cout << " boundary indicators: ";
      for (const std::pair<const types::boundary_id, unsigned int> &pair :
           boundary_count)
      {
        std::cout << pair.first << '(' << pair.second << " times) ";
      }
      std::cout << std::endl;

      // Check that all boundary indices found in the mesh
      // have a matching name, to make sure we're not forgetting
      // a boundary.
      for (const auto &[id, count] : boundary_count)
      {
        if (mesh_domains_tag2name.count(id) == 0)
          throw std::runtime_error("Deal.ii read a boundary entity with tag " +
                                   std::to_string(id) +
                                   " in the mesh, but no physical entity with "
                                   "this tag was read from the mesh file.");
      }

      for (const auto &[id, name] : mesh_domains_tag2name)
        std::cout << "ID " << id << " -> " << name << "\n";
    }
  }

  template <int dim>
  void MMS<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);
    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
          << std::endl;

    locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    locally_relevant_dofs =
      DoFTools::extract_locally_relevant_dofs(this->dof_handler);

    //
    // Initialize parallel vectors
    //
    present_solution.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            mpi_communicator);
    evaluation_point.reinit(locally_owned_dofs,
                            locally_relevant_dofs,
                            mpi_communicator);

    local_evaluation_point.reinit(locally_owned_dofs, mpi_communicator);
    newton_update.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    exact_solution.reinit(locally_owned_dofs, mpi_communicator);
    exact_solution_with_ghosts.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    // BDF solutions
    previous_solutions.clear();
    previous_solutions.resize(bdfCoeffs.size() - 1); // 1 or 2
    for (auto &previous_sol : previous_solutions)
    {
      previous_sol.clear();
      previous_sol.reinit(locally_owned_dofs,
                          locally_relevant_dofs,
                          mpi_communicator);
    }
  }

  template <int dim>
  void MMS<dim>::create_sparsity_pattern()
  {
    //
    // Sparsity pattern and allocate matrix
    // After the constraints have been defined
    //
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler,
                                    dsp,
                                    nonzero_constraints,
                                    false);
    SparsityTools::distribute_sparsity_pattern(dsp,
                                               locally_owned_dofs,
                                               mpi_communicator,
                                               locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs,
                         locally_owned_dofs,
                         dsp,
                         mpi_communicator);
  }

  template <int dim>
  void MMS<dim>::set_initial_condition()
  {
    const FEValuesExtractors::Vector velocities(u_lower);

    // Velocity
    VectorTools::interpolate(mapping,
                             dof_handler,
                             solution_fun,
                             newton_update,
                             fe.component_mask(velocities));

    // #if defined(WITH_MESH_POSITION)
    //   // Mesh position
    //   // Initialize directly from the triangulation
    //   const FEValuesExtractors::Vector mesh_position(dx_lower);
    //   VectorTools::get_position_vector(mapping,
    //                                    dof_handler,
    //                                    newton_update,
    //                                    fe.component_mask(mesh_position));
    //   newton_update.compress(VectorOperation::insert);
    // #endif

    // Apply non-homogeneous Dirichlet BC and set as current solution
    nonzero_constraints.distribute(newton_update);
    present_solution = newton_update;

    // Dirty copy of the initial condition for BDF2 for now (-:
    for (auto &sol : previous_solutions)
      sol = present_solution;
  }

  template <int dim>
  void
  MMS<dim>::constrain_pressure_point(AffineConstraints<double> &constraints,
                                     bool                       set_to_zero)
  {
    // Determine the pressure dof the first time
    if (constrained_pressure_dof == numbers::invalid_dof_index)
    {
      // Choose a fixed physical reference location
      // Here it's the origin (Point<dim> initialized at 0)
      const Point<dim> reference_point;

      const FEValuesExtractors::Scalar pressure(p_lower);
      IndexSet                         pressure_dofs =
        DoFTools::extract_dofs(dof_handler, fe.component_mask(pressure));

      // Get support points for locally relevant DoFs
      std::map<types::global_dof_index, Point<dim>> support_points;
      DoFTools::map_dofs_to_support_points(mapping,
                                           dof_handler,
                                           support_points);

      double local_min_dist             = std::numeric_limits<double>::max();
      types::global_dof_index local_dof = numbers::invalid_dof_index;

      for (auto idx : pressure_dofs)
      {
        if (!locally_owned_dofs.is_element(idx))
          continue;

        const double dist = support_points[idx].distance(reference_point);
        if (dist < local_min_dist)
        {
          local_min_dist = dist;
          local_dof      = idx;
        }
      }

      // Prepare for MPI_MINLOC reduction
      struct MinLoc
      {
        double                  dist;
        types::global_dof_index dof;
      } local_pair{local_min_dist, local_dof}, global_pair;

      // MPI reduction to find the global closest DoF
      MPI_Allreduce(&local_pair,
                    &global_pair,
                    1,
                    MPI_DOUBLE_INT,
                    MPI_MINLOC,
                    mpi_communicator);

      constrained_pressure_dof = global_pair.dof;

      // Set support point for MMS evaluation
      if (locally_owned_dofs.is_element(constrained_pressure_dof))
      {
        constrained_pressure_support_point =
          support_points[constrained_pressure_dof];
      }
    }

    // Constrain that DoF globally
    if (locally_owned_dofs.is_element(constrained_pressure_dof))
    {
      const double pAnalytic =
        solution_fun.value(constrained_pressure_support_point, p_lower);
      constraints.add_line(constrained_pressure_dof);
      constraints.set_inhomogeneity(constrained_pressure_dof,
                                    set_to_zero ? 0. : pAnalytic);
    }

    // constraints.make_consistent_in_parallel(locally_owned_dofs,
    //                                         constraints.get_local_lines(),
    //                                         mpi_communicator);
  }

  // template <int dim>
  // void
  // MMS<dim>::constrain_pressure_point(AffineConstraints<double> &constraints,
  //                                     bool                       set_to_zero)
  // {
  //   bool first = true;

  //   // Constraint a single pressure DoF to manufactured solution
  //   if (mpi_rank == 0)
  //   {
  //     for (const auto &cell : dof_handler.active_cell_iterators())
  //     {
  //       if (!cell->is_locally_owned())
  //         continue;

  //       std::vector<types::global_dof_index> dofs(fe.dofs_per_cell);
  //       cell->get_dof_indices(dofs);

  //       for (unsigned int i = 0; i < fe.dofs_per_cell; ++i)
  //       {
  //         const unsigned int comp = fe.system_to_component_index(i).first;
  //         if (is_pressure(comp) && !constraints.is_constrained(dofs[i]))
  //         {
  //           if (!fe.has_support_points())
  //             throw std::runtime_error(
  //               "Could not locate position of pressure DoF because FE space "
  //               "does not have support points");

  //           // // Don't take first Dof for debug
  //           // if(first)
  //           // {
  //           //   first = false;
  //           //   break; // Break dof loop, go to next cell
  //           // }

  //           // Get exact pressure
  //           Point<dim> refPoint = fe.unit_support_point(i);
  //           Point<dim> pPoint =
  //             mapping.transform_unit_to_real_cell(cell, refPoint);
  //           double pAnalytic = solution_fun.value(pPoint, p_lower);

  //           if (set_to_zero)
  //             pAnalytic = 0.;

  //           std::cout << "Applying pressure constraint at dof " << dofs[i]
  //                     << " : pos = " << pPoint << " - p = " << pAnalytic
  //                     << std::endl;

  //           constraints.add_line(dofs[i]);
  //           constraints.set_inhomogeneity(dofs[i], pAnalytic);
  //           goto escape;
  //         }
  //       }
  //     }
  //   escape:;
  //   }

  //   // Tell the buddies that there is a constrained pressure DoF
  //   constraints.make_consistent_in_parallel(locally_owned_dofs,
  //     constraints.get_local_lines(), mpi_communicator);
  // }

  template <int dim>
  void MMS<dim>::create_zero_constraints()
  {
    zero_constraints.clear();
    zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    // Boundaries where Dirichlet BC are applied,
    // where the Newton increment should be zero.
    const FEValuesExtractors::Vector velocities(u_lower);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             mesh_domains_name2tag.at("Bord"),
                                             Functions::ZeroFunction<dim>(
                                               n_components),
                                             zero_constraints,
                                             fe.component_mask(velocities));

    const FEValuesExtractors::Vector disp_position(x_lower);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             mesh_domains_name2tag.at("Bord"),
                                             Functions::ZeroFunction<dim>(
                                               n_components),
                                             zero_constraints,
                                             fe.component_mask(disp_position));

    // Calls make_consistent_in_parallel, which closes the constraints
    bool set_to_zero = true;
    this->constrain_pressure_point(zero_constraints, set_to_zero);

    zero_constraints.close();
  }

  template <int dim>
  void MMS<dim>::create_nonzero_constraints()
  {
    nonzero_constraints.clear();
    nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    const FEValuesExtractors::Vector velocities(u_lower);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             mesh_domains_name2tag.at("Bord"),
                                             solution_fun,
                                             nonzero_constraints,
                                             fe.component_mask(velocities));

    const FEValuesExtractors::Vector disp_position(x_lower);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             mesh_domains_name2tag.at("Bord"),
                                             solution_fun,
                                             nonzero_constraints,
                                             fe.component_mask(disp_position));

    // Calls make_consistent_in_parallel, which closes the constraints
    bool set_to_zero = false;
    this->constrain_pressure_point(nonzero_constraints, set_to_zero);

    nonzero_constraints.close();
  }

  template <int dim>
  void MMS<dim>::set_exact_solution()
  {
    VectorTools::interpolate(mapping,
                             dof_handler,
                             solution_fun,
                             local_evaluation_point);
    present_solution = local_evaluation_point;
  }

  template <int dim>
  void MMS<dim>::apply_nonzero_constraints()
  {
    nonzero_constraints.distribute(local_evaluation_point);
    present_solution = local_evaluation_point;
  }

  template <int dim>
  void MMS<dim>::assemble_matrix(bool first_step)
  {
    // pcout << "Assembling matrix" << std::endl;
    TimerOutput::Scope t(computing_timer, "Assemble matrix");

    system_matrix = 0;

    const QGaussSimplex<dim>     cell_quadrature(4);
    const QGaussSimplex<dim - 1> face_quadrature(4);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    ScratchData<dim>                     scratchData(
      fe, cell_quadrature, mapping, face_quadrature, dofs_per_cell, bdfCoeffs);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_matrix_fd(dofs_per_cell, dofs_per_cell);
    Vector<double>     ref_local_rhs(dofs_per_cell);
    Vector<double>     perturbed_local_rhs(dofs_per_cell);

    FullMatrix<double> diff_matrix(dofs_per_cell, dofs_per_cell);

    // The local dofs values, which will be perturbed
    std::vector<double> cell_dof_values(dofs_per_cell);

    // Use local evaluation point because we'll have to modify the position
    // directly in the owned solution vector.
    // local_evaluation_point = evaluation_point;

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      bool distribute = true;
      this->assemble_local_matrix(first_step,
                                  cell,
                                  scratchData,
                                  evaluation_point,
                                  previous_solutions,
                                  local_dof_indices,
                                  local_matrix,
                                  distribute);

      // this->assemble_local_matrix_fd(first_step, cell, scratchData,
      //   evaluation_point, previous_solutions, local_dof_indices,
      //   local_matrix_fd, ref_local_rhs, perturbed_local_rhs,
      //   cell_dof_values);

      // diff_matrix.equ(1.0, local_matrix);
      // diff_matrix.add(-1.0, local_matrix_fd);
      // std::cout << "Max difference is " << diff_matrix.linfty_norm() <<
      // std::endl;
    }

    system_matrix.compress(VectorOperation::add);
  }

  template <int dim>
  void MMS<dim>::assemble_local_matrix_fd(
    bool                                                  first_step,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim>                                     &scratchData,
    LA::MPI::Vector                                      &current_solution,
    std::vector<LA::MPI::Vector>                         &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    FullMatrix<double>                                   &local_matrix,
    Vector<double>                                       &ref_local_rhs,
    Vector<double>                                       &perturbed_local_rhs,
    std::vector<double>                                  &cell_dof_values)
  {
    if (!cell->is_locally_owned())
      return;

    local_matrix        = 0.;
    ref_local_rhs       = 0.;
    perturbed_local_rhs = 0.;

    const double h = 1e-8;

    cell->get_dof_indices(local_dof_indices);

    const unsigned int dofs_per_cell = local_dof_indices.size();

    // Get the local dofs values
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
      cell_dof_values[j] = evaluation_point[local_dof_indices[j]];

    // Compute the non-perturbed residual, do not distribute in global RHS
    // Actually: we can probably distribute here, and save an extra residual
    // computation
    bool distribute             = false;
    bool update_cell_dof_values = false;
    bool use_full_solution      = false;
    this->assemble_local_rhs(first_step,
                             cell,
                             scratchData,
                             current_solution,
                             previous_solutions,
                             local_dof_indices,
                             ref_local_rhs,
                             cell_dof_values,
                             distribute,
                             update_cell_dof_values,
                             use_full_solution);

    // Loop over each DoF to compute directional derivative
#if defined(WITH_MESH_POSITION)
    //
    // Solve for the mesh position, not displacement.
    // The non-position dofs are perturbed by modifying the local cell dof
    // values (cell_dof_values) The position dofs are perturbed by modyfing both
    // cell_dof_values and the local_evaluation_point, which then affects the
    // fe_values reinits through the MappingFEField.
    //
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      const unsigned int comp = fe.system_to_component_index(j).first;

      const double og_value = cell_dof_values[j];
      cell_dof_values[j] += h;

      if (is_position_or_displacement(comp))
      {
        // Also modify mapping_fe_field
        local_evaluation_point[local_dof_indices[j]] = cell_dof_values[j];
        local_evaluation_point.compress(VectorOperation::insert);
      }

      // Reinit is called in the local rhs function
      this->assemble_local_rhs(first_step,
                               cell,
                               scratchData,
                               current_solution,
                               previous_solutions,
                               local_dof_indices,
                               perturbed_local_rhs,
                               cell_dof_values,
                               distribute,
                               update_cell_dof_values,
                               use_full_solution);

      // Finite differences (with sign change as residual is -NL(u))
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        local_matrix(i, j) = -(perturbed_local_rhs(i) - ref_local_rhs(i)) / h;
      }

      // Restore solution
      cell_dof_values[j] = og_value;
      if (is_position_or_displacement(comp))
      {
        // Also modify mapping_fe_field
        local_evaluation_point[local_dof_indices[j]] = og_value;
        local_evaluation_point.compress(VectorOperation::insert);
      }
    }
#else
    //
    // Solve for the displacement and do not account for perturbations of
    // position
    //
    for (unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      const double og_value = cell_dof_values[j];
      cell_dof_values[j] += h;

      // Reinit is called in the local rhs function
      this->assemble_local_rhs(first_step,
                               cell,
                               scratchData,
                               current_solution,
                               previous_solutions,
                               local_dof_indices,
                               perturbed_local_rhs,
                               cell_dof_values,
                               distribute,
                               update_cell_dof_values,
                               use_full_solution);

      // Finite differences (with sign change as residual is -NL(u))
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        local_matrix(i, j) = -(perturbed_local_rhs(i) - ref_local_rhs(i)) / h;
      }

      // Restore solution
      cell_dof_values[j] = og_value;
    }
#endif

    cell->get_dof_indices(local_dof_indices);
    if (first_step)
    {
      nonzero_constraints.distribute_local_to_global(local_matrix,
                                                     local_dof_indices,
                                                     system_matrix);
    }
    else
    {
      zero_constraints.distribute_local_to_global(local_matrix,
                                                  local_dof_indices,
                                                  system_matrix);
    }
  }

  template <int dim>
  void MMS<dim>::assemble_local_matrix(
    bool                                                  first_step,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim>                                     &scratchData,
    LA::MPI::Vector                                      &current_solution,
    std::vector<LA::MPI::Vector>                         &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    FullMatrix<double>                                   &local_matrix,
    bool                                                  distribute)
  {
    if (!cell->is_locally_owned())
      return;

    scratchData.reinit(cell,
                       current_solution,
                       previous_solutions,
                       source_term_fun);

    local_matrix = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &present_velocity_values =
        scratchData.present_velocity_values[q];
      const auto &present_velocity_gradients =
        scratchData.present_velocity_gradients[q];

      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];

      const auto &phi_p = scratchData.phi_p[q];

      const auto &present_mesh_velocity_values =
      scratchData.present_mesh_velocity_values[q];

      const auto &phi_x      = scratchData.phi_disp[q];
      const auto &grad_phi_x = scratchData.grad_phi_disp[q];
      const auto &div_phi_x  = scratchData.div_phi_disp[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const unsigned int component_i = scratchData.components[i];
        const bool         i_is_u      = is_velocity(component_i);
        const bool         i_is_p      = is_pressure(component_i);
        const bool         i_is_d = is_position_or_displacement(component_i);

        for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
        {
          const unsigned int component_j = scratchData.components[j];
          const bool         j_is_u      = is_velocity(component_j);
          const bool         j_is_p      = is_pressure(component_j);
          const bool         j_is_d = is_position_or_displacement(component_j);

          double local_matrix_ij = 0.;

          // Velocity - velocity block
          if (i_is_u && j_is_u)
          {
#if defined(WITH_TRANSIENT)
            // Time-dependent
            local_matrix_ij += bdfCoeffs[0] * phi_u[i] * phi_u[j];
#endif

#if defined(WITH_CONVECTION)
            // Convective
            local_matrix_ij += (grad_phi_u[j] * present_velocity_values +
                                present_velocity_gradients * phi_u[j]) *
                               phi_u[i];
#endif

#if defined(WITH_ALE)
            // ALE acceleration : - w dot grad(delta u)
            local_matrix_ij += - (grad_phi_u[j] * present_mesh_velocity_values) * phi_u[i];
#endif

            if (component_i == component_j)
            {
              // Diffusive
              local_matrix_ij += param.viscosity * grad_phi_u[i][component_i] *
                                 grad_phi_u[j][component_j];
            }
          }

          // Velocity - pressure block
          if (i_is_u && j_is_p)
          {
            // Pressure gradient
            local_matrix_ij += -div_phi_u[i] * phi_p[j];
          }

          // Velocity - mesh velocity block (u-chi)
          // Block is the same for either displacement or position
#if defined(WITH_ALE)
          if(i_is_u && j_is_d)
          {
            // ALE acceleration : - delta w dot grad(u)  = - c0 * phi_xj dot grad(u)
            local_matrix_ij += - bdfCoeffs[0]  * present_velocity_gradients * phi_x[i] * phi_u[i];
          }
#endif

          // Pressure - velocity block
          if (i_is_p && j_is_u)
          {
            // Incompressibility
            local_matrix_ij += -phi_p[i] * div_phi_u[j];
          }

          // Displacement - displacement block (chi-chi)
          if (i_is_d && j_is_d)
          {
            // Linear elasticity for pseudo-solid
            local_matrix_ij +=
              param.pseudo_solid_lambda * div_phi_x[j] * div_phi_x[i] +
              param.pseudo_solid_mu *
                scalar_product((grad_phi_x[i] + transpose(grad_phi_x[i])),
                               grad_phi_x[j]);
          }

          local_matrix_ij *= JxW;
          local_matrix(i, j) += local_matrix_ij;
        }
      }
    }

    // const FEValuesExtractors::Vector velocities(u_lower);
    // const FEValuesExtractors::Vector lambda(l_lower);
    // const unsigned int n_faces_q_points =
    // fe_face_values.get_quadrature().size();

    // // Face contributions
    // if(cell->at_boundary())
    // {
    //   for (const auto &face : cell->face_iterators())
    //   {
    //     if(face->at_boundary() && face->boundary_id() ==
    //     param.cylinder_boundary_id)
    //     {
    //       for (unsigned int q = 0; q < n_faces_q_points; ++q)
    //       {
    //         const double JxW  = scratchData.face_JxW[q];
    //         const auto &phi_u = scratchData.phi_u_face[q];
    //         const auto &phi_x = scratchData.phi_x_face[q];
    //         const auto &phi_l = scratchData.phi_l_face[q];

    //         // Here we skip the block structure for now
    //         // because the integral is on a few cells only
    //         // Can be added later
    //         for(unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
    //         {
    //           for(unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
    //           {
    //             local_matrix(i, j) += phi_l[j] * phi_u[i] * JxW;
    //             local_matrix(i, j) += (phi_u[j] - bdfCoeffs[0] * phi_x[j]) *
    //             phi_l[i] * JxW;
    //           }
    //         }
    //       }
    //     }
    //   }
    // }

    if (distribute)
    {
      cell->get_dof_indices(local_dof_indices);
      if (first_step)
      {
        nonzero_constraints.distribute_local_to_global(local_matrix,
                                                       local_dof_indices,
                                                       system_matrix);
      }
      else
      {
        zero_constraints.distribute_local_to_global(local_matrix,
                                                    local_dof_indices,
                                                    system_matrix);
      }
    }
  }

  template <int dim>
  void MMS<dim>::assemble_rhs(bool first_step)
  {
    // pcout << "Assembling RHS" << std::endl;
    TimerOutput::Scope t(computing_timer, "Assemble RHS");

    system_rhs = 0;

    const QGaussSimplex<dim>     cell_quadrature(4);
    const QGaussSimplex<dim - 1> face_quadrature(4);

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    ScratchData<dim> scratchData(
      fe, cell_quadrature, mapping, face_quadrature, dofs_per_cell, bdfCoeffs);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<double> cell_dof_values(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    // local_evaluation_point = evaluation_point;

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      bool distribute             = true;
      bool update_cell_dof_values = false;
      bool use_full_solution      = true;
      this->assemble_local_rhs(first_step,
                               cell,
                               scratchData,
                               evaluation_point,
                               previous_solutions,
                               local_dof_indices,
                               local_rhs,
                               cell_dof_values,
                               distribute,
                               update_cell_dof_values,
                               use_full_solution);
    }

    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void MMS<dim>::assemble_local_rhs(
    bool                                                  first_step,
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim>                                     &scratchData,
    LA::MPI::Vector                                      &current_solution,
    std::vector<LA::MPI::Vector>                         &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    Vector<double>                                       &local_rhs,
    std::vector<double>                                  &cell_dof_values,
    bool                                                  distribute,
    bool update_cell_dof_values,
    bool use_full_solution)
  {
    if (update_cell_dof_values)
    {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int j = 0; j < local_dof_indices.size(); ++j)
        cell_dof_values[j] = local_evaluation_point[local_dof_indices[j]];
    }

    if (use_full_solution)
    {
      scratchData.reinit(cell,
                         current_solution,
                         previous_solutions,
                         source_term_fun);
    }
    else
      scratchData.reinit(cell,
                         cell_dof_values,
                         previous_solutions,
                         source_term_fun);

    local_rhs = 0;

    const unsigned int          nBDF = bdfCoeffs.size();
    std::vector<Tensor<1, dim>> velocity(nBDF);
    std::vector<Tensor<1, dim>> displacement(nBDF);

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &present_velocity_values =
        scratchData.present_velocity_values[q];
      const auto &present_velocity_gradients =
        scratchData.present_velocity_gradients[q];
      const auto &present_pressure_values =
        scratchData.present_pressure_values[q];
      const auto &present_displacement_values =
        scratchData.present_displacement_values[q];
      const auto &present_displacement_gradients =
        scratchData.present_displacement_gradients[q];
      const auto &present_mesh_velocity_values =
        scratchData.present_mesh_velocity_values[q];

      const auto &source_term_velocity = scratchData.source_term_velocity[q];
      const auto &source_term_displacement =
        scratchData.source_term_displacement[q];

      double present_velocity_divergence = trace(present_velocity_gradients);
      double present_displacement_divergence =
        trace(present_displacement_gradients);

      // BDF
      velocity[0]     = present_velocity_values;
      displacement[0] = present_displacement_values;
      for (unsigned int i = 1; i < nBDF; ++i)
      {
        velocity[i]     = scratchData.previous_velocity_values[i - 1][q];
        displacement[i] = scratchData.previous_displacement_values[i - 1][q];
      }

      const Tensor<1, dim> uDotGradU =
        present_velocity_gradients * present_velocity_values;
      const Tensor<1, dim> wDotGradU =
        present_velocity_gradients * present_mesh_velocity_values;

      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];

      const auto &phi_p = scratchData.phi_p[q];

      const auto &phi_x      = scratchData.phi_disp[q];
      const auto &grad_phi_x = scratchData.grad_phi_disp[q];
      const auto &div_phi_x  = scratchData.div_phi_disp[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        // const unsigned int component_i = scratchData.components[i];
        // const bool i_is_u = is_velocity(component_i);
        // const bool i_is_p = is_pressure(component_i);
        // const bool i_is_d = is_displacement(component_i);
        // const bool i_is_l = is_lambda(component_i);

        const auto present_displacement_gradient_sym =
          present_displacement_gradients +
          transpose(present_displacement_gradients);

        double local_rhs_i =
          -(
        // Navier-Stokes ALE:
#if defined(WITH_CONVECTION)
            // Convective
            uDotGradU * phi_u[i]
#endif

#if defined(WITH_ALE)
            // ALE acceleration
            - wDotGradU * phi_u[i]
#endif

            // Diffusive
            + param.viscosity *
                scalar_product(present_velocity_gradients, grad_phi_u[i])

            // Pressure gradient
            - div_phi_u[i] * present_pressure_values

            // Incompressibility
            - phi_p[i] * present_velocity_divergence

            // Linear elasticity
            + param.pseudo_solid_lambda * present_displacement_divergence *
                div_phi_x[i] +
            param.pseudo_solid_mu *
              scalar_product(present_displacement_gradient_sym, grad_phi_x[i])

            // Source term for momentum
            + phi_u[i] * source_term_velocity

            // Source term for displacement
            + phi_x[i] * source_term_displacement

            ) *
          JxW;

#if defined(WITH_TRANSIENT)
        // Transient terms:
        for (unsigned int iBDF = 0; iBDF < nBDF; ++iBDF)
        {
          local_rhs_i -= bdfCoeffs[iBDF] * velocity[iBDF] * phi_u[i] * JxW;
        }
#endif

        local_rhs(i) += local_rhs_i;
      }
    }

    // const FEValuesExtractors::Vector velocities(u_lower);
    // const FEValuesExtractors::Vector lambda(l_lower);
    // const unsigned int n_faces_q_points =
    // fe_face_values.get_quadrature().size();

    // //
    // // Face contributions (Lagrange multiplier)
    // //
    // if(cell->at_boundary())
    // {
    //   for (const auto &face : cell->face_iterators())
    //   {
    //     if(face->at_boundary() && face->boundary_id() ==
    //     param.cylinder_boundary_id)
    //     {
    //       const auto &current_u = scratchData.present_face_velocity_values;
    //       const auto &current_w =
    //       scratchData.present_face_mesh_velocity_values; const auto
    //       &current_l = scratchData.present_face_lambda_values;

    //       for (unsigned int q = 0; q < n_faces_q_points; ++q)
    //       {
    //         const double JxW  = scratchData.face_JxW[q];
    //         const auto &phi_u = scratchData.phi_u_face[q];
    //         const auto &phi_l = scratchData.phi_l_face[q];

    //         for(unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
    //         {
    //           local_rhs(i) -= current_l[q] * phi_u[i] * JxW;
    //           local_rhs(i) -= (current_u[q] - current_w[q]) * phi_l[i] * JxW;
    //           // local_rhs(i) -= pow(current_u[q] - current_w[q], 2) *
    //           phi_l[i] * JxW;
    //         }
    //       }
    //     }
    //   }
    // }

    if (distribute)
    {
      cell->get_dof_indices(local_dof_indices);
      if (first_step)
        nonzero_constraints.distribute_local_to_global(local_rhs,
                                                       local_dof_indices,
                                                       system_rhs);
      else
        zero_constraints.distribute_local_to_global(local_rhs,
                                                    local_dof_indices,
                                                    system_rhs);
    }
  }

  template <int dim>
  void MMS<dim>::solve_direct(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Solve direct");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    // Solve with MUMPS
    SolverControl                    solver_control;
    PETScWrappers::SparseDirectMUMPS solver(solver_control);
    solver.solve(system_matrix, completely_distributed_solution, system_rhs);

    newton_update = completely_distributed_solution;

    if (first_step)
      nonzero_constraints.distribute(newton_update);
    else
      zero_constraints.distribute(newton_update);
  }

  template <int dim>
  void MMS<dim>::solve_newton()
  {
    double             current_res = 1e6;
    double             norm_correction;
    bool               first_step = true;
    unsigned int       iter       = 1;
    const unsigned int max_iter   = 10;
    const double       tol        = 1e-9;
    bool               converged  = false;

    while (current_res > tol && iter <= max_iter)
    {
      evaluation_point = present_solution;

      this->assemble_matrix(first_step);
      this->assemble_rhs(first_step);
      current_res = system_rhs.linfty_norm();

      if (iter == 1)
      {
        current_res = system_rhs.linfty_norm();
      }

      this->solve_direct(first_step);
      first_step = false;

      norm_correction = newton_update.linfty_norm(); // On this proc only!
      pcout << std::scientific << std::setprecision(8)
            << "Newton iteration: " << iter << " - ||du|| = " << norm_correction
            << " - ||NL(u)|| = " << current_res << std::endl;

      if (norm_correction > 1e10 || current_res > 1e10)
      {
        pcout << "Diverged after " << iter << " iteration(s)" << std::endl;
        if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          throw std::runtime_error("Nonlinear solver diverged");
      }

      local_evaluation_point = present_solution;
      local_evaluation_point.add(1., newton_update);
      nonzero_constraints.distribute(local_evaluation_point);
      evaluation_point = local_evaluation_point;

      this->assemble_rhs(first_step);
      current_res = system_rhs.linfty_norm();

      if (current_res <= tol)
      {
        pcout << "Converged in " << iter
              << " iteration(s) because next nonlinear residual is below "
                 "tolerance: "
              << current_res << " < " << tol << std::endl;
        converged = true;
      }

      present_solution = evaluation_point;
      ++iter;
    }

    if (!converged && iter == max_iter + 1)
    {
      pcout << "Did not converge after " << iter << " iteration(s)"
            << std::endl;
      if (Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        throw std::runtime_error("Nonlinear solver did not convege");
    }
  }

  template <int dim>
  void MMS<dim>::move_mesh()
  {
    TimerOutput::Scope t(computing_timer, "Move mesh");
    pcout << "    Moving mesh..." << std::endl;

    ///////////////////
    VectorTools::interpolate(mapping,
                             dof_handler,
                             solution_fun,
                             exact_solution);
    exact_solution_with_ghosts = exact_solution;
    ///////////////////

    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (const auto v : cell->vertex_indices())
      {
        const unsigned int global_vertex_index = cell->vertex_index(v);

        if (vertex_touched[global_vertex_index])
          continue;

        vertex_touched[global_vertex_index] = true;

        Point<dim> vertex_displacement_or_position;

        for (unsigned int i = 0; i < fe.n_dofs_per_vertex(); ++i)
        {
          const unsigned int dof_index = cell->vertex_dof_index(v, i);
          const unsigned int comp = fe.system_to_component_index(i).first;

          if (is_position_or_displacement(comp))
          {
            const unsigned int d = comp - x_lower;
            vertex_displacement_or_position[d] = exact_solution_with_ghosts[dof_index];
            // if(d == 0)
            //   std::cout << "At vertex " << cell->vertex(v) << " : dep_x = " << vertex_displacement_or_position[0] << " vs " << chi_fun(0., cell->vertex(v), 0) << std::endl;
            // if(d == 1)
            //   std::cout << "At vertex " << cell->vertex(v) << " : dep_y = " << vertex_displacement_or_position[1] << " vs " << chi_fun(0., cell->vertex(v), 1) << std::endl;
            if(std::abs(vertex_displacement_or_position[d] - chi_fun(0, cell->vertex(v), d)) > 1e-13)
            {
              throw std::runtime_error("Mismatch");
            }
          }
        }

        // for (unsigned int d = 0; d < dim; ++d)
        // {
        //   // Index of the displacement component
        //   const unsigned int displacement_component = offset + d;

        //   // Find system index of that displacement component at this vertex
        //   const unsigned int system_index =
        //     fe.component_to_system_index(displacement_component, 0); // 0 = first shape function for that component

        //   // Use `vertex_dof_index` to get DoF index for this vertex and component
        //   const unsigned int dof_index = cell->vertex_dof_index(v, system_index);

        //   vertex_displacement_or_position[d] = present_solution[dof_index];
        // }

        #if defined(WITH_MESH_POSITION)
          cell->vertex(v) = vertex_displacement_or_position;
        #else
          cell->vertex(v) = initial_mesh_position[global_vertex_index] + vertex_displacement_or_position;
        #endif
      }
    }
  }

  // template <int dim>
  // void MMS<dim>::move_mesh()
  // {
  //   std::vector<bool> vertex_touched(triangulation.n_vertices(), false);

  //   for (const auto &cell : dof_handler.active_cell_iterators())
  //     if (cell->is_locally_owned())
  //       for (const auto v : cell->vertex_indices())
  //         if (vertex_touched[cell->vertex_index(v)] == false)
  //           {
  //             vertex_touched[cell->vertex_index(v)] = true;

  //             Point<dim> vertex_displacement;
  //             for (unsigned int d = 0; d < dim; ++d)
  //               vertex_displacement[d] =
  //                 displacement(cell->vertex_dof_index(v, d));

  //             cell->vertex(v) += vertex_displacement;
  //           }
  // }

  // template <int dim>
  // void MMS<dim>::move_mesh()
  // {
  //   pcout << "    Moving mesh in parallel..." << std::endl;

  //   // const MPI_Comm mpi_comm = triangulation.get_communicator();  
  //   // const unsigned int my_rank = Utilities::MPI::this_mpi_process(mpi_comm);
  //   const unsigned int n_vertices = triangulation.n_vertices();

  //   // 1. Prepare storage for new positions on all ranks.
  //   std::vector<Point<dim>> new_positions(n_vertices);

  //   // Initialize with current positions.
  //   for (unsigned int v = 0; v < n_vertices; ++v)
  //     new_positions[v] = triangulation.get_vertices()[v];

  //   // 2. Compute displacement for vertices that this rank **owns**:
  //   std::vector<bool> updated(n_vertices, false);
  //   for (const auto &cell : dof_handler.active_cell_iterators())
  //     if (cell->is_locally_owned())
  //       for (unsigned int v = 0; v < cell->n_vertices(); ++v)
  //       {
  //         const unsigned int vid = cell->vertex_index(v);
  //         if (!updated[vid])
  //         {
  //           updated[vid] = true;
  //           Point<dim> disp;
  //           for (unsigned int d = 0; d < dim; ++d)
  //             disp[d] = incremental_displacement(cell->vertex_dof_index(v, d));

  //           new_positions[vid] = cell->vertex(v) + disp;
  //         }
  //       }

  //   // 3. Communicate new coordinates: each rank owning a vertex broadcasts it.
  //   for (unsigned int v = 0; v < n_vertices; ++v)
  //   {
  //     // Solve ownership: choose smallest rank that has it (since there is no method,
  //     // we approximate by asking each rank if it's locally owned)
  //     // We'll gather a mask of which ranks updated each vertex:
  //     bool i_updated = updated[v];
  //     std::vector<char> all_updated(Utilities::MPI::n_mpi_processes(mpi_communicator));
  //     Utilities::MPI::Allgather(&i_updated, 1, all_updated.data(), 1, mpi_communicator);

  //     int owner = -1;
  //     for (unsigned int r = 0; r < all_updated.size(); ++r)
  //       if (all_updated[r])
  //       {
  //         owner = r;
  //         break;  // first rank that updated
  //       }

  //     Assert(owner >= 0, ExcInternalError());

  //     // Broadcast from that owner
  //     MPI_Bcast(&new_positions[v][0], dim, MPI_DOUBLE, owner, mpi_communicator);
  //   }

  //   // 4. Finally, write the updated positions back into the triangulation
  //   for (unsigned int v = 0; v < n_vertices; ++v)
  //     triangulation.get_vertices()[v] = new_positions[v];
  // }


  template <int dim>
  void MMS<dim>::output_results(const unsigned int convergence_index,
                                const unsigned int time_step)
  {
    // Plot FE solution
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");
    for (unsigned int d = 0; d < dim; ++d)
      solution_names.push_back("displacement");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    for (unsigned int d = 0; d < dim; ++d)
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // Plot exact solution
    std::vector<std::string> exact_solution_names(dim, "exact_velocity");
    exact_solution_names.emplace_back("exact_pressure");
    for (unsigned int d = 0; d < dim; ++d)
      exact_solution_names.push_back("exact_displacement");

    VectorTools::interpolate(mapping,
                             dof_handler,
                             solution_fun,
                             exact_solution);
    data_out.add_data_vector(exact_solution,
                             exact_solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    // Partition
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(mapping, 2);

    std::string root = "./ns_mms_" + std::to_string(convergence_index) + "_" + std::to_string(Utilities::MPI::n_mpi_processes(mpi_communicator)) + "proc/";

    data_out.write_vtu_with_pvtu_record(
      root, "solution", time_step, mpi_communicator, 2);
  }

  template <int dim>
  void MMS<dim>::compute_errors(bool only_update_Linf)
  {
    const unsigned int n_active_cells = triangulation.n_active_cells();

    const ComponentSelectFunction<dim> pressure_mask(p_lower, n_components);
    const ComponentSelectFunction<dim> velocity_mask(std::make_pair(u_lower,
                                                                    u_upper),
                                                     n_components);
    const ComponentSelectFunction<dim> disp_position_mask(
      std::make_pair(x_lower, x_upper), n_components);

    Vector<double> cellwise_errors(n_active_cells);

    const QGaussSimplex<dim> quadrature(4);

    //
    // Linfty errors
    //
    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::Linfty_norm,
                                      &pressure_mask);
    const double p_linf =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::Linfty_norm);
    linf_error_p = std::max(linf_error_p, p_linf);

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::Linfty_norm,
                                      &velocity_mask);
    const double u_linf =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::Linfty_norm);
    linf_error_u = std::max(linf_error_u, u_linf);

    VectorTools::integrate_difference(mapping,
                                      dof_handler,
                                      present_solution,
                                      solution_fun,
                                      cellwise_errors,
                                      quadrature,
                                      VectorTools::Linfty_norm,
                                      &disp_position_mask);
    const double disp_linf =
      VectorTools::compute_global_error(triangulation,
                                        cellwise_errors,
                                        VectorTools::Linfty_norm);
    linf_error_disp = std::max(linf_error_disp, disp_linf);

    pcout << "linf_error_u = " << linf_error_u << std::endl;
    pcout << "linf_error_p = " << linf_error_p << std::endl;
    pcout << "linf_error_d = " << linf_error_disp << std::endl;

    {
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        present_solution,
                                        solution_fun,
                                        cellwise_errors,
                                        quadrature,
                                        VectorTools::L2_norm,
                                        &pressure_mask);
      const double p_l2_error =
        VectorTools::compute_global_error(triangulation,
                                          cellwise_errors,
                                          VectorTools::L2_norm);

      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        present_solution,
                                        solution_fun,
                                        cellwise_errors,
                                        quadrature,
                                        VectorTools::L2_norm,
                                        &velocity_mask);
      const double u_l2_error =
        VectorTools::compute_global_error(triangulation,
                                          cellwise_errors,
                                          VectorTools::L2_norm);

      // l2_err_u.push_back(u_l2_error);
      // l2_err_p.push_back(p_l2_error);

      pcout << "Errors: ||e_p||_L2 = " << p_l2_error
            << ",   ||e_u||_L2 = " << u_l2_error << " (at step)" << std::endl;
      pcout << "Errors: ||e_p||_Li = " << linf_error_p
            << ",   ||e_u||_Li = " << linf_error_u << std::endl;
    }

    if (!only_update_Linf)
    {
      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        present_solution,
                                        solution_fun,
                                        cellwise_errors,
                                        quadrature,
                                        VectorTools::L2_norm,
                                        &pressure_mask);
      const double p_l2_error =
        VectorTools::compute_global_error(triangulation,
                                          cellwise_errors,
                                          VectorTools::L2_norm);

      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        present_solution,
                                        solution_fun,
                                        cellwise_errors,
                                        quadrature,
                                        VectorTools::L2_norm,
                                        &velocity_mask);
      const double u_l2_error =
        VectorTools::compute_global_error(triangulation,
                                          cellwise_errors,
                                          VectorTools::L2_norm);

      VectorTools::integrate_difference(mapping,
                                        dof_handler,
                                        present_solution,
                                        solution_fun,
                                        cellwise_errors,
                                        quadrature,
                                        VectorTools::L2_norm,
                                        &disp_position_mask);
      const double disp_l2_error =
        VectorTools::compute_global_error(triangulation,
                                          cellwise_errors,
                                          VectorTools::L2_norm);

      pcout << "Errors: ||e_p||_L2 = " << p_l2_error
            << ",   ||e_u||_L2 = " << u_l2_error
            << ",   ||e_d||_L2 = " << disp_l2_error << " (at final time step)"
            << std::endl;
      pcout << "Errors: ||e_p||_Li = " << linf_error_p
            << ",   ||e_u||_Li = " << linf_error_u
            << ",   ||e_d||_Li = " << linf_error_disp << std::endl;

      convergence_table.add_value("nElm", n_active_cells);
      convergence_table.add_value("dt", param.dt);
      convergence_table.add_value("L2_u", u_l2_error);
      convergence_table.add_value("L2_p", p_l2_error);
      convergence_table.add_value("L2_d", disp_l2_error);
      convergence_table.add_value("Li_u", linf_error_u);
      convergence_table.add_value("Li_p", linf_error_p);
      convergence_table.add_value("Li_d", linf_error_disp);
    }
  }

  //
  // Reset solver between two solves with different meshes
  //
  template <int dim>
  void MMS<dim>::reset()
  {
    // Mesh
    triangulation.clear();

    // Errors
    linf_error_p    = 0.;
    linf_error_u    = 0.;
    linf_error_disp = 0.;
    l2_err_u.clear();
    l2_err_p.clear();
    l2_err_disp.clear();

    // Constrained pressure DOF
    constrained_pressure_dof = numbers::invalid_dof_index;
  }

  template <int dim>
  void MMS<dim>::run()
  {
    unsigned int iMesh            = 3;
    unsigned int nTimeConvergence = param.nConvergenceCycles;

    if (param.bdf_order == 0)
    {
      param.nTimeSteps = 1;
      nTimeConvergence = 1;
    }

    for (unsigned int iT = 1; iT <= nTimeConvergence;
         ++iT, this->param.dt /= 2., this->param.nTimeSteps *= 2.)
    {
      this->reset();

      this->param.prev_dt =
        this->param.dt; // Change this if using variable timestep
      this->set_bdf_coefficients(param.bdf_order);

      this->current_time = param.t0;
      this->solution_fun.set_time(current_time);
      this->source_term_fun.set_time(current_time);

      this->make_grid(iMesh);
      this->setup_system();
      this->create_zero_constraints();
      this->create_nonzero_constraints();
      this->create_sparsity_pattern();
      this->set_initial_condition();

      this->output_results(iT, 0);

      for (unsigned int i = 0; i < param.nTimeSteps; ++i)
      {
        this->current_time += param.dt;
        this->solution_fun.set_time(current_time);
        this->source_term_fun.set_time(current_time);

        pcout << std::endl
              << "Time step " << i + 1 << " - Advancing to t = " << current_time
              << '.' << std::endl;

        // this->move_mesh();
        this->create_nonzero_constraints();
        this->apply_nonzero_constraints();

        if (i == 0 && param.bdf_order == 2)
        {
          // For BDF2: set first step to exact solution
          this->set_exact_solution();

          // TODO: Or compute first step with BDF1 and smaller time step
        }
        else
        {
          this->solve_newton();
        }

        // this->set_exact_solution();

        this->move_mesh();
        this->compute_errors(true);
        this->output_results(iT, i + 1);

        if (param.bdf_order > 0)
        {
          for (unsigned int i = previous_solutions.size() - 1; i >= 1; --i)
            previous_solutions[i] = previous_solutions[i - 1];
          previous_solutions[0] = present_solution;
        }
      }

      this->compute_errors(false);

      for (unsigned int i = 0; i < l2_err_u.size(); ++i)
      {
        pcout << l2_err_u[i] << " - " << l2_err_p[i] << std::endl;
      }
    }

    convergence_table.evaluate_convergence_rates(
      "L2_u", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "L2_u", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "L2_p", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "L2_p", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "L2_d", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "L2_d", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "Li_u", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "Li_u", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "Li_p", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "Li_p", ConvergenceTable::reduction_rate_log2);
    convergence_table.evaluate_convergence_rates(
      "Li_d", ConvergenceTable::reduction_rate);
    convergence_table.evaluate_convergence_rates(
      "Li_d", ConvergenceTable::reduction_rate_log2);

    // Arrange convergence table
    convergence_table.set_precision("L2_u", 6);
    convergence_table.set_precision("L2_p", 6);
    convergence_table.set_precision("L2_d", 6);
    convergence_table.set_precision("Li_u", 6);
    convergence_table.set_precision("Li_p", 6);
    convergence_table.set_precision("Li_d", 6);
    convergence_table.set_scientific("L2_u", true);
    convergence_table.set_scientific("L2_p", true);
    convergence_table.set_scientific("L2_d", true);
    convergence_table.set_scientific("Li_u", true);
    convergence_table.set_scientific("Li_p", true);
    convergence_table.set_scientific("Li_d", true);

    pcout << std::endl;
    if (mpi_rank == 0)
      convergence_table.write_text(std::cout);
  }
} // namespace NS_MMS

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    using namespace NS_MMS;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    SimulationParameters param;

    param.velocity_degree     = 2;
    param.displacement_degree = 1;

    param.viscosity           = VISCOSITY;
    param.pseudo_solid_mu     = MU_PS;
    param.pseudo_solid_lambda = LAMBDA_PS;

    // Time integration
    param.bdf_order  = 1;
    param.t0         = 0.;
    param.dt         = 0.1;
    param.nTimeSteps = 10;
    param.t1         = param.dt * param.nTimeSteps;

    param.nConvergenceCycles = 1;

    MMS<2> problem2D(param);
    problem2D.run();

    // MMS<3> problem3D(param);
    // problem3D.run();
  }
  catch (const std::exception &exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;

    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }

  return 0;
}
