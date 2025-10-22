
#include <MMS.h>
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
#include <deal.II/numerics/matrix_tools.h>


#include <cmath>
#include <fstream>
#include <iostream>
#include <iomanip>   // std::setprecision

bool VERBOSE = false;

namespace fsi_coupled
{
  using namespace dealii;

  template <int dim>
  class SimulationParameters
  {
  public:
    unsigned int velocity_degree;
    unsigned int position_degree;

    // std::vector<std::string> position_boundary_names;
    std::vector<std::string> position_fixed_boundary_names;
    std::vector<std::string> position_moving_boundary_names;
    std::vector<std::string> Inlet_velocity_boundary_names;
    std::vector<std::string> noslip_velocity_boundary_names;
    std::vector<std::string> noflux_velocity_boundary_names;
    std::vector<std::string> position_semi_fixed_boundary_names;


    bool with_position_coupling;
    bool with_fixed_position;

    int write_result_it;

    double       viscosity;
    double       pseudo_solid_mu;
    double       pseudo_solid_lambda;
    double       spring_constant;

    unsigned int bdf_order;
    double       t0;
    double       t1;
    double       dt;
    double       prev_dt;
    unsigned int nTimeSteps;

    double newton_tolerance;

    // Geometry and flow parameters
    double H;
    double L;
    double D;
    double U;
    double Re;
    double rho;

    // Mesh file
    std::string mesh_file;

  public:
    SimulationParameters<dim>(){};
  };

  template <int dim>
  class InitialCondition : public Function<dim>
  {
  public:
    const unsigned int u_lower = 0;
    const unsigned int p_lower = dim;
    const unsigned int x_lower = dim + 1;
    
  public:
    InitialCondition(const unsigned int n_components)
      : Function<dim>(n_components)
    {}

    double initial_velocity(const Point<dim> &/*p*/, const unsigned int component) const
    {
      if(component == 0)
        return 1.;
      if(component == 1)
        return 0.;
      if(component == 2)
        return 0.;
      DEAL_II_ASSERT_UNREACHABLE();
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      values[p_lower] = 0.;
      for(unsigned int d = 0; d < dim; ++d)
      {
        values[u_lower + d] = this->initial_velocity(p, d);
        values[x_lower + d] = 0.;
      }
    }
  };

  template <int dim>
  class Inlet : public Function<dim>
  {
  public:
    // offsets (inchangÃ©s)
    const unsigned int u_lower = 0;
    const unsigned int p_lower = dim;
    const unsigned int x_lower = dim + 1;

    // paramÃ¨tres du profil d'entrÃ©e
    Inlet(const double        time,
          const unsigned int  n_components)
      : Function<dim>(n_components, time)
    {}

    double inlet_velocity(const Point<dim> &p,
                          const double       /*t*/,
                          const unsigned int component) const
    {
      // const double t = this->get_time();
      if constexpr (dim == 2)
      {
        // const double y = p[1];
        if (component == 0) // u_x
          return 1.;
        else                // u_y
          return 0.0;
      }
      else // dim == 3 : mÃªme idÃ©e, Ã©coulement en x, profil en y, u_z=0
      {
        // const double y = p[1];
        if (component == 0) // u_x
          return 0.;
        else                // u_y ou u_z
          return 0.0;
      }
    }

    virtual void vector_value(const Point<dim> &p,
                              Vector<double>   &values) const override
    {
      const double t = this->get_time();

      values[p_lower] = 0.0; // pression imposÃ©e Ã  0 Ã  lâ€™entrÃ©e (si souhaitÃ©)
      for (unsigned int d = 0; d < dim; ++d)
      {
        values[u_lower + d] = this->inlet_velocity(p, t, d);
        values[x_lower + d] = 0.0;
      }
    }
  };


  template <int dim>
  class FixedMeshPosition : public Function<dim>
  {
  public:
    // offsets
    static constexpr unsigned int u_lower = 0;
    static constexpr unsigned int p_lower = dim;
    static constexpr unsigned int x_lower = dim + 1;

    // inutile de passer x_lower : on le connaît à la compile
    explicit FixedMeshPosition(const unsigned int n_components)
      : Function<dim>(n_components)
    {}

    void vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      // const double t = this->get_time(); // <<< à l'intérieur du corps de la fonction

      // Ici, on impose une position "fixe" (ou pilotée) du maillage :
      // composante x : p[d], composantes suivantes : p[d] + 0.1 * t
      for (unsigned int d = 0; d < dim; ++d)
      {
        values[x_lower + d] = p[d];
      }
    }
  };


  template <int dim>
  class MovingMeshPosition : public Function<dim>
  {
  public:
    // offsets
    static constexpr unsigned int u_lower = 0;
    static constexpr unsigned int p_lower = dim;
    static constexpr unsigned int x_lower = dim + 1;

    // inutile de passer x_lower : on le connaît à la compile
    explicit MovingMeshPosition(const unsigned int n_components)
      : Function<dim>(n_components)
    {}

    void vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      const double t = this->get_time(); // <<< à l'intérieur du corps de la fonction

      // Ici, on impose une position "fixe" (ou pilotée) du maillage :
      // composante x : p[d], composantes suivantes : p[d] + 0.1 * t
      for (unsigned int d = 0; d < dim; ++d)
      {
        if (d == 0)
          values[x_lower + d] = p[d];
        else
          values[x_lower + d] = p[d] + 0.1 * t;
      }
    }
  };

  template <int dim>
  class ScratchData
  {
  public:
    const unsigned int n_components = 2 * dim + 1;
    const unsigned int u_lower      = 0;
    const unsigned int p_lower      = dim;
    const unsigned int x_lower      = dim + 1;

  public:
    ScratchData(const FESystem<dim>       &fe,
                const Quadrature<dim>     &cell_quadrature,
                const Mapping<dim>        &fixed_mapping,
                const Mapping<dim>        &mapping,
                const Quadrature<dim - 1> &face_quadrature,
                const unsigned int         dofs_per_cell,
                const unsigned int         boundary_id,
                const std::vector<double> &bdfCoeffs)
      : fe_values(mapping,
                  fe,
                  cell_quadrature,
                  update_values | update_gradients | update_quadrature_points |
                    update_JxW_values | update_jacobians |
                    update_inverse_jacobians)
      , fe_values_fixed_mapping(fixed_mapping,
                                fe,
                                cell_quadrature,
                                update_values | update_gradients |
                                  update_quadrature_points | update_JxW_values |
                                  update_jacobians | update_inverse_jacobians)
      , fe_face_values(mapping,
                      fe,
                      face_quadrature,
                      update_values | update_gradients |
                        update_quadrature_points | update_JxW_values |
                        update_jacobians | update_inverse_jacobians)
      , fe_face_values_fixed_mapping(fixed_mapping,
                                    fe,
                                    face_quadrature,
                                    update_values | update_gradients |
                                      update_quadrature_points |
                                      update_JxW_values | update_jacobians |
                                      update_inverse_jacobians)
      , n_q_points(cell_quadrature.size())
      // We assume that simplicial meshes with all tris or tets
      , n_faces((dim == 3) ? 4 : 3)
      , n_faces_q_points(face_quadrature.size())
      , dofs_per_cell(dofs_per_cell)
      , boundary_id(boundary_id)
      , bdfCoeffs(bdfCoeffs)
    {
      this->allocate();
    }

    void allocate();

  private:
    template <typename VectorType>
    void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
                const VectorType                   &current_solution,
                const std::vector<LA::MPI::Vector> &previous_solutions)
    {
      active_fe_values->reinit(cell);

      for (const unsigned int i : active_fe_values->dof_indices())
        components[i] =
          active_fe_values->get_fe().system_to_component_index(i).first;

      const FEValuesExtractors::Vector velocity(u_lower);
      const FEValuesExtractors::Scalar pressure(p_lower);
      const FEValuesExtractors::Vector position(x_lower);

      if constexpr (std::is_same<VectorType, LA::MPI::Vector>::value)
      {
        (*active_fe_values)[velocity].get_function_values(
          current_solution, present_velocity_values);
        (*active_fe_values)[velocity].get_function_gradients(
          current_solution, present_velocity_gradients);
        (*active_fe_values)[pressure].get_function_values(
          current_solution, present_pressure_values);
        (*active_fe_values)[position].get_function_values(
          current_solution, present_position_values);
        (*active_fe_values)[position].get_function_gradients(
          current_solution, present_position_gradients);
      }
      else if constexpr (std::is_same<VectorType, std::vector<double>>::value)
      {
        (*active_fe_values)[velocity].get_function_values_from_local_dof_values(
          current_solution, present_velocity_values);
        (*active_fe_values)[velocity]
          .get_function_gradients_from_local_dof_values(
            current_solution, present_velocity_gradients);
        (*active_fe_values)[pressure].get_function_values_from_local_dof_values(
          current_solution, present_pressure_values);
        (*active_fe_values)[position].get_function_values_from_local_dof_values(
          current_solution, present_position_values);
        (*active_fe_values)[position]
          .get_function_gradients_from_local_dof_values(
            current_solution, present_position_gradients);
      }
      else
      {
        static_assert(false,"reinit expects LA::MPI::Vector or std::vector<double>");
      }

      // Previous solutions (pour BDF sur u et x uniquement)
      for (unsigned int i = 0; i < previous_solutions.size(); ++i)
      {
        (*active_fe_values)[velocity].get_function_values(
          previous_solutions[i], previous_velocity_values[i]);
        (*active_fe_values)[position].get_function_values(
          previous_solutions[i], previous_position_values[i]);
      }

      // Current mesh velocity from displacement
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        present_mesh_velocity_values[q] =
          bdfCoeffs[0] * present_position_values[q];
        for (unsigned int iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
        {
          present_mesh_velocity_values[q] +=
            bdfCoeffs[iBDF] * previous_position_values[iBDF - 1][q];
        }
      }

      // Remplissage des fonctions de forme (volume)
      for (unsigned int q = 0; q < n_q_points; ++q)
      {
        JxW[q] = active_fe_values->JxW(q);
        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_u[q][k]      = (*active_fe_values)[velocity].value(k, q);
          grad_phi_u[q][k] = (*active_fe_values)[velocity].gradient(k, q);
          div_phi_u[q][k]  = (*active_fe_values)[velocity].divergence(k, q);

          phi_p[q][k]      = (*active_fe_values)[pressure].value(k, q);

          phi_x[q][k]      = (*active_fe_values)[position].value(k, q);
          grad_phi_x[q][k] = (*active_fe_values)[position].gradient(k, q);
          div_phi_x[q][k]  = (*active_fe_values)[position].divergence(k, q);
        }
      }

      //
      // Face values and shape functions,
      // for the faces touching the prescribed boundary_id
      //
      for (const auto i_face : cell->face_indices())
      {
        const auto &face = cell->face(i_face);

        if (!(face->at_boundary() && face->boundary_id() == boundary_id))
          continue;

        active_fe_face_values->reinit(cell, face);
        fe_face_values_fixed_mapping.reinit(cell, face);

        if constexpr (std::is_same<VectorType, LA::MPI::Vector>::value)
        {
          (*active_fe_face_values)[velocity].get_function_values(
            current_solution, present_face_velocity_values[i_face]);
          (*active_fe_face_values)[position].get_function_values(
            current_solution, present_face_position_values[i_face]);
          (*active_fe_face_values)[position].get_function_gradients(
            current_solution, present_face_position_gradient[i_face]);
        }
        else if constexpr (std::is_same<VectorType, std::vector<double>>::value)
        {
          (*active_fe_face_values)[velocity]
            .get_function_values_from_local_dof_values(
              current_solution, present_face_velocity_values[i_face]);
          (*active_fe_face_values)[position]
            .get_function_values_from_local_dof_values(
              current_solution, present_face_position_values[i_face]);
          (*active_fe_face_values)[position]
            .get_function_gradients_from_local_dof_values(
              current_solution, present_face_position_gradient[i_face]);
        }
        else
        {
          static_assert(
            false, "reinit expects LA::MPI::Vector or std::vector<double>");
        }

        for (unsigned int i = 0; i < previous_solutions.size(); ++i)
        {
          (*active_fe_face_values)[position].get_function_values(
            previous_solutions[i], previous_face_position_values[i_face][i]);
        }

        for (unsigned int q = 0; q < n_faces_q_points; ++q)
        {
          face_JxW[i_face][q]       = active_fe_face_values->JxW(q);
          face_jacobians[i_face][q] = active_fe_face_values->jacobian(q);
          const Tensor<2, dim> J    = active_fe_face_values->jacobian(q);

          if constexpr (dim == 2)
          {
            switch (i_face)
            {
              case 0:
                dxsids[0] = 1.;  dxsids[1] = 0.;
                dxsids_array[0][0] = 1.; dxsids_array[0][1] = 0.;
                break;
              case 1:
                dxsids[0] = -1.; dxsids[1] = 1.;
                dxsids_array[0][0] = -1.; dxsids_array[0][1] = 1.;
                break;
              case 2:
                dxsids[0] = 0.;  dxsids[1] = -1.;
                dxsids_array[0][0] = 0.;  dxsids_array[0][1] = -1.;
                break;
              default:
                DEAL_II_ASSERT_UNREACHABLE();
            }
          }
          else
          {
            switch (i_face)
            {
              // Using dealii's face ordering
              case 3: // Opposite to v0
                dxsids_array[0][0] = -1.; dxsids_array[1][0] = -1.;
                dxsids_array[0][1] =  1.; dxsids_array[1][1] =  0.;
                dxsids_array[0][2] =  0.; dxsids_array[1][2] =  1.;
                break;
              case 2: // Opposite to v1
                dxsids_array[0][0] = 0.; dxsids_array[1][0] = 0.;
                dxsids_array[0][1] = 1.; dxsids_array[1][1] = 0.;
                dxsids_array[0][2] = 0.; dxsids_array[1][2] = 1.;
                break;
              case 1: // Opposite to v2
                dxsids_array[0][0] = 1.; dxsids_array[1][0] = 0.;
                dxsids_array[0][1] = 0.; dxsids_array[1][1] = 0.;
                dxsids_array[0][2] = 0.; dxsids_array[1][2] = 1.;
                break;
              case 0: // Opposite to v3
                dxsids_array[0][0] = 1.; dxsids_array[1][0] = 0.;
                dxsids_array[0][1] = 0.; dxsids_array[1][1] = 1.;
                dxsids_array[0][2] = 0.; dxsids_array[1][2] = 0.;
                break;
              default:
                DEAL_II_ASSERT_UNREACHABLE();
            }
          }

          face_dXds[i_face][q] = face_jacobians[i_face][q] * dxsids;

          Tensor<2, dim - 1> G;
          G = 0;
          for (unsigned int di = 0; di < dim - 1; ++di)
            for (unsigned int dj = 0; dj < dim - 1; ++dj)
              for (unsigned int im = 0; im < dim; ++im)
                for (unsigned int in = 0; in < dim; ++in)
                  for (unsigned int ip = 0; ip < dim; ++ip)
                    G[di][dj] += dxsids_array[di][im] * J[in][im] * J[in][ip] *
                                dxsids_array[dj][ip];
          face_G[i_face][q] = G;
          const Tensor<2, dim - 1> G_inverse = invert(G);

          Tensor<2, dim - 1> res;

          for (unsigned int k = 0; k < dofs_per_cell; ++k)
          {
            phi_u_face[i_face][q][k] =
              (*active_fe_face_values)[velocity].value(k, q);
            phi_x_face[i_face][q][k] =
              (*active_fe_face_values)[position].value(k, q);
            grad_phi_x_face[i_face][q][k] =
              (*active_fe_face_values)[position].gradient(k, q);

            const auto &grad_phi_x = grad_phi_x_face[i_face][q][k];

            Tensor<2, dim> A =
              transpose(J) *
              (transpose(present_face_position_gradient[i_face][q]) *
                grad_phi_x +
              transpose(grad_phi_x) *
                present_face_position_gradient[i_face][q]) *
              J;

            res = 0;
            for (unsigned int di = 0; di < dim - 1; ++di)
              for (unsigned int dj = 0; dj < dim - 1; ++dj)
                for (unsigned int im = 0; im < dim - 1; ++im)
                  for (unsigned int in = 0; in < dim; ++in)
                    for (unsigned int io = 0; io < dim; ++io)
                      res[di][dj] += G_inverse[di][im] * dxsids_array[im][in] *
                                    A[in][io] * dxsids_array[dj][io];

            // delta_dx pour le terme pseudo-solide (inchangé)
            delta_dx[i_face][q][k] = 0.5 * trace(res);
          }
          // Face mesh velocity
          present_face_mesh_velocity_values[i_face][q] =
            bdfCoeffs[0] * present_face_position_values[i_face][q];
          for (unsigned int iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
          {
            present_face_mesh_velocity_values[i_face][q] +=
              bdfCoeffs[iBDF] *
              previous_face_position_values[i_face][iBDF - 1][q];
          }
        }
      }
    }

  public:
    template <typename VectorType>
    void reinit_current_mapping(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const VectorType                                     &current_solution,
      const std::vector<LA::MPI::Vector>                   &previous_solutions)
    {
      active_fe_values      = &fe_values;
      active_fe_face_values = &fe_face_values;
      this->reinit(cell,
                  current_solution,
                  previous_solutions);
    }

    template <typename VectorType>
    void reinit_fixed_mapping(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      const VectorType                                     &current_solution,
      const std::vector<LA::MPI::Vector>                   &previous_solutions)
    {
      active_fe_values      = &fe_values_fixed_mapping;
      active_fe_face_values = &fe_face_values_fixed_mapping;
      this->reinit(cell,
                  current_solution,
                  previous_solutions);
    }

    const FEValues<dim> &get_current_fe_values() const { return fe_values; }
    const FEValues<dim> &get_fixed_fe_values() const
    {
      return fe_values_fixed_mapping;
    }

  public:
    FEValues<dim> *active_fe_values;
    FEValues<dim>  fe_values;
    FEValues<dim>  fe_values_fixed_mapping;

    FEFaceValues<dim> *active_fe_face_values;
    FEFaceValues<dim>  fe_face_values;
    FEFaceValues<dim>  fe_face_values_fixed_mapping;

    const unsigned int n_q_points;
    const unsigned int n_faces;
    const unsigned int n_faces_q_points;
    const unsigned int dofs_per_cell;

    const unsigned int boundary_id;

    const std::vector<double> &bdfCoeffs;

    std::vector<double>              JxW;
    std::vector<std::vector<double>> face_JxW;

    // Jacobian matrix on face
    std::vector<std::vector<Tensor<2, dim>>> face_jacobians;

    // If dim = 2, face_dXds is the variation of an edge position w.r.t.
    // the 1-dimensional reference coordinate s. That is,
    // this is the non-unit tangent vector to the edge.
      Tensor<1, dim>                           dxsids;
    std::vector<std::vector<Tensor<1, dim>>> face_dXds;

    // The reference jacobians partial xsi_dim/partial xsi_(dim-1)
    std::array<Tensor<1, dim>, dim - 1>          dxsids_array;
    std::vector<std::vector<Tensor<2, dim - 1>>> face_G;

    // At face x quad node x phi_position_j
    std::vector<std::vector<std::vector<double>>> delta_dx;

    std::vector<unsigned int> components;

    // Current and previous values and gradients for each quad node
    std::vector<Tensor<1, dim>>              present_velocity_values;
    std::vector<Tensor<2, dim>>              present_velocity_gradients;
    std::vector<double>                      present_pressure_values;
    std::vector<std::vector<Tensor<1, dim>>> previous_velocity_values;

    std::vector<Tensor<1, dim>>              present_position_values;
    std::vector<Tensor<2, dim>>              present_position_gradients;
    std::vector<std::vector<Tensor<1, dim>>> previous_position_values;
    std::vector<Tensor<1, dim>>              present_mesh_velocity_values;

    // Current values on faces
    std::vector<std::vector<Tensor<1, dim>>> present_face_velocity_values;
    std::vector<std::vector<Tensor<1, dim>>> present_face_position_values;
    std::vector<std::vector<Tensor<2, dim>>> present_face_position_gradient;
    std::vector<std::vector<Tensor<1, dim>>> present_face_mesh_velocity_values;
    std::vector<std::vector<std::vector<Tensor<1, dim>>>> previous_face_position_values;


    // Source term on cell
    std::vector<Vector<double>>      source_term_full; // n_components
    std::vector<Tensor<1, dim>>      source_term_velocity;
    std::vector<double>              source_term_pressure;
    std::vector<Tensor<1, dim>>      source_term_position;

    // Gradient of source term
    std::vector<std::vector<Tensor<1, dim>>> grad_source_term_full;
    std::vector<Tensor<2, dim>>              grad_source_velocity;
    std::vector<Tensor<1, dim>>              grad_source_pressure;

    // Shape functions and gradients for each quad node and each dof
    std::vector<std::vector<Tensor<1, dim>>> phi_u;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_u;
    std::vector<std::vector<double>>         div_phi_u;
    std::vector<std::vector<double>>         phi_p;
    std::vector<std::vector<Tensor<1, dim>>> phi_x;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_x;
    std::vector<std::vector<double>>         div_phi_x;


    // Shape functions on faces for relevant faces, each quad node and each dof
    std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_u_face;
    std::vector<std::vector<std::vector<Tensor<1, dim>>>> phi_x_face;
    std::vector<std::vector<std::vector<Tensor<2, dim>>>> grad_phi_x_face;

    // Source term on faces
    std::vector<Vector<double>>              face_source_velocity;
    std::vector<Vector<double>>              face_source_mesh_velocity;
    std::vector<std::vector<Tensor<1, dim>>> face_grad_source_term_full;

    // Noslip BC helper buffers (inchangés)
    std::vector<std::vector<Vector<double>>> solution_on_noslip_bc_full;
    std::vector<std::vector<Tensor<1, dim>>> prescribed_velocity_noslip_bc;

    std::vector<std::vector<std::vector<Tensor<1, dim>>>>
                                            grad_solution_on_noslip_bc_full;
    std::vector<std::vector<Tensor<2, dim>>> grad_solution_velocity;
  };

  template <int dim>
  void ScratchData<dim>::allocate()
  {
    components.resize(dofs_per_cell);

    present_velocity_values.resize(n_q_points);
    present_velocity_gradients.resize(n_q_points);
    present_pressure_values.resize(n_q_points);
    present_position_values.resize(n_q_points);
    present_position_gradients.resize(n_q_points);
    present_mesh_velocity_values.resize(n_q_points);
    present_face_mesh_velocity_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

    present_face_velocity_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    present_face_position_values.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    present_face_position_gradient.resize(
      n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));

    source_term_full.resize(n_q_points, Vector<double>(n_components));
    source_term_velocity.resize(n_q_points);
    source_term_pressure.resize(n_q_points);
    source_term_position.resize(n_q_points);

    grad_source_term_full.resize(n_q_points,
                                std::vector<Tensor<1, dim>>(n_components));
    grad_source_velocity.resize(n_q_points);
    grad_source_pressure.resize(n_q_points);

    // BDF (historique pour u et x)
    previous_velocity_values.resize(bdfCoeffs.size() - 1,
                                    std::vector<Tensor<1, dim>>(n_q_points));
    previous_position_values.resize(bdfCoeffs.size() - 1,
                                    std::vector<Tensor<1, dim>>(n_q_points));
    previous_face_position_values.resize(
      n_faces,
      std::vector<std::vector<Tensor<1, dim>>>(
        bdfCoeffs.size() - 1, std::vector<Tensor<1, dim>>(n_faces_q_points)));

    phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_u.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_u.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_p.resize(n_q_points, std::vector<double>(dofs_per_cell));

    phi_x.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_x.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_x.resize(n_q_points, std::vector<double>(dofs_per_cell));


    phi_u_face.resize(n_faces,
                      std::vector<std::vector<Tensor<1, dim>>>(
                        n_faces_q_points,
                        std::vector<Tensor<1, dim>>(dofs_per_cell)));

    phi_x_face.resize(n_faces,
                      std::vector<std::vector<Tensor<1, dim>>>(
                        n_faces_q_points,
                        std::vector<Tensor<1, dim>>(dofs_per_cell)));

    grad_phi_x_face.resize(n_faces,
                          std::vector<std::vector<Tensor<2, dim>>>(
                            n_faces_q_points,
                            std::vector<Tensor<2, dim>>(dofs_per_cell)));

    solution_on_noslip_bc_full.resize(
      n_faces,
      std::vector<Vector<double>>(n_faces_q_points,
                                  Vector<double>(n_components)));
    prescribed_velocity_noslip_bc.resize(
      n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));

    grad_solution_on_noslip_bc_full.resize(
      n_faces,
      std::vector<std::vector<Tensor<1, dim>>>(
        n_faces_q_points, std::vector<Tensor<1, dim>>(n_components)));
    grad_solution_velocity.resize(
      n_faces, std::vector<Tensor<2, dim>>(n_faces_q_points));

    face_source_velocity.resize(n_faces_q_points, Vector<double>(n_components));
    face_source_mesh_velocity.resize(n_faces_q_points, Vector<double>(n_components));
    face_grad_source_term_full.resize(n_faces_q_points,
                                      std::vector<Tensor<1, dim>>(n_components));

    JxW.resize(n_q_points);
    face_JxW.resize(n_faces, std::vector<double>(n_faces_q_points));
    face_jacobians.resize(n_faces,
                          std::vector<Tensor<2, dim>>(n_faces_q_points));
    face_dXds.resize(n_faces, std::vector<Tensor<1, dim>>(n_faces_q_points));
    face_G.resize(n_faces, std::vector<Tensor<2, dim - 1>>(n_faces_q_points));
    delta_dx.resize(n_faces,
                    std::vector<std::vector<double>>(
                      n_faces_q_points, std::vector<double>(dofs_per_cell)));
  }


  template <int dim>
  class FSI
  {
  public:
    FSI(const SimulationParameters<dim>         &param);

    void run();

  private:
    void set_bdf_coefficients(const unsigned int order);
    void make_grid();
    void setup_system();
    void create_nonzero_constraints();
    void add_no_flux_constraints(AffineConstraints<double> &constraints);
    void add_semi_fixed_constraints(AffineConstraints<double> &constraints);
    void apply_row_replacement_u_equals_w(const types::boundary_id boundary_id);
    // void enforce_u_equals_w_manual(const types::boundary_id boundary_id,
    //                            LA::MPI::Vector &vector_to_modify);

    void create_sparsity_pattern();
    void set_initial_condition();
    void update_boundary_conditions();

    void assemble_matrix();
    void assemble_local_matrix(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData<dim>                                     &scratchData,
      LA::MPI::Vector                                      &current_solution,
      std::vector<LA::MPI::Vector>                         &previous_solutions,
      std::vector<types::global_dof_index>                 &local_dof_indices,
      FullMatrix<double>                                   &local_matrix,
      bool                                                  distribute);
    void assemble_local_matrix_fd(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData<dim>                                     &scratchData,
      LA::MPI::Vector                                      &current_solution,
      std::vector<LA::MPI::Vector>                         &previous_solutions,
      std::vector<types::global_dof_index>                 &local_dof_indices,
      FullMatrix<double>                                   &local_matrix,
      Vector<double>                                       &ref_local_rhs,
      Vector<double>                                       &perturbed_local_rhs,
      std::vector<double>                                  &cell_dof_values);
    void assemble_local_matrix_pseudo_solid(
      const typename DoFHandler<dim>::active_cell_iterator &cell,
      ScratchData<dim>                                     &scratchData,
      LA::MPI::Vector                                      &current_solution,
      std::vector<LA::MPI::Vector>                         &previous_solutions,
      std::vector<types::global_dof_index>                 &local_dof_indices,
      FullMatrix<double>                                   &local_matrix,
      bool                                                  distribute);

    void assemble_rhs(LA::MPI::Vector *raw_out = nullptr);
    void assemble_local_rhs(
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
    void assemble_local_rhs_pseudo_solid(
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

    void solve_direct();
    void solve_newton();
    void output_results(const unsigned int time_step,
                        const bool         write_newton_iteration = false,
                        const unsigned int newton_step = 0);
    void compare_mesh_fluid_velocity(const unsigned int boundary_id);
    void compute_drag_lift_and_write(const unsigned int boundary_id,const double actual_time);
    void compute_implicit_boundary_forces(const unsigned int boundary_id,const double actual_time);
    void dump_boundary_velocity_debug(const types::boundary_id boundary_id,
                                              const std::string &label,
                                              const bool after_solve); 



    SimulationParameters<dim> param;

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
    bool is_position(const unsigned int component) const
    {
      return x_lower <= component && component < x_upper;
    }

  public:
    QSimplex<dim>     quadrature;
    QSimplex<dim - 1> face_quadrature;

    parallel::fullydistributed::Triangulation<dim> triangulation;
    std::unique_ptr<Mapping<dim>>                  fixed_mapping;
    std::unique_ptr<Mapping<dim>>                  mapping;

    // Description of the .msh mesh entities
    std::map<unsigned int, std::string> mesh_domains_tag2name;
    std::map<std::string, unsigned int> mesh_domains_name2tag;

    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> nonzero_constraints;
    AffineConstraints<double> velocity_constraints;

    // Position-lambda constraints on the cylinder
    AffineConstraints<double> position_constraints;
    std::map<types::global_dof_index, Point<dim>> initial_positions;

    // The id of the boundary where noslip Dirichlet BC are prescribed
    // for the velocity
    unsigned int noslip_bc_boundary_id;

    LA::MPI::SparseMatrix system_matrix;

    // With ghosts (read only)
    LA::MPI::Vector present_solution;
    LA::MPI::Vector evaluation_point;

    // Without ghosts (owned)
    LA::MPI::Vector local_evaluation_point;
    LA::MPI::Vector newton_update;
    LA::MPI::Vector system_rhs;

    LA::MPI::Vector local_mesh_velocity;
    LA::MPI::Vector mesh_velocity;

    LA::MPI::Vector exact_solution;
    LA::MPI::Vector reactions_raw_;
    double Fx_imp_last = 0.0, Fy_imp_last = 0.0;


    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    std::vector<Point<dim>> initial_mesh_position;
    IndexSet                pos_dof_indices;

    std::vector<LA::MPI::Vector> previous_solutions;
    std::vector<double>          bdfCoeffs;

    double current_time;
    unsigned int current_time_step;

    InitialCondition<dim>   initial_condition_fun;
    Inlet<dim>              inlet_fun;
    FixedMeshPosition<dim>  fixed_mesh_position_fun;
    MovingMeshPosition<dim> moving_mesh_position_fun;
  };

  template <int dim>
  FSI<dim>::FSI(const SimulationParameters<dim>         &param)
    : param(param)
    , mpi_communicator(MPI_COMM_WORLD)
    , mpi_rank(Utilities::MPI::this_mpi_process(mpi_communicator))
    , fe(FE_SimplexP<dim>(param.velocity_degree), // Velocity
         dim,
         FE_SimplexP<dim>(param.velocity_degree - 1), // Pressure
         1,
         FE_SimplexP<dim>(param.position_degree), // Position
         dim)
    , quadrature(QGaussSimplex<dim>(4))
    , face_quadrature(QGaussSimplex<dim - 1>(4))
    , triangulation(mpi_communicator)
    , fixed_mapping(new MappingFE<dim>(FE_SimplexP<dim>(1)))
    , dof_handler(triangulation)
    , pcout(std::cout, (mpi_rank == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::never, // TimerOutput::summary,
                      TimerOutput::wall_times)
    , current_time(param.t0)
    , initial_condition_fun(InitialCondition<dim>(n_components))
    , inlet_fun(Inlet<dim>(current_time, n_components))
    , fixed_mesh_position_fun(FixedMeshPosition<dim>(n_components))
    , moving_mesh_position_fun(MovingMeshPosition<dim>(n_components))

  {
  }

  template <int dim>
  void FSI<dim>::set_bdf_coefficients(const unsigned int order)
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
  void FSI<dim>::make_grid()
  {
    Triangulation<dim> serial_tria;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(serial_tria);

    std::string meshFile = "";

    meshFile = param.mesh_file;

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
      if (VERBOSE)
      {
        std::cout << "Mesh info:" << std::endl
                  << " dimension: " << dim << std::endl
                  << " no. of cells: " << serial_tria.n_active_cells()
                  << std::endl;
      }

      std::map<types::boundary_id, unsigned int> boundary_count;
      for (const auto &face : serial_tria.active_face_iterators())
        if (face->at_boundary())
          boundary_count[face->boundary_id()]++;

      if (VERBOSE)
      {
        std::cout << " boundary indicators: ";
        for (const std::pair<const types::boundary_id, unsigned int> &pair :
             boundary_count)
        {
          std::cout << pair.first << '(' << pair.second << " times) ";
        }
        std::cout << std::endl;
      }

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

      if (VERBOSE)
      {
        for (const auto &[id, name] : mesh_domains_tag2name)
          std::cout << "ID " << id << " -> " << name << "\n";
      }
    }

    std::vector<std::string> all_boundaries;
    for (auto str : param.position_fixed_boundary_names)
      all_boundaries.push_back(str);
    for (auto str : param.position_moving_boundary_names)
      all_boundaries.push_back(str);
    for (auto str : param.Inlet_velocity_boundary_names)
      all_boundaries.push_back(str);
    for (auto str : param.noslip_velocity_boundary_names)
      all_boundaries.push_back(str);
    for (auto str : param.noflux_velocity_boundary_names)
      all_boundaries.push_back(str);

    // Check that specified boundaries exist
    for (auto str : all_boundaries)
    {
      if (mesh_domains_name2tag.count(str) == 0)
      {
        throw std::runtime_error("A boundary condition should be prescribed "
                                 "on the boundary named \"" +
                                 str +
                                 "\", but no physical entity with this name "
                                 "was read from the mesh file.");
      }
    }

    if (param.noslip_velocity_boundary_names.size() > 1)
      throw std::runtime_error(
        "Only considering a single boundary for noslip velocity BC for now.");

    for (auto str : param.noslip_velocity_boundary_names)
    {
      noslip_bc_boundary_id = mesh_domains_name2tag.at(str);
    }
  }

  template <int dim>
  void FSI<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);
    if (VERBOSE)
    {
      pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
            << std::endl;
    }

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

    local_mesh_velocity.reinit(locally_owned_dofs, mpi_communicator);
    mesh_velocity.reinit(locally_owned_dofs,
                         locally_relevant_dofs,
                         mpi_communicator);

    exact_solution.reinit(locally_owned_dofs, mpi_communicator);

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

    // Mesh position
    // Initialize directly from the triangulation
    // The parallel vector storing the mesh position is local_evaluation_point,
    // because this is the one to modify when computing finite differences.
    const FEValuesExtractors::Vector position(x_lower);
    VectorTools::get_position_vector(*fixed_mapping,
                                     dof_handler,
                                     local_evaluation_point,
                                     fe.component_mask(position));
    local_evaluation_point.compress(VectorOperation::insert);
    evaluation_point = local_evaluation_point;
    local_evaluation_point.compress(VectorOperation::insert);
    

    // Set mapping as a solution-dependent mapping
    mapping = std::make_unique<MappingFEField<dim, dim, LA::MPI::Vector>>(
      dof_handler, evaluation_point, fe.component_mask(position));
  }

  template <int dim>
  void FSI<dim>::create_sparsity_pattern()
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
  void FSI<dim>::set_initial_condition()
  {
    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    // Update mesh position *BEFORE* evaluating scalar field
    // with moving mapping (-:

    // Set mesh position with fixed mapping
    VectorTools::interpolate(*fixed_mapping,
                             dof_handler,
                             fixed_mesh_position_fun,
                             newton_update,
                             fe.component_mask(position));

    evaluation_point = newton_update;

    // Set velocity with moving mapping
    VectorTools::interpolate(*mapping,
                             dof_handler,
                             initial_condition_fun,
                             newton_update,
                             fe.component_mask(velocity));

    // Apply non-homogeneous Dirichlet BC and set as current solution
    nonzero_constraints.distribute(newton_update);
    evaluation_point = newton_update;
    present_solution = newton_update;

    // Dirty copy of the initial condition for BDF2 for now (-:
    for (auto &sol : previous_solutions)
      sol = present_solution;
  }


  template <int dim>
  void FSI<dim>::add_no_flux_constraints(AffineConstraints<double> &constraints)
  {
    std::set<types::boundary_id> no_normal_flux_boundaries;
    for (auto str : param.noflux_velocity_boundary_names)
    {
      no_normal_flux_boundaries.insert(this->mesh_domains_name2tag.at(str));
    }
    VectorTools::compute_no_normal_flux_constraints(
      dof_handler, u_lower, no_normal_flux_boundaries, constraints, *mapping);
  }

  template <int dim>
  void FSI<dim>::add_semi_fixed_constraints(AffineConstraints<double> &constraints)
  {
    std::set<types::boundary_id> semi_fixed_boundaries;
    for (auto str : param.position_semi_fixed_boundary_names)
      semi_fixed_boundaries.insert(this->mesh_domains_name2tag.at(str));

    // (δx)·n_ref = 0 sur ces frontières
    VectorTools::compute_no_normal_flux_constraints(
        dof_handler, x_lower, semi_fixed_boundaries, constraints, *fixed_mapping);
  }

  template <int dim>
  void FSI<dim>::apply_row_replacement_u_equals_w(const types::boundary_id boundary_id)
  {
    using number = double;

    const FEValuesExtractors::Vector velocity(u_lower);
    const ComponentMask vel_mask = fe.component_mask(velocity);

    // 0) Ghosts à jour
    evaluation_point.update_ghost_values();
    for (auto &v : previous_solutions) v.update_ghost_values();

    // 1) Ensemble *fiable* des DDL de vitesse sur la frontière demandée
    std::set<types::boundary_id> ids = { boundary_id };
    const IndexSet bdry_vel_is =
        DoFTools::extract_boundary_dofs(dof_handler, vel_mask, ids);

    // 2) Calcul de δ(u)=w−u^k uniquement pour ces DDL-là
    std::map<types::global_dof_index, number> boundary_values_delta;
    std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());

    for (const auto &cell : dof_handler.active_cell_iterators() |
                            IteratorFilters::LocallyOwnedCell())
    {
      for (unsigned int f = 0; f < cell->n_faces(); ++f)
      {
        const auto &face = cell->face(f);
        if (!(face->at_boundary() && face->boundary_id() == boundary_id)) continue;

        cell->get_dof_indices(local_dof_indices);

        for (unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
        {
          if (!fe.has_support_on_face(j, f)) continue;

          const unsigned comp = fe.system_to_component_index(j).first;
          if (!(u_lower <= comp && comp < u_upper)) continue; // seulement la vitesse

          const unsigned d = comp - u_lower;
          const unsigned s = fe.system_to_component_index(j).second;
          const unsigned j_pos = fe.component_to_system_index(x_lower + d, s);

          const auto gi_u = local_dof_indices[j];
          const auto gi_x = local_dof_indices[j_pos];

          // *** Filtre crucial : n'imposer que les vrais DDL de frontière ***
          if (!bdry_vel_is.is_element(gi_u)) continue;
          if (!locally_owned_dofs.is_element(gi_u)) continue;

          number w = bdfCoeffs[0] * evaluation_point[gi_x];
          for (unsigned ib = 1; ib < bdfCoeffs.size(); ++ib)
            w += bdfCoeffs[ib] * previous_solutions[ib - 1][gi_x];

          const number u_k   = evaluation_point[gi_u];
          const number delta = w - u_k;

          boundary_values_delta[gi_u] = delta;
        }
      }
    }

    // 3) Vecteur δ pour l’ajustement RHS lors de l’élimination de colonnes
    LA::MPI::Vector delta_bc(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    delta_bc = 0.0;
    {
      using ST = typename LA::MPI::Vector::size_type;
      std::vector<ST> idx; idx.reserve(boundary_values_delta.size());
      std::vector<number> val; val.reserve(boundary_values_delta.size());
      for (const auto &kv : boundary_values_delta)
        if (locally_relevant_dofs.is_element(kv.first)) {
          idx.push_back(static_cast<ST>(kv.first));
          val.push_back(kv.second);
        }
      if (!idx.empty()) delta_bc.set(idx, val);
    }
    delta_bc.compress(VectorOperation::insert);

    // 4) Colonnes -> zéro + ajustement du RHS
    MatrixTools::apply_boundary_values(boundary_values_delta,
                                      system_matrix,
                                      delta_bc,
                                      system_rhs,
                                      /*eliminate_columns=*/true);

    // 5) Lignes -> identité et RHS(i)=δ (sans clear_row !)
    MatrixTools::apply_boundary_values(boundary_values_delta,
                                      system_matrix,
                                      system_rhs, // dummy
                                      system_rhs,
                                      /*eliminate_columns=*/false);

    system_matrix.compress(VectorOperation::insert);
    system_rhs.compress(VectorOperation::insert);
  }



  // template <int dim>
  // void FSI<dim>::enforce_u_equals_w_manual(const types::boundary_id boundary_id,
  //                                         LA::MPI::Vector &vector_to_modify)
  // {
  //   // 0) Raccourcis
  //   const FEValuesExtractors::Vector velocity(u_lower);
  //   const FEValuesExtractors::Vector position(x_lower);

  //   // 1) Indices des DOFs de VITESSE situés sur la frontière target (boundary_id)
  //   const ComponentMask vel_mask = fe.component_mask(velocity);
  //   std::set<types::boundary_id> ids = { boundary_id };
  //   const IndexSet bdry_vel_is =
  //       DoFTools::extract_boundary_dofs(dof_handler, vel_mask, ids);

  //   // 2) Points de support pour chaque DOF (mêmes points pour toutes composantes
  //   //    dans les éléments Lagrangiens comme FE_SimplexP)
  //   //    -> on travaille avec le mapping COURANT pour coller à la géométrie actuelle.
  //   std::vector<Point<dim>> support_points(dof_handler.n_dofs());
  //   DoFTools::map_dofs_to_support_points(*fixed_mapping, dof_handler, support_points);

  //   // 3) Buffers pour évaluer la position x^{n-i}(p) aux points (on évalue tout le vecteur)
  //   //    On va appeler VectorTools::point_value(*fixed_mapping, ...) pour la position.
  //   //    (Tu utilises déjà *fixed_mapping pour le champ x, on reste cohérent.)
  //   Vector<double> vals_now(n_components);
  //   std::vector<Vector<double>> vals_prev(previous_solutions.size(),
  //                                         Vector<double>(n_components));

  //   // 4) Boucle DOFs frontière (on ne touche QU’AUX DOFs possédés localement)
  //   for (auto it = bdry_vel_is.begin(); it != bdry_vel_is.end(); ++it)
  //   {
  //     const types::global_dof_index i = *it;
  //     if (!locally_owned_dofs.is_element(i))
  //       continue;

  //     // composante du DOF i
  //     const unsigned comp = fe.system_to_component_index(i).first;
  //     if (!(u_lower <= comp && comp < u_upper))
  //       continue; // par sécurité (ne devrait pas arriver vu vel_mask)

  //     // 4.a) Point géométrique du DOF
  //     const Point<dim> p = support_points[i];

  //     // 4.b) Évalue x^{n  }(p)   et   x^{n-1}, x^{n-2} ... (si BDF2)
  //     //      On récupère tout le vecteur et on lira les composantes position.
  //     try
  //     {
  //       VectorTools::point_value(*fixed_mapping, dof_handler, present_solution, p, vals_now);
  //       for (unsigned k = 0; k < previous_solutions.size(); ++k)
  //         VectorTools::point_value(*fixed_mapping, dof_handler, previous_solutions[k], p, vals_prev[k]);
  //     }
  //     catch (...)
  //     {
  //       // Certains points de support peuvent se trouver hors des cellules locales (cas parallèles).
  //       // On ignore proprement : la valeur restera inchangée sur ce rang (sera fixée ailleurs).
  //       continue;
  //     }

  //     // 4.c) Construit w(p) = Σ bdf_i * x^{n-i}(p)
  //     Tensor<1,dim> w;
  //     w = 0.0; 
  //     // terme i=0 -> x^n
  //     for (unsigned d = 0; d < dim; ++d)
  //       w[d] += bdfCoeffs[0] * vals_now[x_lower + d];

  //     // autres termes -> x^{n-i}
  //     for (unsigned ib = 1; ib < bdfCoeffs.size(); ++ib)
  //       for (unsigned d = 0; d < dim; ++d)
  //         w[d] += bdfCoeffs[ib] * vals_prev[ib - 1][x_lower + d];

  //     // 4.d) On écrit la composante correspondante dans le DOF de VITESSE
  //     const unsigned d = comp - u_lower; // 0..dim-1
  //     vector_to_modify[i] = w[d];
  //   }

  //   // 5) Compression parallèle
  //   vector_to_modify.compress(VectorOperation::insert);
  // }


  template <int dim>
  void FSI<dim>::create_nonzero_constraints()
  {
    nonzero_constraints.clear();
    nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    // ----- Mesh position BC -----
    for (auto str : param.position_fixed_boundary_names)
    {
      VectorTools::interpolate_boundary_values(*fixed_mapping,
                                              dof_handler,
                                              mesh_domains_name2tag.at(str),
                                              fixed_mesh_position_fun,
                                              nonzero_constraints,
                                              fe.component_mask(position));
    }
    for (auto str : param.position_moving_boundary_names)
    {
      VectorTools::interpolate_boundary_values(*fixed_mapping,
                                              dof_handler,
                                              mesh_domains_name2tag.at(str),
                                              moving_mesh_position_fun,
                                              nonzero_constraints,
                                              fe.component_mask(position));
    }
    
    // ----- Velocity BC (inlets) -----
    for (auto str : param.Inlet_velocity_boundary_names)
    {
      VectorTools::interpolate_boundary_values(*mapping,
                                              dof_handler,
                                              mesh_domains_name2tag.at(str),
                                              inlet_fun,
                                              nonzero_constraints,
                                              fe.component_mask(velocity));
    }

    // Contrainte "no normal flux" (si demandée)
    this->add_semi_fixed_constraints(nonzero_constraints);
    this->add_no_flux_constraints(nonzero_constraints);
    nonzero_constraints.close();
  }


  template <int dim>
  void
  FSI<dim>::update_boundary_conditions()
  {
    this->create_nonzero_constraints();
  }

  template <int dim>
  void FSI<dim>::assemble_matrix()
  {
    TimerOutput::Scope t(computing_timer, "Assemble matrix");

    system_matrix = 0;

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);
    FullMatrix<double> local_matrix_fd(dofs_per_cell, dofs_per_cell);
    Vector<double>     ref_local_rhs(dofs_per_cell);
    Vector<double>     perturbed_local_rhs(dofs_per_cell);

    FullMatrix<double> diff_matrix(dofs_per_cell, dofs_per_cell);

    // The local dofs values, which will be perturbed
    std::vector<double> cell_dof_values(dofs_per_cell);

    ScratchData<dim> scratchData(fe,
                                 quadrature,
                                 *fixed_mapping,
                                 *mapping,
                                 face_quadrature,
                                 dofs_per_cell,
                                 noslip_bc_boundary_id,
                                 bdfCoeffs);

    // Assemble pseudo-solid on initial mesh
    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      bool distribute = true;
      this->assemble_local_matrix_pseudo_solid(cell,
                                               scratchData,
                                               evaluation_point,
                                               previous_solutions,
                                               local_dof_indices,
                                               local_matrix,
                                               distribute);
    }

    // Assemble Navier-Stokes on current mesh
    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      bool distribute = true;
      this->assemble_local_matrix(cell,
                                  scratchData,
                                  evaluation_point,
                                  previous_solutions,
                                  local_dof_indices,
                                  local_matrix,
                                  distribute);
    }

    system_matrix.compress(VectorOperation::add);
  }

  template <int dim>
  void FSI<dim>::assemble_local_matrix_fd(
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
    this->assemble_local_rhs(cell,
                             scratchData,
                             current_solution,
                             previous_solutions,
                             local_dof_indices,
                             ref_local_rhs,
                             cell_dof_values,
                             distribute,
                             update_cell_dof_values,
                             use_full_solution);

    for (unsigned int j = 0; j < dofs_per_cell; ++j)
    {
      const unsigned int comp = fe.system_to_component_index(j).first;

      const double og_value = cell_dof_values[j];
      cell_dof_values[j] += h;

      if (is_position(comp))
      {
        // Also modify mapping_fe_field
        local_evaluation_point[local_dof_indices[j]] = cell_dof_values[j];
        local_evaluation_point.compress(VectorOperation::insert);
        evaluation_point = local_evaluation_point;
        local_evaluation_point.compress(VectorOperation::insert);
      }

      // Reinit is called in the local rhs function
      this->assemble_local_rhs(cell,
                               scratchData,
                               current_solution,
                               previous_solutions,
                               local_dof_indices,
                               perturbed_local_rhs,
                               cell_dof_values,
                               distribute,
                               update_cell_dof_values,
                               use_full_solution);

      // pcout << "Perturbed               RHS is " << perturbed_local_rhs <<
      // std::endl;

      // Finite differences (with sign change as residual is -NL(u))
      for (unsigned int i = 0; i < dofs_per_cell; ++i)
      {
        local_matrix(i, j) = -(perturbed_local_rhs(i) - ref_local_rhs(i)) / h;
      }

      // Restore solution
      cell_dof_values[j] = og_value;
      if (is_position(comp))
      {
        // Also modify mapping_fe_field
        local_evaluation_point[local_dof_indices[j]] = og_value;
        local_evaluation_point.compress(VectorOperation::insert);
        evaluation_point = local_evaluation_point;
        local_evaluation_point.compress(VectorOperation::insert);
      }
    }
  }

  template <int dim>
  void FSI<dim>::assemble_local_matrix(
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

    // FEValues/FEFaceValues sur le mapping courant (x = éval. actuelle)
    scratchData.reinit_current_mapping(cell, current_solution, previous_solutions);

    local_matrix = 0;

    // ===========================
    // 1) Contributions volumiques
    // ===========================
    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];
      const auto &phi_p      = scratchData.phi_p[q];
      const auto &phi_x      = scratchData.phi_x[q];
      const auto &grad_phi_x = scratchData.grad_phi_x[q];

      const double rho = param.rho;
      const double mu  = param.rho * param.viscosity;

      const auto &uh     = scratchData.present_velocity_values[q];
      const auto &grad_u = scratchData.present_velocity_gradients[q];
      const double div_u = trace(grad_u);
      const double ph    = scratchData.present_pressure_values[q];

      const auto &dxdt = scratchData.present_mesh_velocity_values[q];


      // BDF: d u / dt au temps courant
      Tensor<1, dim> dudt = bdfCoeffs[0] * uh;
      for (unsigned int ib = 1; ib < bdfCoeffs.size(); ++ib)
        dudt += bdfCoeffs[ib] * scratchData.previous_velocity_values[ib - 1][q];

      // Sources (si présentes)
      const auto &src_u  = scratchData.source_term_velocity[q];
      const auto &src_p  = scratchData.source_term_pressure[q];
      const auto &gs_u   = scratchData.grad_source_velocity[q];
      const auto &gs_p   = scratchData.grad_source_pressure[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const unsigned int comp_i = scratchData.components[i];
        const bool i_is_u = is_velocity(comp_i);
        const bool i_is_p = is_pressure(comp_i);

        for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
        {
          const unsigned int comp_j = scratchData.components[j];
          const bool j_is_u = is_velocity(comp_j);
          const bool j_is_p = is_pressure(comp_j);
          const bool j_is_x = is_position(comp_j);

          const bool coup = param.with_position_coupling;

          double a_ij = 0.0;

          if (i_is_u && j_is_u)
          {
            // Masse (BDF0)
            a_ij += rho * (bdfCoeffs[0] * (phi_u[i] * phi_u[j]));
            // Convection rho * [(grad u) phi_u_j + (uh · ∇)phi_u_j]·phi_u_i  (forme consistante)
            a_ij += rho * ((grad_phi_u[j] * uh + grad_u * phi_u[j]) * phi_u[i]);
            // Diffusion mu * ∇u : ∇v
            a_ij += mu * scalar_product(grad_phi_u[i], grad_phi_u[j]);
            // Terme ALE  - rho (w · ∇)u  -> dérivée wrt u_j
            a_ij += rho * ( (grad_phi_u[j] * (-dxdt)) * phi_u[i] );
          }

          if (i_is_u && j_is_p)
          {
            // - ∇p · v  -> - (div v) p
            a_ij += - div_phi_u[i] * phi_p[j];
          }

          if (i_is_u && j_is_x && coup)
          {
            // Variation wrt x (déformation ALE/Jacobien)
            // d/dx [ rho*(dudt, v) ] ~ rho*(dudt · v) tr(∇phi_x_j)
            a_ij += rho * (dudt * phi_u[i] * trace(grad_phi_x[j]));

            // d/dx du terme ALE  -rho (w · ∇)u
            a_ij += rho * ( grad_u * (-bdfCoeffs[0] * phi_x[j]) * phi_u[i] );
            a_ij += rho * ( (-grad_u * grad_phi_x[j]) * (-dxdt) * phi_u[i] );
            a_ij += rho * ( (grad_u * (-dxdt)) * phi_u[i] * trace(grad_phi_x[j]) );

            // d/dx convection (uh · ∇)u
            a_ij += rho * ( (-grad_u * grad_phi_x[j]) * uh * phi_u[i] );
            a_ij += rho * ( (grad_u * uh) * phi_u[i] * trace(grad_phi_x[j]) );

            // d/dx diffusion  mu * ∇u : ∇v
            const Tensor<2, dim> d_grad_u     = - grad_u * grad_phi_x[j];
            const Tensor<2, dim> d_grad_phi_v = - grad_phi_u[i] * grad_phi_x[j];
            a_ij += mu * scalar_product(d_grad_u,     grad_phi_u[i]);
            a_ij += mu * scalar_product(grad_u,       d_grad_phi_v);
            a_ij += mu * scalar_product(grad_u,       grad_phi_u[i]) * trace(grad_phi_x[j]);

            // d/dx pression  -p div v
            a_ij += - ph * trace(- grad_phi_u[i] * grad_phi_x[j]);
            a_ij += - ph * div_phi_u[i] * trace(grad_phi_x[j]);

            // d/dx source_u  (int f · v)
            a_ij += phi_u[i] * gs_u * phi_x[j];
            a_ij += src_u * phi_u[i] * trace(grad_phi_x[j]);
          }

          if (i_is_p && j_is_u)
          {
            // Continuité : - (div u, q)
            a_ij += - phi_p[i] * div_phi_u[j];
          }

          if (i_is_p && j_is_x)
          {
            // d/dx continuité
            a_ij += - trace(-grad_u * grad_phi_x[j]) * phi_p[i];
            a_ij += - div_u * phi_p[i] * trace(grad_phi_x[j]);

            // d/dx source_p
            a_ij += phi_p[i] * gs_p * phi_x[j];
            a_ij += src_p * phi_p[i] * trace(grad_phi_x[j]);
          }

          local_matrix(i, j) += a_ij * JxW;
        }
      }
    }
    // ===========================
    // 3) Distribution globale
    // ===========================
    if (distribute)
    {
      cell->get_dof_indices(local_dof_indices);
      nonzero_constraints.distribute_local_to_global(local_matrix,
                                                    local_dof_indices,
                                                    system_matrix);
    }
  }


  template <int dim>
  void FSI<dim>::assemble_local_matrix_pseudo_solid(
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

    scratchData.reinit_fixed_mapping(cell,
                                     current_solution,
                                     previous_solutions);

    local_matrix = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &grad_phi_x = scratchData.grad_phi_x[q];
      const auto &div_phi_x  = scratchData.div_phi_x[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const unsigned int component_i = scratchData.components[i];
        const bool         i_is_x      = is_position(component_i);

        for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
        {
          const unsigned int component_j = scratchData.components[j];
          const bool         j_is_x      = is_position(component_j);

          double local_matrix_ij = 0.;

          // Position - Position block (x-x)
          if (i_is_x && j_is_x)
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

    if (distribute)
    {
      cell->get_dof_indices(local_dof_indices);
      
      nonzero_constraints.distribute_local_to_global(local_matrix,
                                                     local_dof_indices,
                                                     system_matrix);


    }
  }

  // NEW SIGNATURE (optionnelle) : si raw_out != nullptr, on remplit aussi R_raw
  template <int dim>
  void FSI<dim>::assemble_rhs(LA::MPI::Vector *raw_out /*= nullptr*/)
  {
    TimerOutput::Scope t(computing_timer, "Assemble RHS");

    system_rhs = 0;

    // Si on veut un résidu brut, on prépare le vecteur ghosté
    LA::MPI::Vector R_raw_local;
    if (raw_out != nullptr)
    {
      R_raw_local.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
      R_raw_local = 0;
    }

    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    Vector<double>                       local_rhs(dofs_per_cell);
    std::vector<double>                  cell_dof_values(dofs_per_cell);
    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    ScratchData<dim> scratchData(fe,
                                quadrature,
                                *fixed_mapping,
                                *mapping,
                                face_quadrature,
                                dofs_per_cell,
                                noslip_bc_boundary_id,
                                bdfCoeffs);

    // --- pseudo-solide (mapping fixe) ---
    for (const auto &cell : dof_handler.active_cell_iterators() |
                            IteratorFilters::LocallyOwnedCell())
    {
      bool distribute             = true;   // comme avant -> assemble RHS contraint
      bool update_cell_dof_values = false;
      bool use_full_solution      = true;

      this->assemble_local_rhs_pseudo_solid(cell,
                                            scratchData,
                                            evaluation_point,
                                            previous_solutions,
                                            local_dof_indices,
                                            local_rhs,
                                            cell_dof_values,
                                            distribute,
                                            update_cell_dof_values,
                                            use_full_solution);

      // Si demandé, on fait AUSSI l’ajout "brut" (sans contraintes)
      if (raw_out != nullptr)
      {
        // réévalue local_rhs, mais avec distribute=false
        local_rhs = 0.0;
        distribute = false;
        this->assemble_local_rhs_pseudo_solid(cell,
                                              scratchData,
                                              evaluation_point,
                                              previous_solutions,
                                              local_dof_indices,
                                              local_rhs,
                                              cell_dof_values,
                                              distribute,
                                              update_cell_dof_values,
                                              use_full_solution);
        cell->get_dof_indices(local_dof_indices);
        R_raw_local.add(local_dof_indices, local_rhs);
      }
    }

    // --- Navier–Stokes (mapping courant) ---
    for (const auto &cell : dof_handler.active_cell_iterators() |
                            IteratorFilters::LocallyOwnedCell())
    {
      bool distribute             = true;
      bool update_cell_dof_values = false;
      bool use_full_solution      = true;

      this->assemble_local_rhs(cell,
                              scratchData,
                              evaluation_point,
                              previous_solutions,
                              local_dof_indices,
                              local_rhs,
                              cell_dof_values,
                              distribute,
                              update_cell_dof_values,
                              use_full_solution);

      if (raw_out != nullptr)
      {
        local_rhs = 0.0;
        distribute = false;
        this->assemble_local_rhs(cell,
                                scratchData,
                                evaluation_point,
                                previous_solutions,
                                local_dof_indices,
                                local_rhs,
                                cell_dof_values,
                                distribute,
                                update_cell_dof_values,
                                use_full_solution);
        cell->get_dof_indices(local_dof_indices);
        R_raw_local.add(local_dof_indices, local_rhs);
      }
    }

    system_rhs.compress(VectorOperation::add);
    if (raw_out != nullptr)
    {
      R_raw_local.compress(VectorOperation::add);
      *raw_out = R_raw_local;  // copie ghostée -> ghostée
    }
  }


  template <int dim>
  void FSI<dim>::assemble_local_rhs(
    const typename DoFHandler<dim>::active_cell_iterator &cell,
    ScratchData<dim>                                     &scratchData,
    LA::MPI::Vector                                      &current_solution,
    std::vector<LA::MPI::Vector>                         &previous_solutions,
    std::vector<types::global_dof_index>                 &local_dof_indices,
    Vector<double>                                       &local_rhs,
    std::vector<double>                                  &cell_dof_values,
    bool                                                  distribute,
    bool                                                  update_cell_dof_values,
    bool                                                  use_full_solution)
  {
    // -- éventuellement rafraîchir les valeurs locales (mode FD) --
    if (update_cell_dof_values)
    {
      cell->get_dof_indices(local_dof_indices);
      for (unsigned int j = 0; j < local_dof_indices.size(); ++j)
        cell_dof_values[j] = local_evaluation_point[local_dof_indices[j]];
    }

    // -- FEValues / FEFaceValues via ScratchData (mapping courant) --
    if (use_full_solution)
      scratchData.reinit_current_mapping(cell, current_solution, previous_solutions);
    else
      scratchData.reinit_current_mapping(cell, cell_dof_values, previous_solutions);

    // Remise à zéro du RHS local
    local_rhs = 0;

    // ==================================================================
    // 1) Contributions VOLUmiques : Navier-Stokes + cinématique w = \dot x_BDF
    // ==================================================================
    const unsigned int          nBDF = bdfCoeffs.size();
    std::vector<Tensor<1, dim>> velocity(nBDF);

    const double rho = param.rho;
    const double mu  = param.rho * param.viscosity; // mu = rho * nu

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &present_velocity_values      = scratchData.present_velocity_values[q];
      const auto &present_velocity_gradients   = scratchData.present_velocity_gradients[q];
      const auto &present_pressure_values      = scratchData.present_pressure_values[q];
      const auto &present_position_values      = scratchData.present_position_values[q];
      const auto &present_mesh_velocity_values = scratchData.present_mesh_velocity_values[q];
      const auto  &source_term_velocity        = scratchData.source_term_velocity[q];
      const auto  &source_term_pressure        = scratchData.source_term_pressure[q];

      const double present_velocity_divergence  = trace(present_velocity_gradients);

      // BDF pour u
      velocity[0] = present_velocity_values;
      for (unsigned int i = 1; i < nBDF; ++i)
        velocity[i] = scratchData.previous_velocity_values[i - 1][q];

      const auto &phi_p      = scratchData.phi_p[q];
      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const unsigned int component_i = scratchData.components[i];

        double local_rhs_i =
          -(
            // Convection :  rho * (u · ∇)u testé par v
            rho * (present_velocity_gradients * present_velocity_values) * phi_u[i]

            // ALE        : -rho * (w · ∇)u testé par v
            - rho * (present_velocity_gradients * present_mesh_velocity_values) * phi_u[i]

            // Diffusion  : +mu * (∇u : ∇v)
            + mu * scalar_product(present_velocity_gradients, grad_phi_u[i])

            // Pression   : -p * div(v)
            - present_pressure_values * div_phi_u[i]

            // Source quantité de mouvement
            + source_term_velocity * phi_u[i]

            // Continuité : -(div u) * q
            - present_velocity_divergence * phi_p[i]

            // Source pression
            + source_term_pressure * phi_p[i]
          ) * JxW;

        // Terme transitoire pour u : - rho * sum_i bdf_i * (u^{n-i}, v)
        for (unsigned int iBDF = 0; iBDF < nBDF; ++iBDF)
          local_rhs_i -= rho * bdfCoeffs[iBDF] * velocity[iBDF] * phi_u[i] * JxW;

        local_rhs(i) += local_rhs_i;
      }
    }

    // ==================================================================
    // 3) Assemblage local -> global (contraintes non-homogènes déjà gérées)
    // ==================================================================
    if (distribute)
    {
      cell->get_dof_indices(local_dof_indices);
      nonzero_constraints.distribute_local_to_global(local_rhs,
                                                    local_dof_indices,
                                                    system_rhs);
    }
  }



  template <int dim>
  void FSI<dim>::assemble_local_rhs_pseudo_solid(
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
      scratchData.reinit_fixed_mapping(cell,
                                       current_solution,
                                       previous_solutions);
    }
    else
      scratchData.reinit_fixed_mapping(cell,
                                       cell_dof_values,
                                       previous_solutions);

    local_rhs = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &present_position_gradients =
        scratchData.present_position_gradients[q];

      const auto &source_term_position = scratchData.source_term_position[q];

      const double present_displacement_divergence =
        trace(present_position_gradients);

      const auto &phi_x      = scratchData.phi_x[q];
      const auto &grad_phi_x = scratchData.grad_phi_x[q];
      const auto &div_phi_x  = scratchData.div_phi_x[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const auto present_displacement_gradient_sym =
          present_position_gradients + transpose(present_position_gradients);

        double local_rhs_i =
          -(
            // Linear elasticity
            +param.pseudo_solid_lambda * present_displacement_divergence *
              div_phi_x[i] +
            param.pseudo_solid_mu *
              scalar_product(present_displacement_gradient_sym, grad_phi_x[i])

            // Linear elasticity source term
            + phi_x[i] * source_term_position) *
          JxW;

        local_rhs(i) += local_rhs_i;
      }
    }

    if (distribute)
    {
      cell->get_dof_indices(local_dof_indices);
      nonzero_constraints.distribute_local_to_global(local_rhs,
                                                      local_dof_indices,
                                                      system_rhs);
    }
  }

  template <int dim>
  void FSI<dim>::solve_direct()
  {
    TimerOutput::Scope t(computing_timer, "Solve direct");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs,
                                                    mpi_communicator);

    // Solve with MUMPS
    SolverControl                    solver_control;
    PETScWrappers::SparseDirectMUMPS solver(solver_control);
    solver.solve(system_matrix, completely_distributed_solution, system_rhs);

    newton_update = completely_distributed_solution;


    nonzero_constraints.distribute(newton_update);
  }

  template <int dim>
  void
  FSI<dim>::solve_newton()
  {
    double global_res;
    double current_res;
    double last_res;
    unsigned int outer_iteration = 0;
    last_res              = 1e6;
    current_res           = 1e6;
    global_res            = 1e6;

    while ((global_res > this->param.newton_tolerance) &&
           outer_iteration < 50)
      {
        evaluation_point = present_solution;
        // this->enforce_u_equals_w_manual(noslip_bc_boundary_id, evaluation_point);

        this->assemble_matrix();
        this->assemble_rhs();
        // this->dump_boundary_velocity_debug(noslip_bc_boundary_id, "pre_bc", /*after_solve=*/false);
        this->apply_row_replacement_u_equals_w(noslip_bc_boundary_id);
        // this->dump_boundary_velocity_debug(noslip_bc_boundary_id, "post_bc", /*after_solve=*/false);
        if (outer_iteration==0){
          current_res      = this->system_rhs.l2_norm();
          last_res         = current_res;
        }

        if (VERBOSE)
          {
            pcout << "Newton iteration: " << outer_iteration << "  - Residual:  " << current_res << std::endl;
          }
        
        this->solve_direct();
        // this->dump_boundary_velocity_debug(noslip_bc_boundary_id, "after_solve", /*after_solve=*/true);
        double last_alpha_res = current_res;

        unsigned int alpha_iter = 0;
        for (double alpha = 1.0; alpha >=.125; alpha *= 0.5)
          {

            local_evaluation_point       = present_solution;
            local_evaluation_point.add(alpha, newton_update);
            nonzero_constraints.distribute(local_evaluation_point);
            evaluation_point = local_evaluation_point;
            // this->enforce_u_equals_w_manual(noslip_bc_boundary_id, evaluation_point);
            local_evaluation_point.compress(VectorOperation::insert);
            LA::MPI::Vector raw_residual;
            this->assemble_rhs(&raw_residual); // remplit raw_residual (ghosté)
            this->apply_row_replacement_u_equals_w(noslip_bc_boundary_id);
            raw_residual.update_ghost_values();

            // auto &system_rhs = solver->get_system_rhs();
            current_res      = system_rhs.l2_norm();

            if (VERBOSE)
              {
                pcout << "\talpha = " << std::setw(6) << alpha
                              << std::setw(0) << " res = "
                              << std::setprecision(6)
                              << std::setw(6) << current_res << std::endl;

                // solver->output_newton_update_norms(
                //   this->params.display_precision);
              }

            // If it's not the first iteration of alpha check if the residual is
            // smaller than the last alpha iteration. If it's not smaller, we fall
            // back to the last alpha iteration.
            if (current_res > last_alpha_res and alpha_iter != 0)
              {
                alpha                  = 2 * alpha;
                local_evaluation_point = present_solution;
                local_evaluation_point.add(alpha, newton_update);
                nonzero_constraints.distribute(local_evaluation_point);
                evaluation_point = local_evaluation_point;
                local_evaluation_point.compress(VectorOperation::insert);

                if (VERBOSE)
                  {
                    pcout
                      << "\t\talpha value was kept at alpha = " << alpha
                      << " since alpha = " << alpha / 2
                      << " increased the residual" << std::endl;
                  }
                current_res = last_alpha_res;
                break;
              }
            if (current_res < 0.1 * last_res ||
                last_res < param.newton_tolerance)
              {
                break;
              }
            last_alpha_res = current_res;
            alpha_iter++;
          }
        // global_res       = solver->get_current_residual();
        global_res       = current_res;
        present_solution = evaluation_point;
        present_solution.update_ghost_values();  
        last_res         = current_res;
        ++outer_iteration;
      }

    // If the non-linear solver has not converged abort simulation if
    // abort_at_convergence_failure=true
    if ((global_res > param.newton_tolerance) &&
        outer_iteration >= 10000)
      {
        throw(std::runtime_error(
          "Stopping simulation because the non-linear solver has failed to converge"));
      }
  }

  template <int dim>
  void FSI<dim>::output_results(const unsigned int time_step,
                                const bool         write_newton_iteration,
                                const unsigned int newton_step)
  {
    //
    // Plot FE solution
    //
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.push_back("pressure");
    for (unsigned int d = 0; d < dim; ++d)
      solution_names.push_back("mesh_position");
    
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);
    for (unsigned int d = 0; d < 2 * dim; ++d)
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    //
    // Compute mesh velocity in post-processing
    // This is not ideal, this is done by modifying the displacement and
    // reexporting.
    //
    LA::MPI::Vector mesh_velocity;
    mesh_velocity.reinit(locally_owned_dofs, mpi_communicator);
    const FEValuesExtractors::Vector position(x_lower);
    IndexSet                         disp_dofs =
      DoFTools::extract_dofs(dof_handler, fe.component_mask(position));

    for (const auto &i : disp_dofs)
    {
      if (!locally_owned_dofs.is_element(i))
        continue;

      double value = bdfCoeffs[0] * present_solution[i];
      for (unsigned int iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
        value += bdfCoeffs[iBDF] * previous_solutions[iBDF - 1][i];
      mesh_velocity[i] = value;
    }
    mesh_velocity.compress(VectorOperation::insert);
    std::vector<std::string> mesh_velocity_name(dim, "ph_velocity");
    mesh_velocity_name.emplace_back("ph_pressure");
    for (unsigned int i = 0; i < dim; ++i)
      mesh_velocity_name.push_back("mesh_velocity");

    data_out.add_data_vector(mesh_velocity,
                             mesh_velocity_name,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    //
    // Partition
    //
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(*mapping, 2);

    // Export a Newton iteration in dedicated folder
    if(write_newton_iteration)
    {
      std::string root = "../data/fsi_decoupled_strong/fsi_coupled_newton_iterations/";
      std::string fileName = "solution_time_step_" + std::to_string(time_step);
      data_out.write_vtu_with_pvtu_record(
        root, fileName, newton_step, mpi_communicator, 2);
    }
    else
    {
      // Export regular time step
      std::string root = "../data/fsi_decoupled_strong/";
      std::string fileName = "solution";
      data_out.write_vtu_with_pvtu_record(
        root, fileName, time_step, mpi_communicator, 2);
    }
  }

  template <int dim>
  void FSI<dim>::compare_mesh_fluid_velocity(const unsigned int boundary_id)
  {
    double l2_local = 0.0, li_local = 0.0;

    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Vector position(x_lower);

    FEFaceValues<dim> fe_face_values(*mapping,   // <-- mapping COURANT partout
                                    fe,
                                    face_quadrature,
                                    update_values | update_quadrature_points | update_JxW_values);

    const unsigned int n_q = face_quadrature.size();

    std::vector<std::vector<Tensor<1, dim>>> position_values(
        bdfCoeffs.size(), std::vector<Tensor<1, dim>>(n_q));
    std::vector<Tensor<1, dim>> u(n_q), w(n_q), diff(n_q);

    Tensor<1, dim> boundary_mesh_vel_local, boundary_fluid_vel_local;
    double boundary_area_local = 0.0;

    present_solution.update_ghost_values();
    for (auto &v : previous_solutions) v.update_ghost_values();

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned()) continue;

      for (const auto f : cell->face_indices())
      {
        const auto &face = cell->face(f);
        if (!(face->at_boundary() && face->boundary_id() == boundary_id)) continue;

        fe_face_values.reinit(cell, f);

        fe_face_values[velocity].get_function_values(present_solution, u);
        fe_face_values[position].get_function_values(present_solution, position_values[0]);
        for (unsigned iBDF = 1; iBDF < bdfCoeffs.size(); ++iBDF)
          fe_face_values[position].get_function_values(previous_solutions[iBDF - 1], position_values[iBDF]);

        for (unsigned q = 0; q < n_q; ++q)
        {
          w[q] = 0;
          for (unsigned iBDF = 0; iBDF < bdfCoeffs.size(); ++iBDF)
            w[q] += bdfCoeffs[iBDF] * position_values[iBDF][q];

          diff[q] = u[q] - w[q];
          const double JxW = fe_face_values.JxW(q);

          for (unsigned d = 0; d < dim; ++d)
          {
            boundary_mesh_vel_local[d]  += w[q][d] * JxW;
            boundary_fluid_vel_local[d] += u[q][d] * JxW;
          }
          boundary_area_local += JxW;

          l2_local += diff[q] * diff[q] * JxW;
          li_local = std::max(li_local, std::abs(diff[q].norm()));
        }
      }
    }

    Tensor<1, dim> boundary_mesh_vel_global, boundary_fluid_vel_global;
    for (unsigned d = 0; d < dim; ++d)
    {
      boundary_mesh_vel_global[d]  = Utilities::MPI::sum(boundary_mesh_vel_local[d], mpi_communicator);
      boundary_fluid_vel_global[d] = Utilities::MPI::sum(boundary_fluid_vel_local[d], mpi_communicator);
    }
    const double boundary_area_global = Utilities::MPI::sum(boundary_area_local, mpi_communicator);

    Tensor<1, dim> mean_mesh_velocity, mean_fluid_velocity;
    for (unsigned d = 0; d < dim; ++d)
    {
      mean_mesh_velocity[d]  = boundary_mesh_vel_global[d]  / boundary_area_global;
      mean_fluid_velocity[d] = boundary_fluid_vel_global[d] / boundary_area_global;
    }

    const double l2_error = std::sqrt(Utilities::MPI::sum(l2_local, mpi_communicator));
    const double li_error = Utilities::MPI::max(li_local, mpi_communicator);

    pcout << "--------------------------------------------------\n";
    pcout << "Boundary ID: " << boundary_id << "\n";
    pcout << "||u_h - w_h||_L2   = " << l2_error << "\n";
    pcout << "||u_h - w_h||_Linf = " << li_error << "\n";
    pcout << "Moyenne vitesse maillage : ";
    for (unsigned d = 0; d < dim; ++d) pcout << " " << mean_mesh_velocity[d];
    pcout << "\nMoyenne vitesse fluide   : ";
    for (unsigned d = 0; d < dim; ++d) pcout << " " << mean_fluid_velocity[d];
    pcout << "\n--------------------------------------------------" << std::endl;

    if (mpi_rank == 0)
    {
      std::ofstream outfile("../data/fsi_decoupled_strong/forces.txt", std::ios::app);
      outfile.setf(std::ios::fixed);
      outfile << std::setprecision(15)
              << mean_fluid_velocity[0] << '\t'
              << mean_fluid_velocity[1] << '\t'
              << mean_mesh_velocity[0]  << '\t'
              << mean_mesh_velocity[1]  << '\t'
              << l2_error << '\t'
              << li_error << '\n';
    }
  }



  template <int dim>
  void FSI<dim>::compute_drag_lift_and_write(const unsigned int boundary_id, const double actual_time)
  {
    const FEValuesExtractors::Vector velocity(u_lower);
    const FEValuesExtractors::Scalar pressure(p_lower);

    const double mu  = param.rho * param.viscosity;

    FEFaceValues<dim> fe_face_values(*mapping,
                                    fe,
                                    face_quadrature,
                                    update_values |
                                    update_gradients |
                                    update_normal_vectors |
                                    update_JxW_values);

    const unsigned int n_q = face_quadrature.size();

    // buffers
    std::vector<Tensor<2,dim>> grad_u(n_q);
    std::vector<double>        p(n_q);

    double Fx_local = 0.0, Fy_local = 0.0;

    // --- boucle faces sur la frontière ---
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (const unsigned int f : cell->face_indices())
      {
        const auto &face = cell->face(f);
        if (!(face->at_boundary() && face->boundary_id() == boundary_id))
          continue;

        fe_face_values.reinit(cell, f);
        fe_face_values[velocity].get_function_gradients(present_solution, grad_u);
        fe_face_values[pressure].get_function_values(present_solution, p);

        for (unsigned int q=0; q<n_q; ++q)
        {
          const Tensor<1,dim> n = fe_face_values.normal_vector(q);
          // sigma = -p I + mu (grad u + grad u^T)
          Tensor<2,dim> sigma;
          for (unsigned int i=0;i<dim;++i)
            sigma[i][i] = -p[q];
          sigma += mu * (grad_u[q] + transpose(grad_u[q]));

          const Tensor<1,dim> t = sigma * n; // traction
          const double JxW = fe_face_values.JxW(q);

          Fx_local += t[0] * JxW;
          if constexpr (dim >= 2) Fy_local += t[1] * JxW;
        }
      }
    }

    // --- réduction MPI ---
    const double Fx = Utilities::MPI::sum(Fx_local, mpi_communicator);
    const double Fy = Utilities::MPI::sum(Fy_local, mpi_communicator);

    // --- coefficients aéro ---
    // 2D : aire de référence = D (par unité d’épaisseur). 3D : on prend D * 1 (span unitaire).
    const double qinf = 0.5 * param.rho * param.U * param.U;
    const double Aref = (dim==2 ? param.D : param.D * 1.0);

    const double Cx = Fx / (qinf * Aref);
    const double Cy = Fy / (qinf * Aref);

    if (mpi_rank == 0)
    {
      std::ofstream outfile("../data/fsi_decoupled_strong/forces.txt", std::ios::app);
      outfile.setf(std::ios::fixed);
      outfile << std::setprecision(15);

      if (actual_time == param.dt)
      {
      outfile << "time\tFx\tFy\tCFx\tCFy\tux_cyl\tuy_cyl\twx_cyl\twy_cyl\terr_l2\terr_linf\n";
      outfile << current_time << '\t'
              << Fx << '\t'
              << Fy << '\t'
              <<  Cx          << '\t'
              <<  Cy           << '\t';
      }
      else if (actual_time > param.dt)
      {
      outfile << current_time << '\t'
              << Fx << '\t'
              << Fy << '\t'
              <<  Cx          << '\t'
              <<  Cy           << '\t';
      }
    }
  }


  template <int dim>
  void FSI<dim>::compute_implicit_boundary_forces(const unsigned int boundary_id, const double actual_time)
  {
    // 1) Résidu brut au point courant
    LA::MPI::Vector R_raw;
    this->assemble_rhs(&R_raw);   // remplit system_rhs (contraint) ET R_raw (brut)
    R_raw.update_ghost_values();

    // 2) DDL de vitesse situés sur la frontière demandée
    const FEValuesExtractors::Vector velocity(u_lower);
    const ComponentMask vel_mask = fe.component_mask(velocity);

    std::set<types::boundary_id> ids =
        { static_cast<types::boundary_id>(boundary_id) };
    const IndexSet bdry_vel_is =
        DoFTools::extract_boundary_dofs(dof_handler, vel_mask, ids);

    // 3) Somme des réactions : réaction_i = -R_raw[i]
    double Fx_local = 0.0;
    double Fy_local = 0.0;

    for (auto it = bdry_vel_is.begin(); it != bdry_vel_is.end(); ++it)
    {
      const types::global_dof_index i = *it;
      const unsigned comp = fe.system_to_component_index(i).first;

      const double reaction_i = -R_raw[i];

      if (comp == u_lower)          Fx_local += reaction_i;        // composante x
      if constexpr (dim >= 2) {
        if (comp == u_lower + 1)    Fy_local += reaction_i;        // composante y
      }
    }

    // 4) Réduction MPI -> mémorise dans les membres
    Fx_imp_last = Utilities::MPI::sum(Fx_local, mpi_communicator);
    if constexpr (dim >= 2)
      Fy_imp_last = Utilities::MPI::sum(Fy_local, mpi_communicator);

    const double qinf = 0.5 * param.rho * param.U * param.U;
    const double Aref = (dim==2 ? param.D : param.D * 1.0);

    const double Cx = Fx_imp_last / (qinf * Aref);
    const double Cy = Fy_imp_last / (qinf * Aref);

    if (mpi_rank == 0)
    {
      std::ofstream outfile("../data/fsi_decoupled_strong/forces_res.txt", std::ios::app);
      outfile.setf(std::ios::fixed);
      outfile << std::setprecision(15);

      if (actual_time == param.dt)
      {
      outfile << "time\tFx\tFy\tCFx\tCFy\n";
      outfile << current_time << '\t'
              << Fx_imp_last << '\t'
              << Fy_imp_last << '\t'
              <<  Cx          << '\t'
              <<  Cy           << '\n';
      }
      else if (actual_time > param.dt)
      {
      outfile << current_time << '\t'
              << Fx_imp_last << '\t'
              << Fy_imp_last << '\t'
              <<  Cx          << '\t'
              <<  Cy           << '\n';
      }
    }
  }

  // template <int dim>
  // void FSI<dim>::dump_boundary_velocity_debug(const types::boundary_id boundary_id,
  //                                             const std::string &label,
  //                                             const bool after_solve)
  // {
  //   const FEValuesExtractors::Vector velocity(u_lower);
  //   const ComponentMask vel_mask = fe.component_mask(velocity);

  //   std::set<types::boundary_id> ids = { boundary_id };
  //   const IndexSet bdry_vel_is =
  //       DoFTools::extract_boundary_dofs(dof_handler, vel_mask, ids);

  //   // Points de support -> IMPORTANT: mapping FIXE pour rester cohérent avec w
  //   std::vector<Point<dim>> support_points(dof_handler.n_dofs());
  //   DoFTools::map_dofs_to_support_points(*fixed_mapping, dof_handler, support_points);

  //   std::ostringstream fname;
  //   fname << "../data/fsi_decoupled_strong/bc_dump_rank" << mpi_rank << ".csv";
  //   std::ofstream out(fname.str(), std::ios::app);
  //   out.setf(std::ios::scientific);
  //   out << std::setprecision(15);

  //   // En-tête (facultatif – tu peux le mettre une fois au début du run)
  //   out << "time,step,label,dof,comp,x";
  //   if constexpr (dim>=2) out << ",y";
  //   out << ",u_k,w,delta,b_i,Aii,u_new\n";

  //   Vector<double> vals_now(n_components);
  //   std::vector<Vector<double>> vals_prev(previous_solutions.size(),
  //                                         Vector<double>(n_components));

  //   for (auto it = bdry_vel_is.begin(); it != bdry_vel_is.end(); ++it)
  //   {
  //     const types::global_dof_index i = *it;
  //     if (!locally_owned_dofs.is_element(i)) continue;

  //     const unsigned comp = fe.system_to_component_index(i).first;
  //     if (!(u_lower <= comp && comp < u_upper)) continue;  // sécurité

  //     const Point<dim> &p = support_points[i];

  //     // calcule w(p) avec mapping FIXE (x^n et x^{n-i} évalués sur la géo de ref)
  //     double w_comp = 0.0;
  //     try {
  //       VectorTools::point_value(*fixed_mapping, dof_handler, present_solution, p, vals_now);
  //       for (unsigned k=0; k<previous_solutions.size(); ++k)
  //         VectorTools::point_value(*fixed_mapping, dof_handler, previous_solutions[k], p, vals_prev[k]);

  //       const unsigned d = comp - u_lower;
  //       double w_d = bdfCoeffs[0] * vals_now[x_lower + d];
  //       for (unsigned ib=1; ib<bdfCoeffs.size(); ++ib)
  //         w_d += bdfCoeffs[ib] * vals_prev[ib-1][x_lower + d];
  //       w_comp = w_d;
  //     } catch (...) {
  //       // en //, certains points peuvent ne pas être accessibles
  //       continue;
  //     }

  //     // valeurs utiles
  //     const double u_k   = evaluation_point[i];           // valeur courante sur ce DDL
  //     const double delta = w_comp - u_k;                  // ce qu'on DOIT imposer pour δu
  //     const double b_i   = system_rhs.size() ? system_rhs[i] : 0.0;
  //     const double Aii   = system_matrix.m() ? system_matrix.diag_element(i) : 0.0;
  //     const double u_new = after_solve ? present_solution[i] : std::numeric_limits<double>::quiet_NaN();

  //     out << current_time << "," << current_time_step << "," << label << ","
  //         << i << "," << (comp - u_lower) << ","
  //         << p[0];
  //     if constexpr (dim>=2) out << "," << p[1];
  //     out << "," << u_k << "," << w_comp << "," << delta << ","
  //         << b_i << "," << Aii << "," << u_new << "\n";
  //   }
  // }


  template <int dim>
  void FSI<dim>::run()
  {
      this->param.prev_dt = this->param.dt;
      this->set_bdf_coefficients(param.bdf_order);

      this->current_time = param.t0;
      this->current_time_step = 1;
      this->inlet_fun.set_time(current_time);
      this->fixed_mesh_position_fun.set_time(current_time);
      this->moving_mesh_position_fun.set_time(current_time);

      this->make_grid();
      this->setup_system();
      this->create_nonzero_constraints();
      this->create_sparsity_pattern();

      this->set_initial_condition();
      this->output_results(0);

      for (unsigned int i = 0; i < param.nTimeSteps; ++i, ++(this->current_time_step))
      {
        this->current_time += param.dt;
        this->inlet_fun.set_time(current_time);
        this->fixed_mesh_position_fun.set_time(current_time);
        this->moving_mesh_position_fun.set_time(current_time);

        if (VERBOSE)
        {
          pcout << std::endl
                << "Time step " << i + 1
                << " - Advancing to t = " << current_time << '.' << std::endl;
        }

        ////////////////////////////////////////////////////////////
        this->update_boundary_conditions();
        ////////////////////////////////////////////////////////////

        if (i == 0 && param.bdf_order == 2)
        {
          this->set_initial_condition();
        }
        else if (i == 1 && param.bdf_order == 3)
        {
          this->set_initial_condition();
        }
        else
        {
          this->solve_newton();
        }

        // Efforts implicites (réactions) sur la paroi no-slip
        this->compute_implicit_boundary_forces(noslip_bc_boundary_id,current_time);
        this->compute_drag_lift_and_write(noslip_bc_boundary_id,current_time);
        this->compare_mesh_fluid_velocity(noslip_bc_boundary_id);
        

        if ( ((i + 1) % param.write_result_it == 0))
          this->output_results(i + 1);

        // Rotate solutions
        if (param.bdf_order > 0)
        {
          for (unsigned int i = previous_solutions.size() - 1; i >= 1; --i)
          {
            previous_solutions[i] = previous_solutions[i - 1];
          }
          previous_solutions[0] = present_solution;
        }
      }
  }
} // namespace fsi_coupled

int main(int argc, char *argv[])
{
  try
  {
    using namespace dealii;
    using namespace fsi_coupled;

    Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    const unsigned int dim = 2;

    SimulationParameters<dim> param;

    param.velocity_degree = 2;
    param.position_degree = 2;

    param.with_position_coupling = false; // <===========================================================
    param.with_fixed_position = true;

    param.write_result_it = 10 ;


    if (dim == 2){
      param.mesh_file = "../data/meshes/cyl_not_confined.msh";}
    else{
      param.mesh_file = "../data/meshes/cyl_not_confined_3D.msh";}

    //
    // Mesh position BC
    //
    if (dim == 2) 
    {
      param.position_fixed_boundary_names = {"Inlet", "Outlet", "NoFlux"};
      if(!param.with_position_coupling && param.with_fixed_position)
      {
        param.position_fixed_boundary_names.push_back("InnerBoundary");
      }
      else if (!param.with_position_coupling && !param.with_fixed_position)
      {
        param.position_moving_boundary_names = {"InnerBoundary"};
      }
      //
      // Velocity BC
      //
      param.Inlet_velocity_boundary_names = {"Inlet"};
      param.noflux_velocity_boundary_names = {"NoFlux"};
      param.noslip_velocity_boundary_names = {"InnerBoundary"};
    }
    else if (dim == 3)
    {
      if(!param.with_position_coupling && param.with_fixed_position)
      {
        param.position_fixed_boundary_names.push_back("InnerBoundary");
        param.position_fixed_boundary_names = {"Inlet","Outlet","Top","Bottom"};
      }
      else if (!param.with_position_coupling && !param.with_fixed_position)
      {
        param.position_moving_boundary_names = {"InnerBoundary"};
        param.position_fixed_boundary_names = {"Inlet","Outlet","Top","Bottom"};
        param.position_semi_fixed_boundary_names = {"Front","Back"};
      }
      //
      // Velocity BC
      //
      param.Inlet_velocity_boundary_names = {"Inlet"};
      param.noflux_velocity_boundary_names = {"NoFlux", "Wall"};
    }

    // For the unconfined case
    param.Re = 100.;
    param.H  = 16.;
    param.D  = 1.;
    param.U  = 1.;
    param.rho = 1.;

    param.viscosity           = param.U * param.D / param.Re;
    param.pseudo_solid_mu     = 1.;
    param.pseudo_solid_lambda = 1.;

    const double Ur = 7.5;
    param.spring_constant     = ((param.rho*M_PI)/((Ur*Ur)/(M_PI*M_PI*param.U*param.U)));

    // Time integration
    param.bdf_order  = 2;
    param.t0         = 0.;
    param.dt         = 0.05;
    param.nTimeSteps = 2000;
    param.t1         = param.dt * param.nTimeSteps;

    param.newton_tolerance = 1e-10;

    VERBOSE = true;

    FSI<dim> problem(param);
    problem.run();
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
