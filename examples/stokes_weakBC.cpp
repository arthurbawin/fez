
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
#include <deal.II/base/index_set.h>
#include <deal.II/base/utilities.h>

#include <deal.II/distributed/grid_refinement.h>

#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/lac/affine_constraints.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/petsc_precondition.h>
#include <deal.II/lac/petsc_solver.h>
#include <deal.II/lac/petsc_sparse_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/petsc_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/solver_minres.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/lac/vector.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/quadrature_lib.h>

#include <deal.II/distributed/fully_distributed_tria.h>

#include <deal.II/fe/fe_pyramid_p.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_simplex_p_bubbles.h>
#include <deal.II/fe/fe_wedge_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <cmath>
#include <fstream>
#include <iostream>

// #define STRONG_NO_SLIP

namespace MovingMeshTest
{
  using namespace dealii;

  // Linear ramp from k1 to k2 between times t1 and t2 (with t1 < t2, k1 > k2)
  double springStiffnessRamp(const double t, const double t1, const double t2,
                             const double k1, const double k2)
  {
    AssertThrow(t1 < t2, ExcMessage("t1 must be less than t2"));

    if (t < t1)
      return k1;
    else if (t > t2)
      return k2;
    else
      return k1 + (k2 - k1) * ((t - t1) / (t2 - t1));
  }

  double springStiffnessRamp(const double t,
                 const double t1, const double t2, const double t3,
                 const double k1, const double k2, const double k3)
  {
    AssertThrow(t1 < t2 && t2 < t3, ExcMessage("Times must satisfy t1 < t2 < t3"));

    if (t < t1)
      return k1;
    else if (t < t2)
      return k1 + (k2 - k1) * ((t - t1) / (t2 - t1));
    else if (t < t3)
      return k2 + (k3 - k2) * ((t - t2) / (t3 - t2));
    else
      return k3;
  }

  template <int dim>
  class CylinderDisplacement : public Function<dim>
  {
  public:
    CylinderDisplacement(const double time = 0.)
      : Function<dim>(4*dim+1, time)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      const double t = this->get_time();
      const unsigned int offset = dim + 1;

      if constexpr (dim == 2)
      {
        values[offset + 0] = 0.;
        values[offset + 1] = 0.01 * sin(M_PI * t);
      } else
      {
        values[offset + 0] = 0.1;
        values[offset + 1] = 0.;
        values[offset + 2] = 0.;
      }
    }
  };

  template <int dim>
  class VelocityInlet : public Function<dim>
  {
  public:
    VelocityInlet(const double time = 0.) : Function<dim>(4*dim+1, time)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      const double t = this->get_time();
      const double x = p[0];
      const double y = p[1];
      const double H = 0.41;
      const double uMax = 0.3;
      const unsigned int offset = 0;

      if constexpr (dim == 2)
      {
        values[offset + 0] = 4. * uMax * y * (H-y)/(H*H);
        values[offset + 1] = 0.;
      } else
      {
        values[offset + 0] = 0.;
        values[offset + 1] = 4. * uMax * x * (H-x)/(H*H);
        values[offset + 2] = 0.;
      }
    }
  };

  template <int dim>
  class InitialCondition : public Function<dim>
  {
  public:
    InitialCondition()
      : Function<dim>(4*dim+1)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      for(unsigned int i = 0; i < 4 * dim + 1; ++i)
        values[i] = 0.;
    }
  };

  template <int dim>
  class ScratchData
  {
  public:
    ScratchData(const unsigned int n_q_points,
                const unsigned int n_faces_q_points,
                const unsigned int dofs_per_cell,
                const unsigned int boundary_id)
     : n_q_points(n_q_points)
     , n_faces_q_points(n_faces_q_points)
     , dofs_per_cell(dofs_per_cell)
     , boundary_id(boundary_id)
    {
      this->allocate();
    }

    void allocate();
    void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
                const LA::MPI::Vector &current_solution,
                const std::vector<LA::MPI::Vector> &previous_solutions,
                FEValues<dim> &fe_values,
                FEFaceValues<dim> &fe_face_values);

  public:

    const unsigned int n_q_points;
    const unsigned int dofs_per_cell;

    const unsigned int boundary_id;

    std::vector<double> JxW;

    std::vector<unsigned int> components;

    // Current values and gradients for each quad node
    std::vector<Tensor<1, dim>> present_velocity_values;
    std::vector<Tensor<2, dim>> present_velocity_gradients;
    std::vector<double>         present_pressure_values;
    std::vector<std::vector<Tensor<1, dim>>> previous_velocity_values;

    std::vector<Tensor<1, dim>> present_displacement_values;
    std::vector<Tensor<2, dim>> present_displacement_gradients;
    std::vector<Tensor<1, dim>> present_mesh_velocity_values;
    std::vector<std::vector<Tensor<1, dim>>> previous_displacement_values;

    std::vector<Tensor<1, dim>> present_face_velocity_values;
    std::vector<Tensor<1, dim>> present_face_mesh_velocity_values;
    std::vector<Tensor<1, dim>> present_face_lambda_values;

    // Shape functions and gradients for each quad node and each dof
    std::vector<std::vector<double>>         div_phi_u;
    std::vector<std::vector<Tensor<1, dim>>> phi_u;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_u;
    std::vector<std::vector<double>>         phi_p;
    std::vector<std::vector<Tensor<1, dim>>> phi_disp;
    std::vector<std::vector<double>>         div_phi_disp;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_disp;
    std::vector<std::vector<Tensor<1, dim>>> phi_w;
    std::vector<std::vector<Tensor<1, dim>>> phi_lambda;

    // Face shape functions for each quad node and each dof
    // Only on the face matching the cylinder
    std::vector<std::vector<Tensor<1, dim>>> phi_u_face;
    std::vector<std::vector<Tensor<1, dim>>> phi_w_face;
    std::vector<std::vector<Tensor<1, dim>>> phi_l_face;

    // Face data (for Lagrange multiplier)
    unsigned int n_faces_q_points;
    std::vector<double> face_JxW;
    // unsigned int face_n_dofs;
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

    present_face_velocity_values.resize(n_faces_q_points);
    present_face_mesh_velocity_values.resize(n_faces_q_points);
    present_face_lambda_values.resize(n_faces_q_points);

    // BDF
    previous_velocity_values.resize(2, std::vector<Tensor<1, dim>>(n_q_points));
    previous_displacement_values.resize(2, std::vector<Tensor<1, dim>>(n_q_points));

    div_phi_u.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_u.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    phi_p.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_disp.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_disp.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_disp.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_w.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    phi_lambda.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));

    phi_u_face.resize(n_faces_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    phi_w_face.resize(n_faces_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    phi_l_face.resize(n_faces_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));

    JxW.resize(n_q_points);
    face_JxW.resize(n_faces_q_points);
  }

  template <int dim>
  void ScratchData<dim>::reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                const LA::MPI::Vector &current_solution,
                                const std::vector<LA::MPI::Vector> &previous_solutions,
                                FEValues<dim> &fe_values,
                                FEFaceValues<dim> &fe_face_values)
  {
    fe_values.reinit(cell);

    for (const unsigned int i : fe_values.dof_indices())
      components[i] = fe_values.get_fe().system_to_component_index(i).first;

    const FEValuesExtractors::Vector velocities(0); // 0 -> dim-1
    const FEValuesExtractors::Scalar pressure(dim); // dim -> dim
    const FEValuesExtractors::Vector displacement(dim + 1); // dim+1 -> 2*dim
    const FEValuesExtractors::Vector mesh_velocity(2 * dim + 1); // 2*dim+1 -> 3*dim
    const FEValuesExtractors::Vector lambda(3 * dim + 1); // 3*dim+1 -> 4*dim

    fe_values[velocities].get_function_values(current_solution, present_velocity_values);
    fe_values[velocities].get_function_gradients( current_solution, present_velocity_gradients);
    fe_values[pressure].get_function_values(current_solution, present_pressure_values);
    fe_values[displacement].get_function_values(current_solution, present_displacement_values);
    fe_values[displacement].get_function_gradients(current_solution, present_displacement_gradients);
    fe_values[mesh_velocity].get_function_values(current_solution, present_mesh_velocity_values);

    for(unsigned int i = 0; i < previous_solutions.size(); ++i)
    {
      fe_values[velocities].get_function_values(previous_solutions[i], previous_velocity_values[i]);
      fe_values[displacement].get_function_values(previous_solutions[i], previous_displacement_values[i]);
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
        phi_w[q][k]         = fe_values[mesh_velocity].value(k, q);
        phi_lambda[q][k]    = fe_values[lambda].value(k, q);
      }
    }

    // Reinit fe_face_values on face touching the cylinder, if any.
    for (const auto face_no : cell->face_indices())
    {
      const auto &face = cell->face(face_no);

      if (!(face->at_boundary() && face->boundary_id() == boundary_id))
        continue;

      fe_face_values.reinit(cell, face);

      fe_face_values[velocities].get_function_values(current_solution, present_face_velocity_values);
      fe_face_values[mesh_velocity].get_function_values(current_solution, present_face_mesh_velocity_values);
      fe_face_values[lambda].get_function_values(current_solution, present_face_lambda_values);

      for (unsigned int q = 0; q < n_faces_q_points; ++q)
      {
        face_JxW[q] = fe_face_values.JxW(q);

        for (unsigned int k = 0; k < dofs_per_cell; ++k)
        {
          phi_u_face[q][k] = fe_face_values[velocities].value(k, q);
          phi_w_face[q][k] = fe_face_values[mesh_velocity].value(k, q);
          phi_l_face[q][k] = fe_face_values[lambda].value(k, q);
        }
      }
    }
  }

  template <int dim>
  class MovingMesh
  {
  public:
    MovingMesh(unsigned int velocity_degree,
               unsigned int displacement_degree,
               bool fix_cylinder,
               bool prescribed_displacement,
               unsigned int cylinder_boundary_id,
               double spring_stiffness,
               double viscosity,
               double pseudo_solid_mu,
               double pseudo_solid_lambda,
               unsigned int bdf_order,
               double t0,
               double t1,
               unsigned int nTimeSteps);

    void
    run();

  private:
    void
    make_grid();
    void
    setup_system();
    void
    create_fluid_velocity_mesh_velocity_constraints(const unsigned int boundary_id);
    void
    create_displacement_coupling_placeholder(const unsigned int boundary_id);
    void
    create_lambda_zero_constraints(const unsigned int boundary_id);
    void
    create_displacement_bc_constraints(const unsigned int boundary_id);
    void
    create_zero_constraints();
    void
    create_nonzero_constraints();
    void
    create_sparsity_pattern();
    void
    set_initial_condition();
    void
    apply_zero_constraints();
    void
    apply_nonzero_constraints();
    void
    assemble_matrix(bool first_step);
    void
    assemble_local_matrix(bool first_step, const typename DoFHandler<dim>::active_cell_iterator &cell,
                          ScratchData<dim> &scratchData,
                          FEValues<dim> &fe_values,
                          FEFaceValues<dim> &fe_face_values,
                          LA::MPI::Vector &current_solution,
                          std::vector<LA::MPI::Vector> &previous_solutions,
                          std::vector<types::global_dof_index> &local_dof_indices,
                          FullMatrix<double> &local_matrix);
    void
    assemble_rhs(bool first_step);
    void
    assemble_local_rhs(bool first_step, const typename DoFHandler<dim>::active_cell_iterator &cell,
                       ScratchData<dim> &scratchData,
                       FEValues<dim> &fe_values,
                       FEFaceValues<dim> &fe_face_values,
                       LA::MPI::Vector &current_solution,
                       std::vector<LA::MPI::Vector> &previous_solutions,
                       std::vector<types::global_dof_index> &local_dof_indices,
                       Vector<double> &local_rhs);

    void
    solve_direct(bool first_step);
    void
    solve_iterative(bool first_step);
    void
    solve_newton(
      const double       tolerance,
      const bool         is_initial_step,
      const bool         output_result);
    void
    move_mesh();
    void
    output_results(const unsigned int cycle) const;

    unsigned int velocity_degree;
    unsigned int displacement_degree;
    const bool fix_cylinder;
    const bool prescribed_displacement;
    const unsigned int cylinder_boundary_id;
    const double spring_stiffness;
    const double viscosity;
    const double pseudo_solid_mu;
    const double pseudo_solid_lambda;
    MPI_Comm     mpi_communicator;

    FESystem<dim> fe;

    parallel::fullydistributed::Triangulation<dim> triangulation;
    MappingFE<dim>                                 mapping;

    DoFHandler<dim> dof_handler;

    IndexSet locally_owned_dofs;
    IndexSet locally_relevant_dofs;

    AffineConstraints<double> zero_constraints;
    AffineConstraints<double> nonzero_constraints;

    // Constrain non-boundary lambda dofs to 0
    AffineConstraints<double> lambda_constraints;
    // Constrain boundary displacement dofs to F/k
    AffineConstraints<double> displacement_constraints;

    LA::MPI::SparseMatrix system_matrix;

    // With ghosts (read only)
    LA::MPI::Vector       present_solution;
    LA::MPI::Vector       evaluation_point;

    // Without ghosts (owned)
    LA::MPI::Vector       local_evaluation_point;
    LA::MPI::Vector       newton_update;
    LA::MPI::Vector       system_rhs;

    ConditionalOStream pcout;
    TimerOutput        computing_timer;

    std::vector<Point<dim>> initial_mesh_position;

    std::vector<LA::MPI::Vector> previous_solutions;
    std::vector<double> bdfCoeffs;

    // Time integration
    const double t0;
    const double t1;
    const unsigned int nTimeSteps;
    const double dt;
    double current_time;

    VelocityInlet<dim> inlet_bc;
    CylinderDisplacement<dim> cyl_bc;
  };

  template <int dim>
  MovingMesh<dim>::MovingMesh(unsigned int velocity_degree,
                              unsigned int displacement_degree,
                              bool fix_cylinder,
                              bool prescribed_displacement,
                              unsigned int cylinder_boundary_id,
                              double spring_stiffness,
                              double viscosity,
                              double pseudo_solid_mu,
                              double pseudo_solid_lambda,
                              unsigned int bdf_order,
                              double t0,
                              double t1,
                              unsigned int nTimeSteps)
    : velocity_degree(velocity_degree)
    , displacement_degree(displacement_degree)
    , fix_cylinder(fix_cylinder)
    , prescribed_displacement(prescribed_displacement)
    , cylinder_boundary_id(cylinder_boundary_id)
    , spring_stiffness(spring_stiffness)
    , viscosity(viscosity)
    , pseudo_solid_mu(pseudo_solid_mu)
    , pseudo_solid_lambda(pseudo_solid_lambda)
    , mpi_communicator(MPI_COMM_WORLD)
    , fe(FE_SimplexP<dim>(velocity_degree),
         dim,
         FE_SimplexP<dim>(velocity_degree - 1),
         1,
         FE_SimplexP<dim>(displacement_degree), // Displacement
         dim,
         FE_SimplexP<dim>(displacement_degree), // Mesh velocity
         dim,
         FE_SimplexP<dim>(velocity_degree - 1), // Lagrange multiplier
         dim
         )
    , triangulation(mpi_communicator)
    , mapping(FE_SimplexP<dim>(1))
    , dof_handler(triangulation)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , t0(t0)
    , t1(t1)
    , nTimeSteps(nTimeSteps)
    , dt((t1 - t0) / nTimeSteps)
    , current_time(t0)
    , inlet_bc(VelocityInlet<dim>(current_time))
    , cyl_bc(CylinderDisplacement<dim>(current_time))
  {
    switch(bdf_order)
    {
      case 1:
        bdfCoeffs.resize(2);
        bdfCoeffs[0] =  1. / dt;
        bdfCoeffs[1] = -1. / dt;
        break;

      case 2:
        bdfCoeffs.resize(3);
        bdfCoeffs[0] =  3. / (2. * dt);
        bdfCoeffs[1] = -2. / dt;
        bdfCoeffs[2] =  1. / (2. * dt);
        break;
      default:
        throw std::runtime_error("Can only choose BDF1 or BDF2 time integration method");
    }
  }

  template <int dim>
  void
  MovingMesh<dim>::make_grid()
  {
    pcout << "Making grid..." << std::endl;

    const unsigned int mpi_size =
      Utilities::MPI::n_mpi_processes(mpi_communicator);

    Triangulation<dim> serial_tria;

    GridIn<dim> grid_in;
    grid_in.attach_triangulation(serial_tria);

    std::string meshFile = "";

    if constexpr (dim == 2)
    {
      // meshFile = "../data/meshes/cylinderUltraCoarse.msh";
      // meshFile = "../data/meshes/cylinderSuperCoarse.msh";
      meshFile = "../data/meshes/cylinderCoarse.msh";
      // meshFile = "../data/meshes/cylinderMedium.msh";
      // meshFile = "../data/meshes/cylinderFine.msh";
      
    }
    else
    {
      meshFile = "../data/meshes/cylinderCoarse3D.msh";
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
    // copy_triangulation does not seems to work, so maybe give reference to the mesh
    triangulation.create_triangulation(description);

    // Save initial position of the mesh vertices
    initial_mesh_position.resize(triangulation.n_vertices());
    for (auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto v : cell->vertex_indices())
      {
        const unsigned int global_vertex_index = cell->vertex_index(v);
        initial_mesh_position[global_vertex_index] = cell->vertex(v);
      }
    }
  }

  template <int dim>
  void
  MovingMesh<dim>::setup_system()
  {
    TimerOutput::Scope t(computing_timer, "setup");

    dof_handler.distribute_dofs(fe);
    pcout << "Number of degrees of freedom: " << dof_handler.n_dofs() << std::endl;

    locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(this->dof_handler);

    //
    // Initialize parallel vectors
    //
    present_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    evaluation_point.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    local_evaluation_point.reinit(locally_owned_dofs, mpi_communicator);
    newton_update.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    // BDF solutions
    previous_solutions.resize(bdfCoeffs.size() - 1); // 1 or 2
    for(auto &previous_sol : previous_solutions)
      previous_sol.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
  }

  template <int dim>
  void
  MovingMesh<dim>::create_sparsity_pattern()
  {
    //
    // Sparsity pattern and allocate matrix
    // After the constraints have been defined
    //
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints, false);
    SparsityTools::distribute_sparsity_pattern(
      dsp,
      locally_owned_dofs,
      mpi_communicator,
      locally_relevant_dofs);

    system_matrix.reinit(locally_owned_dofs, locally_owned_dofs, dsp, mpi_communicator);
  }

  template <int dim>
  void
  MovingMesh<dim>::set_initial_condition()
  {
    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    const FEValuesExtractors::Vector displacement(dim+1);
    const FEValuesExtractors::Vector mesh_velocity(2*dim+1);
    const FEValuesExtractors::Vector lambda(3*dim+1);

    // Velocity initial condition
    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialCondition<dim>(),
                             newton_update,
                             fe.component_mask(velocities));

    // Pressure initial condition
    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialCondition<dim>(),
                             newton_update,
                             fe.component_mask(pressure));

    // Displacement initial condition
    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialCondition<dim>(),
                             newton_update,
                             fe.component_mask(displacement));

    // Mesh velocity initial condition
    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialCondition<dim>(),
                             newton_update,
                             fe.component_mask(mesh_velocity));

    // Lambda initial condition
    VectorTools::interpolate(mapping,
                             dof_handler,
                             InitialCondition<dim>(),
                             newton_update,
                             fe.component_mask(lambda));

    // Apply non-homogeneous Dirichlet BC and set as current solution
    nonzero_constraints.distribute(newton_update);
    present_solution = newton_update;

    // Dirty copy of the initial condition for BDF2 as well (-:
    for(auto &sol : previous_solutions)
      sol = present_solution;
  }

  template <int dim>
  void MovingMesh<dim>::create_lambda_zero_constraints(const unsigned int boundary_id)
  {
    lambda_constraints.clear();
    lambda_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

     const std::map<types::global_dof_index, Point<2>> dof_location_map =
      DoFTools::map_dofs_to_support_points(mapping, dof_handler);
  
    std::ofstream dof_location_file("dofs.gnuplot");
    DoFTools::write_gnuplot_dof_support_point_info(dof_location_file,
                                                   dof_location_map);

    // Apply a constraint dof = 0 for all lambda dofs that are not on the prescribed boundary
    std::set<types::global_dof_index> unconstrained_lambda_dofs;
    std::vector<types::global_dof_index> face_dofs(fe.n_dofs_per_face());

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto f : cell->face_indices())
      {
        if (cell->face(f)->at_boundary() &&
            cell->face(f)->boundary_id() == boundary_id)
        {
          cell->face(f)->get_dof_indices(face_dofs);
          for (unsigned int idof = 0; idof < fe.n_dofs_per_face(); ++idof)
          {
            const unsigned int component = fe.face_system_to_component_index(idof).first;
            bool is_lambda_component = 3*dim + 1 <= component && component <= 4*dim;

            if (fe.has_support_on_face(idof, f) && is_lambda_component)
            {
              // Lambda DoF on the prescribed boundary: do not constrain
              unconstrained_lambda_dofs.insert(face_dofs[idof]);
            }
          }
        }
      }
    }

    // Add zero constraints to all lambda DOFs *not* in the boundary set
    const FEValuesExtractors::Vector lambda(3 * dim + 1);
    IndexSet lambda_dofs = DoFTools::extract_dofs(dof_handler, fe.component_mask(lambda));

    for (const auto dof : lambda_dofs)
    {
      // Only constrain owned DOFs
      if(!locally_owned_dofs.is_element(dof))
        continue;

      if (unconstrained_lambda_dofs.count(dof) == 0)
      {
        // lambda_constraints.add_line(dof); // Set dof to zero (by default)
        lambda_constraints.constrain_dof_to_zero(dof); // More readable (-:
      }
    }

    lambda_constraints.close();
  }

  template <int dim>
  void MovingMesh<dim>::create_displacement_bc_constraints(const unsigned int boundary_id)
  {
    // Set constraints on the displacement DoFs on the cylinder.
    // These are set to the weighted sum of the lagrange multiplier DoFs,
    // that is, to the integral of lambda (up to the spring coefficient k)

    // Each displacement DoF is constrained by all lambda DoFs on the cylinder.
    // Those lambda DoF may not be owned or even ghosts, so we need to  indicate that:
    // -to the constraint can store constraints for all lambda DoFs.
    // -the displacement DoFs are coupled to all cylinder lambda DoFs. (?)

    ///////////////////////////////////////////////
    // Get and synchronize the lambda DoFs on the cylinder
    const FEValuesExtractors::Vector displacement(dim + 1);
    const FEValuesExtractors::Vector lambda(3 * dim + 1);
    std::set<types::boundary_id> boundary_ids;
    boundary_ids.insert(boundary_id);

    IndexSet local_lambda_dofs = 
      DoFTools::extract_boundary_dofs(dof_handler, fe.component_mask(lambda), boundary_ids);
    IndexSet local_displacement_dofs = 
      DoFTools::extract_boundary_dofs(dof_handler, fe.component_mask(displacement), boundary_ids);
    
    const unsigned int n_local_lambda_dofs = local_lambda_dofs.n_elements();

    // std::cout << local_lambda_dofs.n_elements() << " lambda dofs on proc " << Utilities::MPI::this_mpi_process(mpi_communicator) << std::endl;
    
    local_lambda_dofs = local_lambda_dofs & locally_owned_dofs;
    local_displacement_dofs = local_displacement_dofs & locally_owned_dofs;
    
    // std::cout << local_lambda_dofs.n_elements() << " owned lambda dofs on proc " << Utilities::MPI::this_mpi_process(mpi_communicator) << std::endl;

    // Convert local IndexSet to vector
    std::vector<types::global_dof_index> local_lambda_dofs_vec = local_lambda_dofs.get_index_vector();

    // Gather all lists to all processes
    std::vector<std::vector<types::global_dof_index>> gathered_dofs =
      Utilities::MPI::all_gather(mpi_communicator, local_lambda_dofs_vec);

    std::vector<types::global_dof_index> gathered_dofs_flattened;
    for (const auto &vec : gathered_dofs)
      gathered_dofs_flattened.insert(gathered_dofs_flattened.end(), vec.begin(), vec.end());
    
    // std::cout << gathered_dofs_flattened.size() << " gathered lambda dofs on proc " << Utilities::MPI::this_mpi_process(mpi_communicator) << std::endl;

    std::sort(gathered_dofs_flattened.begin(), gathered_dofs_flattened.end());

    // IndexSet locally_stored_constraints(dof_handler.n_dofs());
    // locally_stored_constraints = locally_relevant_dofs;
    // // Add lambda dofs
    // locally_stored_constraints.add_indices(gathered_dofs_flattened.begin(), gathered_dofs_flattened.end());
    // locally_stored_constraints.compress();

    // Alternatively, simply add the lambda DoFs to the list of locally relevant DoFs:
    // Do this only if partition contains a chunk of the cylinder
    if(n_local_lambda_dofs > 0)
    {
      locally_relevant_dofs.add_indices(gathered_dofs_flattened.begin(), gathered_dofs_flattened.end());
      locally_relevant_dofs.compress();
    }

    displacement_constraints.clear();
    // displacement_constraints.reinit(locally_owned_dofs, locally_stored_constraints);
    displacement_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

    // Compute the integral weights. This is done only once since the 
    // constraint for each dimension is the same for all displacement dofs (rigid cylinder)
    std::vector<std::map<unsigned int, double>> coeffs(dim);

    const QGaussSimplex<dim-1> face_quadrature(8);

    FEFaceValues<dim> fe_face_values(mapping,
                            fe,
                            face_quadrature,
                            update_values | update_quadrature_points | update_JxW_values);

    std::vector<types::global_dof_index> face_dofs(fe.n_dofs_per_face());

    const unsigned int displacement_lower = dim + 1;
    const unsigned int displacement_upper = displacement_lower + dim;
    const unsigned int lambda_lower       = 3 * dim + 1;
    const unsigned int lambda_upper       = lambda_lower + dim;

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (const auto face_no : cell->face_indices())
      {
        const auto &face = cell->face(face_no);

        if (!(face->at_boundary() && face->boundary_id() == boundary_id))
          continue;

        fe_face_values.reinit(cell, face);
        face->get_dof_indices(face_dofs);

        for (unsigned int q = 0; q < face_quadrature.size(); ++q)
        {
          const double JxW = fe_face_values.JxW(q);

          for (unsigned int i = 0; i < fe.n_dofs_per_face(); ++i)
          {
            const unsigned int comp_i  = fe.face_system_to_component_index(i, face_no).first;

            // if (lambda_lower <= comp_i && comp_i < lambda_upper)
            // {
            //   std::cout << "proc " << Utilities::MPI::this_mpi_process(mpi_communicator) << " lambda dof " << face_dofs[i]
            //   << " - is owned : " << locally_owned_dofs.is_element(face_dofs[i])
            //   << " - is ghost : " << locally_relevant_dofs.is_element(face_dofs[i])
            //   << std::endl;
            // }

            // Here we need to account for ghost DoF, which contribute to the integral on this element
            // if(!locally_owned_dofs.is_element(face_dofs[i]))
            if(!locally_relevant_dofs.is_element(face_dofs[i]))
              continue;

            // Only consider components used in lambda
            if (lambda_lower <= comp_i && comp_i < lambda_upper)
            {
              const types::global_dof_index lambda_dof = face_dofs[i];
              // Return the only nonzero component of the vector-valued shape function for lambda
              const unsigned int d = comp_i - lambda_lower;
              const double phi_i = fe_face_values.shape_value(i, q);
              const double phi_i_comp = fe_face_values.shape_value_component(i, q, d);
              coeffs[d][lambda_dof] += phi_i * JxW;
            }
          }
        }
      }
    }

    // Gather the constraint weights
    std::vector<std::vector<std::pair<unsigned int, double>>> gathered_coeffs(dim);
    std::vector<std::map<unsigned int, double>> gathered_coeffs_map(dim);

    const double spring_stiffness_ramped = spring_stiffness;
    // const double spring_stiffness_ramped = springStiffnessRamp(this->current_time, 1., 5., 1., spring_stiffness);
    // double spring_stiffness_ramped = springStiffnessRamp(this->current_time, 0., 5., 10., 1., 1., spring_stiffness);

    // if(this->current_time < 5.)
    //   spring_stiffness_ramped = 1.;
    // else
    //   spring_stiffness_ramped = 0.1;

    // pcout << "Stiffness = " << spring_stiffness_ramped << std::endl;

    for(unsigned int d = 0; d < dim; ++d)
    {
      std::vector<std::pair<unsigned int, double>> coeffs_vector(coeffs[d].begin(), coeffs[d].end());
      std::vector<std::vector<std::pair<unsigned int, double>>> gathered =
        Utilities::MPI::all_gather(MPI_COMM_WORLD, coeffs_vector);

      // Flatten into single vector
      // Duplicates are ignored by add_entries
      // for (const auto &vec : gathered)
      //   gathered_coeffs[d].insert(gathered_coeffs[d].end(), vec.begin(), vec.end());

      // Put back into map and sum contributions to same DoF from different processes
      for (const auto &vec : gathered)
        for(const auto &pair : vec)
          gathered_coeffs_map[d][pair.first] += pair.second;

      gathered_coeffs[d].insert(gathered_coeffs[d].end(), gathered_coeffs_map[d].begin(), gathered_coeffs_map[d].end());

      for (auto &vec : gathered_coeffs)
        for(auto &pair : vec)
          pair.second /= spring_stiffness_ramped;
    }

    // for(const auto &p : gathered_coeffs[0])
    // {
    //   std::cout << "Proc " << Utilities::MPI::this_mpi_process(mpi_communicator)
    //     << "dof " << p.first << " - weight " << p.second << std::endl;
    // }

    //
    // Then constrain displacement DoFs on cylinder
    //
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (const auto face_no : cell->face_indices())
      {
        const auto &face = cell->face(face_no);

        if (!(face->at_boundary() && face->boundary_id() == boundary_id))
          continue;

        fe_face_values.reinit(cell, face);
        face->get_dof_indices(face_dofs);

        for (unsigned int i = 0; i < fe.n_dofs_per_face(); ++i)
        {
          if(!locally_owned_dofs.is_element(face_dofs[i]))
            continue;

          const unsigned int comp_i  = fe.face_system_to_component_index(i, face_no).first;
          // const unsigned int shape_i = fe.face_system_to_component_index(i, face_no).second;
          
          bool is_displacement = displacement_lower <= comp_i && comp_i < displacement_upper;

          if (is_displacement && fe.has_support_on_face(i, face_no))
          {
            const unsigned int d = comp_i - displacement_lower;
            displacement_constraints.add_line(face_dofs[i]);
            // displacement_constraints.add_constraint(face_dofs[i], constraint_coeffs[d], 0.);
            // displacement_constraints.add_entries(face_dofs[i], coeffs_vectors[d]);
            displacement_constraints.add_entries(face_dofs[i], gathered_coeffs[d]);
          }
        }
      }
    }

    // displacement_constraints.make_consistent_in_parallel(locally_owned_dofs, displacement_constraints.get_local_lines(), mpi_communicator);
    // displacement_constraints.make_consistent_in_parallel(locally_owned_dofs, locally_stored_constraints, mpi_communicator);
    displacement_constraints.make_consistent_in_parallel(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    displacement_constraints.close();
  }

  template <int dim>
  void
  MovingMesh<dim>::create_zero_constraints()
  {
    pcout << "Creating zero constraints..." << std::endl;
    zero_constraints.clear();
    zero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, zero_constraints);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    const FEValuesExtractors::Vector displacement(dim + 1);
    const FEValuesExtractors::Vector mesh_velocity(2 * dim + 1);
    const FEValuesExtractors::Vector lambda(3 * dim + 1);

    const unsigned int n_total_comp = 4 * dim + 1;

    // Boundaries where Dirichlet BC are applied,
    // where the Newton increment should be zero.
    std::vector<unsigned int> dirichlet_boundaries_u, dirichlet_boundaries_chi;
    if constexpr(dim == 2)
    {
      // Boundary 1 is the Outlet, which is free for the flow
      dirichlet_boundaries_u   = {2, 3};
      dirichlet_boundaries_chi = {1, 2, 3};
    }

    for(unsigned int id : dirichlet_boundaries_u)
    {
      VectorTools::interpolate_boundary_values(mapping,
                                               dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(n_total_comp),
                                               zero_constraints,
                                               fe.component_mask(velocities));
    }

    if(fix_cylinder || prescribed_displacement)
      dirichlet_boundaries_chi.push_back(cylinder_boundary_id);

    for(unsigned int id : dirichlet_boundaries_chi)
    {
      VectorTools::interpolate_boundary_values(mapping,
                                               dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(n_total_comp),
                                               zero_constraints,
                                               fe.component_mask(displacement));
    }

    zero_constraints.close();

    // Lambda constraints have to be enforced at each Newton iteration
    // Add them to both sets of constraints?
    zero_constraints.merge(lambda_constraints, AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed, true);
    // Same for displacement constraints?
    zero_constraints.merge(displacement_constraints, AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed, true);

    zero_constraints.close();
  }

  template <int dim>
  void MovingMesh<dim>::create_fluid_velocity_mesh_velocity_constraints(const unsigned int boundary_id)
  {
    std::vector<types::global_dof_index> local_face_dof_indices(fe.n_dofs_per_face());

    // To get the physical positions of the DOFs
    // std::vector<Point<dim>> support_points(dof_handler.n_dofs());
    // DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (!cell->is_locally_owned())
        continue;

      for (const auto face_no : cell->face_indices())
      {
        if (cell->face(face_no)->at_boundary() && (cell->face(face_no)->boundary_id() == boundary_id))
        {
          cell->face(face_no)->get_dof_indices(local_face_dof_indices);

          for(unsigned int i = 0; i < fe.n_dofs_per_face(); ++i)
          {
            const unsigned dof_fluid_velocity = local_face_dof_indices[i];
            
            // Only constrain owned DOFs
            if(!locally_owned_dofs.is_element(dof_fluid_velocity))
              continue;

            const unsigned comp_index  = fe.face_system_to_component_index(i, face_no).first;
            const unsigned shape_index = fe.face_system_to_component_index(i, face_no).second;


            // Only constrain fluid velocity
            if(comp_index >= dim)
              continue;

            // The associated mesh velocity dof component
            const unsigned int mesh_velocity_comp_index = comp_index + 2 * dim + 1;

            bool matched = false;

            for(unsigned int j = 0; j < fe.n_dofs_per_face(); ++j)
            {
              const unsigned comp_index_j  = fe.face_system_to_component_index(j, face_no).first;
              const unsigned shape_index_j = fe.face_system_to_component_index(j, face_no).second;

              // Find DOFs with matching shape_index
              if(shape_index != shape_index_j)
                continue;

              // Find matching mesh_velocity DOF
              if(mesh_velocity_comp_index != comp_index_j)
                continue;

              const unsigned dof_mesh_velocity  = local_face_dof_indices[j];

              nonzero_constraints.add_line(dof_fluid_velocity);
              nonzero_constraints.add_entry(dof_fluid_velocity, dof_mesh_velocity, 1.0);
              // nonzero_constraints.set_inhomogeneity(dof_fluid_velocity, 0.0); // optional

              matched = true;

              break;
            }

            if(!matched)
            {
              throw std::runtime_error("Could not match a fluid velocity DOF with a mesh velocity DOF on the cylinder. "
                "This is probably because mesh velocity is represented with a different FE space.");
            }
          }
        }
      }
    }
  }

  template <int dim>
  void MovingMesh<dim>::create_nonzero_constraints()
  {
    pcout << "Creating nonzero constraints..." << std::endl;
    nonzero_constraints.clear();
    nonzero_constraints.reinit(locally_owned_dofs, locally_relevant_dofs);
    DoFTools::make_hanging_node_constraints(dof_handler, nonzero_constraints);

    const FEValuesExtractors::Vector velocities(0);
    const FEValuesExtractors::Scalar pressure(dim);
    const FEValuesExtractors::Vector displacement(dim + 1);
    const FEValuesExtractors::Vector mesh_velocity(2 * dim + 1);

    const unsigned int n_total_comp = 4 * dim + 1;

    unsigned int inletBoundary = (dim == 2) ? 2 : 5;

    std::vector<unsigned int> noSlipBoundaries, noMeshMovementBoundaries;
    if constexpr(dim == 2)
    {
      // Boundary 1 is the Outlet, which is free
      noSlipBoundaries           = {3};
      noMeshMovementBoundaries   = {1, 2, 3};
    }

    //
    // Create affine constraints relating velocity and mesh velocity on cylinder
    //
  // #if defined(STRONG_NO_SLIP)
  //   VectorTools::interpolate_boundary_values(mapping,
  //                                            dof_handler,
  //                                            cylinder_boundary_id,
  //                                            Functions::ZeroFunction<dim>(n_total_comp),
  //                                            nonzero_constraints,
  //                                            fe.component_mask(velocities));
  // #else
  //   if(!fix_cylinder)
  //     this->create_fluid_velocity_mesh_velocity_constraints(cylinder_boundary_id);
  // #endif

    // Velocity inlet
    this->inlet_bc.set_time(current_time);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             inletBoundary,
                                             inlet_bc,
                                             nonzero_constraints,
                                             fe.component_mask(velocities));

    // No slip boundaries
    for(unsigned int id : noSlipBoundaries)
    {
      VectorTools::interpolate_boundary_values(mapping,
                                               dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(n_total_comp),
                                               nonzero_constraints,
                                               fe.component_mask(velocities));
    }

    // Displacement BC on cylinder
    if(fix_cylinder)
    {
      VectorTools::interpolate_boundary_values(mapping,
                                               dof_handler,
                                               cylinder_boundary_id,
                                               Functions::ZeroFunction<dim>(n_total_comp),
                                               nonzero_constraints,
                                               fe.component_mask(displacement));
    }

    if(prescribed_displacement)
    {
      this->cyl_bc.set_time(current_time);
      VectorTools::interpolate_boundary_values(mapping,
                                               dof_handler,
                                               cylinder_boundary_id,
                                               cyl_bc,
                                               nonzero_constraints,
                                               fe.component_mask(displacement));
    }


    // No displacement
    for(unsigned int id : noMeshMovementBoundaries)
    {
      VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             id,
                                             Functions::ZeroFunction<dim>(n_total_comp),
                                             nonzero_constraints,
                                             fe.component_mask(displacement));
    }

    nonzero_constraints.close();

    // Lambda constraints have to be enforced at each Newton iteration
    // Add them to both sets of constraints?
    nonzero_constraints.merge(lambda_constraints, AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed, true);
    // Same for displacement constraints?
    nonzero_constraints.merge(displacement_constraints, AffineConstraints<double>::MergeConflictBehavior::no_conflicts_allowed, true);

    nonzero_constraints.close();
  }

  template <int dim>
  void
  MovingMesh<dim>::apply_zero_constraints()
  {
    pcout << "Applying zero constraints..." << std::endl;
    zero_constraints.distribute(local_evaluation_point);
    present_solution = local_evaluation_point;
  }

  template <int dim>
  void
  MovingMesh<dim>::apply_nonzero_constraints()
  {
    pcout << "Applying nonzero constraints..." << std::endl;
    nonzero_constraints.distribute(local_evaluation_point);
    present_solution = local_evaluation_point;
  }

  template <int dim>
  void
  MovingMesh<dim>::assemble_matrix(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Assemble matrix");

    system_matrix = 0;

    const QGaussSimplex<dim> quadrature_formula(4);
    const QGaussSimplex<dim-1> face_quadrature(8);

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(mapping,
                            fe,
                            face_quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature.size();
    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();
    
    ScratchData<dim> scratchData(n_q_points, n_face_q_points, dofs_per_cell, cylinder_boundary_id);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      this->assemble_local_matrix(first_step, cell, scratchData, fe_values, fe_face_values, evaluation_point, previous_solutions, local_dof_indices, local_matrix);
    }

    system_matrix.compress(VectorOperation::add);
  }

  template <int dim>
  void
  MovingMesh<dim>::assemble_local_matrix(bool first_step,
                                         const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         ScratchData<dim> &scratchData,
                                         FEValues<dim> &fe_values,
                                         FEFaceValues<dim> &fe_face_values,
                                         LA::MPI::Vector &current_solution,
                                         std::vector<LA::MPI::Vector> &previous_solutions,
                                         std::vector<types::global_dof_index> &local_dof_indices,
                                         FullMatrix<double> &local_matrix)
  {
    if (!cell->is_locally_owned())
      return;

    scratchData.reinit(cell, current_solution, previous_solutions, fe_values, fe_face_values);

    local_matrix = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &present_velocity_values    = scratchData.present_velocity_values[q];
      const auto &present_velocity_gradients = scratchData.present_velocity_gradients[q];

      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];

      const auto &phi_p      = scratchData.phi_p[q];

      const auto &present_mesh_velocity_values = scratchData.present_mesh_velocity_values[q];

      const auto &phi_x      = scratchData.phi_disp[q];
      const auto &grad_phi_x = scratchData.grad_phi_disp[q];
      const auto &div_phi_x  = scratchData.div_phi_disp[q];

      const auto &phi_w      = scratchData.phi_w[q];

      const auto &phi_lambda = scratchData.phi_lambda[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const unsigned int component_i = scratchData.components[i];

        for (unsigned int j = 0; j < scratchData.dofs_per_cell; ++j)
        {
          const unsigned int component_j = scratchData.components[j];

          double local_matrix_ij = 0.;
          
          // Velocity - velocity block
          if(component_i < dim && component_j < dim)
          {
            // Time-dependent
            local_matrix_ij += bdfCoeffs[0] * phi_u[i] * phi_u[j];

            // Convective
            local_matrix_ij +=
              (grad_phi_u[j] * present_velocity_values
                + present_velocity_gradients * phi_u[j]) * phi_u[i];

            // ALE acceleration : - w dot grad(delta u) 
            local_matrix_ij += - (grad_phi_u[j] * present_mesh_velocity_values) * phi_u[i];

            if(component_i == component_j)
            {
              // Diffusive
              local_matrix_ij += viscosity * grad_phi_u[i][component_i] * grad_phi_u[j][component_j];
            }
          }

          // Pressure - velocity block
          if(component_i == dim)
          {
            // Incompressibility
            local_matrix_ij += - phi_p[i] * div_phi_u[j];
          }

          // Velocity - pressure block
          if(component_j == dim)
          {
            // Incompressibility
            local_matrix_ij += - div_phi_u[i] * phi_p[j];
          }

          // Displacement - displacement block (chi-chi)
          if(dim + 1 <= component_i && component_i < 2 * dim + 1 &&
             dim + 1 <= component_j && component_j < 2 * dim + 1)
          {
            // Linear elasticity for pseudo-solid
            local_matrix_ij += 
              pseudo_solid_lambda * div_phi_x[j] * div_phi_x[i]
                + pseudo_solid_mu * scalar_product((grad_phi_x[i] + transpose(grad_phi_x[i])), grad_phi_x[j]);
          }

          // Velocity - mesh velocity block (u-chi)
          if(component_i < dim && 2 * dim + 1 <= component_j)
          {
            // ALE acceleration : - delta w dot grad(u) 
            local_matrix_ij += - (present_velocity_gradients * phi_w[j]) * phi_u[i];
          }

          // Mesh velocity - displacement block (w-chi)
          if(2 * dim + 1 <= component_i && dim + 1 <= component_j && component_j < 2 * dim + 1)
          {
            // Time-dependent (mesh velocity equation)
            local_matrix_ij += - bdfCoeffs[0] * phi_x[j] * phi_w[i];
          }

          // Mesh velocity - Mesh velocity block (w-w)
          if(2 * dim + 1 <= component_i && 2 * dim + 1 <= component_j)
          {
            // Transient mass matrix
            local_matrix_ij += phi_w[i] * phi_w[j];
          }

          local_matrix_ij *= JxW;
          local_matrix(i, j) += local_matrix_ij;
        }
      }
    }

    const FEValuesExtractors::Vector velocities(0); // 0 -> dim-1
    const FEValuesExtractors::Vector lambda(3 * dim + 1); // 3*dim+1 -> 4*dim
    const unsigned int n_faces_q_points = fe_face_values.get_quadrature().size();

    // Face contributions
    if(cell->at_boundary())
    {
      for (const auto &face : cell->face_iterators())
      {
        if(face->at_boundary() && face->boundary_id() == cylinder_boundary_id)
        {
          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            const double JxW  = scratchData.face_JxW[q];
            const auto &phi_u = scratchData.phi_u_face[q];
            const auto &phi_w = scratchData.phi_w_face[q];
            const auto &phi_l = scratchData.phi_l_face[q];

            for(unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            {
              for(unsigned int j = 0; j < fe.n_dofs_per_cell(); ++j)
              {
                local_matrix(i, j) += phi_l[j] * phi_u[i] * JxW;
                // local_matrix(i, j) += phi_u[j] * phi_l[i] * JxW;
                local_matrix(i, j) += (phi_u[j] - phi_w[j]) * phi_l[i] * JxW;
              }
            }

            // for (const unsigned int i : fe_face_values.dof_indices())
            // {
            //   const unsigned int component_i = fe.system_to_component_index(i).first;

            //   for (const unsigned int j : fe_face_values.dof_indices())
            //   {
            //     const unsigned int component_j = fe.system_to_component_index(j).first;

            //     // U-lambda
            //     if(component_i < dim && 3 * dim + 1 <= component_j)
            //     {
            //       local_matrix(i, j) += phi_l[j] * phi_u[i] * fe_face_values.JxW(q);
            //     }

            //     // Lambda-u
            //     if(3 * dim + 1 <= component_i && component_j < dim)
            //     {
            //       local_matrix(i, j) += phi_u[j] * phi_l[i] * fe_face_values.JxW(q);
            //     }
            //   }
            // }
          }
        }
      }
    }

    cell->get_dof_indices(local_dof_indices);
    if(first_step)
    {
      nonzero_constraints.distribute_local_to_global(local_matrix, local_dof_indices, system_matrix);
    }
    else
    {
      zero_constraints.distribute_local_to_global(local_matrix, local_dof_indices, system_matrix);
    }
  }

  template <int dim>
  void
  MovingMesh<dim>::assemble_rhs(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Assemble RHS");

    system_rhs = 0;

    const QGaussSimplex<dim> quadrature_formula(4);
    const QGaussSimplex<dim-1> face_quadrature(8);

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);
    FEFaceValues<dim> fe_face_values(mapping,
                            fe,
                            face_quadrature,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int n_q_points      = quadrature_formula.size();
    const unsigned int n_face_q_points = face_quadrature.size();
    const unsigned int dofs_per_cell   = fe.n_dofs_per_cell();
    
    ScratchData<dim> scratchData(n_q_points, n_face_q_points, dofs_per_cell, cylinder_boundary_id);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      this->assemble_local_rhs(first_step, cell, scratchData, fe_values, fe_face_values, evaluation_point, previous_solutions, local_dof_indices, local_rhs);
    }

    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void
  MovingMesh<dim>::assemble_local_rhs(bool first_step,
                                         const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         ScratchData<dim> &scratchData,
                                         FEValues<dim> &fe_values,
                                         FEFaceValues<dim> &fe_face_values,
                                         LA::MPI::Vector &current_solution,
                                         std::vector<LA::MPI::Vector> &previous_solutions,
                                         std::vector<types::global_dof_index> &local_dof_indices,
                                         Vector<double> &local_rhs)
  {
    scratchData.reinit(cell, current_solution, previous_solutions, fe_values, fe_face_values);

    local_rhs = 0;

    const unsigned int nBDF = bdfCoeffs.size();
    std::vector<Tensor<1, dim>> velocity(nBDF);
    std::vector<Tensor<1, dim>> displacement(nBDF);

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const double JxW = scratchData.JxW[q];

      const auto &present_velocity_values        = scratchData.present_velocity_values[q];
      const auto &present_velocity_gradients     = scratchData.present_velocity_gradients[q];
      const auto &present_pressure_values        = scratchData.present_pressure_values[q];
      const auto &present_displacement_values    = scratchData.present_displacement_values[q];
      const auto &present_displacement_gradients = scratchData.present_displacement_gradients[q];
      const auto &present_mesh_velocity_values   = scratchData.present_mesh_velocity_values[q];

      double present_velocity_divergence =
                  trace(present_velocity_gradients);
      double present_displacement_divergence =
                  trace(present_displacement_gradients);

      // BDF
      velocity[0]     = present_velocity_values;
      displacement[0] = present_displacement_values;
      for(unsigned int i = 1; i < nBDF; ++i)
      {
        velocity[i]     = scratchData.previous_velocity_values[i-1][q];
        displacement[i] = scratchData.previous_displacement_values[i-1][q];
      }

      const Tensor<1, dim> uDotGradU = present_velocity_gradients * present_velocity_values;
      const Tensor<1, dim> wDotGradU = present_velocity_gradients * present_mesh_velocity_values;

      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];

      const auto &phi_p      = scratchData.phi_p[q];

      // const auto &phi_x      = scratchData.phi_disp[q];
      const auto &grad_phi_x = scratchData.grad_phi_disp[q];
      const auto &div_phi_x  = scratchData.div_phi_disp[q];

      const auto &phi_w      = scratchData.phi_w[q];

      const auto &phi_lambda = scratchData.phi_lambda[q];

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
        const auto present_displacement_gradient_sym = present_displacement_gradients + transpose(present_displacement_gradients);

        double local_rhs_i = - (

          // Navier-Stokes ALE:
          // Convective
          uDotGradU * phi_u[i]

          // ALE acceleration
          - wDotGradU * phi_u[i]
          
          // Diffusive
          + viscosity * scalar_product(present_velocity_gradients, grad_phi_u[i])

          // Pressure gradient
          - div_phi_u[i] * present_pressure_values

          // Incompressibility
          - phi_p[i] * present_velocity_divergence

          // Linear elasticity
          + pseudo_solid_lambda * present_displacement_divergence * div_phi_x[i]
          + pseudo_solid_mu * scalar_product(present_displacement_gradient_sym, grad_phi_x[i])

          // Mesh velocity
          + present_mesh_velocity_values * phi_w[i]

          ) * JxW;

          // Transient terms:
          for(unsigned int iBDF = 0; iBDF < nBDF; ++iBDF)
          {
            local_rhs_i -= bdfCoeffs[iBDF] * velocity[iBDF] * phi_u[i] * JxW;
            local_rhs_i -= - bdfCoeffs[iBDF] * displacement[iBDF] * phi_w[i] * JxW;
          }
          
          local_rhs(i) += local_rhs_i;
      }
    }

    const FEValuesExtractors::Vector velocities(0); // 0 -> dim-1
    const FEValuesExtractors::Vector lambda(3 * dim + 1); // 3*dim+1 -> 4*dim
    const unsigned int n_faces_q_points = fe_face_values.get_quadrature().size();

    //
    // Face contributions (Lagrange multiplier)
    //
    if(cell->at_boundary())
    {
      for (const auto &face : cell->face_iterators())
      {
        if(face->at_boundary() && face->boundary_id() == cylinder_boundary_id)
        {
          const auto &current_u = scratchData.present_face_velocity_values;
          const auto &current_w = scratchData.present_face_mesh_velocity_values;
          const auto &current_l = scratchData.present_face_lambda_values;

          for (unsigned int q = 0; q < n_faces_q_points; ++q)
          {
            const double JxW  = scratchData.face_JxW[q];
            const auto &phi_u = scratchData.phi_u_face[q];
            const auto &phi_l = scratchData.phi_l_face[q];

            for(unsigned int i = 0; i < fe.n_dofs_per_cell(); ++i)
            {
              local_rhs(i) -= current_l[q] * phi_u[i] * JxW;
              // local_rhs(i) -= current_u[q] * phi_l[i] * JxW;
              local_rhs(i) -= (current_u[q] - current_w[q]) * phi_l[i] * JxW;
            }

            // // Get shape functions on face for this quad nodes
            // for (const unsigned int i : fe_face_values.dof_indices())
            // {
            //   phi_u[i] = fe_face_values[velocities].value(i, q);
            //   phi_l[i] = fe_face_values[lambda].value(i, q);
            // }

            // for (const unsigned int i : fe_face_values.dof_indices())
            // {
            //   // const unsigned int component_i = fe.system_to_component_index(i).first;

            //   // U-lambda
            //   if(component_i < dim)
            //   {
            //     local_rhs(i) += current_lambda[q] * phi_u[i] * fe_face_values.JxW(q);
            //   }

            //   // Lambda-u
            //   if(3 * dim + 1 <= component_i)
            //   {
            //     local_rhs(i) += current_u[q] * phi_l[i] * fe_face_values.JxW(q);
            //   }
            // }
          }
        }
      }
    }

    cell->get_dof_indices(local_dof_indices);
    if(first_step)
      nonzero_constraints.distribute_local_to_global(local_rhs, local_dof_indices, system_rhs);
    else
      zero_constraints.distribute_local_to_global(local_rhs, local_dof_indices, system_rhs);
  }

  template <int dim>
  void MovingMesh<dim>::solve_direct(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Solve direct");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);
  
    // Solve with MUMPS
    SolverControl                    solver_control;
    PETScWrappers::SparseDirectMUMPS solver(solver_control);
    solver.solve(system_matrix, completely_distributed_solution, system_rhs);

    newton_update = completely_distributed_solution;

    if(first_step)
      nonzero_constraints.distribute(newton_update);
    else
      zero_constraints.distribute(newton_update);
  }

  template <int dim>
  void MovingMesh<dim>::solve_iterative(bool first_step)
  {
    TimerOutput::Scope t(computing_timer, "Solve direct");

    LA::MPI::Vector completely_distributed_solution(locally_owned_dofs, mpi_communicator);

    // Solve with GMRES
    SolverControl              solver_control(1000, 1e-8);
    PETScWrappers::SolverGMRES solver(solver_control, mpi_communicator);

    PETScWrappers::PreconditionJacobi pc;
    pc.initialize(system_matrix);

    solver.solve(system_matrix, completely_distributed_solution, system_rhs, pc);

    newton_update = completely_distributed_solution;

    if(first_step)
      nonzero_constraints.distribute(newton_update);
    else
      zero_constraints.distribute(newton_update);
  }

  template <int dim>
  void MovingMesh<dim>::solve_newton(
    const double       tolerance,
    const bool         is_initial_step,
    const bool         output_result)
  {
    double global_res;
    double current_res;
    double last_res;
    double norm_correction;
    bool   first_step     = true;
    last_res              = 1e6;
    current_res           = 1e6;
    global_res            = 1e6;
    unsigned int iter = 1;
    const unsigned int max_iter = 100;
    const double tol = 1e-10;
    bool converged = false;

    // nonzero_constraints.distribute(newton_update);
    // present_solution = newton_update;

    while (current_res > tol && iter <= max_iter)
    {
      evaluation_point = present_solution;

      this->assemble_matrix(first_step);
      this->assemble_rhs(first_step);
      current_res      = system_rhs.linfty_norm();

      if (iter == 1)
      {
        current_res      = system_rhs.linfty_norm();
        last_res         = current_res;
      }

      this->solve_direct(first_step);
      // this->solve_iterative(first_step);
      first_step = false;

      norm_correction = newton_update.linfty_norm(); // On this proc only!
      pcout << std::scientific
        << std::setprecision(8)
        << "Newton iteration: " << iter << " - ||du|| = " << norm_correction << " - ||NL(u)|| = " << current_res << std::endl;

      if(norm_correction > 1e10 || current_res > 1e10)
      {
        pcout << "Diverged after " << iter << " iteration(s)" << std::endl;
        if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
          throw std::runtime_error("Nonlinear solver diverged");
      }

      double last_alpha_res = current_res;

      local_evaluation_point       = present_solution;
      local_evaluation_point.add(1., newton_update);
      nonzero_constraints.distribute(local_evaluation_point);
      evaluation_point = local_evaluation_point;

      this->assemble_rhs(first_step);

      current_res = system_rhs.linfty_norm();

      last_alpha_res = current_res;

      if(current_res <= tol)
      {
        pcout << "Converged in " << iter << " iteration(s) because next nonlinear residual is below tolerance: " << current_res << " < " << tol << std::endl;
        converged = true;
      }

      // global_res       = solver->get_current_residual();
      present_solution = evaluation_point;
      last_res         = current_res;
      ++iter;
    }

    if(!converged && iter == max_iter + 1)
    {
      pcout << "Did not converge after " << iter << " iteration(s)" << std::endl;
      if(Utilities::MPI::this_mpi_process(mpi_communicator) == 0)
        throw std::runtime_error("Nonlinear solver did not convege");
    }
  }

  template <int dim>
  void MovingMesh<dim>::move_mesh()
  {
    TimerOutput::Scope t(computing_timer, "Move mesh");
    pcout << "    Moving mesh..." << std::endl;

    const unsigned int n_total_components = fe.n_components();

    Assert(n_total_components == 4 * dim + 1,
           ExcMessage("Moving mesh: FESystem is expected to have 4 fields: velocity, pressure, displacement, mesh velocity, lambda."));
    
    const unsigned int displacement_offset = dim + 1;

    const FEValuesExtractors::Vector displacement(dim+1);
    ComponentMask displacement_mask = fe.component_mask(displacement);

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

        Point<dim> vertex_displacement;

        for (unsigned int i = 0; i < fe.n_dofs_per_vertex(); ++i)
        {
          const unsigned int dof_index = cell->vertex_dof_index(v, i);
          const unsigned int comp = fe.system_to_component_index(i).first;

          if (displacement_offset <= comp && comp < displacement_offset + dim)
          {
            const unsigned int d = comp - displacement_offset;
            vertex_displacement[d] = present_solution[dof_index];
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

        //   vertex_displacement[d] = present_solution[dof_index];
        // }

        cell->vertex(v) = initial_mesh_position[global_vertex_index] + vertex_displacement;
      }
    }
  }

  template <int dim>
  void
  MovingMesh<dim>::output_results(const unsigned int cycle) const
  {
    std::vector<std::string> solution_names(dim, "velocity");
    solution_names.emplace_back("pressure");

    for(unsigned int i = 0; i < dim; ++i)
      solution_names.push_back("displacement");
    for(unsigned int i = 0; i < dim; ++i)
      solution_names.push_back("mesh_velocity");
    for(unsigned int i = 0; i < dim; ++i)
      solution_names.push_back("lambda");

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

    for(unsigned int i = 0; i < 3 * dim; ++i)
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);

    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain, "subdomain");

    data_out.build_patches(mapping, 2);

    std::ofstream output("foo.vtk");
    data_out.write_vtk(output);

    data_out.write_vtu_with_pvtu_record(
      "./", "solution", cycle, mpi_communicator, 2);

  }

  template <int dim>
  void
  MovingMesh<dim>::run()
  {
    this->make_grid();

    this->setup_system();
    this->create_lambda_zero_constraints(cylinder_boundary_id);

    // Apply coupling between lambda and displacement dofs
    // If cylinder is fixed, a zero displacement BC is added in create_zero_constraints
    if(!fix_cylinder && !prescribed_displacement)
      this->create_displacement_bc_constraints(cylinder_boundary_id);

    this->create_zero_constraints();
    this->create_nonzero_constraints();

    this->create_sparsity_pattern();

    this->set_initial_condition();

    // pcout << "Constraints:" << std::endl;
    // pcout << zero_constraints.n_constraints() << std::endl;
    // pcout << nonzero_constraints.n_constraints() << std::endl;

    // const unsigned int local_zero_constraints = zero_constraints.n_constraints();
    // std::vector<unsigned int> all_zero_constraints =
    //   Utilities::MPI::all_gather(MPI_COMM_WORLD, local_zero_constraints);

    // const unsigned int local_nonzero_constraints = nonzero_constraints.n_constraints();
    // std::vector<unsigned int> all_nonzero_constraints =
    //   Utilities::MPI::all_gather(MPI_COMM_WORLD, local_nonzero_constraints);

    // const unsigned int n_zero_constraints =
    //   std::accumulate(all_zero_constraints.begin(), all_zero_constraints.end(), 0);
    // const unsigned int n_nonzero_constraints =
    //   std::accumulate(all_nonzero_constraints.begin(), all_nonzero_constraints.end(), 0);

    // pcout << n_zero_constraints << std::endl;
    // pcout << n_nonzero_constraints << std::endl;

    // AssertThrow(n_zero_constraints == n_nonzero_constraints,
    //   ExcMessage("Careful: zero and nonzero constraints don't have the same number of constraints. "
    //     "PETSc might complain that new entries are added outisde o fthe sparsity pattern, "
    //     "although I'm not sure why because the pattern is constructed with the nonzero constraints, "
    //     "and the zero constraints only involved a single DoF at a time... "
    //     "Not sure yet if this is an issue."));

    this->output_results(0);

    for(unsigned int i = 0; i < nTimeSteps; ++i)
    {
      this->current_time += this->dt;

      if(!fix_cylinder && !prescribed_displacement)
        this->create_displacement_bc_constraints(cylinder_boundary_id);

      this->create_nonzero_constraints();
      this->apply_nonzero_constraints();

      pcout << std::endl << "Time step " << i + 1 << " - Advancing to t = " << current_time << '.' << std::endl;

      // For BDF2: initial step just sets solution to initial condition
      // Compute mesh velocity ?
      if(i == 0 && bdfCoeffs.size() == 3)
      {
        std::cout << "Rotating BDF initial condition" << std::endl;
      }
      else
      {
        this->solve_newton(1., true, true);
      }

      this->move_mesh();

      this->output_results(i + 1);

      for(unsigned int i = previous_solutions.size() - 1; i >= 1; --i)
        previous_solutions[i] = previous_solutions[i-1];
      previous_solutions[0] = present_solution;
    }
  }
}

int
main(int argc, char *argv[])
{
  
  try
    {
      using namespace dealii;
      using namespace MovingMeshTest;

      Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

      const bool fix_cylinder = false;
      const bool prescribed_displacement = false;

      const unsigned int cylinder_boundary_id = 4;

      const double spring_stiffness = 1.;

      const double Re = 100.;
      const double D = 0.1;  // Diameter
      const double uMax = 0.3;
      const double U = 2.*uMax/3.;
      const double viscosity = U*D/Re;

      const double pseudo_solid_mu = 1.0;
      const double pseudo_solid_lambda = 1.0;

      const unsigned int bdf_order = 1;
      const double t0 = 0.;
      const double dt = 0.1;
      const int nTimeSteps = 50;
      const double t1 = dt * nTimeSteps;

      const double Ur = pow(M_PI, 3./2.) * U / sqrt(spring_stiffness);

      std::cout << "U_r = " << Ur << std::endl;

      MovingMesh<2> problem(2, 2, fix_cylinder, prescribed_displacement, cylinder_boundary_id, spring_stiffness, viscosity,
        pseudo_solid_mu, pseudo_solid_lambda, bdf_order, t0, t1, nTimeSteps);

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
