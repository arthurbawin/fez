
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

#define WITH_MESH_MOVEMENT
// #define ZERO_VELOCITY_ON_CYLINDER
// #define WEAK_NO_SLIP

namespace MovingMeshTest
{
  using namespace dealii;

  template <int dim>
  class CylinderDisplacement : public Function<dim>
  {
  public:
    CylinderDisplacement(const double time = 0.)
      : Function<dim>(3*dim+1, time)
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
  #if defined(WITH_MESH_MOVEMENT)
    VelocityInlet() : Function<dim>(3*dim+1)
  #else
    VelocityInlet() : Function<dim>(dim + 1)
  #endif
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      const double x = p[0];
      const double y = p[1];
      const double H = 0.41;
      const double uMax = 0.3;
      const unsigned int offset = 0;

      if constexpr (dim == 2)
      {
        values[offset + 0] = 4. * uMax * y * (H-y)/(H*H);
        values[offset + 1] = 0.;
        values[offset + 2] = 0.;
      } else
      {
        values[offset + 0] = 0.;
        values[offset + 1] = 4. * uMax * x * (H-x)/(H*H);
        values[offset + 2] = 0.;
        values[offset + 3] = 0.;
      }
    }
  };

  template <int dim>
  class InitialCondition : public Function<dim>
  {
  public:
    InitialCondition()
      : Function<dim>(3*dim+1)
    {}

    virtual void
    vector_value(const Point<dim> &p, Vector<double> &values) const override
    {
      for(unsigned int i = 0; i < 3 * dim + 1; ++i)
        values[i] = 0.;
    }
  };

  template <int dim>
  class ScratchData
  {
  public:
    ScratchData(const unsigned int n_q_points, const unsigned int dofs_per_cell)
     : n_q_points(n_q_points)
     , dofs_per_cell(dofs_per_cell)
    {
      this->allocate();
    }

    void allocate();
    void reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
                const LA::MPI::Vector &current_solution,
                const std::vector<LA::MPI::Vector> &previous_solutions,
                FEValues<dim> &fe_values);

  public:

    const unsigned int n_q_points;
    const unsigned int dofs_per_cell;

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

    // Shape functions and gradients for each quad node and each dof
    std::vector<std::vector<double>>         div_phi_u;
    std::vector<std::vector<Tensor<1, dim>>> phi_u;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_u;
    std::vector<std::vector<double>>         phi_p;
    std::vector<std::vector<Tensor<1, dim>>> phi_disp;
    std::vector<std::vector<double>>         div_phi_disp;
    std::vector<std::vector<Tensor<2, dim>>> grad_phi_disp;
    std::vector<std::vector<Tensor<1, dim>>> phi_w;
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

    // BDF1
    previous_velocity_values.resize(1, std::vector<Tensor<1, dim>>(n_q_points));
    previous_displacement_values.resize(1, std::vector<Tensor<1, dim>>(n_q_points));

    div_phi_u.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_u.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_u.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    phi_p.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_disp.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
    grad_phi_disp.resize(n_q_points, std::vector<Tensor<2, dim>>(dofs_per_cell));
    div_phi_disp.resize(n_q_points, std::vector<double>(dofs_per_cell));
    phi_w.resize(n_q_points, std::vector<Tensor<1, dim>>(dofs_per_cell));
  }

  template <int dim>
  void ScratchData<dim>::reinit(const typename DoFHandler<dim>::active_cell_iterator &cell,
                                const LA::MPI::Vector &current_solution,
                                const std::vector<LA::MPI::Vector> &previous_solutions,
                                FEValues<dim> &fe_values)
  {
    fe_values.reinit(cell);

    for (const unsigned int i : fe_values.dof_indices())
      components[i] = fe_values.get_fe().system_to_component_index(i).first;

    const FEValuesExtractors::Vector velocities(0); // 0 -> dim-1
    const FEValuesExtractors::Scalar pressure(dim); // dim -> dim
  #if defined(WITH_MESH_MOVEMENT)
    const FEValuesExtractors::Vector displacement(dim + 1); // dim+1 -> 2*dim
    const FEValuesExtractors::Vector mesh_velocity(2 * dim + 1); // 2*dim+1 -> 3*dim
  #endif

    fe_values[velocities].get_function_values(current_solution, present_velocity_values);
    fe_values[velocities].get_function_gradients( current_solution, present_velocity_gradients);
    fe_values[pressure].get_function_values(current_solution, present_pressure_values);
  #if defined(WITH_MESH_MOVEMENT)
    fe_values[displacement].get_function_values(current_solution, present_displacement_values);
    fe_values[displacement].get_function_gradients(current_solution, present_displacement_gradients);
    fe_values[mesh_velocity].get_function_values(current_solution, present_mesh_velocity_values);
  #endif

    for(unsigned int i = 0; i < previous_solutions.size(); ++i)
    {
      fe_values[velocities].get_function_values(previous_solutions[i], previous_velocity_values[i]);
    #if defined(WITH_MESH_MOVEMENT)
      fe_values[displacement].get_function_values(previous_solutions[i], previous_displacement_values[i]);
    #endif
    }

    for (unsigned int q = 0; q < n_q_points; ++q)
    {
      for (unsigned int k = 0; k < dofs_per_cell; ++k)
      {
        phi_u[q][k]         = fe_values[velocities].value(k, q);
        grad_phi_u[q][k]    = fe_values[velocities].gradient(k, q);
        div_phi_u[q][k]     = fe_values[velocities].divergence(k, q);
        phi_p[q][k]         = fe_values[pressure].value(k, q);
      #if defined(WITH_MESH_MOVEMENT)
        phi_disp[q][k]      = fe_values[displacement].value(k, q);
        grad_phi_disp[q][k] = fe_values[displacement].gradient(k, q);
        div_phi_disp[q][k]  = fe_values[displacement].divergence(k, q);
        phi_w[q][k]         = fe_values[mesh_velocity].value(k, q);
      #endif
      }
    }
  }

  template <int dim>
  class MovingMesh
  {
  public:
    MovingMesh(unsigned int velocity_degree,
               unsigned int displacement_degree,
               double viscosity,
               double pseudo_solid_mu,
               double pseudo_solid_lambda,
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
    apply_dof_coupling_on_cylinder(const unsigned int boundary_id);
    void
    create_displacement_coupling_placeholder(const unsigned int boundary_id);
    void
    create_zero_constraints();
    void
    create_nonzero_constraints();
    void
    create_sparsity_pattern();
    void
    create_sparsity_pattern_with_additional_coupling();
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
                       LA::MPI::Vector &current_solution,
                       std::vector<LA::MPI::Vector> &previous_solutions,
                       std::vector<types::global_dof_index> &local_dof_indices,
                       Vector<double> &local_rhs);

    void
    compute_reactions();
    void solve(bool first_step);
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

    CylinderDisplacement<dim> cyl_bc;
    std::vector<Point<dim>> initial_mesh_position;

    std::vector<LA::MPI::Vector> previous_solutions;
    std::vector<double> bdfCoeffs;

    // Time integration
    const double t0;
    const double t1;
    const unsigned int nTimeSteps;
    const double dt;
    double current_time;

    // The data related to the Lagrange multiplier
    FE_SimplexP<dim-1, dim> fe_constraints;
    parallel::fullydistributed::Triangulation<dim-1, dim> cylinder_mesh;
    DoFHandler<dim-1, dim> dof_handler_constraints;
  };

  template <int dim>
  MovingMesh<dim>::MovingMesh(unsigned int velocity_degree,
                              unsigned int displacement_degree,
                              double viscosity,
                              double pseudo_solid_mu,
                              double pseudo_solid_lambda,
                              double t0,
                              double t1,
                              unsigned int nTimeSteps)
    : velocity_degree(velocity_degree)
    , displacement_degree(displacement_degree)
    , viscosity(viscosity)
    , pseudo_solid_mu(pseudo_solid_mu)
    , pseudo_solid_lambda(pseudo_solid_lambda)
    , mpi_communicator(MPI_COMM_WORLD)
    , fe(FE_SimplexP<dim>(velocity_degree),
         dim,
         FE_SimplexP<dim>(velocity_degree - 1),
         1
      #if defined(WITH_MESH_MOVEMENT)
         , FE_SimplexP<dim>(displacement_degree),
         dim,
         FE_SimplexP<dim>(displacement_degree),
         dim
      #endif
         )
    , triangulation(mpi_communicator)
    , mapping(FE_SimplexP<dim>(1))
    , dof_handler(triangulation)
    , pcout(std::cout, (Utilities::MPI::this_mpi_process(mpi_communicator) == 0))
    , computing_timer(mpi_communicator,
                      pcout,
                      TimerOutput::summary,
                      TimerOutput::wall_times)
    , cyl_bc(CylinderDisplacement<dim>(current_time))
    , t0(t0)
    , t1(t1)
    , nTimeSteps(nTimeSteps)
    , dt((t1 - t0) / nTimeSteps)
    , current_time(t0)
    , fe_constraints(FE_SimplexP<dim-1, dim>(velocity_degree))
    , cylinder_mesh(mpi_communicator)
    , dof_handler_constraints(cylinder_mesh)
  {
    // BDF1
    bdfCoeffs.resize(2);
    bdfCoeffs[0] =  1. / dt;
    bdfCoeffs[1] = -1. / dt;

    pcout << "BDF coefficients: " << bdfCoeffs[0] << " , " << bdfCoeffs[1] << std::endl;
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

    // Create boundary mesh on cylinder to handle reactions
    const unsigned int cylinder_boundary_id = 4;
    std::set< types::boundary_id > boundary_ids;
    boundary_ids.insert(cylinder_boundary_id);
    Triangulation<dim-1, dim> serial_boundary_tria;
    GridGenerator::extract_boundary_mesh(serial_tria, serial_boundary_tria, boundary_ids);

    const TriangulationDescription::Description<dim-1, dim> description_bdr =
      TriangulationDescription::Utilities::
        create_description_from_triangulation(serial_boundary_tria, mpi_communicator);
    cylinder_mesh.create_triangulation(description_bdr);

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

#if defined(WEAK_NO_SLIP)
    dof_handler_constraints.distribute_dofs(fe_constraints);
    pcout << "Number of reactions degrees of freedom: " << dof_handler_constraints.n_dofs() << std::endl;

    //
    // Add Lagrange multiplier DoFs
    //
    const unsigned int n_fsi_dofs    = dof_handler.n_dofs();
    const unsigned int n_lambda_dofs = dof_handler_constraints.n_dofs();
    const unsigned int n_total_dofs  = n_fsi_dofs + n_lambda_dofs;
    const unsigned int offset = n_fsi_dofs;

    std::cout << "owned" << std::endl;
    locally_owned_dofs.set_size(n_total_dofs);
    locally_owned_dofs.add_indices(this->dof_handler.locally_owned_dofs(), 0);
    IndexSet locally_owned_constraints_dofs = dof_handler_constraints.locally_owned_dofs();
    locally_owned_dofs.add_indices(locally_owned_constraints_dofs, offset);

    std::cout << "relevant" << std::endl;
    IndexSet fsi_relevant_dofs       = DoFTools::extract_locally_relevant_dofs(dof_handler);
    IndexSet constraint_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler_constraints);

    const unsigned int n_relevant_fsi_dofs        = fsi_relevant_dofs.size();
    const unsigned int n_relevant_constraint_dofs = constraint_relevant_dofs.size();
    locally_relevant_dofs.set_size(n_relevant_fsi_dofs + n_relevant_constraint_dofs);
    locally_relevant_dofs.add_indices(fsi_relevant_dofs);
    locally_relevant_dofs.add_indices(constraint_relevant_dofs);
#else
    locally_owned_dofs = this->dof_handler.locally_owned_dofs();
    locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(this->dof_handler);
#endif

    //
    // Initialize parallel vectors
    //
    present_solution.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
    evaluation_point.reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);

    local_evaluation_point.reinit(locally_owned_dofs, mpi_communicator);
    newton_update.reinit(locally_owned_dofs, mpi_communicator);
    system_rhs.reinit(locally_owned_dofs, mpi_communicator);

    // BDF1
    previous_solutions.resize(1);
    previous_solutions[0].reinit(locally_owned_dofs, locally_relevant_dofs, mpi_communicator);
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
  MovingMesh<dim>::create_sparsity_pattern_with_additional_coupling()
  {
    const unsigned int cylinder_boundary_id = 4;
    const unsigned int n_fsi_dofs = dof_handler.n_dofs();

    // Create sparsity pattern with FSI dofs:
    DynamicSparsityPattern dsp(locally_relevant_dofs);
    DoFTools::make_sparsity_pattern(dof_handler, dsp, nonzero_constraints, false);

    // Setup a map between faces in the
    using FaceIterator = typename Triangulation<dim>::face_iterator;
    using LambdaCell   = typename DoFHandler<dim-1>::active_cell_iterator;

    std::map<FaceIterator, LambdaCell> lambda_face_map;
    for (const LambdaCell &cell : dof_handler_constraints.active_cell_iterators())
    {
      // Assuming lambda FE is one face per cell
      const auto face = cell->face(0);
      lambda_face_map[face] = cell;
    }

    // Then loop over cells and add velocity coupling with Lagrange multiplier
    std::vector<types::global_dof_index> local_dof_indices(fe.n_dofs_per_cell());
    std::vector<types::global_dof_index> local_dof_indices_lambda(fe_constraints.n_dofs_per_cell());
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      if (cell->is_locally_owned())
      {
        cell->get_dof_indices(local_dof_indices);

        for (unsigned int f = 0; f < cell->n_faces(); ++f)
        {
          const auto &face = cell->face(f);
          if (face->at_boundary() && face->boundary_id() == cylinder_boundary_id)
          {
            // Find corresponding face in lambda handler
            auto it = lambda_face_map.find(face);
            Assert(it != lambda_face_map.end(), ExcMessage("Lambda face not found"));
            const LambdaCell &lambda_cell = it->second;

            // const auto lambda_it = lambda_face_map.find(face);
            // Assert(lambda_it != lambda_face_map.end(), ExcMessage("Missing lambda face"));

            // const auto &lambda_face_cell = lambda_it->second;
            lambda_cell->get_dof_indices(local_dof_indices_lambda);

            // Insert velocity-lambda coupling
            for (const auto i : local_dof_indices)
              for (const auto j : local_dof_indices_lambda)
              {
                dsp.add(i, j + n_fsi_dofs);   // (u/p) row, lambda column
                dsp.add(j + n_fsi_dofs, i);   // lambda row, (u/p) column (symmetry)
              }
          }
        }
      }
    }

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

    // Apply non-homogeneous Dirichlet BC and set as current solution
    nonzero_constraints.distribute(newton_update);
    present_solution = newton_update;

    previous_solutions[0] = newton_update;
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
  #if defined(WITH_MESH_MOVEMENT)
    const FEValuesExtractors::Vector displacement(dim + 1);
    const FEValuesExtractors::Vector mesh_velocity(2 * dim + 1);
  #endif

  #if defined(WITH_MESH_MOVEMENT)
    const unsigned int n_total_comp = 3 * dim + 1;
  #else
    const unsigned int n_total_comp = dim + 1;
  #endif

    unsigned int inletBoundary        = (dim == 2) ? 2 : 5;
    unsigned int cylinder_boundary_id = (dim == 2) ? 4 : 5;

    // Boundaries where Dirichlet BC are applied,
    // where the Newton increment should be zero.
    std::vector<unsigned int> dirichlet_boundaries_u, dirichlet_boundaries_chi;
    if constexpr(dim == 2)
    {
      // Boundary 1 is the Outlet, which is free for the flow
      dirichlet_boundaries_u   = {2, 3, 4};
      dirichlet_boundaries_chi = {1, 2, 3, 4};
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

    #if defined(WITH_MESH_MOVEMENT)
    for(unsigned int id : dirichlet_boundaries_chi)
    {
      VectorTools::interpolate_boundary_values(mapping,
                                               dof_handler,
                                               id,
                                               Functions::ZeroFunction<dim>(n_total_comp),
                                               zero_constraints,
                                               fe.component_mask(displacement));
    }
    #endif
    zero_constraints.close();
  }

  template <int dim>
  void MovingMesh<dim>::apply_dof_coupling_on_cylinder(const unsigned int boundary_id)
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
  #if defined(WITH_MESH_MOVEMENT)
    const FEValuesExtractors::Vector displacement(dim + 1);
    const FEValuesExtractors::Vector mesh_velocity(2 * dim + 1);
  #endif

  #if defined(WITH_MESH_MOVEMENT)
    const unsigned int n_total_comp = 3 * dim + 1;
  #else
    const unsigned int n_total_comp = dim + 1;
  #endif

    unsigned int inletBoundary        = (dim == 2) ? 2 : 5;
    unsigned int cylinder_boundary_id = (dim == 2) ? 4 : 5;

    std::vector<unsigned int> noSlipBoundaries, noMeshMovementBoundaries;
    if constexpr(dim == 2)
    {
      // Boundary 1 is the Outlet, which is free
      noSlipBoundaries           = {3};
      noMeshMovementBoundaries   = {1, 2, 3};
    }

#if defined(ZERO_VELOCITY_ON_CYLINDER)
    // Cylinder velocity BC
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             cylinder_boundary_id,
                                             Functions::ZeroFunction<dim>(n_total_comp),
                                             nonzero_constraints,
                                             fe.component_mask(velocities));
#else
    //
    // Create affine constraints relating velocity and mesh velocity on cylinder
    //
    this->apply_dof_coupling_on_cylinder(cylinder_boundary_id);
#endif

    // Velocity inlet
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             inletBoundary,
                                             VelocityInlet<dim>(),
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

  #if defined(WITH_MESH_MOVEMENT)
    // Displacement BC on cylinder
    this->cyl_bc.set_time(current_time);
    VectorTools::interpolate_boundary_values(mapping,
                                             dof_handler,
                                             cylinder_boundary_id,
                                             cyl_bc,
                                             nonzero_constraints,
                                             fe.component_mask(displacement));

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
  #endif

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

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    ScratchData<dim> scratchData(n_q_points, dofs_per_cell);
    FullMatrix<double> local_matrix(dofs_per_cell, dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      this->assemble_local_matrix(first_step, cell, scratchData, fe_values, evaluation_point, previous_solutions, local_dof_indices, local_matrix);
    }

    system_matrix.compress(VectorOperation::add);
  }

  template <int dim>
  void
  MovingMesh<dim>::assemble_local_matrix(bool first_step,
                                         const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         ScratchData<dim> &scratchData,
                                         FEValues<dim> &fe_values,
                                         LA::MPI::Vector &current_solution,
                                         std::vector<LA::MPI::Vector> &previous_solutions,
                                         std::vector<types::global_dof_index> &local_dof_indices,
                                         FullMatrix<double> &local_matrix)
  {
    if (!cell->is_locally_owned())
      return;

    scratchData.reinit(cell, current_solution, previous_solutions, fe_values);

    local_matrix = 0;

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const auto &present_velocity_values    = scratchData.present_velocity_values[q];
      const auto &present_velocity_gradients = scratchData.present_velocity_gradients[q];

      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];

      const auto &phi_p      = scratchData.phi_p[q];

    #if defined(WITH_MESH_MOVEMENT)
      const auto &present_mesh_velocity_values = scratchData.present_mesh_velocity_values[q];

      const auto &phi_x      = scratchData.phi_disp[q];
      const auto &grad_phi_x = scratchData.grad_phi_disp[q];
      const auto &div_phi_x  = scratchData.div_phi_disp[q];

      const auto &phi_w      = scratchData.phi_w[q];
    #endif

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

        #if defined(WITH_MESH_MOVEMENT)
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
        #endif

          local_matrix_ij *= fe_values.JxW(q);
          local_matrix(i, j) += local_matrix_ij;
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

    FEValues<dim> fe_values(mapping,
                            fe,
                            quadrature_formula,
                            update_values | update_gradients |
                              update_quadrature_points | update_JxW_values);

    const unsigned int n_q_points    = quadrature_formula.size();
    const unsigned int dofs_per_cell = fe.n_dofs_per_cell();
    
    ScratchData<dim> scratchData(n_q_points, dofs_per_cell);
    Vector<double> local_rhs(dofs_per_cell);

    std::vector<types::global_dof_index> local_dof_indices(dofs_per_cell);

    for (const auto &cell : dof_handler.active_cell_iterators() |
                              IteratorFilters::LocallyOwnedCell())
    {
      this->assemble_local_rhs(first_step, cell, scratchData, fe_values, evaluation_point, previous_solutions, local_dof_indices, local_rhs);
    }

    system_rhs.compress(VectorOperation::add);
  }

  template <int dim>
  void
  MovingMesh<dim>::assemble_local_rhs(bool first_step,
                                         const typename DoFHandler<dim>::active_cell_iterator &cell,
                                         ScratchData<dim> &scratchData,
                                         FEValues<dim> &fe_values,
                                         LA::MPI::Vector &current_solution,
                                         std::vector<LA::MPI::Vector> &previous_solutions,
                                         std::vector<types::global_dof_index> &local_dof_indices,
                                         Vector<double> &local_rhs)
  {
    scratchData.reinit(cell, current_solution, previous_solutions, fe_values);

    local_rhs = 0;

    // BDF1
    std::vector<Tensor<1, dim>> velocity(2);
    std::vector<Tensor<1, dim>> displacement(2);

    for (unsigned int q = 0; q < scratchData.n_q_points; ++q)
    {
      const auto &present_velocity_values        = scratchData.present_velocity_values[q];
      const auto &present_velocity_gradients     = scratchData.present_velocity_gradients[q];
      const auto &present_pressure_values        = scratchData.present_pressure_values[q];
    #if defined(WITH_MESH_MOVEMENT)
      const auto &present_displacement_values    = scratchData.present_displacement_values[q];
      const auto &present_displacement_gradients = scratchData.present_displacement_gradients[q];
      const auto &present_mesh_velocity_values   = scratchData.present_mesh_velocity_values[q];
    #endif

      double present_velocity_divergence =
                  trace(present_velocity_gradients);
    #if defined(WITH_MESH_MOVEMENT)
      double present_displacement_divergence =
                  trace(present_displacement_gradients);
    #endif

      // BDF1
      velocity[0] = present_velocity_values;
      velocity[1] = scratchData.previous_velocity_values[0][q];
    #if defined(WITH_MESH_MOVEMENT)
      displacement[0] = present_displacement_values;
      displacement[1] = scratchData.previous_displacement_values[0][q];
    #endif

      const Tensor<1, dim> uDotGradU = present_velocity_gradients * present_velocity_values;
    #if defined(WITH_MESH_MOVEMENT)
      const Tensor<1, dim> wDotGradU = present_velocity_gradients * present_mesh_velocity_values;
    #endif

      const auto &phi_u      = scratchData.phi_u[q];
      const auto &grad_phi_u = scratchData.grad_phi_u[q];
      const auto &div_phi_u  = scratchData.div_phi_u[q];

      const auto &phi_p      = scratchData.phi_p[q];

    #if defined(WITH_MESH_MOVEMENT)
      const auto &phi_x      = scratchData.phi_disp[q];
      const auto &grad_phi_x = scratchData.grad_phi_disp[q];
      const auto &div_phi_x  = scratchData.div_phi_disp[q];

      const auto &phi_w      = scratchData.phi_w[q];
    #endif

      for (unsigned int i = 0; i < scratchData.dofs_per_cell; ++i)
      {
      #if defined(WITH_MESH_MOVEMENT)
        const auto present_displacement_gradient_sym = present_displacement_gradients + transpose(present_displacement_gradients);
      #endif

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

        #if defined(WITH_MESH_MOVEMENT)
          // Linear elasticity
          + pseudo_solid_lambda * present_displacement_divergence * div_phi_x[i]
          + pseudo_solid_mu * scalar_product(present_displacement_gradient_sym, grad_phi_x[i])

          // Mesh velocity
          + present_mesh_velocity_values * phi_w[i]
        #endif
          ) * fe_values.JxW(q);

          // Transient terms:
          for(unsigned int iBDF = 0; iBDF < bdfCoeffs.size(); ++iBDF)
          {
            local_rhs_i -= bdfCoeffs[iBDF] * (velocity[iBDF] * phi_u[i]) * fe_values.JxW(q);
          #if defined(WITH_MESH_MOVEMENT)
            local_rhs_i -= - bdfCoeffs[iBDF] * (displacement[iBDF] * phi_w[i]) * fe_values.JxW(q);
          #endif
          }
          
          local_rhs(i) += local_rhs_i;
      }
    }

    cell->get_dof_indices(local_dof_indices);
    if(first_step)
      nonzero_constraints.distribute_local_to_global(local_rhs, local_dof_indices, system_rhs);
    else
      zero_constraints.distribute_local_to_global(local_rhs, local_dof_indices, system_rhs);
  }

  template <int dim>
  void
  MovingMesh<dim>::create_displacement_coupling_placeholder(const unsigned int boundary_id)
  {
    // THIS IS TO COMPUTE THE REACTIONS? Probably only for linear problem...
    // residual.reinit(system_rhs);
    // system_matrix.vmult(residual, solution);
    // residual -= system_rhs;

    // reaction_vector.reinit(solution);
    // reaction_vector = 0.0;

    // for (const auto &line : constraints.get_lines())
    // {
    //   const auto i = line.index;
    //   reaction_vector[i] = -residual[i]; // only on constrained DoFs
    // }
    // reaction_vector.compress(VectorOperation::insert);  // in MPI

    // // Predeclare mesh displacement constraints (no values yet)
    // for (auto dof : mesh_disp_interface_dofs)
    // {
    //   nonzero_constraints.add_line(dof);
    //   // We don't call set_inhomogeneity or add_entry here yet
    // }

    // nonzero_constraints.close();


    // std::vector<types::global_dof_index> local_face_dof_indices(fe.n_dofs_per_face());

    // // To get the physical positions of the DOFs
    // // std::vector<Point<dim>> support_points(dof_handler.n_dofs());
    // // DoFTools::map_dofs_to_support_points(mapping, dof_handler, support_points);

    // for (const auto &cell : dof_handler.active_cell_iterators())
    // {
    //   if (!cell->is_locally_owned())
    //     continue;

    //   for (const auto face_no : cell->face_indices())
    //   {
    //     if (cell->face(face_no)->at_boundary() && (cell->face(face_no)->boundary_id() == boundary_id))
    //     {
    //       cell->face(face_no)->get_dof_indices(local_face_dof_indices);

    //       for(unsigned int i = 0; i < fe.n_dofs_per_face(); ++i)
    //       {
    //         const unsigned dof_fluid_velocity = local_face_dof_indices[i];
            
    //         // Only constrain owned DOFs
    //         if(!locally_owned_dofs.is_element(dof_fluid_velocity))
    //           continue;

    //         const unsigned comp_index  = fe.face_system_to_component_index(i, face_no).first;
    //         const unsigned shape_index = fe.face_system_to_component_index(i, face_no).second;


    //         // Only constrain fluid velocity
    //         if(comp_index >= dim)
    //           continue;

    //         // The associated mesh velocity dof component
    //         const unsigned int mesh_velocity_comp_index = comp_index + 2 * dim + 1;

    //         bool matched = false;

    //         for(unsigned int j = 0; j < fe.n_dofs_per_face(); ++j)
    //         {
    //           const unsigned comp_index_j  = fe.face_system_to_component_index(j, face_no).first;
    //           const unsigned shape_index_j = fe.face_system_to_component_index(j, face_no).second;

    //           // Find DOFs with matching shape_index
    //           if(shape_index != shape_index_j)
    //             continue;

    //           // Find matching mesh_velocity DOF
    //           if(mesh_velocity_comp_index != comp_index_j)
    //             continue;

    //           const unsigned dof_mesh_velocity  = local_face_dof_indices[j];

    //           nonzero_constraints.add_line(dof_fluid_velocity);
    //           nonzero_constraints.add_entry(dof_fluid_velocity, dof_mesh_velocity, 1.0);
    //           // nonzero_constraints.set_inhomogeneity(dof_fluid_velocity, 0.0); // optional

    //           matched = true;

    //           break;
    //         }

    //         if(!matched)
    //         {
    //           throw std::runtime_error("Could not match a fluid velocity DOF with a mesh velocity DOF on the cylinder. "
    //             "This is probably because mesh velocity is represented with a different FE space.");
    //         }
    //       }
    //     }
    //   }
    // }
  }

  template <int dim>
  void MovingMesh<dim>::solve(bool first_step)
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
    const unsigned int max_iter = 10;
    const double tol = 1e-13;
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

      this->solve(first_step);
      first_step = false;

      norm_correction = newton_update.linfty_norm();
      pcout << std::scientific
        << std::setprecision(8)
        << "Newton iteration: " << iter << " - ||du|| = " << norm_correction << " - ||NL(u)|| = " << current_res << std::endl;

      if(norm_correction > 1e4 || current_res > 1e4)
      {
        pcout << "Diverged after " << iter << " iteration(s)" << std::endl;
        throw std::runtime_error("Nonlinear solver diverged");
      }

      double last_alpha_res = current_res;

      local_evaluation_point       = present_solution;
      local_evaluation_point.add(1., newton_update);
      nonzero_constraints.distribute(local_evaluation_point);
      evaluation_point = local_evaluation_point;

      this->assemble_rhs(first_step);

      current_res      = system_rhs.linfty_norm();

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

    if(!converged && iter == max_iter)
    {
      pcout << "Did not converge after " << iter << " iteration(s)" << std::endl;
      throw std::runtime_error("Nonlinear solver did not convege");
    }
  }

  template <int dim>
  void MovingMesh<dim>::move_mesh()
  {
    pcout << "    Moving mesh..." << std::endl;

    const unsigned int n_total_components = fe.n_components();

    Assert(n_total_components == 3 * dim + 1,
           ExcMessage("Moving mesh: FESystem is expected to have 4 fields: velocity, pressure, displacement, mesh velocity."));
    
    // Displacement offset
    const unsigned int offset = dim + 1;

    std::vector<bool> vertex_touched(triangulation.n_vertices(), false);
    for (const auto &cell : dof_handler.active_cell_iterators())
    {
      for (const auto v : cell->vertex_indices())
      {
        const unsigned int global_vertex_index = cell->vertex_index(v);

        if (vertex_touched[global_vertex_index])
          continue;

        vertex_touched[global_vertex_index] = true;

        Point<dim> vertex_displacement;

        for (unsigned int d = 0; d < dim; ++d)
        {
          // Index of the displacement component
          const unsigned int displacement_component = offset + d;

          // Find system index of that displacement component at this vertex
          const unsigned int system_index =
            fe.component_to_system_index(displacement_component, 0); // 0 = first shape function for that component

          // Use `vertex_dof_index` to get DoF index for this vertex and component
          const unsigned int dof_index = cell->vertex_dof_index(v, system_index);

          vertex_displacement[d] = present_solution[dof_index];
        }

        // cell->vertex(v) += vertex_displacement;
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

  #if defined(WITH_MESH_MOVEMENT)
    for(unsigned int i = 0; i < dim; ++i)
      solution_names.push_back("displacement");
    for(unsigned int i = 0; i < dim; ++i)
      solution_names.push_back("mesh_velocity");
  #endif

    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_component_interpretation.push_back(
      DataComponentInterpretation::component_is_scalar);

  #if defined(WITH_MESH_MOVEMENT)
    for(unsigned int i = 0; i < 2 * dim; ++i)
      data_component_interpretation.push_back(DataComponentInterpretation::component_is_part_of_vector);
  #endif

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
#ifdef USE_PETSC_LA
    pcout << "Running using PETSc." << std::endl;
#else
    pcout << "Running using Trilinos." << std::endl;
#endif

    this->make_grid();

    this->setup_system();
    this->create_zero_constraints();
    this->create_nonzero_constraints();

#if defined(WEAK_NO_SLIP)
    this->create_sparsity_pattern_with_additional_coupling();
#else
    this->create_sparsity_pattern();
#endif

    this->set_initial_condition();

    pcout << "Constraints:" << std::endl;
    pcout << zero_constraints.n_constraints() << std::endl;
    pcout << nonzero_constraints.n_constraints() << std::endl;

    const unsigned int local_zero_constraints = zero_constraints.n_constraints();
    std::vector<unsigned int> all_zero_constraints =
      Utilities::MPI::all_gather(MPI_COMM_WORLD, local_zero_constraints);

    const unsigned int local_nonzero_constraints = nonzero_constraints.n_constraints();
    std::vector<unsigned int> all_nonzero_constraints =
      Utilities::MPI::all_gather(MPI_COMM_WORLD, local_nonzero_constraints);

    const unsigned int n_zero_constraints =
      std::accumulate(all_zero_constraints.begin(), all_zero_constraints.end(), 0);
    const unsigned int n_nonzero_constraints =
      std::accumulate(all_nonzero_constraints.begin(), all_nonzero_constraints.end(), 0);

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

      this->create_nonzero_constraints();
      this->apply_nonzero_constraints();

      pcout << std::endl
                << "Time step " << i + 1 << " - Advancing to t = " << current_time << '.' << std::endl;

      this->solve_newton(1., true, true);

    #if defined(WITH_MESH_MOVEMENT)
      this->move_mesh();
    #endif

      this->output_results(i + 1);

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

      const double Re = 20.;
      const double D = 0.1;  // Diameter
      const double uMax = 0.3;
      const double U = 2.*uMax/3.;
      const double viscosity = U*D/Re;

      const double pseudo_solid_mu = 1.0;
      const double pseudo_solid_lambda = 1.0;

      const double t0 = 0.;
      const double dt = 0.1;
      const int nTimeSteps = 100;
      const double t1 = dt * nTimeSteps;

      MovingMesh<2> problem(2, 2, viscosity, pseudo_solid_mu, pseudo_solid_lambda, t0, t1, nTimeSteps);
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
