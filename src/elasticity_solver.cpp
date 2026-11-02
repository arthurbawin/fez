
#include <assembly/elasticity_assemblers.h>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <compare_matrix.h>
#include <deal.II/base/scope_exit.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <elasticity_solver.h>
#include <errors.h>
#include <linear_solver.h>
#include <mesh.h>
#include <post_processing_tools.h>
#include <solver_info.h>
#include <utilities.h>

#include <array>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#if defined(DEAL_II_GMSH_WITH_API)
#  include <gmsh.h>
#endif

namespace
{
  // Partition-independent key of a (support point, component) pair.
  template <int dim>
  std::string cache_entry_key(const std::array<double, dim> &support_point,
                              const unsigned int             component)
  {
    std::ostringstream key;
    key << component << std::setprecision(17);
    for (const double coordinate : support_point)
      key << ":" << coordinate;
    return key.str();
  }

  template <int dim>
  struct PresolvedMeshCacheEntry
  {
    std::array<double, dim> support_point;
    unsigned int            component;
    double                  value;

    template <class Archive>
    void serialize(Archive &archive, const unsigned int)
    {
      for (auto &coordinate : support_point)
        archive &coordinate;
      archive &component;
      archive &value;
    }
  };

  std::string fingerprint_hash(const std::string &text)
  {
    std::uint64_t hash = 14695981039346656037ull;
    for (unsigned char c : text)
    {
      hash ^= c;
      hash *= 1099511628211ull;
    }
    std::ostringstream result;
    result << std::hex << hash;
    return result.str();
  }
} // namespace

template <int dim>
ElasticitySolver<dim>::ElasticitySolver(
  const ParameterReader<dim>                  &param,
  parallel::DistributedTriangulationBase<dim> *reference_triangulation)
  : GenericSolver<LA::ParVectorType>(param.output,
                                     param.nonlinear_solver,
                                     param.timer,
                                     param.mesh,
                                     param.time_integration,
                                     param.mms_param,
                                     SolverInfo::SolverType::elasticity)
  , ordering(ComponentOrderingElasticity<dim>())
  , param(param)
  , owned_triangulation(
      reference_triangulation ?
        nullptr :
        std::make_unique<parallel::fullydistributed::Triangulation<dim>>(
          mpi_communicator))
  , triangulation(reference_triangulation ? *reference_triangulation :
                                            *owned_triangulation)
  , dof_handler(triangulation)
  , time_handler(param.time_integration)
{
  if (param.elasticity.write_final_msh)
  {
    AssertThrow(param.mesh.deal_ii_preset_mesh == "none" &&
                  !param.mesh.use_deal_ii_cube_mesh &&
                  !(param.mms_param.enable &&
                    (param.mms_param.use_deal_ii_cube_mesh ||
                     param.mms_param.use_deal_ii_holed_plate_mesh)) &&
                  std::filesystem::path(param.mesh.filename).extension() ==
                    ".msh",
                ExcMessage(
                  "Writing the final deformed mesh requires a Gmsh .msh "
                  "input mesh."));
#if !defined(DEAL_II_GMSH_WITH_API)
    AssertThrow(false,
                ExcMessage("Gmsh API support is required to write the final "
                           "deformed .msh file."));
#endif
  }

  create_quadrature_rules(param.finite_elements,
                          quadrature,
                          face_quadrature,
                          error_quadrature,
                          error_face_quadrature);

  if (param.finite_elements.use_quads)
  {
    mapping =
      std::make_unique<MappingQ<dim>>(param.finite_elements.mapping_degree);
    fe = std::make_unique<FESystem<dim>>(
      FE_Q<dim>(param.finite_elements.mesh_position_degree) ^ dim);
  }
  else
  {
    mapping = std::make_unique<MappingFE<dim>>(
      FE_SimplexP<dim>(param.finite_elements.mapping_degree));
    fe = std::make_unique<FESystem<dim>>(
      FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^ dim);
  }

  position_extractor = FEValuesExtractors::Vector(0);
  position_mask      = fe->component_mask(position_extractor);

  if (param.mms_param.enable)
  {
    for (auto &[norm, handler] : error_handlers)
      handler.create_entry("x");

    // Assign the manufactured solution
    exact_solution = param.mms.exact_mesh_position;

    // Create source term function for the given MMS and override source terms
    source_terms = std::make_shared<ElasticitySolver<dim>::MMSSourceTerm>(
      param.physical_properties, param.mms);
  }
  else
  {
    source_terms   = param.source_terms.elasticity_source;
    exact_solution = std::make_shared<Functions::ZeroFunction<dim>>(dim);
  }

  // Create direct solver
  direct_solver_reuse =
    std::make_unique<PETScWrappers::SparseDirectMUMPSReuse>(solver_control);

  scratch_data = std::make_unique<ScratchData>(
    *fe, *mapping, *quadrature, *face_quadrature, param);
}

template <int dim>
void ElasticitySolver<dim>::MMSSourceTerm::vector_value(
  const Point<dim> &p,
  Vector<double>   &values) const
{
  Tensor<1, dim> f = mms.exact_mesh_position->divergence_elastic_stress_tensor(
    physical_properties.pseudosolids[0], p);

  for (unsigned int d = 0; d < dim; ++d)
    values[d] = f[d];
}

template <int dim>
void ElasticitySolver<dim>::reset()
{
  param.mms_param.current_step = mms_param.current_step;
  param.mms_param.mesh_suffix  = mms_param.mesh_suffix;
  param.mesh.filename          = mesh_param.filename;
  param.time_integration.dt    = time_param.dt;

  // Mesh: release only this solver's DoFs when borrowing the reference mesh.
  dof_handler.clear();
  if (owned_triangulation)
    triangulation.clear();

  // Direct solver
  direct_solver_reuse =
    std::make_unique<PETScWrappers::SparseDirectMUMPSReuse>(solver_control);

  // Time handler (move assign a new time handler)
  time_handler = TimeHandler(param.time_integration);
}

template <int dim>
void ElasticitySolver<dim>::run()
{
  reset();
  const double initial_time = param.time_integration.t_initial;
  for (auto &[id, bc] : param.pseudosolid_bc)
    bc.set_time(initial_time);
  source_terms->set_time(initial_time);
  exact_solution->set_time(initial_time);
  param.physical_properties.set_time(initial_time);
  param.initial_conditions.initial_chns_tracer_callback->set_time(initial_time);
  setup_assemblers();
  if (owned_triangulation)
    MeshTools::read_mesh(triangulation, param);
  setup_dofs();
  create_zero_constraints();
  create_nonzero_constraints();
  create_sparsity_pattern();
  set_initial_conditions();
  if (param.elasticity.presolved_mesh_position_mode ==
        Parameters::Elasticity::PresolvedMeshPositionMode::reuse &&
      try_load_presolved_mesh_cache())
  {
    postprocess_solution();
    return;
  }
  output_results();

  update_boundary_conditions();

  if (param.cahn_hilliard.mff_source_term ==
      Parameters::CahnHilliard<dim>::MeshForcingSourceTerm::chns_form)
  {
    /**
     * Cahn-Hilliard moving-mesh forcing. The compression forcing is steep, so
     * its multiplier is ramped from a small fraction up to its physical value
     * (1) with a continuation method. The user-source multipliers are disabled.
     */
    scratch_data->source_term_fixed_mesh_multiplier  = 0.;
    scratch_data->source_term_moving_mesh_multiplier = 0.;

    const double c_min =
      param.elasticity.presolver_initial_compression_multiplier;
    const unsigned int n_steps = param.elasticity.presolver_continuation_steps;

    AssertThrow(n_steps == 1 || (std::isfinite(c_min) && c_min > 0.),
                ExcMessage(
                  "Multiple presolver continuation steps require a "
                  "strictly positive initial compression multiplier."));

    for (unsigned int n = 0; n < n_steps; ++n)
    {
      // A single step and the final step both solve the physical target.
      scratch_data->chns_compression_multiplier =
        n + 1 == n_steps ?
          1. :
          std::pow(c_min, 1. - static_cast<double>(n) / (n_steps - 1));
      pcout << std::endl;
      pcout << "Continuation method - Step " << n + 1 << "/" << n_steps
            << " : chns compression multiplier = "
            << scratch_data->chns_compression_multiplier << std::endl;
      pcout << std::endl;

      if (param.nonlinear_solver.compare_jacobian_with_finite_differences)
        compare_analytical_matrix_with_fd();
      solve_nonlinear_problem(time_handler);
    }
  }
  else if (param.elasticity.enable_source_term_on_current_mesh)
  {
    /**
     * Continuation method to handle possibly steep source terms evaluated
     * on the current (deformed) mesh.
     */
    const double c_min =
      param.elasticity.min_current_mesh_source_term_multiplier;
    const double c_max =
      param.elasticity.max_current_mesh_source_term_multiplier;
    const unsigned int n_steps = param.elasticity.n_continuation_steps;

    scratch_data->source_term_moving_mesh_multiplier = c_min;
    scratch_data->source_term_fixed_mesh_multiplier  = 0.;

    // Use a geometric progression to increase the continuation parameter
    const double r =
      n_steps > 1 ? std::pow(c_max / c_min, 1.0 / (n_steps - 1)) : 1.;

    for (unsigned int n = 0; n < n_steps; ++n)
    {
      pcout << std::endl;
      pcout << "Continuation method - Step " << n + 1 << "/" << n_steps
            << " : source term multiplier = "
            << scratch_data->source_term_moving_mesh_multiplier << std::endl;
      pcout << std::endl;

      if (param.nonlinear_solver.compare_jacobian_with_finite_differences)
        compare_analytical_matrix_with_fd();
      solve_nonlinear_problem(time_handler);

      scratch_data->source_term_moving_mesh_multiplier *= r;
    }
  }
  else
  {
    // Source term is evaluated on reference mesh and problem is linear
    // This is the case when performing a convergence study with a
    // manufactured solution, for example.
    scratch_data->source_term_moving_mesh_multiplier = 0.;
    scratch_data->source_term_fixed_mesh_multiplier  = 1.;

    if (param.nonlinear_solver.compare_jacobian_with_finite_differences)
      compare_analytical_matrix_with_fd();
    solve_nonlinear_problem(time_handler);
  }

  // Store reference support points before standalone postprocessing moves the
  // mesh.
  if (param.elasticity.presolved_mesh_position_mode !=
      Parameters::Elasticity::PresolvedMeshPositionMode::off)
    write_presolved_mesh_cache();

  postprocess_solution();
}

template <int dim>
void ElasticitySolver<dim>::setup_assemblers()
{
  assemblers.clear();
  Assembly::Elasticity::setup_assemblers<dim, ScratchData, CopyData>(
    param, ordering, assemblers);
}

template <int dim>
void ElasticitySolver<dim>::setup_dofs()
{
  TimerOutput::Scope t(computing_timer, "Setup");

  auto &comm = mpi_communicator;

  // Initialize dof handler
  dof_handler.distribute_dofs(*fe);

  pcout << "Number of degrees of freedom: " << dof_handler.n_dofs()
        << std::endl;

  locally_owned_dofs    = dof_handler.locally_owned_dofs();
  locally_relevant_dofs = DoFTools::extract_locally_relevant_dofs(dof_handler);

  // Initialize parallel vectors
  present_solution.reinit(locally_owned_dofs, locally_relevant_dofs, comm);
  evaluation_point.reinit(locally_owned_dofs, locally_relevant_dofs, comm);

  local_evaluation_point.reinit(locally_owned_dofs, comm);
  newton_update.reinit(locally_owned_dofs, comm);
  system_rhs.reinit(locally_owned_dofs, comm);
}

template <int dim>
void ElasticitySolver<dim>::create_base_constraints(
  const bool                 homogeneous,
  AffineConstraints<double> &constraints)
{
  constraints.clear();
  constraints.reinit(locally_owned_dofs, locally_relevant_dofs);

  BoundaryConditions::apply_mesh_position_boundary_conditions(
    homogeneous,
    0,
    dim,
    dof_handler,
    *mapping,
    param.pseudosolid_bc,
    *exact_solution,
    *param.mms.exact_mesh_position,
    constraints);

  constraints.close();
}

template <int dim>
void ElasticitySolver<dim>::create_zero_constraints()
{
  create_base_constraints(true, zero_constraints);
}

template <int dim>
void ElasticitySolver<dim>::create_nonzero_constraints()
{
  create_base_constraints(false, nonzero_constraints);
}

template <int dim>
void ElasticitySolver<dim>::create_sparsity_pattern()
{
  DynamicSparsityPattern dsp(locally_relevant_dofs);
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  nonzero_constraints,
                                  /* keep_constrained_dofs = */ false);
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
void ElasticitySolver<dim>::set_initial_conditions()
{
  FixedMeshPosition<dim> fixed_mesh(0, dim);
  VectorTools::interpolate(
    *mapping, dof_handler, fixed_mesh, newton_update, position_mask);
  evaluation_point = newton_update;

  // Apply non-homogeneous Dirichlet BC and set as current solution
  nonzero_constraints.distribute(newton_update);
  present_solution = newton_update;
  evaluation_point = newton_update;
}

template <int dim>
void ElasticitySolver<dim>::set_exact_solution()
{
  VectorTools::interpolate(*mapping,
                           dof_handler,
                           *exact_solution,
                           local_evaluation_point,
                           position_mask);
  evaluation_point = local_evaluation_point;
  present_solution = local_evaluation_point;
}

template <int dim>
void ElasticitySolver<dim>::update_boundary_conditions()
{
  local_evaluation_point = present_solution;
  create_nonzero_constraints();
  nonzero_constraints.distribute(local_evaluation_point);
  evaluation_point = local_evaluation_point;
  present_solution = local_evaluation_point;
}

template <int dim>
void ElasticitySolver<dim>::assemble_matrix()
{
  TimerOutput::Scope t(computing_timer, "Assemble matrix");

  CopyData copy_data(*fe);

  system_matrix = 0;
  this->do_matrix_assembly(*this,
                           dof_handler.begin_active(),
                           dof_handler.end(),
                           *scratch_data,
                           copy_data,
                           this->param.nonlinear_solver.analytic_jacobian);
  system_matrix.compress(VectorOperation::add);
}

template <int dim>
void ElasticitySolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();
  copy_data.cell_is_at_boundary   = cell->at_boundary();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell, evaluation_point, source_terms, exact_solution);

  auto &local_matrix = copy_data.local_matrix();
  local_matrix       = 0;

  for (const auto &assembler : assemblers)
    assembler->assemble_matrix(scratch_data, copy_data);

  cell->get_dof_indices(copy_data.dof_indices());
}

template <int dim>
void ElasticitySolver<dim>::copy_local_to_global_matrix(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;
  zero_constraints.distribute_local_to_global(copy_data.local_matrix(),
                                              copy_data.dof_indices(),
                                              system_matrix);
}

template <int dim>
void ElasticitySolver<dim>::compare_analytical_matrix_with_fd()
{
  CopyData copy_data(*fe);
  Verification::compare_analytical_matrix_with_fd<dim>(
    *this,
    *scratch_data,
    copy_data,
    this->param.nonlinear_solver.write_problematic_elements);
}

template <int dim>
void ElasticitySolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(computing_timer, "Assemble RHS");

  system_rhs = 0;

  CopyData copy_data(*fe);

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &ElasticitySolver::assemble_local_rhs,
                  &ElasticitySolver::copy_local_to_global_rhs,
                  *scratch_data,
                  copy_data);

  system_rhs.compress(VectorOperation::add);
}

template <int dim>
void ElasticitySolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();
  copy_data.cell_is_at_boundary   = cell->at_boundary();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell, evaluation_point, source_terms, exact_solution);

  auto &local_rhs = copy_data.local_rhs();
  local_rhs       = 0;

  for (const auto &assembler : assemblers)
    assembler->assemble_rhs(scratch_data, copy_data);

  cell->get_dof_indices(copy_data.dof_indices());
}

template <int dim>
void ElasticitySolver<dim>::copy_local_to_global_rhs(const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;
  zero_constraints.distribute_local_to_global(copy_data.local_rhs(),
                                              copy_data.dof_indices(),
                                              system_rhs);
}

template <int dim>
void ElasticitySolver<dim>::solve_linear_system()
{
  const auto &linear_solver_param = param.linear_solver.at(this->solver_type);

  if (linear_solver_param.method ==
      Parameters::LinearSolver::Method::direct_mumps)
  {
    if (linear_solver_param.reuse)
    {
      solve_linear_system_direct(this,
                                 linear_solver_param,
                                 system_matrix,
                                 locally_owned_dofs,
                                 zero_constraints,
                                 *direct_solver_reuse);
    }
    else
      solve_linear_system_direct(this,
                                 linear_solver_param,
                                 system_matrix,
                                 locally_owned_dofs,
                                 zero_constraints);
  }
  else if (linear_solver_param.method == Parameters::LinearSolver::Method::cg)
  {
    solve_linear_system_cg(this,
                           linear_solver_param,
                           system_matrix,
                           locally_owned_dofs,
                           zero_constraints);
  }
  else if (linear_solver_param.method ==
           Parameters::LinearSolver::Method::gmres)
  {
    AssertThrow(false,
                ExcMessage("GMRES solver is not implemented for "
                           "ElasticitySolver. Use CG unstead."));
  }
  else
  {
    AssertThrow(false, ExcMessage("No known resolution method"));
  }
}

template <int dim>
void ElasticitySolver<dim>::output_results(const Mapping<dim> *output_mapping)
{
  TimerOutput::Scope t(computing_timer, "Write outputs");

  if (param.output.write_results)
  {
    std::vector<std::string> solution_names(dim, "position");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);
    // Partition
    Vector<float> subdomain(triangulation.n_active_cells());
    for (unsigned int i = 0; i < subdomain.size(); ++i)
      subdomain(i) = triangulation.locally_owned_subdomain();
    data_out.add_data_vector(subdomain,
                             "subdomain",
                             DataOut<dim>::type_cell_data);

    data_out.build_patches(output_mapping ? *output_mapping : *mapping, 2);
    data_out.write_vtu_with_pvtu_record(param.output.output_dir,
                                        param.output.output_prefix +
                                          "elasticity",
                                        0,
                                        mpi_communicator,
                                        2);
  }
}

template <int dim>
void ElasticitySolver<dim>::move_mesh()
{
  std::vector<bool> vertex_moved(triangulation.n_vertices(), false);
  for (auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto v : cell->vertex_indices())
        // if (owned_vertices[cell->vertex_index(v)])
        if (!vertex_moved[cell->vertex_index(v)])
        {
          vertex_moved[cell->vertex_index(v)] = true;
          for (unsigned int d = 0; d < dim; ++d)
            cell->vertex(v)[d] = present_solution(cell->vertex_dof_index(v, d));
        }
}

template <int dim>
void ElasticitySolver<dim>::write_final_msh()
{
#if defined(DEAL_II_GMSH_WITH_API)
  const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_communicator);

  std::vector<std::size_t> node_tags;
  std::vector<double>      node_coordinates;
  std::vector<Point<dim>>  evaluation_points;
  bool                     gmsh_initialized_here = false;
  bool                     gmsh_model_owned      = false;
  double                   gmsh_verbosity        = 0.;

  const ScopeExit cleanup_owned_gmsh_state([&]() noexcept {
    if (rank != 0 || !gmsh::isInitialized())
      return;

    if (gmsh_model_owned)
      try
      {
        gmsh::clear();
        gmsh::option::setNumber("General.Verbosity", gmsh_verbosity);
      }
      catch (...)
      {}

    if (gmsh_initialized_here)
      try
      {
        gmsh::finalize();
      }
      catch (...)
      {}
  });

  const auto on_root = [&](const auto &operation, const std::string &context) {
    std::string error;
    if (rank == 0)
      try
      {
        operation();
      }
      catch (const std::exception &exception)
      {
        error = exception.what();
      }
      catch (const std::string &exception)
      {
        error = exception;
      }
      catch (...)
      {
        error = "unknown Gmsh error";
      }
    error = Utilities::MPI::broadcast(mpi_communicator, error, 0);
    AssertThrow(error.empty(), ExcMessage(context + error));
  };

  on_root(
    [&]() {
      gmsh_initialized_here = !gmsh::isInitialized();
      if (gmsh_initialized_here)
        gmsh::initialize();

      // This routine cannot restore an arbitrary caller-owned Gmsh model.
      // Require no caller-owned model so clearing only removes the mesh opened
      // here. Gmsh retains an unnamed empty model after initialize/clear.
      std::vector<std::string> existing_models;
      gmsh::model::list(existing_models);
      bool model_available = existing_models.empty();
      if (existing_models.size() == 1 && existing_models.front().empty())
      {
        gmsh::vectorpair entities;
        gmsh::model::getEntities(entities);
        model_available = entities.empty();
      }
      AssertThrow(model_available,
                  ExcMessage("Cannot write the final mesh while another Gmsh "
                             "model is open."));

      gmsh::option::getNumber("General.Verbosity", gmsh_verbosity);
      gmsh::option::setNumber("General.Verbosity", 2);
      gmsh_model_owned = true;
      gmsh::open(param.mesh.filename);

      std::vector<double> parametric_coordinates;
      gmsh::model::mesh::getNodes(node_tags,
                                  node_coordinates,
                                  parametric_coordinates,
                                  -1,
                                  -1,
                                  false,
                                  false);

      evaluation_points.reserve(node_tags.size());
      for (unsigned int i = 0; i < node_tags.size(); ++i)
      {
        Point<dim> point;
        for (unsigned int d = 0; d < dim; ++d)
          point[d] = node_coordinates[3 * i + d];
        evaluation_points.push_back(point);
      }
    },
    "Could not open the Gmsh input mesh: ");

  present_solution.update_ghost_values();

  Utilities::MPI::RemotePointEvaluation<dim, dim> cache;
  const auto                                      deformed_positions =
    VectorTools::point_values<dim>(*mapping,
                                   dof_handler,
                                   present_solution,
                                   evaluation_points,
                                   cache,
                                   VectorTools::EvaluationFlags::avg,
                                   ordering.x_lower);

  const unsigned int all_points_found =
    Utilities::MPI::min(cache.all_points_found() ? 1u : 0u, mpi_communicator);
  if (all_points_found == 0)
  {
    AssertThrow(false,
                ExcMessage(
                  "Could not evaluate the deformed mesh position at all Gmsh "
                  "nodes when writing the final .msh file."));
  }

  const std::string output_mesh_filename = param.output.output_dir +
                                           param.output.output_prefix +
                                           "elasticity_final_mesh.msh";
  on_root(
    [&]() {
      AssertDimension(deformed_positions.size(), node_tags.size());

      std::vector<double> coordinates(3, 0.0);
      for (unsigned int i = 0; i < node_tags.size(); ++i)
      {
        for (unsigned int d = 0; d < dim; ++d)
          coordinates[d] = deformed_positions[i][d];
        if constexpr (dim == 2)
          coordinates[2] = node_coordinates[3 * i + 2];

        gmsh::model::mesh::setNode(node_tags[i], coordinates, {});
      }

      gmsh::write(output_mesh_filename);
    },
    "Could not write the final Gmsh mesh: ");

  pcout << "Wrote final deformed mesh to " << output_mesh_filename << std::endl;
#else
  AssertThrow(false,
              ExcMessage("Gmsh API support is required to write the final "
                         "deformed .msh file."));
#endif
}

template <int dim>
std::string ElasticitySolver<dim>::presolved_mesh_fingerprint() const
{
  // Cell geometry, connectivity, materials and boundary ids are independent of
  // the MPI partition. This also detects changes to generated/borrowed meshes.
  std::vector<std::string> local_cells;
  for (const auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
    {
      std::ostringstream entry;
      entry << std::setprecision(17) << cell->id().to_string() << ':'
            << cell->material_id();
      for (unsigned int v = 0; v < cell->n_vertices(); ++v)
        for (unsigned int d = 0; d < dim; ++d)
          entry << ':' << cell->vertex(v)[d];
      for (unsigned int f = 0; f < cell->n_faces(); ++f)
        entry << ':' << cell->face(f)->boundary_id();
      local_cells.push_back(entry.str());
    }
  const auto gathered =
    Utilities::MPI::gather(mpi_communicator, local_cells, 0);
  std::string mesh_hash;
  if (mpi_rank == 0)
  {
    std::vector<std::string> cells;
    for (const auto &part : gathered)
      cells.insert(cells.end(), part.begin(), part.end());
    std::sort(cells.begin(), cells.end());
    std::ostringstream mesh;
    for (const auto &cell : cells)
      mesh << cell << '\n';
    mesh_hash = fingerprint_hash(mesh.str());
  }
  mesh_hash = Utilities::MPI::broadcast(mpi_communicator, mesh_hash, 0);
  const auto        &ch    = param.cahn_hilliard;
  const auto        &solid = param.physical_properties.pseudosolids[0];
  std::ostringstream fingerprint;
  fingerprint << std::setprecision(17)
              << "fez-standard-presolver-v2;dim=" << dim
              << ";mesh=" << mesh_hash << ";ndofs=" << dof_handler.n_dofs()
              << ";input=" << param.elasticity.presolved_mesh_input_parameters
              << ";initial_time=" << param.time_integration.t_initial
              << ";mff=" << static_cast<int>(ch.mff_source_term)
              << ";eps=" << ch.epsilon_interface
              << ";compression=" << ch.mff_physics_compression_factor
              << ";gamma=" << ch.mff_regularization_gamma;
  // Constants may have been updated after reading the input file.
  for (const auto &function :
       {param.initial_conditions.initial_chns_tracer_callback,
        solid.lame_mu_fun,
        solid.lame_lambda_fun})
  {
    fingerprint << ";function=" << function->get_function_expression();
    for (const auto &[name, value] : function->get_constants())
      fingerprint << ';' << name << '=' << value;
  }
  // Transport is deliberately excluded: it is inactive in the presolver.
  return fingerprint_hash(fingerprint.str());
}

template <int dim>
void ElasticitySolver<dim>::write_presolved_mesh_cache() const
{
  const auto file = std::filesystem::path(param.output.output_dir) /
                    param.elasticity.presolved_mesh_position_file;
  const auto                 temporary   = file.string() + ".tmp";
  const std::string          fingerprint = presolved_mesh_fingerprint();
  std::vector<unsigned char> components;
  fill_dofs_to_component(dof_handler, locally_relevant_dofs, components);
  const auto points =
    DoFTools::map_dofs_to_support_points(*mapping, dof_handler);
  std::vector<PresolvedMeshCacheEntry<dim>> local_entries;
  for (const auto i : locally_owned_dofs)
  {
    PresolvedMeshCacheEntry<dim> entry;
    for (unsigned int d = 0; d < dim; ++d)
      entry.support_point[d] = points.at(i)[d];
    entry.component = components[locally_relevant_dofs.index_within_set(i)];
    entry.value     = present_solution[i];
    local_entries.push_back(entry);
  }
  const auto gathered =
    Utilities::MPI::gather(mpi_communicator, local_entries, 0);
  std::string error;
  if (mpi_rank == 0)
    try
    {
      std::vector<PresolvedMeshCacheEntry<dim>> entries;
      for (const auto &part : gathered)
        entries.insert(entries.end(), part.begin(), part.end());
      {
        std::ofstream stream(temporary);
        stream.exceptions(std::ios::failbit | std::ios::badbit);
        {
          boost::archive::text_oarchive archive(stream);
          archive << fingerprint << entries;
        }
        stream.close();
      }
      std::filesystem::rename(temporary, file);
    }
    catch (const std::exception &e)
    {
      error = e.what();
    }
  // Every rank must observe an I/O failure before leaving this collective path.
  error = Utilities::MPI::broadcast(mpi_communicator, error, 0);
  AssertThrow(error.empty(),
              ExcMessage("Could not write presolved mesh cache: " + error));
  pcout << "Wrote presolved mesh position cache to " << file.string()
        << std::endl;
}

template <int dim>
bool ElasticitySolver<dim>::try_load_presolved_mesh_cache()
{
  const auto file = std::filesystem::path(param.output.output_dir) /
                    param.elasticity.presolved_mesh_position_file;
  const std::string expected = presolved_mesh_fingerprint();
  std::vector<PresolvedMeshCacheEntry<dim>> entries;
  bool                                      usable = true;
  std::string                               reason;
  try
  {
    std::ifstream stream(file);
    if (!stream)
    {
      usable = false;
      reason = "missing cache file";
    }
    else
    {
      boost::archive::text_iarchive archive(stream);
      std::string                   stored;
      archive >> stored;
      if (stored != expected)
      {
        usable = false;
        reason = "presolver parameters or reference mesh changed";
      }
      else
        archive >> entries;
    }
  }
  catch (const std::exception &e)
  {
    usable = false;
    reason = "invalid cache: " + std::string(e.what());
  }
  const auto fail = [&]() {
    pcout << "Presolved mesh position cache cannot be reused: "
          << (reason.empty() ? "invalid cache on another rank" : reason)
          << std::endl;
    return false;
  };
  if (!Utilities::MPI::min(usable ? 1 : 0, mpi_communicator))
    return fail();
  std::map<std::string, double> cached;
  for (const auto &entry : entries)
  {
    bool finite = std::isfinite(entry.value);
    for (const auto coordinate : entry.support_point)
      finite &= std::isfinite(coordinate);
    if (!finite || entry.component >= dim ||
        !cached
           .emplace(cache_entry_key<dim>(entry.support_point, entry.component),
                    entry.value)
           .second)
      usable = false;
  }
  usable &= cached.size() == dof_handler.n_dofs();
  std::vector<unsigned char> components;
  fill_dofs_to_component(dof_handler, locally_relevant_dofs, components);
  const auto points =
    DoFTools::map_dofs_to_support_points(*mapping, dof_handler);
  LA::ParVectorType loaded(locally_owned_dofs, mpi_communicator);
  for (const auto i : locally_owned_dofs)
  {
    std::array<double, dim> point;
    for (unsigned int d = 0; d < dim; ++d)
      point[d] = points.at(i)[d];
    const auto component =
      components[locally_relevant_dofs.index_within_set(i)];
    const auto value = cached.find(cache_entry_key<dim>(point, component));
    if (value == cached.end())
      usable = false;
    else
      loaded[i] = value->second;
  }
  if (!Utilities::MPI::min(usable ? 1 : 0, mpi_communicator))
  {
    reason = "missing, duplicated or invalid support-point values";
    return fail();
  }
  loaded.compress(VectorOperation::insert);
  local_evaluation_point = loaded;
  present_solution       = loaded;
  evaluation_point       = loaded;
  pcout << "Loaded presolved mesh position cache from " << file.string()
        << std::endl;
  return true;
}

template <int dim>
void ElasticitySolver<dim>::compute_errors()
{
  TimerOutput::Scope t(computing_timer, "Compute errors");

  const unsigned int n_active_cells = triangulation.n_active_cells();
  Vector<double>     cellwise_errors(n_active_cells);
  const ComponentSelectFunction<dim> position_comp_select(0, dim);

  for (auto &[norm, handler] : error_handlers)
  {
    handler.add_reference_data("n_elm", triangulation.n_global_active_cells());
    handler.add_reference_data("n_dof", dof_handler.n_dofs());
    const double err =
      compute_error_norm<dim, LA::ParVectorType>(triangulation,
                                                 *mapping,
                                                 dof_handler,
                                                 present_solution,
                                                 *exact_solution,
                                                 cellwise_errors,
                                                 *error_quadrature,
                                                 norm,
                                                 &position_comp_select);
    handler.add_error("x", err);
  }
}

template <int dim>
void ElasticitySolver<dim>::postprocess_solution()
{
  // Compute error *before* moving mesh for visualization (-:
  if (param.mms_param.enable)
    compute_errors();

  // Evaluate on the reference mesh, including when the position came from
  // the cache. A borrowed CHNS triangulation must remain unchanged.
  if (param.elasticity.write_final_msh)
    write_final_msh();

  if (owned_triangulation)
  {
    move_mesh();
    output_results();
  }
  else
  {
    const MappingFEField<dim, dim, LA::ParVectorType> deformed_mapping(
      dof_handler, present_solution, position_mask);
    output_results(&deformed_mapping);
  }
}

// Explicit instantiation
template class ElasticitySolver<2>;
template class ElasticitySolver<3>;
