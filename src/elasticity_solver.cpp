
#include <assembly/elasticity_assemblers.h>
#include <compare_matrix.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <elasticity_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <post_processing_tools.h>
#include <solver_info.h>
#include <utilities.h>

#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/vector.hpp>

#include <array>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

namespace
{
  // FNV-1a hash of a file's contents, used to detect mesh changes in the
  // presolved-mesh-position cache fingerprint.
  std::string hash_file_contents(const std::string &filename)
  {
    std::ifstream input(filename, std::ios::binary);
    if (!input)
      return "unavailable:" + filename;

    std::uint64_t hash = 14695981039346656037ull;
    char          character;
    while (input.get(character))
    {
      hash ^= static_cast<unsigned char>(character);
      hash *= 1099511628211ull;
    }
    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
  }

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
} // namespace

template <int dim>
ElasticitySolver<dim>::ElasticitySolver(
  const ParameterReader<dim> &param,
  const bool                  with_enlarged_psi)
  : GenericSolver<LA::ParVectorType>(param.output,
                                     param.nonlinear_solver,
                                     param.timer,
                                     param.mesh,
                                     param.time_integration,
                                     param.mms_param,
                                     SolverInfo::SolverType::elasticity)
  , ordering(ComponentOrderingElasticity<dim>())
  , param(param)
  , triangulation(mpi_communicator)
  , dof_handler(triangulation)
  , time_handler(param.time_integration)
  , with_enlarged_psi(with_enlarged_psi)
{
  create_quadrature_rules(param.finite_elements,
                          quadrature,
                          face_quadrature,
                          error_quadrature,
                          error_face_quadrature);

  if (param.finite_elements.use_quads)
  {
    mapping =
      std::make_unique<MappingQ<dim>>(param.finite_elements.mapping_degree);
    if (with_enlarged_psi)
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(FE_Q<dim>(param.finite_elements.mesh_position_degree) ^
                      dim),
        FE_Q<dim>(param.finite_elements.tracer_degree));
    else
      fe = std::make_unique<FESystem<dim>>(
        FE_Q<dim>(param.finite_elements.mesh_position_degree) ^ dim);
  }
  else
  {
    mapping = std::make_unique<MappingFE<dim>>(
      FE_SimplexP<dim>(param.finite_elements.mapping_degree));
    if (with_enlarged_psi)
      fe = std::make_unique<FESystem<dim>>(
        FESystem<dim>(
          FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^ dim),
        FE_SimplexP<dim>(param.finite_elements.tracer_degree));
    else
      fe = std::make_unique<FESystem<dim>>(
        FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^ dim);
  }

  if (with_enlarged_psi)
    ordering = ComponentOrderingElasticity<dim, true>();

  position_extractor = FEValuesExtractors::Vector(0);
  position_mask      = fe->component_mask(position_extractor);
  if (with_enlarged_psi)
  {
    psi_extractor = FEValuesExtractors::Scalar(dim);
    psi_mask      = fe->component_mask(psi_extractor);
  }

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

  const bool evaluate_chns_forcing =
    param.cahn_hilliard.mff_source_term ==
    Parameters::CahnHilliard<dim>::MeshForcingSourceTerm::chns_form;
  scratch_data = std::make_unique<ScratchData>(*fe,
                                               *mapping,
                                               *quadrature,
                                               *face_quadrature,
                                               param,
                                               evaluate_chns_forcing,
                                               with_enlarged_psi);
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

  // Mesh
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
  setup_assemblers();
  MeshTools::read_mesh(triangulation, param);
  setup_dofs();
  create_zero_constraints();
  create_nonzero_constraints();
  create_sparsity_pattern();
  set_initial_conditions();
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

    const double r =
      n_steps > 1 ? std::pow(1.0 / c_min, 1.0 / (n_steps - 1)) : 1.;

    scratch_data->chns_compression_multiplier = c_min;

    for (unsigned int n = 0; n < n_steps; ++n)
    {
      pcout << std::endl;
      pcout << "Continuation method - Step " << n + 1 << "/" << n_steps
            << " : chns compression multiplier = "
            << scratch_data->chns_compression_multiplier << std::endl;
      pcout << std::endl;

      if (param.nonlinear_solver.compare_jacobian_with_finite_differences)
        compare_analytical_matrix_with_fd();
      solve_nonlinear_problem(time_handler);

      scratch_data->chns_compression_multiplier *= r;
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

  // Write the presolved mesh position cache *before* postprocessing, which
  // moves the mesh and would otherwise deform the support points used as cache
  // keys.
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
    fe->n_components(),
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

  system_matrix = 0;

  CopyData copy_data(*fe);

#if defined(FEZ_WITH_PETSC)
  AssertThrow(
    MultithreadInfo::n_threads() == 1,
    ExcMessage(
      "Assembly is running with more than 1 thread, but uses PETSc wrappers "
      "for parallel matrix and vectors, which are not thread safe."));
#endif

  auto assembly_ptr =
    this->param.nonlinear_solver.analytic_jacobian ?
      &ElasticitySolver::assemble_local_matrix :
      &ElasticitySolver::assemble_local_matrix_finite_differences;

  // Assemble matrix (multithreaded if supported)
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  assembly_ptr,
                  &ElasticitySolver::copy_local_to_global_matrix,
                  *scratch_data,
                  copy_data);
  system_matrix.compress(VectorOperation::add);
}

template <int dim>
void ElasticitySolver<dim>::assemble_local_matrix_finite_differences(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  Verification::compute_local_matrix_finite_differences<dim>(
    cell,
    *this,
    &ElasticitySolver::assemble_local_rhs,
    scratch_data,
    copy_data);
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
    &ElasticitySolver::assemble_local_matrix,
    &ElasticitySolver::assemble_local_rhs,
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
void ElasticitySolver<dim>::copy_local_to_global_rhs(
  const CopyData &copy_data)
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
void ElasticitySolver<dim>::output_results()
{
  TimerOutput::Scope t(computing_timer, "Write outputs");

  if (param.output.write_results)
  {
    std::vector<std::string> solution_names(dim, "position");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    // Enlarged presolver: also write the reconstructed marker psi (it is only
    // visualized here, never injected into the CHNS solver).
    if (with_enlarged_psi)
    {
      solution_names.emplace_back("psi");
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
    }
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

    data_out.build_patches(*mapping, 2);
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
std::string ElasticitySolver<dim>::presolved_mesh_fingerprint() const
{
  const auto        &solid = param.physical_properties.pseudosolids[0];
  const auto        &ch    = param.cahn_hilliard;
  std::ostringstream fingerprint;
  fingerprint << "mesh=" << param.mesh.filename << "#"
              << hash_file_contents(param.mesh.filename)
              << ";degree=" << param.finite_elements.mesh_position_degree
              << ";quads=" << param.finite_elements.use_quads
              << ";ndofs=" << dof_handler.n_dofs()
              << ";model=" << static_cast<int>(solid.constitutive_model)
              << ";lame_mu=" << solid.lame_mu_fun->get_function_expression()
              << ";lame_lambda="
              << solid.lame_lambda_fun->get_function_expression()
              << ";ogden_beta=" << solid.ogden_beta
              // Moving-mesh forcing parameters that shape the presolved mesh.
              << ";mff=" << static_cast<int>(ch.mff_source_term)
              << ";eps=" << ch.epsilon_interface
              << ";compression=" << ch.mff_physics_compression_factor
              << ";gamma=" << ch.mff_regularization_gamma
              // Enlarged (psi) presolver: marker length scale, enlarged
              // compression and equalization exponent all change the mesh.
              << ";enlarged=" << with_enlarged_psi
              << ";psi_width=" << ch.psi_interface_width_factor
              << ";enl_compression=" << ch.mff_enlarged_compression_factor
              << ";equal_exp=" << ch.mff_enlarged_factor_equalization_exponent
              << ";mult0="
              << param.elasticity.presolver_initial_compression_multiplier
              << ";steps=" << param.elasticity.presolver_continuation_steps
              << ";phi="
              << param.initial_conditions.initial_chns_tracer_callback
                   ->get_function_expression();
  return fingerprint.str();
}

template <int dim>
void ElasticitySolver<dim>::write_presolved_mesh_cache() const
{
  const std::string cache_file =
    param.elasticity.presolved_mesh_position_file;
  const std::string temporary_file = "tmp." + cache_file;

  std::vector<unsigned char> dofs_to_component;
  fill_dofs_to_component(dof_handler, locally_relevant_dofs, dofs_to_component);
  const auto support_points =
    DoFTools::map_dofs_to_support_points(*mapping, dof_handler);

  std::vector<PresolvedMeshCacheEntry<dim>> local_entries;
  local_entries.reserve(locally_owned_dofs.n_elements());
  for (const auto dof : locally_owned_dofs)
  {
    PresolvedMeshCacheEntry<dim> entry;
    const auto                  &point = support_points.at(dof);
    for (unsigned int d = 0; d < dim; ++d)
      entry.support_point[d] = point[d];
    entry.component =
      dofs_to_component[locally_relevant_dofs.index_within_set(dof)];
    entry.value = present_solution[dof];
    local_entries.push_back(entry);
  }

  const auto gathered =
    Utilities::MPI::gather(mpi_communicator, local_entries, 0);
  if (mpi_rank == 0)
  {
    std::vector<PresolvedMeshCacheEntry<dim>> entries;
    for (const auto &rank_entries : gathered)
      entries.insert(entries.end(), rank_entries.begin(), rank_entries.end());

    std::ofstream cache(param.output.output_dir + temporary_file);
    AssertThrow(cache, ExcMessage("Could not write presolved mesh cache."));
    boost::archive::text_oarchive archive(cache);
    const std::string fingerprint = presolved_mesh_fingerprint();
    archive << fingerprint;
    archive << entries;
  }

  MPI_Barrier(mpi_communicator);
  replace_temporary_files(param.output.output_dir,
                          temporary_file,
                          cache_file,
                          mpi_communicator);
  pcout << "Wrote presolved mesh position cache to "
        << param.output.output_dir + cache_file << std::endl;
}

template <int dim>
bool ElasticitySolver<dim>::try_load_presolved_mesh_cache()
{
  using Mode      = Parameters::Elasticity::PresolvedMeshPositionMode;
  const auto mode = param.elasticity.presolved_mesh_position_mode;
  AssertThrow(mode == Mode::reuse,
              ExcMessage("try_load_presolved_mesh_cache should only be called "
                         "in 'reuse' mode."));

  reset();
  MeshTools::read_mesh(triangulation, param);
  setup_dofs();

  const std::string cache_file =
    param.output.output_dir + param.elasticity.presolved_mesh_position_file;
  const std::string expected_fingerprint = presolved_mesh_fingerprint();

  bool                                      usable = true;
  std::string                               reason;
  std::vector<PresolvedMeshCacheEntry<dim>> entries;
  {
    std::ifstream cache(cache_file);
    if (!cache)
    {
      usable = false;
      reason = "missing cache file " + cache_file;
    }
    else
      try
      {
        boost::archive::text_iarchive archive(cache);
        std::string                   cached_fingerprint;
        archive >> cached_fingerprint;
        if (cached_fingerprint != expected_fingerprint)
        {
          usable = false;
          reason = "presolver-defining parameters changed";
        }
        else
          archive >> entries;
      }
      catch (const std::exception &exception)
      {
        usable = false;
        reason = "could not read cache: " + std::string(exception.what());
      }
  }

  const auto fail = [&](const std::string &message) -> bool {
    pcout << "Presolved mesh position cache cannot be reused: " << message
          << std::endl;
    return false;
  };

  if (Utilities::MPI::min(usable ? 1 : 0, mpi_communicator) != 1)
    return fail(reason.empty() ? "cache invalid on another MPI rank" : reason);

  std::map<std::string, double> cached_values;
  for (const auto &entry : entries)
    cached_values[cache_entry_key<dim>(entry.support_point, entry.component)] =
      entry.value;

  std::vector<unsigned char> dofs_to_component;
  fill_dofs_to_component(dof_handler, locally_relevant_dofs, dofs_to_component);
  const auto support_points =
    DoFTools::map_dofs_to_support_points(*mapping, dof_handler);

  local_evaluation_point = 0.;
  bool        found = true;
  std::string missing;
  for (const auto dof : locally_owned_dofs)
  {
    const unsigned int component =
      dofs_to_component[locally_relevant_dofs.index_within_set(dof)];
    std::array<double, dim> point;
    for (unsigned int d = 0; d < dim; ++d)
      point[d] = support_points.at(dof)[d];
    const auto value =
      cached_values.find(cache_entry_key<dim>(point, component));
    if (value == cached_values.end())
    {
      found   = false;
      missing = "cache is missing local support points";
      break;
    }
    local_evaluation_point[dof] = value->second;
  }

  if (Utilities::MPI::min(found ? 1 : 0, mpi_communicator) != 1)
    return fail(missing.empty() ? "support points missing on another rank" :
                                  missing);

  local_evaluation_point.compress(VectorOperation::insert);
  present_solution = local_evaluation_point;
  present_solution.update_ghost_values();
  evaluation_point = present_solution;

  pcout << "Loaded presolved mesh position cache from " << cache_file
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

  move_mesh();
  output_results();
}

// Explicit instantiation
template class ElasticitySolver<2>;
template class ElasticitySolver<3>;
