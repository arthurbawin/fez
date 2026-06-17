#include <assembly/chns_enlarged_forms.h>
#include <assembly/moving_mesh_forcing_forms.h>
#include <assembly/pseudosolid_forms.h>
#include <boundary_conditions.h>
#include <boost/archive/text_iarchive.hpp>
#include <boost/archive/text_oarchive.hpp>
#include <boost/serialization/string.hpp>
#include <boost/serialization/vector.hpp>
#include <compare_matrix.h>
#include <deal.II/base/work_stream.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/lac/generic_linear_algebra.h>
#include <deal.II/lac/sparsity_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools_evaluate.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/vector_tools_interpolate.h>
#include <errors.h>
#include <linear_elasticity_solver.h>
#include <linear_solver.h>
#include <mesh.h>
#include <post_processing_tools.h>
#include <solver_info.h>
#include <utilities.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <map>
#include <sstream>

#if defined(DEAL_II_GMSH_WITH_API)
#  include <gmsh.h>
#endif

namespace
{
  constexpr unsigned int presolved_mesh_position_cache_format_version = 5;

  bool
  has_msh_extension(const std::string &filename)
  {
    return filename.size() >= 4 &&
           filename.substr(filename.size() - 4, 4) == ".msh";
  }

  std::string
  hash_file_contents(const std::string &filename)
  {
    if (filename.empty())
      return "";

    std::ifstream input(filename, std::ios::binary);
    if (!input)
      return "unavailable:" + filename;

    std::uint64_t hash = 14695981039346656037ull;
    char          c;
    while (input.get(c))
    {
      hash ^= static_cast<unsigned char>(c);
      hash *= 1099511628211ull;
    }

    std::ostringstream out;
    out << std::hex << hash;
    return out.str();
  }

  struct PresolvedMeshPositionCacheMetadata
  {
    unsigned int       format_version;
    unsigned int       dimension;
    bool               use_quads;
    unsigned int       mesh_position_degree;
    unsigned long long n_global_active_cells;
    unsigned long long n_dofs;
    std::string        mesh_filename;
    std::string        mesh_hash;
    std::string        presolved_fields;
    std::string        presolver_fingerprint;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/)
    {
      ar &format_version;
      ar &dimension;
      ar &use_quads;
      ar &mesh_position_degree;
      ar &n_global_active_cells;
      ar &n_dofs;
      ar &mesh_filename;
      ar &mesh_hash;
      ar &presolved_fields;
      ar &presolver_fingerprint;
    }
  };

  std::string
  metadata_mismatch_reason(
    const PresolvedMeshPositionCacheMetadata &cached,
    const PresolvedMeshPositionCacheMetadata &expected)
  {
    if (cached.format_version != expected.format_version)
      return "cache format version changed";
    if (cached.dimension != expected.dimension)
      return "dimension changed";
    if (cached.use_quads != expected.use_quads)
      return "finite element cell type changed";
    if (cached.mesh_position_degree != expected.mesh_position_degree)
      return "mesh position degree changed";
    if (cached.n_global_active_cells != expected.n_global_active_cells)
      return "number of active cells changed";
    if (cached.n_dofs != expected.n_dofs)
      return "number of linear-elasticity DoFs changed";
    if (cached.mesh_filename != expected.mesh_filename)
      return "mesh filename changed";
    if (cached.mesh_hash != expected.mesh_hash)
      return "mesh file contents changed";
    if (cached.presolved_fields != expected.presolved_fields)
      return "presolved field mode changed";
    if (cached.presolver_fingerprint != expected.presolver_fingerprint)
      return "presolver-defining parameters changed";

    return "";
  }

  template <int dim>
  struct PresolvedMeshPositionCacheEntry
  {
    std::array<double, dim> support_point;
    unsigned int            component;
    double                  value;

    template <class Archive>
    void serialize(Archive &ar, const unsigned int /*version*/)
    {
      for (auto &coordinate : support_point)
        ar &coordinate;
      ar &component;
      ar &value;
    }
  };

  template <std::size_t dim>
  std::string cache_entry_key(const std::array<double, dim> &support_point,
                              const unsigned int             component)
  {
    std::ostringstream key;
    key << component;
    key << std::setprecision(17);
    for (const auto coordinate : support_point)
      key << ":" << coordinate;
    return key.str();
  }

  template <int dim>
  std::string cache_entry_key(const Point<dim>    &support_point,
                              const unsigned int  component)
  {
    std::array<double, dim> support_point_array;
    for (unsigned int d = 0; d < dim; ++d)
      support_point_array[d] = support_point[d];

    return cache_entry_key(support_point_array, component);
  }
} // namespace

template <int dim>
LinearElasticitySolver<dim>::LinearElasticitySolver(
  const ParameterReader<dim> &param,
  const PresolvedCHNSFields   presolved_chns_fields)
  : GenericSolver<LA::ParVectorType>(param.output,
                                     param.nonlinear_solver,
                                     param.timer,
                                     param.mesh,
                                     param.time_integration,
                                     param.mms_param,
                                     SolverInfo::SolverType::linear_elasticity)
  , param(param)
  , triangulation(mpi_communicator)
  , dof_handler(triangulation)
  , time_handler(param.time_integration)
  , presolved_chns_fields(presolved_chns_fields)
{
  create_quadrature_rules(param.finite_elements,
                          quadrature,
                          face_quadrature,
                          error_quadrature,
                          error_face_quadrature);

  if (param.finite_elements.use_quads)
  {
    mapping = std::make_unique<MappingQ<dim>>(1);
    if (presolved_chns_fields == PresolvedCHNSFields::phi_psi)
      fe = std::make_unique<FESystem<dim>>(
        FE_Q<dim>(param.finite_elements.mesh_position_degree),
        dim,
        FE_Q<dim>(param.finite_elements.tracer_degree),
        1,
        FE_Q<dim>(param.finite_elements.tracer_degree),
        1);
    else if (presolved_chns_fields == PresolvedCHNSFields::phi)
      fe = std::make_unique<FESystem<dim>>(
        FE_Q<dim>(param.finite_elements.mesh_position_degree),
        dim,
        FE_Q<dim>(param.finite_elements.tracer_degree),
        1);
    else
      fe = std::make_unique<FESystem<dim>>(
        FE_Q<dim>(param.finite_elements.mesh_position_degree) ^ dim);
  }
  else
  {
    mapping = std::make_unique<MappingFE<dim>>(FE_SimplexP<dim>(1));
    if (presolved_chns_fields == PresolvedCHNSFields::phi_psi)
      fe = std::make_unique<FESystem<dim>>(
        FE_SimplexP<dim>(param.finite_elements.mesh_position_degree),
        dim,
        FE_SimplexP<dim>(param.finite_elements.tracer_degree),
        1,
        FE_SimplexP<dim>(param.finite_elements.tracer_degree),
        1);
    else if (presolved_chns_fields == PresolvedCHNSFields::phi)
      fe = std::make_unique<FESystem<dim>>(
        FE_SimplexP<dim>(param.finite_elements.mesh_position_degree),
        dim,
        FE_SimplexP<dim>(param.finite_elements.tracer_degree),
        1);
    else
      fe = std::make_unique<FESystem<dim>>(
        FE_SimplexP<dim>(param.finite_elements.mesh_position_degree) ^ dim);
  }

  position_extractor = FEValuesExtractors::Vector(0);
  position_mask      = fe->component_mask(position_extractor);
  presolved_field_mask = position_mask;
  if (has_presolved_tracer())
  {
    tracer_extractor = FEValuesExtractors::Scalar(dim);
    tracer_mask      = fe->component_mask(tracer_extractor);
    for (unsigned int c = 0; c < presolved_field_mask.size(); ++c)
      presolved_field_mask.set(c, presolved_field_mask[c] || tracer_mask[c]);
    if (has_presolved_psi())
    {
      psi_extractor = FEValuesExtractors::Scalar(dim + 1);
      psi_mask      = fe->component_mask(psi_extractor);
      for (unsigned int c = 0; c < presolved_field_mask.size(); ++c)
        presolved_field_mask.set(c, presolved_field_mask[c] || psi_mask[c]);
    }

    if (has_presolved_psi())
      presolver_ordering =
        std::make_unique<ComponentOrderingCHNSPresolver<dim, true>>();
    else
      presolver_ordering =
        std::make_unique<ComponentOrderingCHNSPresolver<dim, false>>();

    presolver_coupling_table.reinit(presolver_ordering->n_components,
                                    presolver_ordering->n_components);
    for (unsigned int i = 0; i < presolver_ordering->n_components; ++i)
      for (unsigned int j = 0; j < presolver_ordering->n_components; ++j)
        presolver_coupling_table[i][j] = DoFTools::always;
  }

  if (param.mms_param.enable)
  {
    for (auto &[norm, handler] : error_handlers)
      handler.create_entry("x");

    // Assign the manufactured solution
    exact_solution = param.mms.exact_mesh_position;

    // Create source term function for the given MMS and override source terms
    source_terms = std::make_shared<LinearElasticitySolver<dim>::MMSSourceTerm>(
      param.physical_properties, param.mms);
  }
  else
  {
    source_terms   = param.source_terms.linear_elasticity_source;
    exact_solution = std::make_shared<Functions::ZeroFunction<dim>>(dim);
  }

  // Create direct solver
  direct_solver_reuse =
    std::make_unique<PETScWrappers::SparseDirectMUMPSReuse>(solver_control);

  scratch_data = std::make_unique<ScratchData>(*fe,
                                               *mapping,
                                               dof_handler,
                                               position_mask,
                                               *quadrature,
                                               *face_quadrature,
                                               param,
                                               has_presolved_tracer(),
                                               has_presolved_psi());
}

template <int dim>
std::string LinearElasticitySolver<dim>::get_presolved_fields_name() const
{
  if (presolved_chns_fields == PresolvedCHNSFields::phi_psi)
    return "x+phi+psi";
  if (presolved_chns_fields == PresolvedCHNSFields::phi)
    return "x+phi";
  return "x";
}

template <int dim>
void LinearElasticitySolver<dim>::MMSSourceTerm::vector_value(
  const Point<dim> &p,
  Vector<double>   &values) const
{
  const auto    &pseudosolid = physical_properties.pseudosolids[0];
  Tensor<1, dim> f;

  if (pseudosolid.constitutive_model ==
      Parameters::PseudoSolid<dim>::ConstitutiveModel::neo_hookean)
    f = mms.exact_mesh_position
          ->divergence_neo_hookean_stress_variable_coefficients(
            p, pseudosolid.lame_mu_fun, pseudosolid.lame_lambda_fun);
	  else if (pseudosolid.constitutive_model ==
	           Parameters::PseudoSolid<dim>::ConstitutiveModel::ogden)
	    f = mms.exact_mesh_position
	          ->divergence_ogden_stress_variable_coefficients(
	            p,
	            pseudosolid.lame_mu_fun,
	            pseudosolid.lame_lambda_fun,
	            pseudosolid.ogden_beta);
  else
    f = mms.exact_mesh_position
          ->divergence_linear_elastic_stress_variable_coefficients(
            p, pseudosolid.lame_mu_fun, pseudosolid.lame_lambda_fun);

  for (unsigned int d = 0; d < dim; ++d)
    values[d] = f[d];
}

template <int dim>
void LinearElasticitySolver<dim>::reset()
{
  param.mms_param.current_step = mms_param.current_step;
  param.mms_param.mesh_suffix  = mms_param.mesh_suffix;
  param.mesh.filename          = mesh_param.filename;
  param.time_integration.dt    = time_param.dt;
  strain_cache_is_valid        = false;
  cached_strain_trace.reinit(0);
  cached_strain_tensors.clear();

  // Mesh
  triangulation.clear();

  // Direct solver
  direct_solver_reuse =
    std::make_unique<PETScWrappers::SparseDirectMUMPSReuse>(solver_control);

  // Time handler (move assign a new time handler)
  time_handler = TimeHandler(param.time_integration);
}

template <int dim>
void LinearElasticitySolver<dim>::run()
{
  reset();
  MeshTools::read_mesh(triangulation, param);
  setup_dofs();
  create_zero_constraints();
  create_nonzero_constraints();
  create_sparsity_pattern();
  set_initial_conditions();
  output_results();

  update_boundary_conditions();

  if (has_presolved_tracer() ||
      param.linear_elasticity.enable_source_term_on_current_mesh)
  {
    /**
     * Continuation method to handle possibly steep source terms evaluated
     * on the current (deformed) mesh.
     */
    const bool is_chns_presolver = has_presolved_tracer();
    const double c_min =
      is_chns_presolver ?
        param.linear_elasticity.chns_presolver_initial_compression_multiplier :
        param.linear_elasticity.min_current_mesh_source_term_multiplier;
    const double c_max =
      is_chns_presolver ?
        1.0 :
        param.linear_elasticity.max_current_mesh_source_term_multiplier;
    const unsigned int n_steps =
      is_chns_presolver ?
        param.linear_elasticity.chns_presolver_continuation_steps :
        param.linear_elasticity.n_continuation_steps;

    const auto constitutive_model =
      param.physical_properties.pseudosolids[0].constitutive_model;
    const bool use_arithmetic_continuation =
      is_chns_presolver ||
      constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::neo_hookean ||
      constitutive_model ==
        Parameters::PseudoSolid<dim>::ConstitutiveModel::ogden;

    source_term_moving_mesh_multiplier = c_min;
    source_term_fixed_mesh_multiplier  = 0.;

    // Linear elasticity: geometric progression; finite-deformation
    // hyperelastic laws: arithmetic progression.
    const double r =
      (!use_arithmetic_continuation && n_steps > 1) ?
        std::pow(c_max / c_min, 1.0 / (n_steps - 1)) :
        1.;
    const double step =
      (use_arithmetic_continuation && n_steps > 1) ?
        (c_max - c_min) / static_cast<double>(n_steps - 1) :
        0.;

    for (unsigned int n = 0; n < n_steps; ++n)
    {
      pcout << std::endl;
      pcout << "Continuation method - Step " << n + 1 << "/" << n_steps;
      if (is_chns_presolver)
      {
        pcout << " : CHNS compression multiplier = "
              << source_term_moving_mesh_multiplier
              << " (mff_enlarged = "
              << source_term_moving_mesh_multiplier *
                   param.cahn_hilliard.mff_enlarged_compression_factor
              << ", mff_physics = "
              << source_term_moving_mesh_multiplier *
                   param.cahn_hilliard.mff_physics_compression_factor
              << ")";
      }
      else
        pcout << " : source term multiplier = "
              << source_term_moving_mesh_multiplier;
      pcout << std::endl;
      pcout << std::endl;

      if (param.debug.compare_analytical_jacobian_with_fd)
        compare_analytical_matrix_with_fd();
      solve_nonlinear_problem(time_handler);

      if (!use_arithmetic_continuation)
        source_term_moving_mesh_multiplier *= r;
      else
        source_term_moving_mesh_multiplier += step;
    }
  }
  else
  {
    // Source term is evaluated on reference mesh and problem is linear
    // This is the case when performing a convergence study with a
    // manufactured solution, for example.
    source_term_moving_mesh_multiplier = 0.;
    source_term_fixed_mesh_multiplier  = 1.;

    if (param.debug.compare_analytical_jacobian_with_fd)
      compare_analytical_matrix_with_fd();
    solve_nonlinear_problem(time_handler);
  }

  postprocess_solution();
}

template <int dim>
bool LinearElasticitySolver<dim>::try_load_presolved_mesh_position_cache()
{
  using CacheMode =
    Parameters::LinearElasticity::PresolvedMeshPositionCache::Mode;

  const auto mode = param.linear_elasticity.presolved_mesh_position_cache.mode;
  AssertThrow(mode != CacheMode::off,
              ExcMessage("Cannot load a presolved mesh position cache when "
                         "the cache mode is off."));

  reset();
  MeshTools::read_mesh(triangulation, param);
  setup_dofs();

  const std::string cache_prefix =
    param.output.output_dir +
    param.linear_elasticity.presolved_mesh_position_cache.filename;

  const PresolvedMeshPositionCacheMetadata expected_metadata{
    presolved_mesh_position_cache_format_version,
    dim,
    param.finite_elements.use_quads,
    param.finite_elements.mesh_position_degree,
    static_cast<unsigned long long>(triangulation.n_global_active_cells()),
    static_cast<unsigned long long>(dof_handler.n_dofs()),
    param.mesh.filename,
    hash_file_contents(param.mesh.filename),
    get_presolved_fields_name(),
    param.linear_elasticity.presolved_mesh_position_fingerprint};

  bool        local_cache_is_usable = true;
  std::string local_reason;
  std::vector<PresolvedMeshPositionCacheEntry<dim>> cache_entries;

  {
    std::ifstream cache_file(cache_prefix);
    if (!cache_file)
    {
      local_cache_is_usable = false;
      local_reason = "missing cache file " + cache_prefix;
    }
    else
    {
      try
      {
        boost::archive::text_iarchive archive(cache_file);
        PresolvedMeshPositionCacheMetadata cached_metadata;
        archive >> cached_metadata;

        local_reason =
          metadata_mismatch_reason(cached_metadata, expected_metadata);
        local_cache_is_usable = local_reason.empty();
        if (local_cache_is_usable)
          archive >> cache_entries;
      }
      catch (const std::exception &exc)
      {
        local_cache_is_usable = false;
        local_reason =
          "could not read cache metadata: " + std::string(exc.what());
      }
    }
  }

  const int cache_is_usable =
    Utilities::MPI::min(local_cache_is_usable ? 1 : 0, mpi_communicator);

  if (cache_is_usable == 0)
  {
    const bool read_only = mode == CacheMode::read_only;
    const std::string reason =
      local_reason.empty() ? "cache is unavailable or stale on another rank" :
                             local_reason;

    if (read_only)
      AssertThrow(false,
                  ExcMessage("The presolved mesh position cache is required "
                             "but cannot be used: " +
                             reason));

    pcout << "Presolved fields cache cannot be reused: " << reason
          << std::endl;
    return false;
  }

  std::map<std::string, double> cached_values;
  for (const auto &entry : cache_entries)
    cached_values[cache_entry_key(entry.support_point, entry.component)] =
      entry.value;

  std::vector<unsigned char> dofs_to_component;
  fill_dofs_to_component(dof_handler, locally_relevant_dofs, dofs_to_component);

  const auto support_points =
    DoFTools::map_dofs_to_support_points(*mapping,
                                         dof_handler,
                                         presolved_field_mask);

  local_evaluation_point = 0.;
  bool        local_values_found = true;
  std::string local_missing_reason;

  for (const auto dof : locally_owned_dofs)
  {
    const unsigned int component =
      dofs_to_component[locally_relevant_dofs.index_within_set(dof)];
    const auto value_it =
      cached_values.find(cache_entry_key(support_points.at(dof), component));

    if (value_it == cached_values.end())
    {
      local_values_found   = false;
      local_missing_reason = "cache does not contain all local support points";
      break;
    }

    local_evaluation_point[dof] = value_it->second;
  }

  const int values_found =
    Utilities::MPI::min(local_values_found ? 1 : 0, mpi_communicator);
  if (values_found == 0)
  {
    const bool read_only = mode == CacheMode::read_only;
    const std::string reason =
      local_missing_reason.empty() ?
        "cache does not contain all support points on another rank" :
        local_missing_reason;

    if (read_only)
      AssertThrow(false,
                  ExcMessage("The presolved mesh position cache is required "
                             "but cannot be used: " +
                             reason));

    pcout << "Presolved fields cache cannot be reused: " << reason
          << std::endl;
    return false;
  }

  local_evaluation_point.compress(VectorOperation::insert);
  present_solution = local_evaluation_point;
  present_solution.update_ghost_values();
  local_evaluation_point = present_solution;
  evaluation_point       = present_solution;

  pcout << "Loaded presolved fields cache (" << get_presolved_fields_name()
        << ") from " << cache_prefix
        << std::endl;
  return true;
}

template <int dim>
void LinearElasticitySolver<dim>::write_presolved_mesh_position_cache() const
{
  using CacheMode =
    Parameters::LinearElasticity::PresolvedMeshPositionCache::Mode;

  const auto mode = param.linear_elasticity.presolved_mesh_position_cache.mode;
  AssertThrow(mode != CacheMode::off,
              ExcMessage("Cannot write a presolved mesh position cache when "
                         "the cache mode is off."));

  const std::string cache_filename =
    param.linear_elasticity.presolved_mesh_position_cache.filename;
  const std::string tmp_cache_prefix =
    param.output.output_dir + "tmp." + cache_filename;

  const PresolvedMeshPositionCacheMetadata metadata{
    presolved_mesh_position_cache_format_version,
    dim,
    param.finite_elements.use_quads,
    param.finite_elements.mesh_position_degree,
    static_cast<unsigned long long>(triangulation.n_global_active_cells()),
    static_cast<unsigned long long>(dof_handler.n_dofs()),
    param.mesh.filename,
    hash_file_contents(param.mesh.filename),
    get_presolved_fields_name(),
    param.linear_elasticity.presolved_mesh_position_fingerprint};

  std::vector<unsigned char> dofs_to_component;
  fill_dofs_to_component(dof_handler, locally_relevant_dofs, dofs_to_component);

  const auto support_points =
    DoFTools::map_dofs_to_support_points(*mapping,
                                         dof_handler,
                                         presolved_field_mask);

  std::vector<PresolvedMeshPositionCacheEntry<dim>> local_entries;
  local_entries.reserve(locally_owned_dofs.n_elements());

  for (const auto dof : locally_owned_dofs)
  {
    PresolvedMeshPositionCacheEntry<dim> entry;
    const auto                         &support_point = support_points.at(dof);
    for (unsigned int d = 0; d < dim; ++d)
      entry.support_point[d] = support_point[d];
    entry.component =
      dofs_to_component[locally_relevant_dofs.index_within_set(dof)];
    entry.value = present_solution[dof];
    local_entries.push_back(entry);
  }

  const auto gathered_entries =
    Utilities::MPI::gather(mpi_communicator, local_entries, 0);

  if (mpi_rank == 0)
  {
    std::vector<PresolvedMeshPositionCacheEntry<dim>> entries;
    for (const auto &rank_entries : gathered_entries)
      entries.insert(entries.end(), rank_entries.begin(), rank_entries.end());

    std::ofstream cache_file(tmp_cache_prefix);
    AssertThrow(cache_file,
                ExcMessage("Could not write presolved mesh position cache file " +
                           tmp_cache_prefix));
    boost::archive::text_oarchive archive(cache_file);
    archive << metadata;
    archive << entries;
  }

  MPI_Barrier(mpi_communicator);
  replace_temporary_files(param.output.output_dir,
                          "tmp." + cache_filename,
                          cache_filename,
                          mpi_communicator);

  pcout << "Wrote presolved fields cache (" << get_presolved_fields_name()
        << ") to " << param.output.output_dir + cache_filename << std::endl;
}

template <int dim>
void LinearElasticitySolver<dim>::setup_dofs()
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
void LinearElasticitySolver<dim>::create_base_constraints(
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
void LinearElasticitySolver<dim>::create_zero_constraints()
{
  create_base_constraints(true, zero_constraints);
}

template <int dim>
void LinearElasticitySolver<dim>::create_nonzero_constraints()
{
  create_base_constraints(false, nonzero_constraints);
}

template <int dim>
void LinearElasticitySolver<dim>::create_sparsity_pattern()
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
void LinearElasticitySolver<dim>::set_initial_conditions()
{
  FixedMeshPosition<dim> fixed_mesh(0, fe->n_components());
  VectorTools::interpolate(
    *mapping, dof_handler, fixed_mesh, newton_update, position_mask);

  if (has_presolved_tracer())
  {
    ScalarFunctionFromComponents<dim> tracer_function(
      dim,
      fe->n_components(),
      *param.initial_conditions.initial_chns_tracer_callback);
    VectorTools::interpolate(
      *mapping, dof_handler, tracer_function, newton_update, tracer_mask);
  }

  if (has_presolved_psi())
  {
    const auto &psi_callback =
      param.initial_conditions.use_enlarged_psi ?
        *param.initial_conditions.initial_chns_enlarged_psi_callback :
        *param.initial_conditions.initial_chns_tracer_callback;
    ScalarFunctionFromComponents<dim> psi_function(dim + 1,
                                                   fe->n_components(),
                                                   psi_callback);
    VectorTools::interpolate(
      *mapping, dof_handler, psi_function, newton_update, psi_mask);
  }

  // Apply non-homogeneous Dirichlet BC and set as current solution
  nonzero_constraints.distribute(newton_update);
  present_solution = newton_update;
  evaluation_point = newton_update;
}

template <int dim>
void LinearElasticitySolver<dim>::set_exact_solution()
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
void LinearElasticitySolver<dim>::update_boundary_conditions()
{
  local_evaluation_point = present_solution;
  create_nonzero_constraints();
  nonzero_constraints.distribute(local_evaluation_point);
  evaluation_point = local_evaluation_point;
  present_solution = local_evaluation_point;
}

template <int dim>
void LinearElasticitySolver<dim>::assemble_matrix()
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
      &LinearElasticitySolver::assemble_local_matrix :
      &LinearElasticitySolver::assemble_local_matrix_finite_differences;

  // Assemble matrix (multithreaded if supported)
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  assembly_ptr,
                  &LinearElasticitySolver::copy_local_to_global_matrix,
                  *scratch_data,
                  copy_data);
  system_matrix.compress(VectorOperation::add);
}

template <int dim>
void LinearElasticitySolver<dim>::assemble_local_matrix_finite_differences(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  Verification::compute_local_matrix_finite_differences<dim>(
    cell,
    *this,
    &LinearElasticitySolver::assemble_local_rhs,
    scratch_data,
    copy_data,
    this->evaluation_point,
    this->local_evaluation_point);
}

template <int dim>
void LinearElasticitySolver<dim>::assemble_local_matrix(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();
  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell, evaluation_point, source_terms, exact_solution);

  auto &local_matrix = copy_data.local_matrix();
  local_matrix       = 0;

  const double source_alpha =
    has_presolved_tracer() ? 0. : source_term_moving_mesh_multiplier;
  const double forcing_alpha =
    has_presolved_tracer() ? source_term_moving_mesh_multiplier : 0.;
  const auto &ps = param.physical_properties.pseudosolids[0];
  auto        cahn_hilliard = param.cahn_hilliard;
  if (has_presolved_tracer())
  {
    cahn_hilliard.mff_enlarged_compression_factor *= forcing_alpha;
    cahn_hilliard.mff_physics_compression_factor *= forcing_alpha;
    cahn_hilliard.mff_transport_factor = 0.;
    cahn_hilliard.psi_mu_correction_factor = 0.;
  }

  //
  // Volume contributions
  //
  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW         = scratch_data.JxW[q];
    const double lame_mu     = scratch_data.lame_mu[q];
    const double lame_lambda = scratch_data.lame_lambda[q];

    const auto &phi_x      = scratch_data.phi_x[q];
    const auto &grad_phi_x = scratch_data.grad_phi_x[q];
    const auto &div_phi_x  = scratch_data.div_phi_x[q];

    const Tensor<2, dim> &grad_source_current_mesh =
      scratch_data.grad_source_term_position_current_mesh[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      const auto &phi_x_i      = phi_x[i];
      const auto &grad_phi_x_i = grad_phi_x[i];
      const auto &div_phi_x_i  = div_phi_x[i];

      for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
      {
        const auto &phi_x_j      = phi_x[j];
        const auto &grad_phi_x_j = grad_phi_x[j];
        const auto &div_phi_x_j  = div_phi_x[j];

        local_matrix(i, j) +=
          (Assembly::Pseudosolid::matrix_contribution(ps,
                                                      lame_mu,
                                                      lame_lambda,
                                                      scratch_data
                                                        .position_gradients[q],
                                                      scratch_data
                                                        .position_inv_gradients[q],
                                                      scratch_data
                                                        .position_inv_gradients_T[q],
                                                      scratch_data.position_J[q],
                                                      div_phi_x_i,
                                                      grad_phi_x_i,
                                                      div_phi_x_j,
                                                      grad_phi_x_j) +
           source_alpha * phi_x_i * (grad_source_current_mesh * phi_x_j)) *
          JxW;
      }
    }
  }

  if (has_presolved_tracer())
  {
    const double bdf_c0 = 0.;
    if (!has_presolved_psi())
      Assembly::MovingMeshForcing::assemble_chns_matrix<dim, false>(
        *presolver_ordering,
        presolver_coupling_table,
        cahn_hilliard,
        bdf_c0,
        scratch_data,
        local_matrix);

    for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
      for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
      {
        if (!presolver_ordering->is_tracer(scratch_data.components[i]))
          continue;

        const double phi_i = scratch_data.shape_phi[q][i];
        const double projection_residual =
          scratch_data.tracer_values[q] - scratch_data.analytic_tracer_values[q];

        for (unsigned int j = 0; j < scratch_data.dofs_per_cell; ++j)
        {
          if (presolver_coupling_table[scratch_data.components[i]]
                                      [scratch_data.components[j]] !=
              DoFTools::always)
            continue;

          double local_ij = 0.;
          if (presolver_ordering->is_tracer(scratch_data.components[j]))
            local_ij += phi_i * scratch_data.shape_phi[q][j];

          if (presolver_ordering->is_position(scratch_data.components[j]))
          {
            const Tensor<2, dim> &G =
              scratch_data.grad_phi_x_moving[q][j];
            local_ij +=
              phi_i *
              (projection_residual * Assembly::ALE::jacobian_trace(G) -
               scratch_data.analytic_tracer_gradients[q] *
                 scratch_data.phi_x[q][j]);
          }

          local_matrix(i, j) += local_ij * scratch_data.JxW_moving[q];
        }
      }

    if (has_presolved_psi())
    {
      const double enlarged_length =
        param.cahn_hilliard.epsilon_interface_enlarged -
        param.cahn_hilliard.epsilon_interface;
      const double enlarged_length_sq = enlarged_length * enlarged_length;

      Assembly::assemble_psi_equation_matrix<dim, true>(
        *presolver_ordering,
        presolver_coupling_table,
        scratch_data,
        cahn_hilliard,
        enlarged_length_sq,
        local_matrix);

      Assembly::MovingMeshForcing::assemble_chns_matrix<dim, true>(
        *presolver_ordering,
        presolver_coupling_table,
        cahn_hilliard,
        bdf_c0,
        scratch_data,
        local_matrix);
    }
  }
  cell->get_dof_indices(copy_data.dof_indices());
}

template <int dim>
void LinearElasticitySolver<dim>::copy_local_to_global_matrix(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;
  zero_constraints.distribute_local_to_global(copy_data.local_matrix(),
                                              copy_data.dof_indices(),
                                              system_matrix);
}

template <int dim>
void LinearElasticitySolver<dim>::compare_analytical_matrix_with_fd()
{
  CopyData copy_data(*fe);

  auto errors = Verification::compare_analytical_matrix_with_fd(
    dof_handler,
    fe->n_dofs_per_cell(),
    *this,
    &LinearElasticitySolver::assemble_local_matrix,
    &LinearElasticitySolver::assemble_local_rhs,
    *scratch_data,
    copy_data,
    present_solution,
    evaluation_point,
    local_evaluation_point,
    mpi_communicator,
    /*output_dir = */ "",
    /*print_problematic_elements = */ false,
    param.debug.analytical_jacobian_absolute_tolerance,
    param.debug.analytical_jacobian_relative_tolerance);

  pcout << "Max absolute error analytical vs fd matrix is " << errors.first
        << std::endl;

  // Only print relative error if absolute is too large
  if (errors.first > param.debug.analytical_jacobian_absolute_tolerance)
    pcout << "Max relative error analytical vs fd matrix is " << errors.second
          << std::endl;
}

template <int dim>
void LinearElasticitySolver<dim>::assemble_rhs()
{
  TimerOutput::Scope t(computing_timer, "Assemble RHS");

  system_rhs = 0;

  CopyData copy_data(*fe);

  // Assemble RHS (multithreaded if supported)
  WorkStream::run(dof_handler.begin_active(),
                  dof_handler.end(),
                  *this,
                  &LinearElasticitySolver::assemble_local_rhs,
                  &LinearElasticitySolver::copy_local_to_global_rhs,
                  *scratch_data,
                  copy_data);

  system_rhs.compress(VectorOperation::add);
}

template <int dim>
void LinearElasticitySolver<dim>::assemble_local_rhs(
  const typename DoFHandler<dim>::active_cell_iterator &cell,
  ScratchData                                          &scratch_data,
  CopyData                                             &copy_data)
{
  copy_data.cell_is_locally_owned = cell->is_locally_owned();

  if (!cell->is_locally_owned())
    return;

  scratch_data.reinit(cell, evaluation_point, source_terms, exact_solution);

  auto &local_rhs = copy_data.local_rhs();
  local_rhs       = 0;

  const double source_alpha =
    has_presolved_tracer() ? 0. : source_term_moving_mesh_multiplier;
  const double forcing_alpha =
    has_presolved_tracer() ? source_term_moving_mesh_multiplier : 0.;
  const double gamma =
    has_presolved_tracer() ? 0. : source_term_fixed_mesh_multiplier;
  const auto &ps = param.physical_properties.pseudosolids[0];
  auto        cahn_hilliard = param.cahn_hilliard;
  if (has_presolved_tracer())
  {
    cahn_hilliard.mff_enlarged_compression_factor *= forcing_alpha;
    cahn_hilliard.mff_physics_compression_factor *= forcing_alpha;
    cahn_hilliard.mff_transport_factor = 0.;
    cahn_hilliard.psi_mu_correction_factor = 0.;
  }

  // alpha and gamma cannot both be nonzero
  Assert(!(std::abs(source_alpha) > 1e-14 && std::abs(gamma) > 1e-14),
         ExcInternalError());

  //
  // Volume contributions
  //
  for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
  {
    const double JxW         = scratch_data.JxW[q];
    const double lame_mu     = scratch_data.lame_mu[q];
    const double lame_lambda = scratch_data.lame_lambda[q];

    const auto &source_term_position_moving_mesh =
      scratch_data.source_term_position_current_mesh[q];
    const auto &source_term_position_fixed_mesh =
      scratch_data.source_term_position[q];
    // The source term to use : using coefficients which cannot be both nonzero
    // avois using a condition
    const auto source_term = source_alpha * source_term_position_moving_mesh +
                             gamma * source_term_position_fixed_mesh;

    const Tensor<2, dim> strain =
      Tensor<2, dim>(scratch_data.position_strains[q]);
    const auto trace_strain = scratch_data.position_trace_strains[q];

    const auto &phi_x      = scratch_data.phi_x[q];
    const auto &grad_phi_x = scratch_data.grad_phi_x[q];
    const auto &div_phi_x  = scratch_data.div_phi_x[q];

    for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
    {
      local_rhs(i) -=
        (Assembly::Pseudosolid::rhs_contribution(ps,
                                                 lame_mu,
                                                 lame_lambda,
                                                 trace_strain,
                                                 strain,
                                                 scratch_data
                                                   .position_gradients[q],
                                                 scratch_data
                                                   .position_inv_gradients_T[q],
                                                 scratch_data.position_J[q],
                                                 div_phi_x[i],
                                                 grad_phi_x[i]) +
         phi_x[i] * source_term) *
      JxW;
    }
  }

  if (has_presolved_tracer())
  {
    if (!has_presolved_psi())
      Assembly::MovingMeshForcing::assemble_chns_rhs<dim, false>(
        *presolver_ordering, cahn_hilliard, scratch_data, local_rhs);

    for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
      for (unsigned int i = 0; i < scratch_data.dofs_per_cell; ++i)
      {
        if (!presolver_ordering->is_tracer(scratch_data.components[i]))
          continue;

        const double local_rhs_i =
          scratch_data.shape_phi[q][i] *
          (scratch_data.tracer_values[q] -
           scratch_data.analytic_tracer_values[q]);
        local_rhs(i) -= local_rhs_i * scratch_data.JxW_moving[q];
      }

    if (has_presolved_psi())
    {
      const double enlarged_length =
        param.cahn_hilliard.epsilon_interface_enlarged -
        param.cahn_hilliard.epsilon_interface;
      const double enlarged_length_sq = enlarged_length * enlarged_length;

      Assembly::assemble_psi_equation_rhs<dim>(*presolver_ordering,
                                               scratch_data,
                                               cahn_hilliard,
                                               enlarged_length_sq,
                                               local_rhs);

      Assembly::MovingMeshForcing::assemble_chns_rhs<dim, true>(
        *presolver_ordering, cahn_hilliard, scratch_data, local_rhs);
    }
  }

  cell->get_dof_indices(copy_data.dof_indices());
}

template <int dim>
void LinearElasticitySolver<dim>::copy_local_to_global_rhs(
  const CopyData &copy_data)
{
  if (!copy_data.cell_is_locally_owned)
    return;
  zero_constraints.distribute_local_to_global(copy_data.local_rhs(),
                                              copy_data.dof_indices(),
                                              system_rhs);
}

template <int dim>
void LinearElasticitySolver<dim>::solve_linear_system()
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
    solve_linear_system_unpreconditioned_cg(this,
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
                           "LinearElasticitySolver. Use CG unstead."));
  }
  else
  {
    AssertThrow(false, ExcMessage("No known resolution method"));
  }
}

template <int dim>
void LinearElasticitySolver<dim>::compute_cell_average_strain(
  std::vector<SymmetricTensor<2, dim>> &strain_tensors,
  Vector<double>                       &strain_trace)
{
  const QGauss<dim>     quadrature_formula(fe->degree + 1);
  const QGauss<dim - 1> face_quadrature_formula(fe->degree + 1);
  ScratchData scratch_data(*fe,
                           *mapping,
                           dof_handler,
                           position_mask,
                           quadrature_formula,
                           face_quadrature_formula,
                           param,
                           has_presolved_tracer(),
                           has_presolved_psi());
  const unsigned int n_active_cells = triangulation.n_active_cells();

  strain_tensors.assign(n_active_cells, SymmetricTensor<2, dim>());
  strain_trace.reinit(n_active_cells);

  for (const auto &cell : dof_handler.active_cell_iterators())
  {
    if (cell->is_locally_owned() || cell->is_ghost())
    {
      scratch_data.reinit(cell, present_solution, source_terms, exact_solution);

      SymmetricTensor<2, dim> eps_avg;
      double                  measure = 0.0;

      for (unsigned int q = 0; q < scratch_data.n_q_points; ++q)
      {
        eps_avg += scratch_data.position_strains[q] * scratch_data.JxW[q];
        measure += scratch_data.JxW[q];
      }

      eps_avg /= measure;

      const unsigned int idx = cell->active_cell_index();
      strain_tensors[idx]    = eps_avg;
      strain_trace(idx)      = trace(eps_avg);
    }
  }
}

template <int dim>
void LinearElasticitySolver<dim>::output_results()
{
  TimerOutput::Scope t(computing_timer, "Write outputs");

  if (param.output.write_results)
  {
    std::vector<std::string> solution_names(dim, "position");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      data_component_interpretation(
        dim, DataComponentInterpretation::component_is_part_of_vector);
    if (has_presolved_tracer())
    {
      solution_names.push_back("phi");
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
    }
    if (has_presolved_psi())
    {
      solution_names.push_back("psi");
      data_component_interpretation.push_back(
        DataComponentInterpretation::component_is_scalar);
    }
    DataOut<dim> data_out;
    data_out.attach_dof_handler(dof_handler);
    data_out.add_data_vector(present_solution,
                             solution_names,
                             DataOut<dim>::type_dof_data,
                             data_component_interpretation);

    std::vector<SymmetricTensor<2, dim>> strain_tensors;
    Vector<double>                       strain_trace;

    if (strain_cache_is_valid)
    {
      strain_tensors = cached_strain_tensors;
      strain_trace   = cached_strain_trace;
    }
    else
      compute_cell_average_strain(strain_tensors, strain_trace);

    PostProcessingTools::DG0DataField<dim> strain_field(
      triangulation,
      param.finite_elements.use_quads,
      PostProcessingTools::make_tensor_component_names<dim>("strain"),
      PostProcessingTools::make_tensor_component_interpretation<dim>());

    for (const auto &cell :
         strain_field.get_dof_handler().active_cell_iterators())
      if (cell->is_locally_owned() || cell->is_ghost())
        strain_field.set_cell_values(cell,
                                     strain_tensors[cell->active_cell_index()]);

    PostProcessingTools::add_dg0_data_field(data_out, strain_field);
    data_out.add_data_vector(strain_trace,
                             "strain_trace",
                             DataOut<dim>::type_cell_data);
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
                                          "linear_elasticity",
                                        0,
                                        mpi_communicator,
                                        2);
  }
}

template <int dim>
void LinearElasticitySolver<dim>::move_mesh()
{
  present_solution.update_ghost_values();

  const IndexSet locally_owned = dof_handler.locally_owned_dofs();

  std::vector<bool> vertex_moved(triangulation.n_vertices(), false);
  std::vector<bool> vertex_locally_moved(triangulation.n_vertices(), false);

  for (auto &cell : dof_handler.active_cell_iterators())
    if (cell->is_locally_owned())
      for (const auto v : cell->vertex_indices())
      {
        const auto vid = cell->vertex_index(v);
        if (vertex_moved[vid])
          continue;

        // this rank owns the vertex if all dim position dofs are locally-owned
        bool i_own_vertex = true;
        for (unsigned int d = 0; d < dim; ++d)
          i_own_vertex = i_own_vertex &&
                         locally_owned.is_element(cell->vertex_dof_index(v, d));

        // mark as visited so we don't process it again
        vertex_moved[vid] = true;

        if (!i_own_vertex)
          continue;

        // move vertex to its new position
        for (unsigned int d = 0; d < dim; ++d)
          cell->vertex(v)[d] = present_solution(cell->vertex_dof_index(v, d));

        vertex_locally_moved[vid] = true;
      }

  // sync moved vertices across MPI boundaries
  triangulation.communicate_locally_moved_vertices(vertex_locally_moved);
}

template <int dim>
void LinearElasticitySolver<dim>::write_final_msh()
{
  if (!param.linear_elasticity.write_final_msh)
    return;

  AssertThrow(
    has_msh_extension(param.mesh.filename),
    ExcMessage("Writing the final deformed mesh requires the input mesh to "
               "come from a Gmsh .msh file."));

#if defined(DEAL_II_GMSH_WITH_API)
  const unsigned int rank = Utilities::MPI::this_mpi_process(mpi_communicator);

  std::vector<std::size_t> node_tags;
  std::vector<double>      node_coordinates;
  std::vector<Point<dim>>  evaluation_points;

  if (rank == 0)
  {
    gmsh::initialize();
    gmsh::option::setNumber("General.Verbosity", 2);
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
  }

  present_solution.update_ghost_values();

  Utilities::MPI::RemotePointEvaluation<dim> cache;
  const auto deformed_positions = VectorTools::point_values<dim>(
    *mapping, dof_handler, present_solution, evaluation_points, cache);

  AssertThrow(cache.all_points_found(),
              ExcMessage(
                "Could not evaluate the deformed mesh position at all Gmsh "
                "nodes when writing the final .msh file."));

  if (rank == 0)
  {
    AssertDimension(deformed_positions.size(), node_tags.size());

    for (unsigned int i = 0; i < node_tags.size(); ++i)
    {
      std::vector<double> coordinates(3, 0.0);
      for (unsigned int d = 0; d < dim; ++d)
        coordinates[d] = deformed_positions[i][d];
      if constexpr (dim == 2)
        coordinates[2] = node_coordinates[3 * i + 2];

      gmsh::model::mesh::setNode(node_tags[i], coordinates, {});
    }

    const std::string output_mesh_filename =
      param.output.output_dir + param.output.output_prefix +
      "linear_elasticity_final_mesh.msh";

    gmsh::write(output_mesh_filename);
    gmsh::clear();
    gmsh::finalize();

    pcout << "Wrote final deformed mesh to " << output_mesh_filename
          << std::endl;
  }
#else
  AssertThrow(false,
              ExcMessage("Gmsh API support is required to write the final "
                         "deformed .msh file."));
#endif
}


template <int dim>
void LinearElasticitySolver<dim>::compute_errors()
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
void LinearElasticitySolver<dim>::postprocess_solution()
{
  // Compute error *before* moving mesh for visualization (-:
  if (param.mms_param.enable)
    compute_errors();

  compute_cell_average_strain(cached_strain_tensors, cached_strain_trace);
  strain_cache_is_valid = true;

  write_final_msh();

  using CacheMode =
    Parameters::LinearElasticity::PresolvedMeshPositionCache::Mode;
  const auto cache_mode =
    param.linear_elasticity.presolved_mesh_position_cache.mode;
  if (cache_mode == CacheMode::automatic ||
      cache_mode == CacheMode::force_recompute)
  {
    // Cache keys are reference support points, so write before move_mesh().
    write_presolved_mesh_position_cache();
  }

  move_mesh();
  output_results();
  strain_cache_is_valid = false;
}

// Explicit instantiation
template class LinearElasticitySolver<2>;
template class LinearElasticitySolver<3>;
